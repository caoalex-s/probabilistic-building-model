"""
CLI entry point for Monte Carlo simulation of a hierarchical network abstraction.

This script orchestrates a full simulation run:
- load nodes, edges, and simultaneous pre-run notional damage from SQLite,
- build a transition matrix used for propagation,
- run Monte Carlo sampling of initial failures and cascading propagation,
- write run metadata and results to an SQLite output database,
- optionally execute a percolation scan that varies edge probabilities by axis.

Design
------
The file is intentionally procedural: it defines run policy and precedence,
while numerical/graph logic lives in `pbm.network` and `pbm.sim`.

Configuration
-------------
This CLI is intentionally YAML-free. Run-critical settings are provided via:

- CLI flags (samples, seed, threshold, percolation scan), and
- the input database (pf_node, edge probabilities, and notional damage).

Percolation enablement:
- Percolation is enabled only when `--percolation-scan` is a non-empty string.
- When enabled, run metadata in `runs.notes` includes: `percolation_multi`,
  `scan_spec`, `scan_mode`, `grouped_axes`, and explicit varied/fixed axis
  metadata.

Initial-state semantics
-----------------------
For each Monte Carlo sample, the initial failure state `F0` is built as:

    F0[i] = True                         if i is notionally damaged
          = Bernoulli(pf_node[i])        otherwise

The database representation is `initial_damage(node_id)`. Runtime
`--damage-node-ids` and `--damage-coords` inputs are resolved to the same
canonical node-ID set and are rejected when the database set is non-empty.
Unknown targets are errors. The same package application function is used for
the base simulation, parallel workers, the API, and percolation scans.

Random stream semantics
-----------------------
Realization ``i`` always receives the stream allocated from
``(base_seed, i)`` by :mod:`pbm.sim.random_streams`. Sequential and parallel
execution call the same range function, and chunking only partitions the
zero-based realization indices. ``run_id``, scenario ID, workers, processes,
hosts, and scheduling order never enter stream allocation.

Percolation scan points and notional-damage cases reuse the same realization
streams when given the same base seed. This is intentional common-random-number
coupling for otherwise comparable experiments.

Edge probability semantics
--------------------------
Edges can be directed or undirected.

- Undirected edges use `pe` in both directions.
- Directed edges use the required `pe_fwd` and `pe_rev` values.
- Blocked edges transmit with probability zero in both directions.

Output edge IDs in `edge_usage` / `edge_attempts` are written using the input
database primary key `edges.id`, aggregated over both directions for a given
undirected edge id.

This module is not intended to be imported as a library API.
"""

from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
import argparse, hashlib, json, os, socket, struct
from datetime import datetime

import numpy as np
import sqlite3
import itertools, re

from pbm import __version__
from pbm.io.loader import load_all
from pbm.io.writer import connect, exec_many
from pbm.utils.utils import timestamp
from pbm.network.propagation import build_transition_matrix, simulate_once_mc
from pbm.sim.notional_damage import (
    NotionalDamageError,
    apply_notional_damage,
    notional_damage_mask,
    resolve_notional_damage,
)
from pbm.sim.random_streams import RNG_CONTRACT_VERSION, realization_rng

from concurrent.futures import ProcessPoolExecutor, as_completed


# --- MC parallel (module-level worker state) -------------------------
#
# Workers must not touch SQLite. They compute aggregated statistics for a
# sample-index range and return them to the parent, which performs a single
# bulk insert.
#
# For Windows compatibility (spawn), worker functions must be module-level.

_MC_MATS = None
_MC_PF = None
_MC_DAMAGE_MASK = None
_MC_N = None


def _mc_worker_init(
    N: int,
    edges_tbl: np.ndarray,
    pf: np.ndarray,
    damage_mask: np.ndarray,
):
    """Per-process initializer building the propagation matrices once."""
    global _MC_MATS, _MC_PF, _MC_DAMAGE_MASK, _MC_N
    _MC_N = int(N)
    _MC_PF = np.asarray(pf, dtype=float)
    _MC_DAMAGE_MASK = np.asarray(damage_mask, dtype=bool)
    _MC_MATS = build_transition_matrix(_MC_N, edges_tbl)


def _simulate_realization_range(
    N: int,
    mats,
    pf: np.ndarray,
    damage_mask: np.ndarray,
    base_seed: int,
    i0: int,
    i1: int,
    *,
    log_attempts: bool,
):
    """Run a contiguous range using the global realization-index contract.

    Returns
    -------
    (counts, agg_succ, agg_att)
        counts: np.ndarray shape (N+1,), int64
        agg_succ: np.ndarray shape (nnz,), int64
        agg_att: np.ndarray shape (nnz,), int64, or None

    Notes
    -----
    Chunk boundaries are deliberately absent from stream allocation. Every
    realization constructs the same generator from ``(base_seed, index)``
    regardless of which process executes this function.
    """
    counts = np.zeros(N + 1, dtype=np.int64)
    agg_succ = np.zeros(mats.P.nnz, dtype=np.int64)
    agg_att = (
        np.zeros(mats.P.nnz, dtype=np.int64)
        if log_attempts
        else None
    )

    for i in range(int(i0), int(i1)):
        rng = realization_rng(base_seed, i)
        F0 = apply_notional_damage(pf, damage_mask, rng)
        res = simulate_once_mc(
            F0,
            mats,
            rng=rng,
            log_attempts=log_attempts,
        )

        n_failed = int(res.F_final.sum())
        counts[n_failed] += 1
        if res.successes_per_edge is not None:
            agg_succ += res.successes_per_edge.astype(np.int64, copy=False)
        if log_attempts and res.attempts_per_edge is not None:
            agg_att += res.attempts_per_edge.astype(np.int64, copy=False)

    return counts, agg_succ, agg_att


def _mc_worker_range(task):
    """Run one MC chunk in a worker without persistence-derived RNG inputs."""
    (base_seed, i0, i1) = task
    return _simulate_realization_range(
        _MC_N,
        _MC_MATS,
        _MC_PF,
        _MC_DAMAGE_MASK,
        base_seed,
        i0,
        i1,
        log_attempts=False,
    )


def _resolve_mc_chunk_size(
    samples: int,
    workers: int,
    requested_chunk_size: int,
) -> int:
    """Resolve ``0`` to a bounded-task automatic chunk size."""
    if requested_chunk_size < 0:
        raise ValueError("MC chunk size must be >= 0.")
    if requested_chunk_size > 0:
        return min(int(samples), int(requested_chunk_size))
    if workers <= 1:
        return int(samples)
    target_tasks = max(1, int(workers) * 4)
    return max(1, (int(samples) + target_tasks - 1) // target_tasks)


def _realization_ranges(samples: int, chunk_size: int):
    """Return half-open realization ranges covering ``[0, samples)`` once."""
    return [
        (start, min(start + int(chunk_size), int(samples)))
        for start in range(0, int(samples), int(chunk_size))
    ]


def _percolation_worker_one_point(task):
    """Compute percolation metrics for a single (pe_x, pe_y, pe_z) scan point."""
    (
        run_id,
        N_nodes,
        base_edges,     # np structured array
        axes_arr,       # np array of 'x'/'y'/'z' per edge
        dir_mask,       # boolean mask over edges
        und_mask,       # boolean mask over edges
        pf,             # np array over nodes
        damage_mask,    # boolean mask over nodes
        samples,
        base_seed,
        pe_x,
        pe_y,
        pe_z,
    ) = task

    import numpy as np

    safe_samples = max(1, int(samples))

    et = base_edges.copy()

    for axis, p in (("x", pe_x), ("y", pe_y), ("z", pe_z)):
        if p is None:
            continue  # leave DB values as-is for this axis
        m_ax = (axes_arr == axis)
        if np.any(dir_mask & m_ax):
            raise ValueError(
                f"Scalar percolation scan for axis '{axis}' cannot modify "
                "directed edges; provide an undirected scan model."
            )
        if np.any(und_mask & m_ax):
            idx = np.nonzero(m_ax & und_mask)[0]
            et["pe"][idx] = float(p)

    mats = build_transition_matrix(N_nodes, et)

    any_count = 0
    one_count = 0
    full_count = 0
    sum_nf = 0
    nf_counts = np.zeros(int(N_nodes) + 1, dtype=np.int64)
    for realization_index in range(safe_samples):
        rng = realization_rng(base_seed, realization_index)
        F0 = apply_notional_damage(pf, damage_mask, rng)
        res = simulate_once_mc(F0, mats, rng=rng, log_attempts=False)
        nf = int(res.F_final.sum())
        nf_counts[nf] += 1
        sum_nf += nf
        if nf >= 1:
            any_count += 1
        if nf == 1:
            one_count += 1
        if nf == N_nodes:
            full_count += 1

    p_any = any_count / float(safe_samples)
    p_one = one_count / float(safe_samples)
    p_full = full_count / float(safe_samples)
    mean_nf = sum_nf / float(safe_samples)

    # Store None as sentinel -1.0 to keep composite PK unambiguous
    def _s(x):
        return -1.0 if x is None else float(x)

    return (
        int(run_id),
        _s(pe_x),
        _s(pe_y),
        _s(pe_z),
        float(p_any),
        float(p_one),
        float(p_full),
        float(mean_nf),
        int(safe_samples),
        int(any_count),
        int(one_count),
        int(full_count),
        tuple(int(value) for value in nf_counts),
    )


def _write_percolation_multi_bulk(conn, rows):
    """Bulk insert percolation_multi rows in a single transaction."""
    if not rows:
        return
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(percolation_multi);").fetchall()
    }
    if {"n_samples", "count_any", "count_one", "count_full"} <= columns:
        conn.executemany(
            """
            INSERT OR REPLACE INTO percolation_multi
            (
              run_id, pe_x, pe_y, pe_z, p_any, p_one, p_full, mean_nf,
              n_samples, count_any, count_one, count_full
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [row[:12] for row in rows],
        )
    else:
        # Backward-compatible write to a pre-Prompt-6 results database.
        conn.executemany(
            """
            INSERT OR REPLACE INTO percolation_multi
            (run_id, pe_x, pe_y, pe_z, p_any, p_one, p_full, mean_nf)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [row[:8] for row in rows],
        )


def _write_percolation_pmf_bulk(conn, rows):
    """Retain the complete integer ``N_f`` histogram at every scan point."""
    if not rows:
        return
    table_exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name='percolation_pmf';
        """
    ).fetchone()
    if table_exists is None:
        raise RuntimeError(
            "The output schema does not contain percolation_pmf. Recreate or "
            "migrate the results database before running a percolation scan."
        )

    pmf_rows = []
    for row in rows:
        run_id, pe_x, pe_y, pe_z = row[:4]
        n_samples = int(row[8])
        histogram = row[12]
        if len(histogram) == 0 or sum(histogram) != n_samples:
            raise RuntimeError(
                "Percolation PMF histogram is inconsistent with n_samples."
            )
        pmf_rows.extend(
            (
                int(run_id),
                float(pe_x),
                float(pe_y),
                float(pe_z),
                int(n_failed),
                int(count),
                float(count / n_samples),
                n_samples,
            )
            for n_failed, count in enumerate(histogram)
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO percolation_pmf
        (
          run_id, pe_x, pe_y, pe_z, n_failed, count, p, n_samples
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pmf_rows,
    )


from pbm.network.metrics import static_outflow_influence
from pbm.io.schema import apply_schema

# --- Percolation scan utilities -------------------------------------

def _parse_scan_spec_for_metadata(spec: str) -> tuple[str, list[str]]:
    """
    Lightweight parser for percolation scan specs, used only for metadata.

    Parameters
    ----------
    spec : str
        Scan specification string, e.g. "xy:0,1,0.1; z:1,1,1".

        Returns
    -------
    scan_mode : {"coupled", "grouped"}
        "coupled" -> an xyz:... token is present (diagonal sweep semantics).
        "grouped" -> otherwise (single-axis or grouped axes).
    grouped_axes : list[str]
        Sorted list of axes that appear in multi-axis groups, e.g. ["x","y"].
        For pure single-axis scans this will be [].
    """
    spec = (spec or "").strip()
    if not spec:
        return "grouped", []

    tokens = [t.strip() for t in spec.split(";") if t.strip()]
    grouped_axes_set: set[str] = set()
    has_xyz = False

    for tok in tokens:
        if ":" not in tok:
            continue
        axes_part, _ = tok.split(":", 1)
        axes_part = axes_part.strip()
        if not axes_part:
            continue

        # axes_part is something like "x", "xy", "xyz"
        axes = tuple(axes_part)
        # Only consider valid axis letters
        if not set(axes) <= {"x", "y", "z"}:
            continue

        if len(axes) == 3 and set(axes) == {"x", "y", "z"}:
            has_xyz = True
            grouped_axes_set.update(axes)
        elif len(axes) > 1:
            grouped_axes_set.update(axes)

    scan_mode = "coupled" if has_xyz else "grouped"
    grouped_axes = sorted(grouped_axes_set)
    return scan_mode, grouped_axes


def _grid_from(a: float, b: float, s: float):
    """
    Build an inclusive floating-point grid [a, b] with step s.

    A small numerical tolerance is used so that the nominal end point b is
    included even under floating point accumulation. A guard prevents
    pathological loops for tiny step sizes.

    Parameters
    ----------
    a, b : float
        Start and end values (inclusive).
    s : float
        Step size (must be > 0).

    Returns
    -------
    list[float]
        Inclusive grid values.
    """
    vals, v, guard = [], float(a), 0
    while v <= b + 1e-12 and guard < 10000:
        vals.append(round(v, 12)); v += s; guard += 1
    if abs(vals[-1] - b) > 1e-8:
        vals.append(round(float(b), 12))
    return vals


def _parse_scan_groups(spec: str):
    """Parse the grouped/coupled scan syntax used by execution and metadata."""
    spec = (spec or "").strip()
    if not spec:
        return [], "grouped"
    parts = [part.strip() for part in spec.split(";") if part.strip()]
    parsed = []
    for token in parts:
        match = re.match(
            r"^([xyz]{1,3}):\s*([0-9]*\.?[0-9]+)\s*,\s*"
            r"([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*$",
            token,
        )
        if not match:
            continue
        axes_token, start, stop, step = (
            match.group(1),
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )
        axes = tuple(sorted(set(axes_token)))
        parsed.append(
            {
                "axes": axes,
                "grid": _grid_from(start, stop, step),
            }
        )

    for group in reversed(parsed):
        if group["axes"] == tuple("xyz"):
            return [group], "coupled_xyz"

    last_for_axis = {}
    for group in parsed:
        for axis in group["axes"]:
            last_for_axis[axis] = group

    groups, seen = [], set()
    for axis in "xyz":
        group = last_for_axis.get(axis)
        if group is not None and id(group) not in seen:
            seen.add(id(group))
            groups.append(group)
    return groups, "grouped"


def _iter_scan_combinations(groups):
    """Yield ``(pe_x, pe_y, pe_z)`` values for parsed scan groups."""
    if not groups:
        yield (None, None, None)
        return
    grids = [group["grid"] for group in groups]
    for choices in itertools.product(*grids):
        values = {"x": None, "y": None, "z": None}
        for group, value in zip(groups, choices):
            for axis in group["axes"]:
                values[axis] = float(value)
        yield values["x"], values["y"], values["z"]


def _percolation_axis_metadata(
    scan_spec: str,
    edges_table: np.ndarray,
    axes_map: list[str],
) -> dict:
    """Build JSON-compatible scientific metadata for a scan."""
    groups, parsed_mode = _parse_scan_groups(scan_spec)
    varied_axes = tuple(
        axis for axis in "xyz" if any(axis in group["axes"] for group in groups)
    )
    fixed_axes = tuple(axis for axis in "xyz" if axis not in varied_axes)
    if len(groups) == 1:
        mode = "1d_" + "".join(groups[0]["axes"])
    elif len(groups) == 2:
        mode = "2d_" + "".join(varied_axes)
    else:
        mode = parsed_mode

    axes_array = np.asarray(axes_map)
    fixed_values: dict[str, float | None] = {}
    for axis in fixed_axes:
        mask = (axes_array == axis) & (edges_table["blocked"] == 0)
        axis_rows = edges_table[mask]
        value = None
        if axis_rows.size and not np.any(axis_rows["directed"] == 1):
            unique = np.unique(axis_rows["pe"])
            finite = unique[np.isfinite(unique)]
            if finite.size == 1:
                value = float(finite[0])
        fixed_values[axis] = value

    return {
        "scan_spec": scan_spec,
        "mode": mode,
        "varied_axes": list(varied_axes),
        "fixed_axes": list(fixed_axes),
        "varied_axis_groups": [list(group["axes"]) for group in groups],
        "fixed_axis_values": fixed_values,
    }

def _as_bool(s: str) -> bool:
    """
    Interpret a user-provided string as a boolean.

    Parameters
    ----------
    s : str
        Input token.

    Returns
    -------
    bool
        True for common truthy values ("1","true","yes","y","on"), else False.
    """
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_git_sha(root: str | None = None) -> str | None:
    """
    Retrieve the current Git commit SHA for provenance metadata.

    Parameters
    ----------
    root : str or None
        Path to the repository root. If None, the repo root is inferred
        from the location of this script.

    Returns
    -------
    str or None
        Short Git commit SHA if available, otherwise None.
    """

    if root is None:
        root = str(_REPO_ROOT)
    git_dir = os.path.join(root, ".git")
    try:
        with open(os.path.join(git_dir, "HEAD"), "r", encoding="utf-8") as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref_path = head.split(" ", 1)[1].strip()
            with open(os.path.join(git_dir, ref_path), "r", encoding="utf-8") as f:
                sha = f.read().strip()
        else:
            sha = head
        sha = sha.strip()
        if not sha:
            return None
        return sha[:12]
    except Exception:
        # Any problem (no .git, detached HEAD, etc.) → simply skip SHA.
        return None


def _regular_lattice_shape(nodes) -> tuple[int, int, int] | None:
    """Infer a complete Cartesian lattice from validated node coordinates."""

    coordinates = {
        (
            int(getattr(node, "x")),
            int(getattr(node, "y")),
            int(getattr(node, "z")),
        )
        for node in nodes
    }
    if len(coordinates) != len(nodes) or not coordinates:
        return None
    axes = tuple(
        tuple(sorted({coordinate[index] for coordinate in coordinates}))
        for index in range(3)
    )
    expected = set(itertools.product(*axes))
    if coordinates != expected:
        return None
    return tuple(len(axis) for axis in axes)


def _pf_node_provenance(nodes, input_database: str | Path) -> dict:
    """Return a stable snapshot of the authoritative node probabilities."""

    ordered = sorted(
        (
            int(getattr(node, "id")),
            float(getattr(node, "pf_node")),
        )
        for node in nodes
    )
    digest = hashlib.sha256()
    for node_id, probability in ordered:
        digest.update(struct.pack(">qd", node_id, probability))

    values = [probability for _, probability in ordered]
    unique_values = sorted(set(values))
    uniform = len(unique_values) == 1
    return {
        "schema_version": 1,
        "source": "input_database.nodes.pf_node",
        "input_database": str(Path(input_database).resolve()),
        "authoritative_field": "pf_node",
        "node_count": len(ordered),
        "uniform": uniform,
        "uniform_value": unique_values[0] if uniform else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "unique_value_count": len(unique_values),
        "values_sha256": digest.hexdigest(),
        "sha256_encoding": (
            "concatenated big-endian int64 node id and float64 pf_node, "
            "ordered by node id"
        ),
    }


def _stationary_model_provenance(nodes, edges) -> dict:
    """Return a stable fingerprint of the stationary model except damage.

    The notional-damage set is deliberately excluded so otherwise identical
    undamaged and damaged runs can be paired safely during robustness
    post-processing.
    """

    node_records = sorted(
        (
            int(getattr(node, "id")),
            int(getattr(node, "x")),
            int(getattr(node, "y")),
            int(getattr(node, "z")),
            float(getattr(node, "pf_node")),
        )
        for node in nodes
    )
    edge_records = sorted(
        (
            int(getattr(edge, "id")),
            int(getattr(edge, "org")),
            int(getattr(edge, "dst")),
            str(getattr(edge, "dir")),
            int(getattr(edge, "directed")),
            int(getattr(edge, "blocked")),
            (
                None
                if getattr(edge, "pe", None) is None
                else float(getattr(edge, "pe"))
            ),
            (
                None
                if getattr(edge, "pe_fwd", None) is None
                else float(getattr(edge, "pe_fwd"))
            ),
            (
                None
                if getattr(edge, "pe_rev", None) is None
                else float(getattr(edge, "pe_rev"))
            ),
        )
        for edge in edges
    )
    canonical = json.dumps(
        {
            "contract": "stationary-v1",
            "nodes": node_records,
            "edges": edge_records,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "contract": "stationary-v1",
        "node_count": len(node_records),
        "edge_count": len(edge_records),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "excludes": ["notional_damage"],
    }


def _insert_run(conn, scenario_id: int, samples: int, threshold: int, seed: int, log_attempts: bool, notes: dict) -> int:
    """
    Create a new run record and return its database id.

    Parameters
    ----------
    conn : sqlite3.Connection
        Output DB connection.
    scenario_id : int
        Scenario identifier for bookkeeping.
    samples : int
        Number of Monte Carlo samples.
    threshold : int
        Failure threshold used for p_ge_thresh in collapse_probs.
    seed : int
        RNG seed.
    log_attempts : bool
        Whether to record edge attempts.
    notes : dict
        JSON-serializable metadata stored with the run.

    Returns
    -------
    int
        Primary key id of the inserted run.
    """
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO runs(scenario_id, samples, threshold, seed, log_attempts, started_at, finished_at, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (scenario_id, samples, threshold, seed, int(log_attempts),
         timestamp(), None, json.dumps(notes, separators=(",",":")))
    )
    conn.commit()
    return cur.lastrowid

def _finalize_run(conn, run_id: int) -> None:
    """
    Mark a run as finished by writing its finished_at timestamp.

    Parameters
    ----------
    conn : sqlite3.Connection
        Output DB connection.
    run_id : int
        Run identifier.
    """
    conn.execute("UPDATE runs SET finished_at=? WHERE id=?", (timestamp(), run_id))
    conn.commit()

def _write_pmf(conn, run_id: int, counts: np.ndarray) -> None:
    """
    Write the empirical PMF of number of failed nodes.

    Parameters
    ----------
    conn : sqlite3.Connection
        Output DB connection.
    run_id : int
        Run identifier.
    counts : np.ndarray
        counts[k] is the number of samples with exactly k failed nodes.
    """
    total = int(counts.sum())
    rows = [(run_id, int(i), int(c), (float(c)/total if total else 0.0)) for i, c in enumerate(counts)]
    exec_many(conn, "INSERT INTO pmf(run_id, n_failed, count, p) VALUES (?, ?, ?, ?)", rows)

def _write_collapse_probs(conn, run_id: int, counts: np.ndarray, threshold: int) -> None:
    """
    Write aggregate collapse probabilities derived from PMF counts.

    Metrics:
    - p_any: probability of at least one failed node
    - p_full: probability all nodes failed
    - p_ge_thresh: probability failures >= threshold

    Parameters
    ----------
    conn : sqlite3.Connection
        Output DB connection.
    run_id : int
        Run identifier.
    counts : np.ndarray
        PMF counts by number of failed nodes.
    threshold : int
        Threshold for p_ge_thresh.
    """
    total = int(counts.sum())
    p_any = 1.0 - (counts[0] / total if total else 0.0)
    p_full = (counts[-1] / total) if total else 0.0
    ge = counts[threshold:].sum() if threshold < counts.size else 0
    p_ge = (ge / total) if total else 0.0
    conn.execute("INSERT INTO collapse_probs(run_id, p_any, p_full, p_ge_thresh) VALUES (?, ?, ?, ?)",
                 (run_id, float(p_any), float(p_full), float(p_ge)))
    conn.commit()

def _write_edge_stats(
    conn,
    run_id: int,
    mats,
    successes_nnz: np.ndarray,
    attempts_nnz: np.ndarray | None,
    arc_to_edgeid: dict[tuple[int, int], int],
) -> str:
    """
    Write edge usage using *input DB edges.id* as edge_id.

    Works with both real mats and DummyMats by using mats.P (CSR) only.
    Aggregates over both directions into the same undirected DB edge id because
    output tables key on (run_id, edge_id).
    """
    P = mats.P

    # Fast path: real CSR matrices (scipy.sparse)
    has_csr = hasattr(P, "indptr") and hasattr(P, "indices") and hasattr(P, "shape")

    # Fallback path: tests may use DummyP with only .nnz (no CSR structure)
    if not has_csr:
        # Dummy fallback path (no CSR structure)
        rows = [(int(run_id), int(k + 1), int(s)) for k, s in enumerate(successes_nnz)]
        exec_many(conn, "INSERT INTO edge_usage(run_id, edge_id, successes) VALUES (?, ?, ?)", rows)
        
        if attempts_nnz is not None:
            rows2 = [(int(run_id), int(k + 1), int(a)) for k, a in enumerate(attempts_nnz)]
            exec_many(conn, "INSERT INTO edge_attempts(run_id, edge_id, attempts) VALUES (?, ?, ?)", rows2)
        
        return "dummy_fallback_nnz_index_plus_1"


    indptr = P.indptr
    indices = P.indices
    n = P.shape[0]

    succ_sum: dict[int, int] = {}
    att_sum: dict[int, int] = {} if attempts_nnz is not None else None

    k = 0  # index over nnz, matching successes_nnz / attempts_nnz
    for i in range(n):
        for pos in range(indptr[i], indptr[i + 1]):
            j = int(indices[pos])
            edge_id = arc_to_edgeid.get((int(i), int(j)))
            if edge_id is not None:
                succ_sum[edge_id] = succ_sum.get(edge_id, 0) + int(successes_nnz[k])
                if att_sum is not None:
                    att_sum[edge_id] = att_sum.get(edge_id, 0) + int(attempts_nnz[k])
            k += 1

    rows = [(int(run_id), int(edge_id), int(val)) for edge_id, val in succ_sum.items()]
    exec_many(conn, "INSERT INTO edge_usage(run_id, edge_id, successes) VALUES (?, ?, ?)", rows)

    if att_sum is not None:
        rows2 = [(int(run_id), int(edge_id), int(val)) for edge_id, val in att_sum.items()]
        exec_many(conn, "INSERT INTO edge_attempts(run_id, edge_id, attempts) VALUES (?, ?, ?)", rows2)
    return "csr_arc_to_edgeid_aggregated"

def _write_node_influence(conn, run_id: int, node_ids, influence: np.ndarray) -> None:
    """
    Write per-node influence values to the output database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Output DB connection.
    run_id : int
        Run identifier.
    node_ids : sequence[int]
        External node identifiers (DB ids if available, else indices).
    influence : np.ndarray
        Influence score per node (same order as node_ids).
    """
    rows = [
        (int(run_id), int(node_id), float(val))
        for node_id, val in zip(node_ids, influence)
    ]
    exec_many(
        conn,
        "INSERT INTO node_influence(run_id, node_id, influence) VALUES (?, ?, ?)",
        rows,
    )
    
def _run_percolation_multi(
    L,
    edges_rows,
    axes_map,
    N_nodes: int,
    out_conn,
    run_id: int,
    samples: int,
    seed: int,
    scan_spec: str,
    damage_mask: np.ndarray,
    percolation_workers: int = 1,
):
    """
    Execute a multi-axis percolation scan.

    For each point in the percolation parameter space, this function:
    - modifies node/edge failure probabilities,
    - runs Monte Carlo propagation,
    - and records aggregated results to the output database.

    The resolved notional-damage set is applied before sampling undamaged
    nodes, using the same package function as the base run.

    Parameters
    ----------
    L : scipy.sparse matrix
        System adjacency / propagation matrix.
    edges_rows : list
        Edge records loaded from the input database.
    axes_map : dict
        Mapping from scan axes to node/edge parameters.
    N_nodes : int
        Number of nodes in the system.
    out_conn : sqlite3.Connection
        Open connection to the output database.
    run_id : int
        Identifier of the current run.
    samples : int
        Number of Monte Carlo samples per scan point.
    seed : int
        RNG seed.
    scan_spec : str
        Encoded percolation scan specification.
    damage_mask : np.ndarray (bool)
        Boolean mask of nodes lost simultaneously at pseudo-time t=0.
    """

    edges_dtype = np.dtype([
        ("org",        np.int64),
        ("dst",        np.int64),
        ("pe_fwd",     np.float64),
        ("pe_rev",     np.float64),
        ("pe",         np.float64),
        ("directed",   np.int32),
        ("blocked",    np.int32),
    ])
    base = np.array(edges_rows, dtype=edges_dtype)

    axes_arr = np.array(axes_map)

    # --- parse grouped/coupled spec → combos (pe_x, pe_y, pe_z)
    groups, _mode = _parse_scan_groups(scan_spec)
    combos = list(_iter_scan_combinations(groups))
    if not combos:
        return
    if len(combos) > 5000:
        raise RuntimeError(
            f"Too many percolation combinations ({len(combos)}). Reduce grid density."
        )

    dir_mask = base["directed"] == 1
    und_mask = ~dir_mask
    scanned_axes = {axis for group in groups for axis in group["axes"]}
    if scanned_axes:
        directed_scanned = dir_mask & np.isin(axes_arr, sorted(scanned_axes))
        if np.any(directed_scanned):
            edge_indices = np.nonzero(directed_scanned)[0].tolist()
            raise ValueError(
                "Scalar percolation scans cannot modify directed edges. "
                f"Directed edge row indices on scanned axes: {edge_indices}."
            )

    # Pre-extract pf once (doesn't depend on combos)
    pf = np.array([getattr(n, "pf_node", 0.0) for n in L.nodes], dtype=float)

    # Build tasks (one task per scan point)
    tasks = [
        (
            int(run_id),
            int(N_nodes),
            base,
            axes_arr,
            dir_mask,
            und_mask,
            pf,
            damage_mask,
            int(samples),
            int(seed),
            pe_x,
            pe_y,
            pe_z,
        )
        for (pe_x, pe_y, pe_z) in combos
    ]

    workers = max(1, int(percolation_workers or 1))

    if workers == 1 or len(tasks) == 1:
        rows = [_percolation_worker_one_point(t) for t in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_percolation_worker_one_point, t) for t in tasks]
            for f in as_completed(futures):
                rows.append(f.result())

    # Preserve a stable insertion order even when scan tasks finish out of order.
    rows.sort(key=lambda row: row[1:4])

    # Bulk insert in one transaction (single-writer)
    out_conn.execute("BEGIN;")
    _write_percolation_multi_bulk(out_conn, rows)
    _write_percolation_pmf_bulk(out_conn, rows)
    out_conn.commit()


def main() -> None:
    """
    Main CLI entry point.

    This function:
    - parses CLI arguments,
    - loads configuration and input data,
    - resolves precedence between CLI, config, and database defaults,
    - executes either a single simulation or a percolation scan,
    - and writes results and metadata to the output database.

    All side effects (I/O, database writes, logging) originate here.
    """

    # Defaults (YAML-free)
    db_default = "input/input.db"
    outdb_default = "output/results.db"
    seed_default = 42
    samples_default = 2000

    ap = argparse.ArgumentParser(description="Stationary probabilistic building model simulation")
    ap.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    ap.add_argument(
        "--db",
        default=db_default,
        help="Input DB (nodes, edges, and initial_damage)",
    )
    ap.add_argument("--scenario-id", type=int, default=0)
    ap.add_argument("--samples", type=int, default=samples_default)
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--seed", type=int, default=seed_default)
    ap.add_argument("--log-attempts", default="no", help="yes/no")
    ap.add_argument("--outdb", default=outdb_default, help="Output DB path")
    ap.add_argument(
        "--percolation-scan",
        default="",
        help="Multi-axis sweep. Use ';' to separate groups. Examples: 'x:0,1,0.25; y:0.5,1,0.25' or grouped axes 'yz:0,1,0.2'. Use 'xyz:0,1,0.1' for diagonal coupling (x=y=z=t).",
    )
    ap.add_argument(
        "--percolation-workers",
        type=int,
        default=1,
        help="Number of worker processes for percolation scan (1 = sequential).",
    )

    ap.add_argument(
        "--mc-workers",
        type=int,
        default=1,
        help="Number of worker processes for Monte Carlo sampling (1 = sequential).",
    )
    ap.add_argument(
        "--mc-chunk-size",
        type=int,
        default=0,
        help=(
            "Realizations per Monte Carlo task. Use 0 for an automatic size. "
            "Chunking affects performance only, not random streams or results."
        ),
    )

    ap.add_argument(
        "--damage-node-ids",
        default="",
        help=(
            "Comma-separated canonical node IDs for simultaneous pre-run "
            "notional damage. Cannot be combined with non-empty database "
            "initial_damage."
        ),
    )
    ap.add_argument(
        "--damage-coords",
        default="",
        help=(
            "Semicolon-separated x,y,z coordinates for simultaneous pre-run "
            "notional damage, e.g. '0,0,0;1,2,0'. Coordinates are resolved "
            "strictly to node IDs and cannot be combined with non-empty "
            "database initial_damage."
        ),
    )
    args = ap.parse_args()

    # Percolation enablement (YAML-free)
    args.percolation_scan = str(args.percolation_scan or "")
    percolation_enabled = bool(args.percolation_scan.strip())

    # Parse runtime notional-damage node IDs (comma-separated).
    damage_ids_text = (args.damage_node_ids or "").strip()
    runtime_damage_ids: list[int] = []
    if damage_ids_text:
        for tok in damage_ids_text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                runtime_damage_ids.append(int(tok))
            except ValueError:
                raise SystemExit(
                    f"Invalid node id in --damage-node-ids: {tok!r}"
                )
    
    # Parse runtime notional-damage coordinates.
    damage_coords_text = (args.damage_coords or "").strip()
    runtime_damage_coordinates: list[tuple[int, int, int]] = []
    if damage_coords_text:
        for chunk in damage_coords_text.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [p.strip() for p in chunk.split(",")]
            if len(parts) != 3:
                raise SystemExit(
                    "Invalid coordinate in --damage-coords "
                    f"(expected x,y,z): {chunk!r}"
                )
            try:
                x, y, z = (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                raise SystemExit(
                    "Invalid coordinate component in --damage-coords: "
                    f"{chunk!r}"
                )
            runtime_damage_coordinates.append((x, y, z))
    
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1.")
    if args.seed < 0:
        raise SystemExit("--seed must be a non-negative integer.")
    if args.mc_chunk_size < 0:
        raise SystemExit("--mc-chunk-size must be >= 0.")

    log_attempts = _as_bool(args.log_attempts)
    mc_workers = max(1, int(getattr(args, "mc_workers", 1) or 1))
    if log_attempts and mc_workers > 1:
        raise SystemExit(
            "Policy A: --log-attempts yes is not supported with --mc-workers > 1. "
            "Use --log-attempts no or set --mc-workers 1."
        )
    mc_chunk_size = _resolve_mc_chunk_size(
        args.samples,
        mc_workers,
        args.mc_chunk_size,
    )

    # Load lattice
    L = load_all(args.db)

    try:
        resolved_damage_node_ids = resolve_notional_damage(
            L.nodes,
            database_node_ids=L.damage_node_ids,
            node_ids=runtime_damage_ids,
            coordinates=runtime_damage_coordinates,
        )
        damage_mask = notional_damage_mask(
            L.nodes,
            resolved_damage_node_ids,
        )
    except NotionalDamageError as exc:
        raise SystemExit(str(exc)) from exc
    
    N = len(L.nodes)
    E = len(L.edges)


    # Build sparse transition matrix once
    # Expect edges table (org,dst) to be 0-based ids in loader order.
    # If your loader outputs 1-based ids, adjust indices here accordingly.
    
    # Build a structured array with named fields expected by pbm.network.propagation
    edges_dtype = np.dtype([
        ("org",        np.int64),
        ("dst",        np.int64),
        ("pe_fwd",     np.float64),
        ("pe_rev",     np.float64),
        ("pe",         np.float64),
        ("directed",   np.int32),
        ("blocked",    np.int32),
    ])
    
    def _to_float_or_nan(x):
        """
        Convert a value to float, returning NaN for missing/unparseable inputs.
    
        NaN is used only as the structured-array representation of SQL NULL.
        """
        if x is None:
            return np.nan

        try:
            return float(x)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Edge probability must be numeric or NULL; got {x!r}."
            ) from exc
    
    # 1) Map DB node primary keys -> compact indices 0..N-1
    if hasattr(L.nodes[0], "id"):
        id_to_idx = {int(n.id): i for i, n in enumerate(L.nodes)}
        def to_idx(db_id: int) -> int | None:
            return id_to_idx.get(int(db_id), None)
        node_ids = [None] * N
        for node_id, idx in id_to_idx.items():
            if 0 <= idx < N:
                node_ids[idx] = int(node_id)
        for idx in range(N):
            if node_ids[idx] is None:
                node_ids[idx] = idx
    else:
        id_to_idx = None
        def to_idx(db_id: int) -> int | None:
            return int(db_id)
        node_ids = list(range(N))
    
    # 2) Collect contract-valid edge rows; missing endpoints are errors.
    rows = []
    axes_map = []
    arc_to_edgeid: dict[tuple[int, int], int] = {}  # (org_idx, dst_idx) -> DB edges.id
    
    for e in L.edges:
        org_raw = int(getattr(e, "org"))
        dst_raw = int(getattr(e, "dst"))
        org = to_idx(org_raw)
        dst = to_idx(dst_raw)
        if org is None or dst is None:
            raise ValueError(
                f"Edge id={getattr(e, 'id', None)} references a missing node "
                f"endpoint ({org_raw}, {dst_raw})."
            )
        # Map BOTH directions of this undirected DB edge id to the same edges.id
        edge_db_id = int(getattr(e, "id"))
        if (int(org), int(dst)) in arc_to_edgeid or (int(dst), int(org)) in arc_to_edgeid:
            raise ValueError(
                f"Duplicate directed arc encountered while mapping edge id={edge_db_id}."
            )
        arc_to_edgeid[(int(org), int(dst))] = edge_db_id
        arc_to_edgeid[(int(dst), int(org))] = edge_db_id
        

        directed = int(getattr(e, "directed"))
        blocked = int(getattr(e, "blocked"))
        pe = _to_float_or_nan(getattr(e, "pe", None))
        pe_fwd = _to_float_or_nan(getattr(e, "pe_fwd", None))
        pe_rev = _to_float_or_nan(getattr(e, "pe_rev", None))

        rows.append((org, dst, pe_fwd, pe_rev, pe, directed, blocked))
        axes_map.append(str(getattr(e, "dir", "")))   # 'x' / 'y' / 'z'
    
    # 3) Create the structured array
    if rows:
        edges_tbl = np.array(rows, dtype=edges_dtype)
    else:
        edges_tbl = np.zeros(0, dtype=edges_dtype)
    
    # 4) Sanity guard: ensure indices are valid w.r.t. N
    max_idx = int(edges_tbl["org"].max()) if edges_tbl.size else -1
    max_idx = max(max_idx, int(edges_tbl["dst"].max())) if edges_tbl.size else max_idx
    assert (not edges_tbl.size) or (0 <= edges_tbl["org"].min() and 0 <= edges_tbl["dst"].min()), "Negative node index"
    assert (not edges_tbl.size) or (max_idx < N), f"Edge references node index {max_idx} but N={N}"

    mats = build_transition_matrix(N, edges_tbl)

    # Prepare output DB. Filename-only paths are valid and resolve below the
    # current working directory.
    out_db_path = Path(args.outdb).expanduser()
    out_db_path.parent.mkdir(parents=True, exist_ok=True)
    out_conn = connect(out_db_path)
    
    # Authoritative output schema packaged under pbm/resources/sql/output_schema.sql
    apply_schema(out_conn, "output")

    # Populate id-only mirrors (nodes/edges) to satisfy optional foreign keys
    # in results.db (used by node_influence / edge_usage / edge_attempts).
    #
    # These mirrors are intentionally lightweight; they are NOT full snapshots
    # of the input model.
    try:
        exec_many(out_conn, "INSERT OR IGNORE INTO nodes(id) VALUES (?)", [(int(n.id),) for n in L.nodes])
        exec_many(out_conn, "INSERT OR IGNORE INTO edges(id) VALUES (?)", [(int(e.id),) for e in L.edges])
    except Exception:
        # Mirrors are optional; ignore if schema omits them.
        pass


    notes = dict(
        host=socket.gethostname(),
        cwd=os.getcwd(),
        datetime=datetime.now().isoformat(timespec="seconds"),
        engine="pbm.network.propagation.simulate_once_mc",
        package_version=__version__,
        model_release="stationary-v1",
    )

    notes["mc_workers"] = int(mc_workers)
    notes["mc_chunk_size"] = int(mc_chunk_size)
    notes["rng_contract"] = RNG_CONTRACT_VERSION
    notes["rng_stream_key"] = ["base_seed", "realization_index"]
    
    notes["pe_semantics"] = {
        "undirected_uses": "pe in both directions; pe_fwd and pe_rev are NULL",
        "directed_uses": "required pe_fwd and pe_rev; pe is NULL",
        "blocked": "blocked=1 suppresses transmission in both directions",
    }
    notes["edge_usage_edge_id_source"] = "input_db_edges.id (aggregated over both directions)"
    notes["edge_usage_write_method"] = None  # will be filled after _write_edge_stats()
    notes["notional_damage_node_ids"] = list(resolved_damage_node_ids)
    notes["notional_damage_source"] = (
        "database"
        if L.damage_node_ids
        else "runtime"
        if runtime_damage_ids or runtime_damage_coordinates
        else "none"
    )
    notes["pf_node_provenance"] = _pf_node_provenance(L.nodes, args.db)
    notes["stationary_model_provenance"] = _stationary_model_provenance(
        L.nodes,
        L.edges,
    )
    lattice_shape = _regular_lattice_shape(L.nodes)
    notes["regular_lattice_shape"] = (
        list(lattice_shape) if lattice_shape is not None else None
    )

    git_sha = _get_git_sha()
    if git_sha is not None:
        notes["git_sha"] = git_sha
        
    if percolation_enabled:
        scan_mode, grouped_axes = _parse_scan_spec_for_metadata(args.percolation_scan)
        axis_metadata = _percolation_axis_metadata(
            args.percolation_scan,
            edges_tbl,
            axes_map,
        )
        notes.update(
            percolation_multi=True,
            scan_spec=args.percolation_scan,
            scan_mode=scan_mode,
            grouped_axes=grouped_axes,
            percolation_scan_metadata=axis_metadata,
            percolation_workers=max(1, int(args.percolation_workers or 1)),
        )


    run_id = _insert_run(
        out_conn,
        args.scenario_id,
        args.samples,
        args.threshold,
        args.seed,
        log_attempts,
        notes,
    )

    # Compute static node influence
    infl = static_outflow_influence(mats)
    _write_node_influence(out_conn, run_id, node_ids, infl)
    
    try:
        # Aggregations
        counts = np.zeros(N + 1, dtype=np.int64)
        agg_succ = np.zeros(mats.P.nnz, dtype=np.int64)  # successes per nonzero edge
        agg_att = np.zeros(mats.P.nnz, dtype=np.int64) if log_attempts else None

        # Sample only undamaged nodes; apply the deterministic damage set once.
        pf = np.array([getattr(n, "pf_node", 0.0) for n in L.nodes], dtype=float)
            
        realization_ranges = _realization_ranges(
            args.samples,
            mc_chunk_size,
        )

        if mc_workers <= 1:
            for i0, i1 in realization_ranges:
                c_part, succ_part, att_part = _simulate_realization_range(
                    N,
                    mats,
                    pf,
                    damage_mask,
                    args.seed,
                    i0,
                    i1,
                    log_attempts=log_attempts,
                )
                counts += c_part
                agg_succ += succ_part
                if log_attempts and att_part is not None:
                    agg_att += att_part
        else:
            # Parallel MC (Policy A guarantees log_attempts=False here)
            n_workers = int(mc_workers)
            tasks = [
                (int(args.seed), int(i0), int(i1))
                for (i0, i1) in realization_ranges
            ]

            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_mc_worker_init,
                initargs=(int(N), edges_tbl, pf, damage_mask),
            ) as ex:
                futures = [ex.submit(_mc_worker_range, t) for t in tasks]
                for f in as_completed(futures):
                    c_part, succ_part, _ = f.result()
                    counts += c_part
                    agg_succ += succ_part

        # Write outputs
        _write_pmf(out_conn, run_id, counts)
        _write_collapse_probs(out_conn, run_id, counts, threshold=args.threshold)
        method = _write_edge_stats(out_conn, run_id, mats, agg_succ, agg_att, arc_to_edgeid)
        notes["edge_usage_write_method"] = method

        
        out_conn.execute(
            "UPDATE runs SET notes=? WHERE id=?",
            (json.dumps(notes, separators=(",",":")), int(run_id)),
        )
        out_conn.commit()

        
        if percolation_enabled:
            _run_percolation_multi(
                L, rows, axes_map, N, out_conn, run_id,
                samples=args.samples, seed=args.seed, scan_spec=args.percolation_scan,
                damage_mask=damage_mask,
                percolation_workers=args.percolation_workers,
            )

    finally:
        _finalize_run(out_conn, run_id)
        out_conn.close()

if __name__ == "__main__":
    main()
