"""Selection and validation of risk-equivalence percolation runs.

The verification notebook must not depend on a persistence-derived run ID.
This module selects the newest completed run that satisfies the scientific
inputs of the requested risk-equivalence calculation and validates redundant
sample-count and ``pf_node`` evidence before returning it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
import sqlite3
from typing import Any, Mapping, Sequence

import numpy as np

from .percolation_plots import (
    detect_sweep_mode,
    extract_1d_curves,
    load_percolation_multi,
    load_percolation_pmf_counts,
)


_PF_PROVENANCE_SOURCE = "input_database.nodes.pf_node"
_PF_HASH_ENCODING = (
    "concatenated big-endian int64 node id and float64 pf_node, "
    "ordered by node id"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CompatibleRunNotFoundError(ValueError):
    """Raised when no completed run satisfies the requested scientific inputs."""


@dataclass(frozen=True)
class RiskEquivalenceRunSelection:
    """Validated database run used by the risk-equivalence notebook."""

    run_id: int
    finished_at: str
    samples: int
    scan_samples: int
    sample_count_validated: bool
    lattice_shape: tuple[int, int, int]
    scan_spec: str
    scan_point_count: int
    scan_bounds: tuple[float, float]
    pf_node: float
    pf_node_source: str
    pf_node_values_sha256: str
    pf_provenance_validated: bool
    p_any_at_zero_observed: float
    p_any_at_zero_expected: float
    p_any_at_zero_z_score: float | None
    pf_empirical_check_resolved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _probability(value: object, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability in [0, 1].") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1].")
    return numeric


def _lattice_shape(value: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("lattice_shape must contain exactly (nx, ny, nz).")
    return tuple(
        _positive_integer(item, name=f"lattice_shape[{index}]")
        for index, item in enumerate(value)
    )


def _notes_mapping(raw_notes: object, *, run_id: int) -> Mapping[str, Any]:
    if not isinstance(raw_notes, str) or not raw_notes.strip():
        raise ValueError("missing runs.notes metadata")
    try:
        notes = json.loads(raw_notes)
    except json.JSONDecodeError as exc:
        raise ValueError("runs.notes is not valid JSON") from exc
    if not isinstance(notes, dict):
        raise ValueError("runs.notes must contain a JSON object")
    return notes


def _validate_axis_metadata(notes: Mapping[str, Any]) -> str:
    metadata = notes.get("percolation_scan_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("missing explicit percolation_scan_metadata")
    if metadata.get("mode") != "1d_xyz":
        raise ValueError("scan metadata is not mode='1d_xyz'")
    if metadata.get("varied_axes") != ["x", "y", "z"]:
        raise ValueError("scan metadata does not vary x, y, and z together")
    if metadata.get("varied_axis_groups") != [["x", "y", "z"]]:
        raise ValueError("scan metadata is not a single coupled xyz group")
    if metadata.get("fixed_axes") != []:
        raise ValueError("fully coupled xyz scan must not retain a fixed axis")
    if metadata.get("fixed_axis_values") != {}:
        raise ValueError("fully coupled xyz scan has unexpected fixed values")
    scan_spec = metadata.get("scan_spec")
    if not isinstance(scan_spec, str) or not scan_spec.strip():
        raise ValueError("scan metadata has no scan_spec provenance")
    return scan_spec


def _same_probability(
    actual: object,
    expected: float,
    *,
    name: str,
) -> float:
    value = _probability(actual, name=name)
    tolerance = max(1.0e-15, 16.0 * np.finfo(float).eps)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name}={value!r}, expected {expected!r}")
    return value


def _validate_pf_provenance(
    notes: Mapping[str, Any],
    *,
    expected_pf_node: float,
    expected_node_count: int,
) -> tuple[float, str, str]:
    provenance = notes.get("pf_node_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            "missing pf_node_provenance; re-run the scan with the current code"
        )
    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported pf_node_provenance schema_version")
    source = provenance.get("source")
    if source != _PF_PROVENANCE_SOURCE:
        raise ValueError("pf_node provenance has an unsupported source")
    if provenance.get("authoritative_field") != "pf_node":
        raise ValueError("pf_node provenance does not identify pf_node as authoritative")
    if provenance.get("node_count") != expected_node_count:
        raise ValueError(
            "pf_node provenance node_count does not match the lattice shape"
        )
    if provenance.get("uniform") is not True:
        raise ValueError("risk-equivalence verification requires uniform pf_node")
    if provenance.get("unique_value_count") != 1:
        raise ValueError("uniform pf_node provenance must have one unique value")

    value = _same_probability(
        provenance.get("uniform_value"),
        expected_pf_node,
        name="recorded uniform pf_node",
    )
    _same_probability(
        provenance.get("minimum"),
        expected_pf_node,
        name="recorded minimum pf_node",
    )
    _same_probability(
        provenance.get("maximum"),
        expected_pf_node,
        name="recorded maximum pf_node",
    )

    input_database = provenance.get("input_database")
    if not isinstance(input_database, str) or not input_database.strip():
        raise ValueError("pf_node provenance has no input-database source path")
    if provenance.get("sha256_encoding") != _PF_HASH_ENCODING:
        raise ValueError("pf_node provenance has an unsupported digest encoding")
    digest = provenance.get("values_sha256")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError("pf_node provenance has no valid SHA-256 digest")
    return value, source, digest


def _validate_damage_provenance(notes: Mapping[str, Any]) -> None:
    if notes.get("notional_damage_node_ids") != []:
        raise ValueError(
            "risk-equivalence verification requires no notional-damage nodes"
        )
    if notes.get("notional_damage_source") != "none":
        raise ValueError(
            "risk-equivalence verification requires notional_damage_source='none'"
        )


def _validate_candidate(
    conn: sqlite3.Connection,
    row: tuple[object, ...],
    *,
    expected_samples: int,
    expected_pf_node: float,
    lattice_shape: tuple[int, int, int],
    pe_crit: float,
) -> RiskEquivalenceRunSelection:
    run_id = int(row[0])
    run_samples = _positive_integer(row[1], name=f"run {run_id} samples")
    finished_at = row[2]
    if not isinstance(finished_at, str) or not finished_at.strip():
        raise ValueError("run is incomplete (finished_at is empty)")
    notes = _notes_mapping(row[3], run_id=run_id)

    recorded_shape = notes.get("regular_lattice_shape")
    if recorded_shape != list(lattice_shape):
        raise ValueError(
            f"regular_lattice_shape={recorded_shape!r}, "
            f"expected {list(lattice_shape)!r}"
        )
    if run_samples != expected_samples:
        raise ValueError(
            f"runs.samples={run_samples}, expected {expected_samples}"
        )
    scan_spec = _validate_axis_metadata(notes)
    _validate_damage_provenance(notes)

    node_count = int(np.prod(lattice_shape))
    pf_node, pf_source, pf_digest = _validate_pf_provenance(
        notes,
        expected_pf_node=expected_pf_node,
        expected_node_count=node_count,
    )

    data = load_percolation_multi(conn, run_id)
    if data is None:
        raise ValueError("run has no percolation_multi rows")
    mode, _ = detect_sweep_mode(data)
    if mode != "1d_xyz":
        raise ValueError(f"stored scan data has mode={mode!r}, not '1d_xyz'")
    pe, _, _, _ = extract_1d_curves(data, mode)
    if pe.size < 2:
        raise ValueError("fully coupled xyz scan must contain at least two points")

    tolerance = 16.0 * np.finfo(float).eps
    if pe[0] > tolerance:
        raise ValueError("fully coupled xyz scan does not include p_e=0")
    if pe[-1] + tolerance < pe_crit:
        raise ValueError(
            f"scan ends at p_e={float(pe[-1])}, below pe_crit={pe_crit}"
        )

    sample_rows = conn.execute(
        """
        SELECT DISTINCT n_samples
        FROM percolation_multi
        WHERE run_id = ?
        ORDER BY n_samples;
        """,
        (run_id,),
    ).fetchall()
    scan_sample_counts = [int(value[0]) for value in sample_rows]
    if scan_sample_counts != [expected_samples]:
        raise ValueError(
            "percolation_multi n_samples values are "
            f"{scan_sample_counts!r}, expected [{expected_samples}]"
        )

    pmf_counts = load_percolation_pmf_counts(conn, run_id, data)
    pmf_sample_counts = np.sum(pmf_counts, axis=1)
    if not np.all(pmf_sample_counts == expected_samples):
        raise ValueError(
            "percolation_pmf sample totals do not match runs.samples"
        )

    zero_indices = np.flatnonzero(np.isclose(pe, 0.0, rtol=0.0, atol=tolerance))
    if zero_indices.size != 1:
        raise ValueError("fully coupled xyz scan must contain exactly one p_e=0 point")
    zero_counts = pmf_counts[int(zero_indices[0])]
    p_any_observed = float(1.0 - zero_counts[0] / expected_samples)
    p_any_expected = float(1.0 - (1.0 - pf_node) ** node_count)
    standard_error = math.sqrt(
        p_any_expected * (1.0 - p_any_expected) / expected_samples
    )
    expected_any = expected_samples * p_any_expected
    expected_none = expected_samples * (1.0 - p_any_expected)
    empirical_resolved = expected_any >= 30.0 and expected_none >= 30.0
    z_score = (
        float((p_any_observed - p_any_expected) / standard_error)
        if standard_error > 0.0
        else None
    )
    if empirical_resolved and (z_score is None or abs(z_score) > 6.0):
        raise ValueError(
            "p_e=0 PMF is inconsistent with recorded pf_node "
            f"(observed P(N_f>=1)={p_any_observed:.8g}, "
            f"expected {p_any_expected:.8g}, z={z_score:.3g})"
        )

    return RiskEquivalenceRunSelection(
        run_id=run_id,
        finished_at=finished_at,
        samples=run_samples,
        scan_samples=scan_sample_counts[0],
        sample_count_validated=True,
        lattice_shape=lattice_shape,
        scan_spec=scan_spec,
        scan_point_count=int(pe.size),
        scan_bounds=(float(pe[0]), float(pe[-1])),
        pf_node=pf_node,
        pf_node_source=pf_source,
        pf_node_values_sha256=pf_digest,
        pf_provenance_validated=True,
        p_any_at_zero_observed=p_any_observed,
        p_any_at_zero_expected=p_any_expected,
        p_any_at_zero_z_score=z_score,
        pf_empirical_check_resolved=empirical_resolved,
    )


def select_latest_compatible_xyz_run(
    conn: sqlite3.Connection,
    *,
    expected_samples: int,
    expected_pf_node: float,
    lattice_shape: Sequence[int],
    pe_crit: float,
) -> RiskEquivalenceRunSelection:
    """Return the newest completed fully coupled run matching paper inputs.

    Compatibility requires explicit coupled-``xyz`` metadata and data, the
    requested regular-lattice shape and scan coverage, no notional damage,
    identical sample counts in ``runs``, ``percolation_multi``, and every scan
    PMF, and versioned ``pf_node`` provenance matching ``expected_pf_node``.
    The ``p_e=0`` PMF provides an independent statistical fingerprint of that
    recorded node-failure probability when the expected event counts permit it.
    """

    samples = _positive_integer(expected_samples, name="expected_samples")
    probability = _probability(expected_pf_node, name="expected_pf_node")
    shape = _lattice_shape(lattice_shape)
    critical = _probability(pe_crit, name="pe_crit")
    if critical <= 0.0:
        raise ValueError("pe_crit must lie in (0, 1].")

    required_tables = {"runs", "percolation_multi", "percolation_pmf"}
    actual_tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
    }
    missing = sorted(required_tables - actual_tables)
    if missing:
        raise CompatibleRunNotFoundError(
            f"Results database is missing required table(s): {missing}."
        )

    rows = conn.execute(
        """
        SELECT id, samples, finished_at, notes
        FROM runs
        ORDER BY id DESC;
        """
    ).fetchall()
    if not rows:
        raise CompatibleRunNotFoundError("Results database contains no runs.")

    rejections: list[str] = []
    for row in rows:
        run_id = int(row[0])
        try:
            return _validate_candidate(
                conn,
                row,
                expected_samples=samples,
                expected_pf_node=probability,
                lattice_shape=shape,
                pe_crit=critical,
            )
        except (ValueError, sqlite3.DatabaseError) as exc:
            rejections.append(f"run {run_id}: {exc}")

    details = "; ".join(rejections[:8])
    if len(rejections) > 8:
        details += f"; and {len(rejections) - 8} older run(s)"
    raise CompatibleRunNotFoundError(
        "No completed compatible fully coupled xyz run was found for "
        f"samples={samples}, pf_node={probability}, "
        f"lattice_shape={shape}, and pe_crit={critical}. {details}"
    )
