"""Public single-realization API for the stationary model."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pbm.network.propagation import build_transition_matrix, simulate_once_mc
from pbm.sim.notional_damage import (
    apply_notional_damage,
    notional_damage_mask,
    resolve_notional_damage,
)


def _edges_structured_table(L):
    """
    Build a structured edges table with compact node indices 0..N-1, and the
    fields expected by pbm.network.propagation.build_transition_matrix.
    Rejects any edge referencing a missing node.
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

    def _to_float_or_nan(x):
        if x is None:
            return np.nan
        try:
            return float(x)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Edge probability must be numeric or None; got {x!r}.") from exc

    N = len(L.nodes)
    if N == 0:
        if L.edges:
            raise ValueError("Edges cannot be present when the lattice has no nodes.")
        return np.zeros(0, dtype=edges_dtype)

    # Map DB node ids -> compact indices (0..N-1)
    if hasattr(L.nodes[0], "id"):
        id_to_idx = {int(n.id): i for i, n in enumerate(L.nodes)}

        def to_idx(db_id):
            return id_to_idx.get(int(db_id))
    else:
        def to_idx(db_id):
            return int(db_id)  # assume already 0-based contiguous

    rows = []
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

        rows.append((
            org,
            dst,
            _to_float_or_nan(getattr(e, "pe_fwd", None)),
            _to_float_or_nan(getattr(e, "pe_rev", None)),
            _to_float_or_nan(getattr(e, "pe", None)),
            int(getattr(e, "directed")),
            int(getattr(e, "blocked")),
        ))

    if rows:
        return np.array(rows, dtype=edges_dtype)
    return np.zeros(0, dtype=edges_dtype)


def propagate(init_mask, L, rng, log_attempts: bool = False):
    """
    Propagate one already-resolved stationary initial-failure state.

    Args:
        init_mask (np.ndarray[bool]): initial failed nodes for this realization
        L: loaded lattice (from pbm.io.loader.load_all)
        rng: numpy Generator
        log_attempts (bool): return attempts per edge

    Returns:
        failed_mask (bool[N]), successes_per_edge (int[E_nnz]), attempts_per_edge or None
    """
    N = len(L.nodes)
    edges_tbl = _edges_structured_table(L)
    mats = build_transition_matrix(N, edges_tbl)

    res = simulate_once_mc(init_mask, mats, rng=rng, log_attempts=log_attempts)
    failed_mask = res.F_final.astype(bool, copy=False)
    successes = res.successes_per_edge
    attempts = res.attempts_per_edge if log_attempts else None
    return failed_mask, successes, attempts


@dataclass
class SimOutcome:
    failed_mask: np.ndarray
    edge_successes: Optional[np.ndarray]
    edge_attempts: Optional[np.ndarray]
    damage_node_ids: tuple[int, ...]


def simulate_once(
    L,
    rng,
    log_attempts: bool = False,
    *,
    damage_node_ids=None,
    damage_coordinates=None,
) -> SimOutcome:
    """
    Run one stationary realization through the public package API.

    Runtime IDs/coordinates are resolved by the same strict resolver as the
    CLI. They are rejected when the loaded database already has a non-empty
    ``initial_damage`` set.
    """

    resolved_damage = resolve_notional_damage(
        L.nodes,
        database_node_ids=getattr(L, "damage_node_ids", ()),
        node_ids=() if damage_node_ids is None else damage_node_ids,
        coordinates=() if damage_coordinates is None else damage_coordinates,
    )
    damage_mask = notional_damage_mask(L.nodes, resolved_damage)
    pf = np.array([n.pf_node for n in L.nodes], dtype=float)
    initial_failures = apply_notional_damage(pf, damage_mask, rng)

    failed_mask, edge_succ, edge_att = propagate(
        initial_failures,
        L,
        rng=rng,
        log_attempts=log_attempts,
    )

    return SimOutcome(
        failed_mask=failed_mask.astype(bool, copy=False),
        edge_successes=edge_succ if edge_succ is not None else None,
        edge_attempts=edge_att if (log_attempts and edge_att is not None) else None,
        damage_node_ids=resolved_damage,
    )


edges_structured_table = _edges_structured_table
"""Public wrapper for building the edge table used by build_transition_matrix.
Ensures compact node indexing and compatibility with DB-loaded lattices."""
