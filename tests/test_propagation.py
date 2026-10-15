if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "--disable-warnings"]))

import numpy as np
import pytest
from numpy.random import default_rng

from pbm.network.propagation import (
    build_transition_matrix,
    simulate_once_mc,
    step_prob,
)


EDGE_DTYPE = np.dtype([
    ("org", "i4"),
    ("dst", "i4"),
    ("directed", "i4"),
    ("pe", "f8"),
    ("pe_fwd", "f8"),
    ("pe_rev", "f8"),
    ("blocked", "i4"),
])


def edge_row(
    org,
    dst,
    *,
    directed=0,
    pe=None,
    pe_fwd=None,
    pe_rev=None,
    blocked=0,
):
    def nullable(value):
        return np.nan if value is None else value

    return (
        org,
        dst,
        directed,
        nullable(pe),
        nullable(pe_fwd),
        nullable(pe_rev),
        blocked,
    )


def edge_array(*rows):
    return np.array(list(rows), dtype=EDGE_DTYPE)


def toy_edges(nx=2, ny=2, nz=1, p=1.0, directed=False):
    """
    Build a 2x2x1 lattice → 4 nodes, edges along +x,+y (canonical).
    Node ids: (x,y,z) → id = x + nx*y + nx*ny*z
    """
    def id_of(x, y, z):
        return x + nx * y + nx * ny * z

    rows = []
    for y in range(ny):
        for x in range(nx):
            i = id_of(x, y, 0)
            for j in (
                id_of(x + 1, y, 0) if x + 1 < nx else None,
                id_of(x, y + 1, 0) if y + 1 < ny else None,
            ):
                if j is None:
                    continue
                if directed:
                    rows.append(
                        edge_row(
                            i,
                            j,
                            directed=1,
                            pe_fwd=p,
                            pe_rev=p,
                        )
                    )
                else:
                    rows.append(edge_row(i, j, directed=0, pe=p))
    return np.array(rows, dtype=EDGE_DTYPE), nx * ny * nz


def test_prob_spread_p1():
    edges, n = toy_edges(p=1.0, directed=False)
    mats = build_transition_matrix(n, edges)
    F0 = np.zeros(n)
    F0[0] = 1.0
    F1 = step_prob(F0, mats)
    assert F1[1] == 1.0 or F1[2] == 1.0
    assert np.all((F1 >= 0.0) & (F1 <= 1.0))


def test_no_spread_p0():
    edges, n = toy_edges(p=0.0, directed=False)
    mats = build_transition_matrix(n, edges)
    F0 = np.zeros(n)
    F0[0] = 1.0
    F1 = step_prob(F0, mats)
    assert np.allclose(F1, 0.0)


def test_mc_attempts_and_successes():
    edges, n = toy_edges(p=0.5, directed=False)
    mats = build_transition_matrix(n, edges)
    F0 = np.zeros(n, dtype=bool)
    F0[0] = True
    res = simulate_once_mc(F0, mats, rng=default_rng(123), log_attempts=True)
    assert res.attempts_per_edge is not None
    assert res.successes_per_edge is not None
    assert res.attempts_per_edge.sum() > 0
    assert np.all(res.successes_per_edge <= res.attempts_per_edge)
    assert np.all(res.attempts_per_edge <= 1)


def test_undirected_edge_creates_equal_arcs():
    mats = build_transition_matrix(
        2,
        edge_array(edge_row(0, 1, directed=0, pe=0.35)),
    )
    assert np.allclose(mats.P.toarray(), [[0.0, 0.35], [0.35, 0.0]])


def test_asymmetric_directed_edge_uses_both_explicit_probabilities():
    mats = build_transition_matrix(
        2,
        edge_array(
            edge_row(0, 1, directed=1, pe_fwd=0.2, pe_rev=0.8)
        ),
    )
    assert np.allclose(mats.P.toarray(), [[0.0, 0.2], [0.8, 0.0]])


def test_directed_equal_probabilities_remain_valid_directed_input():
    mats = build_transition_matrix(
        2,
        edge_array(
            edge_row(0, 1, directed=1, pe_fwd=0.4, pe_rev=0.4)
        ),
    )
    assert np.allclose(mats.P.toarray(), [[0.0, 0.4], [0.4, 0.0]])


def test_one_way_edge_requires_explicit_zero():
    mats = build_transition_matrix(
        2,
        edge_array(
            edge_row(0, 1, directed=1, pe_fwd=0.6, pe_rev=0.0)
        ),
    )
    assert np.allclose(mats.P.toarray(), [[0.0, 0.6], [0.0, 0.0]])


def test_blocked_edge_has_zero_transmission_in_both_directions():
    mats = build_transition_matrix(
        2,
        edge_array(
            edge_row(
                0,
                1,
                directed=1,
                pe_fwd=1.0,
                pe_rev=1.0,
                blocked=1,
            )
        ),
    )
    assert mats.P.nnz == 0
    assert np.allclose(mats.P.toarray(), 0.0)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (edge_row(0, 1, directed=0, pe=None), "directed=0 requires non-NULL pe"),
        (
            edge_row(0, 1, directed=0, pe=0.2, pe_fwd=0.2),
            "requires pe_fwd and pe_rev to be NULL",
        ),
        (
            edge_row(0, 1, directed=1, pe=0.2, pe_fwd=0.2, pe_rev=0.3),
            "directed=1 requires pe to be NULL",
        ),
        (
            edge_row(0, 1, directed=1, pe_fwd=0.2, pe_rev=None),
            "missing pe_rev",
        ),
        (
            edge_row(0, 1, directed=0, pe=1.01),
            "finite probability in \\[0,1\\]",
        ),
        (
            edge_row(0, 1, directed=0, pe=np.inf),
            "finite probability in \\[0,1\\]",
        ),
        (edge_row(0, 1, directed=2, pe=0.2), "directed must be 0 or 1"),
        (edge_row(0, 1, directed=0, pe=0.2, blocked=2), "blocked must be 0 or 1"),
    ],
)
def test_malformed_edge_rows_are_rejected(row, message):
    with pytest.raises(ValueError, match=message):
        build_transition_matrix(2, edge_array(row))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (edge_row(0, 0, directed=0, pe=0.2), "self-loop"),
        (edge_row(1, 0, directed=0, pe=0.2), "canonical ordering"),
        (edge_row(0, 2, directed=0, pe=0.2), "endpoints"),
    ],
)
def test_invalid_edge_endpoints_are_rejected(row, message):
    with pytest.raises(ValueError, match=message):
        build_transition_matrix(2, edge_array(row))


def test_duplicate_adjacencies_are_rejected_before_sparse_construction():
    duplicate_rows = edge_array(
        edge_row(0, 1, directed=1, pe_fwd=0.6, pe_rev=0.4),
        edge_row(0, 1, directed=1, pe_fwd=0.5, pe_rev=0.5),
    )
    with pytest.raises(ValueError, match="duplicate adjacency"):
        build_transition_matrix(2, duplicate_rows)


def test_already_failed_destinations_are_not_attempted():
    mats = build_transition_matrix(
        2,
        edge_array(edge_row(0, 1, directed=0, pe=1.0)),
    )
    res = simulate_once_mc(
        np.array([True, True]),
        mats,
        rng=default_rng(1),
        log_attempts=True,
    )
    assert res.attempts_per_edge.sum() == 0


def test_max_steps_exhaustion_is_an_error():
    edges = edge_array(
        edge_row(0, 1, directed=0, pe=1.0),
        edge_row(1, 2, directed=0, pe=1.0),
    )
    mats = build_transition_matrix(3, edges)
    with pytest.raises(RuntimeError, match="did not converge"):
        simulate_once_mc(
            np.array([True, False, False]),
            mats,
            rng=default_rng(1),
            max_steps=1,
        )
