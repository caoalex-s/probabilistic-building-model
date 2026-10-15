# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:31:00 2025

@author: alecao
"""

# File: tests/test_metrics.py

if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "--disable-warnings"]))

import numpy as np

from pbm.network.propagation import build_transition_matrix, PropagationMatrices
from pbm.network.metrics import static_outflow_influence
from test_propagation import toy_edges


def test_static_outflow_influence_undirected_p1():
    """
    2x2x1 lattice, undirected, p=1.0:
    Each node has two neighbours, so influence[i] = 2 for all i.
    """
    edges, n = toy_edges(p=1.0, directed=False)
    mats = build_transition_matrix(n, edges)

    infl = static_outflow_influence(mats)

    assert infl.shape == (n,)
    assert np.allclose(infl, 2.0)


def test_static_outflow_influence_directed_p05():
    """
    2x2x1 lattice, directed, p=0.5:
    Each node has two outgoing edges with probability 0.5, so
      influence[i] = 0.5 + 0.5 = 1.0  for all i.
    """
    edges, n = toy_edges(p=0.5, directed=True)
    mats = build_transition_matrix(n, edges)

    infl = static_outflow_influence(mats)

    assert infl.shape == (n,)
    assert np.allclose(infl, 1.0)


def test_static_outflow_influence_dense_and_sparse_match():
    """
    Guardrail: the metric should behave identically whether P is a
    SciPy CSR matrix (normal path) or a dense NumPy array.
    """
    edges, n = toy_edges(p=0.3, directed=False)
    mats_sparse = build_transition_matrix(n, edges)

    # Dense version of the same matrix
    P_dense = mats_sparse.P.toarray()
    mats_dense = PropagationMatrices(
        P=P_dense,
        n=mats_sparse.n,
        edge_ids=mats_sparse.edge_ids,
        rowptr=mats_sparse.rowptr,
        colind=mats_sparse.colind,
        data=mats_sparse.data,
    )

    infl_sparse = static_outflow_influence(mats_sparse)
    infl_dense = static_outflow_influence(mats_dense)

    assert infl_sparse.shape == (n,)
    assert infl_dense.shape == (n,)
    assert np.allclose(infl_sparse, infl_dense)
