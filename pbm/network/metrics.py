# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:26:49 2025

@author: alecao
"""

# File: model/metrics.py

from __future__ import annotations

from typing import Final
import numpy as np

from .propagation import PropagationMatrices

__all__: Final = ["static_outflow_influence"]


def static_outflow_influence(mats: PropagationMatrices) -> np.ndarray:
    """
    Compute a static node influence measure from the transition matrix.

    For each node i, influence is defined as the expected number of
    *directly affected* neighbours in a single propagation step, i.e.
        influence[i] = sum_j P[i, j]

    This is purely a function of the transition matrix P and does not
    depend on Monte Carlo samples.

    Parameters
    ----------
    mats : PropagationMatrices
        Container with the transition matrix `P` (dense or sparse).

    Returns
    -------
    influence : np.ndarray, shape (N,)
        Static influence score for each node index 0..N-1.
    """
    P = mats.P

    # We expect a 2D matrix; guard lightly to catch gross misuse early.
    if P.ndim != 2:
        raise ValueError(
            f"Transition matrix P must be 2D, got ndim={P.ndim!r}"
        )

    # Works for both NumPy arrays and SciPy sparse matrices:
    # - For dense arrays: P.sum(axis=1) -> 1D array or 2D row vector.
    # - For sparse matrices: P.sum(axis=1) -> (N, 1) matrix.
    row_sums = np.asarray(P.sum(axis=1)).ravel()

    # Defensive check: we want a 1D vector matching the number of rows.
    n_rows = P.shape[0]
    if row_sums.shape != (n_rows,):
        raise RuntimeError(
            f"Unexpected influence shape {row_sums.shape!r}, "
            f"expected ({n_rows},)"
        )

    return row_sums
