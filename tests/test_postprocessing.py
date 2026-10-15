# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:33:59 2025

@author: caol
"""

# tests/test_postprocessing.py
from __future__ import annotations

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so that `post_processing` and
# other top-level packages can be imported when this file is executed
# directly (e.g. via Spyder's %runfile).
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../probabilistic-building-model
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Now it's safe to import project modules
# ---------------------------------------------------------------------------
import sqlite3
import inspect

import numpy as np
import pytest

from pbm.post.core import run_postprocessing
from pbm.post import percolation_plots
from pbm.post.percolation_plots import (
    detect_sweep_mode,
    extract_1d_curves,
)


# ---------------------------------------------------------------------------
# Helpers: minimal results DB
# ---------------------------------------------------------------------------

def make_minimal_results_db(db_path: Path, reset: bool = False) -> sqlite3.Connection:
    """
    Create (or reuse) a minimal results.db suitable for post-processing tests.

    Parameters
    ----------
    db_path : Path
        Location of the SQLite DB file.
    reset : bool, default False
        If True:
            • Existing file is deleted
            • A fresh DB with tables + synthetic data is created.
        If False:
            • If DB exists → return an open connection without modifying tables.
            • If DB does not exist → create a fresh DB with tables + synthetic data.

    Returns
    -------
    conn : sqlite3.Connection
        Connection to the (possibly newly created) database.
    """
    db_path = Path(db_path)

    # ----------------------------
    # Case 1: Reset requested
    # ----------------------------
    if reset and db_path.exists():
        db_path.unlink()

    # ----------------------------
    # Case 2: DB already exists AND reset=False
    # → Simply open it, do NOT try to recreate tables.
    # ----------------------------
    if db_path.exists() and not reset:
        return sqlite3.connect(db_path)

    # ----------------------------
    # Case 3: DB does not exist (or was removed)
    # → Create it and populate minimal schema
    # ----------------------------
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE runs (
            id          INTEGER PRIMARY KEY,
            scenario_id INTEGER,
            samples     INTEGER,
            notes       TEXT
        );

        CREATE TABLE pmf (
            run_id   INTEGER NOT NULL,
            n_failed INTEGER NOT NULL,
            p        REAL NOT NULL
        );

        CREATE TABLE percolation_multi (
            run_id INTEGER NOT NULL,
            pe_x   REAL NOT NULL,
            pe_y   REAL NOT NULL,
            pe_z   REAL NOT NULL,
            p_any  REAL NOT NULL,
            p_one  REAL NOT NULL,
            p_full REAL NOT NULL
        );

        CREATE TABLE node_influence (
            run_id    INTEGER NOT NULL,
            node_id   INTEGER NOT NULL,
            influence REAL NOT NULL
        );
        """
    )

    # Insert a single synthetic run
    cur.execute(
        "INSERT INTO runs (id, scenario_id, samples, notes) VALUES (?, ?, ?, ?);",
        (1, 42, 1_000_000, "{}"),
    )

    # PMF: n_failed = 0..3
    pmf_rows = [(1, 0, 0.4), (1, 1, 0.3), (1, 2, 0.2), (1, 3, 0.1)]
    cur.executemany(
        "INSERT INTO pmf (run_id, n_failed, p) VALUES (?, ?, ?);",
        pmf_rows,
    )

    # Percolation: 1D fully coupled xyz sweep at t ∈ {0, 0.5, 1.0}
    ts = [0.0, 0.5, 1.0]
    perco_rows = []
    for t in ts:
        p_any = t
        p_one = max(0.0, 1.0 - t)
        p_full = 0.5 * t
        perco_rows.append((1, t, t, t, p_any, p_one, p_full))
    cur.executemany(
        """
        INSERT INTO percolation_multi
        (run_id, pe_x, pe_y, pe_z, p_any, p_one, p_full)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        perco_rows,
    )

    # Node influence (3 nodes)
    influ_rows = [
        (1, 10, 0.1),
        (1, 11, 0.5),
        (1, 12, 0.3),
    ]
    cur.executemany(
        "INSERT INTO node_influence (run_id, node_id, influence) VALUES (?, ?, ?);",
        influ_rows,
    )

    conn.commit()
    return conn



# ---------------------------------------------------------------------------
# Tests for detect_sweep_mode
# ---------------------------------------------------------------------------

def test_detect_sweep_mode_1d_xyz():
    # Three points: fully coupled xyz sweep
    data = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5, 0.4, 0.6, 0.1],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    mode, tvals = detect_sweep_mode(data)
    assert mode == "1d_xyz"
    # tvals is the unique set of pe_x values
    assert np.allclose(tvals, [0.0, 0.5, 1.0])


def test_detect_sweep_mode_2d_xy():
    # Simple 2D grid in x,y, z fixed
    xs = [0.0, 0.5, 1.0]
    ys = [0.0, 0.5]
    rows = []
    for x in xs:
        for y in ys:
            rows.append([x, y, 0.2, x * y, 0.0, 0.0])
    data = np.array(rows, dtype=float)

    mode, sweep_info = detect_sweep_mode(data)
    assert mode == "2d_xy"
    ax_vals, ay_vals = sweep_info
    assert np.allclose(ax_vals, xs)
    assert np.allclose(ay_vals, ys)


# ---------------------------------------------------------------------------
# Tests for strict one-dimensional scan extraction
# ---------------------------------------------------------------------------

def test_extract_1d_curves_rejects_duplicate_scan_values():
    data = np.array(
        [
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
        ],
        dtype=float,
    )
    with pytest.raises(ValueError, match="duplicate"):
        extract_1d_curves(data, "1d_xyz")


def test_extract_1d_curves_rejects_out_of_order_scan_values():
    data = np.array(
        [
            [0.5, 0.5, 0.5, 0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
        ],
        dtype=float,
    )
    with pytest.raises(ValueError, match="strictly ordered"):
        extract_1d_curves(data, "1d_xyz")


# ---------------------------------------------------------------------------
# Tests for run_postprocessing
# ---------------------------------------------------------------------------

def test_percolation_postprocessing_has_no_robustness_path():
    assert not hasattr(percolation_plots, "compute_robustness_index")
    assert "compute_robustness_index" not in inspect.signature(
        run_postprocessing
    ).parameters


def test_run_postprocessing_minimal_db(tmp_path: Path):
    db_path = tmp_path / "results.db"
    conn = make_minimal_results_db(db_path)

    # We deliberately disable plotting here to avoid dealing with Matplotlib
    # backends in this test. The logic is still fully exercised.
    summary = run_postprocessing(
        conn,
        run_id=1,
        make_plots=False,
        compute_risk_equivalence=False,
        figs_root=tmp_path / "figs",
    )

    # Basic structure
    assert summary["run_id"] == 1

    # PMF support should match the inserted n_failed values
    pmf_support = summary.get("pmf_support")
    assert pmf_support is not None
    assert np.array_equal(pmf_support, np.array([0, 1, 2, 3]))

    # Percolation mode detected as 1d_xyz for the synthetic data
    assert summary.get("percolation_mode") == "1d_xyz"
    assert summary["percolation_threshold"] is None
    assert summary["percolation_threshold_status"] == "no identifiable threshold"
    assert (
        summary["percolation_threshold_reason"] == "insufficient_scan_points"
    )

    # Percolation p_any is not repurposed as damage robustness.
    assert "robustness_index" not in summary

    # Influence summary: max_influence and top_nodes from the synthetic data
    assert pytest.approx(summary.get("max_influence"), rel=1e-12) == 0.5
    top_nodes = summary.get("top_nodes")
    assert top_nodes is not None
    # node_id 11 has the largest influence in the helper DB
    assert 11 in top_nodes


def test_run_postprocessing_creates_figs(tmp_path: Path):
    db_path = tmp_path / "results.db"
    conn = make_minimal_results_db(db_path)

    figs_root = tmp_path / "figs"
    summary = run_postprocessing(
        conn,
        run_id=1,
        make_plots=True,
        compute_risk_equivalence=False,
        figs_root=figs_root,
    )

    # Check that figure files have been created in the expected directory
    run_dir = figs_root / "run_1"
    assert run_dir.is_dir()

    # We expect at least these plots for the synthetic data
    expected_prefixes = [
        "pmf_1",
        "percolation_1d_xyz_run_1",
        "influence_run_1",
    ]
    created = {p.stem for p in run_dir.glob("*.pdf")}
    for prefix in expected_prefixes:
        assert prefix in created

    # Summary should still be sensible
    assert summary["run_id"] == 1
