# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 19:55:42 2025

@author: caol
"""

import subprocess
import sqlite3
import sys
import time
from pathlib import Path

import pytest


TOL = 2e-3
SAMPLES = 500
THRESHOLD = 2
SEED = 123


def _project_root() -> Path:
    # tests/ -> project root
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> None:
    """Run subprocess and show stdout/stderr if it fails (critical for debugging on CI/Windows)."""
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  cwd: {cwd}\n"
            f"  returncode: {p.returncode}\n"
            f"--- stdout ---\n{p.stdout}\n"
            f"--- stderr ---\n{p.stderr}\n"
        )


def _run_builder(db_path: Path):
    cwd = _project_root()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Builder applies the input schema itself.
    cmd = [
        sys.executable,
        "-m",
        "pbm.network.builder",
        "--nx", "4",
        "--ny", "4",
        "--nz", "1",
        "--db", str(db_path),
        "--truncate", "yes",
    ]
    _run(cmd, cwd=cwd)



def _run_simulation(db_path: Path, outdb: Path):
    outdb.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "cli/cli_simulation.py",
        "--db", str(db_path),
        "--outdb", str(outdb),
        "--scenario-id", "13",
        "--samples", str(SAMPLES),
        "--threshold", str(THRESHOLD),
        "--seed", str(SEED),
        "--log-attempts", "no",
        "--percolation-scan", "",
    ]
    _run(cmd, cwd=_project_root())


def _pmf(conn, run_id):
    # Discover PMF bin column name (not always 'k')
    cols = [r[1] for r in conn.execute("PRAGMA table_info(pmf)").fetchall()]

    # Common candidates across iterations
    candidates = ["k", "n_failed", "bin", "value", "n", "fails", "collapsed"]
    bin_col = next((c for c in candidates if c in cols), None)
    assert bin_col is not None, f"Could not find PMF bin column in pmf table. Columns: {cols}"

    # Count column (usually 'count', sometimes 'n' or 'freq')
    count_candidates = ["count", "freq", "n_count", "counts"]
    count_col = next((c for c in count_candidates if c in cols), None)
    assert count_col is not None, f"Could not find PMF count column in pmf table. Columns: {cols}"

    rows = conn.execute(
        f"SELECT {bin_col}, {count_col} FROM pmf WHERE run_id=? ORDER BY {bin_col}",
        (run_id,),
    ).fetchall()

    total = sum(c for _, c in rows)
    assert total > 0
    return {int(b): c / total for b, c in rows}



def _collapse_probs(conn, run_id):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(collapse_probs)").fetchall()]
    assert "run_id" in cols, f"collapse_probs missing run_id column. Columns: {cols}"

    # Case A: wide schema (your current one): one row per run, columns are metrics
    wide_metrics = [c for c in cols if c != "run_id"]
    if wide_metrics:
        row = conn.execute(
            f"SELECT {', '.join(wide_metrics)} FROM collapse_probs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        assert row is not None, f"No collapse_probs row for run_id={run_id}"
        return {m: float(v) for m, v in zip(wide_metrics, row)}

    # Case B: long schema (older): (run_id, label, value)
    label_candidates = ["name", "metric", "event", "label", "key", "type"]
    value_candidates = ["value", "prob", "p", "pf", "collapse_prob"]

    label_col = next((c for c in label_candidates if c in cols), None)
    value_col = next((c for c in value_candidates if c in cols), None)
    assert label_col and value_col, f"collapse_probs schema not recognized. Columns: {cols}"

    rows = conn.execute(
        f"SELECT {label_col}, {value_col} FROM collapse_probs WHERE run_id=?",
        (run_id,),
    ).fetchall()
    return {str(k): float(v) for k, v in rows}



def test_smoke_end_to_end(tmp_path):
    """
    Block 13:
    - rebuild lattice
    - run simulation twice with fixed seed
    - PMF + collapse_probs reproducible within tolerance
    - runtime regression guard
    """
    input_db = tmp_path / "input.db"
    results_db = tmp_path / "results.db"

    _run_builder(input_db)

    # baseline
    t0 = time.perf_counter()
    _run_simulation(input_db, results_db)
    t1 = time.perf_counter()

    # candidate
    t2 = time.perf_counter()
    _run_simulation(input_db, results_db)
    t3 = time.perf_counter()

    base_dt = t1 - t0
    new_dt = t3 - t2

    conn = sqlite3.connect(results_db)
    run_ids = [r[0] for r in conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 2")]
    assert len(run_ids) == 2, "Smoke test requires two runs in results.db"
    run_new, run_base = run_ids[0], run_ids[1]

    pmf_base = _pmf(conn, run_base)
    pmf_new = _pmf(conn, run_new)

    coll_base = _collapse_probs(conn, run_base)
    coll_new = _collapse_probs(conn, run_new)
    conn.close()

    # PMF L∞
    all_k = set(pmf_base) | set(pmf_new)
    max_diff_pmf = max(abs(pmf_new.get(k, 0.0) - pmf_base.get(k, 0.0)) for k in all_k)

    # collapse_probs L∞
    all_names = set(coll_base) | set(coll_new)
    max_diff_coll = max(abs(coll_new.get(n, 0.0) - coll_base.get(n, 0.0)) for n in all_names)

    assert max_diff_pmf <= TOL, f"PMF regression detected: {max_diff_pmf:.3g} > {TOL}"
    assert max_diff_coll <= TOL, f"Collapse regression detected: {max_diff_coll:.3g} > {TOL}"

    # runtime guard
    if base_dt > 0:
        assert new_dt <= 1.5 * base_dt, (
            f"Runtime regression: {new_dt:.3f}s > 1.5×{base_dt:.3f}s "
            f"(baseline {base_dt:.3f}s)"
        )
