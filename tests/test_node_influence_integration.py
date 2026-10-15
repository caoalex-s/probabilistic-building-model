from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np


def _set_db_probabilities(db_path: Path, pf_node: float, pe_prop: float) -> None:
    with sqlite3.connect(str(db_path)) as con:
        con.execute("PRAGMA foreign_keys=ON;")
        con.execute("UPDATE nodes SET pf_node = ?;", (float(pf_node),))
        con.execute(
            """
            UPDATE edges
            SET pe = ?,
                directed = 0,
                pe_fwd = NULL,
                pe_rev = NULL,
                blocked = 0
            ;
            """,
            (float(pe_prop),),
        )
        con.commit()


def run_simulation(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]

    input_db = tmp_path / "input.db"
    out_db = tmp_path / "results.db"

    # Build a small lattice
    cmd_build = [
        sys.executable,
        "-m",
        "pbm.network.builder",
        "--nx", "3",
        "--ny", "3",
        "--nz", "1",
        "--db", str(input_db),
        "--truncate", "yes",
    ]
    subprocess.run(cmd_build, check=True, cwd=str(repo_root), capture_output=True, text=True)
    _set_db_probabilities(input_db, pf_node=1e-3, pe_prop=1e-2)

    cmd_sim = [
        sys.executable,
        str(repo_root / "cli" / "cli_simulation.py"),
        "--db", str(input_db),
        "--outdb", str(out_db),
        "--scenario-id", "13",
        "--samples", "50",
        "--threshold", "1",
        "--seed", "123",
        "--log-attempts", "no",
        "--percolation-scan", "",  # keep baseline path stable
    ]
    subprocess.run(cmd_sim, check=True, cwd=str(repo_root), capture_output=True, text=True)
    return out_db


def test_node_influence_table_populated(tmp_path: Path):
    out_db = run_simulation(tmp_path)
    assert out_db.is_file(), "results.db was not created"

    con = sqlite3.connect(str(out_db))
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='node_influence';"
        )
        assert cur.fetchone() is not None, "node_influence table is missing"

        cur.execute("SELECT max(id) FROM runs;")
        (run_id,) = cur.fetchone()
        assert run_id is not None, "no runs found in results.db"

        cur.execute(
            "SELECT COUNT(*) FROM node_influence WHERE run_id = ?;",
            (run_id,),
        )
        n_influence = cur.fetchone()[0]
        assert n_influence > 0, "node_influence has no rows for this run"

        cur.execute(
            "SELECT COUNT(DISTINCT node_id) FROM node_influence WHERE run_id = ?;",
            (run_id,),
        )
        n_distinct = cur.fetchone()[0]
        assert n_influence == n_distinct, "duplicate (run_id, node_id) entries"

        cur.execute(
            "SELECT influence FROM node_influence WHERE run_id = ?;",
            (run_id,),
        )
        vals = [r[0] for r in cur.fetchall()]
    finally:
        con.close()

    arr = np.array(vals, dtype=float)
    assert np.all(np.isfinite(arr)), "influence contains non-finite values"
    assert np.all(arr >= 0.0), "influence contains negative values"
