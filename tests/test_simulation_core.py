from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path


def _set_db_probabilities(db_path: Path, pf_node: float, pe_prop: float) -> None:
    """Set pf_node on all nodes and undirected pe on all edges."""
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


def test_block8_core(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]

    input_db = tmp_path / "input.db"
    results_db = tmp_path / "results.db"

    # Build a small lattice input DB
    cmd_build = [
        sys.executable,
        "-m",
        "pbm.network.builder",
        "--nx", "4",
        "--ny", "4",
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
        "--outdb", str(results_db),
        "--scenario-id", "0",
        "--samples", "300",
        "--threshold", "2",
        "--seed", "123",
        "--log-attempts", "no",
        "--percolation-scan", "",  # override YAML defaults
    ]
    res = subprocess.run(cmd_sim, cwd=str(repo_root), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    con = sqlite3.connect(str(results_db))
    try:
        # one run
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1

        # pmf sums to samples
        s = con.execute("SELECT SUM(count) FROM pmf").fetchone()[0]
        assert s == 300

        # collapse_probs exists
        assert con.execute("SELECT COUNT(*) FROM collapse_probs").fetchone()[0] == 1

        # edge_usage non-empty (simulation logs per-edge successes even if log_attempts=no)
        assert con.execute("SELECT COUNT(*) FROM edge_usage").fetchone()[0] > 0
    finally:
        con.close()
