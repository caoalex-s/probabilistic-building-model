from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path


def _build_two_node_input(repo_root: Path, db_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pbm.network.builder",
            "--nx", "2",
            "--ny", "1",
            "--nz", "1",
            "--db", str(db_path),
            "--truncate", "yes",
        ],
        check=True,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )


def _run_cli(repo_root: Path, input_db: Path, output_db: Path):
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "cli" / "cli_simulation.py"),
            "--db", str(input_db),
            "--outdb", str(output_db),
            "--samples", "10",
            "--threshold", "2",
            "--seed", "123",
            "--percolation-scan", "",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )


def _configure_one_way(db_path: Path, *, failed_node_id: int) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE nodes SET pf_node=0.0;")
        con.execute(
            "UPDATE nodes SET pf_node=1.0 WHERE id=?;",
            (int(failed_node_id),),
        )
        con.execute(
            """
            UPDATE edges
            SET pe=NULL, pe_fwd=1.0, pe_rev=0.0, directed=1, blocked=0;
            """
        )
        con.commit()


def test_cli_respects_explicit_one_way_direction(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]

    forward_db = tmp_path / "forward.db"
    forward_out = tmp_path / "forward-results.db"
    _build_two_node_input(repo_root, forward_db)
    _configure_one_way(forward_db, failed_node_id=1)
    forward = _run_cli(repo_root, forward_db, forward_out)
    assert forward.returncode == 0, forward.stderr

    reverse_db = tmp_path / "reverse.db"
    reverse_out = tmp_path / "reverse-results.db"
    _build_two_node_input(repo_root, reverse_db)
    _configure_one_way(reverse_db, failed_node_id=2)
    reverse = _run_cli(repo_root, reverse_db, reverse_out)
    assert reverse.returncode == 0, reverse.stderr

    with sqlite3.connect(forward_out) as con:
        forward_p_full = con.execute(
            "SELECT p_full FROM collapse_probs;"
        ).fetchone()[0]
    with sqlite3.connect(reverse_out) as con:
        reverse_p_full = con.execute(
            "SELECT p_full FROM collapse_probs;"
        ).fetchone()[0]

    assert forward_p_full == 1.0
    assert reverse_p_full == 0.0


def test_cli_rejects_duplicate_database_rows_before_sparse_construction(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    input_db = tmp_path / "duplicates.db"
    output_db = tmp_path / "duplicates-results.db"
    with sqlite3.connect(input_db) as con:
        con.executescript(
            """
            CREATE TABLE nodes (
              id INTEGER PRIMARY KEY,
              x INTEGER NOT NULL,
              y INTEGER NOT NULL,
              z INTEGER NOT NULL,
              pf_node REAL NOT NULL
            );
            CREATE TABLE initial_damage (
              node_id INTEGER PRIMARY KEY,
              FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            CREATE TABLE edges (
              id INTEGER PRIMARY KEY,
              org INTEGER NOT NULL,
              dst INTEGER NOT NULL,
              dir TEXT NOT NULL,
              pe REAL,
              pe_fwd REAL,
              pe_rev REAL,
              directed INTEGER NOT NULL,
              blocked INTEGER NOT NULL
            );
            INSERT INTO nodes(id,x,y,z,pf_node)
            VALUES
              (1,0,0,0,1.0),
              (2,1,0,0,0.0);
            INSERT INTO edges(id,org,dst,dir,pe,pe_fwd,pe_rev,directed,blocked)
            VALUES
              (1,1,2,'x',0.2,NULL,NULL,0,0),
              (2,1,2,'x',NULL,0.3,0.4,1,0);
            """
        )

    result = _run_cli(repo_root, input_db, output_db)
    assert result.returncode != 0
    assert "duplicate adjacency" in (result.stdout + result.stderr)
