from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


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


def _build_lattice(repo_root: Path, db_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pbm.network.builder",
            "--nx", "4",
            "--ny", "3",
            "--nz", "1",
            "--db", str(db_path),
            "--truncate", "yes",
        ],
        check=True,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )


def _run_seeded_case(
    repo_root: Path,
    input_db: Path,
    output_db: Path,
    *,
    workers: int,
    chunk_size: int,
) -> int:
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "cli" / "cli_simulation.py"),
            "--db", str(input_db),
            "--outdb", str(output_db),
            "--scenario-id", "41",
            "--samples", "173",
            "--threshold", "3",
            "--seed", "8675309",
            "--log-attempts", "no",
            "--percolation-scan", "",
            "--mc-workers", str(workers),
            "--mc-chunk-size", str(chunk_size),
            "--damage-coords", "0,0,0",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with sqlite3.connect(output_db) as con:
        return int(con.execute("SELECT MAX(id) FROM runs").fetchone()[0])


def _histogram(output_db: Path, run_id: int) -> tuple[int, ...]:
    with sqlite3.connect(output_db) as con:
        return tuple(
            int(row[0])
            for row in con.execute(
                "SELECT count FROM pmf WHERE run_id=? ORDER BY n_failed",
                (int(run_id),),
            )
        )


def test_mc_workers_parallel_basic(tmp_path: Path):
    """MC parallelism should run and write correct aggregate counts."""
    repo_root = Path(__file__).resolve().parents[1]
    input_db = tmp_path / "input.db"
    results_db = tmp_path / "results.db"

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
        "--samples", "120",
        "--threshold", "2",
        "--seed", "123",
        "--log-attempts", "no",
        "--percolation-scan", "",
        "--mc-workers", "2",
    ]
    res = subprocess.run(cmd_sim, cwd=str(repo_root), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    con = sqlite3.connect(str(results_db))
    try:
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        s = con.execute("SELECT SUM(count) FROM pmf").fetchone()[0]
        assert s == 120
    finally:
        con.close()


def test_mc_workers_policy_a_blocks_log_attempts(tmp_path: Path):
    """Policy A: log-attempts=yes must be rejected when mc-workers>1."""
    repo_root = Path(__file__).resolve().parents[1]
    input_db = tmp_path / "input.db"
    results_db = tmp_path / "results.db"

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
        "--outdb", str(results_db),
        "--scenario-id", "0",
        "--samples", "10",
        "--threshold", "2",
        "--seed", "123",
        "--log-attempts", "yes",
        "--percolation-scan", "",
        "--mc-workers", "2",
    ]
    res = subprocess.run(cmd_sim, cwd=str(repo_root), capture_output=True, text=True)
    assert res.returncode != 0
    assert "Policy A" in (res.stderr + res.stdout)


def test_seeded_histogram_is_independent_of_execution_and_database_history(
    tmp_path: Path,
):
    """One damaged experiment is exact across workers, chunks, and run IDs."""
    repo_root = Path(__file__).resolve().parents[1]
    input_db = tmp_path / "input-contract.db"
    reused_db = tmp_path / "results-reused.db"
    fresh_db = tmp_path / "results-fresh.db"

    _build_lattice(repo_root, input_db)
    _set_db_probabilities(input_db, pf_node=0.18, pe_prop=0.35)

    reused_settings = [
        (1, 1),
        (2, 7),
        (4, 13),
        (4, 13),
    ]
    reused_run_ids = [
        _run_seeded_case(
            repo_root,
            input_db,
            reused_db,
            workers=workers,
            chunk_size=chunk_size,
        )
        for workers, chunk_size in reused_settings
    ]
    fresh_run_id = _run_seeded_case(
        repo_root,
        input_db,
        fresh_db,
        workers=2,
        chunk_size=31,
    )

    assert reused_run_ids == [1, 2, 3, 4]
    assert fresh_run_id == 1

    histograms = [
        _histogram(reused_db, run_id)
        for run_id in reused_run_ids
    ]
    histograms.append(_histogram(fresh_db, fresh_run_id))
    assert all(histogram == histograms[0] for histogram in histograms[1:])
    assert sum(histograms[0]) == 173

    with sqlite3.connect(reused_db) as con:
        notes = [
            json.loads(row[0])
            for row in con.execute("SELECT notes FROM runs ORDER BY id")
        ]
    assert {note["mc_workers"] for note in notes} == {1, 2, 4}
    assert {note["mc_chunk_size"] for note in notes} == {1, 7, 13}
    assert {
        note["rng_contract"]
        for note in notes
    } == {"seedsequence-pcg64-realization-v1"}
