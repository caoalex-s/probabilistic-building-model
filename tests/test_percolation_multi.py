from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def _set_db_probabilities(db_path: Path, pf_node: float, pe_prop: float) -> None:
    """Set pf_node and undirected pe in the input DB."""
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


def _run_sim(
    repo_root: Path,
    tmp_path: Path,
    multi_spec: str,
    extra_args=None,
    *,
    samples: int = 1000,
    nz: int = 1,
) -> Path:
    input_db = tmp_path / "input.db"
    out_db = tmp_path / "results.db"

    # build lattice
    cmd_build = [
        sys.executable,
        "-m",
        "pbm.network.builder",
        "--nx", "3",
        "--ny", "3",
        "--nz", str(nz),
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
        "--scenario-id", "1",
        "--samples", str(samples),
        "--threshold", "1",
        "--seed", "123",
        "--log-attempts", "no",
        "--percolation-scan", multi_spec,
    ]
    if extra_args:
        cmd_sim.extend(list(extra_args))
    subprocess.run(cmd_sim, check=True, cwd=str(repo_root), capture_output=True, text=True)
    return out_db


def test_multi_coupled_xyz(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out_db = _run_sim(repo_root, tmp_path, "xyz:0,1,0.5")

    with sqlite3.connect(str(out_db)) as con:
        rows = con.execute(
            """
            SELECT
              pe_x, pe_y, pe_z, p_any,
              n_samples, count_any, count_one, count_full
            FROM percolation_multi
            WHERE run_id=(SELECT MAX(id) FROM runs)
            ORDER BY pe_x
            """
        ).fetchall()
        notes = json.loads(
            con.execute(
                "SELECT notes FROM runs WHERE id=(SELECT MAX(id) FROM runs)"
            ).fetchone()[0]
        )
        pmf_groups = con.execute(
            """
            SELECT pe_x, COUNT(*), SUM(count), MIN(n_samples), MAX(n_samples)
            FROM percolation_pmf
            WHERE run_id=(SELECT MAX(id) FROM runs)
            GROUP BY pe_x, pe_y, pe_z
            ORDER BY pe_x
            """
        ).fetchall()

    assert len(rows) >= 3
    assert all(row[4] == 1000 for row in rows)
    assert all(row[3] == row[5] / row[4] for row in rows)
    assert all(0 <= row[6] <= row[4] and 0 <= row[7] <= row[4] for row in rows)
    metadata = notes["percolation_scan_metadata"]
    assert metadata["varied_axes"] == ["x", "y", "z"]
    assert metadata["fixed_axes"] == []
    assert metadata["varied_axis_groups"] == [["x", "y", "z"]]
    assert notes["regular_lattice_shape"] == [3, 3, 1]
    provenance = notes["pf_node_provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["source"] == "input_database.nodes.pf_node"
    assert provenance["authoritative_field"] == "pf_node"
    assert provenance["node_count"] == 9
    assert provenance["uniform"] is True
    assert provenance["uniform_value"] == 1e-3
    assert provenance["minimum"] == 1e-3
    assert provenance["maximum"] == 1e-3
    assert provenance["unique_value_count"] == 1
    assert len(provenance["values_sha256"]) == 64
    assert len(pmf_groups) == len(rows)
    assert all(group[1] == 10 for group in pmf_groups)  # N_f=0,...,9
    assert all(group[2:] == (1000, 1000, 1000) for group in pmf_groups)

    # monotone non-decreasing p_any (allow some MC noise)
    pans = [r[3] for r in rows]
    for i in range(1, len(pans)):
        assert pans[i] + 0.05 >= pans[i - 1]


def test_multi_xy_grid(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out_db = _run_sim(repo_root, tmp_path, "x:0,1,1; y:0,1,1")

    with sqlite3.connect(str(out_db)) as con:
        n = con.execute(
            """
            SELECT COUNT(*) FROM percolation_multi
            WHERE run_id=(SELECT MAX(id) FROM runs)
            """
        ).fetchone()[0]
        notes = json.loads(
            con.execute(
                "SELECT notes FROM runs WHERE id=(SELECT MAX(id) FROM runs)"
            ).fetchone()[0]
        )

    assert n == 4  # 2×2 grid
    metadata = notes["percolation_scan_metadata"]
    assert metadata["mode"] == "2d_xy"
    assert metadata["varied_axes"] == ["x", "y"]
    assert metadata["fixed_axes"] == ["z"]
    assert metadata["fixed_axis_values"]["z"] is None  # no z edges when nz=1


def test_fixed_axis_probability_is_preserved_in_metadata(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out_db = _run_sim(
        repo_root,
        tmp_path,
        "xy:0,1,1",
        samples=20,
        nz=2,
    )

    with sqlite3.connect(str(out_db)) as con:
        notes = json.loads(
            con.execute(
                "SELECT notes FROM runs WHERE id=(SELECT MAX(id) FROM runs)"
            ).fetchone()[0]
        )

    metadata = notes["percolation_scan_metadata"]
    assert metadata["varied_axes"] == ["x", "y"]
    assert metadata["fixed_axes"] == ["z"]
    assert metadata["fixed_axis_values"] == {"z": 0.01}


def test_multi_xy_grid_parallel(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out_db = _run_sim(repo_root, tmp_path, "x:0,1,1; y:0,1,1", extra_args=["--percolation-workers", "2"])

    with sqlite3.connect(str(out_db)) as con:
        n = con.execute(
            """
            SELECT COUNT(*) FROM percolation_multi
            WHERE run_id=(SELECT MAX(id) FROM runs)
            """
        ).fetchone()[0]

    assert n == 4


def test_scalar_scan_rejects_directed_edges_on_scanned_axis(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    input_db = tmp_path / "directed.db"
    out_db = tmp_path / "directed-results.db"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pbm.network.builder",
            "--nx", "2",
            "--ny", "1",
            "--nz", "1",
            "--db", str(input_db),
            "--truncate", "yes",
        ],
        check=True,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    with sqlite3.connect(input_db) as con:
        con.execute(
            """
            UPDATE edges
            SET pe=NULL, pe_fwd=0.4, pe_rev=0.6, directed=1, blocked=0;
            """
        )
        con.commit()

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "cli" / "cli_simulation.py"),
            "--db", str(input_db),
            "--outdb", str(out_db),
            "--samples", "10",
            "--threshold", "1",
            "--seed", "123",
            "--percolation-scan", "x:0,1,1",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "cannot modify directed edges" in (result.stdout + result.stderr)


def test_percolation_scan_preserves_blocked_edges(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    input_db = tmp_path / "blocked.db"
    out_db = tmp_path / "blocked-results.db"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pbm.network.builder",
            "--nx", "2",
            "--ny", "1",
            "--nz", "1",
            "--db", str(input_db),
            "--truncate", "yes",
        ],
        check=True,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    with sqlite3.connect(input_db) as con:
        con.execute("UPDATE nodes SET pf_node=0.0;")
        con.execute("UPDATE nodes SET pf_node=1.0 WHERE id=(SELECT MIN(id) FROM nodes);")
        con.execute(
            """
            UPDATE edges
            SET pe=0.9, pe_fwd=NULL, pe_rev=NULL, directed=0, blocked=1;
            """
        )
        con.commit()

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "cli" / "cli_simulation.py"),
            "--db", str(input_db),
            "--outdb", str(out_db),
            "--samples", "20",
            "--threshold", "2",
            "--seed", "123",
            "--percolation-scan", "x:1,1,1",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with sqlite3.connect(out_db) as con:
        p_any, p_one, p_full = con.execute(
            """
            SELECT p_any, p_one, p_full
            FROM percolation_multi
            WHERE run_id=(SELECT MAX(id) FROM runs);
            """
        ).fetchone()
    assert p_any == 1.0
    assert p_one == 1.0
    assert p_full == 0.0


def _percolation_rows(out_db: Path, run_id: int):
    with sqlite3.connect(out_db) as con:
        return con.execute(
            """
            SELECT
              pe_x, pe_y, pe_z, p_any, p_one, p_full, mean_nf,
              n_samples, count_any, count_one, count_full
            FROM percolation_multi
            WHERE run_id=?
            ORDER BY pe_x, pe_y, pe_z
            """,
            (int(run_id),),
        ).fetchall()


def _percolation_pmf_rows(out_db: Path, run_id: int):
    with sqlite3.connect(out_db) as con:
        return con.execute(
            """
            SELECT pe_x, pe_y, pe_z, n_failed, count, n_samples
            FROM percolation_pmf
            WHERE run_id=?
            ORDER BY pe_x, pe_y, pe_z, n_failed
            """,
            (int(run_id),),
        ).fetchall()


def test_percolation_streams_ignore_workers_run_id_and_database_history(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    reused_dir = tmp_path / "reused"
    fresh_dir = tmp_path / "fresh"
    reused_dir.mkdir()
    fresh_dir.mkdir()
    scan = "xyz:0.01,0.41,0.2"

    reused_db = _run_sim(
        repo_root,
        reused_dir,
        scan,
        extra_args=[
            "--percolation-workers", "1",
            "--mc-workers", "1",
            "--mc-chunk-size", "3",
        ],
        samples=137,
    )
    reused_db = _run_sim(
        repo_root,
        reused_dir,
        scan,
        extra_args=[
            "--percolation-workers", "4",
            "--mc-workers", "4",
            "--mc-chunk-size", "17",
        ],
        samples=137,
    )
    fresh_db = _run_sim(
        repo_root,
        fresh_dir,
        scan,
        extra_args=[
            "--percolation-workers", "2",
            "--mc-workers", "2",
            "--mc-chunk-size", "29",
        ],
        samples=137,
    )

    with sqlite3.connect(reused_db) as con:
        reused_run_ids = [
            int(row[0])
            for row in con.execute("SELECT id FROM runs ORDER BY id")
        ]
    with sqlite3.connect(fresh_db) as con:
        fresh_run_id = int(con.execute("SELECT MAX(id) FROM runs").fetchone()[0])

    assert reused_run_ids == [1, 2]
    assert fresh_run_id == 1
    expected = _percolation_rows(reused_db, reused_run_ids[0])
    assert _percolation_rows(reused_db, reused_run_ids[1]) == expected
    assert _percolation_rows(fresh_db, fresh_run_id) == expected
    expected_pmfs = _percolation_pmf_rows(reused_db, reused_run_ids[0])
    assert (
        _percolation_pmf_rows(reused_db, reused_run_ids[1])
        == expected_pmfs
    )
    assert _percolation_pmf_rows(fresh_db, fresh_run_id) == expected_pmfs
