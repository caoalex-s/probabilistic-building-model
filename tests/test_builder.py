from __future__ import annotations

import sqlite3
from pathlib import Path
import subprocess
import sys

import pytest


def _id_col(conn: sqlite3.Connection) -> str | None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(nodes);")]
    return "id" if "id" in cols else ("node_id" if "node_id" in cols else None)


def _deg_map(conn: sqlite3.Connection, id_col: str) -> dict[int, int]:
    deg: dict[int, int] = {}
    for (nid,) in conn.execute(f"SELECT {id_col} FROM nodes"):
        deg[int(nid)] = 0
    for (nid, cnt) in conn.execute("SELECT org, COUNT(*) FROM edges GROUP BY org"):
        deg[int(nid)] += int(cnt)
    for (nid, cnt) in conn.execute("SELECT dst, COUNT(*) FROM edges GROUP BY dst"):
        deg[int(nid)] += int(cnt)
    return deg


def _coord_to_id(conn: sqlite3.Connection, id_col: str, x: int, y: int, z: int) -> int:
    (nid,) = conn.execute(
        f"SELECT {id_col} FROM nodes WHERE x=? AND y=? AND z=?", (x, y, z)
    ).fetchone()
    return int(nid)


def _run_builder(db: Path, nx: int, ny: int, nz: int) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pbm.network.builder",
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--nz",
        str(nz),
        "--db",
        str(db),
        "--truncate",
        "yes",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def test_builder_does_not_import_unrelated_model_package(tmp_path: Path, monkeypatch):
    """An installed package called model must not affect lattice construction."""
    import os

    package = tmp_path / "model"
    package.mkdir()
    marker = tmp_path / "foreign-package-imported"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        encoding="utf8",
    )
    (package / "writer.py").write_text(
        "raise RuntimeError('Unrelated writer must not be imported')\n",
        encoding="utf8",
    )
    previous = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + (os.pathsep + previous if previous else ""))
    db = tmp_path / "input.db"
    _run_builder(db, 2, 2, 2)
    assert not marker.exists()
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 8
        assert connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 12


def test_2x2x1_counts_and_degrees(tmp_path: Path):
    db = tmp_path / "input.db"
    nx, ny, nz = 2, 2, 1
    _run_builder(db, nx, ny, nz)

    with sqlite3.connect(str(db)) as conn:
        id_col = _id_col(conn)
        assert id_col

        n_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert n_nodes == nx * ny * nz == 4
        assert n_edges == (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1) == 4

        bad = conn.execute("SELECT COUNT(*) FROM edges WHERE org >= dst").fetchone()[0]
        assert bad == 0
        duplicates = conn.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT org, dst FROM edges GROUP BY org, dst HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        assert duplicates == 0
        invalid_defaults = conn.execute(
            """
            SELECT COUNT(*) FROM edges
            WHERE directed != 0
               OR blocked != 0
               OR pe IS NULL
               OR pe_fwd IS NOT NULL
               OR pe_rev IS NOT NULL
            """
        ).fetchone()[0]
        assert invalid_defaults == 0

        deg = _deg_map(conn, id_col)
        assert set(deg.values()) == {2}


def test_3x3x1_topology(tmp_path: Path):
    db = tmp_path / "input.db"
    nx, ny, nz = 3, 3, 1
    _run_builder(db, nx, ny, nz)

    with sqlite3.connect(str(db)) as conn:
        id_col = _id_col(conn)
        assert id_col

        n_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert n_nodes == 9
        assert n_edges == (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1) == 12

        nid_corners = [
            _coord_to_id(conn, id_col, 0, 0, 0),
            _coord_to_id(conn, id_col, 2, 0, 0),
            _coord_to_id(conn, id_col, 0, 2, 0),
            _coord_to_id(conn, id_col, 2, 2, 0),
        ]
        nid_edges = [
            _coord_to_id(conn, id_col, 1, 0, 0),
            _coord_to_id(conn, id_col, 2, 1, 0),
            _coord_to_id(conn, id_col, 1, 2, 0),
            _coord_to_id(conn, id_col, 0, 1, 0),
        ]
        nid_center = _coord_to_id(conn, id_col, 1, 1, 0)

        deg = _deg_map(conn, id_col)
        for nid in nid_corners:
            assert deg[nid] == 2
        for nid in nid_edges:
            assert deg[nid] == 3
        assert deg[nid_center] == 4

        bad = conn.execute("SELECT COUNT(*) FROM edges WHERE org >= dst").fetchone()[0]
        assert bad == 0

        ok = conn.execute("SELECT COUNT(*) FROM edges WHERE dir IN ('x','y','z')").fetchone()[0]
        assert ok == n_edges


def test_incompatible_edge_schema_is_rejected_before_truncation(tmp_path: Path):
    from pbm.network.builder import build_and_insert

    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            CREATE TABLE nodes (
              id INTEGER PRIMARY KEY,
              x INTEGER NOT NULL,
              y INTEGER NOT NULL,
              z INTEGER NOT NULL,
              pf_node REAL NOT NULL DEFAULT 0.01
            );
            CREATE TABLE edges (
              id INTEGER PRIMARY KEY,
              org INTEGER NOT NULL,
              dst INTEGER NOT NULL,
              dir TEXT NOT NULL,
              pe REAL,
              pe_fwd REAL,
              pe_rev REAL,
              directed INTEGER NOT NULL
            );
            INSERT INTO nodes(id,x,y,z) VALUES (1,0,0,0), (2,1,0,0);
            INSERT INTO edges(id,org,dst,dir,pe,directed)
            VALUES (1,1,2,'x',0.1,0);
            """
        )

    with pytest.raises(RuntimeError, match="incompatible edge schema"):
        build_and_insert(2, 1, 1, db, truncate=True)

    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1


def test_truncation_clears_notional_damage_before_rebuild(tmp_path: Path):
    from pbm.network.builder import build_and_insert

    db = tmp_path / "input.db"
    build_and_insert(2, 2, 1, db, truncate=True)
    with sqlite3.connect(db) as con:
        con.execute("PRAGMA foreign_keys=ON;")
        node_id = con.execute("SELECT MIN(id) FROM nodes;").fetchone()[0]
        con.execute(
            "INSERT INTO initial_damage(node_id) VALUES (?);",
            (node_id,),
        )

    build_and_insert(3, 1, 1, db, truncate=True)
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM initial_damage;"
        ).fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM nodes;").fetchone()[0] == 3


def test_legacy_damage_schema_is_rejected_before_truncation(tmp_path: Path):
    from pbm.io.schema import ensure_schema
    from pbm.network.builder import build_and_insert

    db = tmp_path / "legacy-damage.db"
    con = ensure_schema(db, "input")
    with con:
        con.execute("DROP TABLE initial_damage;")
        con.execute(
            """
            CREATE TABLE initial_damage (
              id INTEGER PRIMARY KEY,
              type TEXT,
              action TEXT,
              x INTEGER,
              y INTEGER,
              z INTEGER,
              delay INTEGER
            );
            """
        )
        con.execute(
            "INSERT INTO nodes(id,x,y,z,pf_node) VALUES (1,0,0,0,0.01);"
        )
        con.execute(
            """
            INSERT INTO initial_damage(id,type,action,x,y,z,delay)
            VALUES (1,'node','force_fail',0,0,0,0);
            """
        )
    con.close()

    with pytest.raises(RuntimeError, match="incompatible initial_damage"):
        build_and_insert(2, 1, 1, db, truncate=True)

    with sqlite3.connect(db) as check:
        assert check.execute("SELECT COUNT(*) FROM nodes;").fetchone()[0] == 1
        assert check.execute(
            "SELECT COUNT(*) FROM initial_damage;"
        ).fetchone()[0] == 1
