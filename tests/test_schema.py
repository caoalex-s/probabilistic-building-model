from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from pbm.io.schema import ensure_schema


def tables(con: sqlite3.Connection) -> set[str]:
    return {name for (name,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view');"
    )}


def test_input_schema_creation(tmp_path: Path):
    db = tmp_path / "input.db"
    con = ensure_schema(db, "input")
    ts = tables(con)
    assert {"nodes", "edges", "initial_damage"}.issubset(ts)

    node_columns = {
        row[1]: row
        for row in con.execute("PRAGMA table_info(nodes);").fetchall()
    }
    assert set(node_columns) == {"id", "x", "y", "z", "pf_node"}
    assert "beta" not in node_columns
    assert "force_fail" not in node_columns

    damage_columns = {
        row[1]: row
        for row in con.execute("PRAGMA table_info(initial_damage);").fetchall()
    }
    assert set(damage_columns) == {"node_id"}
    assert damage_columns["node_id"][5] == 1  # primary key

    edge_columns = {
        row[1]: row
        for row in con.execute("PRAGMA table_info(edges);").fetchall()
    }
    assert {
        "id",
        "org",
        "dst",
        "dir",
        "pe",
        "pe_fwd",
        "pe_rev",
        "directed",
        "blocked",
    }.issubset(edge_columns)

    # Canonical undirected default.
    with con:
        con.execute(
            """
            INSERT INTO nodes(id,x,y,z) VALUES
              (1,0,0,0), (2,1,0,0), (3,2,0,0), (4,3,0,0),
              (5,4,0,0), (6,5,0,0), (7,6,0,0), (8,7,0,0);
            """
        )
        con.execute("INSERT INTO edges(org,dst,dir) VALUES (1,2,'x');")
        con.execute("INSERT INTO initial_damage(node_id) VALUES (1);")
    row = con.execute(
        "SELECT pe, pe_fwd, pe_rev, directed, blocked FROM edges WHERE org=1 AND dst=2"
    ).fetchone()
    assert row == (0.1, None, None, 0, 0)

    # A bidirectional directed row is valid even when the two values are equal.
    with con:
        con.execute(
            """
            INSERT INTO edges(org,dst,dir,pe,pe_fwd,pe_rev,directed,blocked)
            VALUES (2,3,'x',NULL,0.4,0.4,1,0);
            """
        )

    with pytest.raises(sqlite3.IntegrityError):
        with con:
            con.execute("INSERT INTO edges(org,dst,dir) VALUES (2,1,'x');")

    with pytest.raises(sqlite3.IntegrityError):
        with con:
            con.execute("INSERT INTO edges(org,dst,dir) VALUES (1,2,'x');")

    invalid_rows = [
        (3, 4, "x", None, None, None, 0, 0),
        (4, 5, "x", 0.2, 0.3, None, 0, 0),
        (5, 6, "x", 0.2, 0.3, 0.4, 1, 0),
        (6, 7, "x", None, 0.3, None, 1, 0),
        (7, 8, "x", 1.1, None, None, 0, 0),
    ]
    for invalid in invalid_rows:
        with pytest.raises(sqlite3.IntegrityError):
            with con:
                con.execute(
                    """
                    INSERT INTO edges(
                      org,dst,dir,pe,pe_fwd,pe_rev,directed,blocked
                    ) VALUES (?,?,?,?,?,?,?,?);
                    """,
                    invalid,
                )

    with pytest.raises(sqlite3.IntegrityError):
        with con:
            con.execute("UPDATE nodes SET pf_node=1.1 WHERE id=1;")

    with pytest.raises(sqlite3.IntegrityError):
        with con:
            con.execute("INSERT INTO initial_damage(node_id) VALUES (1);")

    with pytest.raises(sqlite3.IntegrityError):
        with con:
            con.execute("INSERT INTO initial_damage(node_id) VALUES (999);")


def test_output_schema_creation(tmp_path: Path):
    db = tmp_path / "results.db"
    con = ensure_schema(db, "output")
    ts = tables(con)

    # Minimum expected set for current pipeline.
    expected = {
        "runs",
        "pmf",
        "collapse_probs",
        "edge_usage",
        "edge_attempts",
        "percolation",
        "percolation_multi",
        "percolation_pmf",
        "node_influence",
        "nodes",
        "edges",
    }
    assert expected.issubset(ts)

    percolation_columns = {
        row[1]
        for row in con.execute("PRAGMA table_info(percolation_multi);").fetchall()
    }
    assert {
        "n_samples",
        "count_any",
        "count_one",
        "count_full",
    }.issubset(percolation_columns)

    percolation_pmf_columns = {
        row[1]
        for row in con.execute("PRAGMA table_info(percolation_pmf);").fetchall()
    }
    assert {
        "run_id",
        "pe_x",
        "pe_y",
        "pe_z",
        "n_failed",
        "count",
        "p",
        "n_samples",
    } == percolation_pmf_columns
