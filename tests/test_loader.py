import sqlite3
from pathlib import Path

import pytest

from pbm.io.loader import load_all


def make_db(tmp_path: Path, edge_rows, *, include_blocked: bool = True) -> Path:
    db = tmp_path / "input.db"
    con = sqlite3.connect(db)
    con.execute("PRAGMA foreign_keys=OFF;")
    con.executescript(
        """
        CREATE TABLE nodes (
          id INTEGER PRIMARY KEY,
          x INTEGER NOT NULL,
          y INTEGER NOT NULL,
          z INTEGER NOT NULL,
          pf_node REAL NOT NULL DEFAULT 0.0
        );
        CREATE TABLE initial_damage (
          node_id INTEGER PRIMARY KEY,
          FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        );
        """
    )
    blocked_column = ", blocked INTEGER NOT NULL DEFAULT 0" if include_blocked else ""
    con.executescript(
        f"""
        CREATE TABLE edges (
          id INTEGER PRIMARY KEY,
          org INTEGER NOT NULL,
          dst INTEGER NOT NULL,
          dir TEXT NOT NULL,
          pe REAL,
          pe_fwd REAL,
          pe_rev REAL,
          directed INTEGER NOT NULL
          {blocked_column}
        );
        """
    )
    con.executemany(
        "INSERT INTO nodes(id,x,y,z,pf_node) VALUES (?,?,?,?,?)",
        [
            (1, 0, 0, 0, 0.01),
            (2, 1, 0, 0, 0.02),
            (3, 2, 0, 0, 0.03),
        ],
    )
    if edge_rows:
        columns = (
            "id,org,dst,dir,pe,pe_fwd,pe_rev,directed,blocked"
            if include_blocked
            else "id,org,dst,dir,pe,pe_fwd,pe_rev,directed"
        )
        placeholders = ",".join("?" for _ in edge_rows[0])
        con.executemany(
            f"INSERT INTO edges({columns}) VALUES ({placeholders})",
            edge_rows,
        )
    con.commit()
    con.close()
    return db


def test_loader_applies_exact_edge_truth_table(tmp_path: Path):
    db = make_db(
        tmp_path,
        [
            (10, 1, 2, "x", 0.25, None, None, 0, 0),
            (11, 2, 3, "x", None, 0.0, 0.9, 1, 0),
        ],
    )
    loaded = load_all(db)
    e_by_id = {e.id: e for e in loaded.edges}

    undirected = e_by_id[10]
    assert undirected.directed == 0
    assert undirected.pe == pytest.approx(0.25)
    assert undirected.pe_fwd is None
    assert undirected.pe_rev is None
    assert undirected.eff_fwd == pytest.approx(0.25)
    assert undirected.eff_rev == pytest.approx(0.25)

    one_way = e_by_id[11]
    assert one_way.directed == 1
    assert one_way.pe is None
    assert one_way.pe_fwd == pytest.approx(0.0)
    assert one_way.pe_rev == pytest.approx(0.9)
    assert one_way.eff_fwd == pytest.approx(0.0)
    assert one_way.eff_rev == pytest.approx(0.9)


def test_database_directed_field_is_authoritative_when_probabilities_are_equal(
    tmp_path: Path,
):
    db = make_db(
        tmp_path,
        [(10, 1, 2, "x", None, 0.4, 0.4, 1, 0)],
    )
    edge = load_all(db).edges[0]
    assert edge.directed == 1
    assert edge.pe is None
    assert edge.eff_fwd == pytest.approx(0.4)
    assert edge.eff_rev == pytest.approx(0.4)


def test_blocked_edge_preserves_nominal_values_but_has_zero_effective_values(
    tmp_path: Path,
):
    db = make_db(
        tmp_path,
        [(10, 1, 2, "x", None, 0.3, 0.7, 1, 1)],
    )
    edge = load_all(db).edges[0]
    assert edge.blocked == 1
    assert edge.pe_fwd == pytest.approx(0.3)
    assert edge.pe_rev == pytest.approx(0.7)
    assert edge.eff_fwd == 0.0
    assert edge.eff_rev == 0.0


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ((10, 1, 2, "x", None, None, None, 0, 0), "directed=0 requires"),
        (
            (10, 1, 2, "x", 0.2, 0.2, None, 0, 0),
            "requires pe_fwd and pe_rev to be NULL",
        ),
        (
            (10, 1, 2, "x", 0.2, 0.2, 0.3, 1, 0),
            "directed=1 requires pe to be NULL",
        ),
        (
            (10, 1, 2, "x", None, 0.2, None, 1, 0),
            "missing pe_rev",
        ),
        ((10, 1, 2, "x", 1.1, None, None, 0, 0), "finite probability"),
        ((10, 1, 2, "x", float("inf"), None, None, 0, 0), "finite probability"),
        ((10, 1, 2, "x", 0.2, None, None, 2, 0), "directed must be 0 or 1"),
        ((10, 1, 2, "x", 0.2, None, None, 0, 2), "blocked must be 0 or 1"),
        ((10, 1, 1, "x", 0.2, None, None, 0, 0), "self-loop"),
        ((10, 2, 1, "x", 0.2, None, None, 0, 0), "canonical ordering"),
        ((10, 1, 99, "x", 0.2, None, None, 0, 0), "unknown node endpoint"),
        ((10, 1, 2, "u", 0.2, None, None, 0, 0), "dir must be"),
    ],
)
def test_loader_rejects_malformed_edge_rows(tmp_path: Path, row, message):
    db = make_db(tmp_path, [row])
    with pytest.raises(ValueError, match=message):
        load_all(db)


def test_loader_rejects_duplicate_adjacency_even_without_sql_constraint(
    tmp_path: Path,
):
    db = make_db(
        tmp_path,
        [
            (10, 1, 2, "x", 0.2, None, None, 0, 0),
            (11, 1, 2, "x", None, 0.3, 0.4, 1, 0),
        ],
    )
    with pytest.raises(ValueError, match="duplicate adjacency"):
        load_all(db)


def test_loader_rejects_legacy_edge_schema_without_blocked(tmp_path: Path):
    db = make_db(
        tmp_path,
        [(10, 1, 2, "x", 0.2, None, None, 0)],
        include_blocked=False,
    )
    with pytest.raises(RuntimeError, match="missing required.*blocked"):
        load_all(db)


def test_loader_loads_sorted_canonical_notional_damage(tmp_path: Path):
    db = make_db(tmp_path, [])
    with sqlite3.connect(db) as con:
        con.executemany(
            "INSERT INTO initial_damage(node_id) VALUES (?);",
            [(3,), (1,)],
        )

    loaded = load_all(db)
    assert loaded.damage_node_ids == (1, 3)
    assert "notional-damage nodes: 2" in loaded.summary


def test_loader_rejects_unknown_notional_damage_node(tmp_path: Path):
    db = make_db(tmp_path, [])
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO initial_damage(node_id) VALUES (99);")

    with pytest.raises(ValueError, match="unknown node id 99"):
        load_all(db)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf")])
def test_loader_rejects_invalid_pf_node(tmp_path: Path, value: float):
    db = make_db(tmp_path, [])
    with sqlite3.connect(db) as con:
        con.execute("UPDATE nodes SET pf_node=? WHERE id=1;", (value,))

    with pytest.raises(ValueError, match="pf_node must be a finite probability"):
        load_all(db)


def test_loader_rejects_legacy_node_probability_fields(tmp_path: Path):
    db = make_db(tmp_path, [])
    with sqlite3.connect(db) as con:
        con.execute("ALTER TABLE nodes ADD COLUMN beta REAL;")

    with pytest.raises(RuntimeError, match="no longer stores beta or force_fail"):
        load_all(db)
