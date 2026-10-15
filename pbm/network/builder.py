"""
Lattice builder for a non-periodic 6-connected 3D grid.

- Nodes: integer coordinates (x, y, z), 0-based
- Edges: undirected, canonical ordering (org < dst)
- Edges include 'dir' ∈ {'x','y','z'} to satisfy NOT NULL in schema
- Writes to the specified input database; create its parent directory first

CLI:
    pbm-build --nx 4 --ny 3 --nz 2 --db input/input.db --truncate yes
"""

from __future__ import annotations
import argparse
import sqlite3
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from pbm.io.schema import apply_schema, connect_sqlite


NODES_TABLE = "nodes"
EDGES_TABLE = "edges"
NODE_ID_COL = "id"
EDGE_ORG_COL = "org"
EDGE_DST_COL = "dst"
EDGE_DIR_COL = "dir"

def _detect_node_id_col(conn: sqlite3.Connection) -> str:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({NODES_TABLE});").fetchall()]
    if NODE_ID_COL in cols:
        return NODE_ID_COL
    for cand in ("node_id", "nid", "pk"):
        if cand in cols:
            return cand
    raise RuntimeError("Could not find nodes PK column ('id','node_id','nid','pk').")
    
def _maybe_truncate(conn: sqlite3.Connection):
    """Clear one complete stationary model, including notional damage."""

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}

    if "initial_damage" in tables:
        conn.execute("DELETE FROM initial_damage;")
    if EDGES_TABLE in tables:
        conn.execute(f"DELETE FROM {EDGES_TABLE};")
    if NODES_TABLE in tables:
        conn.execute(f"DELETE FROM {NODES_TABLE};")

    conn.commit()

def _ensure_schema(conn: sqlite3.Connection):
    # Authoritative schema is packaged under pbm/resources/sql/input_schema.sql
    apply_schema(conn, "input")
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    }
    if "staged_actions" in tables:
        raise RuntimeError(
            "Existing input database contains unsupported staged_actions. "
            "Recreate or migrate the database before running the builder."
        )

    node_cols = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({NODES_TABLE});").fetchall()
    }
    required_node_cols = {"id", "x", "y", "z", "pf_node"}
    if node_cols != required_node_cols:
        raise RuntimeError(
            "Existing input database uses an incompatible nodes schema; "
            f"expected {sorted(required_node_cols)}, got {sorted(node_cols)}. "
            "Recreate or migrate the database before running the builder."
        )

    damage_info = conn.execute(
        "PRAGMA table_info(initial_damage);"
    ).fetchall()
    damage_cols = {row[1] for row in damage_info}
    if damage_cols != {"node_id"}:
        raise RuntimeError(
            "Existing input database uses an incompatible initial_damage "
            "schema; expected only node_id. Recreate or migrate the database "
            "before running the builder."
        )
    if len(damage_info) != 1 or int(damage_info[0][5]) != 1:
        raise RuntimeError(
            "Existing initial_damage.node_id is not the primary key. "
            "Recreate or migrate the database before running the builder."
        )
    damage_foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(initial_damage);"
    ).fetchall()
    if not any(
        row[2] == NODES_TABLE and row[3] == "node_id" and row[4] == NODE_ID_COL
        for row in damage_foreign_keys
    ):
        raise RuntimeError(
            "Existing initial_damage.node_id does not reference nodes(id). "
            "Recreate or migrate the database before running the builder."
        )

    edge_cols = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({EDGES_TABLE});").fetchall()
    }
    required = {
        "id",
        "org",
        "dst",
        "dir",
        "pe",
        "pe_fwd",
        "pe_rev",
        "directed",
        "blocked",
    }
    missing = required - edge_cols
    if missing:
        raise RuntimeError(
            "Existing input database uses an incompatible edge schema; "
            f"missing columns {sorted(missing)}. Recreate or migrate the "
            "database before running the builder."
        )

    unique_adjacency = False
    for index_row in conn.execute(f"PRAGMA index_list({EDGES_TABLE});"):
        if not int(index_row[2]):
            continue
        index_name = index_row[1]
        escaped_index_name = str(index_name).replace("'", "''")
        columns = [
            info_row[2]
            for info_row in conn.execute(
                f"PRAGMA index_info('{escaped_index_name}');"
            ).fetchall()
        ]
        if columns == [EDGE_ORG_COL, EDGE_DST_COL]:
            unique_adjacency = True
            break
    if not unique_adjacency:
        raise RuntimeError(
            "Existing input database does not enforce UNIQUE(org,dst). "
            "Recreate or migrate the database before running the builder."
        )


def _lin_nodes(nx: int, ny: int, nz: int) -> List[Tuple[int,int,int]]:
    return [(x, y, z) for z in range(nz) for y in range(ny) for x in range(nx)]

def _insert_nodes(conn: sqlite3.Connection, coords: List[Tuple[int,int,int]]):
    sql = f"INSERT INTO {NODES_TABLE}(x,y,z) VALUES(?,?,?)"
    conn.executemany(sql, coords)

def _coord_id_map(conn: sqlite3.Connection, id_col: str) -> Dict[Tuple[int,int,int], int]:
    mapping = {}
    for row in conn.execute(f"SELECT {id_col}, x, y, z FROM {NODES_TABLE}"):
        mapping[(row[1], row[2], row[3])] = row[0]
    if not mapping:
        raise RuntimeError("No nodes found after insertion.")
    return mapping

def _neighbors_with_dir(x: int, y: int, z: int, nx: int, ny: int, nz: int):
    # Only forward neighbors (+x,+y,+z) to avoid duplicates; undirected edge kept with org<dst
    if x + 1 < nx: yield (x+1, y, z, 'x')
    if y + 1 < ny: yield (x, y+1, z, 'y')
    if z + 1 < nz: yield (x, y, z+1, 'z')

def _build_edges(conn: sqlite3.Connection, nx: int, ny: int, nz: int, id_col: str):
    coords = _lin_nodes(nx, ny, nz)
    cmap = _coord_id_map(conn, id_col)

    # Insert one valid undirected row per canonical adjacency.
    edges: List[Tuple[int,int,str,float,Optional[float],Optional[float],int,int]] = []
    seen: set[Tuple[int, int]] = set()
    for (x, y, z) in coords:
        org_id = cmap[(x, y, z)]
        for (xn, yn, zn, d) in _neighbors_with_dir(x, y, z, nx, ny, nz):
            dst_id = cmap[(xn, yn, zn)]
            org, dst = (org_id, dst_id) if org_id < dst_id else (dst_id, org_id)
            adjacency = (org, dst)
            if adjacency in seen:
                raise RuntimeError(
                    f"Duplicate lattice adjacency generated for org={org}, dst={dst}."
                )
            seen.add(adjacency)
            edges.append((org, dst, d, 0.1, None, None, 0, 0))

    if edges:
        q = (
            f"INSERT INTO {EDGES_TABLE}("
            f"{EDGE_ORG_COL},{EDGE_DST_COL},{EDGE_DIR_COL},"
            "pe,pe_fwd,pe_rev,directed,blocked"
            ") VALUES(?,?,?,?,?,?,?,?)"
        )
        conn.executemany(q, edges)

def build_and_insert(nx: int, ny: int, nz: int, db_path: Path, truncate: bool = True):
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("All dimensions must be positive.")
    db_path = Path(db_path)

    with connect_sqlite(db_path) as conn:
        _ensure_schema(conn) 
        if truncate:
            _maybe_truncate(conn)

        _insert_nodes(conn, _lin_nodes(nx, ny, nz))
        id_col = _detect_node_id_col(conn)
        _build_edges(conn, nx, ny, nz, id_col)
        conn.commit()

        n_nodes = conn.execute(f"SELECT COUNT(*) FROM {NODES_TABLE}").fetchone()[0]
        n_edges = conn.execute(f"SELECT COUNT(*) FROM {EDGES_TABLE}").fetchone()[0]
        print(f"[builder] nodes: {n_nodes}, edges: {n_edges}")

def _parse():
    from pbm import __version__

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--db", type=str, default="input/input.db")
    ap.add_argument("--truncate", type=str, choices=["yes", "no"], default="yes")
    return ap.parse_args()

def main() -> None:
    """Console entry point for building a regular stationary lattice."""

    args = _parse()
    build_and_insert(
        args.nx,
        args.ny,
        args.nz,
        Path(args.db),
        truncate=(args.truncate == "yes"),
    )


if __name__ == "__main__":
    main()
