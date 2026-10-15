"""Strict loader for the stationary probabilistic building model.

The active input database contains:

* nodes with one executable probability, ``pf_node``;
* contract-valid stationary edges; and
* one simultaneous notional-damage set, ``initial_damage(node_id)``.

Legacy reliability-index, forced-failure, and staged-action fields are rejected
instead of inferred, clipped, or executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import sqlite3
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Node:
    """One compartment with its authoritative spontaneous-failure probability."""

    id: int
    x: int
    y: int
    z: int
    pf_node: float


@dataclass(frozen=True)
class Edge:
    """One canonical stationary adjacency."""

    id: int
    org: int
    dst: int
    pe: Optional[float]
    pe_fwd: Optional[float]
    pe_rev: Optional[float]
    blocked: int
    dir: str
    directed: int
    eff_fwd: float
    eff_rev: float


@dataclass(frozen=True)
class Loaded:
    """Validated stationary model data."""

    nodes: list[Node]
    edges: list[Edge]
    damage_node_ids: tuple[int, ...]
    summary: str


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite database without permitting mutation."""

    uri = f"file:{Path(path)}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def _has_table(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;",
            (table,),
        ).fetchone()
        is not None
    )


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table});")
    }


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer; got {value!r}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer; got {value!r}.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be an integer; got {value!r}.")
    return int(numeric)


def _node_probability(value: object, *, node_id: object) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"nodes row id={node_id}: pf_node must be a finite probability "
            f"in [0,1]; got {value!r}."
        ) from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"nodes row id={node_id}: pf_node must be a finite probability "
            f"in [0,1]; got {value!r}."
        )
    return probability


def _edge_probability(
    value: object,
    *,
    field: str,
    row_id: object,
) -> Optional[float]:
    if value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"edges row id={row_id}: {field} must be NULL or a finite "
            f"probability in [0,1]; got {value!r}."
        ) from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"edges row id={row_id}: {field} must be a finite probability "
            f"in [0,1]; got {value!r}."
        )
    return probability


def _binary_edge_flag(value: object, *, field: str, row_id: object) -> int:
    flag = _integer(value, label=f"edges row id={row_id}: {field}")
    if flag not in (0, 1):
        raise ValueError(
            f"edges row id={row_id}: {field} must be 0 or 1; got {value!r}."
        )
    return flag


def _validate_schema(connection: sqlite3.Connection) -> None:
    required_tables = {"nodes", "edges", "initial_damage"}
    missing_tables = sorted(
        table for table in required_tables if not _has_table(connection, table)
    )
    if missing_tables:
        raise RuntimeError(
            "Input database is missing required stationary table(s) "
            f"{missing_tables}. Recreate or migrate the database."
        )
    if _has_table(connection, "staged_actions"):
        raise RuntimeError(
            "The staged_actions table is not supported by the stationary "
            "version-1.0 model. Recreate or migrate the input database."
        )

    expected_nodes = {"id", "x", "y", "z", "pf_node"}
    actual_nodes = _columns(connection, "nodes")
    if actual_nodes != expected_nodes:
        raise RuntimeError(
            "Existing input database uses an incompatible nodes schema. "
            f"Expected {sorted(expected_nodes)}, got {sorted(actual_nodes)}. "
            "The executable schema no longer stores beta or force_fail."
        )

    expected_damage = {"node_id"}
    actual_damage = _columns(connection, "initial_damage")
    if actual_damage != expected_damage:
        raise RuntimeError(
            "Existing input database uses an incompatible initial_damage "
            f"schema. Expected {sorted(expected_damage)}, got "
            f"{sorted(actual_damage)}. Recreate or migrate the database."
        )
    damage_info = list(
        connection.execute("PRAGMA table_info(initial_damage);")
    )
    if len(damage_info) != 1 or int(damage_info[0]["pk"]) != 1:
        raise RuntimeError(
            "initial_damage.node_id must be the table primary key. Recreate "
            "or migrate the database."
        )
    damage_foreign_keys = list(
        connection.execute("PRAGMA foreign_key_list(initial_damage);")
    )
    has_node_foreign_key = any(
        row["table"] == "nodes"
        and row["from"] == "node_id"
        and row["to"] == "id"
        for row in damage_foreign_keys
    )
    if not has_node_foreign_key:
        raise RuntimeError(
            "initial_damage.node_id must reference nodes(id). Recreate or "
            "migrate the database."
        )

    required_edges = {
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
    actual_edges = _columns(connection, "edges")
    missing_edges = sorted(required_edges - actual_edges)
    if missing_edges:
        raise RuntimeError(
            "edges table missing required stationary edge-contract columns: "
            f"{missing_edges}. Recreate or migrate the database."
        )


def _load_nodes(connection: sqlite3.Connection) -> list[Node]:
    nodes: list[Node] = []
    seen_ids: set[int] = set()
    seen_coordinates: set[tuple[int, int, int]] = set()

    query = "SELECT id, x, y, z, pf_node FROM nodes ORDER BY id;"
    for row in connection.execute(query):
        node_id = _integer(row["id"], label="nodes.id")
        coordinate = tuple(
            _integer(row[axis], label=f"nodes row id={node_id}: {axis}")
            for axis in ("x", "y", "z")
        )
        if node_id in seen_ids:
            raise ValueError(f"Duplicate node id {node_id}.")
        if coordinate in seen_coordinates:
            raise ValueError(f"Duplicate node coordinate {coordinate}.")
        seen_ids.add(node_id)
        seen_coordinates.add(coordinate)
        nodes.append(
            Node(
                id=node_id,
                x=coordinate[0],
                y=coordinate[1],
                z=coordinate[2],
                pf_node=_node_probability(row["pf_node"], node_id=node_id),
            )
        )
    return nodes


def _load_edges(
    connection: sqlite3.Connection,
    *,
    node_ids: set[int],
) -> list[Edge]:
    edges: list[Edge] = []
    seen_adjacencies: dict[tuple[int, int], int] = {}
    query = """
        SELECT id, org, dst, dir, pe, pe_fwd, pe_rev, directed, blocked
        FROM edges
        ORDER BY id;
    """

    for row in connection.execute(query):
        row_id = _integer(row["id"], label="edges.id")
        org = _integer(row["org"], label=f"edges row id={row_id}: org")
        dst = _integer(row["dst"], label=f"edges row id={row_id}: dst")
        directed = _binary_edge_flag(
            row["directed"], field="directed", row_id=row_id
        )
        blocked = _binary_edge_flag(
            row["blocked"], field="blocked", row_id=row_id
        )

        if org == dst:
            raise ValueError(
                f"edges row id={row_id}: self-loop ({org}, {dst}) is not allowed."
            )
        if org > dst:
            raise ValueError(
                f"edges row id={row_id}: endpoints must use canonical ordering "
                f"org < dst; got ({org}, {dst})."
            )
        unknown_endpoints = sorted({org, dst} - node_ids)
        if unknown_endpoints:
            raise ValueError(
                f"edges row id={row_id}: unknown node endpoint(s) "
                f"{unknown_endpoints}."
            )

        adjacency = (org, dst)
        if adjacency in seen_adjacencies:
            raise ValueError(
                f"edges row id={row_id}: duplicate adjacency {adjacency}; "
                f"already defined by edge id={seen_adjacencies[adjacency]}."
            )
        seen_adjacencies[adjacency] = row_id

        axis = row["dir"]
        if axis not in {"x", "y", "z"}:
            raise ValueError(
                f"edges row id={row_id}: dir must be one of 'x', 'y', or 'z'; "
                f"got {axis!r}."
            )

        pe = _edge_probability(row["pe"], field="pe", row_id=row_id)
        pe_fwd = _edge_probability(
            row["pe_fwd"], field="pe_fwd", row_id=row_id
        )
        pe_rev = _edge_probability(
            row["pe_rev"], field="pe_rev", row_id=row_id
        )

        if directed == 0:
            if pe is None:
                raise ValueError(
                    f"edges row id={row_id}: directed=0 requires non-NULL pe."
                )
            if pe_fwd is not None or pe_rev is not None:
                raise ValueError(
                    f"edges row id={row_id}: directed=0 requires pe_fwd and "
                    "pe_rev to be NULL."
                )
            nominal_fwd = nominal_rev = pe
        else:
            if pe is not None:
                raise ValueError(
                    f"edges row id={row_id}: directed=1 requires pe to be NULL."
                )
            missing_direction = [
                field
                for field, value in (("pe_fwd", pe_fwd), ("pe_rev", pe_rev))
                if value is None
            ]
            if missing_direction:
                raise ValueError(
                    f"edges row id={row_id}: directed=1 requires non-NULL "
                    "pe_fwd and pe_rev; missing "
                    f"{', '.join(missing_direction)}."
                )
            nominal_fwd = pe_fwd
            nominal_rev = pe_rev

        edges.append(
            Edge(
                id=row_id,
                org=org,
                dst=dst,
                pe=pe,
                pe_fwd=pe_fwd,
                pe_rev=pe_rev,
                blocked=blocked,
                dir=str(axis),
                directed=directed,
                eff_fwd=0.0 if blocked else float(nominal_fwd),
                eff_rev=0.0 if blocked else float(nominal_rev),
            )
        )

    return edges


def _load_damage(
    connection: sqlite3.Connection,
    *,
    node_ids: set[int],
) -> tuple[int, ...]:
    damage_ids: list[int] = []
    seen: set[int] = set()
    for row in connection.execute(
        "SELECT node_id FROM initial_damage ORDER BY node_id;"
    ):
        node_id = _integer(
            row["node_id"], label="initial_damage.node_id"
        )
        if node_id not in node_ids:
            raise ValueError(
                f"initial_damage references unknown node id {node_id}."
            )
        if node_id in seen:
            raise ValueError(
                f"initial_damage contains duplicate node id {node_id}."
            )
        seen.add(node_id)
        damage_ids.append(node_id)
    return tuple(damage_ids)


def load_all(db_path: str | Path) -> Loaded:
    """Load and validate one stationary model from an input database."""

    connection = connect_readonly(db_path)
    try:
        _validate_schema(connection)
        nodes = _load_nodes(connection)
        node_ids = {node.id for node in nodes}
        edges = _load_edges(connection, node_ids=node_ids)
        damage_node_ids = _load_damage(connection, node_ids=node_ids)
    finally:
        connection.close()

    directed_count = sum(edge.directed for edge in edges)
    blocked_count = sum(edge.blocked for edge in edges)
    summary = (
        f"Loaded {len(nodes)} nodes | {len(edges)} edges "
        f"(directed: {directed_count}; blocked: {blocked_count}) | "
        f"notional-damage nodes: {len(damage_node_ids)}"
    )
    logger.info(summary)
    return Loaded(
        nodes=nodes,
        edges=edges,
        damage_node_ids=damage_node_ids,
        summary=summary,
    )


def print_summary(loaded: Loaded) -> None:
    """Print a compact model summary."""

    print(loaded.summary)
    if loaded.edges:
        effective = [
            probability
            for edge in loaded.edges
            for probability in (edge.eff_fwd, edge.eff_rev)
        ]
        print(
            f"Effective pe range: [{min(effective):.6g}, "
            f"{max(effective):.6g}]"
        )
