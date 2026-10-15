from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence, Any

def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a read/write connection with WAL & FK enabled."""
    db_path = str(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA synchronous = NORMAL;")
    return con

def init_db(db_path: str | Path, schema_path: str | Path) -> sqlite3.Connection:
    """Create/overwrite an empty DB using the provided SQL schema file."""
    db_path, schema_path = Path(db_path), Path(schema_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = schema_path.read_text(encoding="utf-8")
    con = connect(db_path)
    with con:
        con.executescript(sql)
    return con

def exec_many(con: sqlite3.Connection, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> None:
    """Execute parameterized INSERT/UPDATE in bulk under a single transaction."""
    with con:
        con.executemany(sql, seq_of_params)

