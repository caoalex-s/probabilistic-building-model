from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import sqlite3
from typing import Literal, Optional


SchemaKind = Literal["input", "output"]


@dataclass(frozen=True)
class SchemaPaths:
    input_schema: Path
    output_schema: Path


def _schema_base(repo_root: Optional[Path] = None) -> Path:
    """Return the directory containing the authoritative packaged schemas.

    ``repo_root`` remains available for source-tree tooling and tests. The
    installed package path is used otherwise, so schema loading does not depend
    on the current working directory or on a cloned repository layout.
    An explicit source root must contain pbm/resources/sql/.
    """

    if repo_root is not None:
        root = Path(repo_root).resolve()
        base = root / "pbm" / "resources" / "sql"
        if base.is_dir():
            return base
        raise FileNotFoundError(
            "Could not locate packaged SQL schemas below repository root "
            f"{root}. Expected pbm/resources/sql/."
        )

    resource = files("pbm.resources.sql")
    base = Path(str(resource))
    if not base.is_dir():
        raise FileNotFoundError(
            "Installed package does not contain pbm/resources/sql/."
        )
    return base


def get_schema_paths(repo_root: Optional[Path] = None) -> SchemaPaths:
    """Locate the input and output SQLite schema files."""

    base = _schema_base(repo_root=repo_root)
    inp = base / "input_schema.sql"
    out = base / "output_schema.sql"

    missing = [path for path in (inp, out) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing schema file(s): " + ", ".join(str(path) for path in missing)
        )

    return SchemaPaths(input_schema=inp, output_schema=out)


def read_schema_sql(kind: SchemaKind, repo_root: Optional[Path] = None) -> str:
    """Read one authoritative SQLite schema as UTF-8 text."""

    paths = get_schema_paths(repo_root=repo_root)
    path = paths.input_schema if kind == "input" else paths.output_schema
    return path.read_text(encoding="utf-8")


def apply_schema(
    conn: sqlite3.Connection,
    kind: SchemaKind,
    *,
    repo_root: Optional[Path] = None,
) -> None:
    """Apply the requested schema to an open SQLite connection."""

    sql = read_schema_sql(kind, repo_root=repo_root)
    with conn:
        conn.executescript(sql)


def connect_sqlite(path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with foreign-key enforcement enabled."""

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_schema(
    db_path: str | Path,
    kind: SchemaKind,
    *,
    repo_root: Optional[Path] = None,
) -> sqlite3.Connection:
    """Open a database, apply the requested schema, and return the connection."""

    conn = connect_sqlite(db_path)
    apply_schema(conn, kind, repo_root=repo_root)
    return conn
