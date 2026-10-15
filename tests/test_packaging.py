"""Release-packaging checks that also run from an installed wheel."""

from __future__ import annotations

from pathlib import Path
import pytest

from pbm import __version__
from pbm.io.schema import get_schema_paths, read_schema_sql


def test_public_version_is_v1() -> None:
    assert __version__ == "1.0.0"


def test_authoritative_schemas_are_packaged() -> None:
    paths = get_schema_paths()
    assert paths.input_schema.is_file()
    assert paths.output_schema.is_file()
    assert "CREATE TABLE IF NOT EXISTS nodes" in read_schema_sql("input")
    assert "CREATE TABLE IF NOT EXISTS runs" in read_schema_sql("output")


def test_explicit_source_root_uses_same_packaged_schemas() -> None:
    root = Path(__file__).resolve().parents[1]
    source = get_schema_paths(repo_root=root)
    assert source.input_schema.read_text(encoding="utf8") == read_schema_sql("input")
    assert source.output_schema.read_text(encoding="utf8") == read_schema_sql("output")


def test_explicit_source_root_rejects_legacy_unpacked_schemas(tmp_path: Path) -> None:
    legacy = tmp_path / "resources" / "sql"
    legacy.mkdir(parents=True)
    for kind in ("input", "output"):
        (legacy / f"{kind}_schema.sql").write_text(read_schema_sql(kind), encoding="utf8")
    with pytest.raises(FileNotFoundError, match="Expected pbm/resources/sql"):
        get_schema_paths(repo_root=tmp_path)
