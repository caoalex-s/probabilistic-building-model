from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repository root is importable when running pytest without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Keep expensive publication regressions opt-in."""

    group = parser.getgroup("scientific regression")
    group.addoption(
        "--run-publication",
        action="store_true",
        default=False,
        help="run tests marked 'publication' (larger Monte Carlo samples)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip publication regressions unless explicitly requested."""

    if config.getoption("--run-publication"):
        return
    skip = pytest.mark.skip(
        reason="publication regression; pass --run-publication to execute"
    )
    for item in items:
        if "publication" in item.keywords:
            item.add_marker(skip)
