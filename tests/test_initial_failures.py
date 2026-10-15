"""Retired force-failure flags must not silently change stationary inputs.

Supported deterministic initial damage is covered by test_notional_damage.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "arguments",
    [
        ["--initial-fail-node-ids", "1"],
        ["--initial-fail-coords", "0,0,0"],
        ["--initial-fail-node-ids", "3", "--initial-fail-coords", "0,0,0"],
    ],
)
def test_retired_initial_failure_flags_are_rejected(arguments, tmp_path: Path) -> None:
    output_db = tmp_path / "results.db"
    result = subprocess.run(
        [sys.executable, "-m", "cli.cli_simulation", "--outdb", str(output_db), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert not output_db.exists()
