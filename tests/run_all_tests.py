# -*- coding: utf-8 -*-
"""
run_all_tests.py

Unified test runner for the probabilistic-building-model codebase.

- Executes all pytest tests under ./tests
- Logs output to output/logs/test.log
- Optionally generates an HTML report if pytest-html is available
"""

from __future__ import annotations

import sys
import logging
import datetime
from pathlib import Path

import pytest


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
OUTPUT_DIR = REPO_ROOT / "output"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_PATH = OUTPUT_DIR / "test_report.html"

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
log_file = LOG_DIR / "test.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.info("=" * 72)
logging.info("Running full test suite")
logging.info(f"Timestamp      : {timestamp}")
logging.info(f"Repository root: {REPO_ROOT}")
logging.info(f"Tests directory: {TESTS_DIR}")
logging.info("=" * 72)


# ---------------------------------------------------------------------
# Build pytest arguments
# ---------------------------------------------------------------------
pytest_args = [
    str(TESTS_DIR),
    "-v",
    "--maxfail=1",
    "--tb=short",
    "--color=yes",
]

# Add HTML report only if pytest-html is installed
try:
    import pytest_html  # noqa: F401

    pytest_args.extend(
        [
            f"--html={REPORT_PATH}",
            "--self-contained-html",
        ]
    )
    logging.info("pytest-html detected: HTML report will be generated.")
except ImportError:
    logging.info("pytest-html not installed: skipping HTML report.")

logging.info("Pytest arguments:")
for arg in pytest_args:
    logging.info(f"  {arg}")


# ---------------------------------------------------------------------
# Run pytest
# ---------------------------------------------------------------------
logging.info("\nStarting pytest...\n")
exit_code = pytest.main(pytest_args)

logging.info("\n" + "-" * 72)
logging.info(f"Pytest finished with exit code {exit_code}")
logging.info("-" * 72)

if exit_code != 0:
    logging.error("Some tests FAILED.")
    sys.exit(exit_code)

logging.info("All tests PASSED.")
