from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_path: str | Path = "output/logs/app.log") -> None:
    """
    Minimal file logger setup. Idempotent (safe to call multiple times).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    if getattr(logger, "_scaffold_logging_configured", False):
        return

    logger.setLevel(logging.INFO)

    # File handler (rotating, small cap just to avoid runaway logs)
    fh = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=1, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (useful when running scripts)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Mark as configured to avoid duplicates
    logger._scaffold_logging_configured = True  # type: ignore[attr-defined]
