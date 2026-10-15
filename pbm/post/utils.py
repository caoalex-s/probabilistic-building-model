# -*- coding: utf-8 -*-
"""pbm.post.utils

Small post-processing utilities.

This project is intentionally YAML-free: run configuration is defined by
the CLI arguments and the databases themselves.
"""

from __future__ import annotations

from pathlib import Path


def ensure_figs_dir(figs_root: str | Path, run_id: int) -> Path:
    """Return the run-specific figure directory, creating it if necessary."""
    figs_root = Path(figs_root)
    run_dir = figs_root / f"run_{int(run_id)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
