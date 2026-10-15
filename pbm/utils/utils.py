from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Return a NumPy random Generator.
    If seed is None, returns an entropy-seeded generator.
    """
    return np.random.default_rng(seed)


def timestamp() -> str:
    """
    ISO 8601 UTC timestamp with 'Z' suffix, e.g. '2025-10-21T07:30:12.345678Z'.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: os.PathLike | str, obj: Any) -> None:
    """
    Atomically write JSON to `path`. Creates parent directories if needed.
    Pretty prints (sorted keys, 2-space indent).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    # Atomic replace (works on Windows, Linux, macOS)
    os.replace(tmp, path)
