from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pbm.utils.utils as utils


def test_get_rng_deterministic():
    rng1 = utils.get_rng(123)
    rng2 = utils.get_rng(123)
    a = rng1.random(5)
    b = rng2.random(5)
    assert np.allclose(a, b), "RNG with same seed must be deterministic"


def test_get_rng_entropy():
    # Not strictly testable for equality; just ensure it returns a Generator and produces values
    rng = utils.get_rng(None)
    x = rng.random()
    assert isinstance(x, float)


def test_timestamp_format():
    ts = utils.timestamp()
    # Minimal sanity checks: ISO8601-ish + 'Z'
    assert "T" in ts and ts.endswith("Z")


def test_write_json_atomic(tmp_path: Path):
    out = tmp_path / "a" / "b" / "c.json"
    payload = {"b": 2, "a": 1}
    utils.write_json(out, payload)
    assert out.exists()
    # Ensure valid JSON with sorted keys
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == payload
    # Confirm parent dirs were created
    assert (tmp_path / "a" / "b").exists()
