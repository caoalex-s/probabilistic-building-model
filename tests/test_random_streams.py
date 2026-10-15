from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pbm.sim.random_streams import (
    MAX_REALIZATION_INDEX,
    RNG_CONTRACT_VERSION,
    realization_rng,
    realization_seed_sequence,
)


def test_realization_stream_contract_known_vector():
    """Freeze the documented SeedSequence/PCG64 allocation as version 1."""
    assert RNG_CONTRACT_VERSION == "seedsequence-pcg64-realization-v1"
    actual = realization_rng(123, 0).bit_generator.random_raw(4)
    expected = np.array(
        [
            9991306608154205551,
            9906218914756280507,
            18337267986660269456,
            1438027098199660598,
        ],
        dtype=np.uint64,
    )
    assert np.array_equal(actual, expected)


def test_realization_streams_support_order_independent_random_access():
    indices = [17, 2, 9, 0]
    first_pass = {
        index: realization_rng(987, index).random(12)
        for index in indices
    }
    second_pass = {
        index: realization_rng(987, index).random(12)
        for index in reversed(indices)
    }

    for index in indices:
        assert np.array_equal(first_pass[index], second_pass[index])

    assert not np.array_equal(first_pass[0], first_pass[2])


@pytest.mark.parametrize(
    ("base_seed", "realization_index", "error_type"),
    [
        (-1, 0, ValueError),
        (1, -1, ValueError),
        (1, MAX_REALIZATION_INDEX + 1, ValueError),
        (1.5, 0, TypeError),
        (True, 0, TypeError),
    ],
)
def test_realization_stream_contract_rejects_invalid_keys(
    base_seed,
    realization_index,
    error_type,
):
    with pytest.raises(error_type):
        realization_seed_sequence(base_seed, realization_index)


def test_damage_campaign_uses_one_common_base_seed():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "damage_scenarios.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

    assert "SEED + scenario_id" not in source
    assert source.count('"--seed", str(SEED)') == 5
    assert "--damage-coords" in source
    assert "--initial-fail-coords" not in source
    assert "force_fail" not in source
    assert "common random numbers" in source
