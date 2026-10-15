"""Analysis regressions against retained historical manuscript-run counts.

These checks do not rerun the full Monte Carlo publication campaigns.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pbm.post.percolation_threshold import (
    axis_scan_metadata_from_mapping,
    estimate_percolation_threshold,
)
from pbm.post.risk_equivalence import analyze_risk_equivalence
from pbm.post.robustness import bernoulli_kl_robustness

pytestmark = pytest.mark.scientific
REFERENCE = json.loads(
    (Path(__file__).parent / "data/manuscript_reference.json").read_text(encoding="utf8")
)


@pytest.mark.parametrize("case", REFERENCE["figure3"], ids=lambda c: f"figure3_case_{c['case']}")
def test_retained_figure3_counts_recover_manuscript_threshold(case) -> None:
    counts = np.asarray(case["count_full"], dtype=np.int64)
    trials = case["samples"]
    result = estimate_percolation_threshold(
        case["grid"],
        counts / trials,
        trials=trials,
        successes=counts,
        scan_metadata=axis_scan_metadata_from_mapping(case["metadata"]),
    )
    assert result.identified, result.reason
    # A 0.015 allowance covers the paper's 2-decimal reporting and
    # smoothing/interpolation; exact historical float outputs are not targets.
    assert result.threshold == pytest.approx(case["manuscript_threshold"], abs=0.015)
    assert result.slope > 0
    assert result.smoothing_sensitivity.stable
    assert result.scan_spacing_sensitivity.stable


def test_retained_coupled_pmf_matches_manuscript_risk_equivalence() -> None:
    case = REFERENCE["figure3"][0]
    counts = np.asarray(case["pmf_counts"], dtype=np.int64)
    assert counts.shape == (len(case["grid"]), 65)
    assert np.all(counts.sum(axis=1) == case["samples"])
    assert np.array_equal(counts[:, -1], case["count_full"])
    result = analyze_risk_equivalence(
        case["grid"], counts, pe_crit=0.67, lattice_shape=(4, 4, 4)
    )
    assert result.numerical.identified, result.numerical.reason
    assert result.numerical.estimate == pytest.approx(0.093, abs=0.002)
    assert result.analytical.estimate == pytest.approx(0.094, abs=0.0005)


@pytest.mark.parametrize("case", REFERENCE["damage"], ids=lambda c: f"pe_{c['pe']}")
def test_retained_damage_pmfs_match_manuscript_robustness_ordering(case) -> None:
    runs = case["runs"]
    assert [r["scenario"] for r in runs] == [0, 1, 2, 3]
    assert not runs[0]["damage_node_ids"]
    assert all(len(r["damage_node_ids"]) == 1 for r in runs[1:])
    for run in runs:
        assert len(run["pmf_counts"]) == 65
        assert sum(run["pmf_counts"]) == run["samples"] == 1_000_000
    for s_lim in ((2,) if case["pe"] < 0.1 else (2, 16)):
        probabilities = [sum(r["pmf_counts"][s_lim:]) / r["samples"] for r in runs]
        robustness = [bernoulli_kl_robustness(probabilities[0], pj).value for pj in probabilities[1:]]
        assert probabilities[0] <= probabilities[1] < probabilities[2] < probabilities[3]
        assert robustness[0] > robustness[1] > robustness[2]
