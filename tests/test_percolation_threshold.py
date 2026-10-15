from __future__ import annotations

import numpy as np
import pytest

from pbm.post.percolation_threshold import (
    estimate_percolation_threshold,
)


def _counts(probability: np.ndarray, trials: int) -> tuple[np.ndarray, np.ndarray]:
    successes = np.rint(np.asarray(probability) * trials).astype(np.int64)
    return successes, successes / float(trials)


def test_known_sigmoid_has_interior_second_derivative_root():
    grid = np.linspace(0.0, 1.0, 21)
    expected = 0.55
    probability = 0.01 + 0.80 / (1.0 + np.exp(-12.0 * (grid - expected)))
    successes, probability = _counts(probability, 2_000_000)

    result = estimate_percolation_threshold(
        grid,
        probability,
        trials=2_000_000,
        successes=successes,
        bootstrap_replicates=40,
    )

    assert result.identified
    assert result.threshold == pytest.approx(expected, abs=0.02)
    assert result.root_bracket[0] <= result.threshold <= result.root_bracket[1]
    assert result.slope > 0.0
    assert result.boundary_status == "interior"
    assert result.uncertainty_interval is not None
    assert result.smoothing_sensitivity.stable
    assert result.scan_spacing_sensitivity.stable
    assert result.fit_diagnostics.bootstrap_identification_rate >= 0.8


@pytest.mark.parametrize(
    "probability",
    [
        lambda x: 0.05 + 0.80 * x,
        lambda x: 0.05 + 0.80 * x**2,
        lambda x: 0.05 + 0.80 * (2.0 * x - x**2),
    ],
    ids=["linear", "convex", "concave"],
)
def test_non_inflected_curves_do_not_get_fabricated_threshold(probability):
    grid = np.linspace(0.0, 1.0, 21)
    successes, observed = _counts(probability(grid), 200_000)

    result = estimate_percolation_threshold(
        grid,
        observed,
        trials=200_000,
        successes=successes,
        bootstrap_replicates=0,
    )

    assert result.threshold is None
    assert result.root_bracket is None
    assert result.slope is None
    assert result.reason == "no_interior_sign_changing_second_derivative_root"


def test_noisy_binomial_sigmoid_remains_identifiable():
    grid = np.linspace(0.0, 1.0, 21)
    trials = 50_000
    expected = 0.55
    probability = 0.01 + 0.80 / (1.0 + np.exp(-10.0 * (grid - expected)))
    successes = np.random.default_rng(42).binomial(trials, probability)

    result = estimate_percolation_threshold(
        grid,
        successes / trials,
        trials=trials,
        successes=successes,
        bootstrap_replicates=40,
    )

    assert result.identified
    assert result.threshold == pytest.approx(expected, abs=0.04)
    assert result.fit_diagnostics.monotone_within_mc_uncertainty


def test_boundary_only_inflection_is_not_an_interior_threshold():
    grid = np.linspace(0.0, 1.0, 41)
    probability = 0.01 + 0.80 / (1.0 + np.exp(-15.0 * (grid - 0.04)))
    successes, observed = _counts(probability, 2_000_000)

    result = estimate_percolation_threshold(
        grid,
        observed,
        trials=2_000_000,
        successes=successes,
        boundary_margin=0.08,
        bootstrap_replicates=0,
    )

    assert result.threshold is None
    assert result.boundary_status == "boundary_only"
    assert result.fit_diagnostics.boundary_roots
    assert result.reason == "second_derivative_root_only_at_scan_boundary"


@pytest.mark.parametrize(
    ("grid", "probability", "match"),
    [
        ([0.0, 0.5, 0.5, 1.0], [0.0, 0.2, 0.2, 0.8], "duplicate"),
        ([0.0, 0.7, 0.4, 1.0], [0.0, 0.5, 0.3, 0.8], "strictly increasing"),
        ([0.0, 0.5, 1.0], [0.0, 1.1, 1.0], r"in \[0, 1\]"),
    ],
)
def test_scan_validation_rejects_invalid_grid_or_response(
    grid, probability, match
):
    with pytest.raises(ValueError, match=match):
        estimate_percolation_threshold(
            grid,
            probability,
            trials=1_000,
            bootstrap_replicates=0,
        )


def test_monotonicity_violation_beyond_mc_uncertainty_is_diagnostic():
    grid = np.linspace(0.0, 1.0, 9)
    probability = np.array([0.0, 0.1, 0.2, 0.5, 0.3, 0.6, 0.7, 0.8, 0.9])
    successes, observed = _counts(probability, 100_000)

    result = estimate_percolation_threshold(
        grid,
        observed,
        trials=100_000,
        successes=successes,
        bootstrap_replicates=0,
    )

    assert result.threshold is None
    assert not result.fit_diagnostics.monotone_within_mc_uncertainty
    assert result.reason == "response_not_monotone_within_mc_uncertainty"

