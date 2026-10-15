"""Scientific unit tests for the Cao--Beck entropy-based robustness API."""

import json
import math

import numpy as np
import pytest
from scipy.stats import norm

from pbm.post.robustness import (
    RobustnessCoherenceError,
    adverse_probability_from_pmf,
    bernoulli_kl_robustness,
    entropy_robustness_thresholds,
    evaluate_damage_scenario_robustness,
)


@pytest.mark.parametrize("probability", [0.0, 1e-6, 0.25, 1.0])
def test_identity_is_exactly_one_including_boundaries(probability):
    result = bernoulli_kl_robustness(probability, probability)

    assert result.value == 1.0
    assert result.kl_divergence == 0.0
    assert result.status == "identity"


def test_exact_probability_boundary_conventions_are_not_clipped():
    no_baseline_failures = bernoulli_kl_robustness(0.0, 0.25)
    certain_damaged_failure = bernoulli_kl_robustness(0.2, 1.0)

    assert no_baseline_failures.value == pytest.approx(0.75)
    assert certain_damaged_failure.value == 0.0
    assert math.isinf(certain_damaged_failure.kl_divergence)
    assert certain_damaged_failure.to_dict()["kl_divergence"] is None
    json.dumps(certain_damaged_failure.to_dict(), allow_nan=False)


def test_robustness_decreases_monotonically_as_coherent_damage_increases():
    p0 = 0.01
    values = [
        bernoulli_kl_robustness(p0, pj).value
        for pj in (0.02, 0.05, 0.20, 0.80, 1.0)
    ]

    assert all(value is not None for value in values)
    assert np.all(np.diff(values) < 0.0)


@pytest.mark.parametrize(
    ("p0", "pj"),
    [(-0.1, 0.2), (0.1, 1.1), (math.nan, 0.2), (0.1, math.inf)],
)
def test_invalid_probabilities_are_rejected(p0, pj):
    with pytest.raises(ValueError, match="probability"):
        bernoulli_kl_robustness(p0, pj)


@pytest.mark.parametrize("c", [0.0, -1.0, math.inf])
def test_invalid_scale_is_rejected(c):
    with pytest.raises(ValueError, match="positive"):
        bernoulli_kl_robustness(0.01, 0.02, c=c)


def test_strict_and_diagnostic_coherence_modes_do_not_mask_violation():
    with pytest.raises(RobustnessCoherenceError, match="below"):
        bernoulli_kl_robustness(
            0.02,
            0.01,
            coherence_mode="strict",
        )

    diagnostic = bernoulli_kl_robustness(
        0.02,
        0.01,
        coherence_mode="diagnostic",
    )
    assert diagnostic.value is None
    assert diagnostic.status == "incoherent"
    assert not diagnostic.accepted
    assert diagnostic.reason is not None


def test_monte_carlo_tolerance_is_explicit_and_does_not_return_silent_one():
    result = bernoulli_kl_robustness(
        0.020,
        0.019,
        coherence_mode="strict",
        mc_tolerance=0.002,
    )

    assert result.accepted
    assert not result.coherent
    assert result.status == "within_monte_carlo_tolerance"
    assert result.value is not None
    assert result.value < 1.0
    assert result.reason is not None


def test_adverse_macrostate_is_exact_exceedance_without_clipping():
    support = np.arange(5)
    probabilities = np.array([0.5, 0.2, 0.15, 0.1, 0.05])

    assert adverse_probability_from_pmf(
        support,
        probabilities,
        s_lim=2,
    ) == pytest.approx(0.30)
    assert adverse_probability_from_pmf(
        support,
        probabilities,
        s_lim=4,
    ) == pytest.approx(0.05)


def test_damage_evaluation_requires_named_scenario_and_common_macrostate():
    support = np.arange(4)
    baseline = np.array([0.8, 0.15, 0.04, 0.01])
    damaged = np.array([0.5, 0.2, 0.2, 0.1])

    result = evaluate_damage_scenario_robustness(
        support,
        baseline,
        support,
        damaged,
        s_lim=2,
        damaged_scenario="corner node 7",
    )
    assert result.adverse_macrostate == "N_f >= 2"
    assert result.p0 == pytest.approx(0.05)
    assert result.pj == pytest.approx(0.30)
    assert result.value is not None

    with pytest.raises(ValueError, match="explicitly specified"):
        evaluate_damage_scenario_robustness(
            support,
            baseline,
            support,
            damaged,
            s_lim=2,
            damaged_scenario="",
        )


def test_published_beta_3p7_reference_thresholds():
    # Cao--Beck (2026), Fig. 1: beta=3.7 gives I_-~=0.50 and I_+~=0.84,
    # with pi_-~=0 and pi_+~=1.
    result = entropy_robustness_thresholds(
        float(norm.sf(3.7)),
        pi_min=-3.0,
        pi_max=3.0,
        n_grid=2001,
    )

    assert result.identified
    assert result.pi_minus == pytest.approx(0.0, abs=5e-4)
    assert result.pi_plus == pytest.approx(1.0, abs=5e-4)
    assert result.I_minus == pytest.approx(0.50, abs=0.01)
    assert result.I_plus == pytest.approx(0.84, abs=0.01)


def test_derivative_root_absence_returns_diagnostics_without_fallback():
    result = entropy_robustness_thresholds(
        float(norm.sf(3.7)),
        pi_min=1.5,
        pi_max=2.0,
        n_grid=501,
    )

    assert not result.identified
    assert result.pi_minus is None
    assert result.pi_plus is None
    assert result.second_derivative.reason == (
        "no_sign_changing_second_derivative_root"
    )
    assert result.third_derivative.reason == "lower_threshold_not_identified"
    assert result.reason is not None


def test_boundary_threshold_diagnostics_are_strict_json():
    result = entropy_robustness_thresholds(0.0)
    payload = result.to_dict(include_grid=True)

    assert payload["beta"] is None
    assert payload["beta_status"] == "positive_infinity"
    json.dumps(payload, allow_nan=False)
