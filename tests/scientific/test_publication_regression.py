"""Opt-in higher-sample regressions for publication-facing validation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from pbm.post.percolation_threshold import estimate_percolation_threshold
from pbm.post.risk_equivalence import (
    estimate_analytical_risk_equivalence,
    estimate_numerical_risk_equivalence,
)
from pbm.post.robustness import bernoulli_kl_robustness

from ._helpers import (
    PaperLattice,
    assert_multinomial_marginals_in_acceptance_region,
    estimator_shape_cases,
    hoeffding_radius,
    node_degree,
    simultaneous_binomial_acceptance_interval,
    simulate_conditional_single_seed_scan,
    simulate_histogram,
    simulate_uniform_scan,
)


pytestmark = pytest.mark.publication


def test_publication_scale_probability_invariants(
    paper_lattice: PaperLattice,
) -> None:
    """Higher-sample repeat of Equations 8 and 11-14 plausibility checks."""

    samples = 25_000
    pf_node = 0.005
    propagation_probabilities = np.asarray([0.0, 0.2, 0.5, 1.0])
    histograms = simulate_uniform_scan(
        paper_lattice.model,
        propagation_probabilities,
        pf_node=pf_node,
        samples=samples,
        seed=9876,
    )

    expected_pmf = binom.pmf(np.arange(65), 64, pf_node)
    assert_multinomial_marginals_in_acceptance_region(
        histograms[0],
        expected_pmf,
        family_alpha=0.01,
    )

    count_any = histograms[:, 1:].sum(axis=1)
    expected_any = 1.0 - (1.0 - pf_node) ** 64
    lower, upper = simultaneous_binomial_acceptance_interval(
        samples=samples,
        probability=expected_any,
        family_size=1,
        family_alpha=0.01,
    )
    assert np.all(count_any == count_any[0])
    assert lower <= int(count_any[0]) <= upper

    p_one = histograms[:, 1] / samples
    p_ge_two = histograms[:, 2:].sum(axis=1) / samples
    p_full = histograms[:, -1] / samples
    pair_tolerance = 2.0 * hoeffding_radius(
        samples=samples,
        family_size=9,
        family_alpha=0.01,
    )
    assert np.all(np.diff(p_one) <= pair_tolerance)
    assert np.all(np.diff(p_ge_two) >= -pair_tolerance)
    assert np.all(np.diff(p_full) >= -pair_tolerance)
    assert int(histograms[-1, -1]) == int(count_any[-1])
    assert int(histograms[-1, 1:-1].sum()) == 0


@pytest.mark.parametrize(
    "case",
    estimator_shape_cases(),
    ids=lambda case: case.name,
)
def test_higher_sample_synthetic_estimator_shapes(case) -> None:
    """Higher-sample synthetic binomial counts exercise Equation 15."""

    grid = np.linspace(0.0, 1.0, 21)
    trials = 10_000_000
    rng = np.random.default_rng(case.publication_seed)
    successes = rng.binomial(trials, case.response(grid))
    result = estimate_percolation_threshold(
        grid,
        successes / trials,
        trials=trials,
        successes=successes,
        scan_metadata=case.metadata,
        bootstrap_replicates=60,
        minimum_bootstrap_identification=0.75,
    )

    assert result.scan_metadata == case.metadata
    if case.expected_threshold is None:
        assert result.threshold is None
        assert (
            result.reason
            == "no_interior_sign_changing_second_derivative_root"
        )
    else:
        assert result.threshold == pytest.approx(
            case.expected_threshold,
            abs=0.03,
        )
        assert (
            result.fit_diagnostics.bootstrap_identification_rate is not None
        )
        assert result.fit_diagnostics.bootstrap_identification_rate >= 0.75


def test_publication_scale_risk_equivalence(
    paper_lattice: PaperLattice,
) -> None:
    """Higher-sample Equations 16-27 reference-range regression."""

    propagation_probabilities = np.asarray([0.0, 0.08, 0.09, 0.10, 0.11])
    histograms = simulate_conditional_single_seed_scan(
        paper_lattice.model,
        propagation_probabilities,
        samples=30_000,
        seed=314159,
    )
    numerical = estimate_numerical_risk_equivalence(
        propagation_probabilities,
        histograms,
        pe_crit=0.11,
    )
    analytical = estimate_analytical_risk_equivalence(
        4,
        4,
        4,
        pe_crit=0.65,
    )

    assert numerical.identified
    assert numerical.estimate is not None
    assert 0.089 <= numerical.estimate <= 0.097
    assert numerical.root_bracket == (0.09, 0.10)
    assert analytical.estimate is not None
    assert 0.092 <= analytical.estimate <= 0.096
    assert abs(analytical.estimate - numerical.estimate) <= 0.01


def test_publication_scale_notional_damage_ordering(
    paper_lattice: PaperLattice,
) -> None:
    """Higher-sample Section 3.2 consequence and robustness ordering."""

    samples = 25_000
    pe = 0.2
    scenarios = (
        ("corner", (0, 0, 0)),
        ("perimeter", (1, 0, 0)),
        ("internal", (1, 1, 0)),
    )
    probabilities: dict[str, float] = {}
    robustness: dict[str, float] = {}
    for name, coordinate in scenarios:
        counts = simulate_histogram(
            paper_lattice.model,
            axis_probabilities=pe,
            pf_node=0.0,
            samples=samples,
            seed=271828,
            damage_coordinates=(coordinate,),
        )
        adverse_count = int(counts[2:].sum())
        exact_probability = (
            1.0
            - (1.0 - pe) ** node_degree(paper_lattice.model, coordinate)
        )
        lower, upper = simultaneous_binomial_acceptance_interval(
            samples=samples,
            probability=exact_probability,
            family_size=len(scenarios),
            family_alpha=0.01,
        )
        assert lower <= adverse_count <= upper
        probabilities[name] = adverse_count / samples
        result = bernoulli_kl_robustness(0.0, probabilities[name])
        assert result.value is not None
        robustness[name] = result.value

    assert probabilities["corner"] < probabilities["perimeter"]
    assert probabilities["perimeter"] < probabilities["internal"]
    assert robustness["corner"] > robustness["perimeter"]
    assert robustness["perimeter"] > robustness["internal"]
