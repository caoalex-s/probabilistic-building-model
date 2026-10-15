"""Fast manuscript-level regressions for the stationary paper."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

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
    run_cli_histogram,
    simultaneous_binomial_acceptance_interval,
    simulate_conditional_single_seed_scan,
    simulate_histogram,
    simulate_uniform_scan,
)


pytestmark = pytest.mark.scientific


def test_4x4x4_topology_matches_manuscript_lattice(
    paper_lattice: PaperLattice,
) -> None:
    """Figure 1 and Equations 25-27: finite-lattice topology and degrees."""

    model = paper_lattice.model
    degrees = {node.id: 0 for node in model.nodes}
    for edge in model.edges:
        degrees[edge.org] += 1
        degrees[edge.dst] += 1

    assert len(model.nodes) == 64
    assert len(model.edges) == 144
    assert Counter(degrees.values()) == Counter({3: 8, 4: 24, 5: 24, 6: 8})


def test_pe_zero_pmf_is_binomial_with_simultaneous_exact_intervals(
    paper_lattice: PaperLattice,
) -> None:
    """Equations 8 and 11; Section 2.8 plausibility condition (i)."""

    samples = 12_000
    pf_node = 0.05
    counts = simulate_histogram(
        paper_lattice.model,
        axis_probabilities=0.0,
        pf_node=pf_node,
        samples=samples,
        seed=20260730,
    )
    expected = binom.pmf(np.arange(65), 64, pf_node)

    assert counts.sum() == samples
    assert_multinomial_marginals_in_acceptance_region(
        counts,
        expected,
        family_alpha=0.01,
    )


def test_any_failure_invariance_redistribution_and_connected_limit(
    paper_lattice: PaperLattice,
) -> None:
    """Equations 11-14 and all four Section 2.8 plausibility conditions."""

    propagation_probabilities = np.asarray([0.0, 0.2, 0.5, 1.0])
    samples = 5_000
    pf_node = 0.005
    histograms = simulate_uniform_scan(
        paper_lattice.model,
        propagation_probabilities,
        pf_node=pf_node,
        samples=samples,
        seed=9876,
    )

    count_any = histograms[:, 1:].sum(axis=1)
    expected_any = 1.0 - (1.0 - pf_node) ** 64
    lower, upper = simultaneous_binomial_acceptance_interval(
        samples=samples,
        probability=expected_any,
        family_size=1,
        family_alpha=0.01,
    )

    # Common random numbers make invariance exact, stronger than an
    # independent-sample comparison within confidence intervals.
    assert np.all(count_any == count_any[0])
    assert lower <= int(count_any[0]) <= upper

    p_one = histograms[:, 1] / samples
    p_ge_two = histograms[:, 2:].sum(axis=1) / samples
    p_full = histograms[:, -1] / samples
    comparisons = 3 * (len(propagation_probabilities) - 1)
    pair_tolerance = 2.0 * hoeffding_radius(
        samples=samples,
        family_size=comparisons,
        family_alpha=0.01,
    )
    assert np.all(np.diff(p_one) <= pair_tolerance)
    assert np.all(np.diff(p_ge_two) >= -pair_tolerance)
    assert np.all(np.diff(p_full) >= -pair_tolerance)

    # On the connected pe=1 lattice every non-empty initial state reaches all
    # nodes, so the equality holds realization by realization.
    pe_one = histograms[-1]
    assert int(pe_one[-1]) == int(count_any[-1])
    assert int(pe_one[1:-1].sum()) == 0
    assert int(pe_one[0] + pe_one[-1]) == samples


@pytest.mark.parametrize(
    "case",
    estimator_shape_cases(),
    ids=lambda case: case.name,
)
def test_synthetic_estimator_shapes_and_axis_metadata(case) -> None:
    """Equation 15 estimator behavior on generic response shapes."""

    grid = np.linspace(0.0, 1.0, 21)
    trials = 5_000_000
    successes = np.rint(case.response(grid) * trials).astype(np.int64)
    result = estimate_percolation_threshold(
        grid,
        successes / trials,
        trials=trials,
        successes=successes,
        scan_metadata=case.metadata,
        bootstrap_replicates=24,
        minimum_bootstrap_identification=0.75,
    )

    assert result.scan_metadata == case.metadata
    if case.expected_threshold is None:
        assert result.threshold is None
        assert result.root_bracket is None
        assert (
            result.reason
            == "no_interior_sign_changing_second_derivative_root"
        )
    else:
        assert result.threshold == pytest.approx(
            case.expected_threshold,
            abs=0.03,
        )
        assert result.boundary_status == "interior"


def test_numerical_and_analytical_risk_equivalence_reference_ranges(
    paper_lattice: PaperLattice,
) -> None:
    """Equations 16-27: conditional MC and finite-lattice approximation."""

    propagation_probabilities = np.asarray([0.0, 0.08, 0.09, 0.10, 0.11])
    histograms = simulate_conditional_single_seed_scan(
        paper_lattice.model,
        propagation_probabilities,
        samples=10_000,
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
    assert 0.087 <= numerical.estimate <= 0.099
    assert numerical.root_bracket == (0.09, 0.10)
    assert analytical.identified
    assert analytical.estimate is not None
    assert 0.092 <= analytical.estimate <= 0.096
    assert abs(analytical.estimate - numerical.estimate) <= 0.01
    assert analytical.conservatism == "case-dependent"


def test_notional_damage_consequence_and_robustness_ordering(
    paper_lattice: PaperLattice,
) -> None:
    """Section 3.2 and Equation 28: corner, perimeter, internal ordering."""

    samples = 5_000
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
        degree = node_degree(paper_lattice.model, coordinate)
        exact_probability = 1.0 - (1.0 - pe) ** degree
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


def test_seeded_histogram_is_exact_across_sequential_and_parallel_cli(
    paper_lattice: PaperLattice,
    tmp_path: Path,
) -> None:
    """Algorithm 1 and the RNG contract: execution mode changes no sample."""

    repo_root = Path(__file__).resolve().parents[2]
    sequential = run_cli_histogram(
        repo_root=repo_root,
        input_db=paper_lattice.db_path,
        output_db=tmp_path / "sequential.db",
        workers=1,
        chunk_size=17,
    )
    parallel = run_cli_histogram(
        repo_root=repo_root,
        input_db=paper_lattice.db_path,
        output_db=tmp_path / "parallel.db",
        workers=2,
        chunk_size=41,
    )

    assert sequential == parallel
    assert sum(sequential) == 401
    assert sum(count > 0 for count in sequential) > 1
