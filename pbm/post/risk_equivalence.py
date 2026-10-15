r"""Numerical and analytical risk-equivalence calculations.

The stationary paper's common-consequence criterion is

.. math::

    P(N_f=1) = \sum_{n=2}^{N}(n-1)P(N_f=n).

The numerical implementation evaluates both sides directly from the integer
PMF retained at every fully coupled percolation-scan point.  It identifies only
an observed sign-changing crossing within ``[0, pe_crit]`` and uses linear
interpolation inside the crossing bracket.  A closest-grid-point fallback is
deliberately not used.

The analytical implementation follows the paper's locally tree-like
approximation for a regular rectangular lattice.  Its upper bound is
``pe_max = min(pe_crit, 1 / kappa)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


_COMMON_ASSUMPTIONS = (
    "The direct and indirect consequence of each failed compartment is common "
    "and additive.",
    "The initial compartment consequence is excluded from the progressive-"
    "collapse side of the equality.",
    "The scanned propagation probability is homogeneous and fully coupled "
    "across x, y, and z.",
)

_ANALYTICAL_ASSUMPTIONS = (
    "The lattice is regular with dimensions nx by ny by nz.",
    "The spontaneous compartment-failure probability is small enough that one "
    "initial failed compartment dominates.",
    "The initial compartment is uniformly distributed over all lattice nodes.",
    "Early propagation is approximated as locally tree-like below the "
    "percolation threshold.",
)

_CASE_DEPENDENT_CONSERVATISM_REASON = (
    "The locally tree-like approximation does not establish a universal "
    "conservative or non-conservative bias, but may be nonconservative for "
    "regular lattices. Its estimate should be compared with the numerical "
    "result for the topology under consideration."
)


@dataclass(frozen=True)
class RiskEquivalenceCurve:
    """PMF-derived numerical curve and simultaneous Monte Carlo diagnostics."""

    propagation_probability: tuple[float, ...]
    single_failure_probability: tuple[float, ...]
    additional_failure_consequence: tuple[float, ...]
    residual: tuple[float, ...]
    residual_standard_error: tuple[float, ...]
    residual_confidence_interval: tuple[tuple[float, float], ...]
    n_samples: tuple[int, ...]
    effective_failure_samples: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NumericalRiskEquivalenceResult:
    """Structured result of the bounded scan-PMF calculation."""

    estimate: Optional[float]
    identified: bool
    status: str
    reason: Optional[str]
    bounds: tuple[float, float]
    root_bracket: Optional[tuple[float, float]]
    bracket_residuals: Optional[tuple[float, float]]
    crossing_brackets: tuple[tuple[float, float], ...]
    residual_at_estimate: Optional[float]
    interpolation: str
    boundary_status: str
    closest_grid_point: Optional[float]
    closest_grid_residual: Optional[float]
    grid_spacing_min: Optional[float]
    grid_spacing_median: Optional[float]
    grid_spacing_max: Optional[float]
    resolution: Optional[float]
    resolution_status: str
    monte_carlo_interval: Optional[tuple[float, float]]
    monte_carlo_crossing_resolved: Optional[bool]
    monte_carlo_ambiguous_brackets: tuple[tuple[float, float], ...]
    confidence_level: float
    uncertainty_method: str
    curve: RiskEquivalenceCurve
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegularLatticeDegreeDistribution:
    """Finite regular-lattice degree moments used by the approximation."""

    nx: int
    ny: int
    nz: int
    node_count: int
    degree_counts: tuple[tuple[int, int], ...]
    mean_degree: float
    second_factorial_moment: float
    mean_excess_degree: float
    inverse_mean_excess_degree: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyticalRiskEquivalenceResult:
    """Structured result of the paper's regular-lattice approximation."""

    estimate: Optional[float]
    identified: bool
    status: str
    reason: Optional[str]
    bounds: tuple[float, float]
    root_bracket: Optional[tuple[float, float]]
    residual_at_estimate: Optional[float]
    expected_no_propagation_at_estimate: Optional[float]
    pe_crit: float
    pe_max: float
    degree_distribution: RegularLatticeDegreeDistribution
    equation: str
    assumptions: tuple[str, ...]
    conservatism: str
    conservatism_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskEquivalenceResult:
    """Combined numerical/analytical result for machine-readable output."""

    numerical: NumericalRiskEquivalenceResult
    analytical: AnalyticalRiskEquivalenceResult
    relative_difference: Optional[float]
    absolute_relative_difference: Optional[float]
    difference_definition: str
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


def _axis_degree_counts(length: int) -> dict[int, int]:
    if length == 1:
        return {0: 1}
    if length == 2:
        return {1: 2}
    return {1: 2, 2: length - 2}


def regular_lattice_degree_counts(
    nx: int,
    ny: int,
    nz: int,
) -> dict[int, int]:
    """Return exact node counts by degree for an ``nx × ny × nz`` lattice.

    For three non-degenerate dimensions this reproduces the paper's
    ``alpha_3`` through ``alpha_6`` formulas.  The convolution also handles a
    dimension of length one or two without producing negative counts.
    """

    dimensions = (
        _positive_integer(nx, name="nx"),
        _positive_integer(ny, name="ny"),
        _positive_integer(nz, name="nz"),
    )
    counts: dict[int, int] = {0: 1}
    for length in dimensions:
        updated: dict[int, int] = {}
        for accumulated_degree, accumulated_count in counts.items():
            for axis_degree, axis_count in _axis_degree_counts(length).items():
                degree = accumulated_degree + axis_degree
                updated[degree] = (
                    updated.get(degree, 0)
                    + accumulated_count * axis_count
                )
        counts = updated
    return dict(sorted(counts.items()))


def regular_lattice_degree_distribution(
    nx: int,
    ny: int,
    nz: int,
) -> RegularLatticeDegreeDistribution:
    """Calculate finite-lattice degree counts and the first two moments."""

    nx_i = _positive_integer(nx, name="nx")
    ny_i = _positive_integer(ny, name="ny")
    nz_i = _positive_integer(nz, name="nz")
    counts = regular_lattice_degree_counts(nx_i, ny_i, nz_i)
    node_count = nx_i * ny_i * nz_i
    if sum(counts.values()) != node_count:  # pragma: no cover - invariant
        raise RuntimeError("Regular-lattice degree counts do not sum to |N|.")

    mean_degree = (
        sum(degree * count for degree, count in counts.items()) / node_count
    )
    second_factorial = (
        sum(
            degree * (degree - 1) * count
            for degree, count in counts.items()
        )
        / node_count
    )
    kappa = (
        second_factorial / mean_degree if mean_degree > 0.0 else 0.0
    )
    inverse = 1.0 / kappa if kappa > 0.0 else None
    return RegularLatticeDegreeDistribution(
        nx=nx_i,
        ny=ny_i,
        nz=nz_i,
        node_count=node_count,
        degree_counts=tuple((int(k), int(v)) for k, v in counts.items()),
        mean_degree=float(mean_degree),
        second_factorial_moment=float(second_factorial),
        mean_excess_degree=float(kappa),
        inverse_mean_excess_degree=(
            float(inverse) if inverse is not None else None
        ),
    )


def expected_no_propagation(
    pe: float,
    degree_counts: Mapping[int, int] | Sequence[tuple[int, int]],
) -> float:
    """Evaluate ``E[(1-pe)^Delta]`` for a finite degree distribution."""

    probability = float(pe)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("pe must be a finite probability in [0, 1].")
    items = (
        degree_counts.items()
        if isinstance(degree_counts, Mapping)
        else degree_counts
    )
    parsed: list[tuple[int, int]] = []
    for degree, count in items:
        degree_i = _nonnegative_integer(degree, name="degree")
        count_i = _nonnegative_integer(count, name="degree count")
        parsed.append((degree_i, count_i))
    total = sum(count for _, count in parsed)
    if total <= 0:
        raise ValueError("The degree distribution must contain at least one node.")
    return float(
        sum(
            count * (1.0 - probability) ** degree
            for degree, count in parsed
        )
        / total
    )


def _validate_pe_crit(pe_crit: float) -> float:
    value = float(pe_crit)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("pe_crit must be a finite probability in (0, 1].")
    return value


def _analytical_residual(
    pe: float,
    distribution: RegularLatticeDegreeDistribution,
) -> float:
    """Residual of the corrected analytical risk-equivalence equation.

    The initiating compartment is selected uniformly from all lattice nodes.
    Its expected first generation is therefore E[Delta] * pe, while subsequent
    generations grow with mean factor kappa * pe.
    """

    probability = float(pe)
    kappa = distribution.mean_excess_degree

    denominator = 1.0 - kappa * probability
    if denominator <= 0.0:
        return float("inf")

    expected_propagated_extent = (
        distribution.mean_degree * probability / denominator
    )

    expected_no_propagation_probability = expected_no_propagation(
        probability,
        distribution.degree_counts,
    )

    return float(
        expected_propagated_extent
        - expected_no_propagation_probability
    )


def _sign_change_brackets(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int]]:
    brackets: list[tuple[int, int]] = []
    exact_roots = [int(i) for i in np.flatnonzero(y == 0.0)]
    exact_set = set(exact_roots)
    for i in range(x.size - 1):
        if i in exact_set or (i + 1) in exact_set:
            continue
        if y[i] * y[i + 1] < 0.0:
            brackets.append((i, i + 1))
    return brackets, exact_roots


def estimate_analytical_risk_equivalence(
    nx: int,
    ny: int,
    nz: int,
    *,
    pe_crit: float,
    root_grid_size: int = 4097,
) -> AnalyticalRiskEquivalenceResult:
    """Solve the paper's regular-lattice approximation on its valid bound."""

    critical = _validate_pe_crit(pe_crit)
    if (
        isinstance(root_grid_size, (bool, np.bool_))
        or not isinstance(root_grid_size, (int, np.integer))
        or root_grid_size < 33
    ):
        raise ValueError("root_grid_size must be an integer of at least 33.")
    distribution = regular_lattice_degree_distribution(nx, ny, nz)
    inverse = distribution.inverse_mean_excess_degree
    pe_max = min(critical, inverse) if inverse is not None else critical

    upper_evaluation = float(pe_max)
    if inverse is not None and np.isclose(
        pe_max, inverse, rtol=0.0, atol=8.0 * np.finfo(float).eps
    ):
        upper_evaluation = float(np.nextafter(pe_max, 0.0))

    grid = np.linspace(0.0, upper_evaluation, int(root_grid_size))
    residual = np.asarray(
        [_analytical_residual(value, distribution) for value in grid],
        dtype=float,
    )
    brackets, exact_roots = _sign_change_brackets(grid, residual)

    roots: list[tuple[float, tuple[float, float]]] = []
    for index in exact_roots:
        roots.append(
            (float(grid[index]), (float(grid[index]), float(grid[index])))
        )
    for left, right in brackets:
        root = brentq(
            lambda value: _analytical_residual(value, distribution),
            float(grid[left]),
            float(grid[right]),
        )
        roots.append(
            (float(root), (float(grid[left]), float(grid[right])))
        )
    roots.sort(key=lambda item: item[0])

    if len(roots) == 1:
        estimate, bracket = roots[0]
        expectation = expected_no_propagation(
            estimate, distribution.degree_counts
        )
        return AnalyticalRiskEquivalenceResult(
            estimate=estimate,
            identified=True,
            status="identified",
            reason=None,
            bounds=(0.0, float(pe_max)),
            root_bracket=bracket,
            residual_at_estimate=_analytical_residual(
                estimate, distribution
            ),
            expected_no_propagation_at_estimate=expectation,
            pe_crit=critical,
            pe_max=float(pe_max),
            degree_distribution=distribution,
            equation=(
                "E[Delta]*pe/(1-kappa*pe) - "
                "E[(1-pe)^Delta] = 0"
            ),
            assumptions=_COMMON_ASSUMPTIONS + _ANALYTICAL_ASSUMPTIONS,
            conservatism="case-dependent",
            conservatism_reason=_CASE_DEPENDENT_CONSERVATISM_REASON,
        )

    reason = "no_crossing_within_bounds" if not roots else "multiple_crossings"
    return AnalyticalRiskEquivalenceResult(
        estimate=None,
        identified=False,
        status="not_identified",
        reason=reason,
        bounds=(0.0, float(pe_max)),
        root_bracket=None,
        residual_at_estimate=None,
        expected_no_propagation_at_estimate=None,
        pe_crit=critical,
        pe_max=float(pe_max),
        degree_distribution=distribution,
        equation=(
            "E[Delta]*pe/(1-kappa*pe) - "
            "E[(1-pe)^Delta] = 0"
        ),
        assumptions=_COMMON_ASSUMPTIONS + _ANALYTICAL_ASSUMPTIONS,
        conservatism="case-dependent",
        conservatism_reason=_CASE_DEPENDENT_CONSERVATISM_REASON,
    )


def _validate_numerical_inputs(
    propagation_probability: Sequence[float] | np.ndarray,
    pmf_counts: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pe = np.asarray(propagation_probability, dtype=float)
    raw_counts = np.asarray(pmf_counts)
    if pe.ndim != 1 or pe.size == 0:
        raise ValueError(
            "propagation_probability must be a non-empty one-dimensional array."
        )
    if not np.all(np.isfinite(pe)) or np.any((pe < 0.0) | (pe > 1.0)):
        raise ValueError(
            "propagation_probability must contain finite values in [0, 1]."
        )
    differences = np.diff(pe)
    if np.any(differences == 0.0):
        raise ValueError("The propagation-probability grid contains duplicates.")
    if np.any(differences < 0.0):
        raise ValueError(
            "The propagation-probability grid must be strictly increasing."
        )

    if raw_counts.ndim != 2 or raw_counts.shape[0] != pe.size:
        raise ValueError(
            "pmf_counts must be a two-dimensional array with one row per "
            "propagation probability."
        )
    if raw_counts.shape[1] < 2:
        raise ValueError("pmf_counts must retain at least N_f=0 and N_f=1.")
    try:
        numeric_counts = raw_counts.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("pmf_counts must contain non-negative integers.") from exc
    if (
        not np.all(np.isfinite(numeric_counts))
        or np.any(numeric_counts < 0.0)
        or np.any(numeric_counts != np.floor(numeric_counts))
    ):
        raise ValueError("pmf_counts must contain non-negative integers.")
    counts = numeric_counts.astype(np.int64)
    trials = counts.sum(axis=1)
    if np.any(trials <= 1):
        raise ValueError(
            "Every scan PMF must contain at least two Monte Carlo realizations."
        )
    return pe, counts, trials


def _residual_statistics(
    counts: np.ndarray,
    trials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_values = np.arange(counts.shape[1], dtype=float)
    weights = np.zeros(counts.shape[1], dtype=float)
    weights[1] = 1.0
    if counts.shape[1] > 2:
        weights[2:] = -(n_values[2:] - 1.0)
    probabilities = counts / trials[:, None]
    lhs = probabilities[:, 1]
    rhs = np.sum(
        probabilities[:, 2:] * (n_values[2:] - 1.0), axis=1
    )
    residual = lhs - rhs

    centered = weights[None, :] - residual[:, None]
    sample_variance = np.sum(counts * centered**2, axis=1) / (trials - 1)
    standard_error = np.sqrt(sample_variance / trials)
    return lhs, rhs, residual, standard_error


def _interpolate_at_bound(
    pe: np.ndarray,
    arrays: Sequence[np.ndarray],
    upper: float,
) -> tuple[np.ndarray, list[np.ndarray], bool]:
    inside = pe < upper
    exact = np.flatnonzero(pe == upper)
    if exact.size:
        stop = int(exact[0]) + 1
        return pe[:stop], [values[:stop] for values in arrays], False

    right = int(np.searchsorted(pe, upper, side="right"))
    left = right - 1
    fraction = (upper - pe[left]) / (pe[right] - pe[left])
    bounded_pe = np.concatenate((pe[:right], [upper]))
    bounded_arrays: list[np.ndarray] = []
    for values in arrays:
        interpolated = (
            (1.0 - fraction) * values[left] + fraction * values[right]
        )
        bounded_arrays.append(
            np.concatenate((values[:right], [interpolated]))
        )
    return bounded_pe, bounded_arrays, True


def _linear_root(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    if y0 == 0.0:
        return float(x0)
    if y1 == 0.0:
        return float(x1)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _mc_root_interval(
    x0: float,
    x1: float,
    interval0: tuple[float, float],
    interval1: tuple[float, float],
) -> tuple[Optional[tuple[float, float]], bool]:
    lo0, hi0 = interval0
    lo1, hi1 = interval1
    if lo0 <= 0.0 <= hi0 and lo1 <= 0.0 <= hi1:
        return (float(x0), float(x1)), False

    candidates0 = [lo0, hi0]
    candidates1 = [lo1, hi1]
    if lo0 <= 0.0 <= hi0:
        candidates0.append(0.0)
    if lo1 <= 0.0 <= hi1:
        candidates1.append(0.0)

    roots: list[float] = []
    for value0 in candidates0:
        for value1 in candidates1:
            if value0 == 0.0:
                roots.append(float(x0))
            elif value1 == 0.0:
                roots.append(float(x1))
            elif value0 * value1 < 0.0:
                roots.append(_linear_root(x0, value0, x1, value1))
    if not roots:
        return None, False
    resolved = (
        not (lo0 <= 0.0 <= hi0)
        and not (lo1 <= 0.0 <= hi1)
        and (
            (hi0 < 0.0 and lo1 > 0.0)
            or (lo0 > 0.0 and hi1 < 0.0)
        )
    )
    return (float(min(roots)), float(max(roots))), bool(resolved)


def _ambiguous_mc_brackets(
    pe: np.ndarray,
    confidence_intervals: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    brackets: list[tuple[float, float]] = []
    for i in range(pe.size - 1):
        lo0, hi0 = confidence_intervals[i]
        lo1, hi1 = confidence_intervals[i + 1]
        possible = (
            (lo0 <= 0.0 <= hi0)
            or (lo1 <= 0.0 <= hi1)
            or (lo0 < 0.0 < hi1)
            or (lo1 < 0.0 < hi0)
        )
        if possible:
            brackets.append((float(pe[i]), float(pe[i + 1])))
    return tuple(brackets)


def estimate_numerical_risk_equivalence(
    propagation_probability: Sequence[float] | np.ndarray,
    pmf_counts: Sequence[Sequence[int]] | np.ndarray,
    *,
    pe_crit: float,
    confidence_level: float = 0.95,
) -> NumericalRiskEquivalenceResult:
    """Find the unique PMF crossing within ``[0, pe_crit]``.

    The Monte Carlo diagnostic uses a normal confidence interval for the mean
    of the per-realization common-consequence residual.  A Bonferroni critical
    value makes the pointwise intervals simultaneous over the bounded scan.
    The reported root interval propagates the two endpoint intervals through
    the same local linear interpolation.  This avoids assuming independence
    between scan points, which use common random numbers in this project.
    """

    pe, counts, trials = _validate_numerical_inputs(
        propagation_probability, pmf_counts
    )
    critical = _validate_pe_crit(pe_crit)
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")

    lhs, rhs, residual, standard_error = _residual_statistics(counts, trials)
    effective_samples = trials - counts[:, 0]
    spacing = np.diff(pe)
    spacing_min = float(np.min(spacing)) if spacing.size else None
    spacing_median = float(np.median(spacing)) if spacing.size else None
    spacing_max = float(np.max(spacing)) if spacing.size else None
    assumptions = _COMMON_ASSUMPTIONS + (
        "Each scan PMF is represented by its retained integer Monte Carlo "
        "histogram over N_f.",
        "The root is locally linear within its scan bracket.",
    )
    uncertainty_method = (
        "Bonferroni-simultaneous normal intervals for the sample-mean "
        "common-consequence residual, propagated through local linear "
        "interpolation; no independence between scan points is assumed"
    )

    coverage_tolerance = 16.0 * np.finfo(float).eps
    covers_lower = pe[0] <= coverage_tolerance
    covers_upper = pe[-1] + coverage_tolerance >= critical
    if not covers_lower or not covers_upper:
        alpha = 1.0 - confidence
        z_value = float(norm.ppf(1.0 - alpha / (2.0 * pe.size)))
        intervals = np.column_stack(
            (residual - z_value * standard_error,
             residual + z_value * standard_error)
        )
        curve = RiskEquivalenceCurve(
            propagation_probability=tuple(float(v) for v in pe),
            single_failure_probability=tuple(float(v) for v in lhs),
            additional_failure_consequence=tuple(float(v) for v in rhs),
            residual=tuple(float(v) for v in residual),
            residual_standard_error=tuple(float(v) for v in standard_error),
            residual_confidence_interval=tuple(
                (float(lo), float(hi)) for lo, hi in intervals
            ),
            n_samples=tuple(int(v) for v in trials),
            effective_failure_samples=tuple(
                int(v) for v in effective_samples
            ),
        )
        closest = int(np.argmin(np.abs(residual)))
        return NumericalRiskEquivalenceResult(
            estimate=None,
            identified=False,
            status="not_identified",
            reason="scan_does_not_cover_bounds",
            bounds=(0.0, critical),
            root_bracket=None,
            bracket_residuals=None,
            crossing_brackets=(),
            residual_at_estimate=None,
            interpolation="linear",
            boundary_status="not_evaluated",
            closest_grid_point=float(pe[closest]),
            closest_grid_residual=float(residual[closest]),
            grid_spacing_min=spacing_min,
            grid_spacing_median=spacing_median,
            grid_spacing_max=spacing_max,
            resolution=None,
            resolution_status="not_evaluated",
            monte_carlo_interval=None,
            monte_carlo_crossing_resolved=None,
            monte_carlo_ambiguous_brackets=_ambiguous_mc_brackets(
                pe, intervals
            ),
            confidence_level=confidence,
            uncertainty_method=uncertainty_method,
            curve=curve,
            assumptions=assumptions,
            warnings=(
                "The scan must include p_e=0 and reach pe_crit before a "
                "bounded crossing can be assessed.",
            ),
        )

    bounded_pe, bounded_values, bound_interpolated = _interpolate_at_bound(
        pe,
        (
            lhs,
            rhs,
            residual,
            standard_error,
            trials.astype(float),
            effective_samples.astype(float),
        ),
        critical,
    )
    (
        bounded_lhs,
        bounded_rhs,
        bounded_residual,
        bounded_se,
        bounded_trials_f,
        bounded_effective_f,
    ) = bounded_values
    # A convex sum of endpoint standard errors is conservative for any unknown
    # covariance induced by common random numbers.
    if bound_interpolated:
        right = int(np.searchsorted(pe, critical, side="right"))
        left = right - 1
        fraction = (critical - pe[left]) / (pe[right] - pe[left])
        bounded_se[-1] = (
            (1.0 - fraction) * standard_error[left]
            + fraction * standard_error[right]
        )
        bounded_trials_f[-1] = min(trials[left], trials[right])
        bounded_effective_f[-1] = min(
            effective_samples[left], effective_samples[right]
        )

    alpha = 1.0 - confidence
    z_value = float(
        norm.ppf(1.0 - alpha / (2.0 * bounded_pe.size))
    )
    confidence_intervals = np.column_stack(
        (
            bounded_residual - z_value * bounded_se,
            bounded_residual + z_value * bounded_se,
        )
    )
    curve = RiskEquivalenceCurve(
        propagation_probability=tuple(float(v) for v in bounded_pe),
        single_failure_probability=tuple(float(v) for v in bounded_lhs),
        additional_failure_consequence=tuple(float(v) for v in bounded_rhs),
        residual=tuple(float(v) for v in bounded_residual),
        residual_standard_error=tuple(float(v) for v in bounded_se),
        residual_confidence_interval=tuple(
            (float(lo), float(hi)) for lo, hi in confidence_intervals
        ),
        n_samples=tuple(int(round(v)) for v in bounded_trials_f),
        effective_failure_samples=tuple(
            int(round(v)) for v in bounded_effective_f
        ),
    )

    brackets, exact_roots = _sign_change_brackets(
        bounded_pe, bounded_residual
    )
    candidates: list[
        tuple[float, tuple[float, float], tuple[float, float], int, int]
    ] = []
    for index in exact_roots:
        if bounded_effective_f[index] <= 0:
            continue
        candidates.append(
            (
                float(bounded_pe[index]),
                (float(bounded_pe[index]), float(bounded_pe[index])),
                (0.0, 0.0),
                index,
                index,
            )
        )
    for left, right in brackets:
        estimate = _linear_root(
            float(bounded_pe[left]),
            float(bounded_residual[left]),
            float(bounded_pe[right]),
            float(bounded_residual[right]),
        )
        candidates.append(
            (
                estimate,
                (float(bounded_pe[left]), float(bounded_pe[right])),
                (
                    float(bounded_residual[left]),
                    float(bounded_residual[right]),
                ),
                left,
                right,
            )
        )
    candidates.sort(key=lambda item: item[0])
    crossing_brackets = tuple(candidate[1] for candidate in candidates)
    closest = int(np.argmin(np.abs(bounded_residual)))
    ambiguous = _ambiguous_mc_brackets(bounded_pe, confidence_intervals)
    warnings: list[str] = []
    if np.any(bounded_effective_f <= 0):
        warnings.append(
            "At least one scan point contains no N_f>=1 realization; an "
            "exact zero residual at such a point is not treated as a root."
        )
    if bound_interpolated:
        warnings.append(
            "pe_crit lies between scan points; the residual at the upper bound "
            "is linearly interpolated."
        )

    if len(candidates) != 1:
        reason = (
            "no_crossing_within_bounds"
            if not candidates
            else "multiple_crossings"
        )
        if not candidates and ambiguous:
            warnings.append(
                "Monte Carlo intervals admit at least one unresolved possible "
                "crossing; increase the sample count or refine the scan."
            )
        if len(candidates) > 1:
            warnings.append(
                "More than one observed crossing is present; no crossing is "
                "selected arbitrarily."
            )
        return NumericalRiskEquivalenceResult(
            estimate=None,
            identified=False,
            status="not_identified",
            reason=reason,
            bounds=(0.0, critical),
            root_bracket=None,
            bracket_residuals=None,
            crossing_brackets=crossing_brackets,
            residual_at_estimate=None,
            interpolation="linear",
            boundary_status="not_evaluated",
            closest_grid_point=float(bounded_pe[closest]),
            closest_grid_residual=float(bounded_residual[closest]),
            grid_spacing_min=spacing_min,
            grid_spacing_median=spacing_median,
            grid_spacing_max=spacing_max,
            resolution=None,
            resolution_status="not_applicable",
            monte_carlo_interval=None,
            monte_carlo_crossing_resolved=None,
            monte_carlo_ambiguous_brackets=ambiguous,
            confidence_level=confidence,
            uncertainty_method=uncertainty_method,
            curve=curve,
            assumptions=assumptions,
            warnings=tuple(warnings),
        )

    estimate, bracket, bracket_residuals, left, right = candidates[0]
    if bracket[0] == bracket[1]:
        mc_interval = (
            (bracket[0], bracket[1])
            if confidence_intervals[left, 0] <= 0.0
            <= confidence_intervals[left, 1]
            else None
        )
        mc_resolved = False
        resolution = 0.0
        resolution_status = "exact_grid_point"
    else:
        mc_interval, mc_resolved = _mc_root_interval(
            bracket[0],
            bracket[1],
            (
                float(confidence_intervals[left, 0]),
                float(confidence_intervals[left, 1]),
            ),
            (
                float(confidence_intervals[right, 0]),
                float(confidence_intervals[right, 1]),
            ),
        )
        resolution = float(bracket[1] - bracket[0])
        resolution_status = "interpolated_within_one_scan_interval"
        warnings.append(
            "The numerical estimate is linearly interpolated within the root "
            "bracket; refine that interval to test scan-resolution sensitivity."
        )
    if not mc_resolved:
        warnings.append(
            "The simultaneous endpoint intervals do not resolve the crossing "
            "independently of Monte Carlo uncertainty."
        )
    if min(bounded_effective_f[left], bounded_effective_f[right]) < 30:
        mc_resolved = False
        warnings.append(
            "A root-bracket endpoint has fewer than 30 N_f>=1 realizations; "
            "the normal Monte Carlo approximation is not considered resolved."
        )

    lower_tolerance = 16.0 * np.finfo(float).eps
    if abs(estimate) <= lower_tolerance:
        boundary_status = "lower_bound"
    elif abs(estimate - critical) <= lower_tolerance:
        boundary_status = "upper_bound"
    else:
        boundary_status = "interior"

    residual_at_estimate = (
        0.0
        if bracket[0] != bracket[1]
        else float(bounded_residual[left])
    )
    return NumericalRiskEquivalenceResult(
        estimate=float(estimate),
        identified=True,
        status="identified",
        reason=None,
        bounds=(0.0, critical),
        root_bracket=bracket,
        bracket_residuals=bracket_residuals,
        crossing_brackets=crossing_brackets,
        residual_at_estimate=residual_at_estimate,
        interpolation="linear",
        boundary_status=boundary_status,
        closest_grid_point=float(bounded_pe[closest]),
        closest_grid_residual=float(bounded_residual[closest]),
        grid_spacing_min=spacing_min,
        grid_spacing_median=spacing_median,
        grid_spacing_max=spacing_max,
        resolution=resolution,
        resolution_status=resolution_status,
        monte_carlo_interval=mc_interval,
        monte_carlo_crossing_resolved=mc_resolved,
        monte_carlo_ambiguous_brackets=ambiguous,
        confidence_level=confidence,
        uncertainty_method=uncertainty_method,
        curve=curve,
        assumptions=assumptions,
        warnings=tuple(warnings),
    )


def analyze_risk_equivalence(
    propagation_probability: Sequence[float] | np.ndarray,
    pmf_counts: Sequence[Sequence[int]] | np.ndarray,
    *,
    pe_crit: float,
    lattice_shape: tuple[int, int, int],
    confidence_level: float = 0.95,
) -> RiskEquivalenceResult:
    """Return numerical and analytical estimates in one serializable object."""

    if len(lattice_shape) != 3:
        raise ValueError("lattice_shape must contain exactly (nx, ny, nz).")
    numerical = estimate_numerical_risk_equivalence(
        propagation_probability,
        pmf_counts,
        pe_crit=pe_crit,
        confidence_level=confidence_level,
    )
    analytical = estimate_analytical_risk_equivalence(
        *lattice_shape,
        pe_crit=pe_crit,
    )

    relative: Optional[float] = None
    absolute_relative: Optional[float] = None
    if (
        numerical.estimate is not None
        and analytical.estimate is not None
        and numerical.estimate != 0.0
    ):
        relative = float(
            (analytical.estimate - numerical.estimate)
            / numerical.estimate
        )
        absolute_relative = abs(relative)
    return RiskEquivalenceResult(
        numerical=numerical,
        analytical=analytical,
        relative_difference=relative,
        absolute_relative_difference=absolute_relative,
        difference_definition=(
            "(analytical_estimate - numerical_estimate) / "
            "numerical_estimate"
        ),
        assumptions=_COMMON_ASSUMPTIONS,
    )
