"""Percolation-threshold estimation for binomial Monte Carlo scans.

The paper defines the threshold as an *interior* sign-changing root of

    d² P(N_f = |N|) / d p_e² = 0.

This module implements that definition directly.  It deliberately does not
fall back to a maximum-slope location when an inflection cannot be identified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import brentq
from scipy.signal import find_peaks, peak_prominences


@dataclass(frozen=True)
class AxisScanMetadata:
    """Scientific meaning of the one-dimensional scan parameter."""

    scan_spec: Optional[str]
    mode: str
    varied_axes: tuple[str, ...]
    fixed_axes: tuple[str, ...]
    varied_axis_groups: tuple[tuple[str, ...], ...] = ()
    fixed_axis_values: tuple[tuple[str, Optional[float]], ...] = ()
    source: str = "inferred"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SensitivityDiagnostics:
    """Thresholds obtained under a named set of analysis perturbations."""

    settings: tuple[str, ...]
    thresholds: tuple[Optional[float], ...]
    stable: bool
    spread: Optional[float]
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdFitDiagnostics:
    """Numerical and statistical diagnostics for a threshold fit."""

    method: str
    n_points: int
    response_min: float
    response_max: float
    trials_min: int
    trials_max: int
    spacing_min: Optional[float]
    spacing_median: Optional[float]
    spacing_max: Optional[float]
    monotone_within_mc_uncertainty: bool
    maximum_monotonicity_violation_z: float
    monotonicity_z_limit: float
    smoothing_parameter: Optional[float]
    smoothing_cv_deviance: tuple[tuple[float, float], ...]
    rmse: Optional[float]
    weighted_rmse: Optional[float]
    binomial_deviance: Optional[float]
    interior_roots: tuple[float, ...]
    boundary_roots: tuple[float, ...]
    bootstrap_replicates: int
    bootstrap_identification_rate: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PercolationThresholdResult:
    """Structured result for the paper's percolation-threshold definition."""

    threshold: Optional[float]
    root_bracket: Optional[tuple[float, float]]
    slope: Optional[float]
    boundary_status: str
    uncertainty_interval: Optional[tuple[float, float]]
    smoothing_sensitivity: SensitivityDiagnostics
    scan_spacing_sensitivity: SensitivityDiagnostics
    fit_diagnostics: ThresholdFitDiagnostics
    reason: Optional[str]
    scan_metadata: Optional[AxisScanMetadata] = None

    @property
    def identified(self) -> bool:
        return self.threshold is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _RootAnalysis:
    threshold: Optional[float]
    bracket: Optional[tuple[float, float]]
    slope: Optional[float]
    boundary_status: str
    interior_roots: tuple[float, ...]
    boundary_roots: tuple[float, ...]
    reason: Optional[str]


_MIN_RELATIVE_PEAK_PROMINENCE = 0.05
_DOMINANT_PEAK_PROMINENCE_RATIO = 5.0


_METHOD = (
    "variance-weighted natural cubic smoothing spline; Jeffreys-binomial "
    "variance weights; smoothing selected by leave-one-out binomial deviance "
    "with the one-standard-error rule"
)


def _broadcast_integer_array(
    values: int | Sequence[int] | np.ndarray,
    size: int,
    *,
    name: str,
    strictly_positive: bool,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(size, float(arr), dtype=float)
    if arr.ndim != 1 or arr.size != size:
        raise ValueError(
            f"{name} must be scalar or a one-dimensional array of length {size}."
        )
    if not np.all(np.isfinite(arr)) or not np.all(arr == np.floor(arr)):
        raise ValueError(f"{name} must contain finite integers.")
    if strictly_positive and np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    if not strictly_positive and np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative.")
    return arr.astype(np.int64)


def _validate_inputs(
    grid: Sequence[float] | np.ndarray,
    p_full: Sequence[float] | np.ndarray,
    trials: int | Sequence[int] | np.ndarray,
    successes: Optional[Sequence[int] | np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(grid, dtype=float)
    p = np.asarray(p_full, dtype=float)
    if x.ndim != 1 or p.ndim != 1 or x.size != p.size:
        raise ValueError(
            "grid and p_full must be one-dimensional arrays of equal length."
        )
    if x.size == 0:
        raise ValueError("The percolation scan is empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("The percolation grid must contain only finite values.")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_full must contain finite probabilities in [0, 1].")

    dx = np.diff(x)
    if np.any(dx == 0.0):
        raise ValueError("The percolation grid contains duplicate values.")
    if np.any(dx < 0.0):
        raise ValueError("The percolation grid must be strictly increasing.")

    n = _broadcast_integer_array(
        trials, x.size, name="trials", strictly_positive=True
    )
    if successes is None:
        k = np.rint(p * n).astype(np.int64)
    else:
        k = _broadcast_integer_array(
            successes, x.size, name="successes", strictly_positive=False
        )
        if np.any(k > n):
            raise ValueError("successes cannot exceed trials.")
        tolerance = 0.5000001 / n + 1e-12
        if np.any(np.abs(p - k / n) > tolerance):
            raise ValueError("p_full is inconsistent with successes / trials.")
    return x, p, k, n


def _jeffreys_variance(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    a = k.astype(float) + 0.5
    b = (n - k).astype(float) + 0.5
    total = a + b
    return (a * b) / (total * total * (total + 1.0))


def _precision_weights(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    precision = 1.0 / _jeffreys_variance(k, n)
    return precision / float(np.median(precision))


def _binomial_deviance_scalar(k: int, n: int, probability: float) -> float:
    eps = np.finfo(float).eps
    q = float(np.clip(probability, eps, 1.0 - eps))
    mu = n * q
    value = 0.0
    if k > 0:
        value += k * np.log(k / mu)
    failures = n - k
    if failures > 0:
        value += failures * np.log(failures / (n - mu))
    return float(2.0 * value)


def _binomial_deviance(
    k: np.ndarray, n: np.ndarray, probability: np.ndarray
) -> float:
    return float(
        sum(
            _binomial_deviance_scalar(int(ki), int(ni), float(qi))
            for ki, ni, qi in zip(k, n, probability)
        )
    )


def _fit_spline(
    u: np.ndarray,
    p: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    smoothing: float,
):
    return make_smoothing_spline(
        u,
        p,
        w=_precision_weights(k, n),
        lam=float(smoothing),
    )


def _default_smoothing_candidates() -> np.ndarray:
    return np.logspace(-8.0, 1.0, 16)


def _select_smoothing(
    u: np.ndarray,
    p: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    candidates: np.ndarray,
) -> tuple[float, tuple[tuple[float, float], ...]]:
    if u.size < 6:
        raise ValueError("At least six points are required to select smoothing.")

    held_out = range(1, u.size - 1)
    score_rows: list[list[float]] = []
    means: list[float] = []
    for smoothing in candidates:
        point_scores: list[float] = []
        for index in held_out:
            keep = np.ones(u.size, dtype=bool)
            keep[index] = False
            try:
                spline = _fit_spline(
                    u[keep], p[keep], k[keep], n[keep], float(smoothing)
                )
                prediction = float(spline(float(u[index])))
                point_scores.append(
                    _binomial_deviance_scalar(
                        int(k[index]), int(n[index]), prediction
                    )
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                point_scores.append(float("inf"))
        score_rows.append(point_scores)
        means.append(float(np.mean(point_scores)))

    mean_scores = np.asarray(means, dtype=float)
    if not np.any(np.isfinite(mean_scores)):
        raise RuntimeError("All smoothing candidates failed cross-validation.")

    best = int(np.nanargmin(mean_scores))
    best_scores = np.asarray(score_rows[best], dtype=float)
    finite_best = best_scores[np.isfinite(best_scores)]
    if finite_best.size > 1:
        standard_error = float(
            np.std(finite_best, ddof=1) / np.sqrt(finite_best.size)
        )
    else:
        standard_error = 0.0
    cutoff = float(mean_scores[best] + standard_error)

    acceptable = np.flatnonzero(np.isfinite(mean_scores) & (mean_scores <= cutoff))
    selected = int(acceptable[-1]) if acceptable.size else best
    diagnostics = tuple(
        (float(candidate), float(score))
        for candidate, score in zip(candidates, mean_scores)
    )
    return float(candidates[selected]), diagnostics


def _deduplicate_roots(roots: list[float], tolerance: float) -> list[float]:
    roots = sorted(roots)
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > tolerance:
            unique.append(root)
    return unique


def _analyse_roots(
    spline,
    x: np.ndarray,
    p: np.ndarray,
    *,
    boundary_margin: float,
    root_grid_size: int,
) -> _RootAnalysis:
    x_min = float(x[0])
    x_max = float(x[-1])
    span = x_max - x_min
    median_spacing = float(np.median(np.diff(x)))
    amplitude = max(float(np.ptp(p)), 1e-12)
    curvature_floor = max(1e-10, 1e-4 * amplitude / (span * span))

    second_u = spline.derivative(2)
    dense_u = np.linspace(1e-9, 1.0 - 1e-9, max(513, int(root_grid_size)))
    dense_second = np.asarray(second_u(dense_u), dtype=float) / (span * span)

    roots: list[float] = []
    exact_zero_tolerance = max(1e-14, curvature_floor * 1e-6)
    for index in range(1, dense_u.size - 1):
        if (
            abs(dense_second[index]) <= exact_zero_tolerance
            and dense_second[index - 1] * dense_second[index + 1] < 0.0
            and min(
                abs(dense_second[index - 1]),
                abs(dense_second[index + 1]),
            )
            >= curvature_floor
        ):
            roots.append(x_min + float(dense_u[index]) * span)

    for left, right, f_left, f_right in zip(
        dense_u[:-1], dense_u[1:], dense_second[:-1], dense_second[1:]
    ):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0 or f_right == 0.0 or f_left * f_right > 0.0:
            continue
        try:
            root_u = float(brentq(second_u, float(left), float(right)))
        except ValueError:
            continue

        probe_u = max(median_spacing / span, 4.0 / dense_u.size)
        left_u = max(1e-9, root_u - probe_u)
        right_u = min(1.0 - 1e-9, root_u + probe_u)
        left_curvature = float(second_u(left_u)) / (span * span)
        right_curvature = float(second_u(right_u)) / (span * span)
        if (
            left_curvature * right_curvature >= 0.0
            or min(abs(left_curvature), abs(right_curvature)) < curvature_floor
        ):
            continue
        roots.append(x_min + root_u * span)

    roots = _deduplicate_roots(roots, tolerance=max(1e-8 * span, 1e-12))
    first_u = spline.derivative(1)
    root_slopes = {
        root: float(first_u((root - x_min) / span)) / span
        for root in roots
    }
    slope_floor = max(1e-12, 1e-3 * amplitude / span)

    # A percolation transition is a convex-to-concave inflection, i.e. a
    # local maximum of the fitted first derivative.
    maxima_roots: list[float] = []
    for root in roots:
        root_u = (root - x_min) / span
        probe_u = max(0.25 * median_spacing / span, 2.0 / dense_u.size)
        left_curvature = float(
            second_u(max(1e-9, root_u - probe_u))
        )
        right_curvature = float(
            second_u(min(1.0 - 1e-9, root_u + probe_u))
        )

        if (
            left_curvature > 0.0
            and right_curvature < 0.0
            and root_slopes[root] > slope_floor
        ):
            maxima_roots.append(root)

    roots = maxima_roots

    # Keep one clearly dominant derivative peak. Comparable peaks remain
    # classified as multiple roots.
    if roots:
        dense_slope = np.asarray(first_u(dense_u), dtype=float) / span
        peak_indices, _ = find_peaks(dense_slope)
        prominences = peak_prominences(
            dense_slope,
            peak_indices,
        )[0]

        root_prominence = {
            root: (
                float(
                    prominences[
                        int(
                            np.argmin(
                                np.abs(
                                    dense_u[peak_indices]
                                    - (root - x_min) / span
                                )
                            )
                        )
                    ]
                )
                if peak_indices.size
                else 0.0
            )
            for root in roots
        }

        prominence_floor = max(
            1e-12,
            _MIN_RELATIVE_PEAK_PROMINENCE * amplitude / span,
        )
        roots = [
            root
            for root in roots
            if root_prominence[root] >= prominence_floor
        ]

        if len(roots) > 1:
            ordered = sorted(
                roots,
                key=lambda root: root_prominence[root],
                reverse=True,
            )
            if (
                root_prominence[ordered[0]]
                >= _DOMINANT_PEAK_PROMINENCE_RATIO
                * root_prominence[ordered[1]]
            ):
                roots = [ordered[0]]
    boundary_roots = tuple(
        root
        for root in roots
        if root <= x_min + boundary_margin or root >= x_max - boundary_margin
    )
    interior_roots = tuple(root for root in roots if root not in boundary_roots)

    if not interior_roots:
        if boundary_roots:
            return _RootAnalysis(
                threshold=None,
                bracket=None,
                slope=None,
                boundary_status="boundary_only",
                interior_roots=(),
                boundary_roots=boundary_roots,
                reason="second_derivative_root_only_at_scan_boundary",
            )
        return _RootAnalysis(
            threshold=None,
            bracket=None,
            slope=None,
            boundary_status="no_root",
            interior_roots=(),
            boundary_roots=(),
            reason="no_interior_sign_changing_second_derivative_root",
        )

    if len(interior_roots) > 1:
        return _RootAnalysis(
            threshold=None,
            bracket=None,
            slope=None,
            boundary_status="interior",
            interior_roots=interior_roots,
            boundary_roots=boundary_roots,
            reason="multiple_interior_second_derivative_roots",
        )

    threshold = float(interior_roots[0])
    upper = int(np.searchsorted(x, threshold, side="right"))
    upper = min(max(upper, 1), x.size - 1)
    bracket = (float(x[upper - 1]), float(x[upper]))
    root_u = (threshold - x_min) / span
    slope = float(spline.derivative(1)(root_u)) / span
    return _RootAnalysis(
        threshold=threshold,
        bracket=bracket,
        slope=slope,
        boundary_status="interior",
        interior_roots=interior_roots,
        boundary_roots=boundary_roots,
        reason=None,
    )


def _summarise_sensitivity(
    settings: Sequence[str],
    thresholds: Sequence[Optional[float]],
    tolerance: float,
) -> SensitivityDiagnostics:
    identified = [float(value) for value in thresholds if value is not None]
    spread = (
        float(max(identified) - min(identified)) if len(identified) >= 2 else None
    )
    stable = (
        len(identified) == len(thresholds)
        and bool(identified)
        and (spread is None or spread <= tolerance)
    )
    return SensitivityDiagnostics(
        settings=tuple(settings),
        thresholds=tuple(thresholds),
        stable=stable,
        spread=spread,
        tolerance=float(tolerance),
    )


def _empty_sensitivity(tolerance: float) -> SensitivityDiagnostics:
    return SensitivityDiagnostics((), (), False, None, float(tolerance))


def _early_result(
    *,
    x: np.ndarray,
    p: np.ndarray,
    n: np.ndarray,
    monotone: bool,
    maximum_violation_z: float,
    monotonicity_z_limit: float,
    reason: str,
    scan_metadata: Optional[AxisScanMetadata],
    sensitivity_tolerance: float,
) -> PercolationThresholdResult:
    spacing = np.diff(x)
    diagnostics = ThresholdFitDiagnostics(
        method=_METHOD,
        n_points=int(x.size),
        response_min=float(np.min(p)),
        response_max=float(np.max(p)),
        trials_min=int(np.min(n)),
        trials_max=int(np.max(n)),
        spacing_min=float(np.min(spacing)) if spacing.size else None,
        spacing_median=float(np.median(spacing)) if spacing.size else None,
        spacing_max=float(np.max(spacing)) if spacing.size else None,
        monotone_within_mc_uncertainty=monotone,
        maximum_monotonicity_violation_z=float(maximum_violation_z),
        monotonicity_z_limit=float(monotonicity_z_limit),
        smoothing_parameter=None,
        smoothing_cv_deviance=(),
        rmse=None,
        weighted_rmse=None,
        binomial_deviance=None,
        interior_roots=(),
        boundary_roots=(),
        bootstrap_replicates=0,
        bootstrap_identification_rate=None,
    )
    empty = _empty_sensitivity(sensitivity_tolerance)
    return PercolationThresholdResult(
        threshold=None,
        root_bracket=None,
        slope=None,
        boundary_status="not_evaluated",
        uncertainty_interval=None,
        smoothing_sensitivity=empty,
        scan_spacing_sensitivity=empty,
        fit_diagnostics=diagnostics,
        reason=reason,
        scan_metadata=scan_metadata,
    )


def estimate_percolation_threshold(
    grid: Sequence[float] | np.ndarray,
    p_full: Sequence[float] | np.ndarray,
    *,
    trials: int | Sequence[int] | np.ndarray,
    successes: Optional[Sequence[int] | np.ndarray] = None,
    scan_metadata: Optional[AxisScanMetadata] = None,
    smoothing_candidates: Optional[Sequence[float]] = None,
    smoothing_factors: Sequence[float] = (0.25, 1.0, 4.0),
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 1729,
    minimum_bootstrap_identification: float = 0.80,
    monotonicity_z_limit: float = 3.0,
    boundary_margin: Optional[float] = None,
    sensitivity_tolerance: Optional[float] = None,
    root_grid_size: int = 4097,
) -> PercolationThresholdResult:
    """Estimate an identifiable interior inflection of ``P(N_f=|N|)``.

    Observed probabilities are fitted with a natural cubic smoothing spline.
    The weights are inverse Jeffreys-regularized binomial variances, and the
    smoothing parameter is selected by leave-one-out binomial deviance using
    the one-standard-error rule. The returned root must change the sign of the
    fitted second derivative, correspond to a convex-to-concave derivative
    peak, lie at least 1.5 median scan spacings from either boundary, and
    remain stable under smoothing and alternating-grid coarsening.

    The optional bootstrap is disabled by default because aggregate scan
    results do not retain the dependence between common-random-number scan
    points.

    Invalid grids or probability/count data raise ``ValueError``.  A valid
    curve without an identifiable threshold returns ``threshold=None`` and an
    explicit ``reason``.
    """
    x, p, k, n = _validate_inputs(grid, p_full, trials, successes)
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative.")
    if not isinstance(bootstrap_seed, (int, np.integer)) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer.")
    if not 0.0 <= minimum_bootstrap_identification <= 1.0:
        raise ValueError("minimum_bootstrap_identification must lie in [0, 1].")
    if monotonicity_z_limit <= 0.0:
        raise ValueError("monotonicity_z_limit must be positive.")

    span = float(x[-1] - x[0]) if x.size > 1 else 0.0
    spacing = np.diff(x)
    median_spacing = float(np.median(spacing)) if spacing.size else 0.0
    tolerance = (
        float(sensitivity_tolerance)
        if sensitivity_tolerance is not None
        else max(2.0 * median_spacing, 0.10 * span)
    )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("sensitivity_tolerance must be finite and non-negative.")

    if x.size < 7:
        return _early_result(
            x=x,
            p=p,
            n=n,
            monotone=True,
            maximum_violation_z=0.0,
            monotonicity_z_limit=monotonicity_z_limit,
            reason="insufficient_scan_points",
            scan_metadata=scan_metadata,
            sensitivity_tolerance=tolerance,
        )

    posterior_variance = _jeffreys_variance(k, n)
    standard_error = np.sqrt(posterior_variance[:-1] + posterior_variance[1:])
    decreases = p[:-1] - p[1:]
    violation_z = np.divide(
        np.maximum(decreases, 0.0),
        standard_error,
        out=np.zeros_like(decreases),
        where=standard_error > 0.0,
    )
    maximum_violation_z = float(np.max(violation_z)) if violation_z.size else 0.0
    monotone = maximum_violation_z <= monotonicity_z_limit
    if not monotone:
        return _early_result(
            x=x,
            p=p,
            n=n,
            monotone=False,
            maximum_violation_z=maximum_violation_z,
            monotonicity_z_limit=monotonicity_z_limit,
            reason="response_not_monotone_within_mc_uncertainty",
            scan_metadata=scan_metadata,
            sensitivity_tolerance=tolerance,
        )

    resolved_change = max(
        1e-12,
        3.0 * float(np.median(np.sqrt(posterior_variance))),
    )
    if float(np.ptp(p)) <= resolved_change:
        return _early_result(
            x=x,
            p=p,
            n=n,
            monotone=True,
            maximum_violation_z=maximum_violation_z,
            monotonicity_z_limit=monotonicity_z_limit,
            reason="response_change_not_resolved_above_mc_uncertainty",
            scan_metadata=scan_metadata,
            sensitivity_tolerance=tolerance,
        )

    u = (x - x[0]) / span
    candidates = (
        _default_smoothing_candidates()
        if smoothing_candidates is None
        else np.asarray(smoothing_candidates, dtype=float)
    )
    if (
        candidates.ndim != 1
        or candidates.size == 0
        or not np.all(np.isfinite(candidates))
        or np.any(candidates <= 0.0)
        or np.any(np.diff(candidates) <= 0.0)
    ):
        raise ValueError(
            "smoothing_candidates must be finite, positive, and increasing."
        )

    try:
        smoothing, cv_scores = _select_smoothing(u, p, k, n, candidates)
        spline = _fit_spline(u, p, k, n, smoothing)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
        return _early_result(
            x=x,
            p=p,
            n=n,
            monotone=True,
            maximum_violation_z=maximum_violation_z,
            monotonicity_z_limit=monotonicity_z_limit,
            reason="smoothing_fit_failed",
            scan_metadata=scan_metadata,
            sensitivity_tolerance=tolerance,
        )

    margin = (
        float(boundary_margin)
        if boundary_margin is not None
        else max(1.5 * median_spacing, 0.05 * span)
    )
    if margin < 0.0 or margin >= 0.5 * span:
        raise ValueError(
            "boundary_margin must be non-negative and below half the scan span."
        )

    base = _analyse_roots(
        spline,
        x,
        p,
        boundary_margin=margin,
        root_grid_size=root_grid_size,
    )

    smoothing_settings: list[str] = []
    smoothing_thresholds: list[Optional[float]] = []
    if len(smoothing_factors) == 0:
        raise ValueError("smoothing_factors cannot be empty.")
    for factor in smoothing_factors:
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("smoothing_factors must be finite and positive.")
        perturbed = float(smoothing * factor)
        smoothing_settings.append(f"{factor:g}x")
        try:
            perturbed_spline = _fit_spline(u, p, k, n, perturbed)
            analysed = _analyse_roots(
                perturbed_spline,
                x,
                p,
                boundary_margin=margin,
                root_grid_size=root_grid_size,
            )
            smoothing_thresholds.append(analysed.threshold)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            smoothing_thresholds.append(None)
    smoothing_sensitivity = _summarise_sensitivity(
        smoothing_settings, smoothing_thresholds, tolerance
    )

    spacing_settings = ["full_grid"]
    spacing_thresholds: list[Optional[float]] = [base.threshold]
    interior_indices = np.arange(1, x.size - 1)
    coarsened_sets = (
        np.concatenate(
            ([0], interior_indices[interior_indices % 2 == 0], [x.size - 1])
        ),
        np.concatenate(
            ([0], interior_indices[interior_indices % 2 == 1], [x.size - 1])
        ),
    )
    for label, indices in zip(("alternate_even", "alternate_odd"), coarsened_sets):
        indices = np.unique(indices)
        threshold: Optional[float] = None
        if indices.size >= 6:
            sub_x = x[indices]
            sub_u = (sub_x - sub_x[0]) / (sub_x[-1] - sub_x[0])
            try:
                sub_smoothing, _ = _select_smoothing(
                    sub_u, p[indices], k[indices], n[indices], candidates
                )
                sub_spline = _fit_spline(
                    sub_u, p[indices], k[indices], n[indices], sub_smoothing
                )
                threshold = _analyse_roots(
                    sub_spline,
                    sub_x,
                    p[indices],
                    boundary_margin=margin,
                    root_grid_size=root_grid_size,
                ).threshold
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                threshold = None
        spacing_settings.append(label)
        spacing_thresholds.append(threshold)
    spacing_sensitivity = _summarise_sensitivity(
        spacing_settings, spacing_thresholds, tolerance
    )

    uncertainty_interval: Optional[tuple[float, float]] = None
    bootstrap_identification_rate: Optional[float] = None
    bootstrap_thresholds: list[float] = []
    if bootstrap_replicates > 0 and base.threshold is not None:
        fitted_probability = np.clip(
            np.asarray(spline(u), dtype=float), 0.0, 1.0
        )
        rng = np.random.default_rng(int(bootstrap_seed))
        for _ in range(int(bootstrap_replicates)):
            boot_k = rng.binomial(n, fitted_probability).astype(np.int64)
            boot_p = boot_k / n
            try:
                boot_spline = _fit_spline(u, boot_p, boot_k, n, smoothing)
                boot = _analyse_roots(
                    boot_spline,
                    x,
                    boot_p,
                    boundary_margin=margin,
                    root_grid_size=root_grid_size,
                )
                if boot.threshold is not None:
                    bootstrap_thresholds.append(float(boot.threshold))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
        bootstrap_identification_rate = (
            len(bootstrap_thresholds) / float(bootstrap_replicates)
        )
        if len(bootstrap_thresholds) >= 2:
            lower, upper = np.quantile(bootstrap_thresholds, [0.025, 0.975])
            uncertainty_interval = (float(lower), float(upper))

    fitted = np.clip(np.asarray(spline(u), dtype=float), 0.0, 1.0)
    weights = _precision_weights(k, n)
    diagnostics = ThresholdFitDiagnostics(
        method=_METHOD,
        n_points=int(x.size),
        response_min=float(np.min(p)),
        response_max=float(np.max(p)),
        trials_min=int(np.min(n)),
        trials_max=int(np.max(n)),
        spacing_min=float(np.min(spacing)),
        spacing_median=median_spacing,
        spacing_max=float(np.max(spacing)),
        monotone_within_mc_uncertainty=True,
        maximum_monotonicity_violation_z=maximum_violation_z,
        monotonicity_z_limit=float(monotonicity_z_limit),
        smoothing_parameter=float(smoothing),
        smoothing_cv_deviance=cv_scores,
        rmse=float(np.sqrt(np.mean((fitted - p) ** 2))),
        weighted_rmse=float(
            np.sqrt(np.sum(weights * (fitted - p) ** 2) / np.sum(weights))
        ),
        binomial_deviance=_binomial_deviance(k, n, fitted),
        interior_roots=base.interior_roots,
        boundary_roots=base.boundary_roots,
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_identification_rate=bootstrap_identification_rate,
    )

    threshold = base.threshold
    reason = base.reason
    if threshold is not None and not smoothing_sensitivity.stable:
        threshold = None
        reason = "threshold_not_stable_to_smoothing"
    elif threshold is not None and not spacing_sensitivity.stable:
        threshold = None
        reason = "threshold_not_stable_to_scan_spacing"
    elif (
        threshold is not None
        and bootstrap_identification_rate is not None
        and bootstrap_identification_rate < minimum_bootstrap_identification
    ):
        threshold = None
        reason = "threshold_not_stable_under_binomial_mc_uncertainty"

    return PercolationThresholdResult(
        threshold=threshold,
        root_bracket=base.bracket,
        slope=base.slope,
        boundary_status=base.boundary_status,
        uncertainty_interval=uncertainty_interval,
        smoothing_sensitivity=smoothing_sensitivity,
        scan_spacing_sensitivity=spacing_sensitivity,
        fit_diagnostics=diagnostics,
        reason=reason,
        scan_metadata=scan_metadata,
    )


def axis_scan_metadata_from_mapping(
    value: Mapping[str, Any],
    *,
    source: str = "run_notes",
) -> AxisScanMetadata:
    """Construct validated axis metadata from persisted JSON-compatible data."""
    varied_axes = tuple(str(axis) for axis in value.get("varied_axes", ()))
    fixed_axes = tuple(str(axis) for axis in value.get("fixed_axes", ()))
    all_axes = varied_axes + fixed_axes
    if set(all_axes) != {"x", "y", "z"} or len(all_axes) != 3:
        raise ValueError("Axis metadata must classify x, y, and z exactly once.")

    groups = tuple(
        tuple(str(axis) for axis in group)
        for group in value.get("varied_axis_groups", ())
    )
    fixed_values_raw = value.get("fixed_axis_values", {})
    if isinstance(fixed_values_raw, Mapping):
        fixed_values = tuple(
            (
                axis,
                None
                if fixed_values_raw.get(axis) is None
                else float(fixed_values_raw[axis]),
            )
            for axis in fixed_axes
        )
    else:
        fixed_values = tuple(
            (str(axis), None if fixed is None else float(fixed))
            for axis, fixed in fixed_values_raw
        )
    for axis, fixed in fixed_values:
        if axis not in fixed_axes:
            raise ValueError(f"Fixed-axis value provided for non-fixed axis {axis!r}.")
        if fixed is not None and (
            not np.isfinite(fixed) or not 0.0 <= fixed <= 1.0
        ):
            raise ValueError(
                f"Fixed-axis probability for {axis!r} must lie in [0, 1]."
            )
    return AxisScanMetadata(
        scan_spec=(
            None if value.get("scan_spec") is None else str(value.get("scan_spec"))
        ),
        mode=str(value.get("mode", "unknown")),
        varied_axes=varied_axes,
        fixed_axes=fixed_axes,
        varied_axis_groups=groups,
        fixed_axis_values=fixed_values,
        source=source,
    )
