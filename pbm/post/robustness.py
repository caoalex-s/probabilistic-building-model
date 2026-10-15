r"""Entropy-based robustness for stationary notional-damage analyses.

The authoritative scalar definition is :func:`bernoulli_kl_robustness`:

.. math::

   I_\mathrm{rob}
   = \exp\{-c D_\mathrm{KL}[\mathrm{Bern}(p_0)\|
                           \mathrm{Bern}(p_j)]\}.

``p0`` and ``pj`` must describe the *same* explicitly defined adverse
macro-state before and after one specified notional-damage scenario.  In this
repository that macro-state is ``N_f >= S_lim``.  Percolation scan endpoints
and ``p_any`` are not damage scenarios and are deliberately unsupported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


CoherenceMode = Literal["strict", "diagnostic"]


class RobustnessCoherenceError(ValueError):
    """Raised when a damaged probability violates the coherence assumption."""


def _finite_json_number(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def _number_status(value: float) -> str:
    numeric = float(value)
    if math.isnan(numeric):
        return "not_a_number"
    if numeric == math.inf:
        return "positive_infinity"
    if numeric == -math.inf:
        return "negative_infinity"
    return "finite"


def _validate_probability(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite probability in [0, 1].")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1].")
    return probability


def _validate_positive_scale(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("c must be a positive finite number.")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("c must be a positive finite number.")
    return scale


def _validate_tolerance(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("mc_tolerance must be a finite value in [0, 1].")
    tolerance = float(value)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
        raise ValueError("mc_tolerance must be a finite value in [0, 1].")
    return tolerance


def _relative_entropy_term(reference: float, comparison: float) -> float:
    """Return ``reference * log(reference / comparison)`` exactly at bounds."""

    if reference == 0.0:
        return 0.0
    if comparison == 0.0:
        return math.inf
    return reference * math.log(reference / comparison)


def _bernoulli_kl_divergence(p0: float, pj: float) -> float:
    divergence = _relative_entropy_term(p0, pj) + _relative_entropy_term(
        1.0 - p0,
        1.0 - pj,
    )
    # Round-off can make the sum very slightly negative near p0 == pj.
    if math.isfinite(divergence) and divergence < 0.0:
        if divergence >= -1e-14:
            return 0.0
        raise ArithmeticError("Computed a negative Bernoulli KL divergence.")
    return divergence


@dataclass(frozen=True)
class BernoulliRobustnessResult:
    """Result and coherence diagnostic for one pair of macrostate probabilities."""

    value: Optional[float]
    kl_divergence: float
    p0: float
    pj: float
    c: float
    coherence_mode: CoherenceMode
    mc_tolerance: float
    coherent: bool
    accepted: bool
    status: str
    reason: Optional[str]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["kl_divergence"] = _finite_json_number(self.kl_divergence)
        payload["kl_divergence_status"] = _number_status(self.kl_divergence)
        return payload


def bernoulli_kl_robustness(
    p0: float,
    pj: float,
    *,
    c: float = 1.0,
    coherence_mode: CoherenceMode = "strict",
    mc_tolerance: float = 0.0,
) -> BernoulliRobustnessResult:
    """Evaluate the Cao--Beck Bernoulli KL-divergence robustness index.

    Parameters
    ----------
    p0, pj
        Probabilities of one identical adverse macro-state for the undamaged
        system and for a specified damaged system, respectively.
    c
        Positive scale in ``exp(-c D_KL)``.  The published and stationary-paper
        analyses use ``c=1``.
    coherence_mode
        ``"strict"`` raises :class:`RobustnessCoherenceError` when
        ``pj < p0 - mc_tolerance``.  ``"diagnostic"`` returns an invalid result
        with ``value=None`` and an explicit reason.
    mc_tolerance
        Optional absolute probability tolerance for Monte Carlo sampling
        variation.  A small reversal within this tolerance is accepted but
        remains explicitly labelled ``within_monte_carlo_tolerance``.

    Notes
    -----
    Boundary probabilities are evaluated by the exact KL conventions
    ``0 log(0/q)=0`` and ``p log(p/0)=+inf``.  They are not clipped.
    """

    baseline = _validate_probability(p0, name="p0")
    damaged = _validate_probability(pj, name="pj")
    scale = _validate_positive_scale(c)
    tolerance = _validate_tolerance(mc_tolerance)
    if coherence_mode not in ("strict", "diagnostic"):
        raise ValueError("coherence_mode must be 'strict' or 'diagnostic'.")

    coherent = damaged >= baseline
    accepted = damaged + tolerance >= baseline
    divergence = _bernoulli_kl_divergence(baseline, damaged)

    if not accepted:
        reason = (
            "The damaged adverse-state probability is below the undamaged "
            f"baseline by {baseline - damaged:.6g}, exceeding the stated "
            f"Monte Carlo tolerance {tolerance:.6g}."
        )
        if coherence_mode == "strict":
            raise RobustnessCoherenceError(reason)
        return BernoulliRobustnessResult(
            value=None,
            kl_divergence=divergence,
            p0=baseline,
            pj=damaged,
            c=scale,
            coherence_mode=coherence_mode,
            mc_tolerance=tolerance,
            coherent=False,
            accepted=False,
            status="incoherent",
            reason=reason,
        )

    if damaged == baseline:
        status = "identity"
        reason = None
    elif not coherent:
        status = "within_monte_carlo_tolerance"
        reason = (
            "The observed damaged probability is below p0, but the reversal "
            "is within the explicitly stated Monte Carlo tolerance."
        )
    else:
        status = "coherent"
        reason = None

    value = 0.0 if math.isinf(divergence) else math.exp(-scale * divergence)
    return BernoulliRobustnessResult(
        value=value,
        kl_divergence=divergence,
        p0=baseline,
        pj=damaged,
        c=scale,
        coherence_mode=coherence_mode,
        mc_tolerance=tolerance,
        coherent=coherent,
        accepted=True,
        status=status,
        reason=reason,
    )


def _validate_s_lim(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("S_lim must be a positive integer.")
    threshold = int(value)
    if threshold < 1:
        raise ValueError("S_lim must be a positive integer.")
    return threshold


def _validated_pmf(
    support: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    normalization_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    s_raw = np.asarray(support)
    p = np.asarray(probabilities, dtype=float)
    if s_raw.ndim != 1 or p.ndim != 1 or s_raw.size != p.size or s_raw.size == 0:
        raise ValueError("PMF support and probabilities must be non-empty 1-D arrays.")
    if not np.all(np.isfinite(s_raw.astype(float))):
        raise ValueError("PMF support must contain finite integers.")
    if not np.all(np.equal(s_raw, np.floor(s_raw.astype(float)))):
        raise ValueError("PMF support must contain integers.")
    s = s_raw.astype(np.int64)
    expected = np.arange(int(np.max(s)) + 1, dtype=np.int64)
    if not np.array_equal(s, expected):
        raise ValueError("PMF support must be the ordered complete range 0,...,|N|.")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("PMF entries must be finite probabilities in [0, 1].")
    tolerance = _validate_tolerance(normalization_tolerance)
    if abs(float(np.sum(p)) - 1.0) > tolerance:
        raise ValueError("PMF probabilities must sum to one within tolerance.")
    return s, p


def adverse_probability_from_pmf(
    support: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    s_lim: int,
    normalization_tolerance: float = 1e-10,
) -> float:
    """Return the explicitly defined macrostate probability ``P(N_f>=S_lim)``."""

    threshold = _validate_s_lim(s_lim)
    s, p = _validated_pmf(
        support,
        probabilities,
        normalization_tolerance=normalization_tolerance,
    )
    if threshold > int(s[-1]):
        raise ValueError(f"S_lim={threshold} exceeds the system size |N|={int(s[-1])}.")
    return float(np.sum(p[s >= threshold]))


@dataclass(frozen=True)
class DamageScenarioRobustnessResult:
    """Robustness conditional on one named damage and one adverse macro-state."""

    baseline_scenario: str
    damaged_scenario: str
    s_lim: int
    adverse_macrostate: str
    p0: float
    pj: float
    robustness: BernoulliRobustnessResult

    @property
    def value(self) -> Optional[float]:
        return self.robustness.value

    def to_dict(self) -> dict:
        return {
            "baseline_scenario": self.baseline_scenario,
            "damaged_scenario": self.damaged_scenario,
            "s_lim": self.s_lim,
            "adverse_macrostate": self.adverse_macrostate,
            "p0": self.p0,
            "pj": self.pj,
            "robustness": self.robustness.to_dict(),
        }


def evaluate_damage_scenario_robustness(
    baseline_support: Sequence[int] | np.ndarray,
    baseline_probabilities: Sequence[float] | np.ndarray,
    damaged_support: Sequence[int] | np.ndarray,
    damaged_probabilities: Sequence[float] | np.ndarray,
    *,
    s_lim: int,
    damaged_scenario: str | int,
    baseline_scenario: str | int = "undamaged",
    c: float = 1.0,
    coherence_mode: CoherenceMode = "strict",
    mc_tolerance: float = 0.0,
) -> DamageScenarioRobustnessResult:
    """Evaluate one explicitly specified notional-damage scenario.

    Both PMFs are reduced independently to the same adverse macro-state
    ``N_f >= S_lim``.  This is the package path used by plotting and the CLI.
    """

    threshold = _validate_s_lim(s_lim)
    if damaged_scenario is None or not str(damaged_scenario).strip():
        raise ValueError("damaged_scenario must be explicitly specified.")
    if baseline_scenario is None or not str(baseline_scenario).strip():
        raise ValueError("baseline_scenario must be explicitly specified.")

    baseline_s, baseline_p = _validated_pmf(
        baseline_support,
        baseline_probabilities,
        normalization_tolerance=1e-10,
    )
    damaged_s, damaged_p = _validated_pmf(
        damaged_support,
        damaged_probabilities,
        normalization_tolerance=1e-10,
    )
    if not np.array_equal(baseline_s, damaged_s):
        raise ValueError(
            "Undamaged and damaged PMFs must use the same system-size support."
        )

    p0 = adverse_probability_from_pmf(
        baseline_s,
        baseline_p,
        s_lim=threshold,
    )
    pj = adverse_probability_from_pmf(
        damaged_s,
        damaged_p,
        s_lim=threshold,
    )
    evaluation = bernoulli_kl_robustness(
        p0,
        pj,
        c=c,
        coherence_mode=coherence_mode,
        mc_tolerance=mc_tolerance,
    )
    return DamageScenarioRobustnessResult(
        baseline_scenario=str(baseline_scenario),
        damaged_scenario=str(damaged_scenario),
        s_lim=threshold,
        adverse_macrostate=f"N_f >= {threshold}",
        p0=p0,
        pj=pj,
        robustness=evaluation,
    )


@dataclass(frozen=True)
class DerivativeRootDiagnostics:
    """Candidate roots and the selected root of one published derivative."""

    derivative_order: int
    selected_root: Optional[float]
    selected_bracket: Optional[tuple[float, float]]
    candidate_roots: tuple[float, ...]
    candidate_brackets: tuple[tuple[float, float], ...]
    reason: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EntropyRobustnessThresholds:
    """Derivative-based Cao--Beck robustness thresholds and diagnostics."""

    p0: float
    beta: float
    pi_grid: np.ndarray
    pj_grid: np.ndarray
    I_grid: np.ndarray
    pi_minus: Optional[float]
    pi_plus: Optional[float]
    I_minus: Optional[float]
    I_plus: Optional[float]
    second_derivative: DerivativeRootDiagnostics
    third_derivative: DerivativeRootDiagnostics
    reason: Optional[str]

    @property
    def identified(self) -> bool:
        return self.pi_minus is not None and self.pi_plus is not None

    def to_dict(self, *, include_grid: bool = False) -> dict:
        payload = {
            "p0": self.p0,
            "beta": _finite_json_number(self.beta),
            "beta_status": _number_status(self.beta),
            "pi_minus": self.pi_minus,
            "pi_plus": self.pi_plus,
            "I_minus": self.I_minus,
            "I_plus": self.I_plus,
            "identified": self.identified,
            "second_derivative": self.second_derivative.to_dict(),
            "third_derivative": self.third_derivative.to_dict(),
            "reason": self.reason,
        }
        if include_grid:
            payload.update(
                {
                    "pi_grid": self.pi_grid.tolist(),
                    "pj_grid": self.pj_grid.tolist(),
                    "I_grid": [
                        _finite_json_number(value) for value in self.I_grid
                    ],
                }
            )
        return payload


def _curve_quantities(pi: float, p0: float, c: float) -> tuple[float, ...]:
    """Return ``I``, first, second, and third derivatives in pi-space."""

    pj = float(norm.sf(pi))
    survival_j = float(norm.cdf(pi))
    phi = float(norm.pdf(pi))
    divergence = _bernoulli_kl_divergence(p0, pj)
    robustness = 0.0 if math.isinf(divergence) else math.exp(-c * divergence)

    chi = (1.0 - p0) / survival_j - p0 / pj
    psi = (1.0 - p0) / survival_j**2 + p0 / pj**2
    psi_p = 2.0 * (1.0 - p0) / survival_j**3 - 2.0 * p0 / pj**3

    omega = -pi * chi + phi * (c * chi**2 - psi)
    omega_prime = (
        -chi
        + pi * phi * psi
        - pi * phi * (c * chi**2 - psi)
        + phi**2 * (psi_p - 2.0 * c * chi * psi)
    )

    first = robustness * c * phi * chi
    second = robustness * c * phi * omega
    third = robustness * c * phi * (
        c * phi * chi * omega - pi * omega + omega_prime
    )
    return robustness, first, second, third


def _sign_changing_roots(
    x: np.ndarray,
    y: np.ndarray,
    function,
) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...]]:
    roots: list[float] = []
    brackets: list[tuple[float, float]] = []
    for index in range(x.size - 1):
        x0, x1 = float(x[index]), float(x[index + 1])
        y0, y1 = float(y[index]), float(y[index + 1])
        if not math.isfinite(y0) or not math.isfinite(y1):
            continue
        if y0 == 0.0:
            root = x0
            bracket = (x0, x0)
        elif y0 * y1 < 0.0:
            root = float(brentq(function, x0, x1))
            bracket = (x0, x1)
        else:
            continue
        if roots and abs(root - roots[-1]) <= 1e-10:
            continue
        roots.append(root)
        brackets.append(bracket)

    if x.size and math.isfinite(float(y[-1])) and float(y[-1]) == 0.0:
        root = float(x[-1])
        if not roots or abs(root - roots[-1]) > 1e-10:
            roots.append(root)
            brackets.append((root, root))
    return tuple(roots), tuple(brackets)


def entropy_robustness_thresholds(
    p0: float,
    *,
    pi_min: float = -3.0,
    pi_max: float = 3.0,
    n_grid: int = 1001,
    c: float = 1.0,
) -> EntropyRobustnessThresholds:
    """Locate the published derivative roots without fallback positions.

    ``pi_minus`` is the sign-changing root of ``d²I/dpi²`` with the largest
    local ``|dI/dpi|``.  ``pi_plus`` is the subsequent sign-changing root of
    ``d³I/dpi³`` with the largest local ``|d²I/dpi²|``.  This reproduces the
    Cao--Beck thresholds while making an absent root a valid diagnostic result.
    Only the coherent domain ``P_j >= P_0`` is searched.
    """

    baseline = _validate_probability(p0, name="p0")
    scale = _validate_positive_scale(c)
    if not isinstance(n_grid, (int, np.integer)) or isinstance(
        n_grid, (bool, np.bool_)
    ):
        raise ValueError("n_grid must be an integer >= 25.")
    if int(n_grid) < 25:
        raise ValueError("n_grid must be an integer >= 25.")
    lower, upper = float(pi_min), float(pi_max)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("pi_min and pi_max must be finite with pi_min < pi_max.")

    pi_grid = np.linspace(lower, upper, int(n_grid))
    pj_grid = norm.sf(pi_grid)
    if np.any((pj_grid <= 0.0) | (pj_grid >= 1.0)):
        raise ValueError(
            "The pi search range is too extreme for finite normal-tail "
            "probabilities."
        )
    beta = float(norm.isf(baseline))
    coherent_mask = pj_grid >= baseline

    I_grid = np.full(pi_grid.shape, np.nan, dtype=float)
    derivatives = np.full((3, pi_grid.size), np.nan, dtype=float)
    for index in np.flatnonzero(coherent_mask):
        quantities = _curve_quantities(float(pi_grid[index]), baseline, scale)
        I_grid[index] = quantities[0]
        derivatives[:, index] = quantities[1:]

    search_x = pi_grid[coherent_mask]
    search_derivatives = derivatives[:, coherent_mask]
    if search_x.size < 2:
        reason = "coherent_search_domain_has_fewer_than_two_grid_points"
        missing_second = DerivativeRootDiagnostics(
            derivative_order=2,
            selected_root=None,
            selected_bracket=None,
            candidate_roots=(),
            candidate_brackets=(),
            reason=reason,
        )
        missing_third = DerivativeRootDiagnostics(
            derivative_order=3,
            selected_root=None,
            selected_bracket=None,
            candidate_roots=(),
            candidate_brackets=(),
            reason=reason,
        )
        return EntropyRobustnessThresholds(
            p0=baseline,
            beta=beta,
            pi_grid=pi_grid,
            pj_grid=pj_grid,
            I_grid=I_grid,
            pi_minus=None,
            pi_plus=None,
            I_minus=None,
            I_plus=None,
            second_derivative=missing_second,
            third_derivative=missing_third,
            reason=reason,
        )

    second_roots, second_brackets = _sign_changing_roots(
        search_x,
        search_derivatives[1],
        lambda value: _curve_quantities(value, baseline, scale)[2],
    )
    if second_roots:
        first_at_roots = [
            abs(_curve_quantities(root, baseline, scale)[1])
            for root in second_roots
        ]
        second_index = int(np.argmax(first_at_roots))
        pi_minus = second_roots[second_index]
        second_bracket = second_brackets[second_index]
        second_reason = None
    else:
        pi_minus = None
        second_bracket = None
        second_reason = "no_sign_changing_second_derivative_root"

    third_roots, third_brackets = _sign_changing_roots(
        search_x,
        search_derivatives[2],
        lambda value: _curve_quantities(value, baseline, scale)[3],
    )
    if pi_minus is None:
        pi_plus = None
        third_bracket = None
        third_reason = "lower_threshold_not_identified"
    else:
        eligible = [
            index
            for index, root in enumerate(third_roots)
            if root > pi_minus + 1e-10
        ]
        if eligible:
            third_index = max(
                eligible,
                key=lambda index: abs(
                    _curve_quantities(third_roots[index], baseline, scale)[2]
                ),
            )
            pi_plus = third_roots[third_index]
            third_bracket = third_brackets[third_index]
            third_reason = None
        else:
            pi_plus = None
            third_bracket = None
            third_reason = "no_subsequent_sign_changing_third_derivative_root"

    second_diagnostic = DerivativeRootDiagnostics(
        derivative_order=2,
        selected_root=pi_minus,
        selected_bracket=second_bracket,
        candidate_roots=second_roots,
        candidate_brackets=second_brackets,
        reason=second_reason,
    )
    third_diagnostic = DerivativeRootDiagnostics(
        derivative_order=3,
        selected_root=pi_plus,
        selected_bracket=third_bracket,
        candidate_roots=third_roots,
        candidate_brackets=third_brackets,
        reason=third_reason,
    )

    I_minus = (
        _curve_quantities(pi_minus, baseline, scale)[0]
        if pi_minus is not None
        else None
    )
    I_plus = (
        _curve_quantities(pi_plus, baseline, scale)[0]
        if pi_plus is not None
        else None
    )
    reasons = [
        diagnostic
        for diagnostic in (second_reason, third_reason)
        if diagnostic is not None
    ]
    return EntropyRobustnessThresholds(
        p0=baseline,
        beta=beta,
        pi_grid=pi_grid,
        pj_grid=pj_grid,
        I_grid=I_grid,
        pi_minus=pi_minus,
        pi_plus=pi_plus,
        I_minus=I_minus,
        I_plus=I_plus,
        second_derivative=second_diagnostic,
        third_derivative=third_diagnostic,
        reason=";".join(reasons) if reasons else None,
    )
