"""Database and plotting interface for notional-damage robustness campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import logging
import math
from pathlib import Path
import sqlite3
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .percolation_plots import _batlow
from .plot_style import PlotMode, plotting_context, save_figure
from .pmf_plots import load_pmf
from .robustness import (
    CoherenceMode,
    DamageScenarioRobustnessResult,
    EntropyRobustnessThresholds,
    adverse_probability_from_pmf,
    entropy_robustness_thresholds,
    evaluate_damage_scenario_robustness,
)


logger = logging.getLogger(__name__)


class RobustnessPairingError(ValueError):
    """Raised when undamaged and damaged runs are not a valid model pair."""


@dataclass(frozen=True)
class RobustnessRunMetadata:
    run_id: int
    scenario_id: int
    damage_node_ids: tuple[int, ...]
    model_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DamagedRunRobustness:
    run: RobustnessRunMetadata
    label: str
    result: DamageScenarioRobustnessResult

    def to_dict(self) -> dict:
        return {
            "run": self.run.to_dict(),
            "label": self.label,
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class MacrostateRobustnessAnalysis:
    s_lim: int
    adverse_macrostate: str
    p0: float
    thresholds: EntropyRobustnessThresholds
    damaged_runs: tuple[DamagedRunRobustness, ...]

    def to_dict(self) -> dict:
        return {
            "s_lim": self.s_lim,
            "adverse_macrostate": self.adverse_macrostate,
            "p0": self.p0,
            "thresholds": self.thresholds.to_dict(),
            "damaged_runs": [item.to_dict() for item in self.damaged_runs],
        }


@dataclass(frozen=True)
class RobustnessCampaignResult:
    baseline_run: RobustnessRunMetadata
    damaged_runs: tuple[RobustnessRunMetadata, ...]
    coherence_mode: CoherenceMode
    mc_tolerance: float
    c: float
    macrostates: tuple[MacrostateRobustnessAnalysis, ...]
    output_paths: tuple[Path, ...] = ()

    def to_dict(self) -> dict:
        return {
            "baseline_run": self.baseline_run.to_dict(),
            "damaged_runs": [run.to_dict() for run in self.damaged_runs],
            "coherence_mode": self.coherence_mode,
            "mc_tolerance": self.mc_tolerance,
            "c": self.c,
            "macrostates": [item.to_dict() for item in self.macrostates],
            "output_paths": [str(path) for path in self.output_paths],
        }


def _parse_s_lim(value: int | str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        raw = [int(value)]
    elif isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        try:
            raw = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("Every S_lim value must be a positive integer.") from exc
    else:
        raw = []
        for item in value:
            if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (int, np.integer)
            ):
                raise ValueError("Every S_lim value must be a positive integer.")
            raw.append(int(item))
    if not raw:
        raise ValueError("At least one explicit S_lim value is required.")
    if any(item < 1 for item in raw):
        raise ValueError("Every S_lim value must be a positive integer.")
    if len(set(raw)) != len(raw):
        raise ValueError("S_lim values must be unique.")
    return tuple(raw)


def _load_run_metadata(
    conn: sqlite3.Connection,
    run_id: int,
) -> RobustnessRunMetadata:
    row = conn.execute(
        "SELECT scenario_id, notes FROM runs WHERE id = ?;",
        (int(run_id),),
    ).fetchone()
    if row is None:
        raise RobustnessPairingError(f"Run {run_id} does not exist.")
    scenario_id, raw_notes = row
    try:
        notes = json.loads(raw_notes)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RobustnessPairingError(
            f"Run {run_id} has invalid JSON metadata."
        ) from exc
    if not isinstance(notes, dict):
        raise RobustnessPairingError(f"Run {run_id} metadata must be an object.")

    damage = notes.get("notional_damage_node_ids")
    if not isinstance(damage, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in damage
    ):
        raise RobustnessPairingError(
            f"Run {run_id} does not record a valid notional-damage node set."
        )
    if len(set(damage)) != len(damage):
        raise RobustnessPairingError(
            f"Run {run_id} records duplicate notional-damage nodes."
        )

    provenance = notes.get("stationary_model_provenance")
    if not isinstance(provenance, dict):
        raise RobustnessPairingError(
            f"Run {run_id} lacks stationary model provenance; re-run it with "
            "the current simulation interface."
        )
    fingerprint = provenance.get("sha256")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise RobustnessPairingError(
            f"Run {run_id} has invalid stationary model provenance."
        )
    return RobustnessRunMetadata(
        run_id=int(run_id),
        scenario_id=int(scenario_id),
        damage_node_ids=tuple(sorted(int(item) for item in damage)),
        model_sha256=fingerprint,
    )


def _validate_pairing(
    baseline: RobustnessRunMetadata,
    damaged: RobustnessRunMetadata,
) -> None:
    if baseline.run_id == damaged.run_id:
        raise RobustnessPairingError(
            "The baseline and damaged scenario must be different runs."
        )
    if baseline.damage_node_ids:
        raise RobustnessPairingError(
            f"Baseline run {baseline.run_id} is not undamaged."
        )
    if not damaged.damage_node_ids:
        raise RobustnessPairingError(
            f"Damaged run {damaged.run_id} has no specified notional damage."
        )
    if baseline.model_sha256 != damaged.model_sha256:
        raise RobustnessPairingError(
            f"Runs {baseline.run_id} and {damaged.run_id} do not represent "
            "the same stationary model apart from notional damage."
        )


def analyze_entropy_robustness_campaign(
    conn: sqlite3.Connection,
    *,
    baseline_run_id: int,
    damaged_run_ids: Sequence[int],
    s_lim: int | str | Sequence[int],
    scenario_labels: dict[int, str] | None = None,
    coherence_mode: CoherenceMode = "strict",
    mc_tolerance: float = 0.0,
    c: float = 1.0,
    pi_min: float = -3.0,
    pi_max: float = 3.0,
    n_grid: int = 1001,
) -> RobustnessCampaignResult:
    """Analyze compatible undamaged/damaged PMFs for explicit macrostates."""

    s_lim_values = _parse_s_lim(s_lim)
    damaged_ids = tuple(int(run_id) for run_id in damaged_run_ids)
    if not damaged_ids:
        raise ValueError("At least one damaged_run_id is required.")
    if len(set(damaged_ids)) != len(damaged_ids):
        raise ValueError("damaged_run_ids must be unique.")

    baseline = _load_run_metadata(conn, int(baseline_run_id))
    if baseline.damage_node_ids:
        raise RobustnessPairingError(
            f"Baseline run {baseline.run_id} is not undamaged."
        )
    damaged_metadata = tuple(
        _load_run_metadata(conn, run_id) for run_id in damaged_ids
    )
    for damaged in damaged_metadata:
        _validate_pairing(baseline, damaged)

    baseline_support, baseline_pmf = load_pmf(conn, baseline.run_id)
    if baseline_support.size == 0:
        raise ValueError(f"No PMF found for baseline run {baseline.run_id}.")

    damaged_pmfs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for damaged in damaged_metadata:
        support, probabilities = load_pmf(conn, damaged.run_id)
        if support.size == 0:
            raise ValueError(f"No PMF found for damaged run {damaged.run_id}.")
        damaged_pmfs[damaged.run_id] = (support, probabilities)

    macrostates: list[MacrostateRobustnessAnalysis] = []
    for threshold in s_lim_values:
        # This baseline is intentionally recomputed for every S_lim.
        p0 = adverse_probability_from_pmf(
            baseline_support,
            baseline_pmf,
            s_lim=threshold,
        )
        threshold_result = entropy_robustness_thresholds(
            p0,
            pi_min=pi_min,
            pi_max=pi_max,
            n_grid=n_grid,
            c=c,
        )

        evaluations: list[DamagedRunRobustness] = []
        for damaged in damaged_metadata:
            support, probabilities = damaged_pmfs[damaged.run_id]
            label = (
                scenario_labels.get(damaged.run_id, f"Run {damaged.run_id}")
                if scenario_labels is not None
                else f"Run {damaged.run_id}"
            )
            result = evaluate_damage_scenario_robustness(
                baseline_support,
                baseline_pmf,
                support,
                probabilities,
                s_lim=threshold,
                baseline_scenario=f"run {baseline.run_id} (undamaged)",
                damaged_scenario=(
                    f"run {damaged.run_id}, nodes "
                    f"{list(damaged.damage_node_ids)}"
                ),
                c=c,
                coherence_mode=coherence_mode,
                mc_tolerance=mc_tolerance,
            )
            if result.p0 != p0:
                raise AssertionError("Threshold-specific baseline was not preserved.")
            evaluations.append(
                DamagedRunRobustness(
                    run=damaged,
                    label=label,
                    result=result,
                )
            )

        macrostates.append(
            MacrostateRobustnessAnalysis(
                s_lim=threshold,
                adverse_macrostate=f"N_f >= {threshold}",
                p0=p0,
                thresholds=threshold_result,
                damaged_runs=tuple(evaluations),
            )
        )

    return RobustnessCampaignResult(
        baseline_run=baseline,
        damaged_runs=damaged_metadata,
        coherence_mode=coherence_mode,
        mc_tolerance=float(mc_tolerance),
        c=float(c),
        macrostates=tuple(macrostates),
    )


def _add_robustness_regions(
    ax: plt.Axes,
    thresholds: EntropyRobustnessThresholds,
) -> None:
    if not thresholds.identified:
        ax.text(
            0.02,
            0.98,
            f"Thresholds not identified:\n{thresholds.reason}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        return

    assert thresholds.pi_minus is not None
    assert thresholds.pi_plus is not None
    assert thresholds.I_minus is not None
    assert thresholds.I_plus is not None
    xmin, xmax = float(thresholds.pi_grid[0]), float(thresholds.pi_grid[-1])
    x0, x1 = thresholds.pi_minus, thresholds.pi_plus
    y0, y1 = thresholds.I_minus, thresholds.I_plus
    if not (xmin <= x0 < x1 <= xmax and 0.0 <= y0 < y1 <= 1.0):
        ax.text(
            0.02,
            0.98,
            "Derivative roots do not define ordered robustness regions.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
        )
        return

    low = (0.95, 0.95, 0.95)
    medium = (0.7, 0.7, 0.7)
    high = (0.5, 0.5, 0.5)
    ax.add_patch(
        mpatches.Rectangle(
            (xmin, 0.0),
            x0 - xmin,
            y0,
            facecolor=low,
            alpha=0.4,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (xmin, y0),
            x1 - xmin,
            y1 - y0,
            facecolor=medium,
            alpha=0.4,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (x0, 0.0),
            x1 - x0,
            y0,
            facecolor=medium,
            alpha=0.4,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (xmin, y1),
            xmax - xmin,
            1.0 - y1,
            facecolor=high,
            alpha=0.4,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (x1, 0.0),
            xmax - x1,
            y1,
            facecolor=high,
            alpha=0.4,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.annotate(
        "Low",
        ((xmin + x0) / 2.0, y0 / 2.0),
        ha="center",
        va="center",
        fontsize="small",
    )
    ax.annotate(
        "Medium",
        ((xmin + x1) / 2.0, (y0 + y1) / 2.0),
        ha="center",
        va="center",
        fontsize="small",
    )
    ax.annotate(
        "High",
        ((xmin + x1) / 2.0, min((y1 + 1.0) / 2.0, 0.92)),
        ha="center",
        va="center",
        fontsize="small",
    )


def _plot_campaign(
    campaign: RobustnessCampaignResult,
    *,
    figs_dir: Path,
    filename: str,
    formats: Sequence[str],
    plot_mode: PlotMode,
) -> tuple[Path, ...]:
    count = len(campaign.macrostates)
    columns = min(2, count)
    rows = int(math.ceil(count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=tuple(plt.figaspect(0.8) * 1.25),
        dpi=80,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for index, macrostate in enumerate(campaign.macrostates):
        ax = flat_axes[index]
        thresholds = macrostate.thresholds
        _add_robustness_regions(ax, thresholds)
        finite = np.isfinite(thresholds.I_grid)
        ax.plot(
            thresholds.pi_grid[finite],
            thresholds.I_grid[finite],
            "k:",
            linewidth=1.2,
        )

        diagnostics: list[str] = []
        for scenario_index, damaged in enumerate(macrostate.damaged_runs):
            evaluation = damaged.result.robustness
            if evaluation.value is None:
                diagnostics.append(
                    f"{damaged.label}: {evaluation.status}"
                )
                continue
            pi_value = float(norm.isf(damaged.result.pj))
            plot_pi = float(
                np.clip(
                    pi_value,
                    thresholds.pi_grid[0],
                    thresholds.pi_grid[-1],
                )
            )
            color = _batlow(90 + 55 * (scenario_index % 3))
            ax.scatter(
                plot_pi,
                evaluation.value,
                s=70,
                zorder=10,
                color=color,
                alpha=0.9,
                edgecolor=(0.3, 0.3, 0.3),
                label=damaged.label,
            )

            # Label the adverse macrostate beside the first (green) point.
            if scenario_index == 0:
                ax.annotate(
                    rf"$N_{{\mathrm{{f}}}}\geq {macrostate.s_lim}$",
                    xy=(plot_pi + 0.35, evaluation.value),
                    fontsize=16,
                    ha="left",
                    va="center",
                )
            if not math.isfinite(pi_value):
                diagnostics.append(
                    f"{damaged.label}: pi={pi_value:g} shown at boundary"
                )
            if evaluation.status == "within_monte_carlo_tolerance":
                diagnostics.append(
                    f"{damaged.label}: coherence accepted within tolerance"
                )

        if diagnostics:
            ax.text(
                0.02,
                0.02,
                "\n".join(diagnostics),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize="x-small",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
        ax.set_xlim(
            float(thresholds.pi_grid[0]),
            float(thresholds.pi_grid[-1]),
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$\pi=-\Phi^{-1}(p_j)$")
        ax.set_ylabel(r"$I_{\mathrm{rob}}$")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(
                unique.values(),
                unique.keys(),
                loc="lower right",
                frameon=True,
                fontsize="small",
            )

    for ax in flat_axes[count:]:
        ax.set_visible(False)
    fig.tight_layout()

    figs_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in formats:
        suffix = str(extension).lstrip(".")
        if not suffix:
            raise ValueError("Figure formats must be non-empty extensions.")
        output = figs_dir / f"{filename}.{suffix}"
        save_figure(fig, output, mode=plot_mode)
        outputs.append(output)
    plt.close(fig)
    if not outputs:
        raise ValueError("At least one output figure format is required.")
    return tuple(outputs)


def plot_entropy_robustness_index(
    conn: sqlite3.Connection,
    *,
    baseline_run_id: int,
    damaged_run_ids: Sequence[int],
    s_lim: int | str | Sequence[int],
    figs_dir: str | Path = "output/figs",
    filename: str = "entropy_robustness_index",
    formats: Sequence[str] = ("pdf",),
    pi_min: float = -3.0,
    pi_max: float = 3.0,
    n_grid: int = 1001,
    scenario_labels: dict[int, str] | None = None,
    coherence_mode: CoherenceMode = "strict",
    mc_tolerance: float = 0.0,
    c: float = 1.0,
    plot_mode: PlotMode = "standard",
) -> RobustnessCampaignResult:
    """Analyze and plot explicit notional-damage robustness macrostates.

    Multiple ``S_lim`` values produce separate panels because every threshold
    has a distinct undamaged probability ``p0`` and therefore a distinct
    reference curve and pair of derivative-based robustness thresholds.
    """

    campaign = analyze_entropy_robustness_campaign(
        conn,
        baseline_run_id=baseline_run_id,
        damaged_run_ids=damaged_run_ids,
        s_lim=s_lim,
        scenario_labels=scenario_labels,
        coherence_mode=coherence_mode,
        mc_tolerance=mc_tolerance,
        c=c,
        pi_min=pi_min,
        pi_max=pi_max,
        n_grid=n_grid,
    )
    with plotting_context(plot_mode):
        outputs = _plot_campaign(
            campaign,
            figs_dir=Path(figs_dir),
            filename=filename,
            formats=formats,
            plot_mode=plot_mode,
        )
    return replace(campaign, output_paths=outputs)
