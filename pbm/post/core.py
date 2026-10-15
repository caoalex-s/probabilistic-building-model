# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:27:09 2025

@author: caol
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import logging
import json
import sqlite3

from .utils import ensure_figs_dir
from . import pmf_plots, percolation_plots, influence_plots
from .percolation_threshold import estimate_percolation_threshold
from .plot_style import PlotMode, normalize_plot_mode
from .risk_equivalence import analyze_risk_equivalence

logger = logging.getLogger(__name__)


def _load_regular_lattice_shape(
    conn: sqlite3.Connection,
    run_id: int,
) -> tuple[int, int, int] | None:
    row = conn.execute("SELECT notes FROM runs WHERE id = ?;", (run_id,)).fetchone()
    if row is None or not row[0]:
        return None
    try:
        notes = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None
    value = notes.get("regular_lattice_shape")
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            for item in value
        )
    ):
        raise ValueError(
            "runs.notes contains an invalid regular_lattice_shape."
        )
    return (int(value[0]), int(value[1]), int(value[2]))


def run_postprocessing(
    conn: sqlite3.Connection,
    run_id: int,
    make_plots: bool = True,
    compute_risk_equivalence: bool = False,
    figs_root: str | Path = "output/figs",
    plot_mode: PlotMode = "standard",
    risk_equivalence_pe_crit: float | None = None,
    lattice_shape: tuple[int, int, int] | None = None,
    risk_equivalence_confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Run full post-processing for a single run_id.

    Parameters
    ----------
    conn : sqlite3.Connection
        Connection to ``output/results.db`` (or equivalent).
    run_id : int
        Run identifier in the ``runs`` table.
    make_plots : bool, default True
        If False, all computations are performed but no image files are written.
    compute_risk_equivalence : bool, default False
        If True, compute the paper's numerical and analytical risk-equivalence
        estimates from the fully coupled scan PMFs.
    figs_root : str or Path, default "output/figs"
        Root directory for figure output. Each run gets its own subdirectory.
    plot_mode : {"standard", "publication"}, default "standard"
        Plot rendering/export mode. Publication mode uses external LaTeX and
        the PGF backend for PDF/PGF output.
    risk_equivalence_pe_crit : float or None, default None
        Upper bound for risk equivalence. If omitted, an identified
        percolation threshold from the same scan is used.
    lattice_shape : tuple[int, int, int] or None, default None
        Regular ``(nx, ny, nz)`` shape used by the analytical approximation.
        If omitted, the shape recorded in ``runs.notes`` is used.
    risk_equivalence_confidence_level : float, default 0.95
        Simultaneous confidence level for numerical Monte Carlo diagnostics.

    Returns
    -------
    summary : dict
        Dictionary containing scalar summaries plus the structured
        percolation-threshold result and axis metadata.
    """
    plot_mode = normalize_plot_mode(plot_mode)

    # Ensure the run exists early to give a clear error message.
    cur = conn.execute("SELECT COUNT(*) FROM runs WHERE id = ?;", (run_id,))
    if cur.fetchone()[0] == 0:
        raise ValueError(f"run_id {run_id} does not exist in runs table")

    figs_dir = ensure_figs_dir(figs_root, run_id) if make_plots else Path(figs_root)

    summary: Dict[str, Any] = {"run_id": run_id}

    # --- PMF ------------------------------------------------------------------
    try:
        pmf_vals, pmf_probs = pmf_plots.load_pmf(conn, run_id)
        summary["pmf_support"] = pmf_vals
        if make_plots and pmf_vals.size > 0:
            pmf_plots.make_pmf_plot(
                pmf_vals,
                pmf_probs,
                figs_dir / f"pmf_{run_id}.pdf",
                plot_mode=plot_mode,
            )
    except Exception as exc:  # pragma: no cover - defensive; tests may focus on success path
        logger.exception("PMF post-processing failed: %s", exc)

    # --- Percolation / grouped scans -----------------------------------------
    perco_data = percolation_plots.load_percolation_multi(conn, run_id)
    if perco_data is not None:
        mode, sweep_info = percolation_plots.detect_sweep_mode(perco_data)
        summary["percolation_mode"] = mode
        if compute_risk_equivalence and mode != "1d_xyz":
            raise ValueError(
                "Risk equivalence requires a one-dimensional fully coupled "
                "xyz percolation scan."
            )
        if mode.startswith("1d_"):
            t, pany, pone, pfull = percolation_plots.extract_1d_curves(
                perco_data, mode
            )
            metadata = percolation_plots.load_axis_scan_metadata(
                conn, run_id, mode, perco_data
            )
            successes, trials = percolation_plots.load_percolation_binomial_data(
                conn, run_id, pfull
            )
            threshold_result = estimate_percolation_threshold(
                t,
                pfull,
                trials=trials,
                successes=successes,
                scan_metadata=metadata,
            )
            summary["percolation_scan_metadata"] = metadata
            summary["percolation_threshold_result"] = threshold_result
            summary["percolation_threshold"] = threshold_result.threshold
            summary["percolation_threshold_status"] = (
                "identified"
                if threshold_result.identified
                else "no identifiable threshold"
            )
            summary["percolation_threshold_reason"] = threshold_result.reason

            if compute_risk_equivalence:
                if mode != "1d_xyz":
                    raise ValueError(
                        "Risk equivalence requires a fully coupled xyz scan "
                        "with homogeneous p_e."
                    )
                pe_crit = (
                    float(risk_equivalence_pe_crit)
                    if risk_equivalence_pe_crit is not None
                    else threshold_result.threshold
                )
                if pe_crit is None:
                    raise ValueError(
                        "Risk equivalence requires either an identified "
                        "percolation threshold or an explicit pe_crit."
                    )
                resolved_shape = (
                    lattice_shape
                    if lattice_shape is not None
                    else _load_regular_lattice_shape(conn, run_id)
                )
                if resolved_shape is None:
                    raise ValueError(
                        "The analytical risk-equivalence approximation requires "
                        "a regular lattice shape. Pass (nx, ny, nz) explicitly "
                        "or re-run the simulation to record it."
                    )
                pmf_counts = percolation_plots.load_percolation_pmf_counts(
                    conn, run_id, perco_data
                )
                risk_result = analyze_risk_equivalence(
                    t,
                    pmf_counts,
                    pe_crit=pe_crit,
                    lattice_shape=resolved_shape,
                    confidence_level=risk_equivalence_confidence_level,
                )
                summary["risk_equivalence_result"] = risk_result
                summary["risk_equivalence_numerical"] = (
                    risk_result.numerical.estimate
                )
                summary["risk_equivalence_analytical"] = (
                    risk_result.analytical.estimate
                )
                summary["risk_equivalence_relative_difference"] = (
                    risk_result.relative_difference
                )

            if make_plots and t.size > 0:
                percolation_plots.make_1d_plot(
                    t,
                    pany,
                    pone,
                    pfull,
                    figs_dir / f"percolation_{mode}_run_{run_id}.pdf",
                    threshold_result=threshold_result,
                    plot_mode=plot_mode,
                )
        elif mode.startswith("2d_") and make_plots:
            percolation_plots.make_2d_heatmap(
                perco_data,
                mode,
                figs_dir / f"percolation_heatmap_{mode}_run_{run_id}.pdf",
                plot_mode=plot_mode,
            )
    elif compute_risk_equivalence:
        raise ValueError(
            f"Run {run_id} has no percolation scan for risk equivalence."
        )

    # --- Node influence -------------------------------------------------------
    try:
        node_ids, influences = influence_plots.load_node_influence(conn, run_id)
        summary["max_influence"] = float(influences.max()) if influences.size else 0.0
        if influences.size:
            # store indices of top-3 nodes (by influence) for quick inspection
            top_idx = influences.argsort()[::-1][:3]
            summary["top_nodes"] = [int(node_ids[i]) for i in top_idx]
        if make_plots and influences.size > 0:
            influence_plots.make_influence_plot(
                node_ids,
                influences,
                figs_dir / f"influence_run_{run_id}.pdf",
                plot_mode=plot_mode,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Influence post-processing failed: %s", exc)

    return summary
