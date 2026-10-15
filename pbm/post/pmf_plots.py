# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:27:24 2025

@author: caol
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import logging
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

from .percolation_plots import _batlow
from .plot_style import PlotMode, plotting_context, save_figure

logger = logging.getLogger(__name__)


def load_pmf(conn: sqlite3.Connection, run_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load PMF data for a given run.

    Returns
    -------
    values : ndarray
        Support values (e.g. number of failed nodes).
    probs : ndarray
        Corresponding probabilities that sum (approximately) to 1.
    """
    cur = conn.execute(
        "SELECT n_failed, p FROM pmf WHERE run_id = ? ORDER BY n_failed ASC;",
        (run_id,),
    )
    rows = cur.fetchall()
    if not rows:
        logger.info("No PMF data found for run_id %s", run_id)
        return np.array([], dtype=int), np.array([], dtype=float)

    vals = np.array([r[0] for r in rows], dtype=int)
    probs = np.array([r[1] for r in rows], dtype=float)
    return vals, probs


def make_pmf_plot(
    values: np.ndarray,
    probs: np.ndarray,
    out_path: str | Path,
    *,
    plot_mode: PlotMode = "standard",
) -> None:
    """Create a PMF plot with aesthetics similar to percolation_scan.py.

    Bars use batlow color if available, with light grid and mathtext labels.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with plotting_context(plot_mode):
        fig, ax = plt.subplots(figsize=tuple(plt.figaspect(0.8) * 1.25), dpi=80)
        try:
            color = _batlow(150)
            ax.bar(
                values,
                probs,
                align="center",
                width=0.8,
                color=color,
                edgecolor="black",
                linewidth=0.8,
            )

            ax.set_xlabel(r"$N_{\mathrm{f}}$")
            ax.set_ylabel(r"$\mathrm{P}(N_{\mathrm{f}})$")

            if values.size > 0:
                ax.set_xlim(values.min() - 0.5, values.max() + 0.5)
            ax.set_ylim(0.0, max(1e-12, float(np.max(probs)) * 1.05))

            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
            save_figure(fig, out_path, mode=plot_mode)
        finally:
            plt.close(fig)