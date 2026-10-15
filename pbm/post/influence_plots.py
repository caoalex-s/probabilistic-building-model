# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:27:52 2025

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


def load_node_influence(
    conn: sqlite3.Connection, run_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load node influence data for a given run.

    Returns
    -------
    node_ids : ndarray of ints
    influence : ndarray of floats
    """
    cur = conn.execute(
        "SELECT node_id, influence FROM node_influence WHERE run_id = ? ORDER BY node_id ASC;",
        (run_id,),
    )
    rows = cur.fetchall()
    if not rows:
        logger.info("No node_influence data for run_id %s", run_id)
        return np.array([], dtype=int), np.array([], dtype=float)

    node_ids = np.array([r[0] for r in rows], dtype=int)
    influences = np.array([r[1] for r in rows], dtype=float)
    return node_ids, influences


def make_influence_plot(
    node_ids: np.ndarray,
    influences: np.ndarray,
    out_path: str | Path,
    *,
    plot_mode: PlotMode = "standard",
) -> None:
    """Plot influence values sorted in descending order and save as a figure.

    Styling is loosely aligned with the percolation plots (batlow color, thin
    lines, light grid).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by influence descending
    order = np.argsort(influences)[::-1]
    infl_sorted = influences[order]

    with plotting_context(plot_mode):
        fig, ax = plt.subplots(figsize=tuple(plt.figaspect(0.8) * 1.25), dpi=80)
        try:
            color = _batlow(150)
            ax.plot(
                np.arange(len(infl_sorted)),
                infl_sorted,
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=4,
                color=color,
            )

            ax.set_xlabel(r"Rank (sorted by influence)")
            ax.set_ylabel(r"Influence")
            ax.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.7)
            save_figure(fig, out_path, mode=plot_mode)
        finally:
            plt.close(fig)
