# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:27:36 2025

@author: caol
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, Tuple

import logging
import sqlite3

import matplotlib.pyplot as plt

import numpy as np

from .plot_style import PlotMode, plotting_context, save_figure
from .percolation_threshold import (
    AxisScanMetadata,
    PercolationThresholdResult,
    axis_scan_metadata_from_mapping,
)

try:  # pragma: no cover - purely aesthetic
    import cmcrameri.cm as cm
    _HAS_CMCRAMERI = True
except Exception:  # pragma: no cover - optional dependency
    cm = None
    _HAS_CMCRAMERI = False
    
def _batlow(level: int | float) -> str | None:
    """Return a color from the batlow colormap if available, else None.

    Using None lets Matplotlib pick a default color without failing.
    """
    if not _HAS_CMCRAMERI:
        return None
    # cm.batlow expects values in [0, 255]; we accept a level in that range.
    return cm.batlow(int(level))

logger = logging.getLogger(__name__)

SweepMode1D = Literal[
    "1d_xyz", "1d_xy", "1d_xz", "1d_yz", "1d_x", "1d_y", "1d_z"
]
SweepMode2D = Literal["2d_xy", "2d_xz", "2d_yz"]


def load_percolation_multi(conn: sqlite3.Connection, run_id: int) -> Optional[np.ndarray]:
    """Load percolation_multi rows for *run_id*.

    Returns
    -------
    data : ndarray or None
        2D ndarray with columns:

        [pe_x, pe_y, pe_z, p_any, p_one, p_full, mean_nf]

        For backward compatibility with older results DBs that do not include
        ``mean_nf``, the last column is returned as NaN.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(percolation_multi);").fetchall()]
    has_mean = "mean_nf" in cols

    sel = (
        "SELECT pe_x, pe_y, pe_z, p_any, p_one, p_full"
        + (", mean_nf" if has_mean else "")
        + " FROM percolation_multi WHERE run_id = ?"
        + " ORDER BY pe_x, pe_y, pe_z;"
    )
    rows = conn.execute(sel, (run_id,)).fetchall()
    if not rows:
        logger.info("No percolation_multi data for run_id %s", run_id)
        return None

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2:
        raise ValueError("percolation_multi has unexpected shape: %r" % (arr.shape,))

    if has_mean:
        if arr.shape[1] != 7:
            raise ValueError("percolation_multi has unexpected shape: %r" % (arr.shape,))
        return arr

    # Old DB: append NaN mean_nf column.
    if arr.shape[1] != 6:
        raise ValueError("percolation_multi has unexpected shape: %r" % (arr.shape,))
    nan_col = np.full((arr.shape[0], 1), np.nan, dtype=float)
    return np.hstack([arr, nan_col])


def load_percolation_binomial_data(
    conn: sqlite3.Connection,
    run_id: int,
    p_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load complete-collapse successes and trials in scan-row order.

    Current databases store integer counts directly.  For older databases,
    counts are reconstructed exactly from ``runs.samples`` and the stored
    Monte Carlo proportion.
    """
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(percolation_multi);").fetchall()
    }
    if {"count_full", "n_samples"} <= cols:
        rows = conn.execute(
            """
            SELECT count_full, n_samples
            FROM percolation_multi
            WHERE run_id = ?
            ORDER BY pe_x, pe_y, pe_z;
            """,
            (run_id,),
        ).fetchall()
        if len(rows) != p_full.size:
            raise ValueError("Percolation count rows do not match probability rows.")
        values = np.asarray(rows, dtype=np.int64)
        successes, trials = values[:, 0], values[:, 1]
    else:
        run_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(runs);").fetchall()
        }
        if "samples" not in run_columns:
            raise ValueError(
                "The results database does not retain percolation sample counts."
            )
        row = conn.execute(
            "SELECT samples FROM runs WHERE id = ?;", (run_id,)
        ).fetchone()
        if row is None or row[0] is None or int(row[0]) <= 0:
            raise ValueError(f"Run {run_id} has no valid Monte Carlo sample count.")
        trials = np.full(p_full.size, int(row[0]), dtype=np.int64)
        successes = np.rint(p_full * trials).astype(np.int64)

    tolerance = 0.5000001 / trials + 1e-12
    if np.any(np.abs(p_full - successes / trials) > tolerance):
        raise ValueError("Stored p_full is inconsistent with complete-collapse counts.")
    return successes, trials


def load_percolation_pmf_counts(
    conn: sqlite3.Connection,
    run_id: int,
    percolation_data: np.ndarray,
) -> np.ndarray:
    """Load and validate complete integer scan PMFs in aggregate-row order.

    ``percolation_multi`` contains plotting summaries.  Risk-equivalence uses
    the authoritative ``percolation_pmf`` histogram and verifies that each
    histogram reproduces those summaries exactly within storage tolerance.
    """

    table_exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name='percolation_pmf';
        """
    ).fetchone()
    if table_exists is None:
        raise ValueError(
            "The results database does not retain percolation scan PMFs. "
            "Re-run the percolation scan with the current output schema."
        )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(percolation_pmf);").fetchall()
    }
    required = {
        "run_id",
        "pe_x",
        "pe_y",
        "pe_z",
        "n_failed",
        "count",
        "p",
        "n_samples",
    }
    if not required <= columns:
        raise ValueError(
            "percolation_pmf does not use the current risk-equivalence schema."
        )

    rows = conn.execute(
        """
        SELECT pe_x, pe_y, pe_z, n_failed, count, p, n_samples
        FROM percolation_pmf
        WHERE run_id = ?
        ORDER BY pe_x, pe_y, pe_z, n_failed;
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        raise ValueError(
            f"Run {run_id} has no retained percolation scan PMFs."
        )

    grouped: dict[
        tuple[float, float, float],
        list[tuple[int, int, float, int]],
    ] = {}
    for pe_x, pe_y, pe_z, n_failed, count, probability, n_samples in rows:
        key = (float(pe_x), float(pe_y), float(pe_z))
        grouped.setdefault(key, []).append(
            (
                int(n_failed),
                int(count),
                float(probability),
                int(n_samples),
            )
        )

    expected_coordinates = np.asarray(percolation_data[:, :3], dtype=float)
    ordered_coordinates = np.asarray(sorted(grouped), dtype=float)
    if (
        ordered_coordinates.shape != expected_coordinates.shape
        or not np.allclose(
            ordered_coordinates,
            expected_coordinates,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Percolation PMF scan points do not match percolation_multi."
        )

    histograms: list[np.ndarray] = []
    expected_support: Optional[np.ndarray] = None
    for index, key in enumerate(sorted(grouped)):
        group = grouped[key]
        support = np.asarray([row[0] for row in group], dtype=np.int64)
        counts = np.asarray([row[1] for row in group], dtype=np.int64)
        probabilities = np.asarray([row[2] for row in group], dtype=float)
        sample_counts = np.asarray([row[3] for row in group], dtype=np.int64)
        if expected_support is None:
            expected_support = np.arange(support.size, dtype=np.int64)
        if not np.array_equal(support, expected_support):
            raise ValueError(
                "Every percolation scan PMF must retain the common complete "
                "support N_f=0,...,|N|."
            )
        if (
            np.any(counts < 0)
            or np.any(sample_counts <= 0)
            or np.any(sample_counts != sample_counts[0])
            or int(np.sum(counts)) != int(sample_counts[0])
        ):
            raise ValueError(
                "A retained percolation PMF has inconsistent integer counts."
            )
        trials = int(sample_counts[0])
        if np.any(
            np.abs(probabilities - counts / trials) > 1.0e-12
        ):
            raise ValueError(
                "A retained percolation PMF probability is inconsistent with "
                "its integer count."
            )

        p_any = 1.0 - counts[0] / trials
        p_one = counts[1] / trials
        p_full = counts[-1] / trials
        mean_nf = float(np.dot(support, counts) / trials)
        aggregates = np.asarray(
            [p_any, p_one, p_full, mean_nf], dtype=float
        )
        stored = np.asarray(percolation_data[index, 3:7], dtype=float)
        finite = np.isfinite(stored)
        if np.any(
            np.abs(aggregates[finite] - stored[finite]) > 1.0e-12
        ):
            raise ValueError(
                "A retained percolation PMF is inconsistent with "
                "percolation_multi summaries."
            )
        histograms.append(counts)

    return np.vstack(histograms)


_MODE_VARIED_AXES = {
    "1d_xyz": ("x", "y", "z"),
    "1d_xy": ("x", "y"),
    "1d_xz": ("x", "z"),
    "1d_yz": ("y", "z"),
    "1d_x": ("x",),
    "1d_y": ("y",),
    "1d_z": ("z",),
}


def load_axis_scan_metadata(
    conn: sqlite3.Connection,
    run_id: int,
    mode: str,
    data: np.ndarray,
) -> AxisScanMetadata:
    """Load explicit varied/fixed axis semantics, with legacy inference."""
    row = conn.execute("SELECT notes FROM runs WHERE id = ?;", (run_id,)).fetchone()
    if row and row[0]:
        try:
            notes = json.loads(row[0])
            value = notes.get("percolation_scan_metadata")
            if isinstance(value, dict):
                return axis_scan_metadata_from_mapping(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning(
                "Run %s has invalid percolation scan metadata; inferring it.",
                run_id,
            )

    varied = _MODE_VARIED_AXES.get(mode)
    if varied is None:
        raise ValueError(f"Cannot infer axis metadata for sweep mode {mode!r}.")
    fixed = tuple(axis for axis in ("x", "y", "z") if axis not in varied)
    fixed_values: list[tuple[str, Optional[float]]] = []
    axis_columns = {"x": 0, "y": 1, "z": 2}
    for axis in fixed:
        values = np.unique(data[:, axis_columns[axis]])
        value = (
            float(values[0])
            if values.size == 1 and 0.0 <= float(values[0]) <= 1.0
            else None
        )
        fixed_values.append((axis, value))
    return AxisScanMetadata(
        scan_spec=None,
        mode=mode,
        varied_axes=tuple(varied),
        fixed_axes=fixed,
        varied_axis_groups=(tuple(varied),),
        fixed_axis_values=tuple(fixed_values),
        source="inferred",
    )


def _unique_tol(vals: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Return unique values with tolerance *tol* (for floats from DB)."""
    if vals.size == 0:
        return vals
    # sort and merge near-equal values
    s = np.sort(vals)
    uniq = [s[0]]
    for v in s[1:]:
        if abs(v - uniq[-1]) > tol:
            uniq.append(v)
    return np.array(uniq)


def detect_sweep_mode(
    data: np.ndarray,
) -> Tuple[str, np.ndarray | Tuple[np.ndarray, np.ndarray] | None]:
    """Detect whether *data* corresponds to a 1D or 2D sweep.

    Parameters
    ----------
    data : ndarray
        Shape (N, >=6): [pe_x, pe_y, pe_z, p_any, p_one, p_full, ...].

        If present, the 7th column is interpreted as ``mean_nf`` and is
        ignored by the sweep-mode detection.

    Returns
    -------
    mode : str
        One of
        - "1d_xyz", "1d_xy", "1d_xz", "1d_yz", "1d_x", "1d_y", "1d_z"
        - "2d_xy", "2d_xz", "2d_yz"
        - "multi_skip" (true 3D grid or unsupported pattern)
    sweep_info : array-like or tuple or None
        For 1D modes: array of t values (parameter along sweep).
        For 2D modes: (t1_vals, t2_vals) arrays defining the grid.
        For "multi_skip": None.
    """
    pe_x, pe_y, pe_z = data[:, 0], data[:, 1], data[:, 2]
    ux, uy, uz = _unique_tol(pe_x), _unique_tol(pe_y), _unique_tol(pe_z)
    vary_x = ux.size > 1
    vary_y = uy.size > 1
    vary_z = uz.size > 1

    # Helper for approximate equality
    def _all_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
        return np.all(np.abs(a - b) <= tol)

    # 1) Fully coupled xyz: pe_x == pe_y == pe_z, all varying together
    if vary_x and vary_y and vary_z and _all_close(pe_x, pe_y) and _all_close(pe_x, pe_z):
        t = _unique_tol(pe_x)
        return "1d_xyz", t

    # 2) Group-coupled pairs xy / xz / yz (two axes equal, third fixed)
    # xy
    if _all_close(pe_x, pe_y) and uz.size == 1 and _unique_tol(pe_x).size > 1:
        return "1d_xy", _unique_tol(pe_x)
    # xz
    if _all_close(pe_x, pe_z) and uy.size == 1 and _unique_tol(pe_x).size > 1:
        return "1d_xz", _unique_tol(pe_x)
    # yz
    if _all_close(pe_y, pe_z) and ux.size == 1 and _unique_tol(pe_y).size > 1:
        return "1d_yz", _unique_tol(pe_y)

    # 3) Single-axis sweeps
    if vary_x and not vary_y and not vary_z:
        return "1d_x", ux
    if vary_y and not vary_x and not vary_z:
        return "1d_y", uy
    if vary_z and not vary_x and not vary_y:
        return "1d_z", uz

    # 4) 2D grids (two varying axes, third fixed)
    if vary_x and vary_y and not vary_z:
        return "2d_xy", (ux, uy)
    if vary_x and vary_z and not vary_y:
        return "2d_xz", (ux, uz)
    if vary_y and vary_z and not vary_x:
        return "2d_yz", (uy, uz)

    # 5) True 3D or irregular pattern → skip plotting
    return "multi_skip", None


def extract_1d_curves(
    data: np.ndarray, mode: SweepMode1D | str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract 1D curves (t, p_any, p_one, p_full) for a given 1D *mode*.

    Scan rows must already be strictly ordered and unique.  Duplicate or
    out-of-order values are rejected rather than silently averaged.
    """
    pe_x, pe_y, pe_z = data[:, 0], data[:, 1], data[:, 2]
    p_any, p_one, p_full = data[:, 3], data[:, 4], data[:, 5]

    if mode == "1d_xyz":
        t = pe_x
    elif mode in ("1d_xy", "1d_xz", "1d_x"):
        t = pe_x
    elif mode in ("1d_yz", "1d_y"):
        t = pe_y
    elif mode == "1d_z":
        t = pe_z
    else:
        raise ValueError(f"Unsupported 1D mode: {mode!r}")

    arrays = (t, p_any, p_one, p_full)
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("Percolation scan curves must contain only finite values.")
    if t.size > 1:
        differences = np.diff(t)
        if np.any(differences == 0.0):
            raise ValueError("Percolation scan contains duplicate grid values.")
        if np.any(differences < 0.0):
            raise ValueError("Percolation scan rows are not strictly ordered.")
    return tuple(np.asarray(values, dtype=float) for values in arrays)

def make_1d_plot(
    t: np.ndarray,
    p_any: np.ndarray,
    p_one: np.ndarray,
    p_full: np.ndarray,
    out_path: str | Path,
    threshold_result: Optional[PercolationThresholdResult] = None,
    *,
    plot_mode: PlotMode = "standard",
) -> None:
    """Plot 1D percolation curves and save as an image.

    Aesthetics loosely mirror `percolation_scan.py`:
    - batlow colormap (if available),
    - vertical + horizontal lines at the critical point,
    - black marker at an identified threshold,
    - an explicit message when no threshold is identifiable.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = (
        threshold_result.scan_metadata
        if threshold_result is not None
        else None
    )

    with plotting_context(plot_mode):
        fig, ax = plt.subplots(figsize=tuple(plt.figaspect(0.8) * 1.25), dpi=80)
        try:
            # Colors (batlow if available, otherwise Matplotlib defaults)
            col_any = _batlow(100)
            col_one = _batlow(150)
            col_full = _batlow(200)

            # Main curves
            ax.plot(
                t,
                p_any,
                label=r"$\mathrm{P}(N_{\mathrm{f}}\geq 1)$",
                color=col_any,
                linewidth=1.8,
            )
            ax.plot(
                t,
                p_one,
                label=r"$\mathrm{P}(N_{\mathrm{f}}=1)$",
                color=col_one,
                linewidth=1.8,
            )
            ax.plot(
                t,
                p_full,
                label=r"$\mathrm{P}(N_{\mathrm{f}}=|\mathcal{N}|)$",
                color=col_full,
                linewidth=1.8,
            )

            # Threshold-style annotation: vertical + horizontal lines at the critical point.
            if (
                threshold_result is not None
                and threshold_result.threshold is not None
                and t.size > 0
            ):
                tcrit = float(threshold_result.threshold)
                pcrit = float(np.interp(tcrit, t, p_full))

                # Vertical + horizontal "L" at the threshold.
                ax.plot([tcrit, tcrit], [0.0, pcrit], "k:", linewidth=1.0)
                ax.plot([0.0, tcrit], [pcrit, pcrit], "k:", linewidth=1.0)
                ax.scatter([tcrit], [pcrit], color="k", alpha=0.6, zorder=5)

                # Annotation rendering follows the selected plotting mode.
                ax.annotate(
                    rf"$p_{{\mathrm{{f}}}}^{{n=|\mathcal{{N}}|}}={pcrit:.4f}$",
                    xy=(tcrit, pcrit),
                    xytext=(tcrit, pcrit * 1.03),
                    ha="center",
                    va="bottom",
                )
                ax.annotate(
                    rf"$p_{{\mathrm{{e,crit}}}}={tcrit:.2f}$",
                    xy=(tcrit, pcrit * 0.5),
                    xytext=(tcrit * 1.02, pcrit * 0.5),
                    ha="left",
                    va="center",
                )
            elif threshold_result is not None:
                ax.text(
                    0.98,
                    0.04,
                    "No identifiable threshold",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize="small",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                        "pad": 1.5,
                    },
                )

            # Axes and limits
            t_min, t_max = float(np.min(t)), float(np.max(t))
            ax.set_xlim(t_min, t_max)
            # If this is a proper probability axis, use [0,1] + 0.1 spacing.
            if t_min >= -1e-6 and t_max <= 1.0 + 1e-6:
                ax.set_xlim(0.0, 1.0)
                ax.set_xticks(np.linspace(0.0, 1.0, 11))
                if metadata is not None and metadata.varied_axes:
                    varied_labels = [
                        rf"p_{{\mathrm{{e}},{axis}}}"
                        for axis in metadata.varied_axes
                    ]
                    ax.set_xlabel("$" + "=".join(varied_labels) + "$")
                else:
                    ax.set_xlabel(r"$p_{\mathrm{e}}$")
            else:
                ax.set_xlabel(r"$t$")

            ax.set_ylim(0.0, max(1e-12, float(np.max(p_any)) * 1.05))
            ax.set_ylabel(r"$\mathrm{P}(N_{\mathrm{f}})$")

            if metadata is not None and metadata.fixed_axes:
                fixed_lookup = dict(metadata.fixed_axis_values)
                labels = []
                for axis in metadata.fixed_axes:
                    value = fixed_lookup.get(axis)
                    labels.append(
                        rf"$p_{{\mathrm{{e}},{axis}}}={value:g}$"
                        if value is not None
                        else rf"$p_{{\mathrm{{e}},{axis}}}$ fixed"
                    )
                ax.text(
                    0.98,
                    0.98,
                    ", ".join(labels),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize="small",
                )

            ax.legend(loc="upper left", frameon=False)
            ax.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.7)
            save_figure(fig, out_path, mode=plot_mode)
        finally:
            plt.close(fig)



def make_2d_heatmap(
    data: np.ndarray,
    mode: SweepMode2D | str,
    out_path: str | Path,
    *,
    plot_mode: PlotMode = "standard",
) -> None:
    """Plot a 2D heatmap of p_any over a 2D percolation grid.

    Uses batlow colormap if available to stay close to percolation_scan.py.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pe_x, pe_y, pe_z = data[:, 0], data[:, 1], data[:, 2]
    p_any = data[:, 3]

    # Determine axes for the chosen mode
    if mode == "2d_xy":
        a_vals = _unique_tol(pe_x)
        b_vals = _unique_tol(pe_y)
        a = pe_x
        b = pe_y
        x_label, y_label = r"$p_{\mathrm{e},x}$", r"$p_{\mathrm{e},y}$"
    elif mode == "2d_xz":
        a_vals = _unique_tol(pe_x)
        b_vals = _unique_tol(pe_z)
        a = pe_x
        b = pe_z
        x_label, y_label = r"$p_{\mathrm{e},x}$", r"$p_{\mathrm{e},z}$"
    elif mode == "2d_yz":
        a_vals = _unique_tol(pe_y)
        b_vals = _unique_tol(pe_z)
        a = pe_y
        b = pe_z
        x_label, y_label = r"$p_{\mathrm{e},y}$", r"$p_{\mathrm{e},z}$"
    else:
        raise ValueError(f"Unsupported 2D mode: {mode!r}")

    na, nb = a_vals.size, b_vals.size
    grid = np.full((nb, na), np.nan, dtype=float)

    # Map coordinates to grid indices
    def find_index(val: float, arr: np.ndarray, tol: float = 1e-9) -> int:
        for i, v in enumerate(arr):
            if abs(val - v) <= tol:
                return i
        raise ValueError("value not found in axis array")

    for ai, bi, pi in zip(a, b, p_any):
        i = find_index(ai, a_vals)
        j = find_index(bi, b_vals)
        grid[j, i] = pi  # rows: b (y-axis), cols: a (x-axis)

    with plotting_context(plot_mode):
        fig, ax = plt.subplots(figsize=tuple(plt.figaspect(0.8) * 1.25), dpi=80)
        try:
            cmap = cm.batlow if _HAS_CMCRAMERI else None
            im = ax.imshow(
                grid,
                origin="lower",
                extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]],
                aspect="auto",
                cmap=cmap,
                vmin=0.0,
                vmax=np.nanmax(grid),
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(r"$\mathrm{P}(N_{\mathrm{f}}\geq 1)$")

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"$\mathrm{P}(N_{\mathrm{f}}\geq 1)$")
            save_figure(fig, out_path, mode=plot_mode)
        finally:
            plt.close(fig)
