# -*- coding: utf-8 -*-
"""Central plotting configuration and figure export helpers.

Two plotting modes are supported:

``standard``
    Portable Matplotlib rendering for notebooks, exploratory analysis, tests,
    and raster output. Mathematical notation is rendered with Matplotlib's
    built-in mathtext engine; no LaTeX installation is required.

``publication``
    External-LaTeX rendering for manuscript figures. Figures are exported via
    Matplotlib's PGF backend on a per-save basis, avoiding the Agg/DVI path
    while leaving the interactive backend unchanged. Publication export is
    intentionally limited to PDF and PGF files.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, Mapping

import matplotlib as mpl
from matplotlib.figure import Figure

PlotMode = Literal["standard", "publication"]
PLOT_MODES: tuple[PlotMode, ...] = ("standard", "publication")


def normalize_plot_mode(mode: str | PlotMode) -> PlotMode:
    """Return a validated, normalized plotting mode."""
    normalized = str(mode).strip().lower()
    if normalized not in PLOT_MODES:
        choices = ", ".join(PLOT_MODES)
        raise ValueError(f"Unknown plot mode {mode!r}; expected one of: {choices}")
    return normalized  # type: ignore[return-value]


def plot_rc_params(mode: str | PlotMode) -> Mapping[str, object]:
    """Return Matplotlib rc parameters for *mode*.

    The publication configuration keeps the active interactive backend intact.
    The PGF backend is selected only by :func:`save_figure` when the file is
    written.
    """
    normalized = normalize_plot_mode(mode)

    common: dict[str, object] = {
        "font.family": "serif",
        "font.size": 16,
    }

    if normalized == "standard":
        return {
            **common,
            "text.usetex": False,
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }

    return {
        **common,
        "text.usetex": True,
        "font.serif": ["Times"],
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{mathptmx}",
    }


def configure_plotting(mode: str | PlotMode = "standard") -> PlotMode:
    """Apply a plotting mode globally and return its normalized name.

    Package plotting functions use :func:`plotting_context` instead, so they do
    not leak style changes into callers. This global helper is primarily for
    notebooks that create figures directly.
    """
    normalized = normalize_plot_mode(mode)
    mpl.rcParams.update(plot_rc_params(normalized))
    return normalized


@contextmanager
def plotting_context(mode: str | PlotMode = "standard") -> Iterator[PlotMode]:
    """Temporarily apply a plotting mode without changing the backend."""
    normalized = normalize_plot_mode(mode)
    with mpl.rc_context(rc=plot_rc_params(normalized)):
        yield normalized


def save_figure(
    fig: Figure,
    out_path: str | Path,
    *,
    mode: str | PlotMode = "standard",
    dpi: int | float | None = None,
    pad_inches: float = 0.05,
) -> Path:
    """Save *fig* according to the requested plotting mode.

    Standard mode uses the figure's current backend and calls
    :meth:`~matplotlib.figure.Figure.tight_layout` before saving, preserving the
    repository's previous interactive behavior. Publication mode deliberately
    skips ``tight_layout`` and exports through the PGF backend with a tight
    bounding box. Skipping ``tight_layout`` prevents the active Agg canvas from
    requesting LaTeX/DVI text metrics before the PGF renderer is selected.
    """
    normalized = normalize_plot_mode(mode)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, object] = {}
    if dpi is not None:
        save_kwargs["dpi"] = dpi

    if normalized == "publication":
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".pgf"}:
            raise ValueError(
                "Publication mode supports only '.pdf' and '.pgf' output; "
                f"received {path.name!r}. Use standard mode for raster figures."
            )
        save_kwargs.update(
            {
                "backend": "pgf",
                "bbox_inches": "tight",
                "pad_inches": pad_inches,
            }
        )
    else:
        fig.tight_layout()

    fig.savefig(path, **save_kwargs)
    return path
