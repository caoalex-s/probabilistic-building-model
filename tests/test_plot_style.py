from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from pbm.post.plot_style import (
    configure_plotting,
    normalize_plot_mode,
    plot_rc_params,
    plotting_context,
    save_figure,
)


def test_normalize_plot_mode():
    assert normalize_plot_mode("standard") == "standard"
    assert normalize_plot_mode(" Publication ") == "publication"
    with pytest.raises(ValueError, match="Unknown plot mode"):
        normalize_plot_mode("draft")


def test_plot_rc_params_distinguish_modes():
    standard = plot_rc_params("standard")
    publication = plot_rc_params("publication")

    assert standard["text.usetex"] is False
    assert standard["mathtext.fontset"] == "stix"

    assert publication["text.usetex"] is True
    assert publication["pgf.texsystem"] == "pdflatex"
    assert publication["pgf.rcfonts"] is False
    assert "mathptmx" in str(publication["pgf.preamble"])


def test_plotting_context_restores_rcparams():
    original_usetex = mpl.rcParams["text.usetex"]
    original_font_size = mpl.rcParams["font.size"]

    with plotting_context("standard"):
        assert mpl.rcParams["text.usetex"] is False
        assert mpl.rcParams["font.size"] == 16

    assert mpl.rcParams["text.usetex"] == original_usetex
    assert mpl.rcParams["font.size"] == original_font_size


def test_configure_plotting_applies_global_mode():
    with mpl.rc_context():
        assert configure_plotting("standard") == "standard"
        assert mpl.rcParams["text.usetex"] is False
        assert mpl.rcParams["mathtext.fontset"] == "stix"


def test_save_figure_standard_writes_raster(tmp_path: Path):
    out = tmp_path / "standard.png"
    with plotting_context("standard"):
        fig, ax = plt.subplots()
        ax.set_title(r"$N_{\mathrm{f}}$")
        try:
            result = save_figure(fig, out, mode="standard", dpi=80)
        finally:
            plt.close(fig)

    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_save_figure_publication_selects_pgf_without_tight_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fig = plt.figure()
    calls: dict[str, object] = {}

    def fake_savefig(path: Path, **kwargs: object) -> None:
        calls["path"] = Path(path)
        calls["kwargs"] = kwargs

    def fail_tight_layout() -> None:
        raise AssertionError("tight_layout must not run in publication mode")

    monkeypatch.setattr(fig, "savefig", fake_savefig)
    monkeypatch.setattr(fig, "tight_layout", fail_tight_layout)

    try:
        out = tmp_path / "publication.pdf"
        result = save_figure(fig, out, mode="publication")
    finally:
        plt.close(fig)

    assert result == out
    assert calls["path"] == out
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["backend"] == "pgf"
    assert kwargs["bbox_inches"] == "tight"
    assert kwargs["pad_inches"] == pytest.approx(0.05)


def test_save_figure_publication_rejects_raster_output(tmp_path: Path):
    fig = plt.figure()
    try:
        with pytest.raises(ValueError, match="supports only"):
            save_figure(fig, tmp_path / "publication.png", mode="publication")
    finally:
        plt.close(fig)
