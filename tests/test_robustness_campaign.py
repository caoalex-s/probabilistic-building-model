"""Integration tests for threshold-specific notional-damage robustness."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from pbm.post.robustness import RobustnessCoherenceError
from pbm.post.robustness_plots import (
    RobustnessPairingError,
    analyze_entropy_robustness_campaign,
    plot_entropy_robustness_index,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_HASH = "a" * 64


def _notes(damage: list[int], *, model_hash: str = MODEL_HASH) -> str:
    return json.dumps(
        {
            "notional_damage_node_ids": damage,
            "stationary_model_provenance": {
                "schema_version": 1,
                "contract": "stationary-v1",
                "sha256": model_hash,
            },
        }
    )


def _campaign_db(path: Path, *, incoherent: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY,
            scenario_id INTEGER NOT NULL,
            notes TEXT NOT NULL
        );
        CREATE TABLE pmf (
            run_id INTEGER NOT NULL,
            n_failed INTEGER NOT NULL,
            p REAL NOT NULL,
            PRIMARY KEY (run_id, n_failed)
        );
        """
    )
    connection.executemany(
        "INSERT INTO runs(id, scenario_id, notes) VALUES (?, ?, ?);",
        [
            (1, 0, _notes([])),
            (2, 1, _notes([7])),
            (3, 2, _notes([9])),
        ],
    )
    baseline = [0.70, 0.20, 0.08, 0.015, 0.005]
    damaged_1 = (
        [0.75, 0.20, 0.04, 0.009, 0.001]
        if incoherent
        else [0.50, 0.20, 0.15, 0.10, 0.05]
    )
    damaged_2 = [0.60, 0.20, 0.10, 0.07, 0.03]
    rows = [
        (run_id, n_failed, probability)
        for run_id, pmf in enumerate(
            (baseline, damaged_1, damaged_2),
            start=1,
        )
        for n_failed, probability in enumerate(pmf)
    ]
    connection.executemany(
        "INSERT INTO pmf(run_id, n_failed, p) VALUES (?, ?, ?);",
        rows,
    )
    connection.commit()
    return connection


def test_every_s_lim_gets_its_own_undamaged_baseline(tmp_path: Path):
    connection = _campaign_db(tmp_path / "campaign.db")

    campaign = analyze_entropy_robustness_campaign(
        connection,
        baseline_run_id=1,
        damaged_run_ids=[2, 3],
        s_lim=[2, 4],
        coherence_mode="strict",
    )

    assert [item.p0 for item in campaign.macrostates] == pytest.approx(
        [0.10, 0.005]
    )
    for macrostate in campaign.macrostates:
        assert all(
            item.result.p0 == pytest.approx(macrostate.p0)
            for item in macrostate.damaged_runs
        )
        assert all(
            item.result.s_lim == macrostate.s_lim
            for item in macrostate.damaged_runs
        )


def test_intentionally_noncoherent_campaign_is_strict_or_diagnostic(
    tmp_path: Path,
):
    connection = _campaign_db(tmp_path / "incoherent.db", incoherent=True)

    with pytest.raises(RobustnessCoherenceError):
        analyze_entropy_robustness_campaign(
            connection,
            baseline_run_id=1,
            damaged_run_ids=[2],
            s_lim=2,
            coherence_mode="strict",
        )

    campaign = analyze_entropy_robustness_campaign(
        connection,
        baseline_run_id=1,
        damaged_run_ids=[2],
        s_lim=2,
        coherence_mode="diagnostic",
    )
    evaluation = campaign.macrostates[0].damaged_runs[0].result.robustness
    assert evaluation.value is None
    assert evaluation.status == "incoherent"


def test_pairing_rejects_different_stationary_models(tmp_path: Path):
    connection = _campaign_db(tmp_path / "mismatch.db")
    connection.execute(
        "UPDATE runs SET notes=? WHERE id=2;",
        (_notes([7], model_hash="b" * 64),),
    )
    connection.commit()

    with pytest.raises(RobustnessPairingError, match="same stationary model"):
        analyze_entropy_robustness_campaign(
            connection,
            baseline_run_id=1,
            damaged_run_ids=[2],
            s_lim=2,
        )


def test_plot_uses_separate_panels_and_returns_machine_readable_result(
    tmp_path: Path,
):
    connection = _campaign_db(tmp_path / "plot.db")

    campaign = plot_entropy_robustness_index(
        connection,
        baseline_run_id=1,
        damaged_run_ids=[2, 3],
        s_lim=[2, 4],
        figs_dir=tmp_path / "figs",
        formats=("png",),
        plot_mode="standard",
    )

    assert len(campaign.macrostates) == 2
    assert len(campaign.output_paths) == 1
    assert campaign.output_paths[0].is_file()
    assert campaign.to_dict()["macrostates"][1]["p0"] == pytest.approx(0.005)


def test_cli_requires_explicit_s_lim_and_writes_diagnostics(tmp_path: Path):
    database = tmp_path / "cli.db"
    connection = _campaign_db(database)
    connection.close()

    missing = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "cli" / "cli_post_processing.py"),
            "--outdb",
            str(database),
            "--entropy-robustness",
            "--baseline-run-id",
            "1",
            "--damaged-run-ids",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0
    assert "--s-lim is required" in (missing.stdout + missing.stderr)

    output_json = tmp_path / "robustness.json"
    complete = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "cli" / "cli_post_processing.py"),
            "--outdb",
            str(database),
            "--figs-root",
            str(tmp_path / "figs"),
            "--plot-mode",
            "standard",
            "--entropy-robustness",
            "--baseline-run-id",
            "1",
            "--damaged-run-ids",
            "2,3",
            "--s-lim",
            "2,4",
            "--robustness-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert complete.returncode == 0, complete.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert [item["p0"] for item in payload["macrostates"]] == pytest.approx(
        [0.10, 0.005]
    )
    assert payload["output_paths"]


def test_cli_exposes_only_explicit_damage_robustness_interface():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "cli" / "cli_post_processing.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--damaged-run-ids" in result.stdout
    assert "--s-lim" in result.stdout
    assert "--robustness-index" not in result.stdout
