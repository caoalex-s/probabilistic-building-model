from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from pbm.io.loader import load_all
from pbm.network.builder import build_and_insert
from pbm.sim import simulate_once
from pbm.sim.notional_damage import (
    NotionalDamageError,
    apply_notional_damage,
    notional_damage_mask,
    resolve_notional_damage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class CountingRng:
    def __init__(self, values: list[float]):
        self.values = np.asarray(values, dtype=float)
        self.calls: list[int] = []

    def random(self, count: int) -> np.ndarray:
        self.calls.append(int(count))
        return self.values[:count].copy()


def test_obsolete_staging_modules_are_absent():
    assert not (REPO_ROOT / "pbm" / "sim" / "staging.py").exists()
    assert not (REPO_ROOT / "pbm" / "network" / "stage0.py").exists()


def _build_zero_propagation_model(db_path: Path) -> None:
    build_and_insert(3, 3, 1, db_path, truncate=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE nodes SET pf_node=0.0;")
        connection.execute(
            """
            UPDATE edges
            SET pe=0.0, pe_fwd=NULL, pe_rev=NULL, directed=0, blocked=0;
            """
        )


def _node_id(db_path: Path, coordinate: tuple[int, int, int]) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT id FROM nodes WHERE x=? AND y=? AND z=?;",
            coordinate,
        ).fetchone()
    assert row is not None
    return int(row[0])


def _run_cli(
    input_db: Path,
    output_db: Path,
    *,
    coordinate: tuple[int, int, int] | None = None,
    percolation_scan: str = "",
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "cli" / "cli_simulation.py"),
        "--db",
        str(input_db),
        "--outdb",
        str(output_db),
        "--samples",
        "5",
        "--threshold",
        "2",
        "--seed",
        "123",
        "--percolation-scan",
        percolation_scan,
    ]
    if coordinate is not None:
        command.extend(
            [
                "--damage-coords",
                ",".join(str(value) for value in coordinate),
            ]
        )
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _run_data(output_db: Path) -> tuple[tuple[tuple[int, int], ...], dict]:
    with sqlite3.connect(output_db) as connection:
        run_id = int(
            connection.execute("SELECT MAX(id) FROM runs;").fetchone()[0]
        )
        histogram = tuple(
            (int(n_failed), int(count))
            for n_failed, count in connection.execute(
                """
                SELECT n_failed, count
                FROM pmf
                WHERE run_id=?
                ORDER BY n_failed;
                """,
                (run_id,),
            )
        )
        notes = json.loads(
            connection.execute(
                "SELECT notes FROM runs WHERE id=?;",
                (run_id,),
            ).fetchone()[0]
        )
    return histogram, notes


def test_apply_notional_damage_preserves_draw_slots_but_samples_only_survivors():
    probabilities = np.array([0.0, 0.4, 1.0, 0.8])
    damage = np.array([True, False, True, False])
    probabilities_before = probabilities.copy()
    damage_before = damage.copy()
    rng = CountingRng([0.99, 0.3, 0.0, 0.9])

    initial = apply_notional_damage(probabilities, damage, rng)

    assert rng.calls == [4]
    assert initial.tolist() == [True, True, True, False]
    assert np.array_equal(probabilities, probabilities_before)
    assert np.array_equal(damage, damage_before)


def test_resolution_rejects_unknown_targets_and_database_runtime_conflict():
    nodes = [
        type("Node", (), {"id": 1, "x": 0, "y": 0, "z": 0})(),
        type("Node", (), {"id": 2, "x": 1, "y": 0, "z": 0})(),
    ]

    with pytest.raises(NotionalDamageError, match="Unknown runtime.*99"):
        resolve_notional_damage(nodes, node_ids=[99])
    with pytest.raises(NotionalDamageError, match="Unknown.*coordinate"):
        resolve_notional_damage(nodes, coordinates=[(9, 9, 9)])
    with pytest.raises(NotionalDamageError, match="already defines"):
        resolve_notional_damage(
            nodes,
            database_node_ids=[1],
            coordinates=[(1, 0, 0)],
        )


@pytest.mark.parametrize(
    ("label", "coordinate"),
    [
        ("corner", (0, 0, 0)),
        ("perimeter", (1, 0, 0)),
        ("internal", (1, 1, 0)),
    ],
)
def test_damage_scenarios_are_identical_through_api_cli_and_database(
    tmp_path: Path,
    label: str,
    coordinate: tuple[int, int, int],
):
    input_db = tmp_path / f"{label}.db"
    runtime_output = tmp_path / f"{label}-runtime.db"
    database_output = tmp_path / f"{label}-database.db"
    _build_zero_propagation_model(input_db)
    expected_id = _node_id(input_db, coordinate)

    # Public API with runtime coordinate authoring.
    runtime_model = load_all(input_db)
    expected_index = next(
        index
        for index, node in enumerate(runtime_model.nodes)
        if node.id == expected_id
    )
    runtime_api = simulate_once(
        runtime_model,
        np.random.default_rng(123),
        damage_coordinates=[coordinate],
    )
    assert runtime_api.damage_node_ids == (expected_id,)
    assert np.flatnonzero(runtime_api.failed_mask).tolist() == [expected_index]

    # CLI with the same runtime coordinate authoring.
    runtime_cli = _run_cli(
        input_db,
        runtime_output,
        coordinate=coordinate,
    )
    assert runtime_cli.returncode == 0, runtime_cli.stderr
    runtime_histogram, runtime_notes = _run_data(runtime_output)
    assert runtime_notes["notional_damage_node_ids"] == [expected_id]
    assert runtime_notes["notional_damage_source"] == "runtime"
    assert dict(runtime_histogram)[1] == 5

    # Canonical persistent representation, consumed without runtime targets.
    with sqlite3.connect(input_db) as connection:
        connection.execute(
            "INSERT INTO initial_damage(node_id) VALUES (?);",
            (expected_id,),
        )

    database_model = load_all(input_db)
    database_api = simulate_once(
        database_model,
        np.random.default_rng(123),
    )
    assert database_api.damage_node_ids == (expected_id,)
    assert np.array_equal(
        database_api.failed_mask,
        runtime_api.failed_mask,
    )

    database_cli = _run_cli(input_db, database_output)
    assert database_cli.returncode == 0, database_cli.stderr
    database_histogram, database_notes = _run_data(database_output)
    assert database_histogram == runtime_histogram
    assert database_notes["notional_damage_node_ids"] == [expected_id]
    assert database_notes["notional_damage_source"] == "database"
    assert (
        runtime_notes["stationary_model_provenance"]["sha256"]
        == database_notes["stationary_model_provenance"]["sha256"]
    )


def test_cli_and_api_reject_runtime_targets_when_database_damage_exists(
    tmp_path: Path,
):
    input_db = tmp_path / "conflict.db"
    output_db = tmp_path / "conflict-output.db"
    _build_zero_propagation_model(input_db)
    node_id = _node_id(input_db, (0, 0, 0))
    with sqlite3.connect(input_db) as connection:
        connection.execute(
            "INSERT INTO initial_damage(node_id) VALUES (?);",
            (node_id,),
        )

    loaded = load_all(input_db)
    with pytest.raises(NotionalDamageError, match="already defines"):
        simulate_once(
            loaded,
            np.random.default_rng(123),
            damage_coordinates=[(1, 0, 0)],
        )

    result = _run_cli(
        input_db,
        output_db,
        coordinate=(1, 0, 0),
    )
    assert result.returncode != 0
    assert "already defines notional damage" in (result.stdout + result.stderr)


def test_percolation_path_uses_the_same_notional_damage_set(tmp_path: Path):
    input_db = tmp_path / "percolation.db"
    output_db = tmp_path / "percolation-output.db"
    _build_zero_propagation_model(input_db)

    result = _run_cli(
        input_db,
        output_db,
        coordinate=(0, 0, 0),
        percolation_scan="xyz:0,0,1",
    )
    assert result.returncode == 0, result.stderr

    with sqlite3.connect(output_db) as connection:
        row = connection.execute(
            "SELECT p_any, p_one, p_full, mean_nf FROM percolation_multi;"
        ).fetchone()
    assert row == (1.0, 1.0, 0.0, 1.0)


def test_damage_mask_is_canonical_for_combined_id_and_coordinate_targets(
    tmp_path: Path,
):
    input_db = tmp_path / "combined.db"
    _build_zero_propagation_model(input_db)
    loaded = load_all(input_db)
    node_id = _node_id(input_db, (0, 0, 0))

    resolved = resolve_notional_damage(
        loaded.nodes,
        node_ids=[node_id, node_id],
        coordinates=[(0, 0, 0)],
    )
    mask = notional_damage_mask(loaded.nodes, resolved)

    assert resolved == (node_id,)
    assert int(mask.sum()) == 1
