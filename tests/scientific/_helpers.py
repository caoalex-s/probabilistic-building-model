"""Shared helpers for stationary-paper scientific regressions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import binom

from pbm.io.loader import Loaded
from pbm.network.propagation import (
    PropagationMatrices,
    build_transition_matrix,
    simulate_once_mc,
)
from pbm.post.percolation_threshold import AxisScanMetadata
from pbm.sim.engine import edges_structured_table
from pbm.sim.notional_damage import (
    apply_notional_damage,
    notional_damage_mask,
    resolve_notional_damage,
)
from pbm.sim.random_streams import realization_rng


@dataclass(frozen=True)
class PaperLattice:
    """The built 4x4x4 input model used by the regression suites."""

    db_path: Path
    model: Loaded


@dataclass(frozen=True)
class EstimatorShapeCase:
    """A synthetic estimator response with explicit varied/fixed axes."""

    name: str
    metadata: AxisScanMetadata
    response: Callable[[np.ndarray], np.ndarray]
    expected_threshold: float | None
    publication_seed: int


def estimator_shape_cases() -> tuple[EstimatorShapeCase, ...]:
    """Return generic sigmoid and concave estimator fixtures.

    These shapes deliberately exercise different outcomes and axis metadata.
    Their roots are synthetic test inputs, not the manuscript's Figure 3
    thresholds. Retained manuscript counts are tested separately.
    """

    return (
        EstimatorShapeCase(
            name="synthetic_xyz_sigmoid",
            metadata=AxisScanMetadata(
                scan_spec="xyz:0,1,0.05",
                mode="1d_xyz",
                varied_axes=("x", "y", "z"),
                fixed_axes=(),
                varied_axis_groups=(("x", "y", "z"),),
            ),
            response=lambda x: 0.0064
            / (1.0 + np.exp(-16.0 * (x - 0.65))),
            expected_threshold=0.65,
            publication_seed=901,
        ),
        EstimatorShapeCase(
            name="synthetic_xy_concave",
            metadata=AxisScanMetadata(
                scan_spec="xy:0,1,0.05",
                mode="1d_xy",
                varied_axes=("x", "y"),
                fixed_axes=("z",),
                varied_axis_groups=(("x", "y"),),
                fixed_axis_values=(("z", 0.8),),
            ),
            response=lambda x: 0.0064
            * (1.0 - np.exp(-3.0 * x))
            / (1.0 - np.exp(-3.0)),
            expected_threshold=None,
            publication_seed=902,
        ),
        EstimatorShapeCase(
            name="synthetic_z_sigmoid",
            metadata=AxisScanMetadata(
                scan_spec="z:0,1,0.05",
                mode="1d_z",
                varied_axes=("z",),
                fixed_axes=("x", "y"),
                varied_axis_groups=(("z",),),
                fixed_axis_values=(("x", 0.8), ("y", 0.8)),
            ),
            response=lambda x: 0.85
            / (1.0 + np.exp(-10.0 * (x - 0.50))),
            expected_threshold=0.50,
            publication_seed=903,
        ),
    )


def _probability(value: float, *, name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1].")
    return probability


def _sample_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("samples must be a positive integer.")
    if int(value) <= 0:
        raise ValueError("samples must be a positive integer.")
    return int(value)


def _node_probabilities(
    node_count: int,
    pf_node: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(pf_node, dtype=float)
    if probabilities.ndim == 0:
        probabilities = np.full(
            node_count,
            _probability(float(probabilities), name="pf_node"),
            dtype=float,
        )
    if probabilities.shape != (node_count,):
        raise ValueError(
            f"pf_node must be scalar or have shape ({node_count},)."
        )
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("pf_node entries must be finite probabilities.")
    return probabilities


def transition_matrices(
    model: Loaded,
    axis_probabilities: float | Mapping[str, float],
) -> PropagationMatrices:
    """Build the production transition matrix for one axis configuration."""

    if isinstance(axis_probabilities, Mapping):
        if set(axis_probabilities) != {"x", "y", "z"}:
            raise ValueError(
                "axis_probabilities must define exactly x, y, and z."
            )
        probabilities = {
            axis: _probability(axis_probabilities[axis], name=f"pe_{axis}")
            for axis in ("x", "y", "z")
        }
    else:
        value = _probability(axis_probabilities, name="pe")
        probabilities = {axis: value for axis in ("x", "y", "z")}

    edges = edges_structured_table(model)
    axes = np.asarray([edge.dir for edge in model.edges])
    if np.any(edges["directed"] != 0):
        raise ValueError("Scientific lattice fixtures must be undirected.")
    for axis, probability in probabilities.items():
        edges["pe"][axes == axis] = probability
    return build_transition_matrix(len(model.nodes), edges)


def simulate_histogram(
    model: Loaded,
    *,
    axis_probabilities: float | Mapping[str, float],
    pf_node: float | Sequence[float] | np.ndarray,
    samples: int,
    seed: int,
    damage_coordinates: Sequence[tuple[int, int, int]] = (),
) -> np.ndarray:
    """Run a seeded production-kernel histogram without persistence overhead."""

    sample_count = _sample_count(samples)
    node_count = len(model.nodes)
    probabilities = _node_probabilities(node_count, pf_node)
    resolved_damage = resolve_notional_damage(
        model.nodes,
        database_node_ids=model.damage_node_ids,
        coordinates=damage_coordinates,
    )
    damage_mask = notional_damage_mask(model.nodes, resolved_damage)
    matrices = transition_matrices(model, axis_probabilities)

    counts = np.zeros(node_count + 1, dtype=np.int64)
    for realization_index in range(sample_count):
        rng = realization_rng(seed, realization_index)
        initial = apply_notional_damage(
            probabilities,
            damage_mask,
            rng,
        )
        outcome = simulate_once_mc(initial, matrices, rng=rng)
        counts[int(outcome.F_final.sum())] += 1
    return counts


def simulate_uniform_scan(
    model: Loaded,
    propagation_probabilities: Sequence[float],
    *,
    pf_node: float | Sequence[float] | np.ndarray,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Run coupled uniform scans using the same realization streams."""

    return np.stack(
        [
            simulate_histogram(
                model,
                axis_probabilities=float(pe),
                pf_node=pf_node,
                samples=samples,
                seed=seed,
            )
            for pe in propagation_probabilities
        ]
    )


def simulate_conditional_single_seed_scan(
    model: Loaded,
    propagation_probabilities: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Simulate ``P(N_f | exactly one uniformly located initial failure)``.

    This removes the rare-event factor from the small-``pf_node`` limit used in
    manuscript Equations 20-24. The same seed location and realization stream
    are reused at each propagation probability.
    """

    sample_count = _sample_count(samples)
    node_count = len(model.nodes)
    histograms: list[np.ndarray] = []
    for propagation_probability in propagation_probabilities:
        matrices = transition_matrices(
            model,
            float(propagation_probability),
        )
        counts = np.zeros(node_count + 1, dtype=np.int64)
        for realization_index in range(sample_count):
            rng = realization_rng(seed, realization_index)
            initial = np.zeros(node_count, dtype=bool)
            initial[int(rng.integers(node_count))] = True
            outcome = simulate_once_mc(initial, matrices, rng=rng)
            counts[int(outcome.F_final.sum())] += 1
        histograms.append(counts)
    return np.stack(histograms)


def simultaneous_binomial_acceptance_interval(
    *,
    samples: int,
    probability: float,
    family_size: int,
    family_alpha: float = 0.01,
) -> tuple[int, int]:
    """Return a Bonferroni-simultaneous exact binomial count interval."""

    sample_count = _sample_count(samples)
    p = _probability(probability, name="probability")
    if family_size <= 0:
        raise ValueError("family_size must be positive.")
    if not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must lie in (0, 1).")
    if p == 0.0:
        return 0, 0
    if p == 1.0:
        return sample_count, sample_count
    coverage = 1.0 - family_alpha / int(family_size)
    lower, upper = binom.interval(coverage, sample_count, p)
    return int(lower), int(upper)


def assert_multinomial_marginals_in_acceptance_region(
    counts: np.ndarray,
    probabilities: np.ndarray,
    *,
    family_alpha: float = 0.01,
) -> None:
    """Check every multinomial category through its binomial marginal."""

    observed = np.asarray(counts, dtype=np.int64)
    expected = np.asarray(probabilities, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError("counts and probabilities must have the same shape.")
    samples = int(observed.sum())
    failures: list[str] = []
    for index, (count, probability) in enumerate(
        zip(observed, expected, strict=True)
    ):
        lower, upper = simultaneous_binomial_acceptance_interval(
            samples=samples,
            probability=float(probability),
            family_size=observed.size,
            family_alpha=family_alpha,
        )
        if not lower <= int(count) <= upper:
            failures.append(
                f"N_f={index}: observed {int(count)}, expected count interval "
                f"[{lower}, {upper}]"
            )
    if failures:
        raise AssertionError("; ".join(failures))


def hoeffding_radius(
    *,
    samples: int,
    family_size: int,
    family_alpha: float = 0.01,
) -> float:
    """Familywise Hoeffding radius for bounded Bernoulli estimates."""

    sample_count = _sample_count(samples)
    if family_size <= 0:
        raise ValueError("family_size must be positive.")
    if not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must lie in (0, 1).")
    return float(
        math.sqrt(
            math.log(2.0 * int(family_size) / family_alpha)
            / (2.0 * sample_count)
        )
    )


def node_degree(model: Loaded, coordinate: tuple[int, int, int]) -> int:
    """Return the undirected degree of a coordinate in the built lattice."""

    matches = [
        node.id
        for node in model.nodes
        if (node.x, node.y, node.z) == tuple(coordinate)
    ]
    if len(matches) != 1:
        raise ValueError(f"Coordinate {coordinate!r} is not unique.")
    node_id = matches[0]
    return sum(
        int(edge.org == node_id or edge.dst == node_id)
        for edge in model.edges
    )


def run_cli_histogram(
    *,
    repo_root: Path,
    input_db: Path,
    output_db: Path,
    workers: int,
    chunk_size: int,
    samples: int = 401,
    seed: int = 424242,
) -> tuple[int, ...]:
    """Run the supported CLI and return its integer PMF histogram."""

    environment = os.environ.copy()
    environment.setdefault(
        "MPLCONFIGDIR",
        str(output_db.parent / "matplotlib-cache"),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "cli" / "cli_simulation.py"),
            "--db",
            str(input_db),
            "--outdb",
            str(output_db),
            "--scenario-id",
            "9",
            "--samples",
            str(samples),
            "--threshold",
            "2",
            "--seed",
            str(seed),
            "--log-attempts",
            "no",
            "--percolation-scan",
            "",
            "--mc-workers",
            str(workers),
            "--mc-chunk-size",
            str(chunk_size),
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    with sqlite3.connect(output_db) as connection:
        run_id = int(
            connection.execute("SELECT MAX(id) FROM runs;").fetchone()[0]
        )
        return tuple(
            int(row[0])
            for row in connection.execute(
                """
                SELECT count
                FROM pmf
                WHERE run_id=?
                ORDER BY n_failed;
                """,
                (run_id,),
            )
        )
