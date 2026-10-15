"""Stationary pre-run notional-damage resolution and application.

Notional damage is one simultaneous set of deterministically failed nodes. The
database stores that set as ``initial_damage(node_id)``. Runtime coordinates
are authoring conveniences only: they are resolved to canonical node IDs
before simulation.

The executable node probability is always ``pf_node``. Damaged nodes are not
sampled; all other nodes receive independent Bernoulli draws.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from typing import Any

import numpy as np


class NotionalDamageError(ValueError):
    """Raised when a notional-damage target or source is invalid."""


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise NotionalDamageError(f"{label} must be an integer; got {value!r}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise NotionalDamageError(
            f"{label} must be an integer; got {value!r}."
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise NotionalDamageError(f"{label} must be an integer; got {value!r}.")
    return int(numeric)


def _node_maps(
    nodes: Sequence[Any],
) -> tuple[dict[int, int], dict[tuple[int, int, int], int]]:
    id_to_index: dict[int, int] = {}
    coordinate_to_id: dict[tuple[int, int, int], int] = {}

    for index, node in enumerate(nodes):
        node_id = _integer(getattr(node, "id", None), label="node id")
        if node_id in id_to_index:
            raise NotionalDamageError(
                f"Loaded model contains duplicate node id {node_id}."
            )
        id_to_index[node_id] = index

        coordinate = tuple(
            _integer(getattr(node, axis, None), label=f"node {node_id} {axis}")
            for axis in ("x", "y", "z")
        )
        if coordinate in coordinate_to_id:
            other = coordinate_to_id[coordinate]
            raise NotionalDamageError(
                "Loaded model contains duplicate coordinate "
                f"{coordinate} for node ids {other} and {node_id}."
            )
        coordinate_to_id[coordinate] = node_id

    return id_to_index, coordinate_to_id


def resolve_notional_damage(
    nodes: Sequence[Any],
    *,
    database_node_ids: Iterable[object] = (),
    node_ids: Iterable[object] = (),
    coordinates: Iterable[Sequence[object]] = (),
) -> tuple[int, ...]:
    """Resolve one stationary notional-damage set to sorted node IDs.

    Database damage and explicit runtime targets are mutually exclusive.
    Explicit IDs and coordinates may be combined; repeated references to the
    same node are normalized to one ID. Every target must exist.
    """

    id_to_index, coordinate_to_id = _node_maps(nodes)
    known_ids = set(id_to_index)

    database_targets = tuple(
        _integer(value, label="initial_damage.node_id")
        for value in database_node_ids
    )
    explicit_ids = tuple(
        _integer(value, label="notional-damage node id")
        for value in node_ids
    )

    explicit_coordinates: list[tuple[int, int, int]] = []
    for coordinate in coordinates:
        try:
            values = tuple(coordinate)
        except TypeError as exc:
            raise NotionalDamageError(
                "Each notional-damage coordinate must contain exactly x, y, z."
            ) from exc
        if len(values) != 3:
            raise NotionalDamageError(
                "Each notional-damage coordinate must contain exactly x, y, z; "
                f"got {values!r}."
            )
        explicit_coordinates.append(
            tuple(
                _integer(value, label=f"notional-damage coordinate {axis}")
                for axis, value in zip(("x", "y", "z"), values)
            )
        )

    if database_targets and (explicit_ids or explicit_coordinates):
        raise NotionalDamageError(
            "The input database already defines notional damage; do not also "
            "supply runtime node IDs or coordinates."
        )

    selected = set(database_targets if database_targets else explicit_ids)
    unknown_ids = sorted(selected - known_ids)
    if unknown_ids:
        source = "initial_damage" if database_targets else "runtime"
        raise NotionalDamageError(
            f"Unknown {source} notional-damage node id(s): {unknown_ids}."
        )

    if not database_targets:
        unknown_coordinates: list[tuple[int, int, int]] = []
        for coordinate in explicit_coordinates:
            node_id = coordinate_to_id.get(coordinate)
            if node_id is None:
                unknown_coordinates.append(coordinate)
            else:
                selected.add(node_id)
        if unknown_coordinates:
            raise NotionalDamageError(
                "Unknown notional-damage coordinate(s): "
                f"{sorted(unknown_coordinates)}."
            )

    return tuple(sorted(selected))


def notional_damage_mask(
    nodes: Sequence[Any],
    damage_node_ids: Iterable[object],
) -> np.ndarray:
    """Return the node-order mask for an already resolved damage set."""

    id_to_index, _ = _node_maps(nodes)
    mask = np.zeros(len(nodes), dtype=bool)
    unknown: list[int] = []
    for value in damage_node_ids:
        node_id = _integer(value, label="notional-damage node id")
        index = id_to_index.get(node_id)
        if index is None:
            unknown.append(node_id)
        else:
            mask[index] = True
    if unknown:
        raise NotionalDamageError(
            f"Unknown notional-damage node id(s): {sorted(set(unknown))}."
        )
    return mask


def apply_notional_damage(
    pf_node: Sequence[float] | np.ndarray,
    damage_mask: Sequence[bool] | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build ``F(0)`` by applying damage once and sampling undamaged nodes.

    The inputs are not mutated. One fixed-length node draw vector is consumed
    to preserve common-random-number alignment across damage scenarios, but
    Bernoulli comparisons are performed only for undamaged nodes.
    """

    probabilities = np.asarray(pf_node, dtype=float)
    mask = np.asarray(damage_mask, dtype=bool)
    if probabilities.ndim != 1:
        raise ValueError("pf_node must be a one-dimensional array.")
    if mask.ndim != 1 or mask.shape != probabilities.shape:
        raise ValueError(
            "damage_mask must be one-dimensional and match pf_node in length."
        )
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("pf_node values must be finite probabilities in [0,1].")

    initial_failures = mask.copy()
    undamaged = ~mask
    node_count = int(probabilities.size)
    if node_count:
        draws = np.asarray(rng.random(node_count), dtype=float)
        if draws.shape != (node_count,):
            raise ValueError(
                "rng.random(node_count) must return a one-dimensional vector."
            )
        initial_failures[undamaged] = (
            draws[undamaged] < probabilities[undamaged]
        )
    return initial_failures
