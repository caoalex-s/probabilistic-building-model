"""Execution-independent random streams for stationary Monte Carlo runs.

The public RNG contract is deliberately small:

* ``base_seed`` is a non-negative integer supplied with the experiment;
* ``realization_index`` is the zero-based position of a realization; and
* the realization uses ``PCG64(SeedSequence(base_seed, spawn_key=(lo, hi)))``,
  where ``lo`` and ``hi`` are the low and high 32-bit words of the index.

Consequently, the stream for realization ``i`` depends only on
``(base_seed, i)``. Database ``run_id`` values, scenario identifiers, worker
and process identifiers, hosts, chunks, and scheduling order are never stream
inputs.

One generator is used first for spontaneous node-failure draws and then for
all propagation draws in that realization. Percolation scan points and
otherwise comparable notional-damage cases intentionally reuse the same
mapping, providing common random numbers when their base seed is the same.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


RNG_CONTRACT_VERSION = "seedsequence-pcg64-realization-v1"
MAX_REALIZATION_INDEX = (1 << 64) - 1


def _nonnegative_integer(value: int, *, name: str) -> int:
    """Return ``value`` as an ``int`` after strict contract validation."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer; got {value!r}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {result}.")
    return result


def realization_seed_sequence(
    base_seed: int,
    realization_index: int,
) -> np.random.SeedSequence:
    """Return the stable child seed sequence for one realization.

    The fixed two-word spawn key avoids dependence on how child streams are
    requested or partitioned. It also permits direct random access to any
    realization without spawning all preceding streams.
    """
    seed = _nonnegative_integer(base_seed, name="base_seed")
    index = _nonnegative_integer(realization_index, name="realization_index")
    if index > MAX_REALIZATION_INDEX:
        raise ValueError(
            "realization_index must fit in an unsigned 64-bit integer; "
            f"got {index}."
        )

    low_word = index & 0xFFFFFFFF
    high_word = (index >> 32) & 0xFFFFFFFF
    return np.random.SeedSequence(
        entropy=seed,
        spawn_key=(low_word, high_word),
    )


def realization_rng(
    base_seed: int,
    realization_index: int,
) -> np.random.Generator:
    """Return the contract-defined generator for one realization."""
    seed_sequence = realization_seed_sequence(base_seed, realization_index)
    return np.random.Generator(np.random.PCG64(seed_sequence))
