"""Stationary simulation API."""

from .engine import SimOutcome, simulate_once
from .notional_damage import (
    NotionalDamageError,
    apply_notional_damage,
    notional_damage_mask,
    resolve_notional_damage,
)

__all__ = [
    "NotionalDamageError",
    "SimOutcome",
    "apply_notional_damage",
    "notional_damage_mask",
    "resolve_notional_damage",
    "simulate_once",
]
