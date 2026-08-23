"""Deterministic uncertainty simulation from empirical NBA histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SimulationSummary:
    """Summary statistics for one player's simulated season totals."""

    projected_per_game: float
    median: float
    p05: float
    p25: float
    p75: float
    p95: float


def simulate_season_totals(
    projected_per_game: float,
    game_values: Iterable[float],
    historical_games_played: Iterable[float],
    draws: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate season totals by resampling game outputs and availability.

    Game-log values are scaled to the projected mean so historical variance is
    preserved without allowing an old level estimate to override the model.
    Availability is sampled from the player's empirical games-played history,
    capped at the 82-game regular-season maximum.
    """
    if draws <= 0:
        raise ValueError("draws must be positive")
    generator = rng or np.random.default_rng()
    values = np.asarray(list(game_values), dtype=float)
    values = values[np.isfinite(values)]
    availability = np.asarray(list(historical_games_played), dtype=float)
    availability = availability[np.isfinite(availability)]
    if len(values) == 0:
        values = np.array([float(projected_per_game)])
    if len(availability) == 0:
        availability = np.array([82.0])
    scale_base = float(values.mean())
    scale = float(projected_per_game / scale_base) if scale_base else 1.0
    normalized = values * scale
    games = np.rint(generator.choice(availability, size=draws, replace=True)).clip(1, 82).astype(int)
    totals = np.empty(draws, dtype=float)
    for index, count in enumerate(games):
        totals[index] = generator.choice(normalized, size=count, replace=True).sum()
    return totals


def summarize_simulation(projected_per_game: float, totals: np.ndarray) -> SimulationSummary:
    """Convert simulated totals into the requested percentile bands."""
    p05, p25, median, p75, p95 = np.percentile(totals, [5, 25, 50, 75, 95])
    return SimulationSummary(float(projected_per_game), float(median), float(p05), float(p25), float(p75), float(p95))

