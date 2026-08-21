"""Fantasy scoring functions for points and nine-category leagues."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

DEFAULT_POINTS_WEIGHTS: dict[str, float] = {
    "PTS": 1.0,
    "REB": 1.2,
    "AST": 1.5,
    "STL": 3.0,
    "BLK": 3.0,
    "TOV": -1.0,
    "3PM": 0.5,
}


def points_value(row: Mapping[str, float], weights: Mapping[str, float] | None = None) -> float:
    """Calculate a points-league per-game value from one stat row."""
    active = weights or DEFAULT_POINTS_WEIGHTS
    return float(sum(float(row.get(category, 0.0)) * weight for category, weight in active.items()))


def add_points_value(frame: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Return a copy of a season table with a ``fantasy_value`` column."""
    result = frame.copy()
    result["fantasy_value"] = result.apply(lambda row: points_value(row, weights), axis=1)
    return result


def add_nine_category_value(
    frame: pd.DataFrame,
    top_n: int = 150,
    minutes_column: str = "MIN",
) -> pd.DataFrame:
    """Add nine-category z-score value, using the top players by minutes.

    Percent categories are already represented as percentages when supplied by
    NBA stats. Turnovers are inverted because fewer turnovers are beneficial.
    The resulting value is a sum of category z-scores.
    """
    result = frame.copy()
    required = ["FG_PCT", "FT_PCT", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
    missing = [column for column in required if column not in result]
    if missing:
        raise ValueError(f"Missing nine-category columns: {', '.join(missing)}")
    reference = result.nlargest(top_n, minutes_column) if minutes_column in result else result
    values = reference[required].astype(float).copy()
    values["TOV"] = -values["TOV"]
    means = values.mean()
    stds = values.std(ddof=0).replace(0, 1.0)
    normalized = result[required].astype(float).copy()
    normalized["TOV"] = -normalized["TOV"]
    result["fantasy_value"] = normalized.sub(means).div(stds).sum(axis=1)
    return result


def scoring_config(config: Mapping[str, object]) -> dict[str, float]:
    """Extract point weights from application configuration."""
    scoring = config.get("points", {}) if isinstance(config, Mapping) else {}
    return {str(key): float(value) for key, value in scoring.items()} or DEFAULT_POINTS_WEIGHTS.copy()

