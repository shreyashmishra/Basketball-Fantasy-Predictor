"""Leakage-safe season feature engineering."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .scoring import add_points_value

FEATURE_COLUMNS = [
    "feature_lag_value_1",
    "feature_lag_value_2",
    "feature_lag_value_3",
    "feature_age",
    "feature_games_lag_1",
    "feature_games_lag_2",
    "feature_games_lag_3",
    "feature_minutes_lag_1",
    "feature_minutes_lag_2",
    "feature_minutes_lag_3",
    "feature_minutes_trend",
    "feature_usage_lag_1",
    "feature_team_change",
    "feature_limited_history",
    "feature_yahoo_percent_owned",
]

# These are performance fields from the target season and are never allowed in
# the model matrix. Keeping this list close to the assertion makes the causal
# boundary easy to audit during an interview or code review.
SAME_SEASON_FIELDS = {
    "PTS", "REB", "AST", "STL", "BLK", "TOV", "3PM", "FG_PCT", "FT_PCT",
    "games_played", "minutes_per_game", "usage_rate", "fantasy_value", "target_value",
}


def _prior(values: list[float], offset: int) -> float:
    """Return a lagged value or NaN when the requested history is unavailable."""
    return values[-offset] if len(values) >= offset else float("nan")


def build_feature_table(
    season_stats: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    yahoo_percent_owned: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one target row per player-season with strictly pre-season features.

    The current row contributes only identity, eligibility metadata, and the
    target. Lagged performance fields are computed before the current row is
    appended to each player's history. Percent-owned data is likewise joined
    using the prior season only.
    """
    required = {"player_id", "player_name", "season", "team", "games_played", "minutes_per_game"}
    missing = required.difference(season_stats.columns)
    if missing:
        raise ValueError(f"Season stats missing columns: {', '.join(sorted(missing))}")
    scored = add_points_value(season_stats, weights)
    scored = scored.sort_values(["player_id", "season"], kind="stable").reset_index(drop=True)
    owned = _owned_lookup(yahoo_percent_owned)
    rows: list[dict[str, object]] = []
    for player_id, group in scored.groupby("player_id", sort=False):
        history: list[dict[str, object]] = []
        values: list[float] = []
        games: list[float] = []
        minutes: list[float] = []
        usage: list[float] = []
        teams: list[object] = []
        for record in group.to_dict("records"):
            season = str(record["season"])
            previous_season = _previous_season(season)
            rows.append({
                "player_id": player_id,
                "player_name": record["player_name"],
                "season": season,
                "team": record["team"],
                "games_played": record["games_played"],
                "minutes_per_game": record["minutes_per_game"],
                "target_value": record["fantasy_value"],
                "feature_lag_value_1": _prior(values, 1),
                "feature_lag_value_2": _prior(values, 2),
                "feature_lag_value_3": _prior(values, 3),
                # Age is a known pre-season descriptor, unlike current-season box scores.
                "feature_age": record.get("age", float("nan")),
                "feature_games_lag_1": _prior(games, 1),
                "feature_games_lag_2": _prior(games, 2),
                "feature_games_lag_3": _prior(games, 3),
                "feature_minutes_lag_1": _prior(minutes, 1),
                "feature_minutes_lag_2": _prior(minutes, 2),
                "feature_minutes_lag_3": _prior(minutes, 3),
                "feature_minutes_trend": (minutes[-1] - minutes[-2]) if len(minutes) >= 2 else float("nan"),
                "feature_usage_lag_1": _prior(usage, 1),
                "feature_team_change": int(bool(teams and teams[-1] != record["team"])),
                "feature_limited_history": int(len(history) < 3),
                "feature_yahoo_percent_owned": owned.get((player_id, previous_season), float("nan")),
            })
            values.append(float(record["fantasy_value"]))
            games.append(float(record["games_played"]))
            minutes.append(float(record["minutes_per_game"]))
            usage.append(float(record.get("usage_rate", float("nan"))))
            teams.append(record["team"])
            history.append(record)
    result = pd.DataFrame(rows)
    assert_no_same_season_features(result)
    return result


def _previous_season(season: str) -> str:
    """Return the NBA season immediately preceding a label such as ``2020-21``."""
    year = int(season[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _owned_lookup(frame: pd.DataFrame | None) -> dict[tuple[object, str], float]:
    """Build a player-season lookup from optional Yahoo ownership data."""
    if frame is None or frame.empty:
        return {}
    required = {"player_id", "season", "percent_owned"}
    if not required.issubset(frame.columns):
        raise ValueError("Yahoo ownership data requires player_id, season, and percent_owned")
    return {
        (row.player_id, str(row.season)): float(row.percent_owned)
        for row in frame.itertuples(index=False)
    }


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only approved pre-season feature columns in stable order."""
    assert_no_same_season_features(frame)
    return frame[FEATURE_COLUMNS].astype(float)


def build_projection_features(
    season_stats: pd.DataFrame,
    feature_table: pd.DataFrame,
    target_season: str,
) -> pd.DataFrame:
    """Create next-season rows from each player's latest available history."""
    latest = feature_table.sort_values("season", key=lambda values: values.map(_season_year)).groupby("player_id").tail(1)
    latest = latest.set_index("player_id")
    current_stats = season_stats.sort_values("season", key=lambda values: values.map(_season_year)).groupby("player_id").tail(1).set_index("player_id")
    rows: list[dict[str, object]] = []
    for player_id, record in latest.iterrows():
        raw = current_stats.loc[player_id] if player_id in current_stats.index else record
        rows.append({
            "player_id": player_id,
            "player_name": record["player_name"],
            "season": target_season,
            "team": raw.get("team", record.get("team")),
            "games_played": float("nan"),
            "minutes_per_game": float("nan"),
            "target_value": float("nan"),
            "feature_lag_value_1": record["target_value"],
            "feature_lag_value_2": record["feature_lag_value_1"],
            "feature_lag_value_3": record["feature_lag_value_2"],
            "feature_age": float(record["feature_age"]) + 1 if pd.notna(record["feature_age"]) else float("nan"),
            "feature_games_lag_1": record["games_played"],
            "feature_games_lag_2": record["feature_games_lag_1"],
            "feature_games_lag_3": record["feature_games_lag_2"],
            "feature_minutes_lag_1": record["minutes_per_game"],
            "feature_minutes_lag_2": record["feature_minutes_lag_1"],
            "feature_minutes_lag_3": record["feature_minutes_lag_2"],
            "feature_minutes_trend": record["minutes_per_game"] - record["feature_minutes_lag_1"] if pd.notna(record["feature_minutes_lag_1"]) else float("nan"),
            "feature_usage_lag_1": raw.get("usage_rate", record["feature_usage_lag_1"]),
            # A future roster move is unknown without an external transaction feed.
            "feature_team_change": 0,
            "feature_limited_history": int(record["feature_limited_history"]),
            "feature_yahoo_percent_owned": record["feature_yahoo_percent_owned"],
        })
    return pd.DataFrame(rows)


def _season_year(value: object) -> int:
    """Sort helper for season labels."""
    return int(str(value)[:4])


def assert_no_same_season_features(frame: pd.DataFrame) -> None:
    """Raise when a model feature violates the pre-season information boundary."""
    feature_columns = [column for column in frame.columns if column.startswith("feature_")]
    invalid = [column for column in feature_columns if column.removeprefix("feature_") in SAME_SEASON_FIELDS]
    if invalid:
        raise AssertionError(f"Same-season fields used as features: {', '.join(invalid)}")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise AssertionError(f"Feature table is missing approved features: {', '.join(missing)}")
