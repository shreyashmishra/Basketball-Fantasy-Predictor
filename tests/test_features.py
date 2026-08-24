import pandas as pd
import pytest

from fantasy.features import FEATURE_COLUMNS, assert_no_same_season_features, build_feature_table, model_matrix


def _stats() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": 1, "player_name": "A", "season": "2019-20", "team": "X", "games_played": 70, "minutes_per_game": 30, "PTS": 20, "REB": 5, "AST": 4, "STL": 1, "BLK": 1, "TOV": 2, "3PM": 2, "age": 24, "usage_rate": 25},
        {"player_id": 1, "player_name": "A", "season": "2020-21", "team": "Y", "games_played": 60, "minutes_per_game": 28, "PTS": 22, "REB": 6, "AST": 5, "STL": 1, "BLK": 1, "TOV": 3, "3PM": 3, "age": 25, "usage_rate": 27},
    ])


def test_feature_table_has_no_same_season_performance() -> None:
    table = build_feature_table(_stats())
    assert set(FEATURE_COLUMNS).issubset(table.columns)
    assert table.iloc[1].feature_lag_value_1 == table.iloc[0].target_value
    assert table.iloc[1].feature_team_change == 1
    assert_no_same_season_features(table)
    assert "target_value" not in model_matrix(table).columns


def test_same_season_feature_is_rejected() -> None:
    table = build_feature_table(_stats())
    table["feature_PTS"] = 1
    with pytest.raises(AssertionError):
        assert_no_same_season_features(table)

