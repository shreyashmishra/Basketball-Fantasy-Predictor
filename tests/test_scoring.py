import pandas as pd

from fantasy.scoring import add_nine_category_value, points_value


def test_points_scoring_math() -> None:
    row = {"PTS": 20, "REB": 10, "AST": 5, "STL": 2, "BLK": 1, "TOV": 3, "3PM": 2}
    assert points_value(row) == 20 + 12 + 7.5 + 6 + 3 - 3 + 1


def test_nine_category_value_inverts_turnovers() -> None:
    frame = pd.DataFrame({"FG_PCT": [0.5, 0.4], "FT_PCT": [0.8, 0.7], "3PM": [3, 1], "PTS": [20, 10], "REB": [8, 4], "AST": [6, 2], "STL": [1, 0.5], "BLK": [1, 0.2], "TOV": [2, 4], "MIN": [35, 20]})
    result = add_nine_category_value(frame)
    assert result.loc[0, "fantasy_value"] > result.loc[1, "fantasy_value"]

