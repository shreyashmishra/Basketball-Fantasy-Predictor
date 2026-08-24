import pandas as pd

from fantasy.backtest import season_year, walk_forward_splits


def test_walk_forward_training_is_strictly_prior() -> None:
    frame = pd.DataFrame({"season": ["2017-18", "2018-19", "2019-20"], "player_id": [1, 1, 1]})
    splits = list(walk_forward_splits(frame, "2018-19", "2019-20"))
    assert [target for target, _, _ in splits] == ["2018-19", "2019-20"]
    assert all(all(season_year(s) < season_year(target) for s in train.season) for target, train, _ in splits)

