"""Walk-forward evaluation and calibration reporting."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import model_matrix
from .models import FantasyModel


def season_year(season: str) -> int:
    """Extract the starting calendar year from an NBA season label."""
    return int(str(season)[:4])


def walk_forward_splits(frame: pd.DataFrame, start: str, end: str) -> Iterable[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield target seasons with training rows strictly earlier than each target."""
    for target in sorted(frame["season"].astype(str).unique(), key=season_year):
        if not (season_year(start) <= season_year(target) <= season_year(end)):
            continue
        train = frame[frame["season"].map(season_year) < season_year(target)].copy()
        test = frame[frame["season"].astype(str) == target].copy()
        if train.empty or test.empty:
            continue
        yield target, train, test


def _rank_correlation(actual: pd.Series, predicted: np.ndarray) -> float:
    """Calculate Spearman correlation while handling constant vectors."""
    if len(actual) < 2 or actual.nunique() < 2 or len(np.unique(predicted)) < 2:
        return float("nan")
    return float(spearmanr(actual.to_numpy(float), predicted).statistic)


def _top_hit_rate(actual: pd.Series, predicted: np.ndarray, top_n: int) -> float:
    """Return overlap divided by top-N actual players."""
    if len(actual) == 0:
        return float("nan")
    n = min(top_n, len(actual))
    actual_top = set(actual.nlargest(n).index)
    predicted_top = set(pd.Series(predicted, index=actual.index).nlargest(n).index)
    return len(actual_top & predicted_top) / n


def metric_row(season: str, model: str, actual: pd.Series, predicted: np.ndarray, top_n: int = 50) -> dict[str, object]:
    """Return the required metrics for one model-season prediction vector."""
    errors = predicted - actual.to_numpy(float)
    return {
        "season": season,
        "model": model,
        "n_players": len(actual),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "spearman": _rank_correlation(actual, predicted),
        "top_50_hit_rate": _top_hit_rate(actual, predicted, top_n),
    }


def run_backtest(
    frame: pd.DataFrame,
    models: list[FantasyModel],
    start: str,
    end: str,
    min_games: int = 20,
    min_minutes: float = 15.0,
    top_n: int = 50,
) -> pd.DataFrame:
    """Run strict walk-forward backtesting and return per-season plus pooled rows."""
    rows: list[dict[str, object]] = []
    for target, train, test in walk_forward_splits(frame, start, end):
        eligible = test[(test["games_played"] >= min_games) & (test["minutes_per_game"] >= min_minutes)].copy()
        if eligible.empty:
            continue
        train_x, train_y = model_matrix(train), train["target_value"]
        test_x, actual = model_matrix(eligible), eligible["target_value"]
        for prototype in models:
            model = type(prototype)(**_constructor_args(prototype))
            fit_x, predict_x = train_x, test_x
            # ADP is an identity lookup rather than a numeric feature. Keep it
            # outside the approved matrix so it cannot leak into learned models.
            if model.name == "yahoo_adp":
                fit_x = train_x.copy()
                predict_x = test_x.copy()
                fit_x["player_id"] = train["player_id"].to_numpy()
                predict_x["player_id"] = eligible["player_id"].to_numpy()
            model.fit(fit_x, train_y)
            rows.append(metric_row(target, model.name, actual, model.predict(predict_x), top_n))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    pooled: list[dict[str, object]] = []
    for name, group in result[result["season"] != "pooled"].groupby("model"):
        # MAE/RMSE/top-N are weighted by the player count; rank correlations are averaged.
        pooled.append({
            "season": "pooled",
            "model": name,
            "n_players": int(group["n_players"].sum()),
            "mae": float(np.average(group["mae"], weights=group["n_players"])),
            "rmse": float(np.sqrt(np.average(group["rmse"] ** 2, weights=group["n_players"]))),
            "spearman": float(group["spearman"].mean()),
            "top_50_hit_rate": float(np.average(group["top_50_hit_rate"], weights=group["n_players"])),
        })
    return pd.concat([result, pd.DataFrame(pooled)], ignore_index=True)


def _constructor_args(model: FantasyModel) -> dict[str, object]:
    """Copy supported constructor state when refitting a model per season."""
    if model.name == "yahoo_adp":
        return {"adp_by_player": getattr(model, "adp_by_player", {})}
    if model.name in {"ridge", "hist_gradient_boosting"}:
        return {"seed": 42}
    return {}


def save_report(report: pd.DataFrame, path: str | Path) -> None:
    """Write a backtest report as CSV, creating its parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(target, index=False)


def calibration_fraction(actual: Iterable[float], lower: Iterable[float], upper: Iterable[float]) -> float:
    """Return the fraction of actual values inside supplied prediction intervals."""
    actual_array, low_array, high_array = (np.asarray(list(values), dtype=float) for values in (actual, lower, upper))
    if len(actual_array) == 0:
        return float("nan")
    return float(np.mean((actual_array >= low_array) & (actual_array <= high_array)))
