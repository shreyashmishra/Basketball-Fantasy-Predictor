"""Interchangeable fantasy projection estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FantasyModel(ABC):
    """Small common interface shared by baselines and machine-learning models."""

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FantasyModel":
        """Fit the estimator on pre-season features and historical targets."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict per-game fantasy value."""


class CarryForward(FantasyModel):
    """Baseline A: carry forward the most recent observed value."""

    name = "carry_forward"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CarryForward":
        self.fallback_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["feature_lag_value_1"].fillna(self.fallback_).to_numpy(dtype=float)


class Marcel(FantasyModel):
    """Baseline B: weighted three-year mean with minutes regression and age curve."""

    name = "marcel"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Marcel":
        self.league_mean_ = float(y.mean())
        self.league_mean_minutes_ = 20.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        weights = np.array([5.0, 4.0, 3.0])
        lags = X[["feature_lag_value_1", "feature_lag_value_2", "feature_lag_value_3"]].to_numpy(float)
        valid = ~np.isnan(lags)
        weighted = np.nansum(np.nan_to_num(lags) * weights, axis=1) / np.maximum((valid * weights).sum(axis=1), 1.0)
        minutes = X["feature_minutes_lag_1"].fillna(self.league_mean_minutes_).to_numpy(float)
        # Minutes are a transparent reliability proxy: small samples regress harder.
        reliability = np.clip(minutes / (minutes + self.league_mean_minutes_), 0.0, 1.0)
        result = reliability * weighted + (1.0 - reliability) * self.league_mean_
        age = X["feature_age"].fillna(27.0).to_numpy(float)
        # A deliberately modest curve avoids baking a sharp, unsupported age prior into forecasts.
        result *= 1.0 + np.clip(27.0 - age, -10.0, 10.0) * 0.005
        return result


class YahooADP(FantasyModel):
    """Baseline C: map Yahoo draft rank to a historical value relationship."""

    name = "yahoo_adp"

    def __init__(self, adp_by_player: Mapping[object, float] | None = None) -> None:
        self.adp_by_player = dict(adp_by_player or {})

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "YahooADP":
        self.fallback_ = float(y.mean())
        ranks = pd.Series(self.adp_by_player, dtype=float)
        self.scale_ = float(y.std() / ranks.std()) if len(ranks) > 1 and ranks.std() else 1.0
        self.intercept_ = float(y.mean()) + self.scale_ * float(ranks.mean()) if len(ranks) else self.fallback_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ranks = X["player_id"].map(self.adp_by_player).astype(float)
        return ranks.map(lambda value: self.intercept_ - self.scale_ * value if pd.notna(value) else self.fallback_).to_numpy()


class RidgeModel(FantasyModel):
    """Ridge regression with deterministic imputation and standardization."""

    name = "ridge"

    def __init__(self, seed: int = 42) -> None:
        self.estimator = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)


class GradientBoostingModel(FantasyModel):
    """Histogram gradient boosting regression with a seeded estimator."""

    name = "hist_gradient_boosting"

    def __init__(self, seed: int = 42) -> None:
        self.estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(random_state=seed, max_iter=200, learning_rate=0.05, l2_regularization=0.5),
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)


def default_models(seed: int = 42, adp_by_player: Mapping[object, float] | None = None) -> list[FantasyModel]:
    """Create the full requested model roster in stable report order."""
    return [CarryForward(), Marcel(), YahooADP(adp_by_player), RidgeModel(seed), GradientBoostingModel(seed)]

