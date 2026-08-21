"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    """Validated application settings loaded from YAML."""

    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    output_dir: Path
    scoring: dict[str, Any]
    seed: int
    monte_carlo_draws: int
    min_games: int
    min_minutes: float
    top_n: int
    league_mean_minutes: float
    backtest_start: str
    backtest_end: str


def load_settings(path: str | Path = "config.yaml") -> Settings:
    """Read YAML configuration and return absolute-path-independent settings."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    paths = data.get("paths", {})
    model = data.get("model", {})
    seasons = data.get("seasons", {})
    return Settings(
        raw_dir=Path(paths.get("raw", "data/raw")),
        processed_dir=Path(paths.get("processed", "data/processed")),
        reports_dir=Path(paths.get("reports", "reports")),
        output_dir=Path(paths.get("output", "output")),
        scoring=data.get("scoring", {}),
        seed=int(model.get("seed", 42)),
        monte_carlo_draws=int(model.get("monte_carlo_draws", 10_000)),
        min_games=int(model.get("min_games", 20)),
        min_minutes=float(model.get("min_minutes", 15.0)),
        top_n=int(model.get("top_n", 50)),
        league_mean_minutes=float(model.get("league_mean_minutes", 20.0)),
        backtest_start=str(seasons.get("backtest_start", "2017-18")),
        backtest_end=str(seasons.get("backtest_end", "2024-25")),
    )


def ensure_directories(settings: Settings) -> None:
    """Create output directories required by local pipeline commands."""
    for directory in (settings.raw_dir, settings.processed_dir, settings.reports_dir, settings.output_dir):
        directory.mkdir(parents=True, exist_ok=True)

