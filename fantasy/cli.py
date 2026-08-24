"""Typer command line entry points for the local pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from .backtest import run_backtest, save_report
from .config import ensure_directories, load_settings
from .data import DataCache, fetch_range, read_cached_aggregates, read_cached_game_history
from .features import build_feature_table, build_projection_features, model_matrix
from .models import Marcel, RidgeModel
from .monte_carlo import simulate_season_totals, summarize_simulation
from .scoring import scoring_config
from .yahoo import sync_league

app = typer.Typer(help="Local, reproducible NBA fantasy value projection pipeline.")


def _settings(config: str) -> object:
    settings = load_settings(config)
    ensure_directories(settings)
    return settings


@app.command()
def fetch(
    start: str = typer.Option("2013-14"),
    end: str = typer.Option("2024-25"),
    force: bool = typer.Option(False, help="Refetch cached seasons."),
    config: str = typer.Option("config.yaml"),
) -> None:
    """Fetch season aggregates and game logs into the local parquet cache."""
    settings = _settings(config)
    frame = fetch_range(DataCache(settings.raw_dir), start, end, force=force)
    typer.echo(f"Fetched {len(frame):,} season-player rows into {settings.raw_dir}.")


@app.command("yahoo-sync")
def yahoo_sync(league_id: str, config: str = typer.Option("config.yaml")) -> None:
    """Sync optional Yahoo scoring metadata and draft context."""
    result = sync_league(league_id)
    typer.echo(result.message)
    if result.scoring:
        typer.echo(json.dumps(result.scoring, indent=2, sort_keys=True))


@app.command("build-features")
def build_features(config: str = typer.Option("config.yaml")) -> None:
    """Build the leakage-safe season feature table."""
    settings = _settings(config)
    stats = read_cached_aggregates(DataCache(settings.raw_dir), "2013-14", "2024-25")
    table = build_feature_table(stats, scoring_config(settings.scoring))
    target = settings.processed_dir / "features.parquet"
    table.to_parquet(target, index=False)
    typer.echo(f"Wrote {len(table):,} feature rows to {target}.")


@app.command()
def backtest(config: str = typer.Option("config.yaml")) -> None:
    """Run strict walk-forward evaluation and write reports/backtest.csv."""
    settings = _settings(config)
    feature_path = settings.processed_dir / "features.parquet"
    if not feature_path.exists():
        raise typer.BadParameter("features.parquet is missing; run build-features first")
    frame = pd.read_parquet(feature_path)
    report = run_backtest(frame, [Marcel(), RidgeModel(settings.seed)], settings.backtest_start, settings.backtest_end, settings.min_games, settings.min_minutes, settings.top_n)
    path = settings.reports_dir / "backtest.csv"
    save_report(report, path)
    typer.echo(report.to_string(index=False))
    typer.echo(f"\nWrote {path}.")


@app.command()
def project(season: str = typer.Option("2025-26"), config: str = typer.Option("config.yaml")) -> None:
    """Project a future season and write compact percentile-band JSON."""
    settings = _settings(config)
    feature_path = settings.processed_dir / "features.parquet"
    if not feature_path.exists():
        raise typer.BadParameter("features.parquet is missing; run build-features first")
    features = pd.read_parquet(feature_path)
    stats = read_cached_aggregates(DataCache(settings.raw_dir), "2013-14", "2024-25")
    game_values, availability = read_cached_game_history(DataCache(settings.raw_dir), "2013-14", "2024-25", scoring_config(settings.scoring))
    future = build_projection_features(stats, features, season)
    train = features[features["season"].map(lambda value: int(str(value)[:4])) < int(season[:4])]
    model = RidgeModel(settings.seed).fit(model_matrix(train), train["target_value"])
    future["projected_value"] = model.predict(model_matrix(future))
    # Game logs are optional at projection time; the fallback still produces a
    # valid uncertainty band and is replaced automatically when cached logs exist.
    results = []
    for row in future.itertuples(index=False):
        projected = float(row.projected_value)
        totals = simulate_season_totals(
            projected,
            game_values.get(row.player_id, [projected]),
            availability.get(row.player_id, [82]),
            settings.monte_carlo_draws,
        )
        summary = summarize_simulation(projected, totals)
        results.append({"player_id": row.player_id, "player_name": row.player_name, "projected_value": projected, "p05": summary.p05, "p25": summary.p25, "p50": summary.median, "p75": summary.p75, "p95": summary.p95})
    output = settings.output_dir / "projections.json"
    output.write_text(json.dumps({"season": season, "projections": results}, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {len(results):,} projections to {output}.")


if __name__ == "__main__":
    app()
