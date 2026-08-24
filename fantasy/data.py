"""NBA API fetching, local parquet caching, and canonical column normalization."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .scoring import points_value

STAT_COLUMNS = {
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team",
    "GP": "games_played",
    "MIN": "minutes_per_game",
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "TOV": "TOV",
    "FG_PCT": "FG_PCT",
    "FT_PCT": "FT_PCT",
    "FG3M": "3PM",
    "USG_PCT": "usage_rate",
    "AGE": "age",
}


def season_range(start: str, end: str) -> list[str]:
    """Return NBA season labels from inclusive ``start`` to inclusive ``end``."""
    start_year = int(start[:4])
    end_year = int(end[:4])
    if end_year < start_year:
        raise ValueError("end season must not precede start season")
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year + 1)]


class DataCache:
    """Parquet cache with a JSON manifest for reproducible fetch auditing."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"

    def path(self, kind: str, season: str) -> Path:
        """Return the deterministic cache path for a data kind and season."""
        safe_season = season.replace("-", "_")
        return self.directory / f"{kind}_{safe_season}.parquet"

    def read(self, kind: str, season: str) -> pd.DataFrame | None:
        """Read a cached frame, or return ``None`` when it is not cached."""
        target = self.path(kind, season)
        return pd.read_parquet(target) if target.exists() else None

    def write(self, kind: str, season: str, frame: pd.DataFrame) -> Path:
        """Write a frame and update the fetch manifest."""
        target = self.path(kind, season)
        frame.to_parquet(target, index=False)
        manifest = self._manifest()
        manifest[f"{kind}:{season}"] = {
            "path": str(target),
            "rows": int(len(frame)),
            "fetched_at": datetime.now(UTC).isoformat(),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


def normalize_stats(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    """Normalize NBA API aggregate columns to the project's canonical schema."""
    result = frame.rename(columns=STAT_COLUMNS).copy()
    result["season"] = season
    required = ["player_id", "player_name", "team", "games_played", "minutes_per_game"]
    missing = [column for column in required if column not in result]
    if missing:
        raise ValueError(f"NBA stats response is missing columns: {', '.join(missing)}")
    for column in STAT_COLUMNS.values():
        if column in result and column not in {"player_name", "team"}:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _endpoint_frame(endpoint: Any) -> pd.DataFrame:
    """Extract the first NBA API result set as a pandas frame."""
    try:
        return endpoint.get_data_frames()[0]
    except Exception as exc:  # pragma: no cover - exercised only with API failures
        raise RuntimeError("NBA API returned an unreadable response") from exc


def fetch_season(cache: DataCache, season: str, force: bool = False, sleep_seconds: float = 0.6) -> pd.DataFrame:
    """Fetch and cache one season of aggregates and game logs from nba_api.

    The API dependency is imported only when a cache miss occurs, so all local
    feature/model/test workflows remain usable without network credentials.
    """
    aggregate = cache.read("season_stats", season) if not force else None
    logs = cache.read("game_logs", season) if not force else None
    if aggregate is not None and logs is not None:
        return aggregate
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats, playergamelogs
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError("nba_api is required for uncached fetches; install project dependencies") from exc
    if aggregate is None:
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="PerGame", season_type_all_star="Regular Season"
        )
        aggregate = normalize_stats(_endpoint_frame(endpoint), season)
        cache.write("season_stats", season, aggregate)
        time.sleep(sleep_seconds)
    if logs is None:
        endpoint = playergamelogs.PlayerGameLogs(
            season_nullable=season, season_type_nullable="Regular Season", league_id_nullable="00"
        )
        logs = _endpoint_frame(endpoint)
        logs["season"] = season
        cache.write("game_logs", season, logs)
    return aggregate


def fetch_range(cache: DataCache, start: str, end: str, force: bool = False) -> pd.DataFrame:
    """Fetch a season range and return concatenated canonical aggregates."""
    frames = [fetch_season(cache, season, force=force) for season in season_range(start, end)]
    return pd.concat(frames, ignore_index=True)


def read_cached_aggregates(cache: DataCache, start: str, end: str) -> pd.DataFrame:
    """Read an already-fetched season range, failing clearly if data is absent."""
    frames = []
    for season in season_range(start, end):
        frame = cache.read("season_stats", season)
        if frame is None:
            raise FileNotFoundError(f"No cached season data for {season}; run `python -m fantasy fetch` first")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def read_cached_game_history(
    cache: DataCache,
    start: str,
    end: str,
    weights: dict[str, float] | None = None,
) -> tuple[dict[object, list[float]], dict[object, list[float]]]:
    """Read cached game logs as per-player fantasy values and availability history."""
    values: dict[object, list[float]] = {}
    games: dict[object, list[float]] = {}
    for season in season_range(start, end):
        logs = cache.read("game_logs", season)
        if logs is None:
            continue
        normalized = logs.rename(columns=STAT_COLUMNS)
        if "player_id" not in normalized or "PTS" not in normalized:
            continue
        for record in normalized.to_dict("records"):
            player_id = record["player_id"]
            values.setdefault(player_id, []).append(points_value(record, weights))
    aggregates = read_cached_aggregates(cache, start, end)
    for player_id, group in aggregates.groupby("player_id"):
        games[player_id] = group["games_played"].dropna().astype(float).tolist()
    return values, games
