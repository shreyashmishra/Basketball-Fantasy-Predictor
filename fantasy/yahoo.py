"""Optional Yahoo Fantasy metadata integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True)
class YahooSyncResult:
    """Result of an optional Yahoo synchronization attempt."""

    enabled: bool
    message: str
    scoring: dict[str, float] | None = None
    adp: list[dict[str, Any]] | None = None
    percent_owned: list[dict[str, Any]] | None = None


def yahoo_available(env_path: str | Path = ".env") -> bool:
    """Return whether both Yahoo OAuth client values are configured."""
    load_dotenv(env_path)
    return bool(os.getenv("YAHOO_CLIENT_ID") and os.getenv("YAHOO_CLIENT_SECRET"))


def sync_league(league_id: str, env_path: str | Path = ".env", token_path: str | Path = "oauth2.json") -> YahooSyncResult:
    """Synchronize Yahoo metadata, or cleanly return the documented fallback.

    Yahoo packages have changed response shapes over time, so this adapter keeps
    the rest of the pipeline independent of their object model. Credentials and
    tokens never enter source control.
    """
    load_dotenv(env_path)
    if not yahoo_available(env_path):
        return YahooSyncResult(False, "Yahoo credentials absent; using config.yaml scoring and skipping ADP.")
    try:
        from yahoo_oauth import OAuth2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Yahoo credentials are configured but yahoo_oauth is not installed") from exc
    oauth = OAuth2(None, None, from_file=str(token_path))
    if not oauth.token_is_valid():
        oauth.refresh_access_token()
    Path(token_path).write_text(json.dumps(oauth.token), encoding="utf-8")
    # The metadata endpoints are deliberately isolated here. Teams can replace
    # this small adapter if Yahoo changes its API without touching modeling code.
    try:
        from yahoo_fantasy_api import Game

        game = Game(oauth, "nba")
        league = game.to_league(league_id)
        settings = league.settings()
        scoring = _settings_to_weights(settings)
        return YahooSyncResult(True, f"Yahoo metadata synchronized for league {league_id}.", scoring=scoring)
    except Exception as exc:  # pragma: no cover - requires live Yahoo account
        raise RuntimeError(f"Yahoo synchronization failed for league {league_id}") from exc


def _settings_to_weights(settings: Any) -> dict[str, float]:
    """Translate common Yahoo stat-setting shapes into project stat weights."""
    result: dict[str, float] = {}
    for item in settings if isinstance(settings, list) else settings.get("stat_categories", []):
        key = str(item.get("display_name", item.get("stat_id", ""))).upper()
        value = item.get("modifier", item.get("value", 0))
        aliases = {"POINTS": "PTS", "REBOUNDS": "REB", "ASSISTS": "AST", "STEALS": "STL", "BLOCKS": "BLK", "TURNOVERS": "TOV", "THREES": "3PM"}
        if key in aliases:
            result[aliases[key]] = float(value)
    return result

