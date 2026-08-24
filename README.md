# NBA Fantasy Predictor

This project builds leakage-safe next-season NBA fantasy value projections from locally cached NBA statistics. It supports points-league scoring, nine-category z-scores, strict walk-forward backtests, optional Yahoo draft context, and reproducible Monte Carlo uncertainty bands.

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest
```

The first data fetch requires internet access. After that, cached parquet files under `data/raw/` are used and are never refetched unless `--force` is supplied. The cache manifest records fetch timestamps and row counts.

## Yahoo setup (optional)

1. Create a Yahoo Developer application at <https://developer.yahoo.com/apps/>.
2. Enable Fantasy Sports API access and choose an application type that supports OAuth 2.0.
3. Put the client values in a local `.env` file (never commit it):

```text
YAHOO_CLIENT_ID=your-client-id
YAHOO_CLIENT_SECRET=your-client-secret
```

Run `python -m fantasy yahoo-sync --league-id YOUR_LEAGUE_ID`. The OAuth token is cached in `oauth2.json`, also ignored by Git. Yahoo supplies league scoring settings, ADP/draft context, and optional ownership; it is never used as a statistics source. If credentials are missing, the configured points weights are used and the ADP benchmark is skipped with a clear notice.

## Pipeline

```bash
python -m fantasy fetch --start 2013-14 --end 2024-25
python -m fantasy yahoo-sync --league-id <id>       # optional
python -m fantasy build-features
python -m fantasy backtest
python -m fantasy project --season 2025-26
```

`reports/backtest.csv` contains per-season and pooled MAE, RMSE, Spearman rank correlation, and top-50 hit rate. Baseline B (Marcel) is the primary bar. `output/projections.json` is compact and self-contained for a future static client.

The model target is per-game fantasy value. Monte Carlo percentile bands describe simulated season totals: game-level fantasy values are resampled from cached histories and games played from each player's historical availability distribution. The random seed is configured in `config.yaml`.

## Data and modeling notes

`nba_api` is authoritative for statistics. Features include only lagged values, prior games/minutes/usage, an age descriptor known before the season, prior minutes trend, roster-change flag, limited-history flag, and prior Yahoo ownership. The leakage test fails if a same-season performance field enters the feature matrix. No statistics are invented or hardcoded.

