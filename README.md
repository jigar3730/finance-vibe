# Finance Vibe

Two stock scanners over a Yahoo Finance OHLCV universe, run weekly and daily,
with a Flask dashboard to review their output:

| Scanner | What it finds | Output | Spec |
| ------- | ------------- | ------ | ---- |
| **Coiled Cobra** | Compressed leaders likely to expand: four hard gates and a 100-point coil scorecard | `coiled_cobra_setups_<date>.csv` | [`docs/coiled_cobra.md`](docs/coiled_cobra.md) |
| **Breakout readiness** | Each ticker's trend / volatility / volume / momentum / structure state and breakout status (descriptive) | `breakout_setups_<date>.csv` | [`docs/breakout.md`](docs/breakout.md) |

Both produce *signals* for review, not trades. The backtest found no proven edge in them
(`RESEARCH.md`), so the dashboard's home page is daily market context (SPY / QQQ / SCHG /
SCHD facts, VIX, credit and sector breadth); the scans live on a back page, Setups (unproven).

## Quick start

```bash
uv sync                                         # .venv with an editable install
uv run python -m finance_vibe.pipeline          # weekly: refresh tickers + data, run both scanners
uv run python -m finance_vibe.pipeline --mode daily
uv run python -m finance_vibe.pipeline --reuse-raw          # keep raw data; scanners only
uv run python -m finance_vibe.pipeline --as-of 2025-11-07   # replay a past date
uv run python -m finance_vibe.web.app           # dashboard: market home, changes, setups (unproven), run pages at http://127.0.0.1:5000
uv run pytest -q
```

Without uv: `pip install -r requirements.txt` (exported from `uv.lock`) and
`export PYTHONPATH=src`.

In production the pipeline runs in the `finance_vibe` container on a cron
schedule (weekly Fridays 18:00 ET, daily 18:30 ET with a 22:30 retry), and
`docker compose up -d` serves the dashboard on port 5000. See
[`docs/operations.md`](docs/operations.md).

## Pipeline

1. Wipe `data/raw/{mode}/` (skipped with `--reuse-raw` / `--as-of`)
2. `data.tickers` → `data/active_tickers.csv`
3. `data.ingest` (weekly) / `data.daily_ingest` (daily) → `data/raw/{mode}/<TICKER>_<period>_<interval>.csv`, plus a bar-quality report `data/logs/{mode}/data_quality_<date>.csv`
4. `scanners.coiled_cobra` → `data/logs/{mode}/coiled_cobra_setups_<date>.csv`
5. `scanners.breakout` → `data/logs/{mode}/breakout_setups_<date>.csv`
6. `data.market_ingest` (daily only) → VIX, HYG, IEF and the 11 sector funds for the market home page, in their
   own folder `data/raw/market/` (the scanners never see them), plus `data/logs/market/data_quality_<date>.csv`

| Mode | Lookback | Interval |
| ---- | -------- | -------- |
| `weekly` (default) | 10y | 1wk |
| `daily` | 5y | 1d |

## Layout

```
src/finance_vibe/
  config.py           modes, paths, as-of cut, OHLCV contract, setup schema
  log.py              logging setup
  indicators.py       EMA/SMA/RSI/ATR/MACD/BB/KC/OBV (pandas-ta 0.4.71b0 semantics)
  market.py           QQQ benchmark frame, relative strength, Cobra market gate
  pipeline.py         orchestrator (python -m finance_vibe.pipeline)
  data/               tickers.py, ingest.py, daily_ingest.py, market_ingest.py, raw.py, quality.py, ticker_manifest.csv
  scanners/           coiled_cobra.py, breakout.py
  web/                app.py, playbook.py (Setups page sections, Market rows), health.py (VIX, credit, sector breadth), insights.py (freshness, changes, ETF relative view), templates/, static/
scripts/              cron runners, health checks, email alerts, test/lint/golden helpers
tests/                pytest suite
docs/                 scanner specs and operations
```

## Requirements

- Python 3.12 or 3.13 (3.12 is the reference: the Docker image and `.python-version`)
- [uv](https://docs.astral.sh/uv/); dependencies in `pyproject.toml`, pinned in `uv.lock`
