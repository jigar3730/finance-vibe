# Finance Vibe

## Project Overview

Finance Vibe is a Python pipeline for **coil → expansion signal generation**.
It builds an active ticker universe, ingests OHLCV data, scores Coiled Cobra
setups, and ranks them with informational stock-level context (entry, stop,
targets). It generates *signals*, not tradeable options positions.

The orchestrator is `src/finance_vibe/run_vibe.py`. Macro Vibe Score
(`analysis_engine.py`) runs as its own pipeline stage; Coiled Cobra imports
different functions from the same module (`load_benchmark_frame`,
`relative_strength`, `check_coiled_cobra_market_gate`) for its own RS and
market-gate pillars, not the Vibe Score itself.

## Analysis layers

| Layer | Module | Output |
| ----- | ------ | ------ |
| **Macro** | `analysis_engine.py` | `data/logs/{mode}/vibe_report_<date>.csv` |
| **Breakout readiness (descriptive)** | `breakout_scanner.py` | `breakout_setups_<date>.csv` ([spec](docs/architecture/breakout_scanner.md)) |
| **Coiled Cobra (coil → expansion, primary signal engine)** | `coiled_cobra.py` / `coiled_cobra_backtest.py` | `coiled_cobra_setups_<date>.csv`, `coiled_cobra_backfill_<date>.csv`, `coiled_cobra_backtest_trades_<date>.csv` |

Macro scoring rules: [`docs/handbook/scoring_logic.md`](docs/handbook/scoring_logic.md).

## Documentation

| Section | Path | Contents |
| ------- | ---- | -------- |
| **Catalog** | [`docs/README.md`](docs/README.md) | Full map of the handbook and architecture docs |
| **Handbook** | [`docs/handbook/`](docs/handbook/) | Scoring rubrics and trade-plan calculations |
| **Architecture** | [`docs/architecture/`](docs/architecture/) | Pipeline operations, backtests, breakout scanner |

## Repository structure

```
finance-vibe/
├── src/finance_vibe/          # Application code
├── docs/
│   ├── handbook/              # Scoring rubrics and calculations
│   └── architecture/          # Pipeline ops, backtests, scanners
├── scripts/                   # Cron runners, health checks, test/lint/golden helpers
├── data/
│   ├── active_tickers.csv     # Universe from ticker_provider
│   ├── raw/{weekly|daily}/    # Ingested OHLCV CSVs
│   └── logs/{weekly|daily}/   # Reports, trade plans, backtests
└── tests/
```

## Pipeline flow (`run_vibe.py`)

1. Clean `data/raw/{data_mode}/` unless `--reuse-raw`
2. `ticker_provider.py` → `data/active_tickers.csv` (skipped with `--reuse-raw`)
3. `data_ingestor.py` (weekly) / `daily_ingest.py` (daily; drops today's bar before 17:00 ET) → download OHLCV (skipped with `--reuse-raw`)
4. `analysis_engine.py` → macro Vibe Score (`vibe_report_<date>.csv`)
5. `coiled_cobra.py` → coil scorecard (profile: weekly / daily)
6. `breakout_scanner.py` → pre-breakout readiness scan
7. `trade_planner.py` → entry / stop / target context (signal, not a trade plan)
8. `trade_plan_helper.py` → guardrails, R:R, Expected Value ranking

## Running

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode daily
python src/finance_vibe/run_vibe.py --mode daily --reuse-raw   # keep existing OHLCV; skip wipe + ingest
python src/finance_vibe/run_vibe.py --as-of 2025-11-07         # replay a past date
```

Scheduled runs go through `scripts/run_daily_pipeline.sh` and
`scripts/run_weekly_pipeline.sh` (locking, health checks, email alerts); see
[`docs/architecture/operation_manual.md`](docs/architecture/operation_manual.md).

Individual stages:

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py weekly
python src/finance_vibe/daily_ingest.py                    # daily ingest
python src/finance_vibe/analysis_engine.py weekly          # also runs automatically as pipeline step 4
python src/finance_vibe/coiled_cobra.py weekly
python src/finance_vibe/breakout_scanner.py weekly
python src/finance_vibe/trade_planner.py weekly
python src/finance_vibe/trade_plan_helper.py weekly
```

## Timeframe profiles (`config.py`)

| Mode | Lookback | Interval | Raw path |
| ---- | -------- | -------- | -------- |
| `weekly` (default) | 10y | 1wk | `data/raw/weekly/` |
| `daily` | 5y | 1d | `data/raw/daily/` |

Filenames: `<TICKER>_<period>_<interval>.csv` (e.g. `AAPL_10y_1wk.csv`).

## Output files

| File | Description |
| ---- | ----------- |
| `vibe_report_<date>.csv` | Macro scores for all scanned tickers |
| `coiled_cobra_setups_<date>.csv` | Coil → expansion setups (shared setup schema; weekly/daily) |
| `breakout_setups_<date>.csv` | Breakout readiness states for every scanned ticker |
| `trade_plan_<date>.csv` | Signal + stock-level context (entry/stop/target; no options metadata) |
| `trade_plan_clean_<date>.csv` | Guardrailed plan with R:R, Expected Value, and Priority |
| `ingest_errors_<date>.csv` | Per-ticker ingestion failures (empty/invalid/insufficient data) |

Coiled Cobra emits a shared setup schema (`config.SETUP_ROW_COLUMNS`); raw
CSVs are validated against the OHLCV contract (`config.REQUIRED_OHLCV`) at ingest
and scan time, so malformed files are rejected instead of silently mis-scored.

All outputs live under `data/logs/{mode}/`.

## Macro Vibe Score (summary)

- Scale: **−10 to +10** on the latest bar
- Uses SMA20/50 trend, MACD/RSI momentum, pullback distance from SMA20, CCI cyclical rules, RSI caps
- Full rubric: [`docs/handbook/scoring_logic.md`](docs/handbook/scoring_logic.md)

## Trade planning (summary)

`trade_planner.py` adds informational stock-level context to each Coiled
Cobra signal (not an options trade plan):

- **Entry:** Fib 78.6% floor vs `Close − 0.25×ATR` (Coiled Cobra setups)
- **Stop:** triple-constraint — local 10-session floor vs `1.5×ATR` vs 5% of Close (tightest wins)
- **Targets:** **2R / 3R**
- **Helper:** drop risk > 5% of Close, checklist < 4/6 (Gate D, `MIN_CHECKS_MET/N_SCORED_PILLARS`), or R:R T1 < 2.0; rank survivors by Expected Value (= raw-Score ranking)

## Requirements

- Python 3.12 or 3.13 (3.12 is the reference: the Docker image and `.python-version`)
- [uv](https://docs.astral.sh/uv/). Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`:
  runtime (`pandas`, `numpy`, `yfinance`, `yahooquery`, `Flask`, …) plus the `dev` group (`pytest`).

```bash
uv sync                  # creates .venv with an editable install; no PYTHONPATH needed
uv run pytest -q
uv run finance-vibe --mode weekly --reuse-raw   # = python src/finance_vibe/run_vibe.py ...
```

Without uv: `pip install -r requirements.txt` (exported from `uv.lock`) and
`export PYTHONPATH=src`.

## UI

```bash
python src/finance_vibe/app.py
# http://127.0.0.1:5000
```

Browse trade plans by date and mode (weekly/daily). Breakout scans for both modes:
`http://127.0.0.1:5000/breakout`. Docs: `http://127.0.0.1:5000/docs/`.

## Backtest (offline validation)

`coiled_cobra_backtest.py` is the walk-forward backtest tool: it replays the
Coiled Cobra scorecard across historical OHLC data and simulates stock-level
trade outcomes via `src/finance_vibe/trade_simulator.py` (generic OHLC-bar
fill/stop/target simulation).

```bash
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill   # signal archive
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest   # trade simulation
```

Outputs land under `data/logs/{weekly|daily}/`. Details and limitations:
[`docs/architecture/backtest_and_backfill.md`](docs/architecture/backtest_and_backfill.md).
Stock-level only (no options P&L); not part of `run_vibe.py`.

## Notes

- `run_vibe.py` deletes existing files in `data/raw/{mode}/` before each run unless `--reuse-raw` is passed.
- `trade_plan_helper.py` prefers today’s `trade_plan_<date>.csv`, then the newest dated plan in that log silo.
- Macro and Coiled Cobra layers use different moving averages (SMA vs EMA) by design.

## Further reading

- [`docs/README.md`](docs/README.md) — documentation catalog
- [`docs/architecture/operation_manual.md`](docs/architecture/operation_manual.md) — operations and troubleshooting
- [`docs/handbook/scoring_logic.md`](docs/handbook/scoring_logic.md) — macro score specification
- [`docs/handbook/coiled_cobra_rubric.md`](docs/handbook/coiled_cobra_rubric.md) — Coiled Cobra scorecard specification
- [`docs/architecture/breakout_scanner.md`](docs/architecture/breakout_scanner.md) — breakout readiness scanner
