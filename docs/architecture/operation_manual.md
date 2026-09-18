# Finance Vibe Operation Manual

## Status

Stable

## Purpose

This manual describes how to operate the Finance Vibe pipeline: data ingestion, macro scoring, tactical scanning, trade plan generation, and optional UI review.

## Environment

- Python 3.10+
- Install from repo root:

```bash
python -m pip install -r requirements.txt
```

Core packages: `pandas`, `numpy`, `pandas_ta`, `yfinance`, `yahooquery`, `Flask`.

ML baseline extras (in `requirements.txt`): `xgboost`, `lightgbm`, `scikit-learn`, `matplotlib`.

Compatible with the dev container or any standard Python environment with `PYTHONPATH=./src`.

## Directory layout

| Path | Contents |
| ---- | -------- |
| `src/finance_vibe/` | Pipeline source |
| `docs/` | Handbook, architecture, and labs ([catalog](../README.md)) |
| `data/active_tickers.csv` | Ticker universe |
| `data/raw/weekly/` | Weekly OHLCV CSVs (`*_10y_1wk.csv`) |
| `data/raw/daily/` | Daily OHLCV CSVs (`*_5y_1d.csv`) |
| `data/logs/weekly/` | Weekly reports and trade plans |
| `data/logs/daily/` | Daily reports and trade plans |
| `data/logs/high_beta/` | High-beta signal profile logs (shares daily raw OHLCV) |

## Standard operating procedure

### Full pipeline

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode daily
python src/finance_vibe/run_vibe.py --mode high_beta
python src/finance_vibe/run_vibe.py --mode daily --reuse-raw
```

Execution order (`run_vibe.py`):

| Step | Script | Output |
| ---- | ------ | ------ |
| 0 | (orchestrator) | Clears `data/raw/{data_mode}/` (skipped with `--reuse-raw`) |
| 1 | `ticker_provider.py` | `data/active_tickers.csv` (skipped with `--reuse-raw`) |
| 2 | `data_ingestor.py` | Raw CSVs in `data/raw/{data_mode}/` (skipped with `--reuse-raw`) |
| 3 | `analysis_engine.py` | `vibe_report_<date>.csv` (macro Vibe Score) |
| 4 | `coiled_cobra.py` | `coiled_cobra_setups_<date>.csv` — primary signal engine, runs for every mode |
| 5 | `breakout_scanner.py` | `breakout_setups_<date>.csv` |
| 6 | `trade_planner.py` | `trade_plan_<date>.csv` |
| 7 | `trade_plan_helper.py` | `trade_plan_clean_<date>.csv` |

`analysis_engine.py` runs as its own pipeline stage (step 3), producing
`vibe_report_<date>.csv`. Coiled Cobra does not consume the Vibe Score itself
for gating — it imports `load_benchmark_frame` / `relative_strength` /
`check_coiled_cobra_market_gate` from the same module for its own RS and
market-gate pillars, which is a separate (trend + relative-strength) check.

Coiled Cobra is the project's single tactical signal engine (the former
`swing_scanner.py`, which duplicated indicator logic Coiled Cobra already
covered, has been decommissioned). The pipeline produces ranked *signals*,
not options trade plans: `trade_planner.py`/`trade_plan_helper.py` add
stock-level context and an Expected-Value ranking, not LEAPS/options strike
metadata.

### Manual stages

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py weekly
python src/finance_vibe/analysis_engine.py weekly
python src/finance_vibe/coiled_cobra.py weekly
python src/finance_vibe/trade_planner.py weekly
python src/finance_vibe/trade_plan_helper.py weekly
```

Replace `weekly` with `daily` or `high_beta` — `coiled_cobra.py` now supports
all three (`high_beta` reads daily OHLCV and writes to its own
`data/logs/high_beta/` silo).

## Script reference

### `ticker_provider.py`

- Merges `STATIC_TICKERS` (`config.py`), `ticker_manifest.csv`, and Yahoo Finance screeners
- Writes `data/active_tickers.csv` (capped at `ACTIVE_TICKER_CAP` = **1000** in `config.py`)

### `data_ingestor.py`

- Reads active tickers; downloads via `yfinance` using `TIMEFRAME_PROFILES` in `config.py`
- Drops incomplete weekly candles (last bar if not Friday)
- Uses `auto_adjust=True` for split/dividend-adjusted prices

### `analysis_engine.py` (macro layer)

- Scores every raw CSV in `data/raw/{mode}/` on a **−10 to +10** Vibe Score
- Requires ≥ 60 bars per file
- Parallel scan via `ProcessPoolExecutor`
- Runs as pipeline step 3 (`vibe_report_<date>.csv`); Coiled Cobra does not
  consume this score for gating (see note above)
- **Specification:** [`scoring_logic.md`](../handbook/scoring_logic.md)

### `coiled_cobra.py` (primary signal engine: coil → expansion)

- Filters to symbols in `active_tickers.csv`
- 100-pt v4.0 scorecard, hard-gated: long-term trend template, ticker market
  gate, coil integrity (structure/volatility-contraction independently
  gating), and breadth (Checks Met >= 5/6); BBWidth-percentile volatility
  contraction replaces the old MACD-spread squeeze proxy
- Profiles: `weekly`, `daily`, `high_beta` (`high_beta` reads daily OHLCV via
  `config.resolve_pipeline_mode()`, same bar-frequency calibration as
  `daily`, own `data/logs/high_beta/` silo)
- **Specification:** [`coiled_cobra_rubric.md`](../handbook/coiled_cobra_rubric.md)

### `trade_planner.py` (signal-ranking stage, not a trade planner)

- Reads **today's** `coiled_cobra_setups_<date>.csv`
- Adds informational stock-level context: Fib 78.6% entry floor, local
  10-bar stop, **2R/3R** targets (`calculate_stock_levels()`)
- No options/LEAPS output — the project generates signals, not tradeable
  option positions

### `trade_plan_helper.py`

- Loads `trade_plan_{today}.csv`, else newest dated plan in the mode log dir
- Adds Risk Per Share and direction-aware R:R
- Drops risk > 5% of Close, cobra checklist below coiled_cobra's own Gate D
  breadth floor (`MIN_CHECKS_MET/N_SCORED_PILLARS`, currently 4/6), or R:R T1 < 2.0
- Ranks survivors by Expected Value / `ML_Pred_Return` with a 1.25 coil propensity
- Writes `trade_plan_clean_<date>.csv`

### `src/finance_vibe/trade_simulator.py` (library, not a script)

Generic OHLC-bar trade simulation primitives (`simulate_trade`,
`simulate_scaled_trade`, `passes_macro_gate`) extracted from the former
`pipeline_backtest.py` (swing-specific walk-forward harness, removed
alongside `swing_scanner.py`). `coiled_cobra_backtest.py` is the only
walk-forward backtest tool now; it imports `simulate_trade` from here.

### `src/finance_vibe/coiled_cobra_backtest.py` (Coiled Cobra historical)

Walk-forward validation and historical backfill for the Coiled Cobra coil → expansion scanner.

- `--backfill` exports a historical Coiled Cobra signal archive to `data/logs/{mode}/coiled_cobra_backfill_<date>.csv`.
- `--backtest` runs a walk-forward stock-level backtest and writes `data/logs/{mode}/coiled_cobra_backtest_trades_<date>.csv`.

Usage (container recommended):

```bash
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
```

Notes:
- Uses the same `evaluate_coiled_cobra()` engine as the live scanner but evaluates every eligible historical bar.
- `trade_planner.py` recognizes `Source` == `coiled_cobra` so Coiled Cobra setups use Fib 78.6% entry logic.
- Details and recipes: **[`backtest_and_backfill.md`](backtest_and_backfill.md)**.

### `src/finance_vibe/coiled_cobra_ml_training.py` (offline ML)

Trains XGBoost + LightGBM regressors to predict `Forward_Return_2w` from Coiled Cobra backtest exports.

- Input: `data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv` (from `--backtest`)
- Features: 6 pre-signal columns (`Score`, EMA/Fib distances, `ATR_Pct`); `Grade` excluded
- Split: rolling 26-week test / 26-week val / rest train on `Signal Date` (no random K-fold)
- `MODEL_PARAMS`: `max_depth=4`, `learning_rate=0.01`, `n_estimators=400`, `subsample=0.8`, `colsample_bytree=0.8`
- Objectives: XGBoost `reg:absoluteerror`, LightGBM `regression_l1`; `sample_weight=ATR_Pct`
- Output: artifacts (`xgb`/`lgb` + metadata JSON) + PNG; `ml_ranker.py` attaches soft ranks on live scans

```bash
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_2026-07-17.csv
```

Not part of the default pipeline. Full specification: **[`coiled_cobra_ml.md`](coiled_cobra_ml.md)**.

## UI dashboard

```bash
python src/finance_vibe/app.py
```

Open `http://127.0.0.1:5000` to browse trade plans by mode and date.

## Data maintenance

Force re-download for one mode:

```bash
rm data/raw/weekly/*.csv
python src/finance_vibe/run_vibe.py
```

## Troubleshooting

| Symptom | Action |
| ------- | ------ |
| Missing `active_tickers.csv` | Run `ticker_provider.py` |
| Ingest skips a symbol | No yfinance data for that ticker; check symbol validity |
| Empty `coiled_cobra_setups_*.csv` | No tickers matched the coil scorecard (expected in quiet markets) |
| `trade_plan_helper` file not found | Run the planner first; helper prefers today’s file then falls back to the newest dated plan |
| Macro report missing tickers | Check ingest logs; file needs ≥ 60 rows |
| ML script cannot find trades CSV | Run `coiled_cobra_backtest.py weekly --backtest`; pass `--csv`; see **[`coiled_cobra_ml.md`](coiled_cobra_ml.md)** |

## Extending the project

### Add tickers

- Edit `STATIC_TICKERS` in `config.py`, or
- Add rows to `ticker_manifest.csv`

### Change lookback or cadence

Edit `TIMEFRAME_PROFILES` in `config.py`:

```python
"weekly": {"period": "10y", "interval": "1wk", ...}
"daily":  {"period": "5y",  "interval": "1d",  ...}
```

### Change scoring or setup rules

1. Macro: edit `score_last_row()` in `analysis_engine.py`; update [`scoring_logic.md`](../handbook/scoring_logic.md)
2. Tactical: edit `evaluate_coiled_cobra()` in `coiled_cobra.py`; update [`coiled_cobra_rubric.md`](../handbook/coiled_cobra_rubric.md)
3. Signal levels: edit `calculate_stock_levels()` in `trade_planner.py`

## Output files

| File | Layer |
| ---- | ----- |
| `vibe_report_<date>.csv` | Macro (manual run) |
| `coiled_cobra_setups_<date>.csv` | Signal (weekly/daily/high_beta) |
| `trade_plan_<date>.csv` | Signal + stock-level context |
| `trade_plan_clean_<date>.csv` | Signal (guardrailed + ranked) |
| `backtest_trades_<date>.csv` | Offline backtest (manual run) |
| `coiled_cobra_backtest_trades_<date>.csv` | Coiled Cobra walk-forward trades (ML source) |
| `coiled_cobra_ml_feature_importance.png` | ML feature-importance chart (manual ML run) |

## Notes

- Each pipeline run clears `data/raw/{mode}/` before ingestion unless `--reuse-raw` is passed.
- Macro (SMA) and tactical (EMA) indicators are intentionally different.
- `trade_plan_helper.py` prefers today’s dated plan, then the newest `trade_plan_*.csv` in that silo.
