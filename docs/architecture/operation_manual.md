# Finance Vibe Operation Manual

## Status

Stable

## Purpose

This manual describes how to operate the Finance Vibe pipeline: data ingestion, macro scoring, Coiled Cobra signal scanning, breakout-readiness scanning, signal ranking, and UI review.

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
| `templates/`, `src/finance_vibe/static/` | Flask dashboard templates and CSS |
| `tests/` | pytest suite |

## Standard operating procedure

### Full pipeline

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode daily
python src/finance_vibe/run_vibe.py --mode high_beta
python src/finance_vibe/run_vibe.py --mode daily --reuse-raw
python src/finance_vibe/run_vibe.py --as-of 2025-11-07
```

### Scheduled weekly run

The weekly pipeline runs automatically on **Fridays at 18:00 America/New_York**
through the host crontab (user `jigar`):

```cron
0 22,23 * * 5 /opt/stacks/finance-vibe/scripts/run_weekly_pipeline.sh
```

The host clock is UTC, so cron fires at both 22:00 and 23:00 UTC.
`scripts/run_weekly_pipeline.sh` exits unless it is the 18:00 hour in New York,
so the run stays at 6 PM through EDT and EST. It runs
`run_vibe.py --mode weekly` inside the `finance_vibe` container, holds a lock so
runs cannot overlap, and writes one log per run to `~/.local/state/finance-vibe/`.
Use `scripts/run_weekly_pipeline.sh --now` for a manual run.

**Failure alerts (email).** The runner emails an alert, with the end of the run log, when:

- `run_vibe.py` exits non-zero (including the container being down);
- the run exits 0 but `scripts/check_weekly_outputs.py` finds a problem: a missing
  output for today (vibe report, Cobra setups, breakout setups, trade plan, clean plan),
  SPY/QQQ's newest weekly bar is not the week that just closed, or more than 20% of
  tickers failed to ingest;
- a run is skipped because the previous one still holds the lock.

Email is sent from the host by `scripts/notify_email.py` (stdlib only, so it works with
the container down). It uses SMTP settings in `~/.config/finance-vibe/notify.env`
(mode 600, not in the repo), with the same `SMTP_*` / `EMAIL_*` names as quant-hub. If
that file is missing, the alert is logged as not sent. Run the health check by hand with
`docker exec -i finance_vibe python - < scripts/check_weekly_outputs.py`.

The container runs the code baked into its image. After changing pipeline
code, rebuild it with `docker compose up -d --build` so the next scheduled run
picks up the change.

The ingestor keeps the current week's bar once that Friday's close has passed
(17:00 ET onward; `data_ingestor.weekly_bar_is_complete`), so a Friday-evening
run scans the week that just closed. Before 2026-10-01, a bug dropped every
final weekly bar, because bars are Monday-dated and the check required a Friday
date. Runs up to then, including the 2026-09-19 run, scanned the previous week.

### Replaying a past week (`--as-of`)

`--as-of YYYY-MM-DD` re-runs steps 3–7 as if it were that date, from the raw data
already on disk. It implies `--reuse-raw` (nothing is wiped or re-downloaded), passes
the date to each stage, and stamps every output with it (`trade_plan_2025-11-07.csv`, …),
so the dashboard shows the replayed week like any other run.

- **No lookahead.** Only bars *complete* on the as-of date are used. Weekly bars are
  Monday-dated but hold the whole week, so a weekly bar counts only once its Friday is
  on or before the as-of date: `--as-of 2025-11-07` scans the bar dated `2025-11-03`,
  while a mid-week date such as `2025-11-05` scans the previous week. Daily bars count
  on their own date. Benchmarks (QQQ/SPY) are cut the same way.
- **Strict trade-plan lookup.** The helper does not fall back to the newest plan on an
  as-of run; it needs the plan for exactly that date.
- **ML ranking is skipped** (a model trained later would leak the future); ranking is by Score.
- **Limits.** It uses *today's* `active_tickers.csv` (not the historical universe) and
  *today's* split/dividend-adjusted prices, so entry/stop/target levels are in adjusted
  prices, not the quotes seen at the time. Names with under 160 weekly bars as of that
  date are not scored (the usual history floor).
- The individual stages accept the same flag, e.g.
  `python src/finance_vibe/coiled_cobra.py weekly --as-of 2025-11-07`.
- Verified on the real data: deleting every bar after the as-of date and re-running
  gives byte-identical scanner, breakout, and trade-plan files (vibe report matches to
  float noise, ~1e-13). Tests: `tests/test_as_of.py`.

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
python src/finance_vibe/breakout_scanner.py weekly
python src/finance_vibe/trade_planner.py weekly
python src/finance_vibe/trade_plan_helper.py weekly
```

`data_ingestor.py` and `analysis_engine.py` take a data timeframe (`weekly` or
`daily`). The signal stages take `weekly`, `daily`, or `high_beta`. `coiled_cobra.py`
supports all three (`high_beta` reads daily OHLCV and writes to its own
`data/logs/high_beta/` silo).

## Script reference

### `ticker_provider.py`

- Merges `STATIC_TICKERS` (`config.py`), `src/finance_vibe/ticker_manifest.csv`, and Yahoo Finance screeners (`SCREENER_IDS`, `SCREENER_COUNT` = 250 each)
- Writes `data/active_tickers.csv` (capped at `ACTIVE_TICKER_CAP` = **1000** in `config.py`)

### `data_ingestor.py`

- Reads active tickers; downloads via `yfinance` using `TIMEFRAME_PROFILES` in `config.py`
- Drops the last weekly candle only while its week is still trading (complete after Friday 17:00 ET)
- Uses `auto_adjust=True` for split/dividend-adjusted prices
- Logs per-ticker failures to `data/logs/{mode}/ingest_errors_<date>.csv`

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
  gating), and breadth (Checks Met >= 4/6, `MIN_CHECKS_MET`); BBWidth-percentile volatility
  contraction replaces the old MACD-spread squeeze proxy
- Profiles: `weekly`, `daily`, `high_beta` (`high_beta` reads daily OHLCV via
  `config.resolve_pipeline_mode()`, same bar-frequency calibration as
  `daily`, own `data/logs/high_beta/` silo)
- **Specification:** [`coiled_cobra_rubric.md`](../handbook/coiled_cobra_rubric.md)

### `breakout_scanner.py` (research scanner: pre-breakout states)

- Labels trend / volatility / volume / momentum / structure / MTF states and
  classifies `PRE_BREAKOUT`, `BREAKOUT_CONFIRMED`, `FAILED_BREAKOUT`,
  `DEVELOPING`, `WATCH`; records a 100-pt Breakout Readiness score + factor scores
- Writes `breakout_setups_<date>.csv`; **not** consumed by the planner
- **Specification:** [`breakout_scanner.md`](breakout_scanner.md)

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
- Ranks survivors by Expected Value (`R:R T2 × Score`) with a 1.25 coil propensity (`ML_Pred_Return` is used only when `config.ML_RANKING_ENABLED` is on and every row has a prediction)
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
  --csv data/logs/weekly/coiled_cobra_backtest_trades_YYYY-MM-DD.csv
```

Not part of the default pipeline. ML ranking in the helper is off by default
(`config.ML_RANKING_ENABLED = False`) and skipped on `--as-of` runs.

Research harnesses (read-only, never write served artifacts):

```bash
python -m finance_vibe.coiled_cobra_ml_walkforward [--csv PATH]       # ML rank vs Score, expanding folds
python -m finance_vibe.coiled_cobra_ml_experiment --csv PATH           # pre-registered, lockbox-gated
python -m finance_vibe.coiled_cobra_leader_experiment                  # Leader-Expansion vs Coiled Cobra
python -m finance_vibe.breakout_experiment                             # Breakout scanner states/score vs random
```

Full specification: **[`coiled_cobra_ml.md`](coiled_cobra_ml.md)**; leader experiment
protocol: **[`backtest_and_backfill.md`](backtest_and_backfill.md)**.

## UI dashboard

```bash
python src/finance_vibe/app.py
```

Docker (`docker compose up -d`) runs the dashboard by default on port 5000, with
`data/` mounted from the host and `docs/` mounted read-only.

| Route | Contents |
| ----- | -------- |
| `/` , `/view/<mode>/<date>` | Trade plans (`trade_plan_clean_*` / `trade_plan_*`) for `weekly` and `daily` |
| `/breakout`, `/breakout/<mode>/<date>` | Breakout scan: KPIs, status mix, candidates, live price vs Close (`weekly`, `daily`, `high_beta`) |
| `/docs/` | Rendered handbook / architecture / labs markdown (`docs_routes.py`) |

The trade-plan view does not list the `high_beta` silo yet. The breakout view does.

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
| Empty plan on an `--as-of` run | The helper needs the plan for exactly that date; run the full replay, not the helper alone |
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
3. Breakout states/score: edit `classify_status()` / `score_readiness()` in `breakout_scanner.py`; update [`breakout_scanner.md`](breakout_scanner.md)
4. Signal levels: edit `calculate_stock_levels()` in `trade_planner.py`

Validate gate or threshold changes with a staged walk-forward backtest before
shipping them, and bump `config.RUBRIC_VERSION` when a change alters `Score` or
which setups qualify.

## Output files

| File | Layer |
| ---- | ----- |
| `ingest_errors_<date>.csv` | Ingestion failures (pipeline step 2) |
| `vibe_report_<date>.csv` | Macro (pipeline step 3) |
| `coiled_cobra_setups_<date>.csv` | Signal (weekly/daily/high_beta) |
| `breakout_setups_<date>.csv` | Breakout readiness research scan |
| `trade_plan_<date>.csv` | Signal + stock-level context |
| `trade_plan_clean_<date>.csv` | Signal (guardrailed + ranked) |
| `coiled_cobra_backfill_<date>.csv` | Historical Cobra signal archive (manual run) |
| `coiled_cobra_backtest_trades_<date>.csv` | Coiled Cobra walk-forward trades (ML source) |
| `coiled_cobra_{xgb_model.json,lgb_model.txt,ml_model_metadata.json}` | ML artifacts (manual ML run) |
| `coiled_cobra_ml_feature_importance.png` | ML feature-importance chart (manual ML run) |
| `leader_experiment_*` | Leader experiment bars / runs / analysis (manual run) |
| `breakout_experiment_*` | Breakout experiment bars / analysis (manual run) |

## Notes

- Each pipeline run clears `data/raw/{mode}/` before ingestion unless `--reuse-raw` is passed.
- Macro (SMA) and Coiled Cobra (EMA) indicators are intentionally different.
- `trade_plan_helper.py` prefers today’s dated plan, then the newest `trade_plan_*.csv` in that silo.
