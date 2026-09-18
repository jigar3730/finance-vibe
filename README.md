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
| **Coiled Cobra (coil → expansion, primary signal engine)** | `coiled_cobra.py` / `coiled_cobra_backtest.py` | `coiled_cobra_setups_<date>.csv`, `coiled_cobra_backfill_<date>.csv`, `coiled_cobra_backtest_trades_<date>.csv` |
| **Coiled Cobra ML (offline)** | `coiled_cobra_ml_training.py` / `ml_ranker.py` | XGBoost/LightGBM artifacts + soft `ML_Pred_Return` / `ML_Rank` |

Macro scoring rules: [`docs/handbook/scoring_logic.md`](docs/handbook/scoring_logic.md).

## Documentation

This repo is both a production pipeline and a **quant / ML lab**. Start here:

| Section | Path | Contents |
| ------- | ---- | -------- |
| **Catalog** | [`docs/README.md`](docs/README.md) | Full map of handbook, architecture, and labs |
| **Handbook** | [`docs/handbook/`](docs/handbook/) | Theory, scoring rubrics, and [`QUANT_ML_MANUAL.md`](docs/handbook/QUANT_ML_MANUAL.md) |
| **Architecture** | [`docs/architecture/`](docs/architecture/) | Pipeline ops, backtests, ML baseline |
| **Labs** | [`docs/labs/`](docs/labs/) | Seven hands-on experiments (indicator ablation → SHAP) |

## Repository structure

```
finance-vibe/
├── src/finance_vibe/          # Application code
├── docs/
│   ├── handbook/              # Conceptual guides + QUANT_ML_MANUAL.md
│   ├── architecture/          # System design, ops, ML pipeline
│   └── labs/                  # Hands-on ML / DS experiments
├── data/
│   ├── active_tickers.csv     # Universe from ticker_provider
│   ├── raw/{weekly|daily}/    # Ingested OHLCV CSVs
│   └── logs/{weekly|daily|high_beta}/  # Reports, trade plans, backtests
└── tests/
```

## Pipeline flow (`run_vibe.py`)

1. Clean `data/raw/{data_mode}/` unless `--reuse-raw`
2. `ticker_provider.py` → `data/active_tickers.csv` (skipped with `--reuse-raw`)
3. `data_ingestor.py` → download OHLCV (skipped with `--reuse-raw`)
4. `analysis_engine.py` → macro Vibe Score (`vibe_report_<date>.csv`)
5. `coiled_cobra.py` → coil scorecard (profile: weekly / daily / high_beta)
6. `breakout_scanner.py` → pre-breakout readiness scan
7. `trade_planner.py` → entry / stop / target context (signal, not a trade plan)
8. `trade_plan_helper.py` → guardrails, R:R, EV / ML rank

`high_beta` reads **daily** OHLCV and writes to `data/logs/high_beta/`.

## Running

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode daily
python src/finance_vibe/run_vibe.py --mode high_beta
python src/finance_vibe/run_vibe.py --mode daily --reuse-raw   # keep existing OHLCV; skip wipe + ingest
```

Coiled Cobra (weekly / daily / high_beta):

```bash
python src/finance_vibe/coiled_cobra.py weekly
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
```

Individual stages:

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py weekly
python src/finance_vibe/analysis_engine.py weekly          # also runs automatically as pipeline step 4
python src/finance_vibe/coiled_cobra.py weekly
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
| `coiled_cobra_setups_<date>.csv` | Coil → expansion setups (shared setup schema; weekly/daily/high_beta) |
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
- **Helper:** drop risk > 5% of Close, checklist < 4/6 (Gate D, `MIN_CHECKS_MET/N_SCORED_PILLARS`), or R:R T1 < 2.0; rank survivors by Expected Value (= raw-Score ranking); `ML_Pred_Return` only when `config.ML_RANKING_ENABLED`

## Requirements

- Python 3.10+
- See `requirements.txt` (`pandas`, `numpy`, `pandas_ta`, `yfinance`, `yahooquery`, `Flask`, `xgboost`, `lightgbm`, `scikit-learn`, `matplotlib`, …)

```bash
python -m pip install -r requirements.txt
```

## Optional UI

```bash
python src/finance_vibe/app.py
# http://127.0.0.1:5000
```

Browse historic trade plans by date and mode (weekly/daily; the UI does not
list the `high_beta` log silo). Docs: `http://127.0.0.1:5000/docs/`.

## Pipeline backtest (offline validation)

`coiled_cobra_backtest.py` is the walk-forward backtest tool: it replays the
Coiled Cobra scorecard across historical OHLC data and simulates stock-level
trade outcomes via `src/finance_vibe/trade_simulator.py` (generic OHLC-bar
fill/stop/target simulation, reused from the former `pipeline_backtest.py`).

```bash
# Coiled Cobra signal archive + trade simulation
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest

# Coiled Cobra ML baseline (predict Forward_Return_2w)
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_YYYY-MM-DD.csv
```

The training run writes model artifacts such as `coiled_cobra_xgb_model.json`, `coiled_cobra_lgb_model.txt`, and `coiled_cobra_ml_model_metadata.json` beside the feature-importance plot. Use them with `src/finance_vibe/ml_ranker.py` to attach `ML_Pred_Return` and `ML_Rank` to new Coiled Cobra setups. Treat those columns as a soft ranking/confirmation signal: combine them with the macro score, structure checks, and risk rules rather than using them as a standalone entry gate.

Outputs land under `data/logs/{weekly|daily|high_beta}/`. ML feature isolation, temporal split, and metrics: **[`docs/architecture/coiled_cobra_ml.md`](docs/architecture/coiled_cobra_ml.md)**.

**Limitations (summary):** stock-level only (no options P&L); not part of `run_vibe.py`.

## Notes

- `run_vibe.py` deletes existing files in `data/raw/{mode}/` before each run unless `--reuse-raw` is passed.
- `trade_plan_helper.py` prefers today’s `trade_plan_<date>.csv`, then the newest dated plan in that log silo.
- Macro and Coiled Cobra layers use different moving averages (SMA vs EMA) by design.
 - `trade_planner.py` now normalizes `Source` values for Coiled Cobra (`coiled_cobra`) so the Coiled Cobra branch is applied when backtesting/backfilling.
 - `evaluate_coiled_cobra()` may return `None` for non-qualifying bars; code now treats that return as optional in backtest logic.

## Further reading

- [`docs/README.md`](docs/README.md) — documentation catalog
- [`docs/handbook/QUANT_ML_MANUAL.md`](docs/handbook/QUANT_ML_MANUAL.md) — feature / target / GBDT / validation / SHAP reference
- [`docs/labs/README.md`](docs/labs/README.md) — hands-on experiment index
- [`docs/architecture/coiled_cobra_ml.md`](docs/architecture/coiled_cobra_ml.md) — Coiled Cobra ML baseline (XGBoost / LightGBM)
- [`docs/architecture/operation_manual.md`](docs/architecture/operation_manual.md) — operations and troubleshooting
- [`docs/handbook/scoring_logic.md`](docs/handbook/scoring_logic.md) — macro score specification
- [`docs/handbook/coiled_cobra_rubric.md`](docs/handbook/coiled_cobra_rubric.md) — Coiled Cobra scorecard specification
- `src/finance_vibe/coiled_cobra_backtest.py` — offline walk-forward validation
