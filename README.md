# Finance Vibe

## Project Overview

Finance Vibe is a Python pipeline for **macro regime scoring** and **tactical swing setup** discovery. It builds an active ticker universe, ingests OHLCV data, runs two analysis layers, and generates trade plans with stops, targets, and options guidance.

The orchestrator is `src/finance_vibe/run_vibe.py`.

## Analysis layers

| Layer | Module | Output |
| ----- | ------ | ------ |
| **Macro** | `analysis_engine.py` | `data/logs/{mode}/vibe_report_<date>.csv` |
| **Tactical** | `swing_scanner.py` | `data/logs/{mode}/swing_setups_<date>.csv` |
| **Coiled Cobra (Macro Reversal)** | `coiled_cobra.py` / `coiled_cobra_backtest.py` | `data/logs/{mode}/coiled_cobra_setups_<date>.csv`, `coiled_cobra_backfill_<date>.csv`, `coiled_cobra_backtest_trades_<date>.csv` |
| **Coiled Cobra ML (offline)** | `coiled_cobra_ml_training.py` | Trains XGBoost/LightGBM on backtest trades → MAE/RMSE + `coiled_cobra_ml_feature_importance.png` |

Macro scoring rules: `src/finance_vibe/Scoring_Logic.md`.

## Repository structure

```
finance-vibe/
├── src/finance_vibe/          # Application code
├── data/
│   ├── active_tickers.csv     # Universe from ticker_provider
│   ├── raw/{weekly|daily}/    # Ingested OHLCV CSVs
│   └── logs/{weekly|daily|high_beta}/  # Reports, trade plans, backtests
├── BacktestAndBackfill.md     # Offline validation guide
├── CoiledCobraML.md           # ML baseline (XGBoost / LightGBM)
└── tests/
```

## Pipeline flow

1. Clean `data/raw/{mode}/`
2. `ticker_provider.py` → `data/active_tickers.csv`
3. `data_ingestor.py` → download OHLCV per active ticker
4. `analysis_engine.py` → macro Vibe Score report
5. `swing_scanner.py` → tactical setup scan
6. `trade_planner.py` → entry / stop / target / options fields
7. `trade_plan_helper.py` → cleaned plan with R:R columns

## Running

Full pipeline (weekly default):

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode daily
```

Run Coiled Cobra backfill/backtest directly (container recommended):

```bash
python src/finance_vibe/coiled_cobra.py weekly            # run live scanner (latest bar)
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill   # export historical signal archive
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest    # run walk-forward backtest
```

Individual stages (pass `weekly` or `daily` where noted):

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py weekly
python src/finance_vibe/analysis_engine.py weekly
python src/finance_vibe/swing_scanner.py weekly
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
| `swing_setups_<date>.csv` | Tickers passing tactical setup rules (shared setup schema) |
| `coiled_cobra_setups_<date>.csv` | Macro reversal setups (shared setup schema) |
| `trade_plan_<date>.csv` | Stock levels plus persisted options/LEAPS metadata and pass-through context |
| `trade_plan_clean_<date>.csv` | Cleaned plan with direction-aware R:R columns |
| `ingest_errors_<date>.csv` | Per-ticker ingestion failures (empty/invalid/insufficient data) |

Both scanners emit a single shared setup schema (`config.SETUP_ROW_COLUMNS`); raw
CSVs are validated against the OHLCV contract (`config.REQUIRED_OHLCV`) at ingest
and scan time, so malformed files are rejected instead of silently mis-scored.

All outputs live under `data/logs/{mode}/`.

## Macro Vibe Score (summary)

- Scale: **−10 to +10** on the latest bar
- Uses SMA20/50 trend, MACD/RSI momentum, pullback distance from SMA20, CCI cyclical rules, RSI caps
- Full rubric: `src/finance_vibe/Scoring_Logic.md`

## Tactical swing scanner (summary)

Quality swing profile only (see `swing_setup_readme.md`):

**Long (`SETUP_LONG`):**

- Bull regime: `Close > EMA100`; `EMA20 > EMA50` with rising `EMA50`
- Tight pullback into EMA20 (1.5% weekly / 2% daily)
- RSI ≤ 55; MACD hist rising while still ≤ 0; structure held above swing low
- **Next-bar confirmation** required

**Short (`SETUP_SHORT`):** mirror (bear regime below EMA100, RSI 50–60, hist fade)

## Trade planning (summary)

- **Entry:** pullback toward EMA20 (`max(EMA20, Close − 0.25×ATR)` for longs)
- **Stop / targets (swing):** mode-aware via `config.get_swing_params` — weekly T1/T2 1.25/2.25 ATR (stop cap 1.25); daily T1/T2 0.85/1.6 ATR (stop cap 1.5) plus soft Vibe ≥ 5; **high_beta** (daily data) is **long-only** with a QQQ market-regime + relative-strength gate, ATR EMA proximity, wider RSI, structural (uncapped) stop rejected outside 0.5–2.5×ATR risk, and true 1R/2R targets. Its backtest models gap/slippage fills with a 50%-at-1R / breakeven / 2R-runner scale-out and blended-R reporting.
- **Weekly mode:** LEAPS CALL/PUT, 12–24 month expiry window, delta 0.65–0.80 (long) or −0.80 to −0.65 (short)
- **Daily mode:** Options CALL/PUT, 1–3 month expiry window, same delta bands

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

Browse historic trade plans by date and mode (weekly/daily).

## Pipeline backtest (offline validation)

Walk-forward backtest of swing setups + trade-plan stock levels on historical OHLC data. Modes: `weekly`, `daily`, `high_beta`.

```bash
python src/finance_vibe/pipeline_backtest.py weekly --tickers SPY,QQQ
python src/finance_vibe/pipeline_backtest.py daily --tickers QQQ,SPY
python src/finance_vibe/pipeline_backtest.py high_beta --tickers PLTR,TSLA,HOOD

# Coiled Cobra signal archive + trade simulation
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest

# Coiled Cobra ML baseline (predict Forward_Return_2w)
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_2026-07-17.csv
```

The training run writes model artifacts such as `coiled_cobra_xgb_model.json`, `coiled_cobra_lgb_model.txt`, and `coiled_cobra_ml_model_metadata.json` beside the feature-importance plot. Use them with `src/finance_vibe/ml_ranker.py` to attach `ML_Pred_Return` and `ML_Rank` to new Coiled Cobra setups. Treat those columns as a soft ranking/confirmation signal: combine them with the macro score, structure checks, risk rules, and options/liquidity constraints rather than using them as a standalone entry gate.

Outputs land under `data/logs/{weekly|daily|high_beta}/`. Full CLI, execution model, data backfill steps, and promotion gates: **`BacktestAndBackfill.md`**. ML feature isolation, temporal split, and metrics: **`CoiledCobraML.md`**.

**Limitations (summary):** stock-level only (no options P&L); not part of `run_vibe.py`. The scaled simulator includes gap/slippage and 50%-at-1R scale-out; Coiled Cobra still uses the legacy full-exit simulator.

## Notes

- `run_vibe.py` deletes existing files in `data/raw/{mode}/` before each run.
- `trade_plan_helper.py` expects a same-day `trade_plan_<date>.csv` when run standalone.
- Macro and tactical layers use different moving averages (SMA vs EMA) by design.
 - `trade_planner.py` now normalizes `Source` values for Coiled Cobra (`coiled_cobra`) so the Coiled Cobra branch is applied when backtesting/backfilling.
 - `evaluate_coiled_cobra()` may return `None` for non-qualifying bars; code now treats that return as optional in backtest logic.

## Further reading

- `BacktestAndBackfill.md` — **data backfill, signal backfill, and walk-forward backtests**
- `CoiledCobraML.md` — **Coiled Cobra ML baseline (XGBoost / LightGBM)**
- `OperationManual.md` — operations and troubleshooting
- `src/finance_vibe/Scoring_Logic.md` — macro score specification
- `swing_setup_readme.md` — tactical scanner reference
- `src/finance_vibe/pipeline_backtest.py` — offline walk-forward validation
