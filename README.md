# Finance Vibe

## Project Overview

Finance Vibe’s live product is the **Coiled Cobra** scanner: compressed leaders vs QQQ that are ready to expand (coil → breakout). The scoring spec is **`Coiled Cobra Rubric .MD`** (v2.1). It is not a swing-pullback scanner and not a Fib-dip mean-reversion model.

The orchestrator is `src/finance_vibe/run_vibe.py` (ingest → coil scan → expansion trade plan → helper → notifier). Quality-swing and macro Vibe scoring remain in the repo for **offline** studies only.

## Analysis layers

| Layer | Module | Output |
| ----- | ------ | ------ |
| **Coiled Cobra (live)** | `coiled_cobra.py` | `data/logs/{mode}/coiled_cobra_setups_<date>.csv` |
| **Expansion plan** | `trade_planner.py` / `trade_plan_helper.py` | `trade_plan_<date>.csv`, `trade_plan_clean_<date>.csv` |
| **Cobra history / ML** | `coiled_cobra_backtest.py` / `coiled_cobra_ml_training.py` | backfill, expansion trades, XGB/LGB ranks |
| **Offline swing (not in `run_vibe`)** | `swing_scanner.py` / `pipeline_backtest.py` | `swing_setups_<date>.csv`, swing backtests |

Hard gates: MACD compression ≥ 5, structure ≥ 8, relative strength vs QQQ ≥ 12. Grades: **A ≥ 85**, **B ≥ 70**.

## Repository structure

```
finance-vibe/
├── src/finance_vibe/          # Application code
├── data/
│   ├── active_tickers.csv     # Universe from ticker_provider
│   ├── raw/{daily|weekly}/    # Ingested OHLCV CSVs (daily is primary)
│   └── logs/{daily|weekly|high_beta}/  # Cobra scans, trade plans, backtests
├── Coiled Cobra Rubric .MD    # Live scorecard (source of truth)
├── Learn.md / LearnTA.md / LearnML.md  # Curriculum + primers (/learn on :5000)
├── MLOps.md                   # Docker train / evaluate / deploy
├── BacktestAndBackfill.md     # Offline validation guide
├── CoiledCobraML.md           # ML baseline (XGBoost / LightGBM)
└── tests/
```

## Pipeline flow

1. Clean `data/raw/{mode}/` (skip with `--keep-raw`)
2. `ticker_provider.py` → `data/active_tickers.csv`
3. `data_ingestor.py` → download OHLCV per active ticker
4. `coiled_cobra.py` → v2.1 coil scorecard on the latest bar
5. `trade_planner.py` → Close / Coil_Low / 2R–3R expansion plan
6. `trade_plan_helper.py` → risk cap, ML-then-Score rank, cleaned CSV
7. `ai_notifier.py` → optional Gemini briefing

## Running

Full pipeline (daily default; pass `--mode weekly` for the slower confirmation horizon):

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode weekly
python src/finance_vibe/run_vibe.py --keep-raw   # reuse existing OHLCV
```

Coiled Cobra research:

```bash
python src/finance_vibe/coiled_cobra.py
python src/finance_vibe/coiled_cobra_backtest.py --backfill
python src/finance_vibe/coiled_cobra_backtest.py --backtest
python src/finance_vibe/coiled_cobra_ml_training.py
```

On the `finance_vibe` container, prefix with `docker exec -w /app finance_vibe` and use `/app/data/...` paths. Full train/deploy: **`MLOps.md`**.

Individual stages (omit the mode argument to use daily; pass `weekly` when needed):

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py
python src/finance_vibe/coiled_cobra.py
python src/finance_vibe/trade_planner.py
python src/finance_vibe/trade_plan_helper.py
```

`high_beta` on `run_vibe` is ingest-only (use `pipeline_backtest.py` for the swing study).

## Timeframe profiles (`config.py`)

| Mode | Lookback | Interval | Raw path |
| ---- | -------- | -------- | -------- |
| `daily` (default) | 5y | 1d | `data/raw/daily/` |
| `weekly` | 10y | 1wk | `data/raw/weekly/` |

Filenames: `<TICKER>_<period>_<interval>.csv` (e.g. `AAPL_5y_1d.csv`, `AAPL_10y_1wk.csv`).

## Output files

| File | Description |
| ---- | ----------- |
| `coiled_cobra_setups_<date>.csv` | Passing coils (v2.1 pillars + Coil_High/Low) |
| `trade_plan_<date>.csv` | Expansion levels: Close entry, Coil_Low stop, 2R/3R targets |
| `trade_plan_clean_<date>.csv` | Ranked plan (ML predicted return, else Score) |
| `ingest_errors_<date>.csv` | Per-ticker ingestion failures (empty/invalid/insufficient data) |

The live scanner emits `config.SETUP_ROW_COLUMNS`. Raw CSVs are validated against `config.REQUIRED_OHLCV` at ingest and scan time.

All outputs live under `data/logs/{mode}/`.

## Coiled Cobra scorecard (summary)

See **`Coiled Cobra Rubric .MD`**. Core six pillars sum to 100; Fib bonus 0–5 then clip at 100. Piecewise-linear interpolation, not 5-point cliffs.

## Trade planning (Cobra expansion)

- **Entry:** Close of the passing coil bar
- **Stop:** triple constraint — `Coil_Low − 0.25×ATR` (else Swing Low), `entry − 1.5×ATR`, `entry − 5% of close`; cap so the stop stays below entry
- **Targets:** 2R and 3R
- **Weekly mode:** LEAPS CALL, 12–24 month expiry, delta 0.65–0.80
- **Daily mode:** Options CALL, 1–3 month expiry, same delta band

Quality-swing geometry remains in `config.compute_swing_levels` for offline `pipeline_backtest` only.

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

# Coiled Cobra signal archive + expansion study (daily default)
python src/finance_vibe/coiled_cobra_backtest.py --backfill
python src/finance_vibe/coiled_cobra_backtest.py --backtest
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest   # slower confirmation horizon

# Coiled Cobra ML baseline (predict Rel_Forward_2w on new coils)
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/daily/coiled_cobra_backtest_trades_2026-07-17.csv
```

The training run writes `coiled_cobra_xgb_model.json`, `coiled_cobra_lgb_model.txt`, and `coiled_cobra_ml_model_metadata.json`. `ml_ranker.py` attaches `ML_Pred_Return` / `ML_Rank` as a **soft** rank — never a gate. Retrain after the 10-feature pillar set; old 6-feature (Score + Fib %) models will not score new frames.

Outputs land under `data/logs/{weekly|daily|high_beta}/`. Full CLI: **`BacktestAndBackfill.md`**. Features/splits: **`CoiledCobraML.md`**.

**Limitations (summary):** stock-level only (no options P&L). Cobra `--backtest` is forward expansion vs QQQ, not bounce P&L. Filter `Is_New_Coil` when counting episodes.

## Notes

- `run_vibe.py` deletes `data/raw/{mode}/` unless you pass `--keep-raw`.
- Spec: **`Coiled Cobra Rubric .MD`**. `swing_scanner.py` / `analysis_engine.py` are offline only.

## Further reading

- `Learn.md` — **curriculum** (TA, system, ML). Dashboard: `http://host:5000/learn`
- `LearnTA.md` / `LearnML.md` — **beginner primers**
- `Coiled Cobra Rubric .MD` — **live scorecard (source of truth)**
- `BacktestAndBackfill.md` — **data backfill, signal backfill, and walk-forward backtests**
- `MLOps.md` — **Docker-first train / evaluate / deploy runbook + ML concepts**
- `CoiledCobraML.md` — **Coiled Cobra ML baseline (feature contract, metrics)**
- `OperationManual.md` — operations and troubleshooting
- `swing_setup_readme.md` — offline tactical scanner reference
- `src/finance_vibe/pipeline_backtest.py` — offline quality-swing validation
