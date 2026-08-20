# Finance Vibe Operation Manual

## Status

Stable

## Purpose

This manual describes how to operate the **Coiled Cobra** pipeline: ingest OHLCV, score compressed leaders vs QQQ, draft expansion trade plans, and optionally email a briefing.

The live spec is **`Coiled Cobra Rubric .MD`**. `analysis_engine.py` and `swing_scanner.py` are offline-only.

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
| `data/active_tickers.csv` | Ticker universe |
| `data/raw/daily/` | Daily OHLCV CSVs (`*_5y_1d.csv`) — **primary** |
| `data/raw/weekly/` | Weekly OHLCV CSVs (`*_10y_1wk.csv`) |
| `data/logs/daily/` | Daily reports and trade plans — **primary** |
| `data/logs/weekly/` | Weekly reports and trade plans |

## Standard operating procedure

### Full pipeline

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode weekly
```

Execution order:

| Step | Script | Output |
| ---- | ------ | ------ |
| 0 | (orchestrator) | Clears `data/raw/{mode}/` unless `--keep-raw` |
| 1 | `ticker_provider.py` | `data/active_tickers.csv` |
| 2 | `data_ingestor.py` | Raw CSVs in `data/raw/{mode}/` |
| 3 | `coiled_cobra.py` | `coiled_cobra_setups_<date>.csv` |
| 4 | `trade_planner.py` | `trade_plan_<date>.csv` (Close / Coil_Low / 2R–3R) |
| 5 | `trade_plan_helper.py` | `trade_plan_clean_<date>.csv` |
| 6 | `ai_notifier.py` | Email briefing (if credentials set) |

```bash
python src/finance_vibe/run_vibe.py --keep-raw
```

### Manual stages

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py
python src/finance_vibe/coiled_cobra.py
python src/finance_vibe/trade_planner.py
python src/finance_vibe/trade_plan_helper.py
```

Pass `weekly` after each script for the slower confirmation horizon.

## Script reference

### `ticker_provider.py`

- Merges `STATIC_TICKERS` (`config.py`), `ticker_manifest.csv`, and Yahoo Finance screeners
- Writes `data/active_tickers.csv` (capped at `ACTIVE_TICKER_CAP` = **1000** in `config.py`)

### `data_ingestor.py`

- Reads active tickers; downloads via `yfinance` using `TIMEFRAME_PROFILES` in `config.py`
- Drops incomplete weekly candles (last bar if not Friday)
- Uses `auto_adjust=True` for split/dividend-adjusted prices

### `analysis_engine.py` / `swing_scanner.py` (offline)

Not invoked by `run_vibe.py`. Macro Vibe Score and quality-swing setups remain available for `pipeline_backtest.py` studies. Specs: `src/finance_vibe/Scoring_Logic.md`, `swing_setup_readme.md`.

### `coiled_cobra.py` (live)

- Spec: **`Coiled Cobra Rubric .MD`** v2.1 (coil → expansion)
- Scores the latest bar of each active ticker vs QQQ
- Writes `coiled_cobra_setups_<date>.csv` with pillars plus `Coil_High` / `Coil_Low` / `Coil_Width_ATR`
- Soft-ranks with `ml_ranker.py` when a 10-feature model is present

### `trade_planner.py`

- Auto-detects latest `coiled_cobra_setups_*.csv` in `data/logs/{mode}/`
- **Entry** = Close; **stop** protects `Coil_Low` (else Swing Low) with 1.5×ATR and 5% risk cap; **T1/T2** = 2R / 3R
- Weekly: LEAPS metadata (12–24 mo); daily: options metadata (1–3 mo)
- Explicit path still accepted (offline swing CSVs keep swing geometry)

### `trade_plan_helper.py`

- Loads today’s `trade_plan_<date>.csv` (falls back to newest dated file)
- Drops rows whose risk exceeds 5% of close
- Ranks by `ML_Pred_Return` (clipped ≥ 0), else `Score`; +25% boost only when `Coil_Width_ATR` ≤ 4 or risk ≤ 3%
- Writes `trade_plan_clean_<date>.csv`

### `src/finance_vibe/pipeline_backtest.py` (offline)

Walk-forward validation of swing setup → trade-plan stock levels on historical OHLC CSVs.

Modes: `weekly`, `daily`, `high_beta` (daily data + long-only profile; logs under `data/logs/high_beta/`).

```bash
python src/finance_vibe/pipeline_backtest.py weekly --tickers SPY,QQQ
python src/finance_vibe/pipeline_backtest.py daily --tickers QQQ,SPY
python src/finance_vibe/pipeline_backtest.py high_beta --tickers PLTR,TSLA,HOOD
```

Not part of the default pipeline. Stock simulation only. Full guide (data backfill, CLI, scale-out execution, promotion gates): **`BacktestAndBackfill.md`**.

### `src/finance_vibe/coiled_cobra_backtest.py` (Coiled Cobra historical)

Walk-forward validation and historical backfill for the Coiled Cobra coil scorecard.

- `--backfill` exports a historical Coiled Cobra signal archive to `data/logs/{mode}/coiled_cobra_backfill_<date>.csv`.
- `--backtest` records forward expansion vs QQQ (`Forward_Return_*`, `Rel_Forward_*`) and writes `data/logs/{mode}/coiled_cobra_backtest_trades_<date>.csv`.

Usage:

```bash
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
```

Notes:
- Same `evaluate_coiled_cobra()` engine as the live scanner, every eligible historical bar.
- Filter `Is_New_Coil` to score episodes, not overlapping continuation bars.
- `trade_planner.py` uses Close / Coil_Low expansion geometry for `Source == coiled_cobra`.
- Details: **`BacktestAndBackfill.md`**.

### `src/finance_vibe/coiled_cobra_ml_training.py` (offline ML)

Trains XGBoost + LightGBM to predict **`Rel_Forward_2w`** (fallback `Forward_Return_2w`) on **new coils only**.

- Input: `data/logs/daily/coiled_cobra_backtest_trades_<date>.csv` (weekly silo if you trained on `--mode weekly`)
- Features: 10 columns — 7 rubric pillars + `Pct_From_EMA20/50` + `ATR_Pct` (`Score` / `Grade` excluded)
- Split: rolling 6-month val/test from max `Signal Date` (no random K-fold)
- Retrain after this feature set; old 6-feature Score+Fib models will not score new frames

```bash
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/daily/coiled_cobra_backtest_trades_2026-07-17.csv
```

Not part of the default pipeline. Full specification: **`CoiledCobraML.md`**.

## UI dashboard

```bash
python src/finance_vibe/app.py
```

Open `http://127.0.0.1:5000` to browse trade plans by mode and date.

## Data maintenance

Force re-download for one mode:

```bash
rm data/raw/daily/*.csv
python src/finance_vibe/run_vibe.py
```

## Troubleshooting

| Symptom | Action |
| ------- | ------ |
| Missing `active_tickers.csv` | Run `ticker_provider.py` |
| Ingest skips a symbol | No yfinance data for that ticker; check symbol validity |
| Empty `coiled_cobra_setups_*.csv` | No ticker passed hard gates (compression / structure / RS vs QQQ) |
| `trade_plan_helper` file not found | Run full pipeline first; helper expects a dated `trade_plan_*.csv` |
| ML ranks all null | Retrain 10-feature model (`CoiledCobraML.md`); old Score+Fib artifacts will not score |
| Macro report missing tickers | Check ingest logs; file needs ≥ 60 rows |
| ML script cannot find trades CSV | Run `coiled_cobra_backtest.py weekly --backtest`; pass `--csv`; see **`CoiledCobraML.md`** |

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

1. Cobra scorecard: edit `evaluate_coiled_cobra()` in `coiled_cobra.py`; update **`Coiled Cobra Rubric .MD`**
2. Expansion levels: edit `calculate_stock_levels()` in `trade_planner.py`
3. Offline swing: `evaluate_setup()` in `swing_scanner.py` / `swing_setup_readme.md`

## Output files

| File | Layer |
| ---- | ----- |
| `coiled_cobra_setups_<date>.csv` | Live coil scan |
| `trade_plan_<date>.csv` | Expansion plan |
| `trade_plan_clean_<date>.csv` | Ranked plan |
| `coiled_cobra_backfill_<date>.csv` | Historical coil archive |
| `coiled_cobra_backtest_trades_<date>.csv` | Expansion study (ML source) |
| `coiled_cobra_ml_feature_importance.png` | ML feature-importance chart |

## Notes

- Each pipeline run clears `data/raw/{mode}/` unless `--keep-raw`.
- Daily is the primary horizon for the live scan (5y `1d`); weekly is slower confirmation.
