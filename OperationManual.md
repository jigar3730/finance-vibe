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

Compatible with the dev container or any standard Python environment with `PYTHONPATH=./src`.

## Directory layout

| Path | Contents |
| ---- | -------- |
| `src/finance_vibe/` | Pipeline source |
| `data/active_tickers.csv` | Ticker universe |
| `data/raw/weekly/` | Weekly OHLCV CSVs (`*_10y_1wk.csv`) |
| `data/raw/daily/` | Daily OHLCV CSVs (`*_2y_1d.csv`) |
| `data/logs/weekly/` | Weekly reports and trade plans |
| `data/logs/daily/` | Daily reports and trade plans |

## Standard operating procedure

### Full pipeline

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode daily
```

Execution order:

| Step | Script | Output |
| ---- | ------ | ------ |
| 0 | (orchestrator) | Clears `data/raw/{mode}/` |
| 1 | `ticker_provider.py` | `data/active_tickers.csv` |
| 2 | `data_ingestor.py` | Raw CSVs in `data/raw/{mode}/` |
| 3 | `analysis_engine.py` | `vibe_report_<date>.csv` |
| 4 | `swing_scanner.py` | `swing_setups_<date>.csv` |
| 5 | `trade_planner.py` | `trade_plan_<date>.csv` |
| 6 | `trade_plan_helper.py` | `trade_plan_clean_<date>.csv` |

### Manual stages

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py weekly
python src/finance_vibe/analysis_engine.py weekly
python src/finance_vibe/swing_scanner.py weekly
python src/finance_vibe/trade_planner.py weekly
python src/finance_vibe/trade_plan_helper.py weekly
```

Replace `weekly` with `daily` for the daily profile.

## Script reference

### `ticker_provider.py`

- Merges `STATIC_TICKERS` (`config.py`), `ticker_manifest.csv`, and Yahoo Finance screeners
- Writes `data/active_tickers.csv` (capped at 150 symbols)

### `data_ingestor.py`

- Reads active tickers; downloads via `yfinance` using `TIMEFRAME_PROFILES` in `config.py`
- Drops incomplete weekly candles (last bar if not Friday)
- Uses `auto_adjust=True` for split/dividend-adjusted prices

### `analysis_engine.py` (macro layer)

- Scores every raw CSV in `data/raw/{mode}/` on a **−10 to +10** Vibe Score
- Requires ≥ 60 bars per file
- Parallel scan via `ProcessPoolExecutor`
- **Specification:** `src/finance_vibe/Scoring_Logic.md`

### `swing_scanner.py` (tactical layer)

- Filters to symbols in `active_tickers.csv`
- EMA20/50 trend, RSI band, MACD histogram slope + std-dev cap, ATR output
- Only tickers passing setup rules appear in output
- Weekly long RSI floor: 45; daily long RSI floor: 40

### `trade_planner.py`

- Reads latest `swing_setups_*.csv` in `data/logs/{mode}/`
- Computes stock entry, stop, 1R/2R targets from ATR and EMA levels
- Weekly: LEAPS metadata (12–24 mo expiry); daily: options metadata (1–3 mo expiry)

### `trade_plan_helper.py`

- Loads today’s `trade_plan_<date>.csv`
- Parses delta range; adds Risk Per Share and R:R columns
- Writes `trade_plan_clean_<date>.csv`

### `src/finance_vibe/pipeline_backtest.py` (offline)

Walk-forward validation of swing setup → trade-plan stock levels on historical OHLC CSVs.

- Re-runs `detect_setup_at_bar()` and `calculate_stock_levels()` at each historical bar
- Applies macro gate: LONG requires Vibe Score ≥ 7, SHORT requires Vibe Score ≤ −2
- Simulates limit entry, stop, and ATR targets on forward High/Low bars
- Writes `data/logs/{mode}/backtest_trades_<date>.csv`

```bash
python src/finance_vibe/pipeline_backtest.py weekly
python src/finance_vibe/pipeline_backtest.py weekly --tickers SPY,QQQ
```

Not part of the default pipeline. Stock simulation only; no options P&L.

### `src/finance_vibe/coiled_cobra_backtest.py` (Coiled Cobra historical)

Walk-forward validation and historical backfill for the Coiled Cobra macro-reversal scanner.

- `--backfill` exports a historical Coiled Cobra signal archive to `data/logs/{mode}/coiled_cobra_backfill_<date>.csv`.
- `--backtest` runs a walk-forward stock-level backtest and writes `data/logs/{mode}/coiled_cobra_backtest_trades_<date>.csv`.

Usage (container recommended):

```bash
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
```

Notes:
- Uses the same `evaluate_coiled_cobra()` engine as the live scanner but evaluates every eligible historical bar.
- `trade_planner.py` now recognizes `Source` == `coiled_cobra` so Coiled Cobra setups use Fib 78.6% entry logic.

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
| Empty `swing_setups_*.csv` | No tickers matched tactical filters (expected in quiet markets) |
| `trade_plan_helper` file not found | Run full pipeline first; helper expects today’s dated file |
| Macro report missing tickers | Check ingest logs; file needs ≥ 60 rows |

## Extending the project

### Add tickers

- Edit `STATIC_TICKERS` in `config.py`, or
- Add rows to `ticker_manifest.csv`

### Change lookback or cadence

Edit `TIMEFRAME_PROFILES` in `config.py`:

```python
"weekly": {"period": "10y", "interval": "1wk", ...}
"daily":  {"period": "2y",  "interval": "1d",  ...}
```

### Change scoring or setup rules

1. Macro: edit `score_last_row()` in `analysis_engine.py`; update `Scoring_Logic.md`
2. Tactical: edit `evaluate_setup()` in `swing_scanner.py`; update `swing_setup_readme.md`
3. Execution: edit `trade_planner.py` for level/options logic

## Output files

| File | Layer |
| ---- | ----- |
| `vibe_report_<date>.csv` | Macro |
| `swing_setups_<date>.csv` | Tactical |
| `trade_plan_<date>.csv` | Execution |
| `trade_plan_clean_<date>.csv` | Execution (cleaned) |
| `backtest_trades_<date>.csv` | Offline backtest (manual run) |

## Notes

- Each pipeline run clears `data/raw/{mode}/` before ingestion.
- Macro (SMA) and tactical (EMA) indicators are intentionally different.
- `trade_plan_helper.py` fails if the trade plan date does not match today when run alone.
