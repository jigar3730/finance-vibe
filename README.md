# Finance Vibe

## Project Overview

Finance Vibe is a lightweight Python pipeline for weekly equity analysis and swing setup discovery. It combines an active ticker universe, weekly OHLCV ingestion, technical indicator scanning, and trade plan generation.

This repository separates application logic from generated output data. The core workflow is executed by `src/finance_vibe/run_vibe.py`, which runs the pipeline end-to-end.

## Repository Structure

- `src/finance_vibe/` - Application source code
- `data/raw/` - Raw weekly market data files downloaded by the ingestion step
- `data/logs/` - Generated reports and timestamped CSV outputs
- `notebooks/` - Exploratory analysis notebooks

## Key Pipeline Scripts

- `src/finance_vibe/run_vibe.py` - Orchestrates the full pipeline in sequence
- `src/finance_vibe/ticker_provider.py` - Builds `data/active_tickers.csv` from a manifest, static tickers, and Yahoo Finance screeners
- `src/finance_vibe/data_ingestor.py` - Downloads weekly market data using `yfinance`
- `src/finance_vibe/swing_scanner.py` - Scans raw data for swing trade setups using EMA, RSI, MACD, and ATR rules
- `src/finance_vibe/trade_planner.py` - Converts scanner output into stock levels, targets, and LEAPS guidance
- `src/finance_vibe/trade_plan_helper.py` - Cleans the latest generated trade plan and exports a cleaned CSV

## Pipeline Flow

1. `run_vibe.py` cleans `data/raw/`.
2. It runs `ticker_provider.py` to generate `data/active_tickers.csv`.
3. It runs `data_ingestor.py` to download weekly OHLCV data for each active ticker.
4. It runs `swing_scanner.py` to detect setups and save `data/logs/swing_setups_<YYYY-MM-DD>.csv`.
5. It runs `trade_planner.py` to create `data/logs/trade_plan_<YYYY-MM-DD>.csv`.
6. It runs `trade_plan_helper.py` to validate and save `data/logs/trade_plan_clean_<YYYY-MM-DD>.csv`.

## Running the Project

From the repository root:

```bash
python src/finance_vibe/run_vibe.py
```

This executes the full pipeline and writes output into `data/logs/`.

### Run individual stages

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py
python src/finance_vibe/swing_scanner.py
python src/finance_vibe/trade_planner.py
python src/finance_vibe/trade_plan_helper.py
```

## Data Files

- `data/active_tickers.csv` — Active ticker universe created by `ticker_provider.py`
- `data/raw/<TICKER>_10y_1wk.csv` — Raw weekly OHLCV source files
- `data/logs/swing_setups_<YYYY-MM-DD>.csv` — Scanner output with detected setups
- `data/logs/trade_plan_<YYYY-MM-DD>.csv` — Generated trade plan from scanner output
- `data/logs/trade_plan_clean_<YYYY-MM-DD>.csv` — Cleaned version of the latest trade plan

## Core Logic Summary

### Ticker Discovery

- Static tickers are always included from `src/finance_vibe/config.py`.
- If present, `src/finance_vibe/ticker_manifest.csv` is also loaded.
- `ticker_provider.py` adds active names from Yahoo Finance screeners and saves the final list to `data/active_tickers.csv`.
- The ticker list is limited to 150 items for manageability.

### Data Ingestion

- `data_ingestor.py` reads `data/active_tickers.csv` and downloads weekly data with `yfinance`.
- Raw files are saved using the configured period and interval from `src/finance_vibe/config.py`.
- The ingestion step normalizes columns and drops incomplete weekly candles.

### Swing Scanner Rules

- `swing_scanner.py` calculates `EMA20`, `EMA50`, `MACD_Hist`, `RSI`, and `ATR` for each ticker.
- A long setup is identified when:
  - `EMA20 > EMA50`
  - `EMA50` is rising
  - price is within 2% of EMA20
  - RSI is between 45 and 60
  - MACD histogram momentum is rising but not over-extended
- A short setup is identified when:
  - `EMA20 < EMA50`
  - `EMA50` is falling
  - price is within 2% of EMA20
  - RSI is between 50 and 65
  - MACD histogram momentum is weakening but not over-extended

### Trade Planning

- `trade_planner.py` creates stock entry, stop, and target levels from each setup.
- It assigns LEAPS option type (`CALL` for longs, `PUT` for shorts) and a suggested delta range.
- Expiry windows are calculated for 12–24 months out.

### Trade Plan Cleaning

- `trade_plan_helper.py` loads the latest `trade_plan_<YYYY-MM-DD>.csv` for today.
- It converts numeric fields, attempts to parse the suggested delta range, and exports `trade_plan_clean_<YYYY-MM-DD>.csv`.
- If the expected today file is missing, the helper will fail with a not-found message.

## Requirements

- Python 3.10+
- pandas
- pandas_ta
- yfinance
- yahooquery
- Flask (for the optional dashboard)

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Optional UI Dashboard

A simple web dashboard is available at `src/finance_vibe/app.py`.

Start the dashboard from the repository root:

```bash
python src/finance_vibe/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Use the home page to see the latest run summary and click individual dates for historic outputs.

## Notes and Caveats

- The pipeline is designed for weekly analysis, not intraday trading.
- `run_vibe.py` removes all files from `data/raw/` before ingestion.
- `trade_plan_helper.py` expects a same-day trade plan file when run manually.
- `analysis_engine.py` and `analysis_engine_local.py` exist for validation and comparison but are not executed by the default workflow.

## Recommended Next Steps

- Add permanent symbols in `src/finance_vibe/config.py` under `STATIC_TICKERS`.
- If you want more active tickers, update `ticker_provider.py` or the manifest file.
- To change lookback or cadence, edit `PERIOD` and `INTERVAL` in `src/finance_vibe/config.py`.
