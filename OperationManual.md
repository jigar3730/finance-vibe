# Finance Vibe Operation Manual

## Status

Stable

## Purpose

This manual describes how to operate the Finance Vibe pipeline, the responsibilities of each script, and how to maintain or extend the project.

## Environment Requirements

Recommended environment:

- Python 3.10 or newer
- pandas
- pandas_ta
- yfinance
- yahooquery

Install required packages with:

```bash
python -m pip install pandas pandas_ta yfinance yahooquery
```

> The repository is compatible with a dev container, but it can also run in any standard Python environment with the required packages installed.

## Directory Structure

- `src/finance_vibe/` — Pipeline source code
- `data/raw/` — downloaded weekly ticker CSV files
- `data/logs/` — generated scanner and trade plan outputs
- `data/active_tickers.csv` — ticker universe produced by `ticker_provider.py`

## Standard Operating Procedure

### Full pipeline execution

From the repository root, run:

```bash
python src/finance_vibe/run_vibe.py
```

This sequence is executed:

1. Deletes all files in `data/raw/`.
2. Runs `src/finance_vibe/ticker_provider.py` to regenerate `data/active_tickers.csv`.
3. Runs `src/finance_vibe/data_ingestor.py` to download weekly price data.
4. Runs `src/finance_vibe/swing_scanner.py` to detect setups and save `data/logs/swing_setups_<YYYY-MM-DD>.csv`.
5. Runs `src/finance_vibe/trade_planner.py` to build `data/logs/trade_plan_<YYYY-MM-DD>.csv`.
6. Runs `src/finance_vibe/trade_plan_helper.py` to validate and save `data/logs/trade_plan_clean_<YYYY-MM-DD>.csv`.

### Manual script execution

Run individual modules directly when needed:

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py
python src/finance_vibe/swing_scanner.py
python src/finance_vibe/trade_planner.py
python src/finance_vibe/trade_plan_helper.py
```

Additional analysis scripts are available for manual use but are not part of the default workflow:

```bash
python src/finance_vibe/analysis_engine.py
python src/finance_vibe/analysis_engine_local.py
```

## Script Responsibilities

### `src/finance_vibe/ticker_provider.py`

- Loads static tickers from `src/finance_vibe/config.py`.
- Reads `src/finance_vibe/ticker_manifest.csv` if present.
- Uses Yahoo Finance screeners to discover active tickers.
- Saves the final ticker list to `data/active_tickers.csv`.
- Limits the final list to 150 symbols for manageable ingestion.

### `src/finance_vibe/data_ingestor.py`

- Reads `data/active_tickers.csv`.
- Downloads weekly OHLCV data with `yfinance`.
- Writes each ticker file as `data/raw/<TICKER>_10y_1wk.csv`.
- Standardizes column capitalization and drops incomplete weekly candles.

### `src/finance_vibe/swing_scanner.py`

- Loads active tickers and available raw data files.
- Computes `EMA20`, `EMA50`, `MACD_Hist`, `RSI`, and `ATR`.
- Applies long and short setup rules based on trend, momentum, and proximity to EMA20.
- Saves matching setups to `data/logs/swing_setups_<YYYY-MM-DD>.csv`.

### `src/finance_vibe/trade_planner.py`

- Reads the latest `swing_setups_*.csv` file from `data/logs/`.
- Calculates stock entry, stop, target levels, LEAPS type, and suggested delta.
- Writes the result to `data/logs/trade_plan_<YYYY-MM-DD>.csv`.

### `src/finance_vibe/trade_plan_helper.py`

- Loads the expected trade plan for today using `data/logs/trade_plan_<YYYY-MM-DD>.csv`.
- Cleans numeric columns and parses the delta range.
- Writes a cleaned file to `data/logs/trade_plan_clean_<YYYY-MM-DD>.csv`.

## Data Maintenance

To force a fresh ingestion run, remove raw data files:

```bash
rm data/raw/*.csv
```

## Optional UI Dashboard

A lightweight dashboard is available at `src/finance_vibe/app.py`.

Run the dashboard from the repository root:

```bash
python src/finance_vibe/app.py
```

Open the browser at:

```text
http://127.0.0.1:5000
```

The UI shows the latest pipeline run and provides links to historic run details.

## Troubleshooting

- If `data/active_tickers.csv` is missing, run `ticker_provider.py` first.
- If `data_ingestor.py` shows no data for a ticker, that symbol is skipped.
- If `swing_scanner.py` reports no valid setups, the current filters did not match any symbols.
- If `trade_plan_helper.py` cannot find a file, ensure a matching `trade_plan_<YYYY-MM-DD>.csv` exists for today.

## Extending the Project

### Add permanent tickers

Edit `src/finance_vibe/config.py` and add symbols to `STATIC_TICKERS`.

### Add or adjust indicator rules

1. Update `src/finance_vibe/swing_scanner.py` for new setup logic.
2. Update `src/finance_vibe/trade_planner.py` for any new level or options logic.
3. Re-run the pipeline and verify output in `data/logs/`.

### Change the data window

Edit the period and interval in `src/finance_vibe/config.py`:

- `PERIOD = "10y"`
- `INTERVAL = "1wk"`

## Output Files

- `data/logs/swing_setups_<YYYY-MM-DD>.csv` — scanner results with detected setups
- `data/logs/trade_plan_<YYYY-MM-DD>.csv` — trade plan export for the chosen setups
- `data/logs/trade_plan_clean_<YYYY-MM-DD>.csv` — cleaned trade plan export

## Notes

- `run_vibe.py` clears raw data each run, so existing downloaded files are removed before ingestion.
- `trade_plan_helper.py` currently expects a same-day `trade_plan_` file and will fail if the file date does not match today.
- `analysis_engine.py` and `analysis_engine_local.py` are available for validation and are not required for the default workflow.
