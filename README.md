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

Macro scoring rules: `src/finance_vibe/Scoring_Logic.md`.

## Repository structure

```
finance-vibe/
├── src/finance_vibe/          # Application code
├── data/
│   ├── active_tickers.csv     # Universe from ticker_provider
│   ├── raw/{weekly|daily}/    # Ingested OHLCV CSVs
│   └── logs/{weekly|daily}/   # Reports and trade plans
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
| `daily` | 2y | 1d | `data/raw/daily/` |

Filenames: `<TICKER>_<period>_<interval>.csv` (e.g. `AAPL_10y_1wk.csv`).

## Output files

| File | Description |
| ---- | ----------- |
| `vibe_report_<date>.csv` | Macro scores for all scanned tickers |
| `swing_setups_<date>.csv` | Tickers passing tactical setup rules |
| `trade_plan_<date>.csv` | Stock levels and options/LEAPS metadata |
| `trade_plan_clean_<date>.csv` | Cleaned plan with R:R columns |

All outputs live under `data/logs/{mode}/`.

## Macro Vibe Score (summary)

- Scale: **−10 to +10** on the latest bar
- Uses SMA20/50 trend, MACD/RSI momentum, pullback distance from SMA20, CCI cyclical rules, RSI caps
- Full rubric: `src/finance_vibe/Scoring_Logic.md`

## Tactical swing scanner (summary)

**Long (`SETUP_LONG`):**

- `EMA20 > EMA50`, rising `EMA50`
- Price between `EMA20` and `EMA20 × 1.02`
- RSI 45–60 (weekly) or 40–60 (daily)
- MACD histogram rising two bars, not beyond 2× its 20-bar std dev

**Short (`SETUP_SHORT`):**

- `EMA20 < EMA50`, falling `EMA50`
- Price between `EMA20 × 0.98` and `EMA20`
- RSI 50–65
- MACD histogram falling two bars, not beyond −2× its 20-bar std dev

## Trade planning (summary)

- **Entry / stop:** ATR-adjusted levels from EMA20/50 and setup direction
- **Targets:** 1× and 2× ATR from entry
- **Weekly mode:** LEAPS CALL/PUT, 12–24 month expiry window, delta 0.65–0.80 (long) or −0.80 to −0.65 (short)
- **Daily mode:** Options CALL/PUT, 1–3 month expiry window, same delta bands

## Requirements

- Python 3.10+
- See `requirements.txt` (`pandas`, `numpy`, `pandas_ta`, `yfinance`, `yahooquery`, `Flask`, …)

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

Walk-forward backtest of swing setups + trade-plan stock levels on historical OHLC data, **with a macro Vibe Score gate** (not applied in the live pipeline today):

- Long setups require Vibe Score ≥ 7
- Short setups require Vibe Score ≤ −2

```bash
python src/finance_vibe/pipeline_backtest.py weekly
python src/finance_vibe/pipeline_backtest.py weekly --tickers SPY,QQQ
python src/finance_vibe/pipeline_backtest.py weekly --long-min-score 7 --short-max-score -2

# Coiled Cobra backtest (separate module)
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
```

Output: `data/logs/{mode}/backtest_trades_<date>.csv` plus a summary printed to stdout.

**Limitations:** stock-level simulation only (no options); uses current `active_tickers.csv` universe; no transaction costs or slippage. Not part of the default `run_vibe.py` workflow.

## Notes

- `run_vibe.py` deletes existing files in `data/raw/{mode}/` before each run.
- `trade_plan_helper.py` expects a same-day `trade_plan_<date>.csv` when run standalone.
- Macro and tactical layers use different moving averages (SMA vs EMA) by design.
 - `trade_planner.py` now normalizes `Source` values for Coiled Cobra (`coiled_cobra`) so the Coiled Cobra branch is applied when backtesting/backfilling.
 - `evaluate_coiled_cobra()` may return `None` for non-qualifying bars; code now treats that return as optional in backtest logic.

## Further reading

- `OperationManual.md` — operations and troubleshooting
- `src/finance_vibe/Scoring_Logic.md` — macro score specification
- `swing_setup_readme.md` — tactical scanner reference
- `src/finance_vibe/pipeline_backtest.py` — offline walk-forward validation
