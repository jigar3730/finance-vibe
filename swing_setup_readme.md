# Swing Scanner (Tactical Layer)

Companion to the macro Vibe Score in `analysis_engine.py`. This module flags
**SETUP_LONG** and **SETUP_SHORT** pullbacks into EMA20 when trend and momentum align.

Part of the default pipeline (`run_vibe.py`, step 4).

## Usage

```bash
python src/finance_vibe/swing_scanner.py weekly
python src/finance_vibe/swing_scanner.py daily
```

Or run the full pipeline:

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode daily
```

## Inputs and outputs

| | Path |
| --- | --- |
| Raw OHLCV | `data/raw/{mode}/` (e.g. `SPY_10y_1wk.csv`) |
| Active universe | `data/active_tickers.csv` |
| Output CSV | `data/logs/{mode}/swing_setups_<YYYY-MM-DD>.csv` |

Output columns:

| Symbol | Setup Type | Close | EMA20 | EMA50 | RSI | ATR | Notes |

## Indicators (`add_indicators`)

| Indicator | Settings |
| --- | --- |
| EMA20 / EMA50 | Length 20 / 50 on `Close` |
| MACD histogram | pandas_ta default 12 / 26 / 9 |
| RSI | Length 14 |
| ATR | Length 14 on High / Low / Close |

Rows with NaN indicators are dropped before setup evaluation.

## Long setup (`SETUP_LONG`)

All conditions on the **latest bar** (prior bar used for EMA50 slope):

- `EMA20 > EMA50` and `EMA50` rising vs prior bar
- `EMA20 ≤ Close ≤ EMA20 × 1.02`
- RSI 45–60 (weekly) or 40–60 (daily)
- MACD histogram rose two consecutive bars (`h[-1] > h[-2] > h[-3]`)
- MACD histogram below `2 × std(MACD_Hist, 20)` (not overextended)

## Short setup (`SETUP_SHORT`)

- `EMA20 < EMA50` and `EMA50` falling vs prior bar
- `EMA20 × 0.98 ≤ Close ≤ EMA20`
- RSI 50–65
- MACD histogram fell two consecutive bars
- MACD histogram above `−2 × std(MACD_Hist, 20)` (not overextended)

## Programmatic API

For walk-forward backtests and tests:

- `evaluate_setup(df, mode)` — returns setup dict or `None`
- `detect_setup_at_bar(df, symbol, mode)` — returns a full output row dict or `None`

Requires ≥ 60 bars in the input DataFrame (before indicators).

## Operational notes

- Only symbols in `data/active_tickers.csv` that also have a raw CSV are scanned
- Rejection counts are logged (inactive ticker, insufficient data, no setup)
- Uses **EMA** (tactical); macro engine uses **SMA** — intentional separation
- Does not read `vibe_report_*.csv`; macro filtering is optional (see `pipeline_backtest.py`)

## Related files

- `src/finance_vibe/trade_planner.py` — converts setups to stock levels and options metadata
- `src/finance_vibe/trade_plan_helper.py` — cleans trade plan and adds R:R columns
- `trade_planner_worklog.md` — trade planner design and level formulas
