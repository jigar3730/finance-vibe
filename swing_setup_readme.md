# Swing Scanner (Quality Swing Layer)

Companion to the macro Vibe Score in `analysis_engine.py`. This module flags
**high-probability SETUP_LONG / SETUP_SHORT** pullbacks: bull/bear regime,
tight EMA20 location, RSI band, early MACD-histogram turn, held swing
structure, and **next-bar confirmation**.

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

Output columns follow the shared setup schema (`config.SETUP_ROW_COLUMNS`), including
`Swing Low` / `Swing High` for structural stops in `trade_planner.py`.

## Indicators (`add_indicators`)

| Indicator | Settings |
| --- | --- |
| EMA20 / EMA50 / EMA100 | Length 20 / 50 / 100 on `Close` |
| MACD histogram | pandas_ta default 12 / 26 / 9 |
| RSI | Length 14 |
| ATR | Length 14 on High / Low / Close |
| Swing Low / High | Rolling min/max of Low/High over `SWING_STRUCTURE_BARS` (5) |

## Quality long (`SETUP_LONG`)

Evaluated on the **setup bar** (one bar before the latest); latest bar must **confirm**.

Setup bar:

- `EMA20 > EMA50` and `EMA50` rising
- `Close > EMA100` (bull regime)
- Close within EMA20 … EMA20 × (1 + prox) — prox **1.5%** weekly / **2%** daily
- RSI between floor (45 weekly / 40 daily) and **55**
- MACD histogram rising two bars and still **≤ 0**
- Pullback held above prior swing low (structure)

Confirmation bar:

- Close ≥ EMA20 and Close ≥ setup-bar low

## Quality short (`SETUP_SHORT`)

Mirror:

- `EMA20 < EMA50`, falling EMA50, `Close < EMA100`
- Close within EMA20 × (1 − prox) … EMA20
- RSI **50–60**
- MACD hist falling two bars and still **≥ 0**
- Held below prior swing high
- Confirm: Close ≤ EMA20 and Close ≤ setup-bar high

## Trade geometry (`trade_planner.py`)

Mode-aware via `config.get_swing_params(mode)`:

| Level | Weekly | Daily | High-beta |
| ----- | ------ | ----- | --------- |
| Entry | EMA20 ± 0.25×ATR | same | same |
| Stop | structural, capped 1.25×ATR | structural, capped **1.5×ATR** | **structural, uncapped (rejected outside 0.5–2.5×ATR risk)** |
| T1 / T2 | 1.25 / 2.25 ATR | **0.85 / 1.6 ATR** | **1R / 2R (stop distance)** |
| EMA proximity | 1.5% | 2% | **0.5×ATR** |
| RSI long | 45–55 | 40–55 | **35–58** |
| Soft Vibe gate | none | **≥ 5 (long); shorts disabled** | **≥ 5 (long only)** |
| Confirm slack | 0 | 0 | **0.35×ATR** |
| Structure tolerance | 0.2% | 0.2% | **0.25×ATR** |
| Direction | long/short | long/short | **long-only** |
| Market context | none | none | **QQQ regime (close > EMA50 & EMA100, EMA50 rising) + relative strength (ratio > 20d MA, +63d rel-return)** |
| Cooldown / max hold | 4 / 12 | **8 / 20** | **10 / 20** |

`high_beta` reads **daily** OHLCV (`resolve_pipeline_mode`) but keeps its own
long-only swing profile and an isolated `data/logs/high_beta` output silo.

Weekly/daily stops remain structural (swing low/high + EMA50) with an ATR
buffer, then risk-capped. `high_beta` instead leaves the stop at the pullback
structure (swing low − buffer) and **rejects** any setup whose entry-to-stop
risk falls outside the configurable 0.5–2.5×ATR band, then measures T1/T2 as
true 1R/2R multiples of that risk.

The backtest simulates realistic execution: gap/slippage-aware fills, 50% off
at 1R, the remainder trailed to breakeven, and the runner to 2R (blended R:
−1R before T1, +0.5R at breakeven, +1.5R at 2R). Reporting centers on filled
trades, blended expectancy, profit factor, and MAE/MFE.

```bash
python -m finance_vibe.pipeline_backtest high_beta --tickers PLTR,TSLA,HOOD
```

## Programmatic API

- `evaluate_setup(df, mode)` — setup on last bar of `df` or `None`
- `detect_setup_at_bar(df, symbol, mode)` — full row requiring setup on `iloc[-2]` + confirm on `iloc[-1]`

Requires ≥ 60 bars in the input DataFrame (before indicators).

## Operational notes

- Only symbols in `data/active_tickers.csv` that also have a raw CSV are scanned
- Raw CSVs validated via `config.validate_and_clean_ohlcv`
- Rejection counts logged (inactive, missing_columns, insufficient data, IGNORE)
- Uses **EMA** (tactical); macro engine uses **SMA** — intentional
- Does not read `vibe_report_*.csv`; optional macro gate lives in `pipeline_backtest.py`

## Related files

- `src/finance_vibe/trade_planner.py` — structural levels + options metadata
- `src/finance_vibe/trade_plan_helper.py` — R:R cleanup
- `src/finance_vibe/config.py` — `SWING_*` tunables
