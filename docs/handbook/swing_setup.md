# Swing Scanner (Quality Swing Layer)

Companion to the macro Vibe Score in `analysis_engine.py`. This module flags
**high-probability SETUP_LONG / SETUP_SHORT** pullbacks: bull/bear regime,
tight EMA20 location, RSI band, early MACD-histogram turn, held swing
structure, and **next-bar confirmation**.

Part of the default pipeline (`run_vibe.py`). `analysis_engine.py` is **not**
an orchestrator step; the scanner calls `score_last_row` itself when a profile
sets `vibe_min`.

## Usage

```bash
python src/finance_vibe/swing_scanner.py weekly
python src/finance_vibe/swing_scanner.py daily
python src/finance_vibe/swing_scanner.py high_beta
```

Or run the full pipeline:

```bash
python src/finance_vibe/run_vibe.py
python src/finance_vibe/run_vibe.py --mode daily
python src/finance_vibe/run_vibe.py --mode high_beta
python src/finance_vibe/run_vibe.py --mode daily --reuse-raw
```

`high_beta` reads **daily** OHLCV (`config.resolve_pipeline_mode`) and writes
to `data/logs/high_beta/`. Coiled Cobra is skipped in that mode.

## Inputs and outputs

| | Path |
| --- | --- |
| Raw OHLCV | `data/raw/{weekly\|daily}/` (e.g. `SPY_10y_1wk.csv`, `QQQ_5y_1d.csv`) |
| Active universe | `data/active_tickers.csv` |
| Output CSV | `data/logs/{weekly\|daily\|high_beta}/swing_setups_<YYYY-MM-DD>.csv` |

Output columns follow the shared setup schema (`config.SETUP_ROW_COLUMNS`), including
`Swing Low` / `Swing High` for structural stops in `trade_planner.py`.

## Indicators (`add_indicators`)

| Indicator | Settings |
| --- | --- |
| EMA20 / EMA50 / EMA100 | Length 20 / 50 / 100 on `Close` |
| MACD histogram | pandas_ta default **12 / 26 / 9** |
| RSI | Length 14 |
| ATR | Length 14 on High / Low / Close |
| Swing Low / High | Rolling min/max of Low/High over `structure_bars` (all profiles: **10**) |

MACD readiness also requires a 20-bar histogram std-dev cap:
long hist `< 2 × std`; short hist `> −2 × std`.

## Quality long (`SETUP_LONG`)

Evaluated on the **setup bar** (one bar before the latest); latest bar must **confirm**.

Setup bar:

- `EMA20 > EMA50` and `EMA50` rising
- `Close > EMA100` (bull regime)
- `high_beta` also requires `EMA20 > EMA50 > EMA100` (`require_ema_stack`)
- Close within EMA20 … EMA20 + proximity (see table)
- RSI inside the profile long band
- MACD histogram rising two bars, still **≤ 0**, and not overextended vs 20-bar std
- Pullback held above prior swing low (structure)

Confirmation bar:

- Close ≥ EMA20 and Close ≥ setup-bar low − `confirm_slack_atr × ATR`

## Quality short (`SETUP_SHORT`)

Mirror (weekly only in practice — daily / high_beta disable shorts):

- `EMA20 < EMA50`, falling EMA50, `Close < EMA100`
- Close within EMA20 − proximity … EMA20
- RSI inside the profile short band
- MACD hist falling two bars, still **≥ 0**, not overextended
- Held below prior swing high
- Confirm: Close ≤ EMA20 and Close ≤ setup-bar high + slack

`high_beta` sets `long_only=True` (no short evaluation).
Daily and high_beta set `vibe_min=5` and `short_max_vibe=None`, so
`_soft_vibe_gate` **rejects all shorts** even if the weekly-style short
rules would otherwise match.

## Trade geometry (`config.get_swing_params` / `compute_swing_levels`)

| Level | Weekly | Daily | High-beta |
| ----- | ------ | ----- | --------- |
| Entry | `max(EMA20, Close − 0.25×ATR)` (long) | same | same |
| Stop | dual-constraint: structure vs **1.5×ATR** floor | same **1.5×ATR** floor | same **1.5×ATR** floor, then reject if risk ∉ **[0.5, 1.5] ATR** |
| T1 / T2 | **1.25 / 2.25 ATR** | **0.85 / 1.6 ATR** | **2R / 3R** (`t1_r=2.0`, `t2_r=3.0`) |
| EMA proximity | 1.5% | 2% | **0.5×ATR** |
| RSI long | 45–55 | 40–55 | **35–58** |
| Soft Vibe gate | none | **≥ 5 (long); shorts disabled** | **≥ 5 (long only)** |
| Confirm slack | 0 | 0 | **0.35×ATR** |
| Structure tolerance | 0.2% | 0.2% | **0.25×ATR** |
| Direction | long/short | **long only** (vibe) | **long-only** |
| Market context | none | none | **QQQ regime (close > EMA50 & EMA100, EMA50 rising) + RS (ratio > 20d MA, +63d rel-return)** |
| Cooldown / entry valid / max hold | 4 / 4 / 12 | **8 / 6 / 20** | **10 / 6 / 20** |
| `stop_atr_cap` | 1.5 | 1.5 | 1.5 |
| `max_risk_atr` / `min_risk_atr` | unused | unused | **1.5 / 0.5** |
| Price risk cap | `MAX_RISK_PCT_OF_CLOSE` = **5%** of Close | same | same |

All profiles use a dual-constraint stop: local structure (swing low/high ±
`0.25×ATR`, else EMA50) versus `entry ± stop_atr_cap × ATR`, picking the
tighter bound, then a 5%-of-close ceiling. `high_beta` additionally
**rejects** setups whose resulting risk is outside `[min_risk_atr, max_risk_atr]`.

Coiled Cobra rows use a **separate** Fib path in `trade_planner.calculate_stock_levels`
(not this table). See [`trade_plan_calculations.md`](trade_plan_calculations.md).

The offline swing backtest **defaults to full exit** at `--target-r` (CLI
default 1.5R) with a 2.0 ATR trailing stop. Pass `--use-partials` for the
legacy 50%-at-T1 / runner-to-T2 model.

```bash
python src/finance_vibe/pipeline_backtest.py weekly --tickers SPY,QQQ
python src/finance_vibe/pipeline_backtest.py daily --tickers QQQ,SPY
python src/finance_vibe/pipeline_backtest.py high_beta --tickers PLTR,TSLA,HOOD
```

## Programmatic API

- `evaluate_setup(df, mode)` — setup on last bar of `df` or `None`
- `detect_setup_at_bar(df, symbol, mode, benchmark_df=None)` — full row requiring setup on `iloc[-2]` + confirm on `iloc[-1]`

Requires ≥ 60 bars in the input DataFrame (before indicators).

## Operational notes

- Only symbols in `data/active_tickers.csv` that also have a raw CSV are scanned
- Raw CSVs validated via `config.validate_and_clean_ohlcv`
- Rejection counts logged (inactive, missing_columns, insufficient data, IGNORE)
- Uses **EMA** (tactical); macro engine uses **SMA** — intentional
- Does not read `vibe_report_*.csv`; optional macro gate lives in the scanner (`vibe_min`) and in `pipeline_backtest.py`

## Related files

- [`trade_planner.py`](../../src/finance_vibe/trade_planner.py) — structural levels + options metadata
- [`trade_plan_helper.py`](../../src/finance_vibe/trade_plan_helper.py) — R:R, guardrails, EV rank
- [`config.py`](../../src/finance_vibe/config.py) — `SWING_PROFILES` / `get_swing_params`
- [`trade_plan_calculations.md`](trade_plan_calculations.md) — entry / stop / target math
- [`backtest_and_backfill.md`](../architecture/backtest_and_backfill.md) — walk-forward validation
