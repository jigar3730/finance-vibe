# Breakout Readiness Scanner

**Module:** `src/finance_vibe/breakout_scanner.py`
**Pipeline step:** 5 of `run_vibe.py` (after `coiled_cobra.py`, before `trade_planner.py`)
**Output:** `data/logs/{mode}/breakout_setups_<YYYY-MM-DD>.csv`
**UI:** `/breakout` and `/breakout/<mode>/<date>` in `app.py` (weekly, daily)

A research scanner that labels each ticker's **state** (trend, volatility,
volume, momentum, structure, multi-timeframe alignment) and classifies it as a
pre-breakout, confirmed, failed, developing, or watch setup. It aims to find
names *before* they break out, not only after.

It runs beside Coiled Cobra and does **not** feed the trade planner. Coiled
Cobra stays the primary signal engine. Breakout output is for inspection
and backtesting, to find out which features have predictive value before any
weights are tuned.

## Design principles

- **States first, score second.** `classify_status()` does not take the score
  as an input. The 100-point Breakout Readiness score and its factor
  sub-scores are kept for later weight tuning.
- **Features are the source of truth.** The CSV keeps every raw feature
  (percentiles, slopes, distances, booleans) next to the states and scores.
- **No invented bars.** Daily raw data is resampled to weekly (`W-FRI`) and
  monthly. Weekly raw data is resampled to monthly only. Intraday (4H/1H) is
  out of scope until the raw dataset includes those bars.
- **Causal percentiles.** Width/ATR/range percentiles use rolling windows
  (daily 252, weekly 52, monthly 36 bars), never the full series.

## Flow

```text
data/raw/{data_mode}/*.csv  (filtered to active_tickers.csv)
        │
        ▼
normalize_ohlcv()          clean / validate, optional cut_to_as_of()
        │
        ▼
FeatureEngine.create_timeframes()
        ├── daily native  → Daily + Weekly + Monthly
        └── weekly native → Weekly + Monthly
        │
        ▼
add_indicators()  per timeframe
        SMA 20/50/200 · RSI 14 · MACD 15/30/9 · BB 20/2 · KC 20/1.5
        ATR 14 · RVOL 20 · OBV · 20-bar resistance/support · 20-bar range
        │
        ▼
FeatureEngine.extract()  → BreakoutFeatures (last bar)
        │
        ▼
ScoringEngine.evaluate()
        ├── classify_states()   Trend / Volatility / Volume / Momentum /
        │                       Structure / Breakout Distance / MTF
        ├── classify_status()   PRE_BREAKOUT · BREAKOUT_CONFIRMED ·
        │                       FAILED_BREAKOUT · DEVELOPING · WATCH
        └── score_readiness()   100-pt score + factor scores − penalties
        │
        ▼
breakout_setups_<date>.csv  (sorted by status, then readiness)
```

Files with fewer than 80 primary bars
(`MIN_PRIMARY_BARS`) are rejected as `insufficient_data`.

## Key event definitions (`add_indicators`)

| Field | Definition |
| ----- | ---------- |
| `Squeeze` | Bollinger Bands fully inside Keltner Channel |
| `Compression` | BB Width pctl ≤ 30 **and** ATR pctl ≤ 40 **and** Range20 pctl ≤ 40 |
| `Breakout Triggered` | Close > 20-bar resistance |
| `Breakout Confirmation` | Triggered **and** RVOL20 ≥ 1.5 **and** Close > SMA20 |
| `Breakout Level` | 20-bar resistance on the first bar of the most recent breakout run (the level actually broken) |
| `Failed Breakout` | Was above resistance in the last 5 bars, is not now, **and** closed back below `Breakout Level`. Resistance rolls up to the breakout bar's own high the next day, so closing under it while holding the broken level is not a failure (fixed 2026-10-08; before that most next-day pullbacks were labelled failed). |
| `Wick Reject` | High pierced resistance but Close stayed below |

## Status classification (`classify_status`)

Evaluated in order. The first match wins.

| Status | Rule (summary) |
| ------ | -------------- |
| `FAILED_BREAKOUT` | `Failed Breakout` is true |
| `BREAKOUT_CONFIRMED` | `Breakout Confirmation` is true |
| `PRE_BREAKOUT` | Not through resistance, trend BULLISH, volatility COMPRESSING, momentum ACCELERATING, under/at resistance within 1.25 ATR, MTF ALIGNED/PARTIAL, volume not climactic |
| `WATCH` | Triggered but unconfirmed, or a near-resistance bullish name |
| `DEVELOPING` | Bullish early coil (compressing or BB pctl ≤ 40) still more than 1.25 ATR from resistance |
| `WATCH` | Fallback |

The console table and dashboard show `PRE_BREAKOUT`, `BREAKOUT_CONFIRMED`,
and `FAILED_BREAKOUT` rows, plus `WATCH` rows with readiness ≥ 55 and
`DEVELOPING` rows with readiness ≥ 50. The CSV keeps every scanned row.

## Breakout Readiness score (`score_readiness`)

| Pillar | Max | Components |
| ------ | --- | ---------- |
| Trend | 25 | Daily 10 / Weekly 8 / Monthly 7 (weekly-native: Weekly 15 / Monthly 10) |
| Compression | 25 | BB width pctl 8, ATR pctl 6, Range20 pctl 5, Squeeze 6 |
| Momentum | 20 | RSI band 6, RSI slope 7, MACD histogram slope 7 |
| Volume | 15 | Dry-up 8, RVOL/OBV 7 (regime-aware: dry-up pre-breakout, expansion once triggered) |
| Structure | 15 | Proximity to resistance in ATR 10, extension above SMA in ATR 5 |

Penalties, subtracted after the pillars and clamped to 0–100:

| Condition | Penalty |
| --------- | ------- |
| Failed breakout | 20 (10 if currently re-triggered) |
| Wick rejection (not failed) | 8 |
| MTF `DIVERGENT`, `BEARISH` or `NEUTRAL` (no bullish timeframe) | 10 |
| Extension > 2.5 ATR | 10 |
| RSI > 75 | 5 |
| Trend `BEARISH` | 10 |

## Output columns

`OUTPUT_COLUMNS` in the module is authoritative. The CSV contains these groups:

| Group | Examples |
| ----- | -------- |
| Identity | `Symbol`, `Mode`, `Source`, `AsOf Date`, `Close` |
| Raw features | `SMA20/50/200`, `RSI`, `ATR`, `Resistance`, `Support`, `BB/KC Width Pctl`, `ATR Pctl`, `Range20 Pctl`, `RVOL20`, `RSI Slope`, `MACD Hist Slope`, `Distance Resistance ATR`, `Extension ATR` |
| Booleans | `Daily/Weekly/Monthly Trend Bull`, `MTF Alignment`, `Compression`, `Breakout Triggered`, `Breakout Confirmation`, `Failed Breakout` |
| States | `Trend`, `Volatility`, `Volume State`, `Momentum`, `Structure`, `Breakout Distance`, `MTF`, `Status` |

`MTF` compares the trend on daily (daily-native only), weekly and monthly frames:
`ALIGNED` (all known bullish), `PARTIAL` (some bullish, none bearish), `DIVERGENT`
(bullish on one, bearish on another), `BEARISH` (none bullish, at least one bearish),
`NEUTRAL` (none bullish or bearish), `INSUFFICIENT` (fewer than two known). Before
2026-10-08 `BEARISH` and `NEUTRAL` names were labelled `DIVERGENT`; they keep the same
10-point penalty, so readiness scores are unchanged by the split.
| Scores | `Breakout Readiness`, five pillar scores, `Penalty`, and 15 factor scores |

## Usage

```bash
python src/finance_vibe/breakout_scanner.py weekly
python src/finance_vibe/breakout_scanner.py daily
python src/finance_vibe/breakout_scanner.py weekly --as-of 2025-11-07
```

Tests: `tests/test_breakout_dashboard.py`, `tests/test_as_of.py`.

## Validation status

**Tested 2026-10-01 and found no edge.** A pre-registered weekly walk-forward
(`breakout_experiment.py`, 259 tickers) found that no status or readiness
cut beats random entries on trail-exit R. `BK_READY70` was significantly *worse* than
random, the readiness score's weekly rank IC is about 0, and `PRE_BREAKOUT` had the weakest
13-week forward return of any status. Full protocol and numbers:
[`backtest_and_backfill.md`](backtest_and_backfill.md#breakout-readiness-experiment-research-only).

Treat the dashboard as descriptive only. Do not use it to rank or select entries.

## Open work

- Decide whether to keep `breakout_scanner.py` in `run_vibe.py` given the null result,
  or keep it only as a descriptive dashboard.
- The test covered weekly bars only. A daily test would need the same protocol
  pre-registered before it is run.
- Intraday timeframes need intraday raw bars.
