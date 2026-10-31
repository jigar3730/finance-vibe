# Breakout Readiness Scanner

**Module:** `src/finance_vibe/scanners/breakout.py`
**Pipeline step:** last stage of `finance_vibe.pipeline` (after Coiled Cobra)
**Output:** `data/logs/{mode}/breakout_setups_<YYYY-MM-DD>.csv`
**UI:** `/breakout/<mode>/<date>` in `web/app.py` (weekly, daily)
**Rubric:** Rubric v1.0 (`config.BREAKOUT_RUBRIC_VERSION = "1.0"`), stamped on
`scan_history()` rows and every research output. Bump it whenever a state, Status
or Readiness rule or threshold changes; `tests/test_rubric_docs_sync.py`
fingerprints the rule code and constants and fails until you do.

A research scanner that labels each ticker's **state** (trend, volatility,
volume, momentum, structure, multi-timeframe alignment) and classifies it as a
pre-breakout, confirmed, failed, developing, or watch setup. It aims to find
names *before* they break out, not only after.

It runs beside Coiled Cobra and is independent of it: neither scanner reads
the other's output. Breakout output is descriptive, for inspection.

> **Source of truth:** the module. This doc was audited line by line against
> the code on 2026-10-10. Every threshold below is a module constant or a
> literal in `add_indicators`, `classify_states_frame`, `classify_status_frame`
> or `score_readiness_frame`. Since 2026-10-26 the rules exist only in those
> vectorized functions; the per-row `classify_states` / `classify_status` /
> `score_readiness` are one-row wrappers over them. Where code and doc
> disagree, the code wins.

## Design principles

- **States first, score second.** `classify_status()` does not take the score
  as an input. The 100-point Breakout Readiness score and its factor
  sub-scores are kept for later weight tuning.
- **Features are the source of truth.** The CSV keeps every raw feature
  (percentiles, slopes, distances, booleans) next to the states and scores.
- **No invented bars.** Daily raw data is resampled to weekly (`W-FRI`) and
  monthly (month end). Weekly raw data is resampled to monthly only. A
  still-forming higher-timeframe bar is kept, labelled with the last native
  date. Intraday (4H/1H) is out of scope until the raw dataset includes
  those bars.
- **Causal percentiles.** Width/ATR/range percentiles are rolling ranks of the
  current value (0-100) over a window of daily 252, weekly 52 or monthly 36
  bars, needing at least 20 bars. They never use the full series.

## Flow

```text
data/raw/{data_mode}/*.csv  (filtered to active_tickers.csv)
        │
        ▼
raw.load_raw()             contract check, clean dates, optional as-of cut
        │
        ▼
normalize_ohlcv()          header aliases, UTC dates, dedupe, negative volume → 0
        │
        ▼
_feature_frame()          one row of features per native bar
        ├── add_indicators() on the native frame
        ├── daily native  → daily flags (native) + weekly (W-FRI) and monthly
        │                   (ME) flags from the bar forming at each date
        └── weekly native → weekly flags (native) + monthly flags (forming)
        │
        ├──────────────────────────────┐
        ▼                              ▼
FeatureEngine.extract()        scan_history()
  last row → BreakoutFeatures    every row (bars 80+)
        │                              │
        ▼                              ▼
ScoringEngine.evaluate()       the same rules on the whole frame
        ├── classify_states()   Trend / Volatility / Volume / Momentum /
        │                       Structure / Breakout Distance / MTF
        ├── classify_status()   PRE_BREAKOUT · BREAKOUT_CONFIRMED ·
        │                       BREAKOUT_TRIGGERED · FAILED_BREAKOUT ·
        │                       DEVELOPING · WATCH
        └── score_readiness()   100-pt score + factor scores − penalties
        │                       (each a one-row call of the *_frame rule)
        ▼
breakout_setups_<date>.csv     DataFrame (Date + output columns)
```

Files with fewer than 80 native bars (`MIN_PRIMARY_BARS`) are rejected as
`insufficient_data`. Other rejection keys: `inactive_ticker`, `read_error`,
`missing_columns`, `invalid_ohlcv`, `execution_error`.

## Indicators (`add_indicators`, per timeframe)

Implementations are in `indicators.py` (pandas-ta 0.4.71b0 semantics).

| Column | Definition |
| ------ | ---------- |
| `SMA20/50/200` | Simple moving averages of Close. Monthly frames skip SMA200. |
| `RSI` | RSI(14), Wilder smoothing |
| `MACD_Hist` | MACD(15, 30, 9) histogram (EMA-based, SMA-seeded) |
| `BB_*`, `BB_Width` | Bollinger(20, 2) on SMA20 with **sample** std (ddof=1); width = (upper − lower) / mid |
| `KC_*`, `KC_Width` | Keltner(20, 1.5): EMA20 basis ± 1.5 × EMA20 of the true range |
| `ATR` | ATR(14), Wilder RMA, SMA-seeded |
| `RVOL20` | Volume / SMA20(Volume), current bar included |
| `Volume Dryup` | SMA5(Volume) / SMA20(Volume) < 0.75 |
| `OBV`, `OBV_SMA20` | On-balance volume and its SMA20; "OBV rising" means OBV > OBV_SMA20 |
| `Range20` | max(High, 20) − min(Low, 20), current bar included |
| `Resistance` / `Support` | max High / min Low of the **prior** 20 bars (current bar excluded) |
| `Distance Resistance ATR` | (Resistance − Close) / ATR. Negative means above resistance. |
| `Extension ATR` | (Close − SMA20) / ATR |
| `RSI Slope` | RSI − RSI 5 bars ago |
| `MACD Hist Slope` | MACD_Hist − MACD_Hist 3 bars ago |
| `BB Width Pctl`, `KC Width Pctl`, `ATR Pctl`, `Range20 Pctl` | causal rolling percentile (see above) |

## Key event definitions

| Field | Definition |
| ----- | ---------- |
| `Squeeze` | BB upper < KC upper **and** BB lower > KC lower |
| `Compression` | BB Width Pctl ≤ 30 **and** ATR Pctl ≤ 40 **and** Range20 Pctl ≤ 40 |
| `Breakout Triggered` | Close > Resistance |
| `Breakout Confirmation` | Triggered **and** RVOL20 ≥ 1.5 **and** Close > SMA20 |
| `Breakout Level` | Resistance on the first bar of the most recent run of triggered bars (the level actually broken), carried forward |
| `Failed Breakout` | Triggered on any of the previous 5 bars, not triggered now, **and** Close < `Breakout Level`. Resistance rolls up to the breakout bar's own high the next day, so closing under it while holding the broken level is not a failure (fixed 2026-10-08; before that, most next-day pullbacks were labelled failed). |
| `Wick Reject` | High > Resistance but Close < Resistance |

**Trend flags** (per timeframe, evaluated at each native bar; on daily data the
weekly and monthly flags use the higher-timeframe bar as it was forming at that
date, e.g. Monday–Wednesday of the current week on a Wednesday):
- **Bull:** all of the following hold.
  - Close > SMA20, SMA20 > SMA50, and Close > SMA50.
  - Close > SMA200, when SMA200 exists.
  - SMA20 ≥ its previous value.
  - The monthly frame may lack SMA50; it then uses only Close > rising SMA20.
- **Bear:** Close < SMA20 < SMA50.

## States (`classify_states`)

The primary timeframe is daily when the data is daily, otherwise weekly.

| State | Rule |
| ----- | ---- |
| `Trend` | `BULLISH` if the primary bull flag is set, else `BEARISH` if the primary bear flag is set, else `NEUTRAL` |
| `Volatility` | `COMPRESSING` if `Compression`; else `EXPANDING` if BB Width Pctl ≥ 70 or ATR Pctl ≥ 70; else `NORMAL` |
| `Volume State` | `DRYING_UP` if `Volume Dryup` or RVOL20 < 0.85; else `EXPANDING` if RVOL20 ≥ 1.20; else `NORMAL` |
| `Momentum` | `ACCELERATING` if RSI Slope > 0 **and** MACD Hist Slope > 0; `FADING` if both < 0; else `NEUTRAL` |
| `Structure` | `BELOW_SUPPORT` if Close < Support (checked first); `UNKNOWN` if no distance; `ABOVE_RESISTANCE` if distance < 0; `AT_RESISTANCE` if distance ≤ 0.25 ATR; else `UNDER_RESISTANCE` |
| `Breakout Distance` | text label, e.g. `0.8 ATR` |
| `MTF` | Uses the bull/bear flags of daily (daily data only), weekly and monthly. `INSUFFICIENT` if fewer than two bull flags are known; `ALIGNED` if every known flag is bull; `DIVERGENT` if some are bull and any is bear; `PARTIAL` if some are bull and none bear; `BEARISH` if none bull and any bear; `NEUTRAL` if none bull or bear. Before 2026-10-08, `BEARISH` and `NEUTRAL` were labelled `DIVERGENT`; the penalty is unchanged. |

## Status classification (`classify_status`)

Evaluated in order; the first match wins. "Not through" means `Breakout Triggered` is false.

| # | Status | Rule |
| - | ------ | ---- |
| 1 | `FAILED_BREAKOUT` | `Failed Breakout` |
| 2 | `BREAKOUT_CONFIRMED` | `Breakout Confirmation` |
| 3 | `PRE_BREAKOUT` | Not through **and** Trend `BULLISH` **and** Volatility `COMPRESSING` **and** Momentum `ACCELERATING` **and** Structure `UNDER_RESISTANCE`/`AT_RESISTANCE` **and** 0 ≤ distance ≤ 1.25 ATR **and** MTF `ALIGNED`/`PARTIAL` **and** volume not climactic (Volume State `DRYING_UP`/`NORMAL`, or `EXPANDING` with RVOL20 < 1.5) |
| 4 | `BREAKOUT_TRIGGERED` | Triggered but not confirmed (closed above Resistance without the RVOL20 / SMA20 confirmation). Before 2026-10-20 these rows were labelled `WATCH`; older CSVs still say `WATCH`, and the dashboard relabels them from `Breakout Triggered`. |
| 5 | `DEVELOPING` | Not through **and** Trend `BULLISH` **and** (Compression **or** BB Width Pctl ≤ 40) **and** distance > 1.25 ATR |
| 6 | `WATCH` | Everything else |

## History (`scan_history`)

`scan_history(bars, symbol, scan_mode, native_tf)` returns every bar's features,
states, Status and score, not just the last bar's. It is a function, not a
pipeline stage or CSV.

- One row per native bar that has at least 80 bars of history (`MIN_PRIMARY_BARS`);
  fewer bars in total raise `insufficient bars`, as the scan does.
- Columns: `Date` (the bar's timestamp, datetime), then the output columns below, then
  `Rubric Version`.
- The row dated t is what `run_scanner --as-of t` writes for that ticker, cell for
  cell. On daily data the weekly/monthly trend flags at t come from the
  higher-timeframe bar forming at t, never the completed week or month.
  Weekly raw bars only exist complete, so weekly data has no forming week; its
  monthly flags still use the forming month.
- Pass bars from `raw.load_raw` (the data contract). Truncating the output at t
  equals calling it on data cut at t (`tests/test_breakout_history.py`).
- The forming-bar SMAs are bit-exact with `indicators.sma` on the cut data: every
  window is laid end to end and convolved once.

## Display and sort

The console table and dashboard show these rows; the CSV keeps every scanned row:
- every `PRE_BREAKOUT`, `BREAKOUT_CONFIRMED`, `BREAKOUT_TRIGGERED` and `FAILED_BREAKOUT` row;
- `DEVELOPING` rows with readiness ≥ 50.

`WATCH` rows (the catch-all: nothing has broken out, failed or started coiling)
are never displayed by default. Before 2026-10-20, `WATCH` rows with readiness
≥ 55 were shown.

Rows are sorted by status (`PRE_BREAKOUT`, `BREAKOUT_CONFIRMED`,
`BREAKOUT_TRIGGERED`, `FAILED_BREAKOUT`, `WATCH`, `DEVELOPING`), then readiness
descending, then Symbol.

The dashboard (`/breakout/<mode>/<date>`) shows these rows by default, with
`FAILED_BREAKOUT` rows in their own collapsed section below the main table.
`?show=all` lists every scanned row; a status chip (`?status=`) lists every row
of that status, whatever its readiness. KPIs and chip counts always use the
full file.

## Breakout Readiness score (`score_readiness`)

Five pillars (100 points), minus penalties, clamped to 0-100 and truncated
to an integer.

### Trend (25)

| Timeframe | Daily data | Weekly data | Points |
| --------- | ---------: | ----------: | ------ |
| Daily | 10 | — | 10 if bull; else 6 if Close > SMA50 (3 if also Close ≤ SMA200); else 0 |
| Weekly | 8 | 15 | full if bull; half (integer) if neither bull nor bear; else 0 |
| Monthly | 7 | 10 | full if bull; half (integer) if neither bull nor bear; else 0 |

### Compression (25)

Percentile bands award full points, then partial points, then `max(1, partial // 2)` up to the 50th percentile, then 0:

| Factor | Full | Partial | ≤ 50th | Bands |
| ------ | ---: | ------: | -----: | ----- |
| BB Width Pctl | 8 | 6 | 3 | ≤ 15 / ≤ 25 / ≤ 50 |
| ATR Pctl | 6 | 4 | 2 | ≤ 20 / ≤ 35 / ≤ 50 |
| Range20 Pctl | 5 | 3 | 1 | ≤ 20 / ≤ 35 / ≤ 50 |
| Squeeze | 6 | 3 | — | 6 if `Squeeze`; 3 if KC Width Pctl ≤ 30; else 0 |

### Momentum (20)

| Factor | Points |
| ------ | ------ |
| RSI level (6) | 50-65 → 6; 45-50 or 65-70 → 4; 40-45 → 2; > 75 → 0; otherwise (< 40 or 70-75) → 1 |
| RSI slope (7) | ≥ 2 → 7; > 0 → 5; else 0 |
| MACD hist slope (7) | slope > 0 and histogram > 0 → 7; slope > 0 → 5; else 0 |

### Volume (15), regime-aware

**Dry-up (8):**
- 8 if `Volume Dryup`;
- otherwise 4 if RVOL20 < 1.0, else 0;
- capped at 3 once triggered.

**RVOL/OBV (7)**, when triggered (expansion is the ideal):
- RVOL20 ≥ 2.0 → 7;
- ≥ 1.5 → 6;
- ≥ 1.0 → 3;
- else 0;
- then +1 if OBV is rising, capped at 7.

**RVOL/OBV (7)**, when not triggered (quiet volume is the ideal):
- OBV rising → 7 if RVOL20 < 1.2, else 5;
- otherwise 4 if 1.0 ≤ RVOL20 < 1.5;
- otherwise 2 if `Volume Dryup`, else 0.

### Structure (15)

| Factor | Points |
| ------ | ------ |
| Proximity (10), distance to resistance in ATR | 0-0.75 → 10; 0.75-1.50 → 7; 1.50-2.50 → 4; −0.35 to 0 (just above) → 5; < −0.35 → 2; > 2.50 → 1 |
| Extension (5), (Close − SMA20) / ATR | 0-1.50 → 5; 1.50-2.00 → 3; < 0 → 2; > 2.00 → 0 |

### Penalties

| Condition | Penalty |
| --------- | ------- |
| Failed breakout | 20 |
| Wick rejection (not failed) | 8 |
| MTF `DIVERGENT`, `BEARISH` or `NEUTRAL` (`INSUFFICIENT` is not penalized) | 10 |
| Extension > 2.5 ATR | 10 |
| RSI > 75 | 5 |
| Trend `BEARISH` | 10 |

## Output columns

`OUTPUT_COLUMNS` in the module is authoritative. The CSV contains these groups:

| Group | Columns |
| ----- | ------- |
| Identity | `Symbol`, `Mode`, `Source`, `AsOf Date`, `Close` |
| Raw features | `SMA20/50/200`, `RSI`, `ATR`, `Resistance`, `Support`, `BB/KC Width Pctl`, `ATR Pctl`, `Range20 Pctl`, `RVOL20`, `RSI Slope`, `MACD Hist Slope`, `Distance Resistance ATR`, `Extension ATR` |
| Booleans | `Daily/Weekly/Monthly Trend Bull`, `MTF Alignment` (MTF is `ALIGNED`), `Compression`, `Breakout Triggered`, `Breakout Confirmation`, `Failed Breakout` |
| States | `Trend`, `Volatility`, `Volume State`, `Momentum`, `Structure`, `Breakout Distance`, `MTF`, `Status` |
| Scores | `Breakout Readiness`, the five pillar scores, `Penalty`, and 14 factor scores (3 trend, 4 compression, 3 momentum, 2 volume, 2 structure) |

## Design choices

### Known quirks (kept for parity)

Both are behaviour of the current rubric, kept so `scan_history` and the daily
scan stay identical. Fix them only in a deliberate, versioned change (re-baseline
golden, regenerate history):

- **Length-n SMA.** When a timeframe has exactly n bars, `indicators.sma` sums its
  single window in reverse order (numpy convolution), so that one SMA can differ
  in the last bit from the same window inside a longer series. That can flip a
  trend-flag tie (e.g. `SMA20 ≥ previous`) on the bar where the SMA first warms
  up. `scan_history` reproduces it deliberately.
- **Rounding before thresholds.** Features are rounded (prices, ATR, RVOL20,
  slopes and distances to 4 dp; percentiles to 2 dp; MACD values to 6 dp) before
  the states, status and score compare them with their thresholds. A value just
  under a cut can round onto it (e.g. RVOL20 1.49996 counts as 1.5).

- **Bollinger width uses sample std (ddof=1)**, as pandas-ta did; textbook
  Bollinger uses population std. Kept on purpose (decided 2026-10-10):
  widths are only compared with their own percentile history, so ddof just
  rescales them (about 2.6% at 20 bars). Switching would move Squeeze and
  Compression labels with no evidence that it helps.
- Cleaned up 2026-10-10, with no output change (golden identical at atol=0):
  - the failed-breakout penalty had an unreachable "10 if re-triggered"
    branch (a failed breakout is never triggered on the same bar);
  - `MACD Hist Accel` was computed but never changed the Momentum label.
  - `bbands` used to be called with a `std=` argument that pandas-ta silently
    ignored (harmless, because 2.0 is its default); the call has passed
    `num_std` explicitly since 2026-10.

## Usage

```bash
python -m finance_vibe.scanners.breakout weekly
python -m finance_vibe.scanners.breakout daily
python -m finance_vibe.scanners.breakout weekly --as-of 2025-11-07
```

Tests: `tests/test_breakout.py`, `tests/test_breakout_history.py`,
`tests/test_dashboard.py`, `tests/test_as_of.py`. `tests/breakout_reference.py` is a
frozen copy of the pre-vectorization scalar rules used as a test oracle; delete it
(and its tests) in the next deliberate rule change rather than editing it.

## Validation status

Tested in-sample under `RESEARCH.md` (§7 daily, §11 weekly):

- **Weekly BREAKOUT_CONFIRMED** (first bar of a run) passed in-sample. It is unproven
  until its one holdout run on 2035-09-07. Tested as: buy next open, hold 4 weeks, no stop.
- **Every other event failed**: daily PRE_BREAKOUT, BREAKOUT_TRIGGERED and
  BREAKOUT_CONFIRMED, and weekly PRE_BREAKOUT and BREAKOUT_TRIGGERED.
- Breakout Readiness was dropped from the research plan and is untested.

The Setups page (`/setups`) follows this split: weekly CONFIRMED cards first, failed signals as
counts only. Nothing is ranked by Readiness.

## Open work

- Any edge claim needs a pre-registered walk-forward test against random entries,
  on weekly and daily bars, before the statuses or score are used to select trades.
- Intraday timeframes need intraday raw bars.
