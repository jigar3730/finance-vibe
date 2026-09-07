# Coiled Cobra Scanner
## Scoring Rubric & Technical Design

Version: 3.1 (High-beta coil + ticker-level Market Gate)

Implementation: [`coiled_cobra.py`](../../src/finance_vibe/coiled_cobra.py). Theory and ML mapping: [`QUANT_ML_MANUAL.md`](QUANT_ML_MANUAL.md). Historical trade archive: [`backtest_and_backfill.md`](../architecture/backtest_and_backfill.md).

---

# Overview

The **Coiled Cobra Scanner** identifies **compressed leaders ready to expand**
(coil → breakout), not swing pullbacks and not deep-discount mean reversion.

It looks for securities that are:

- Sitting in a volume / accumulation shelf
- Compressing on MACD (histogram squeeze near zero, MACD line > 0)
- Holding aligned structure (EMA20 > EMA50; 10 EMA > 20 EMA preferred)
- Showing positive relative strength vs QQQ (full RS pillar at +15% / 63d)
- Coiling in a tight N-bar range (range / ATR ≤ 1.5 for full coil points)
- Printing expansion volume (RVOL ≥ 1.2×) as an *additive* trigger
- Clearing overhead supply, or sitting in open sky (≥ 95% of 52-week / ATH)

Deep markdown under EMA20 was **removed** — it systematically filtered out
high-base coils (NVDA / APP / MU-class) that the scanner is meant to catch
before a monster run. EMA50 extension above 25% is a **soft deduction**,
not a Market Gate fail; RS leaders may extend to 50%.

The scanner produces a maximum score of **100 points**.
Only setups scoring ≥ 70 are returned.

CLI accepts **`weekly`** or **`daily`** only. `run_vibe.py --mode high_beta`
skips this scanner (`skip_modes: ["high_beta"]`).

```bash
python src/finance_vibe/coiled_cobra.py weekly
python src/finance_vibe/coiled_cobra.py daily
```

---

# Calibration constants (`coiled_cobra.py`)

| Constant | Weekly | Daily |
| -------- | -----: | ----: |
| `LOOKBACK` (volume shelf + Fib + overhead) | 52 | 252 |
| `COIL_BARS` | 8 | 30 |
| `STRUCTURE_STOP_BARS` (local swing-low for planner) | 10 | 10 |
| `RS_LOOKBACK` | 13 | 63 |
| `RS_RATIO_MA` | 5 | 20 |
| `BENCHMARK` / `SPY_BENCHMARK` | QQQ / SPY | QQQ / SPY |
| `MIN_PASS_SCORE` | 70 | 70 |
| `GRADE_A_SCORE` | 85 | 85 |
| `MIN_COMPRESSION` | 5 (Checks Met only) | 5 |
| `MIN_STRUCTURE` | 8 (Checks Met only) | 8 |
| `MIN_RS_POINTS` | 12 (Checks Met only) | 12 |
| `MACRO_PENALTY` | 0 (unused; gate no longer haircuts) | 0 |
| `OPEN_SKY_PCT` | 0.95 | 0.95 |
| `EMA50_SOFT_EXT` / `EMA50_MOMENTUM_EXT` | 0.25 / 0.50 | 0.25 / 0.50 |
| `RS_FULL_SCORE` | 0.15 | 0.15 |
| `COIL_FULL_ATR` / `COIL_PARTIAL_ATR` | 1.5 / 2.2 | 1.5 / 2.2 |
| `RVOL_BONUS` | 1.2 | 1.2 |

Minimum bars to evaluate a bar: `max(COIL_BARS + 2, 25)`. Live scan also
requires `max(LOOKBACK // 2, COIL_BARS + 40)` rows of history.

---

# Overall Scoring Matrix (v3.1 — Coil)

| Category | Weight | `Parts` key |
|----------|---------:|-------------|
| Volume Profile Shelf | 20 | `volume_shelf` |
| Volatility Coil (range / ATR ≤ 1.5) | 20 | `coil_width` |
| MACD Squeeze State | 15 | `macd_compression` |
| Relative Strength vs QQQ | 15 | `relative_strength` |
| MA Alignment & ATR Proximity | 15 | `structure` |
| Breakout RVOL Trigger | 10 | `rvol_trigger` |
| Overhead Clearance / Space | 5 | `overhead_clearance` |

Deprecated v2 keys **`macd_cross`** and **`fib_bonus`** are no longer written.
CSV extras: **`RVOL`** (raw volume / SMA20) and **`Market Gate`** (ticker
Close ≥ EMA50 and RS 63d ≥ 0). These are not ML `FEATURE_COLS`.

`fibonacci_score()` remains in the module but is **not summed**. The CSV still
emits `Fib Score = 0.0`.

Maximum Score = **100**

Market Gate is the **only** pre-filter, and it fails only on total trend
break: `Close < EMA50` or `RS 63d < 0`. Pillar floors above are **Checks
Met** counters, not binary drops. Extension (`Pct_From_EMA50`), Fib
distance, and `RVOL < 1.0` never fail the gate.

A failed gate records `Market Gate = False` and `Grade = Rejected - Trend
Fail`. The scorecard is still computed (`include_rejects=True`); the live
scanner / backtest omit the row. RVOL is **never** zeroed and no −15
macro haircut is applied.

---

# Grade Classification

| Score | Grade | Meaning |
|--------:|-------|---------|
| 85-100 **and** market gate pass | A - Coil Ready | High-confluence pre-expansion |
| 70-84 **and** market gate pass | B - Valid Coil | Actionable coil |
| Gate fail (Close < EMA50 or RS 63d < 0) | Rejected - Trend Fail | Filtered after scoring |
| Below 70 with gate pass | Rejected - Below Threshold | Insufficient confluence |

---

# Indicator Details

## 1. Volume Profile Shelf (20 Points)

`evaluate_volume_profile_shelf(df, price, lookback=LOOKBACK)` — 30 bins over
the lookback window, volume-weighted on Close.

| Sub-score | Max | Rule |
| --------- | --: | ---- |
| Topology | 8 | `min(8, int((bin_vol / avg_neighbor_vol) * 2.5))` |
| Auction value vs POC | 8 | ≤ 3 bins from POC (including at POC) = 8; ≤ 6 bins = 4; else 0 |
| Behavior | 4 | Close above bin center = 4, else 1 |

Check counted when `volume_shelf ≥ 10`.

## 2. Volatility Coil (20 Points)

`coil_width_score` — N-bar High−Low range / ATR14 (`COIL_BARS`). The
signal bar is excluded when history allows so a breakout day does not
inflate the coil. Quiet volume (`RVOL < 1.0`) inside the base is valid
compression and is **not** a drop.

The tightest of several short windows is kept (daily: 3/5/8/13; weekly:
3/5/8), signal bar excluded.

| width | Points |
|-------|-------:|
| ≤ 1.5 ATR | 20 |
| 1.5 – 2.2 ATR | linear 20 → 10 |
| 2.2 – 4.0 ATR | residual 10 → 4 (high-beta pause) |
| > 4.0 ATR | 0 |

Check counted when `coil_width ≥ 10`.

## 3. MACD Squeeze State (15 Points)

`macd_compression_score` — pandas_ta MACD **12 / 26 / 9**.

```
hist = MACD_Hist   # or MACD − Signal if hist missing
spread = abs(hist) / ATR
```

| spread | Points |
|--------|-------:|
| ≤ 0.05 | 15 |
| ≤ 0.10 | 11 |
| ≤ 0.18 | 7 |
| ≤ 0.30 | 4 |
| else | 0 |

Deduct 5 if the MACD **line** is ≤ 0 (floor at 0). Crossover is not scored
(`macd_cross` retired). Check counted when `macd_compression ≥ 7`.

## 4. Relative Strength vs QQQ (15 Points)

`rs_score` → `relative_strength()` (causal, `Date <= as_of`):

Full pass (`ok`) requires stock/QQQ **ratio > its MA** and **positive**
lookback relative return.

| Condition | Points |
| --------- | -----: |
| rel-return ≥ +15% (even if QQQ is choppy / ratio < MA) | 15 |
| `ok` and rel-return > +10% | 15 |
| `ok` (ratio > MA and rel-return > 0) | 12 |
| not `ok` but rel-return > 0 | 5 |
| −15% < rel-return ≤ 0 (noise / mild lag) | 5 |
| else / no benchmark | 0 |

Weekly: 13 bars / 5-bar MA. Daily: 63 / 20. Negative 63d RS fails the
Market Gate. Check counted when `relative_strength ≥ 12`.

## 5. MA Alignment & ATR Proximity (15 Points)

`structure_score(df, rs_rel=None)`:

- **Required:** EMA20 ≥ 0.98 × EMA50 (2% slack so a flat coil still scores).
  Else 0. SMA50 only if EMA50 is missing.
- 10 EMA ≥ 0.98 × 20 EMA → +5
- EMA20 > EMA50 (after slack) → +5
- Close ≥ 0.98 × 20 EMA → +5
- Soft `Pct_From_EMA50` haircut (never a reject or Market Gate fail):
  - ≤ 0.25 → no deduction
  - 0.25 – 0.50 → scaled deduction (max 5 if `RS 63d > 0.10`, else max 8)
  - > 0.50 → 7-pt (leader) or 10-pt (laggard) haircut, floor at 0

Check counted when `structure ≥ 10`.

## 6. Breakout RVOL Trigger (10 Points)

`rvol_trigger_score` — `RVOL = Volume / SMA20(Volume)`. Additive bonus
only. `RVOL < 1.0` on a tight coil does **not** fail the Market Gate.

| RVOL | Points |
|------|-------:|
| ≥ 2.0× | 10 |
| ≥ 1.5× | 8 |
| ≥ 1.2× | 6 |
| ≥ 1.0× | 4 |
| < 1.0× on a tight coil (`coil_width ≥ 15`) | 4 (quiet-coil credit) |
| else | 0 |

Never zeroed by the gate. Check counted when `rvol_trigger ≥ 6`.

## 7. Overhead Clearance (5 Points)

`overhead_clearance_score(df, price, atr, lookback=LOOKBACK)` — **Open sky
first:** if `Close ≥ 0.95 ×` the lookback high (52-week on daily) **or**
the series ATH, award the full 5 regardless of Fib 78.6% / 61.8% distance.
Otherwise, distance from close to the nearest High above price, in ATR.

| condition | Points |
|-----------|-------:|
| Open sky (`Close ≥ 0.95 × 52w/ATH`) or no high above | 5 |
| ≥ 3 ATR to nearest supply | 5 |
| ≥ 2 ATR | 3 |
| ≥ 1 ATR | 1 |

Check counted when `overhead_clearance ≥ 3`.

---

# Market gate (`check_coiled_cobra_market_gate`)

True unless the **ticker** has suffered a total trend failure:

- `Close < 0.90 × EMA50` (more than 10% below the 50 — a coil *at* the
  50 still passes), or
- 63-day relative strength `≤ −15%` (BA/DG-style lag; −2% noise does not drop)

Fail-open (`True`) when `close` / `ema50` / `rs_63d` are not provided
(unit tests that pass `benchmark_df=None` keep working).

**Not** drop conditions: `Pct_From_EMA50` (including > 0.25),
`Pct_From_Fib786`, `RVOL < 1.0`, or SPY/QQQ sitting below their 21-EMA /
50-SMA. Those stay on the scorecard (soft structure haircut, open-sky
space, RVOL bonus).

`spy_df` / `qqq_df` remain on the signature for call-site compatibility
and are not used for the pass/fail decision.

`Regime OK` on the CSV is a copy of `Market Gate`.

---

# Output contract

Live scan writes `data/logs/{weekly|daily}/coiled_cobra_setups_<YYYY-MM-DD>.csv`
using `config.SETUP_ROW_COLUMNS`. Soft ML ranks (`ML_Pred_Return`, `ML_Rank`)
are attached by `ml_ranker.attach_ml_ranks()` when artifacts exist; otherwise
rows sort by `Score`.

`Checks Met` is `{passed}/7` using the check thresholds above (not the hard
gates).

---

# Scanner Philosophy

Favor:

- Compression before expansion
- Relative-strength leaders
- Structural health
- Accumulation shelves
- Expansion volume at the trigger bar

Avoid:

- Chasing vertical breakouts already extended more than 50% above EMA50
- Requiring deep discounts that miss high-base coils
- Treating yearly Fib retracements or low RVOL as a mandatory gate
- Scoring MACD crossovers as the entry trigger
- Dropping RS leaders because QQQ is choppy

---

# Known Limitations

- Trade planner still uses Fib-anchored bounce geometry for Cobra rows
  (`max(Fib 78.6%, Close − 0.25×ATR)` entry, local 10-bar swing-low stop,
  2R / 3R targets). Coil-breakout entry/stop logic is a follow-up.
- IPO / short-history names (GEV, APP early years) may lack EMA100 / RS history.
- Weekly is the primary horizon for multi-month monster runs; daily is secondary.

---

# Legacy notes

v1 scored deep markdown + MACD < 0 + heavy yearly Fib (macro reversal).
v2 introduced the coil scorecard but still scored MACD crossover and an
optional Fib bonus. v3 added RVOL, overhead, and a SPY/QQQ penalty gate.
Live code is the **v3.1** scorecard above: ticker-level Market Gate, open-sky
space, RS ≥ 15% full credit, and a soft EMA50-extension haircut.

The CMT recommendations that originally motivated v3 (MACD as squeeze *state*,
RVOL as an additive trigger, overhead / open-sky space, EMA alignment with
a soft extension allowance) are **implemented** in the pillars and gates
above — they are not a future wishlist.
