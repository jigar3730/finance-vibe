# Coiled Cobra Scanner
## Rubric v4.0 — Weekly-Only, 10-Year Lookback, Hard-Gated

Supersedes v3.1. Target: identify coils likely to break out within 1-2 weekly
bars, scanned against 10 years of weekly OHLCV history.

> **Source of truth:** `src/finance_vibe/coiled_cobra.py`
> (`evaluate_coiled_cobra`). This doc was re-audited against the code on
> 2026-10-08; where they disagree, the code wins.
>
> **Daily mode.** The rubric is weekly-native, but `coiled_cobra.py daily`
> runs the same logic on daily bars with every week-denominated period scaled
> ×5 (EMA 50/100/150/200, trend-rising lookback 40 bars, BBWidth window 650,
> history floors 300/800 bars). A few constants are set separately for daily
> rather than ×5: `COIL_BARS` = 30, RS lookback = 63, RS ratio MA = 20,
> overhead lookback = 252.

---

# What changed from v3.1 and why

| Change | Reason |
|---|---|
| Added a **Long-Term Trend Template** hard gate (30w/40w EMA stack, both rising) | v3.1 only checked EMA20/50 — a short-term construct. Nothing stopped a coil from passing inside a dead or declining multi-year trend. |
| `Checks Met` is now a **hard AND condition**, not a display-only counter | v3.1's fully additive scoring let strong unrelated pillars compensate for a pillar that outright failed (e.g. `structure = 0`, `coil_width = 0`), producing false positives. |
| `structure` and `vol_contraction` (which replaced v3.1's `coil_width`) can each **independently disqualify** a setup | Same reason — these two pillars *are* the definition of "coiled," so a zero on either should not be recoverable via volume/RS/RVOL. |
| Volatility compression now measured with **Bollinger Band Width percentile**, not MACD | MACD is a momentum/trend indicator, not a volatility indicator. Using `|Hist|/ATR` as a "squeeze" score conflated momentum convergence with range contraction and produced false compression reads. |
| BBWidth percentile is computed over a **rolling 2-3 year window**, not the full 10-year history | Regime drift (2020 crash, 2022 bear) would otherwise distort what "tight" means for a given stock. The 10-year history is used for ATH/overhead detection instead. |
| MACD demoted to a **small binary directional filter** | Keeps a genuinely useful, cheap check (is momentum net-positive) without double-counting volatility. |
| Coil tightness bands **tightened**; the old 2.2-4.0 ATR "high-beta pause" credit removed | That band was rewarding non-compression as if it were a coil variant — the single largest source of noise in v3.1. |
| EMA-extension soft-haircut ceiling **lowered** (was: leaders could extend 50% above EMA50 with minor deduction) | 50%-extended names are stage-2 markup, not fresh coils. Early-detection scanners should punish extension harder. |
| RS scoring **smoothed**; flat "-15% to 0%" plateau removed; added **RS-line-new-high** bonus | The old step function couldn't distinguish a stock lagging by 1% from one lagging by 14%. RS-line cresting is a leading tell IBD/O'Neil-style traders rely on. |
| Fibonacci retracement/extension **removed from scoring** (kept only as an informational column) | Almost always redundant with the open-sky override; not carrying independent signal. |
| Output now splits into **Watchlist vs. Actionable** | Score ≥70 alone conflated "well-coiled, no trigger yet" with "breaking out today." These need to be visibly different tiers. |

---

# Data requirements

- 10 years of weekly OHLCV (~520 bars) per ticker.
- QQQ weekly series, same length, for RS (`BENCHMARK = "QQQ"`). SPY is
  loaded and passed to the market gate but is currently **not used** in any
  pass/fail or scoring decision.
- Optional (recommended, not yet required): a sector/industry group proxy
  (ETF or custom basket) for group RS — see Known Limitations.

Minimum bars to evaluate: `max(COIL_BARS + 2, 60)` (`MIN_BARS_TO_EVALUATE`).
Full scoring (ATH, 30w/40w EMA, BBWidth percentile) requires at least 160
bars (~3 years, `MIN_BARS_FULL_SCORE`). Below 160 bars the ticker is **not
scored**: `evaluate_coiled_cobra` returns `None` regardless of
`include_rejects`. The scanner's rejection summary counts files with < 60
bars as `insufficient_history`; files with 60-159 bars land in the generic
`IGNORE` bucket. No row or `Grade` reads `Insufficient History`.

---

# Stage 1 — Hard Gates (pass/fail, before any scoring)

All four must pass. Any failure = `Rejected`, reason recorded, row excluded
from live scan output (kept only if `include_rejects=True`, same as v3.1).

## Gate A — Long-Term Trend Template
```
Close > EMA30w > EMA40w
EMA40w rising over trailing 8 weeks (EMA40w[t] > EMA40w[t-8])
```
Rationale: this is the weekly analog of Minervini's daily 50/150/200-SMA
stack. It filters names that are coiling *inside* a long-term downtrend or
dead sideways market — the single biggest gap in v3.1.

## Gate B — Ticker Market Gate (retained from v3.1, unchanged thresholds)
```
Close ≥ 0.90 × EMA50w   (fails if more than 10% below the 50)
RS_13w > -15%           (fails only on outright multi-quarter lag)
```
EMA50w here is the plain 50-bar EMA (`EMA50`). A ticker exactly at
`RS_13w = -15%` fails. Fail-open: when benchmark data is unavailable (RS is
`None`) only the EMA50 check applies, same as v3.1. SPY/QQQ trend is not part
of this gate.

## Gate C — Coil Integrity (new — replaces implicit additive credit)
```
vol_contraction   ≥ 12   (GATE_C_VOL_CONTRACTION_MIN — BBWidth pctl ≤ 35th and not rising, or ≤ 10th)
structure_score   ≥ 8    (GATE_C_STRUCTURE_MIN — EMA10w/20w/30w/40w alignment passed)
```
Both must individually clear their own check threshold. A stock that isn't
actually coiling, or isn't in aligned short-term structure, is rejected
regardless of how well it scores elsewhere. This is the direct fix for the
v3.1 failure mode where a non-coiling name reached 65-70+ on unrelated
pillars.

> **Hardened 2026-09-18**: `structure_score`'s hard stack check originally
> only compared EMA10w/20w/40w, leaving EMA30w unconstrained — a stock with
> a hole mid-stack (e.g. EMA20w dipping below EMA30w) could still pass both
> Gate A and Gate C despite not being a clean ascending EMA fan. Now requires
> the full EMA10w ≥ EMA20w ≥ EMA30w ≥ EMA40w hierarchy (see Structure & MA
> Alignment below).

## Gate D — Breadth (`Checks Met`)
```
Checks Met ≥ 4 of 6 scored pillars (see Stage 2)
```
Previously cosmetic; now load-bearing. A name can score ≥70 on raw points
but still fail if it's only clearing 2-3 of 6 checks — that pattern
indicates one or two pillars are being carried by outliers rather than
genuine multi-factor confluence.

> **Recalibrated 2026-09-18** (originally ≥5/6): a 264-ticker, 10-year
> weekly walk-forward backtest compared signals passing Gates A-D against
> signals passing only A-C. The ≥5/6 cutoff showed no measurable win-rate or
> expectancy edge over ≥4/6 (37.8% / +0.145R vs 37.5% / +0.132R on the
> blocked increment — overlapping confidence intervals) while discarding
> ~65% of otherwise equal-or-better `B/Watchlist`-tier signal volume, where
> the blocked increment actually outperformed (+0.121R vs +0.087R). The
> `A/Actionable` tier — the strongest cohort by far — was unaffected by the
> threshold either way. See `gate_d_ablation_trades.csv` backtest run for
> detail.

---

# Stage 2 — Scoring (100 points, only run on gate-passing names)

| Category | Weight | `Parts` key |
|---|---:|---|
| Volatility Contraction (BBWidth percentile) | 25 | `vol_contraction` |
| Structure & MA Alignment | 20 | `structure` |
| Relative Strength vs QQQ (+ RS-line, + group) | 20 | `relative_strength` |
| Volume Profile Shelf | 15 | `volume_shelf` |
| Overhead Clearance / Open Sky | 10 | `overhead_clearance` |
| RVOL Breakout Trigger | 10 | `rvol_trigger` |

`MIN_PASS_SCORE = 70` (unchanged). `GRADE_A_SCORE = 85` (unchanged).

MACD is **not** a scored pillar in v4.0 — see "MACD directional filter" below,
applied as a small pre-score modifier instead.

---

## 1. Volatility Contraction (25 Points)

Replaces `macd_compression` + folds `coil_width`'s old role into one
properly-measured pillar.

```
BBWidth = (UpperBB20 - LowerBB20) / MiddleBB20      # weekly, 20-period, 2 std
         = 4 × std20(Close) / SMA20(Close)         # pandas sample std (ddof=1)
percentile = rank of current BBWidth within trailing 130-week window
             (BBWIDTH_WINDOW, midpoint of the 104-156w range; needs ≥ 40 bars)
```

| BBWidth percentile (own history) | Points |
|---|---:|
| ≤ 10th percentile | 25 |
| ≤ 20th percentile | 20 |
| ≤ 35th percentile | 12 |
| ≤ 50th percentile | 6 |
| > 50th percentile | 0 |

Additionally require the percentile to have been **declining over the
trailing `COIL_BARS` (8 weeks)** — i.e. contraction is a trend, not a
snapshot — or halve the points. This is the actual VCP-style check that
v3.1's static ATR ratio never performed.

Check counted when `vol_contraction ≥ 12`. This check is also one half of
Gate C (see above — the same ≥ 12 threshold, `GATE_C_VOL_CONTRACTION_MIN`).
Note the halving rule: in code the points are halved (integer) when the
percentile is higher than it was `COIL_BARS` ago. A ≤ 10th-percentile reading
still passes when halved (25 → 12), but a ≤ 20th-percentile reading drops to 10
and **fails** both the check and Gate C.

## 2. MACD directional filter (not scored — gate modifier only)

```
MACD line (12,26,9) > 0   → no penalty
MACD line ≤ 0             → -8 pt penalty applied to Stage 2 total, floor 0
```
Keeps the cheap, legitimate part of the old check (is momentum net-positive)
without treating MACD as a squeeze detector.

## 3. Structure & MA Alignment (20 Points)

```
Required: EMA10w ≥ 0.98 × EMA20w ≥ 0.98 × EMA30w ≥ 0.98 × EMA40w   → else 0 (hard fail, Gate C)
```
| Condition | Points |
|---|---:|
| Full stack aligned (above) | 10 |
| Close ≥ 0.98 × EMA20w | +5 |
| EMA20w > EMA40w | +5 |

**Soft extension haircut** (tightened from v3.1). Measured from the **40w
EMA** (`(Close − EMA40w) / EMA40w`), *not* EMA50w or the `Pct_From_EMA50`
output column:
| Extension above EMA40w | Deduction |
|---|---:|
| ≤ 0.20 | none |
| 0.20 - 0.30 | linear 0 → -4 |
| 0.30 - 0.40 | linear -4 → -8 (leader) / -4 → -12 (laggard) |
| > 0.40 | pillar floored at 0, treated as extended, not a coil |

"Leader" means `RS_13w > +10%` (`RS_LEADER_EXT`). Deductions are rounded to
whole points and the pillar is floored at 0.

Check counted when `structure ≥ 8`. Independently gates via Gate C.

## 4. Relative Strength vs QQQ (20 Points)

```
RS_13w = stock 13w return − QQQ 13w return (causal, Date <= as_of)
ratio  = stock Close / QQQ Close (the "RS line")
RS_ratio_5wMA = 5-week SMA of ratio
"ratio > 5wMA" also requires RS_13w > 0 (it is relative_strength()'s `ok` flag)
```

Smoothed scoring (replaces the old flat "-15% to 0%" plateau):

| Condition | Points |
|---|---:|
| RS_13w ≥ +15% | 20 |
| ratio > 5wMA and RS_13w > +10% | 18 |
| ratio > 5wMA and RS_13w > 0 | 14 |
| RS_13w between 0% and +10%, ratio ≤ 5wMA | linear 6 → 12 |
| RS_13w between +10% and +15%, ratio ≤ 5wMA | 12 (clamped) |
| RS_13w between -15% and 0% | linear 0 → 6 (was a flat 5 in v3.1) |
| RS_13w ≤ -15% | 0 (also fails Gate B) |
| No benchmark / too little overlap | 0 |

**+2 bonus** (capped at 20) if the ratio is at its trailing 13-week high on
the as-of bar, i.e. the RS line cresting, a leading institutional-accumulation
tell that v3.1 didn't check. It applies only when the pillar already scores
> 0 and does not check the price coil itself. The final value is rounded to
an integer.

The group-RS part of the pillar is **not implemented** (see Known
Limitations).

Check counted when `relative_strength ≥ 14`.

## 5. Volume Profile Shelf (15 Points, scaled down from v3.1's 20)

Unchanged methodology: 30 equal-width price bins spanning the window's
Low-min to High-max, with Close histogrammed and weighted by Volume. The
pillar takes the **better of two windows**: the last 20 bars (the coil) and
the full available history (up to 10yr, previously capped at the 52-week
`LOOKBACK`). The sub-scores are summed and rounded to an integer.

| Sub-score | Max | Rule |
|---|---:|---|
| Topology | 6 | `min(6, (bin_vol/avg_neighbor_vol) * 2)` |
| Auction value vs POC | 6 | ≤3 bins from POC = 6; ≤6 bins = 3; else 0 |
| Behavior | 3 | Close above bin center = 3, else 1 |

Check counted when `volume_shelf ≥ 8`.

## 6. Overhead Clearance / Open Sky (10 Points, Fib removed)

```
high_52   = max(High) over the last LOOKBACK bars (52w)
ath       = max(High) over full available history (up to 10yr)
local_high = max(High) over the last RS_LOOKBACK bars (13w)
```

| Condition | Points |
|---|---:|
| Open sky: Close ≥ 0.95 × `high_52`, `ath`, **or** `local_high` | 10 |
| ≥ 3 ATR below `local_high` | 8 |
| ≥ 2 ATR | 5 |
| ≥ 1 ATR | 2 |
| < 1 ATR | 0 |

Levels use bar **Highs**, not Closes. There is no swing-high detection: the
code measures room up to `min(high_52, local_high)`, which is always the
13-week high. In practice the 13-week "local open sky" clause gives full
points to any stock within 5% of its quarter high, so most tight coils near
the top of their base score 10. The rubric's original intent (ATH open sky,
then ATR distance to the nearest prior swing high) was **never
implemented**. The local-high clause dates from v3.x. ATR is ATR(14).

Fib 61.8%/78.6% levels (52-bar rolling High/Low range) are **dropped from
scoring**. They appear only as informational CSV columns (`Fib 61.8%`,
`Fib 78.6%`) for manual chart review. `Fib Score` is always written as 0.0.

Check counted when `overhead_clearance ≥ 5`.

## 7. RVOL Breakout Trigger (10 Points, additive — unchanged intent)

```
RVOL = Volume / SMA20w(Volume)
```
| RVOL | Points |
|---|---:|
| ≥ 2.0× | 10 |
| ≥ 1.5× | 8 |
| ≥ 1.2× | 6 |
| ≥ 1.0× | 4 |
| < 1.0× on a confirmed tight coil (`vol_contraction ≥ 20`) | 4 (quiet-coil credit) |
| else | 0 |

`SMA20w(Volume)` includes the current bar.

Check counted when `rvol_trigger ≥ 6`. Never gates — same non-gating
philosophy as v3.1, this pillar is about timing, not candidate quality.

---

# Stage 3 — Output Tiering (new)

All gate-passing, score ≥70, Checks Met ≥4/6 rows split into two tiers on
the CSV:

| Tier | Condition |
|---|---|
| **Actionable** | Above, AND `rvol_trigger ≥ 6` (i.e. RVOL ≥ 1.2×) AND Close > max High of the prior `COIL_BARS` bars (excluding the current bar) |
| **Watchlist** | Above, but no volume trigger yet / still inside the coil range |

This directly fixes the v3.1 problem of well-coiled-but-not-yet-firing
names being visually indistinguishable from names breaking out today.

**Recommended pipeline:** run this weekly scorecard to produce the
candidate list, then re-check `Actionable` names against **daily** bars for
the specific trigger day/price — weekly bars pick the setup, daily bars
confirm the exact entry. Weekly-only scanning can tell you "this stock is
coiled and ready," but the precise day it clears the range is a daily-bar
question.

---

# Grade Classification

| Score | Grade |
|---:|---|
| 85-100, all gates pass, Actionable tier | A - Coil Ready |
| 85-100, all gates pass, Watchlist tier | A - Watch |
| 70-84, all gates pass, Actionable tier | B - Valid Coil |
| 70-84, all gates pass, Watchlist tier | B - Watch |
| Any gate fail | `Rejected - Gate Fail (A/C)`: every failing gate listed, `/`-joined |
| Score < 70 with gates passed | Rejected - Below Threshold |

---

# Known Limitations / Follow-ups

- **Group/sector RS is not yet implemented.** Recommend adding a
  sector-ETF or custom-basket relative-strength column as a secondary
  (non-gating, informational-then-scored-later) check — leadership within
  a hot group is a meaningfully different signal than absolute RS vs. QQQ
  alone, and this rubric doesn't yet capture it.
- **BBWidth percentile window (130w, chosen from the 104-156w range) is a
  starting heuristic**, not back-tested here. Worth validating against your
  existing trade archive (`backtest_and_backfill.md`) before fully replacing
  the old ATR-ratio method in production.
- **IPO / short-history names** (< 160 weekly bars) are not scored at all, so
  they never reach Stage 2. Nothing scores them on partial data, but nothing
  surfaces an explicit `Insufficient History` status either. They show up
  only in the scanner's rejection-summary log (`insufficient_history` / `IGNORE`).
- **Overhead Clearance is looser than intended.** See pillar 6: the 13-week
  local-high clause gives full marks well short of true open sky, and there
  is no prior-swing-high detection. Fixing it changes `Score`, so it needs a
  `RUBRIC_VERSION` bump and backtest validation.
- **SPY is unused.** It is loaded and passed to the market gate but ignored.
- Daily-bar trigger confirmation (mentioned in Stage 3) is a **process
  recommendation**, not implemented in this rubric — it's a second, smaller
  scanner pass, not a rewrite of this one.
