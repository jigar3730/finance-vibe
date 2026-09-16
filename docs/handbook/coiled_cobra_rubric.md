# Coiled Cobra Scanner
## Rubric v4.0 — Weekly-Only, 10-Year Lookback, Hard-Gated

Supersedes v3.1. Target: identify coils likely to break out within 1-2 weekly
bars, scanned against 10 years of weekly OHLCV history.

---

# What changed from v3.1 and why

| Change | Reason |
|---|---|
| Added a **Long-Term Trend Template** hard gate (30w/40w EMA stack, both rising) | v3.1 only checked EMA20/50 — a short-term construct. Nothing stopped a coil from passing inside a dead or declining multi-year trend. |
| `Checks Met` is now a **hard AND condition**, not a display-only counter | v3.1's fully additive scoring let strong unrelated pillars compensate for a pillar that outright failed (e.g. `structure = 0`, `coil_width = 0`), producing false positives. |
| `structure` and `coil_width` can each **independently disqualify** a setup | Same reason — these two pillars *are* the definition of "coiled," so a zero on either should not be recoverable via volume/RS/RVOL. |
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
- QQQ (primary) and SPY (secondary) weekly series, same length, for RS.
- Optional (recommended, not yet required): a sector/industry group proxy
  (ETF or custom basket) for group RS — see Known Limitations.

Minimum bars to evaluate: `max(COIL_BARS + 2, 60)`. Full scoring (ATH,
30w/40w EMA, BBWidth percentile) requires at least 160 bars (~3 years);
tickers with less history are flagged `Insufficient History`, not scored.

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
Fail-open (`True`) when benchmark data is unavailable, same as v3.1.

## Gate C — Coil Integrity (new — replaces implicit additive credit)
```
coil_width_score  ≥ 10   (i.e. range/ATR at or below the "partial" band)
structure_score   ≥ 8    (i.e. EMA10w/20w/40w alignment check passed)
```
Both must individually clear their own check threshold. A stock that isn't
actually coiling, or isn't in aligned short-term structure, is rejected
regardless of how well it scores elsewhere. This is the direct fix for the
v3.1 failure mode where a non-coiling name reached 65-70+ on unrelated
pillars.

## Gate D — Breadth (`Checks Met`)
```
Checks Met ≥ 5 of 6 scored pillars (see Stage 2)
```
Previously cosmetic; now load-bearing. A name can score ≥70 on raw points
but still fail if it's only clearing 3-4 of 6 checks — that pattern
indicates one or two pillars are being carried by outliers rather than
genuine multi-factor confluence.

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
percentile = rank of current BBWidth within trailing 104-156 week window
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
Gate C (see above — must independently clear ≥10 pts equivalent).

## 2. MACD directional filter (not scored — gate modifier only)

```
MACD line (12,26,9) > 0   → no penalty
MACD line ≤ 0             → -8 pt penalty applied to Stage 2 total, floor 0
```
Keeps the cheap, legitimate part of the old check (is momentum net-positive)
without treating MACD as a squeeze detector.

## 3. Structure & MA Alignment (20 Points)

```
Required: EMA10w ≥ 0.98 × EMA20w ≥ 0.98 × EMA40w   → else 0 (hard fail, Gate C)
```
| Condition | Points |
|---|---:|
| Full stack aligned (above) | 10 |
| Close ≥ 0.98 × EMA20w | +5 |
| EMA20w > EMA40w | +5 |

**Soft extension haircut** (tightened from v3.1):
| `Pct_From_EMA50w` | Deduction |
|---|---:|
| ≤ 0.20 | none |
| 0.20 - 0.30 | scaled, max -4 |
| 0.30 - 0.40 | max -8 (leader) / -12 (laggard) |
| > 0.40 | floor at 0 — treat as extended, not a coil |

Check counted when `structure ≥ 8`. Independently gates via Gate C.

## 4. Relative Strength vs QQQ (20 Points)

```
RS_13w = 13-week relative return vs QQQ (causal, Date <= as_of)
RS_ratio_5wMA = 5-week MA of stock/QQQ ratio
RS_line = cumulative stock/QQQ ratio series
```

Smoothed scoring (replaces the old flat "-15% to 0%" plateau):

| Condition | Points |
|---|---:|
| RS_13w ≥ +15% | 20 |
| ratio > 5wMA and RS_13w > +10% | 18 |
| ratio > 5wMA and RS_13w > 0 | 14 |
| RS_13w between 0% and +10%, ratio ≤ 5wMA | linear 6 → 12 |
| RS_13w between -15% and 0% | linear 0 → 6 (was a flat 5 in v3.1) |
| RS_13w < -15% | 0 (also fails Gate B) |

**+2 bonus** if `RS_line` is at a new 13-week high concurrent with the
price coil (RS-line cresting into a base — a leading institutional-
accumulation tell, previously not checked at all).

Check counted when `relative_strength ≥ 14`.

## 5. Volume Profile Shelf (15 Points, scaled down from v3.1's 20)

Unchanged methodology (30-bin, volume-weighted-on-Close, over full
available history now that 10yr is on hand — was capped at 52-week
`LOOKBACK`):

| Sub-score | Max | Rule |
|---|---:|---|
| Topology | 6 | `min(6, (bin_vol/avg_neighbor_vol) * 2)` |
| Auction value vs POC | 6 | ≤3 bins from POC = 6; ≤6 bins = 3; else 0 |
| Behavior | 3 | Close above bin center = 3, else 1 |

Check counted when `volume_shelf ≥ 8`.

## 6. Overhead Clearance / Open Sky (10 Points, Fib removed)

```
lookback_high = max(Close) over full available history (up to 10yr)
prior_swing_high = nearest prior swing high above current price
```

| Condition | Points |
|---|---:|
| Open sky: Close ≥ 0.95 × all-time-high | 10 |
| ≥ 3 ATR to nearest prior swing high | 8 |
| ≥ 2 ATR | 5 |
| ≥ 1 ATR | 2 |
| < 1 ATR | 0 |

Fib 61.8%/78.6% levels are **dropped from scoring**; retained only as an
informational CSV column (`Fib_Ref`) for manual chart review.

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

Check counted when `rvol_trigger ≥ 6`. Never gates — same non-gating
philosophy as v3.1, this pillar is about timing, not candidate quality.

---

# Stage 3 — Output Tiering (new)

All gate-passing, score ≥70, Checks Met ≥5/6 rows split into two tiers on
the CSV:

| Tier | Condition |
|---|---|
| **Actionable** | Above, AND `rvol_trigger ≥ 6` AND Close breaking above the `COIL_BARS`-window high |
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
| Any gate fail | Rejected - Gate Fail (A/B/C/D recorded) |
| Score < 70 with gates passed | Rejected - Below Threshold |

---

# Known Limitations / Follow-ups

- **Group/sector RS is not yet implemented.** Recommend adding a
  sector-ETF or custom-basket relative-strength column as a secondary
  (non-gating, informational-then-scored-later) check — leadership within
  a hot group is a meaningfully different signal than absolute RS vs. QQQ
  alone, and this rubric doesn't yet capture it.
- **BBWidth percentile window (104-156w) is a starting heuristic**, not
  back-tested here — worth validating against your existing trade archive
  (`backtest_and_backfill.md`) before fully replacing the old ATR-ratio
  method in production.
- **IPO / short-history names** still can't pass Stage 2 in full (need
  ≥160 weekly bars for EMA40w and 10yr ATH context) — same limitation as
  v3.1, now formalized as an explicit `Insufficient History` status rather
  than silently scoring on partial data.
- Daily-bar trigger confirmation (mentioned in Stage 3) is a **process
  recommendation**, not implemented in this rubric — it's a second, smaller
  scanner pass, not a rewrite of this one.
