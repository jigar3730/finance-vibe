# Coiled Cobra Scanner
## Scoring Rubric & Technical Design

Version: 2.0 (Coil → Expansion)

---

# Overview

The **Coiled Cobra Scanner** identifies **compressed leaders ready to expand**
(coil → breakout), not swing pullbacks and not deep-discount mean reversion.

It looks for securities that are:

- Sitting in a volume / accumulation shelf
- Compressing on MACD (tight MACD−Signal spread) — MACD may be ≥ 0
- Holding healthy structure (price above a rising EMA50 / EMA100 stack)
- Showing positive relative strength vs QQQ
- Coiling in a tight N-bar range (low range / ATR)
- Optionally printing an early bullish MACD cross

Deep markdown under EMA20 was **removed** — it systematically filtered out
high-base coils (NVDA / APP / MU-class) that the scanner is meant to catch
before a monster run.

The scanner produces a maximum score of **100 points**.
Only setups scoring ≥ 70 are returned.

---

# Overall Scoring Matrix (v2 — Coil)

| Category | Weight | Purpose |
|----------|---------:|---------|
| Volume Profile Shelf | 20 | Accumulation zone |
| MACD Compression (ATR-normalized) | 20 | Coiled energy (no MACD &lt; 0) |
| Structure (rising EMA stack) | 20 | Leader still healthy |
| Relative Strength vs QQQ | 15 | Prefer names already leading |
| Coil Width (range / ATR) | 15 | Tight base before expansion |
| Bullish MACD Crossover | 10 | Early trigger |
| Fib bonus (optional) | 5 | Context only — not a gate |

Maximum Score = **100**

Hard gates: MACD compression ≥ 5, structure ≥ 8, **and relative strength ≥ 12**
(full RS pass vs QQQ). Negative-RS coils that only look tight (BA/DG-class) are rejected.

---

# Grade Classification

| Score | Grade | Meaning |
|--------:|-------|---------|
| 85-100 | A - Coil Ready | High-confluence pre-expansion |
| 70-84 | B - Valid Coil | Actionable coil |
| Below 70 | Reject | Insufficient confluence |

---

# Indicator Details

## 1. Volume Profile Shelf (20 Points)

Auction-market bins over the lookback window. Scores topology near high-volume
nodes, proximity to the POC, and close holding above the bin center.

## 2. MACD Compression (20 Points)

```
spread = abs(MACD - Signal) / ATR
```

| spread | Points |
|--------|-------:|
| ≤ 0.05 | 20 |
| ≤ 0.10 | 15 |
| ≤ 0.18 | 10 |
| ≤ 0.30 | 5 |
| else | 0 |

No requirement that MACD &lt; 0 — uptrend coils compress near or above zero.

## 3. Structure (20 Points)

- Close &gt; EMA50 → +8
- EMA50 rising → +6
- EMA50 &gt; EMA100 → +6

Replaces the old deep-markdown pillar.

## 4. Relative Strength vs QQQ (15 Points)

Stock/QQQ ratio above its MA and positive lookback relative return
(weekly: 13 bars / 5-bar MA; daily: 63 / 20). Stronger RS (&gt; +10%) scores full 15.

## 5. Coil Width (15 Points)

N-bar range / ATR (weekly N=8, daily N=30):

| width | Points |
|-------|-------:|
| ≤ 4 ATR | 15 |
| ≤ 6 ATR | 10 |
| ≤ 8 ATR | 5 |

## 6. Bullish MACD Cross (10 Points)

Prior bar MACD ≤ Signal and current MACD &gt; Signal.

## 7. Fib Bonus (0–5 Points)

Optional proximity to rolling Fib 61.8 / 78.6 (ATR-normalized). Not required.

---

# Scanner Philosophy

Favor:

- Compression before expansion
- Relative-strength leaders
- Structural health
- Accumulation shelves

Avoid:

- Chasing vertical breakouts already extended
- Requiring deep discounts that miss high-base coils
- Treating yearly Fib retracements as a mandatory gate

---

# Known Limitations

- Trade planner still uses Fib-anchored bounce geometry for Cobra rows;
  coil-breakout entry/stop logic is a follow-up.
- IPO / short-history names (GEV, APP early years) may lack EMA100 / RS history.
- Weekly is the primary horizon for multi-month monster runs; daily is secondary.

---

# Legacy v1 note

v1 scored deep markdown + MACD &lt; 0 + heavy yearly Fib (macro reversal). That
model remains documented in git history; live code is the v2 coil scorecard above.
