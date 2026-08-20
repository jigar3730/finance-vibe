# Coiled Cobra Scanner
## Scoring Rubric & Technical Design

Version: 2.1 (Coil → Expansion, granular pillars)

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

Pillars are **piecewise-linear** (not 5-point cliffs). Core six sum to **100**;
the Fib bonus is 0–5 and the printed total is **clipped at 100**.
Only setups scoring ≥ 70 are returned.

---

# Overall Scoring Matrix (v2.1 — Coil)

| Category | Weight | Purpose |
|----------|---------:|---------|
| Volume Profile Shelf | 20 | Accumulation zone |
| MACD Compression (ATR-normalized) | 20 | Coiled energy (no MACD &lt; 0) |
| Structure (rising EMA stack) | 20 | Leader still healthy |
| Relative Strength vs QQQ | 15 | Prefer names already leading |
| Coil Width (range / ATR) | 15 | Tight base before expansion |
| Bullish MACD Crossover | 10 | Early trigger (quality-scaled) |
| Fib bonus (optional) | 5 | Context only — not a gate |

Core maximum = **100**. Fib may add up to 5, then **clip to 100**.

Hard gates: MACD compression ≥ 5, structure ≥ 8, **and relative strength ≥ 12**
(full RS pass vs QQQ). Negative-RS coils that only look tight (BA/DG-class) are rejected.

Scan / backtest CSVs write each pillar: `Volume_Shelf`, `MACD_Compression`,
`Structure`, `RS_Score`, `Coil_Width`, `MACD_Cross`, `Fib_Bonus`.

---

# Grade Classification

| Score | Grade | Meaning |
|--------:|-------|---------|
| 85-100 | A - Coil Ready | High-confluence pre-expansion |
| 70-84 | B - Valid Coil | Actionable coil |
| Below 70 | Reject | Insufficient confluence |

---

# Indicator Details

Interpolation uses linear segments between the knots below, rounded to 2 decimals.
Values outside the first/last knot clamp to that knot’s points.

## 1. Volume Profile Shelf (20 Points)

Auction-market bins over the lookback window. Three sub-scores (floats):

| Sub-score | Max | Rule |
|-----------|----:|------|
| Topology | 8 | `min(8, (bin_vol / neighbor_vol) × 2.5)` — no integer truncation |
| Value vs POC | 8 | Tent: 5 at POC, peak 8 at 1–3 bins, 8→4 at 3–6, 4→0 at 6–10 |
| Behavior | 4 | 4 if close ≥ bin center; decay toward 1 if below |

Prefer *near* value (not glued to the POC). Not a hard gate.

## 2. MACD Compression (20 Points)

```
spread = abs(MACD - Signal) / ATR
```

| spread | Points |
|--------|-------:|
| ≤ 0.05 | 20 |
| 0.10 | 15 |
| 0.18 | 10 |
| 0.30 | 5 |
| ≥ 0.40 | 0 |

No requirement that MACD &lt; 0 — uptrend coils compress near or above zero.

**Worked contrast:** spread 0.075 → **17.5**; spread 0.09 used to share the old
15-point bucket with 0.075 and now scores **16.0**. Tighter coil ranks higher.

## 3. Structure (20 Points)

Same three checks, scaled (needs ATR):

- Close vs EMA50 → 0 if close ≤ EMA50; else 4→8 over 0–0.5 ATR above (cap 8)
- EMA50 rising → 3-bar EMA50 change / ATR; 0 if not rising; 6 at ≥ 0.05 ATR rise
- EMA50 > EMA100 → 0 if missing or inverted; else 3→6 over 0–0.5 ATR gap

Hard gate ≥ 8 still requires close above EMA50 (that sub-score is the only path to 8).

## 4. Relative Strength vs QQQ (15 Points)

Stock/QQQ ratio above its MA **and** positive lookback relative return
(weekly: 13 bars / 5-bar MA; daily: 63 / 20).

| Condition | Points |
|-----------|-------:|
| Full pass, rel-return ≥ +10% | 15 |
| Full pass, rel-return 0% to +10% | 12 + 3 × (rel / 0.10) |
| Rel-return > 0 but not a full pass | 1–5 (fails the ≥ 12 gate) |
| Else | 0 |

**Worked contrast:** full-pass +5% RS → **13.5**, not lumped with +0.1% at 12.

## 5. Coil Width (15 Points)

N-bar range / ATR (weekly N=8, daily N=30):

| width (ATR) | Points |
|-------------|-------:|
| ≤ 4 | 15 |
| 6 | 10 |
| 8 | 5 |
| ≥ 10 | 0 |

## 6. Bullish MACD Cross (10 Points)

Prior bar MACD ≤ Signal and current MACD > Signal. Otherwise **0**
(no approach credit).

If crossed, scale by remaining tightness:

```
10 × clip(1 - (spread - 0.05) / 0.25, 0.60, 1.00)
```

10 at spread ≤ 0.05; 6 at wide/late crosses.

## 7. Fib Bonus (0–5 Points)

Optional proximity to rolling Fib 61.8 / 78.6 (ATR-normalized). Not required.
Does not change hard gates. Cannot push the printed score above 100.

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

- Historical validation (`coiled_cobra_backtest.py --backtest`) is forward /
  QQQ-relative expansion, not bounce P&L. Filter `Is_New_Coil` to score episodes.
- IPO / short-history names (GEV, APP early years) may lack EMA100 / RS history.
- Daily is the primary horizon (5y `1d`, coil N=30, 1–3 month options). Weekly is slower confirmation (10y `1wk`, coil N=8, LEAPS).
- Granular `Score` spreads vs v2.0 buckets; retrain the offline ML ranker if ranks look stale
  (10 features = 7 pillars + EMA distances + ATR_Pct — old Score+Fib models will not score).

---

# Legacy v1 note

v1 scored deep markdown + MACD &lt; 0 + heavy yearly Fib (macro reversal). That
model remains documented in git history; live code is the v2.1 coil scorecard above.
