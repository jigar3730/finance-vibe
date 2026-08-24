# Coiled Cobra Scanner
## Scoring Rubric & Technical Design

Version: 2.2 (Coil → Expansion, proximity-to-highs, cleaned scorecard)

---

# Overview

The **Coiled Cobra Scanner** identifies **compressed leaders ready to expand**
(coil → breakout), not swing pullbacks and not deep-discount mean reversion.

It looks for securities that are:

- Contracting volume during the coil (dry-up vs the prior baseline)
- Compressing on MACD (tight MACD−Signal spread) — MACD may be ≥ 0
- Holding healthy structure (price above a rising EMA50 / EMA100 stack)
- Showing positive relative strength vs QQQ
- Coiling in a tight N-bar range (low range / ATR)
- Sitting near recent highs (63/126/252-bar, weekly 13/26/52)

MACD cross and Fib distances are **exported as flags/raw features**, not added to Score.
RSI (45–70 healthy zone) is an ML feature, not a Score pillar.

Pillars are **piecewise-linear**. Core six sum to **100** (clipped).
Only setups scoring ≥ 70 are returned. Score is a **filter and live ranking baseline**,
not the ML ranker.

---

# Overall Scoring Matrix (v2.2 — Coil)

| Category | Weight | Purpose | Gate? |
|----------|---------:|---------|-------|
| Volume contraction | 20 | Dry-up during the coil | no |
| MACD Compression (ATR-normalized) | 20 | Coiled energy (no MACD &lt; 0) | yes ≥ 5 |
| Structure (rising EMA stack) | 20 | Leader still healthy | yes ≥ 8 + Close &gt; rising EMA50 |
| Relative Strength vs QQQ | 15 | Prefer names already leading | yes ≥ 12 |
| Coil Width (range / ATR) | 13 | Tight base before expansion | no |
| Proximity to highs | 12 | Ready to push through supply | no |
| MACD cross (export only) | 0 | `MACD_Crossed` bool + `MACD_Spread_ATR` | no |
| Fib (export only) | 0 | `Pct_From_Fib618/786` optional ML context | no |

Hard gates: MACD compression ≥ 5, structure ≥ 8 **and** `healthy_structure_gate`,
**and relative strength ≥ 12** (full RS pass vs QQQ). Negative-RS coils that only
look tight (BA/DG-class) are rejected.

CSVs write pillars **and** raw measurements (`Volume_Contraction_Ratio`,
`Dist_High_*`, `Coil_Width_ATR`, `Coil_Width_Pctile`, coil OBV / up-volume /
volume-trend, RSI, `MACD_Crossed`).

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

## 1. Volume contraction (20 Points)

Median coil-window volume / median prior baseline volume (`Volume_Contraction_Ratio`).
Lower is better (dry-up). Not a hard gate. `Volume_Shelf` is the 0–20 score alias.

Complementary **raw** fields (not in Score): `OBV_Coil_Slope`, `Up_Volume_Ratio`,
`Volume_Trend_Ratio` (late-half / early-half median volume; declining ≈ accumulation).

Daily/weekly OHLCV cannot produce a true volume-at-price shelf.

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

## 5. Coil Width (13 Points)

N-bar range / ATR (weekly N=8, daily N=30):

| width (ATR) | Points |
|-------------|-------:|
| ≤ 4 | 13 |
| 6 | 9 |
| 8 | 4 |
| ≥ 10 | 0 |

Also export `Coil_Width_ATR` and `Coil_Width_Pctile` (percentile of current N-bar
range vs the prior 252 daily / 52 weekly bars, excluding the signal bar).

## 6. Proximity to highs (12 Points)

Distance from Close to rolling highs. Column names are always `Dist_High_{63,126,252}_Pct`
and `_ATR`; weekly uses 13/26/52 bars.

- Up to 8 pts from the near (63/13) high
- Up to 2 pts from the mid (126/26) high
- Up to 2 pts from the year (252/52) high

Not a hard gate.

## 7. MACD Cross and Fib (export only)

`MACD_Crossed` is 1 when prior MACD ≤ Signal and current MACD > Signal.
Tightness stays in `MACD_Spread_ATR`. **0 points in Score.**

Fib 61.8 / 78.6 distances remain optional columns. **0 points in Score.**

RSI(14) plus `RSI_Healthy` (45–70) and `RSI_Zone_Score` (0–5) are ML/context only.

---

# Scanner Philosophy

Favor:

- Compression before expansion
- Relative-strength leaders
- Structural health
- Accumulation / volume dry-up during the coil
- Proximity to recent highs without chasing a vertical extension

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
- Granular `Score` spreads vs v2.1; retrain the offline ML ranker after this
  scorecard (pillars + raw geometry — old Score+Fib / 10-feature models will not score).
  Daily primary target is `Rel_Forward_42d`; weekly is `Rel_Forward_13w`.

---

# Legacy v1 note

v1 scored deep markdown + MACD &lt; 0 + heavy yearly Fib (macro reversal). That
model remains documented in git history; live code is the v2.2 coil scorecard above.
