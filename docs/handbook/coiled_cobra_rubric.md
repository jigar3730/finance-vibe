# Coiled Cobra Scanner
## Scoring Rubric & Technical Design

Version: 2.0 (Coil → Expansion)

Implementation: [`coiled_cobra.py`](../../src/finance_vibe/coiled_cobra.py). Theory and ML mapping: [`QUANT_ML_MANUAL.md`](QUANT_ML_MANUAL.md). Historical trade archive: [`backtest_and_backfill.md`](../architecture/backtest_and_backfill.md).

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


# Thoughs as a CMT on currrent Scoring logic 

As a Chartered Market Technician (CMT), I can tell you that this rubric is remarkably well-thought-out. It avoids the classic retail mistake of relying on a single lagging oscillator and instead builds a multi-dimensional framework: **Value/Volume (Volume Profile), Momentum/Volatility Compression (MACD/ATR/Coil), Trend/Market Context (Structure/RS), and Trigger Timing (Crossover).**

Here is a comprehensive breakdown of what works, what needs refinement, and what critical elements should be added to maximize your win rate.

---

## 1. What Is Good (and Why)

* **ATR Normalization for MACD & Coil Width (35% Combined Weight):**
* *Why:* Raw MACD lines or price percentage consolidation ranges are skewed by a stock's volatility. By normalizing MACD and price consolidation against **Average True Range (ATR)**, your rubric becomes asset-agnostic. It allows you to objectively rank a tight setup in a high-beta tech stock alongside a lower-volatility blue chip.


* **Volume Profile Shelf (20% Weight):**
* *Why:* Price action only tells half the story; volume reveals institutional positioning. Identifying a High Volume Node (HVN) or "volume shelf" right below your breakout level ensures you are launching off strong structural support. If the stock sits above a volume shelf, overhead supply is thin, meaning price can expand rapidly through Low Volume Nodes (LVNs).


* **Relative Strength vs. QQQ (15% Weight):**
* *Why:* Broad market drag is the primary cause of breakout failures. Selecting stocks displaying alpha (outperforming the QQQ prior to the breakout) ensures institutions are actively accumulating the name even during market choppiness.


* **"No MACD < 0" Constraint:**
* *Why:* A bullish MACD crossover below the zero line is often just a mean-reverting counter-trend rally. Requiring MACD to compress *above* zero guarantees you are hunting for expansion within an established, high-timeframe uptrend.


* **Fib Bonus as Optional (5% Weight):**
* *Why:* Keeping Fibonacci as a non-gating context metric is correct. Fib levels are dynamic reference zones, not structural liquidity triggers.



---

## 2. What Is Not So Good (and Why)

* **The "Bullish MACD Crossover" as a Trigger (10% Weight):**
* *Why:* MACD is a lagging, double-smoothed moving average derivative. Relying on a MACD crossover as your primary breakout trigger will consistently cause you to buy late—often near the exhaustion point of the initial expansion move.
* *Fix:* Use MACD solely as a **setup state** (measuring compression/coiling), not as your precise entry trigger.


* **Over-reliance on Momentum Oscillators (30% Total on MACD):**
* *Why:* MACD Compression (20%) + MACD Crossover (10%) allocates nearly a third of your scoring model to a single mathematical indicator family. This creates redundant confirmation bias.


* **Rising EMA Stack Needs Clear Definition:**
* *Why:* Simply saying "rising EMA stack" can lead to buying overextended moves where price is far above the moving averages. The structure score needs to account for **moving average alignment AND price proximity to those averages** (avoiding overextension).



---

## 3. Missing Key Components & How They Will Improve Results

To bring this rubric up to an institutional trading desk standard, adding the following key metrics will significantly reduce "false breakouts" (bull traps).

### A. Liquidity & Volume Expansion Trigger (Critical Missing Element)

* **The Concept:** While you measure the *historical* volume shelf, you lack a real-time **Volume Expansion Trigger**. True breakouts require institutional sponsorship at the moment of price expansion.
* **How to Implement:** Add a metric for **Relative Volume (RVOL)** on the breakout bar (e.g., $RVOL \ge 2.0\times$ the 20-day average volume) or **On-Balance Volume (OBV)** breaking out ahead of price.
* **Impact:** Filtering for a minimum $200\%$ volume spike on the breakout day eliminates low-liquidity "fakeouts" driven by retail liquidity sweeps.

### B. Proximity to Major Resistance / Pivot Level

* **The Concept:** Compression is useless if price breaks out directly into a massive multi-month resistance wall or major Supply Zone.
* **How to Implement:** Include a check for **Clear Overhead Space / Clean Chart**. Measure the distance between the breakout trigger price and the next major high or supply zone (minimum 2:1 or 3:1 reward-to-risk space required).
* **Impact:** Prevents entering breakouts that instantly hit overhead supply and reverse into distribution traps.

### C. Broad Market Health / Market Regime Filter

* **The Concept:** Individual stock relative strength can temporarily mask broad market distribution. If the S&P 500 (SPY) or Nasdaq (QQQ) is below a declining 20-day EMA, individual stock breakout win rates drop below 35%.
* **How to Implement:** Add a binary pass/fail condition: **"Is SPY/QQQ trading above its 21-day EMA or 50-day SMA?"**
* **Impact:** Acts as a safety switch to pause the rubric during choppy or declining macro regimes.

---

## 4. Suggested Revised Rubric Architecture

Here is a optimized re-allocation of weights incorporating these improvements:

| Category | Recommended Weight | Purpose & Technical Criteria |
| --- | --- | --- |
| **Volume Profile & Liquidity Base** | **20%** | Price consolidating immediately above a high-volume shelf (HVN) with low supply overhead. |
| **Volatility Compression (Coil)** | **20%** | Coil Width: Daily range / 14-period ATR $< 1.5$ (indicates tight squeeze prior to launch). |
| **MACD Compression & Squeeze State** | **15%** | MACD histogram compressing near zero, MACD line $> 0$ (energy coiling above zero line). |
| **Relative Strength vs QQQ** | **15%** | RS line forming higher high before price, or RS slope expanding over 10/20 days. |
| **Trend & Moving Average Alignment** | **15%** | Full alignment ($10 \text{ EMA} > 20 \text{ EMA} > 50 \text{ SMA}$) AND price within $1.5\text{ ATR}$ of 21 EMA (not overextended). |
| **Breakout Trigger & RVOL** | **10%** | Price clearing the defined consolidation pivot on **$RVOL \ge 2.0\times$** (Volume Expansion). |
| **Overhead Space / Fib Confluence** | **5%** | Minimum $3\times$ ATR distance to next major historical resistance / Fib extension zone. |