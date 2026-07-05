# Technical Analysis & Trade Execution Guide (Finance Vibe Project)

This guide outlines how to interpret macro Vibe Score output from `analysis_engine.py` and tactical setups from `swing_scanner.py`. The default weekly profile uses a **10-year weekly** lookback (`config.py` → `TIMEFRAME_PROFILES`). Full scoring rules: `src/finance_vibe/Scoring_Logic.md`.

## 1. The Trend Pillars (Moving Averages)
Moving averages establish the baseline structural bias of an asset.

| Metric | Definition | Technical Interpretation |
| :--- | :--- | :--- |
| **SMA 20** | 20-Week Simple Moving Average | Represents the medium-term market equilibrium. |
| **SMA 50** | 50-Week Simple Moving Average | Represents the primary, long-term structural trend. |

* **Structural Bull Market:** $\text{Close} > \text{SMA 20} > \text{SMA 50}$. This alignment confirms that short-term price discovery is outpacing long-term averages.
* **Trend Weakness:** If $\text{Close} < \text{SMA 20}$, it serves as an early warning that the structural trend is pausing or reversing, even if $\text{Close} > \text{SMA 50}$.

---

## 2. Momentum & Boundaries (Oscillators)
Oscillators quantify speed, velocity, and cyclical exhaustion points.

### Relative Strength Index (RSI 14)
* **Scale:** 0 to 100
* **Standard Overbought/Oversold:** $>70$ (Overbought), $<30$ (Oversold).
* **Weekly Context:** In powerful macro uptrends, the RSI frequently becomes "embedded" above 50 and rarely drops below 40 during healthy pullbacks.

### Commodity Channel Index (CCI 20)
* **Scale:** Unbounded (typically fluctuates between -300 and +300)
* **Cyclical Expansion:** Values $> +100$ indicate the stock is entering a strong, high-velocity cyclical impulse.
* **Cyclical Contraction:** Values $< -100$ signal major downward velocity.

---

## 3. The Trend Accelerator (MACD 15, 30, 9)
The Moving Average Convergence Divergence tracking is optimized with smoother windows (15, 30) for weekly analysis.

* **MACD Line:** Measures the absolute distance between the fast and slow exponential moving averages.
* **Interpretation:** 
    * **Positive & Rising:** The structural spread is widening; upside momentum is accelerating.
    * **Bearish Divergence:** If the asset price hits a new high but the MACD peak is lower than its previous peak, institutional accumulation is slowing down.

---

## 4. Signal Line Smoothers (EMA 20 Filters)
Applying a 20-period Exponential Moving Average directly to your indicators (`EMA_20_RSI`, `EMA_20_CCI`, `EMA_20_MACD`) creates a dynamic, rolling baseline rather than static boundary lines.

* **Bullish Crossover:** Indicator $>$ its respective `EMA_20`. This signals that immediate momentum is expanding relative to its recent historical window.
* **Bearish Crossover:** Indicator $<$ its respective `EMA_20`. This indicates that the rate of change is slowing down (rolling over), even if the absolute values still look superficially high.

---

## 5. Tactical Order Execution Framework (Swing Levels)

When `Entry_Signal` and `Exit_Signal` are both **False**, the asset is in a **No-Trade / Hold Zone**. During this phase, you use the raw dollar values of `Swing_High_4wk` and `Swing_Low_4wk` to set conditional pending orders.

### The Breakout Buy Blueprint (Entering Positions)
Do not buy at market price when momentum is stagnant. Instead, wait for a breakout above the local 4-week ceiling to confirm that institutional buying has returned.
* **Order Type:** Buy Stop Limit (or Buy Stop)
* **Execution Trigger:** $\text{Price} \ge (\text{Swing\_High\_4wk} + \$0.50)$
* **Objective:** Capture the sudden return of high-velocity macro momentum while protecting capital during sideways chop.

### The Structural Stop-Loss Blueprint (Risk Mitigation)
To protect trading capital against a deep cyclical correction, use the 4-week floor as your ultimate line in the sand.
* **Order Type:** Stop-Loss (or Stop Market)
* **Execution Trigger:** $\text{Price} \le (\text{Swing\_Low\_4wk} - \$0.40)$
* **Objective:** Automatically clear the position before a localized flush turns into a devastating multi-month structural breakdown.

---

## Summary Cheat Sheet: The "Perfect Bullish Vibe"

When verifying pipeline output records or dashboard dataframes, the ideal bullish momentum alignment is satisfied when:

1.  $$\text{Close} > \text{SMA 20} > \text{SMA 50}$$
2.  $$\text{RSI} > \text{EMA 20 RSI}$$
3.  $$\text{CCI} > \text{EMA 20 CCI}$$
4.  $$\text{MACD} > \text{EMA 20 MACD}$$