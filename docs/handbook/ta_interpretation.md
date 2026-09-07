# Technical Analysis & Trade Execution Guide (Finance Vibe Project)

How to read **macro Vibe Score** output from `analysis_engine.py` and
**tactical setups** from `swing_scanner.py`. Live scoring rules:
[`scoring_logic.md`](scoring_logic.md). Live swing geometry:
[`swing_setup.md`](swing_setup.md). Coil scorecard:
[`coiled_cobra_rubric.md`](coiled_cobra_rubric.md).

Default weekly ingest is **10-year weekly** (`config.TIMEFRAME_PROFILES`).
Daily is **5-year daily**. `high_beta` reuses daily OHLCV.

---

## 1. The Trend Pillars (Moving Averages)

Macro Vibe Score uses **SMA**. The swing and Coiled Cobra scanners use **EMA**.
That split is intentional.

| Metric | Used by | Interpretation |
| :--- | :--- | :--- |
| **SMA 20 / SMA 50** | `analysis_engine.build_features` | Medium-term equilibrium vs primary trend |
| **EMA 20 / EMA 50 / EMA 100** | `swing_scanner.add_indicators` | Tactical pullback + regime (`Close > EMA100` for longs) |
| **EMA 10 / SMA 50** | `coiled_cobra.structure_score` | Coil stack: 10 EMA > 20 EMA > 50 SMA |

* **Structural bull (macro):** $\text{Close} > \text{SMA20} > \text{SMA50}$ → +4 Trend.
* **Tactical bull (swing):** `EMA20 > EMA50` (rising), `Close > EMA100`, price inside the EMA20 proximity band.

---

## 2. Momentum & Boundaries (Oscillators)

### Relative Strength Index (RSI 14)

Both layers use Wilder RSI(14).

| Layer | Smoother / band |
| ----- | --------------- |
| Macro | `RSI_S` = **10-period SMA** of RSI (not an EMA-20) |
| Swing weekly | long **45–55**, short **50–60** |
| Swing daily | long **40–55** |
| Swing high_beta | long **35–58** |

Macro RSI risk: RSI > 80 caps the score at 5; 70 < RSI ≤ 80 is −1; RSI < 30 is +1.

### Commodity Channel Index (CCI 20)

Macro only. Typical price, **MAD** window, constant 0.015 (`_cci_fast`).
`CCI_S` is a **10-period SMA** of CCI.

* Constructive: −100 < CCI < 100 **and** CCI > CCI_S → +1
* Exhaustion: CCI > 200 → −2; CCI < −200 → +1

---

## 3. MACD (12, 26, 9)

Both layers use the **standard 12 / 26 / 9** parameterization — not 15 / 30 / 9.

| Layer | Series | Role |
| ----- | ------ | ---- |
| Macro | `MACD_H` = histogram; `MACD_S` = 9-EMA **of the histogram** | Momentum vs decay |
| Swing | `MACD_Hist` (pandas_ta) | Early turn: rising two bars while still ≤ 0, and `< 2 × 20-bar hist std` |
| Cobra | MACD line + hist / ATR | Squeeze state (0–15); deduct 5 if MACD line ≤ 0 |

---

## 4. Signal smoothers (what actually ships)

There are **no** `EMA_20_RSI` / `EMA_20_CCI` / `EMA_20_MACD` columns.

| Column | Definition |
| ------ | ---------- |
| `RSI_S` | SMA(RSI, 10) |
| `CCI_S` | SMA(CCI, 10) |
| `MACD_S` | EMA(MACD_H, 9) |

Bullish crossover language still applies to these smoothers: indicator above
its smoother means the latest impulse is expanding versus its recent window.

---

## 5. Tactical execution (swing levels)

There are **no** `Entry_Signal` / `Exit_Signal` / `Swing_High_4wk` fields.
Swing structure is a **10-bar** rolling low/high (`structure_bars=10`).

Quality-swing longs (after next-bar confirmation):

* **Limit entry:** `max(EMA20, Close − 0.25×ATR)`
* **Stop:** dual-constraint (local swing low vs `entry − 1.5×ATR`, 5% Close cap)
* **Targets:** weekly 1.25 / 2.25 ATR; daily 0.85 / 1.6 ATR; high_beta **2R / 3R**

Coiled Cobra rows use Fib 78.6% as an entry floor and the same 2R / 3R
targets. Full formulas: [`trade_plan_calculations.md`](trade_plan_calculations.md).

---

## Summary cheat sheet: bullish alignment

Macro Vibe (latest bar):

1. $\text{Close} > \text{SMA20} > \text{SMA50}$
2. $\text{RSI} > \text{RSI\_S}$ and $\text{MACD\_H} > \text{MACD\_S}$
3. Distance to SMA20 in $[0, 5\%]$
4. RSI not above 80; CCI not above 200

Quality swing long:

1. `EMA20 > EMA50` (rising), `Close > EMA100`
2. Close inside the EMA20 proximity band
3. RSI in the profile long band; MACD hist rising and ≤ 0
4. Next bar confirms (close ≥ EMA20 and ≥ setup low − slack)
