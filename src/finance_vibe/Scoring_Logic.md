# Macro Vibe Score — Scoring Logic

This document describes the **implemented** scoring rules in `analysis_engine.py`. The score ranks macro tradability on a **−10 to +10** scale. It favors trend structure and pullback timing over raw momentum.

For tactical entry rules (EMA pullbacks), see `swing_scanner.py` and `swing_setup_readme.md`.

---

## Indicators (computed on each ticker’s OHLC CSV)

| Output | Definition |
| ------ | ---------- |
| **SMA20 / SMA50** | Simple moving average of `Close` (20 / 50 periods) |
| **MACD_H** | MACD histogram: EMA(12) − EMA(26) of `Close`, minus 9-period EMA of that line |
| **MACD_S** | 9-period EMA of `MACD_H` |
| **RSI** | 14-period Wilder RSI on `Close` |
| **RSI_S** | 10-period SMA of `RSI` |
| **CCI** | 20-period CCI using typical price and **mean absolute deviation** (0.015 constant) |
| **CCI_S** | 10-period SMA of `CCI` |

Minimum history: **60 rows** per ticker (rows with insufficient data are skipped).

---

## Score components (additive, applied in order)

Each rule reads the **latest completed bar** only.

### 1. Trend (±4)

| Condition | Points |
| --------- | ------ |
| `Close > SMA20 > SMA50` | +4 |
| `Close < SMA20 < SMA50` | −4 |
| Otherwise | 0 |

Partial trend credit (e.g. above SMA20 only) is **not** awarded.

### 2. Momentum (±2)

| Condition | Points |
| --------- | ------ |
| `MACD_H > MACD_S` **and** `RSI > RSI_S` | +2 |
| `MACD_H < MACD_S` **and** `RSI < RSI_S` | −2 |
| Otherwise | 0 |

### 3. Momentum decay (−1)

| Condition | Points |
| --------- | ------ |
| `MACD_H < MACD_S` **and** `Close > SMA20` | −1 |

Penalizes bullish price structure with weakening momentum.

### 4. Timing — distance from SMA20 (−2 to +2)

Let `dist = (Close − SMA20) / SMA20`.

| Condition | Points |
| --------- | ------ |
| `0.0 ≤ dist ≤ 0.05` (pullback into SMA20) | +2 |
| `dist > 0.12` (overextended above SMA20) | −2 |
| `dist < −0.05` (meaningfully below SMA20) | −1 |
| Otherwise | 0 |

### 5. CCI cyclical logic (−2 to +1)

| Condition | Points |
| --------- | ------ |
| `−100 < CCI < 100` **and** `CCI > CCI_S` | +1 |
| `CCI > 200` (exhaustion) | −2 |
| `CCI < −200` (deep cyclical low) | +1 |
| Otherwise | 0 |

CCI is **not** used as a pure momentum booster outside the constructive band.

### 6. RSI risk governors

Applied **after** the components above:

| Condition | Effect |
| --------- | ------ |
| `RSI > 80` | Cap total score at **5** (`min(score, 5)`) |
| `70 < RSI ≤ 80` | −1 |
| `RSI < 30` | +1 |

### 7. High-score persistence (−2)

If score is **≥ 7** after steps 1–6, require `MACD_H > 0` **and** `RSI > 50`. If not met, **−2**.

### 8. Final clip

Score is clipped to **\[−10, +10\]** and stored as an integer.

---

## Sentiment and action labels

| Score | Sentiment | Action |
| ----- | --------- | ------ |
| ≥ 9 | Bullish | STARTER + ADD ON PULLBACK |
| 7 – 8 | Bullish | STARTER POSITION |
| 5 – 6 | Positive | WATCH / SCALE IN |
| 2 – 4 | Neutral | WAIT |
| −1 – 1 | Neutral | NO EDGE |
| −4 – −2 | Bearish | REDUCE / HEDGE |
| ≤ −5 | Bearish | AVOID / SHORT BIAS |

---

## Pipeline output

- **File:** `data/logs/{mode}/vibe_report_<YYYY-MM-DD>.csv`
- **Columns:** Ticker, Price, SMA20, SMA50, CCI, CCI_S, MACD_H, MACD_S, RSI, RSI_S, Score, Sentiment, Action
- **Sort:** Score descending, then Ticker ascending

---

## Design intent

- Penalize **overextension** (RSI caps, distance from SMA20, CCI > 200).
- Reward **pullbacks into trend** (timing band near SMA20).
- Use action labels that imply **position sizing**, not binary conviction.
- Keep macro (SMA-based Vibe Score) separate from tactical (EMA-based swing scanner).

## Backtest macro gate

The offline backtest in `pipeline_backtest.py` applies a macro filter that the live pipeline does **not** enforce today:

| Setup | Required Vibe Score | Aligns with action band |
| ----- | ------------------- | ----------------------- |
| `SETUP_LONG` | ≥ 7 | STARTER POSITION or higher |
| `SETUP_SHORT` | ≤ −2 | REDUCE / HEDGE or lower |

Thresholds are configurable via `config.py` (`BACKTEST_LONG_MIN_SCORE`, `BACKTEST_SHORT_MAX_SCORE`) or CLI flags.
