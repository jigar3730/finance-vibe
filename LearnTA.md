# Technical analysis primer (Coiled Cobra)

This page is for **live Cobra**: compressed leaders vs QQQ that may expand. It is not a complete TA textbook. After this, read the [rubric](/docs/rubric) for exact points and gates.

**Offline labs** (not in `run_vibe.py`): [macro Vibe score](/docs/vibe) and [quality swing pullbacks](/docs/swing). Different questions, different indicators. Do not mix them with coil grades when you read a trade plan.

---

## 1. What a bar is

Each row of a raw CSV is a session (daily) or a week (weekly):

| Field | Meaning |
| ----- | ------- |
| **Open / High / Low / Close** | First, max, min, last traded price in that bar |
| **Volume** | Shares (or equivalent) traded |
| **Date** | Bar timestamp |

Ingest uses **adjusted** prices (`auto_adjust=True`) so splits/dividends do not fake coils. **QQQ** is the Nasdaq-100 proxy: Cobra asks whether the name is a **leader versus that benchmark**, not whether the whole market went up.

Daily is the project primary (5y × `1d`, coil window 30 bars). Weekly is slower confirmation (10y × `1wk`, coil window 8 bars).

---

## 2. Moving averages (EMA)

An **exponential moving average (EMA)** is a smoothed Close that weights recent bars more than old ones.

Cobra cares about a **stack**, not a single line:

- **EMA20** — short location (how stretched or coiled vs the last month of daily bars).
- **EMA50** — swing trend. Hard structure wants Close **above** a **rising** EMA50.
- **EMA100** — slower trend. EMA50 above EMA100 is “the leader is still healthy,” not a deep crash.

**Intuition:** a coil is a tight range **in an uptrend**, not a bounce from a wreck. Deep markdown under EMA20 was removed from the live scorecard on purpose (it filtered out high-base leaders).

`Pct_From_EMA20` / `Pct_From_EMA50` are `(Close − EMA) / EMA`. The ML model sees these; the rubric Score does not use them as pillars.

---

## 3. ATR (volatility scale)

**Average True Range (ATR)** measures how much the name typically moves (high−low plus gaps), not direction.

Cobra **normalizes** several ideas by ATR:

- MACD compression = `|MACD − Signal| / ATR` (tight oscillator, not a tiny-priced stock looking “compressed”).
- Coil width = N-bar range / ATR.
- Stops use ATR buffers so a $5 name and a $400 name are not treated as the same dollar stop.

**ATR_Pct** = ATR / Close. High ATR_Pct means violent names; the trainer **down-weights** them (`1 / ATR_Pct`) so lottery tickets do not dominate the loss.

---

## 4. MACD (compression and cross)

**MACD** is the gap between a fast and a slow EMA of Close; **Signal** is an EMA of MACD. Histogram ≈ MACD − Signal.

**Compression:** a small `|MACD − Signal| / ATR` means the two lines are coiled. Energy is stored. MACD may be **above zero** (uptrend coil). The old “MACD must be negative” idea was a reversal model, not this scanner.

**Cross:** prior bar MACD ≤ Signal and current MACD > Signal is an early bullish trigger, scaled by remaining tightness. No cross → 0 points. It is not required to pass if other pillars are strong enough — but compression still has a **hard gate** (≥ 5).

---

## 5. RSI (context, not the live gate)

**RSI (14)** is a 0–100 momentum oscillator. Classic: >70 stretched, <30 washed out.

On a **Cobra trade plan**, RSI is **context** on the row, not a live hard gate. Oversold RSI is closer to the **offline swing** lab (pullbacks). Do not require RSI 30 to take a high-base coil.

---

## 6. Relative strength vs QQQ

**Relative strength** here is not RSI. It is: is this stock’s price path **beating QQQ** over the lookback, and is the stock/QQQ ratio above its moving average?

Daily lookback is 63 bars with a 20-bar ratio MA; weekly is 13 / 5. Full pass plus positive relative return scores 12–15. **Hard gate: RS ≥ 12.** A tight coil that is lagging QQQ is rejected (false-positive class).

Forward labels for ML use the same idea: `Rel_Forward_2w` = stock 2-week return **minus** QQQ over the same dates.

---

## 7. Volume shelf

Price can sit in a range while volume **accumulates** in a price bin (auction / volume profile). Cobra scores a “shelf”: topology vs neighbors, location vs point of control, close vs bin. It is **not** a hard gate. Think “is this a pause with sponsorship?” not “highest volume of the year.”

---

## 8. Coil vs pullback (two different trades)

| | **Coiled Cobra (live)** | **Quality swing (offline)** |
| - | --------------------- | --------------------------- |
| Idea | Tight range, leader vs QQQ, ready to **expand** | Dip toward EMA20 in a regime, **confirm** next bar |
| Entry | Close of the passing coil bar | Pullback geometry (`compute_swing_levels`) |
| Stop | Coil_Low (else swing low), ATR and 5% caps | Structural swing low/high |
| Fib | Bonus only, not a gate | Used in some swing context |

**Vibe score** (−10 to +10) is a **macro** tape (SMA, MACD, RSI, CCI). Offline `pipeline_backtest` can gate swings on it. It is **not** the Cobra 0–100 Score.

---

## 9. What to look at on the dashboard table

- **Score / Grade** — rubric (A ≥ 85, B ≥ 70). Gates already applied.
- **Stock Entry / Stop / Target 1 / 2** — Close, Coil_Low constraint, 2R and 3R. See [trade plan math](/docs/trade-plan).
- **ML_Rank** — sort of predicted 2-week **relative** return. Tie-breaker, not a gate. Primer: [LearnML](/docs/learn-ml).

Next: [rubric](/docs/rubric) for numbers, then Track B in [Learn.md](/docs/learn).
