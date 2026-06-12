## 🔧 Scoring Logic Overhaul (v2)

This update fixes systemic flaws in the original scoring model that caused score saturation, late entries, and misleading “GO ALL IN” signals. The focus is now on **tradable opportunity**, not raw momentum.

---

### 🚨 Problems Addressed

- Scores frequently maxed out at 9–10 for overextended assets
- RSI and CCI rewarded strength even in exhaustion zones
- No distinction between trend quality and entry timing
- Binary, reckless action labels (e.g. “GO ALL IN”)

---

### ✅ Key Improvements

#### 1. **Score Architecture Refactor**

Scores are now built from distinct components:

- **Trend (0–4)** – structure via SMA20 > SMA50
- **Momentum (−3 to +3)** – MACD + RSI direction with decay detection
- **Timing (−2 to +2)** – distance from SMA20 (pullbacks favored)
- **Risk Governors** – hard penalties for overextension

This prevents score saturation and forces trade-offs.

---

#### 2. **Overextension Penalties Added**

Late-stage momentum is now explicitly penalized:

- RSI > 70 → negative drift
- RSI > 75 + CCI > 180 → strong penalty
- RSI > 80 → hard cap on total score
- Price > SMA20 by >12% → timing penalty

Strength ≠ entry.

---

#### 3. **CCI Logic Corrected**

CCI is no longer treated as a directional indicator:

- −100 to +100 → constructive zone
- > +200 → exhaustion penalty
- < −200 → rebound potential (small bonus)

This restores its mean-reversion intent.

---

#### 4. **Momentum Persistence Requirement**

High scores now require **confirmation durability**:

- Scores ≥ 8 must have MACD > 0 and RSI > 50
- Prevents premature “top of first candle” signals

---

#### 5. **Action Labels Replaced**

Binary conviction removed. Actions now reflect position sizing and patience:

| Score | Action                      |
| ----- | --------------------------- |
| 9+    | Starter + add on pullback   |
| 7–8   | Starter position            |
| 5–6   | Watch for pullback          |
| 2–4   | Wait / No edge              |
| ≤ −2  | Reduce / Avoid / Short bias |

---

### 📈 Result

- Fewer false positives
- Clean score distribution (no mass 9–10s)
- Overbought assets correctly downgraded
- Scanner ranks **opportunity**, not hype

This version is intentionally quieter — and materially more tradable.
