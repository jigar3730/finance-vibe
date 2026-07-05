# Macro Vibe Score — Scoring Logic

This document describes the **implemented** scoring rules in [`analysis_engine.py`](analysis_engine.py). The score ranks macro tradability on a **−10 to +10** integer scale. It favors trend structure and pullback timing over raw momentum.

For tactical entry rules (EMA pullbacks), see [`swing_scanner.py`](swing_scanner.py) and [`swing_setup_readme.md`](../../swing_setup_readme.md).

---

## Role in the pipeline

| | |
| --- | --- |
| Module | `src/finance_vibe/analysis_engine.py` |
| Pipeline step | **3** in `run_vibe.py` (after ingestion, before swing scanner) |
| Input | All CSV files in `data/raw/{mode}/` |
| Output | `data/logs/{mode}/vibe_report_<YYYY-MM-DD>.csv` |

```bash
python src/finance_vibe/analysis_engine.py weekly
python src/finance_vibe/run_vibe.py
```

Scoring uses the **latest bar** of each ticker file. Minimum history: **60 rows** (`MIN_ROWS`); shorter files are skipped.

---

## Indicators (`build_features`)

Computed from `Close` (and `High` / `Low` when present for CCI):

| Column | Definition |
| ------ | ---------- |
| **SMA20 / SMA50** | Simple moving average of `Close` (20 / 50 periods, `min_periods` = window) |
| **MACD_H** | MACD histogram: EMA(12) − EMA(26) of `Close`, minus 9-period EMA of that line |
| **MACD_S** | 9-period EMA of `MACD_H` |
| **RSI** | 14-period Wilder RSI on `Close` |
| **RSI_S** | 10-period SMA of `RSI` |
| **CCI** | 20-period CCI on typical price using **mean absolute deviation** (constant 0.015) |
| **CCI_S** | 10-period SMA of `CCI` |

If `High` / `Low` are missing, CCI uses `Close` as typical price.

Implementation note: CCI uses a vectorized MAD window (not pandas_ta), which avoids unstable values on weekly bars.

---

## Score components (additive, applied in order)

Each rule reads the **latest bar** only. Logic lives in `_compute_score()`; keys below appear in `calculate_vibe_score(..., return_components=True)`.

### 1. Trend — key `Trend` (±4)

| Condition | Points |
| --------- | ------ |
| `Close > SMA20 > SMA50` | +4 |
| `Close < SMA20 < SMA50` | −4 |
| Otherwise | 0 |

Partial trend credit (e.g. above SMA20 only) is **not** awarded.

### 2. Momentum — key `Momentum` (±2)

| Condition | Points |
| --------- | ------ |
| `MACD_H > MACD_S` **and** `RSI > RSI_S` | +2 |
| `MACD_H < MACD_S` **and** `RSI < RSI_S` | −2 |
| Otherwise | 0 |

### 3. Momentum decay — key `MomentumDecay` (−1)

| Condition | Points |
| --------- | ------ |
| `MACD_H < MACD_S` **and** `Close > SMA20` | −1 |
| Otherwise | 0 |

Penalizes bullish structure with weakening momentum.

### 4. Timing — key `Timing` (−2 to +2)

Let `dist = (Close − SMA20) / SMA20`.

| Condition | Points |
| --------- | ------ |
| `0.0 ≤ dist ≤ 0.05` (pullback into SMA20) | +2 |
| `dist > 0.12` (overextended above SMA20) | −2 |
| `dist < −0.05` (meaningfully below SMA20) | −1 |
| Otherwise | 0 |

### 5. CCI — key `CCI` (−2 to +1)

| Condition | Points |
| --------- | ------ |
| `−100 < CCI < 100` **and** `CCI > CCI_S` | +1 |
| `CCI > 200` (exhaustion) | −2 |
| `CCI < −200` (deep cyclical low) | +1 |
| Otherwise | 0 |

Boundaries are **strict** (`CCI = ±100` does not qualify for the constructive +1 band). CCI is not used as a raw momentum booster outside these rules.

### 6. RSI risk — key `RSI_Risk` (variable)

Applied **after** steps 1–5:

| Condition | Effect |
| --------- | ------ |
| `RSI > 80` | Hard cap: `score = min(score, 5)` (component records the adjustment applied) |
| `70 < RSI ≤ 80` | −1 |
| `RSI < 30` | +1 |
| Otherwise | 0 |

### 7. Persistence — key `Persistence` (−2)

If score is **≥ 7** after steps 1–6, require `MACD_H > 0` **and** `RSI > 50`. If not met, **−2**.

### 8. Final clip

Score is clipped to **\[−10, +10\]** and stored as an integer.

---

## Theoretical range (before clip)

| Component | Min | Max |
| --------- | --- | --- |
| Trend | −4 | +4 |
| Momentum | −2 | +2 |
| Momentum decay | −1 | 0 |
| Timing | −2 | +2 |
| CCI | −2 | +1 |
| RSI risk | unbounded cap | +1 |
| Persistence | −2 | 0 |

RSI > 80 cap and persistence check prevent most scores from clustering at the top end.

---

## Sentiment and action labels

Mapped by `sentiment_action()` — exact strings written to the CSV **Action** column:

| Score | Sentiment | Action |
| ----- | --------- | ------ |
| ≥ 9 | Bullish | 🟢 STARTER + ADD ON PULLBACK |
| 7 – 8 | Bullish | 🟢 STARTER POSITION |
| 5 – 6 | Positive | 📈 WATCH / SCALE IN |
| 2 – 4 | Neutral | ⏳ WAIT |
| −1 – 1 | Neutral | 💤 NO EDGE |
| −4 – −2 | Bearish | 🟠 REDUCE / HEDGE |
| −10 – −5 | Bearish | 🔴 AVOID / SHORT BIAS |

---

## Output file

| | |
| --- | --- |
| Path | `data/logs/{mode}/vibe_report_<YYYY-MM-DD>.csv` |
| Columns | Ticker, Price, SMA20, SMA50, CCI, CCI_S, MACD_H, MACD_S, RSI, RSI_S, Score, Sentiment, Action |
| Sort | Score descending, then Ticker ascending |

Universe scan runs in parallel (`ProcessPoolExecutor`); tickers that fail to load or score are skipped with a count at the end.

---

## Programmatic API

```python
from finance_vibe.analysis_engine import build_features, score_last_row, calculate_vibe_score

score = score_last_row(feat.iloc[-1])
detail = calculate_vibe_score("SPY", df, return_components=True)
# detail["Score"], detail["Components"]["Trend"], ...
```

Used by `pipeline_backtest.py` for walk-forward macro gating.

---

## Design intent

- Penalize **overextension** (RSI cap, distance from SMA20, CCI > 200).
- Reward **pullbacks into trend** (timing band near SMA20).
- Use action labels that imply **position sizing**, not binary conviction.
- Keep macro (SMA-based) separate from tactical (EMA-based swing scanner).

---

## Backtest macro gate (offline only)

The live pipeline does **not** filter swing setups by Vibe Score. The offline backtest in `pipeline_backtest.py` adds this gate:

| Setup | Required score | Default threshold | Action band alignment |
| ----- | -------------- | ----------------- | --------------------- |
| `SETUP_LONG` | ≥ threshold | `BACKTEST_LONG_MIN_SCORE` = **7** | STARTER POSITION or higher |
| `SETUP_SHORT` | ≤ threshold | `BACKTEST_SHORT_MAX_SCORE` = **−2** | REDUCE / HEDGE or lower |

Override via CLI: `--long-min-score`, `--short-max-score`.

See [`config.py`](config.py) for `BACKTEST_*` constants.
