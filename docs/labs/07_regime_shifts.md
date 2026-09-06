# Lab 07 — Market Regime Shifts

**Objective.** Measure whether the frozen Coiled Cobra booster stays useful
when volatility regimes change, and sketch a rolling-retrain alternative to
the single 26-week holdout.

## Why this experiment

The trainer has **one** trailing test window (`_temporal_split`). High-beta
swing logic already knows about regimes — QQQ above rising EMA50/100 and
63-day relative strength — but those flags never enter `FEATURE_COLS`. The
only vol channel in the ML table is `ATR_Pct` (feature **and** sample
weight). A model that looks fine on a quiet holdout can fail in a high-vol
slice of the same dates.

## Starter pointers

| What | Where |
| ---- | ----- |
| Vol feature / weight | `ATR_Pct`, `WEIGHT_COL` |
| Temporal cut | `_temporal_split()` |
| QQQ regime helpers | `analysis_engine.market_regime_ok`, `relative_strength` |
| High-beta profile | `config.get_swing_params("high_beta")`, `pipeline_backtest.py` |
| Cobra RS pillar | `coiled_cobra.rs_score` (not in $X$) |
| Manual §6 | [`QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md) |

## Prerequisites

A trained XGB artifact (Lab 06 folder is fine) and the trades CSV used to
train it. Same `FEATURE_COLS` / `TARGET_COL`.

## Exercise A — slice the official test set by vol

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from finance_vibe.coiled_cobra_ml_training import (
    FEATURE_COLS, TARGET_COL, _resolve_source_csv, _load_and_prepare,
    _temporal_split,
)

df = _load_and_prepare(_resolve_source_csv(
    "data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv"
))
train, val, test, bounds = _temporal_split(df)
model = XGBRegressor()
model.load_model("data/logs/weekly/lab06/coiled_cobra_xgb_model.json")

pred = model.predict(test[FEATURE_COLS])
y = test[TARGET_COL].astype(float).to_numpy()
test = test.copy()
test["pred"] = pred
test["abs_err"] = np.abs(y - pred)

q = test["ATR_Pct"].quantile([0.33, 0.67])
bins = [-np.inf, q.iloc[0], q.iloc[1], np.inf]
test["vol_bucket"] = pd.cut(test["ATR_Pct"], bins, labels=["low", "mid", "high"])
print(test.groupby("vol_bucket", observed=False).apply(
    lambda g: pd.Series({
        "n": len(g),
        "mae": g["abs_err"].mean(),
        "mean_y": g[TARGET_COL].mean(),
        "std_y": g[TARGET_COL].std(),
        "mae_over_std": g["abs_err"].mean() / (g[TARGET_COL].std() or np.nan),
    })
))
print("overall_test_mae", mean_absolute_error(y, pred), "bounds", bounds)
```

If high-vol MAE / std($Y$) is much worse than low-vol, the booster is not
regime-robust even though `ATR_Pct` is in $X$.

## Exercise B — calendar / stress windows

Using the **full** prepared frame (not just the last 26w), score a model
trained on data **before** a cut on data **after** it. Example cuts to try
if your CSV covers them:

- 2020-02-01 → 2020-06-30 (vol shock)
- 2022-01-01 → 2022-12-31 (tightening / bear)
- 2023-01-01 → 2023-12-31 (narrow-leadership rally)

```python
cut = pd.Timestamp("2022-01-01")
pre, post = df[df["Signal Date"] < cut], df[df["Signal Date"] >= cut]
# fit on pre[FEATURE_COLS], evaluate MAE on post — only if both sides are large
```

Skip a window if either side has fewer than ~150 rows. Survivorship in
`active_tickers.csv` already biases older history.

## Exercise C — rolling retrain vs freeze

Approximate an expanding walk-forward **without** writing a new module:

1. Sort unique weeks in `Signal Date`.
2. For each year-end (or every 26 weeks): train on all earlier rows, score
   the next 26 weeks.
3. Compare the **mean** of those fold MAEs to the single frozen model from
   Exercise A scored on the union of those 26-week blocks.

If rolling retrain wins by a lot, the current one-shot `_temporal_split`
is understating maintenance cost. If it ties, a quarterly refit is enough.

Do not tune `MODEL_PARAMS` on these folds in the same sitting as Lab 05
(double-dipping). Use the frozen defaults.

## Exercise D — regime feature (research only)

Optionally merge a QQQ regime flag onto each `Signal Date` via
`load_benchmark_frame("QQQ", "weekly")` + `market_regime_ok`. Adding it to
`FEATURE_COLS` is a Lab 01-style schema change and requires `ml_ranker`
updates if you keep it. For this lab it is enough to **stratify** Test MAE
by `regime_ok` True/False, the same way you stratified by `ATR_Pct`.

## What to record

| Slice | n | MAE | MAE / std($Y$) | Notes |
| ----- | - | --- | -------------- | ----- |
| Official test (all) | | | | Frozen model |
| Test × low / mid / high ATR | | | | Exercise A |
| Stress window(s) | | | | Exercise B |
| Mean rolling-retrain MAE | | | | Exercise C |

## Expected failure modes

- Training and testing on the PLTR/TSLA/HOOD high_beta **tuning** basket and
  calling it regime robustness. Those names are a promotion holdout for the
  *swing* profile, not an ML regime test.
- Comparing raw MAE across 2w vs 13w targets (Lab 02) inside a vol slice.
- Forgetting that `sample_weight=ATR_Pct` already up-weights noisy rows —
  high-vol MAE can look bad even when rank correlation is acceptable.
  Optionally also report Spearman($Y$, $\hat{y}$) per bucket.

## Restore

No production files required. A real rolling trainer would be a new module;
do not silently replace `_temporal_split()` from this lab.
