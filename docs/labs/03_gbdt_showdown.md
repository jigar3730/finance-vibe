# Lab 03 — GBDT Showdown

**Objective.** Benchmark XGBoost vs LightGBM vs (optional) CatBoost on the
same Coiled Cobra matrix: training speed, Val/Test MAE & RMSE, and how each
library treats a categorical (`Grade`).

## Why this experiment

The production trainer already fits `XGBRegressor` and `LGBMRegressor` with
matched MAE/L1 objectives and the same `MODEL_PARAMS`. CatBoost is **not**
in `requirements.txt`. This lab asks whether the ranking is library-specific
and whether native categoricals change anything if you put `Grade` back.

## Starter pointers

| What | Where |
| ---- | ----- |
| Dual-model fit | `_train_and_report()` in `coiled_cobra_ml_training.py` |
| Shared hyperparameters | `MODEL_PARAMS` (`max_depth`, `learning_rate`, `n_estimators`, bagging) |
| Split / matrices | `_temporal_split()`, `_build_matrices()` |
| Why `Grade` is out | [`coiled_cobra_ml.md`](../architecture/coiled_cobra_ml.md) § Feature space |
| Inference | `ml_ranker.py` (XGB JSON + LGB text only) |

## Prerequisites

Trades CSV + `PYTHONPATH=src`. No extra packages required for XGB vs LGB.

### Optional CatBoost

```bash
python -m pip install catboost
```

Skip Exercise C if you do not install it. The XGB vs LGB comparison still
satisfies the lab.

## Exercise A — production pair (already implemented)

```bash
export PYTHONPATH=src
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv \
  --artifacts-dir data/logs/weekly/lab03_baseline
```

Time the process (`time …` or note wall-clock). Record Val/Test MAE, RMSE,
and the two ASCII importance orders.

## Exercise B — fair timing harness

The trainer does not print fit seconds. A small sidecar script (do not
replace production) using the same helpers:

```python
import time
from pathlib import Path
from finance_vibe.coiled_cobra_ml_training import (
    FEATURE_COLS, TARGET_COL, MODEL_PARAMS,
    _resolve_source_csv, _load_and_prepare, _temporal_split, _build_matrices,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

df = _load_and_prepare(_resolve_source_csv(
    "data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv"
))
train, val, test, _ = _temporal_split(df)
parts = _build_matrices(train, val, test)
X, y, w = parts["train"]["X"], parts["train"]["y"], parts["train"]["w"]

def fit_timer(model):
    t0 = time.perf_counter()
    model.fit(X, y, sample_weight=w)
    return time.perf_counter() - t0

xgb = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                   random_state=42, n_jobs=-1, **MODEL_PARAMS)
lgb = LGBMRegressor(objective="regression_l1", random_state=42,
                    n_jobs=-1, verbose=-1, **MODEL_PARAMS)
print("xgb_sec", round(fit_timer(xgb), 3), "lgb_sec", round(fit_timer(lgb), 3))
```

LightGBM is typically faster at this scale (a few thousand rows × 6 cols).
Do not treat a 0.2s gap as a production decision.

## Exercise C — optional CatBoost

```python
from catboost import CatBoostRegressor, Pool

# Numeric-only (apples-to-apples with FEATURE_COLS)
cat = CatBoostRegressor(
    loss_function="MAE",
    depth=MODEL_PARAMS["max_depth"],
    learning_rate=MODEL_PARAMS["learning_rate"],
    iterations=MODEL_PARAMS["n_estimators"],
    subsample=MODEL_PARAMS["subsample"],
    random_seed=42,
    verbose=False,
)
t0 = time.perf_counter()
cat.fit(X, y, sample_weight=w)
print("cat_sec", round(time.perf_counter() - t0, 3))
```

Then evaluate MAE/RMSE on `parts["val"]` and `parts["test"]` the same way
`_evaluate()` does.

**Categorical variant:** add `Grade` (A/B) back as a CatBoost categorical
(`cat_features=["Grade"]`) and compare to one-hot / integer-coded XGBoost.
Expect little lift — `Grade` is a bin of `Score`. If CatBoost "wins" only
in this variant, you are measuring encoding, not coil alpha.

## What to record

| Model | Fit seconds | Val MAE | Val RMSE | Test MAE | Test RMSE | Notes |
| ----- | ----------- | ------- | -------- | -------- | --------- | ----- |
| XGBoost MAE | | | | | | Production |
| LightGBM L1 | | | | | | Production |
| CatBoost MAE (optional) | | | | | | Numeric $X$ |
| CatBoost + Grade (optional) | | | | | | Native categorical |

## Expected failure modes

- Comparing `reg:squarederror` XGB to L1 LGB — change the objective, not
  just the library.
- Installing CatBoost into the image and assuming `ml_ranker` can load it.
  Inference only knows `coiled_cobra_xgb_model.json` and
  `coiled_cobra_lgb_model.txt`.
- Treating `Grade` as numeric 0/1 in XGB while CatBoost gets a true
  category — that is not a fair showdown.

## Restore

No production files need to change if you used a sidecar script. Do **not**
add `catboost` to `requirements.txt` from this lab.
