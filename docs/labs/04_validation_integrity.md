# Lab 04 — Leakage & Validation Integrity

**Objective.** Show why random $K$-fold overstates skill on Coiled Cobra
trades, then compare it to the production temporal split and a minimal
**purged** embargo.

Purged walk-forward is specified in
[`QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md) §4 and is **not**
implemented in the trainer. This lab is the implementation exercise.

## Why this experiment

`Forward_Return_2w` for a signal at $t$ uses prices through $t+2$ (or $t+13$
if you ran Lab 02). A random fold can place a June coil in train and a May
coil in val on the same ticker — the "future" return already overlaps the
other row's feature window. Market-factor contemporaneous leakage (every
name that week) remains even with a perfect per-ticker embargo.

## Starter pointers

| What | Where |
| ---- | ----- |
| Approved split | `_temporal_split()` — last 26w test, prior 26w val |
| Isolation | `LEAKAGE_COLS`, `DATE_COL = "Signal Date"` |
| Causal backtest | `coiled_cobra_backtest.py` evaluates `df.iloc[:i+1]` |
| Swing analogue | `pipeline_backtest.py` (warmup + detect-at-bar) |
| Forbidden | `sklearn.model_selection.KFold` / `train_test_split(shuffle=True)` on this table |

## Prerequisites

Trades CSV only. You will write a **sidecar** comparison script; do not
replace `_temporal_split()` in production during the lab.

## Exercise A — leaking $K$-fold

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from finance_vibe.coiled_cobra_ml_training import (
    FEATURE_COLS, TARGET_COL, MODEL_PARAMS, LEAKAGE_COLS,
    _resolve_source_csv,
)

df = pd.read_csv(_resolve_source_csv(
    "data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv"
))
df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
df = df[df[TARGET_COL].notna()].copy()
X = df[FEATURE_COLS]
y = df[TARGET_COL].astype(float).to_numpy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # leaking on purpose
maes = []
for tr, va in kf.split(X):
    m = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                     random_state=42, n_jobs=-1, **MODEL_PARAMS)
    m.fit(X.iloc[tr], y[tr])
    maes.append(mean_absolute_error(y[va], m.predict(X.iloc[va])))
print("leaking_kfold_mae_mean", float(np.mean(maes)), "std", float(np.std(maes)))
```

## Exercise B — production temporal split

```python
from finance_vibe.coiled_cobra_ml_training import (
    _load_and_prepare, _temporal_split, _build_matrices,
)

frame = _load_and_prepare(_resolve_source_csv(
    "data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv"
))
train, val, test, bounds = _temporal_split(frame)
parts = _build_matrices(train, val, test)
m = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                 random_state=42, n_jobs=-1, **MODEL_PARAMS)
m.fit(parts["train"]["X"], parts["train"]["y"],
      sample_weight=parts["train"]["w"])
print("temporal_val_mae", mean_absolute_error(
    parts["val"]["y"], m.predict(parts["val"]["X"])))
print("temporal_test_mae", mean_absolute_error(
    parts["test"]["y"], m.predict(parts["test"]["X"])))
print(bounds)
```

Expect **higher** (worse) temporal MAE than shuffled K-fold if leakage was
helping. If they are close, the feature set may simply be weak — still
prefer the honest split.

## Exercise C — purged embargo (minimal)

Let $h = 2$ weekly bars (or 13 if you changed `TARGET_COL`). After the
temporal cut, drop train rows whose label window overlaps val:

```python
h = 2  # keep in sync with TARGET_HORIZON_WEEKS
embargo = pd.Timedelta(weeks=h)
cut = bounds["val_start"]
purged_train = train[train["Signal Date"] < (cut - embargo)].copy()
print("train_rows", len(train), "purged_train_rows", len(purged_train))
```

Refit on `purged_train` and re-score val. The MAE gap vs un-purged temporal
is the overlap tax. A fuller walk-forward would repeat this on several
expanding cuts; that is extra credit, not required.

## What to record

| Protocol | Train rows | Val MAE | Test MAE | Notes |
| -------- | ---------- | ------- | -------- | ----- |
| Shuffled 5-fold (mean) | ~80% / fold | | n/a | Leaking |
| Temporal 26w / 26w | | | | Production |
| Temporal + $h$-week purge | | | | Lab purge |

## Expected failure modes

- Shuffling *after* `_temporal_split()` — still leaks inside a window if you
  then K-fold the train piece and report that as OOS.
- Purging by row count instead of calendar time — weekly bars are the unit
  of $h$, not iid index positions across tickers.
- Using `Target_Label` as $Y$ while claiming the split is "purged" — the
  label itself is a future execution outcome.

## Restore

Sidecar only. Production `_temporal_split()` stays the contract unless you
later upstream a real purged CV.
