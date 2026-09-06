# Lab 05 — Hyperparameter Optimization with Optuna

**Objective.** Run an Optuna study on XGBoost using the **time-series**
split (or the Lab 04 purge), never shuffled K-fold.

Optuna is **not** in `requirements.txt`.

## Why this experiment

`MODEL_PARAMS` is a conservative default (`max_depth=4`, `learning_rate=0.01`,
`n_estimators=400`, 80% row/column bagging). It is not the result of a search.
A naive `GridSearchCV` with `KFold` would optimize leaked folds (Lab 04).

## Starter pointers

| What | Where |
| ---- | ----- |
| Frozen defaults | `MODEL_PARAMS` in `coiled_cobra_ml_training.py` |
| Fit site | `_train_and_report()` — no early stopping, no search |
| Val partition to score | `_temporal_split()` → `parts["val"]` |
| Theory | [`QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md) §5.3 |

## Optional install

```bash
python -m pip install optuna
```

### Fallback without Optuna

Use a 6–8 point manual grid over `max_depth ∈ {3,4,6}` and
`learning_rate ∈ {0.01, 0.03}` with the same temporal val MAE. The scientific
content of the lab is **the split**, not the optimizer.

## Study design (required rules)

1. Load via `_load_and_prepare` + `_temporal_split` + `_build_matrices`.
2. Each trial fits **only** on train (+ train `sample_weight`).
3. Objective = Val MAE (minimize). Do **not** peek at Test during the study.
4. After you pick a winner, score Test **once**.
5. Cap trials (20–40) — this is a 6-feature weekly table, not a Kaggle race.

## Starter study

```python
import optuna
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from finance_vibe.coiled_cobra_ml_training import (
    MODEL_PARAMS, _resolve_source_csv, _load_and_prepare,
    _temporal_split, _build_matrices,
)

df = _load_and_prepare(_resolve_source_csv(
    "data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv"
))
train, val, test, _ = _temporal_split(df)
parts = _build_matrices(train, val, test)
Xtr, ytr, wtr = parts["train"]["X"], parts["train"]["y"], parts["train"]["w"]
Xva, yva = parts["val"]["X"], parts["val"]["y"]

def objective(trial: optuna.Trial) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
    }
    model = XGBRegressor(
        objective="reg:absoluteerror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    model.fit(Xtr, ytr, sample_weight=wtr)
    return float(mean_absolute_error(yva, model.predict(Xva)))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print(study.best_params, study.best_value)
```

Refit the best params on train, then report Test MAE **once**. Compare to
the frozen `MODEL_PARAMS` baseline from a normal trainer run.

## Extra credit

- Repeat the study with the Lab 04 purged train set. If best params jump,
  the search was fitting overlap, not coil structure.
- LightGBM `LGBMRegressor` with the same search space (`num_leaves` instead
  of — or in addition to — `max_depth`).

## What to record

| Config | Val MAE | Test MAE (one shot) | `max_depth` | `learning_rate` | `n_estimators` |
| ------ | ------- | ------------------- | ----------- | --------------- | -------------- |
| `MODEL_PARAMS` baseline | | | 4 | 0.01 | 400 |
| Optuna best (temporal) | | | | | |
| Optuna best (purged, optional) | | | | | |

## Expected failure modes

- Using `cross_val_score` with default K-fold inside the objective —
  this undoes Lab 04.
- Early-stopping on **test** or reporting the best test trial (selection
  bias). Test is one-shot after the study.
- Promoting best params into `MODEL_PARAMS` without a new time window.
  A search that saw 2024–2026 val is not automatically the next-year prior.

## Restore

Do **not** add `optuna` to `requirements.txt`. Only change `MODEL_PARAMS`
if you have an out-of-sample window the study never scored.
