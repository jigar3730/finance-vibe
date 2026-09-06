# Lab 06 — Model Explainability with SHAP

**Objective.** Explain XGBoost predictions on the 6-feature Coiled Cobra
matrix with SHAP summary and dependence plots. Contrast SHAP with the
gain-importance PNG the trainer already writes.

SHAP is **not** in `requirements.txt`.

## Why this experiment

Gain importance answers "which columns did the trees split on?" SHAP answers
"for this row, how much did each feature push $\hat{y}$ away from the
background expectation?" With collinear EMA/Fib distances, those two stories
diverge — see [`QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md) §5.

## Starter pointers

| What | Where |
| ---- | ----- |
| Saved booster | `coiled_cobra_xgb_model.json` (written by `_train_and_report`) |
| Feature order | `FEATURE_COLS` — must match the JSON |
| Gain chart (baseline) | `coiled_cobra_ml_feature_importance.png` |
| Soft inference | `ml_ranker.predict_returns` / `attach_ml_ranks` |
| Metadata | `coiled_cobra_ml_model_metadata.json` |

## Prerequisites

Train once so artifacts exist:

```bash
export PYTHONPATH=src
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv \
  --artifacts-dir data/logs/weekly/lab06
```

### Optional install

```bash
python -m pip install shap
```

### Fallback without SHAP

Use the ASCII / PNG gain charts plus a manual ICE-style slice: pick
`Pct_From_Fib786`, hold other columns at their train median, vary that
column on a grid, and plot `model.predict`. You will see marginal response
but not interactions.

## Exercise A — load the production booster

```python
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from finance_vibe.coiled_cobra_ml_training import (
    FEATURE_COLS, _resolve_source_csv, _load_and_prepare, _temporal_split,
)

art = Path("data/logs/weekly/lab06")
model = XGBRegressor()
model.load_model(str(art / "coiled_cobra_xgb_model.json"))

df = _load_and_prepare(_resolve_source_csv(
    "data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv"
))
_, _, test, _ = _temporal_split(df)
X_test = test[FEATURE_COLS]
```

Confirm `list(X_test.columns) == FEATURE_COLS`.

## Exercise B — summary plot

```python
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
# Background: a sample of train rows is enough at this scale
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(art / "shap_summary.png", dpi=140, bbox_inches="tight")
plt.close()
```

Read: color = feature value, x = SHAP (impact on predicted forward return).
Ask whether high `ATR_Pct` systematically increases or decreases $\hat{y}$.
Sample-weighting by `ATR_Pct` during **fit** does not force a positive SHAP
sign at inference.

## Exercise C — dependence / interaction

```python
shap.dependence_plot(
    "Pct_From_Fib786",
    shap_values,
    X_test,
    interaction_index="ATR_Pct",
    show=False,
)
plt.savefig(art / "shap_dep_fib786_atr.png", dpi=140, bbox_inches="tight")
plt.close()
```

Repeat for `Score` vs `Pct_From_EMA20`. If `Score` SHAP is flat after
conditioning on the distances, the rubric number is mostly redundant with
geometry (Lab 01).

## Exercise D — one setup, human-readable

Pick a high-`Score` test row and print `explainer.shap_values(row)` next to
the feature values. Compare that story to `ml_ranker`’s `ML_Rank` if you
attach ranks on the same frame. SHAP explains the booster; it does not
override the rubric gates.

## What to record

| Artifact | Path | What you concluded |
| -------- | ---- | ------------------ |
| Gain PNG | `lab06/coiled_cobra_ml_feature_importance.png` | |
| SHAP summary | `lab06/shap_summary.png` | |
| Dependence | `lab06/shap_dep_fib786_atr.png` | |
| One-row attribution | notes | |

## Expected failure modes

- Explaining a model trained on 2w $Y$ as if it were a 13w expansion
  forecast (Lab 02).
- Passing leakage columns into `X_test` "just to see" — the booster was
  not trained on them; you will hit a shape error or a meaningless plot.
- Treating SHAP as causal ("increase Fib distance to raise return").

## Restore

No production edits required. Do **not** add `shap` to `requirements.txt`.
