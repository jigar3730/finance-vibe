# Hands-on labs

Structured experiments against the Coiled Cobra ML baseline. Theory lives in
[`../handbook/QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md). The trainer
contract lives in [`../architecture/coiled_cobra_ml.md`](../architecture/coiled_cobra_ml.md).

Work **locally on a branch or a copy of the constants** — do not commit
ablations unless you intend to change the production baseline.

## Prerequisites (all labs)

1. Python 3.10+ with `PYTHONPATH=src` and `requirements.txt` installed
   (`xgboost`, `lightgbm`, `scikit-learn`, `matplotlib`, `pandas`, `numpy`).
2. A Coiled Cobra trades CSV:

   ```bash
   export PYTHONPATH=src
   python src/finance_vibe/data_ingestor.py weekly
   python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
   ```

   Output: `data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv`.
3. Ability to run:

   ```bash
   python src/finance_vibe/coiled_cobra_ml_training.py \
     --csv data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv \
     --artifacts-dir data/logs/weekly
   ```

Docker: `export PYTHONPATH=/app/src` inside the `finance_vibe` container and
use `/app/data/logs/weekly/`.

## Optional libraries (not in `requirements.txt`)

| Lab | Extra package | Install (lab-only) |
| --- | ------------- | ------------------ |
| 03 | CatBoost | `python -m pip install catboost` |
| 05 | Optuna | `python -m pip install optuna` |
| 06 | SHAP | `python -m pip install shap` |

Each of those labs has a fallback that uses only the current stack.

## Lab sequence

| # | File | Depends on a trained baseline? |
| - | ---- | ------------------------------ |
| 01 | [`01_indicator_sensitivity.md`](01_indicator_sensitivity.md) | Yes (retrain after each ablation) |
| 02 | [`02_target_horizon_shift.md`](02_target_horizon_shift.md) | Yes (retrain on a new `TARGET_COL`) |
| 03 | [`03_gbdt_showdown.md`](03_gbdt_showdown.md) | Trainer + optional CatBoost |
| 04 | [`04_validation_integrity.md`](04_validation_integrity.md) | CSV only (comparison script) |
| 05 | [`05_hyperparameter_optuna.md`](05_hyperparameter_optuna.md) | Trainer + optional Optuna |
| 06 | [`06_shap_explainability.md`](06_shap_explainability.md) | Saved `coiled_cobra_xgb_model.json` |
| 07 | [`07_regime_shifts.md`](07_regime_shifts.md) | Trained model + `ATR_Pct` slices |

Suggested order: **01 → 02 → 04 → 03 → 05 → 06 → 07**.

## Shared constants (do not leak these)

From `src/finance_vibe/coiled_cobra_ml_training.py`:

- `FEATURE_COLS` — only these may enter $X$
- `LEAKAGE_COLS` — never enter $X$
- `TARGET_COL` — default `Forward_Return_2w`
- `_temporal_split()` — the only approved split for scoring trials
