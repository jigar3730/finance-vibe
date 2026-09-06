# Finance Vibe documentation

This tree is the reference manual and lab environment for the Finance Vibe
pipeline: macro scoring, tactical swing detection, Coiled Cobra coil scoring,
walk-forward backtests, and the offline GBDT ranking baseline.

Project overview and run commands stay in the root [`README.md`](../README.md).

## How to use this catalog

| If you want to… | Start here |
| ---------------- | ---------- |
| Understand *why* an indicator or target is constructed | [`handbook/QUANT_ML_MANUAL.md`](handbook/QUANT_ML_MANUAL.md) |
| Look up the exact scoring rule that shipped | Handbook rubrics below |
| Operate, backtest, or train the ML baseline | Architecture guides below |
| Run a structured experiment | [`labs/README.md`](labs/README.md) |

---

## Handbook — concepts and finance reference

| File | Contents |
| ---- | -------- |
| [`handbook/QUANT_ML_MANUAL.md`](handbook/QUANT_ML_MANUAL.md) | Feature engineering, targets, GBDTs, validation, SHAP / Optuna — mapped to this repo |
| [`handbook/scoring_logic.md`](handbook/scoring_logic.md) | Macro Vibe Score (−10 to +10) specification |
| [`handbook/ta_interpretation.md`](handbook/ta_interpretation.md) | How to read SMA / RSI / CCI / MACD output |
| [`handbook/swing_setup.md`](handbook/swing_setup.md) | Quality-swing long/short rules and geometry |
| [`handbook/coiled_cobra_rubric.md`](handbook/coiled_cobra_rubric.md) | Coil → expansion 100-point scorecard |
| [`handbook/trade_plan_calculations.md`](handbook/trade_plan_calculations.md) | Early ATR entry/stop/target illustration |

---

## Architecture — system design and operations

| File | Contents |
| ---- | -------- |
| [`architecture/operation_manual.md`](architecture/operation_manual.md) | SOP, environment, troubleshooting |
| [`architecture/backtest_and_backfill.md`](architecture/backtest_and_backfill.md) | Data backfill, signal archives, walk-forward sims |
| [`architecture/coiled_cobra_ml.md`](architecture/coiled_cobra_ml.md) | ML baseline: features, leakage isolation, temporal split |
| [`architecture/trade_planner_worklog.md`](architecture/trade_planner_worklog.md) | Planner formulas and implementation notes |
| [`architecture/code_review.md`](architecture/code_review.md) | Historical architecture review (2026-07-12 snapshot) |
| [`architecture/planned_enhancements.md`](architecture/planned_enhancements.md) | LEAPS / IV-rank critique |
| [`architecture/project_resurrection_prompt.md`](architecture/project_resurrection_prompt.md) | Session context prompt for future work |

---

## Labs — hands-on experiments

Seven guided experiments live in [`labs/`](labs/). Index and prerequisites:
[`labs/README.md`](labs/README.md).

| Lab | File |
| --- | ---- |
| 01 Indicator sensitivity | [`labs/01_indicator_sensitivity.md`](labs/01_indicator_sensitivity.md) |
| 02 Target horizon shift | [`labs/02_target_horizon_shift.md`](labs/02_target_horizon_shift.md) |
| 03 GBDT showdown | [`labs/03_gbdt_showdown.md`](labs/03_gbdt_showdown.md) |
| 04 Leakage & validation | [`labs/04_validation_integrity.md`](labs/04_validation_integrity.md) |
| 05 Optuna HPO | [`labs/05_hyperparameter_optuna.md`](labs/05_hyperparameter_optuna.md) |
| 06 SHAP explainability | [`labs/06_shap_explainability.md`](labs/06_shap_explainability.md) |
| 07 Market regime shifts | [`labs/07_regime_shifts.md`](labs/07_regime_shifts.md) |

`catboost`, `optuna`, and `shap` are **lab-optional**. They are not in
`requirements.txt`. Each lab that needs one of them includes an install line
and a fallback that uses only the current stack (`xgboost`, `lightgbm`,
`scikit-learn`, `matplotlib`).

---

## Code of record (quick map)

| Concern | Module |
| ------- | ------ |
| Orchestrator | `src/finance_vibe/run_vibe.py` |
| Macro score | `src/finance_vibe/analysis_engine.py` |
| Quality swing | `src/finance_vibe/swing_scanner.py` |
| Coiled Cobra rubric | `src/finance_vibe/coiled_cobra.py` |
| Cobra backtest + forward returns | `src/finance_vibe/coiled_cobra_backtest.py` |
| GBDT trainer | `src/finance_vibe/coiled_cobra_ml_training.py` |
| Soft ML ranking | `src/finance_vibe/ml_ranker.py` |
| Swing walk-forward | `src/finance_vibe/pipeline_backtest.py` |
