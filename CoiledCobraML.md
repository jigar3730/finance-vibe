# Coiled Cobra ML Baseline

**Resume after a break:** **`docs/CoiledCobraML-Handoff.md`**. Audit: **`CodeReview.MD`**. Docker runbook: **`MLOps.md`**. This file is the **feature / label / artifact contract**.

**Module:** `src/finance_vibe/coiled_cobra_ml_training.py`

Standalone daily trainer for **three independent** XGBoost classifiers:

| Horizon | Train label | Live column (if promoted) | Embargo |
| ------- | ----------- | ------------------------- | ------- |
| 10d | `Win_10d` (`Forward_Return_10d > 0`) | `ML_Prob_Win_10d` | 2 weeks |
| 21d | `Win_21d` | `ML_Prob_Win_21d` | 5 weeks |
| 42d | `Win_42d` | `ML_Prob_Win_42d` | 9 weeks |

`Hit_*` (MFE vs signal close) is **research-only**. It is not the promotion objective. `Rel_Forward_*` is unused as a train target.

Training rows are **`Is_New_Coil == True` only**. Splits are expanding chronological walk-forward folds. Score, logistic, XGBoost, and deterministic random are compared. **No XGBoost/LightGBM averaging. No blending of 10d/21d/42d.**

**As of 2026-08-24:** `schema_version` 5, all horizons `production_model: none`. Do not serve these models live. Do not retrain the same 26 features on Win expecting a different gate result.

> Everything from **“Historical: MAE regression baseline”** downward is the old 6-feature `Forward_Return_2w` / Rel_Forward trainer. It is not executable contract.

---

## Purpose

| Goal | Detail |
| ---- | ------ |
| Task | Binary **Win** classification at 10d / 21d / 42d |
| Models | Logistic baseline + one `XGBClassifier` per horizon |
| Why | Rank new-coil setups by P(close up) at that horizon |
| Promote only if | Walk-forward top 10% **within fold** beats Score, random, **and** population on avg forward, median forward, and win rate, and ≥60% of folds pass |
| Not for | Live order routing, options P&L, replacing the rubric score, or Hit-rate optimization |

---

## Prerequisites

1. **Dependencies** (already in `requirements.txt`): pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib.

2. **Source data:** native Coiled Cobra walk-forward CSV from:

   ```bash
   python -m finance_vibe.coiled_cobra_backtest daily --backtest
   ```

   **Pinned file:** `data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv` (21,961 new coils, 114 columns with native `Forward_Return_*` / `Max_Return_*` / `Win_*` / `Hit_*`). Always pass `--csv`. Auto-discover still prefers a missing `..._2026-07-17.csv` then newest mtime and can pick stale `_Large.csv`.

3. **Environment:** Docker image code is baked; `./data` is the volume. After editing `src/`, rebuild or `docker cp`. `PYTHONPATH=/app/src`.

---

## How to run

```bash
docker exec finance_vibe python -m finance_vibe.coiled_cobra_ml_training \
  --csv /app/data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv \
  --artifacts-dir /app/data/logs/daily
```

| Flag | Meaning |
| ---- | ------- |
| `--csv` | **Required in practice** — pin the native trades file |
| `--artifacts-dir` | Writes `coiled_cobra_xgb_{10,21,42}d.json`, `coiled_cobra_ml_metadata_{horizon}.json`, index, OOS CSVs |

Live `ml_ranker.attach_horizon_probabilities` fills `ML_Prob_Win_*` when a booster loads and `SERVE_ML_RANKER` is True (default). `production_model: xgb` still means the OOS gate passed.

---

## Pipeline architecture (current)

```
coiled_cobra_backtest_trades_2026-08-24.csv
        │
        ▼
  load_and_prepare()
    • keep Is_New_Coil == True
    • drop leakage cols / prefixes (forwards, MAE, Max, Win, Hit, ML_*)
    • require Max_Return_* after enrich (no close-based Hit fallback)
        │
        ▼
  walk_forward_folds()  (expanding; embargo 2w/5w/9w)
    skip n_train < 1000 or best_iteration < 3
        │
        ▼
  X = 26 FEATURE_COLS (no Score / Grade / Fib)
  y = Win_{10,21,42}d
        │
        ├── LogisticRegression (baseline)
        ├── XGBClassifier (early stop on val AUC)
        ├── Score ranker (live baseline)
        └── Random ranker (same seed)
                │
                ▼
        top_fraction WITHIN FOLD, then pool
        promote on avg_fwd, med_fwd, win_rate vs Score+random+pop
```

---

## Column isolation (anti-leakage)

### Feature space (X) — 26 pillars + raw (no Score/Grade/Fib)

Pillars: `Volume_Shelf`, `MACD_Compression`, `Structure`, `RS_Score`, `Coil_Width`, `Proximity_Highs`.

Raw: `Volume_Contraction_Ratio`, `MACD_Spread_ATR`, `Coil_Width_ATR`, `Coil_Width_Pctile`, `Dist_High_{63,126,252}_Pct` / `_ATR`, `OBV_Coil_Slope`, `Up_Volume_Ratio`, `Volume_Trend_Ratio`, `RSI`, `RSI_Healthy`, `Pct_From_EMA20`, `Pct_From_EMA50`, `ATR_Pct`, `Distance_To_Pivot_Pct`, `MACD_Crossed`.

**`Score` and `Grade` are excluded from trees.** Score gates the sample (≥70 + hard gates) and is a live ranking baseline, not `FEATURE_COLS`. Tests forbid `Pct_From_Fib618`.

### Target (y)

| Horizon | Train | Trading metrics | Research (not the gate) |
| ------- | ----- | --------------- | ----------------------- |
| 10d | `Win_10d` | `Forward_Return_10d` | `Hit_10Pct_10d`, `Hit_15Pct_10d` |
| 21d | `Win_21d` | `Forward_Return_21d` | `Hit_15Pct_21d`, `Hit_20Pct_21d` |
| 42d | `Win_42d` | `Forward_Return_42d` | `Hit_25Pct_42d`, `Hit_50Pct_42d` |

Hit was retired as the production label because MFE is monotone in volatility (negative OOS medians as the cut tightens). See `CodeReview.MD` T1.

**Metadata caveat:** on-disk `target_column` currently shows the last research Hit name because of a loop-variable leak (`CodeReview.MD` I5). Training still uses `Win_*`.

### Leakage columns — strictly dropped before training

| Dropped column | Why |
| -------------- | --- |
| `Stock Entry`, `Stock Stop`, `Target 1`, `Target 2` | Planned bracket levels |
| `Outcome`, `Exit Date`, `Exit Price` | Realized trade path |
| `R Multiple`, `Target_Label`, `Target_R_Mult` | Post-hoc labels |
| Prefixes `Forward_Return_`, `Rel_Forward_`, `MAE_`, `Max_Return_`, `Held_Coil_Low_`, `Win_`, `Hit_`, `ML_Prob_`, `ML_Pred_`, `ML_Rank` | Outcomes / model outputs |

`Symbol` / `Signal Date` are for splits only.

### Row filtering

| Rule | Behavior |
| ---- | -------- |
| `Is_New_Coil == True` | Kept |
| Target NaN | Dropped (tail of series) |
| Random shuffle / K-fold | Forbidden |

---

## Temporal split

Walk-forward: expanding train, ~26w val, ~26w test, embargo = horizon weeks. Not a single 2023/2024/2025 cut. Fraction cuts (`top_10pct`, …) are taken **per fold**.

Floors: `MIN_TRAIN_ROWS = 1000`, `MIN_BEST_ITERATION = 3` (need four trees at lr=0.01).

---

## Sample weighting

Uniform (`USE_INVERSE_ATR_WEIGHTS = False`). Inverse ATR is a later experiment, not current.

---

## Model configuration

| Param | Value |
| ----- | ----- |
| `max_depth` | 4 |
| `learning_rate` | 0.01 |
| `n_estimators` | 400 (early stopping 40 rounds on val AUC) |
| `subsample` / `colsample_bytree` | 0.8 |
| `min_child_weight` | 16 |
| `reg_lambda` | 2.0 |
| `random_state` | 42 |
| Objective | XGBoost binary logistic (`XGBClassifier`) |

LightGBM is **not** trained or loaded for live ranks. Logistic is an OOS baseline only. E3: logistic often beats this XGB on top-decile **mean**; both lose the promotion gate. Do not tune lr until features or labels change.

---

## Outputs & diagnostics

| Artifact | Role |
| -------- | ---- |
| `coiled_cobra_xgb_{10,21,42}d.json` | Booster (`best_iteration` in metadata) |
| `coiled_cobra_ml_metadata_{horizon}.json` | Fold table, promotion, gain |
| `coiled_cobra_ml_model_metadata.json` | Index of three horizons |
| `coiled_cobra_ml_oos_{horizon}.csv` | Fold-stamped OOS for re-scoring |

Stdout prints the ML vs Score / random / population table (`top_10pct` within fold). Classification P/R at 0.5 is uninformative (E4).

### How to read the promotion table

| Metric | Meaning |
| ------ | ------- |
| avg_fwd / med_fwd / win_rate | Close-to-close over the horizon, **not** MFE hit rate |
| Blocked by | Which of those failed vs Score, random, or population |
| Fold pass | Share of folds that would promote on their own (≥0.60 required) |

---

## Historical: MAE regression baseline

The sections below describe the **retired** 6-feature `XGBRegressor` / `LGBMRegressor` on `Rel_Forward_*` / `Forward_Return_2w`, including the importance chart that ranks EMA50, ATR, EMA20, Fibs, and Score. That stack is not what `coiled_cobra_ml_training.py` runs today.

### Historical pipeline architecture

```
coiled_cobra_backtest_trades_*.csv
        │
        ▼
  load_and_prepare()
    • keep Is_New_Coil == True
    • parse Signal Date → datetime
    • DROP leakage columns (execution / outcome)
    • DROP rows with NaN Rel_Forward_2w (or Forward_Return_2w fallback)
        │
        ▼
  temporal_split() on Signal Date  (NO random K-fold)
    rolling 6-month val / 6-month test from max date
        │
        ▼
  build_matrices()
    X = v2.2 pillars + raw geometry (no Score/Grade)
    y = Rel_Forward_42d (daily) or Rel_Forward_13w (weekly)
    y = Rel_Forward_2w
    sample_weight = ATR_Pct
        │
        ├── XGBRegressor(objective="reg:absoluteerror")
        └── LGBMRegressor(objective="regression_l1")
                │
                ▼
        MAE / RMSE on Val + Test
        ASCII + PNG feature importances
```

---

## Column isolation (anti-leakage)

### Feature space (X) — pillars and raw (no Score/Grade)

Pillars: `Volume_Shelf`, `MACD_Compression`, `Structure`, `RS_Score`, `Coil_Width`, `Proximity_Highs`.

Raw: `Volume_Contraction_Ratio`, `MACD_Spread_ATR`, `Coil_Width_ATR`, `Coil_Width_Pctile`, `Dist_High_{63,126,252}_Pct` / `_ATR` (weekly bars 13/26/52, same names), `OBV_Coil_Slope`, `Up_Volume_Ratio`, `Volume_Trend_Ratio`, `RSI`, `RSI_Healthy`, `Pct_From_EMA20`, `Pct_From_EMA50`, `ATR_Pct`, `Distance_To_Pivot_Pct`, `MACD_Crossed`.

**`Score` and `Grade` are excluded from trees.** Score gates the sample (≥70 + hard gates) and is a live ranking baseline, not `FEATURE_COLS`.

### Target (y)

| Mode | Preferred | Fallbacks |
| ---- | --------- | --------- |
| Daily | `Rel_Forward_42d` | `Rel_Forward_13w`, `Rel_Forward_2w`, `Forward_Return_2w` |
| Weekly | `Rel_Forward_13w` | `Rel_Forward_26w`, `Rel_Forward_2w` |

Also exported: 21d (daily), 4w/8w (weekly), absolute forwards, `MAE_*`, `Held_Coil_Low_*`.

Embargo weeks match the training target (9 weeks for 42d, 13 weeks for Rel_Forward_13w, etc.).

### Leakage columns — strictly dropped before training

These encode post-signal execution / outcome information and must never enter `X`:

| Dropped column | Why |
| -------------- | --- |
| `Stock Entry`, `Stock Stop`, `Target 1`, `Target 2` | Planned bracket levels derived for simulation |
| `Outcome`, `Exit Date`, `Exit Price` | Realized trade path |
| `R Multiple`, `Target_Label`, `Target_R_Mult` | Post-hoc trade labels / R multiples |

Other identity columns (`Symbol`, `Signal Date`, `Setup Type`) are used only for splitting / sorting, not as model features.

### Row filtering

| Rule | Behavior |
| ---- | -------- |
| `Is_New_Coil == True` | **Kept** — train on new coil episodes, not aged continuation bars |
| Target is NaN/None | **Dropped** — insufficient future bars near series end |
| Random shuffle / K-fold | **Forbidden** — would leak future structure across time |

---

## Temporal split (rigid, date-based)

Split on **`Signal Date`** only — never random folds.

| Partition | Signal Date range | Role |
| --------- | ----------------- | ---- |
| Train | Inception → 2023-12-31 | Fit |
| Validation | 2024-01-01 → 2024-12-31 | Tuning / comparison |
| Test (OOS) | 2025-01-01 → 2026-07-31 | Final holdout |

Example sizes from `coiled_cobra_backtest_trades_2026-07-17.csv` (after NaN-target drop):

| Partition | Rows × features |
| --------- | --------------- |
| Train | 3398 × 6 |
| Val | 764 × 6 |
| Test | 1117 × 6 |

Raw CSV was 5397 rows; 118 rows dropped for missing `Forward_Return_13w`.

---

## Sample weighting

Both frameworks receive **uniform** sample weights by default. Inverse `ATR_Pct` (`USE_INVERSE_ATR_WEIGHTS = True`) is a later experiment.

---

## Model configuration

Shared hyperparameters:

| Param | Value |
| ----- | ----- |
| `max_depth` | 4 |
| `learning_rate` | 0.01 |
| `n_estimators` | 400 (early stopping, 40 rounds) |
| `subsample` | 0.8 (`bagging_freq=1` for LightGBM) |
| `colsample_bytree` | 0.8 |
| `random_state` | 42 |

| Framework | Objective | Rationale |
| --------- | --------- | --------- |
| XGBoost `XGBRegressor` | `reg:absoluteerror` | MAE loss — robust to heavy-tailed financial outliers that inflate Test RMSE under squared error |
| LightGBM `LGBMRegressor` | `regression_l1` | Same L1 / MAE family for apples-to-apples comparison |

Missing values: left as-is; both libraries handle NaNs natively in tree growth. No median imputation is applied to features.

---

## Outputs & diagnostics

Stdout always prints:

1. **Dataset shape integrity** — `X_train` / `X_val` / `X_test` row × column counts and feature list
2. **Validation scores** — MAE and RMSE on 2024 val and 2025–2026 test for each model
3. **ASCII feature-importance charts** — rank-ordered by split/gain importance

Artifact files:

```
{artifacts-dir}/coiled_cobra_ml_feature_importance.png
{artifacts-dir}/coiled_cobra_xgb_model.json
{artifacts-dir}/coiled_cobra_lgb_model.txt
{artifacts-dir}/coiled_cobra_ml_model_metadata.json
```

Side-by-side horizontal bar chart (XGBoost vs LightGBM), plus serialized model weights and a JSON metadata summary for downstream use.

### How to read metrics

| Metric | Interpretation |
| ------ | -------------- |
| **MAE** | Primary fit quality under the training objective; less dominated by extreme movers |
| **RMSE** | Tail sensitivity; can remain high on OOS even when MAE improves |
| **Spearman** | **Production metric** — rank correlation of prediction vs realized Rel_Forward. This is what `ML_Rank` / helper Priority actually use. |

Expect **Test RMSE ≫ Val RMSE** when a few extreme high-beta paths appear in 2025–2026. Prefer MAE for comparing objective upgrades; use RMSE to audit tail risk.

---

## Baseline results (reference run)

Source: `coiled_cobra_backtest_trades_2026-07-17.csv`  
Config: 6 features, MAE objectives, `ATR_Pct` sample weights.

### Current baseline (MAE objectives, no Grade)

| Model | Val 2024 MAE / RMSE | Test 2025–26 MAE / RMSE |
| ----- | ------------------- | ----------------------- |
| XGBoost | 0.277 / 0.521 | 0.400 / 1.503 |
| LightGBM | 0.267 / 0.491 | 0.386 / 1.494 |

### Ablation vs earlier squared-error + Grade run

| Change | Effect |
| ------ | ------ |
| Drop `Grade` | Removes collinear categorical of `Score`; importances redistribute across geometry / ATR |
| `reg:squarederror` → `reg:absoluteerror` / `regression_l1` | Val MAE/RMSE improved materially; Test MAE improved; Test RMSE still tail-dominated |

Approximate prior (7 features including Grade, squared error):

| Model | Val MAE / RMSE | Test MAE / RMSE |
| ----- | -------------- | --------------- |
| XGBoost | 0.313 / 0.667 | 0.432 / 1.509 |
| LightGBM | 0.330 / 0.672 | 0.459 / 1.492 |

### Typical feature importance pattern (current)

- **LightGBM:** `ATR_Pct` often leads, then EMA / Fib distances, then `Score`
- **XGBoost (MAE):** Fib / ATR / EMA importances more balanced; `Score` still contributory

Importance ranks can shift run-to-run with library versions; treat charts as diagnostic, not a frozen production ranking.

---

## How to use the trained models for decisions

The exported models are intended to be a soft decision aid, not a stand-alone trading system.

1. **Train the baseline**
   - Run the training script to produce model artifacts under the relevant log directory.
   - Keep the artifacts next to the weekly or daily backtest exports so the same folder is easy to discover.

2. **Attach model scores to new setups**
   - Use the existing ranking helper in `src/finance_vibe/ml_ranker.py` to load the saved boosters and attach `ML_Pred_Return` plus `ML_Rank` to Coiled Cobra setups.
   - The helper will derive the same feature frame used in training when the raw scanner columns are present.

3. **Use the model as a secondary signal**
   - Favor setups with a high `ML_Rank` only after they already pass the core rubric checks: macro regime alignment, structure, risk management, and liquidity.
   - Treat the model as a tie-breaker or confirmation layer, not a hard entry gate.

4. **Make more informed decisions**
   - Compare the model rank against `Score` and the scanner geometry. A setup that is strong on the rubric and also ranks highly in the ML model is a better candidate for review.
   - If the model disagrees with the rubric, investigate the reason: it may be highlighting a regime where the current rules are too conservative or too aggressive.
   - Keep a simple review checklist: market regime, risk/reward, position size, options liquidity, and model rank.

A practical workflow is:

- scan for Coiled Cobra setups,
- filter for quality by rubric / structure / risk,
- attach model predictions,
- rank the survivors by `ML_Pred_Return` or `ML_Rank`,
- then commit to the final decision using the combined evidence.

## Design review notes

| Decision | Status | Notes |
| -------- | ------ | ----- |
| Pre-signal-only features | Correct | Leakage cols explicitly dropped |
| Keep `no_fill` | Correct | Continuous target still defined without a fill |
| Temporal split (no K-fold) | Correct | Prevents future→past leakage |
| Drop `Grade` | Correct | Redundant with `Score` |
| MAE / L1 objectives | Correct for tails | Stabilizes val; OOS RMSE still needs tail strategy |
| `ATR_Pct` as `sample_weight` | Contractual baseline | Motivating text argued for balancing high-beta variance; if weights appear to *amplify* high-ATR names, consider researching `1/ATR_Pct` as an experiment (not current default) |
| Dual XGB + LGBM | Useful | Cross-checks that conclusions are not framework-specific |
| Not wired into live path | Intentional | Research script only |

### Known limitations

- Stock forward return only — no options / LEAPS P&L target
- Single horizon (`13w`); 5w / 26w not trained in this baseline
- No hyperparameter search, early stopping, or calibration layer
- No model serialization / inference API yet
- Test RMSE remains sensitive to extreme movers (e.g. high-beta names)

### Sensible next experiments (not implemented)

1. Inverse-vol sample weights (`1 / ATR_Pct`) vs current `ATR_Pct`
2. Huber / Pseudo-Huber objectives (`reg:pseudohubererror`, LightGBM `huber`)
3. Winsorize or clip `y` at train quantiles before fit; still evaluate unclipped OOS
4. Multi-horizon multi-output or separate heads for 5w / 13w / 26w
5. Walk-forward expanding window instead of a single fixed cut
6. Persist best model + feature schema for offline scoring of new setups

---

## Relationship to other modules

| Module | Relationship |
| ------ | ------------ |
| `coiled_cobra_backtest.py` | **Upstream** — produces the trades CSV (features + forward returns + leakage cols) |
| `coiled_cobra.py` | Live scanner; same geometry fields at signal time; ML does not call it |
| `pipeline_backtest.py` | Separate quality-swing / high_beta path — not an ML input |
| `run_vibe.py` | Does **not** invoke ML training |

Full backtest column definitions and CLI: **`BacktestAndBackfill.md`**.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `FileNotFoundError` for CSV | Backtest not run / wrong mount | See **`MLOps.md`**: `--backtest` then `--csv` under `/app/data/logs/daily/` |
| Empty train/val/test | Date range outside CSV | Check `Signal Date` min/max; regenerate backtest with full history |
| Stale script in Docker | Image baked without host edits | `docker cp` the `.py` or `docker compose up -d --build` |
| Import errors for xgboost/lightgbm | Old image / missing deps | Rebuild image; confirm `requirements.txt` installed |
| Huge Test RMSE, OK Val MAE | Tail events in OOS | Expected under heavy tails; compare MAE; consider clipping / Huber experiments |
