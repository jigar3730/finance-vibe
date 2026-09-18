# Coiled Cobra ML Baseline

**Module:** `src/finance_vibe/coiled_cobra_ml_training.py`

Standalone training script that learns a short-horizon alpha gradient from Coiled Cobra backtest exports. It predicts **`Forward_Return_2w`** (2-bar / ~2-week forward close-to-close return) using only pre-signal technical attributes — never post-trade execution fields.

This is an **offline research baseline**, not part of `run_vibe.py` or the live scanner.

---

## Purpose

| Goal | Detail |
| ---- | ------ |
| Task | Regression: predict continuous `Forward_Return_2w` |
| Models | Vanilla `XGBRegressor` + `LGBMRegressor` side-by-side |
| Why | Rank which coil geometries historically realized stronger forward alpha; inform score/geometry research |
| Not for | Live order routing, options P&L, or replacing the rubric score |

---

## Prerequisites

1. **Dependencies** (already in `requirements.txt`):

   ```
   pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib
   ```

2. **Source data:** a Coiled Cobra walk-forward trades CSV produced by:

   ```bash
   python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
   ```

   Typical path:

   ```
   data/logs/weekly/coiled_cobra_backtest_trades_YYYY-MM-DD.csv
   ```

   Auto-selection takes the **newest** `coiled_cobra_backtest_trades_*.csv` (by the date stamp in the filename) from the mode's log directory, falling back to legacy roots only if that directory holds none. There is no hard-coded filename. If the newest file is stamped with a different rubric version (or is unversioned, i.e. pre-v4.0), training **refuses** it — regenerate with `coiled_cobra_backtest.py weekly --backtest`.

3. **Environment:** run from the repo / container with `PYTHONPATH` including the project (Docker image already sets `PYTHONPATH=/app`).

---

## How to run

```bash
# Auto-select the newest CSV in the weekly log dir; models land there too
python src/finance_vibe/coiled_cobra_ml_training.py

# Explicit CSV + artifact directory
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_YYYY-MM-DD.csv \
  --artifacts-dir data/logs/weekly

# Inside the finance_vibe container
docker exec finance_vibe python /app/src/finance_vibe/coiled_cobra_ml_training.py
```

| Flag | Meaning |
| ---- | ------- |
| `--csv` | Path to trades CSV (optional; newest in the mode log dir if omitted; a non-existent path is an error, never a silent fallback) |
| `--artifacts-dir` | Where to write models + metadata + PNG (default: the mode's log dir — where `ml_ranker` looks) |
| `--mode` | Pipeline mode the model is bound to. Only `weekly` is supported (`Forward_Return_2w` counts bars) |
| `--allow-rubric-mismatch` | Experiments only: train on another rubric vintage. The model records that version and `ml_ranker` will refuse it |

---

## Pipeline architecture

```
coiled_cobra_backtest_trades_*.csv
        │
        ▼
  _load_and_prepare()
    • parse Signal Date → datetime
    • DROP leakage columns (execution / outcome)
    • KEEP no_fill rows
    • DROP rows with NaN Forward_Return_2w
        │
        ▼
  _temporal_split() on Signal Date  (NO random K-fold)
    Train  < max_date − 52 weeks
    Val    last-52w .. last-26w
    Test   last-26w .. max_date
        │
        ▼
  _build_matrices()
    X = 6 pre-signal features
    y = Forward_Return_2w
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

### Feature space (X) — exactly 6 columns

| Feature | Role |
| ------- | ---- |
| `Score` | Rubric score at signal bar |
| `Pct_From_EMA20` | Close vs EMA20 (fraction) |
| `Pct_From_EMA50` | Close vs EMA50 (fraction) |
| `Pct_From_Fib618` | Close vs Fib 61.8% (fraction) |
| `Pct_From_Fib786` | Close vs Fib 78.6% (fraction) |
| `ATR_Pct` | ATR / Close (volatility scale) |

**`Grade` is intentionally excluded.** It is a redundant binned categorical of `Score` (e.g. A vs B). Including both wasted tree split budget and added multi-collinearity noise without predictive lift.

### Target (y)

| Column | Definition |
| ------ | ---------- |
| `Forward_Return_2w` | `(Close[t+2] − Close[t]) / Close[t]` on the weekly series |

Also present in the CSV but **not** used as the baseline target: `Forward_Return_5w`, `Forward_Return_13w`, `Forward_Return_26w`. See [Lab 02](../labs/02_target_horizon_shift.md) to switch horizons.

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
| `Outcome == no_fill` | **Kept** — model still learns coil geometry vs continuous forward return |
| `Forward_Return_2w` is NaN/None | **Dropped** — insufficient future bars near series end |
| Random shuffle / K-fold | **Forbidden** — would leak future structure across time |

---

## Temporal split (dynamic, date-based)

Split on **`Signal Date`** only — never random folds. `_temporal_split()` in `coiled_cobra_ml_training.py` rolls backward from the latest signal date:

| Partition | Signal Date range | Role |
| --------- | ----------------- | ---- |
| Train | Inception → `max_date − 52w − embargo` | Fit |
| *Embargo* | `max_date − 52w − 2w` → `max_date − 52w` | Purged (dropped) |
| Validation | `max_date − 52w` → `max_date − 26w − embargo` | Tuning / comparison |
| *Embargo* | `max_date − 26w − 2w` → `max_date − 26w` | Purged (dropped) |
| Test (OOS) | `max_date − 26w` → `max_date` | Final holdout |

**Embargo:** the label at date *t* is realised at *t + 2 bars*, so rows within `EMBARGO_WEEKS` (= `TARGET_HORIZON_WEEKS`) before a boundary would have labels overlapping the next partition (and the same-week market move shared by every ticker). They are dropped from the earlier partition; the number purged is printed and stored in the metadata.

Example sizes from an older fixed-cut run on `coiled_cobra_backtest_trades_2026-07-17.csv` (after NaN-target drop). Counts will shift under the current rolling window:

| Partition | Rows × features |
| --------- | --------------- |
| Train | 3398 × 6 |
| Val | 764 × 6 |
| Test | 1117 × 6 |

---

## Sample weighting

Both frameworks receive `sample_weight=ATR_Pct` at `.fit()` time.

- High-ATR (high-beta) names produce larger absolute return variance.
- Weighting by `ATR_Pct` is the current baseline contract so optimization is not dominated solely by unscaled squared residuals on quiet large-caps (paired with MAE objectives below).
- Non-finite or non-positive weights are replaced with the **train median** of `ATR_Pct`.

---

## Model configuration

Shared hyperparameters (`MODEL_PARAMS` in `coiled_cobra_ml_training.py`):

| Param | Value |
| ----- | ----- |
| `max_depth` | 4 |
| `learning_rate` | 0.01 |
| `n_estimators` | 400 |
| `subsample` | 0.8 |
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
2. **Validation scores** — MAE and RMSE on the rolling val and test windows for each model
3. **ASCII feature-importance charts** — rank-ordered by split/gain importance

Artifact files:

```
{artifacts-dir}/coiled_cobra_ml_feature_importance.png
{artifacts-dir}/coiled_cobra_xgb_model.json
{artifacts-dir}/coiled_cobra_lgb_model.txt
{artifacts-dir}/coiled_cobra_ml_model_metadata.json
```

Side-by-side horizontal bar chart (XGBoost vs LightGBM), plus serialized model weights and a JSON metadata summary for downstream use.

### Integrity guards (rubric version, mode binding, hashes)

- `config.RUBRIC_VERSION` (currently `"4.0"`) is written to a `Rubric_Version` column on every backtest/backfill CSV. **Bump it whenever a gate/pillar/threshold change alters `Score` or which setups qualify** (e.g. Gate D 5/6 → 4/6) — it is the only thing that lets training and inference detect that drift.
- Model metadata records `rubric_version`, `mode`, `feature_columns`, `trained_at`, `git_sha`, `source_csv`, split bounds/row counts, and a `sha256` for each model file. The old metadata is deleted before new binaries are written, so a crashed retrain can't leave old metadata paired with new models.
- `ml_ranker` serves predictions only when the metadata exists and its rubric version, mode and feature list match the live pipeline, and each model file's hash matches. Models are read **only** from the requested mode's own log directory (optionally `FINANCE_VIBE_MODEL_DIR`) — there is no cross-mode fallback, so a weekly model can never score `daily`/`high_beta` setups. Any failure logs the reason and leaves `ML_Pred_Return` / `ML_Rank` null (ranking falls back to `Score`).

### How to read metrics

| Metric | Interpretation |
| ------ | -------------- |
| **MAE** | Primary fit quality under the training objective; less dominated by extreme movers |
| **RMSE** | Still reported for tail sensitivity; can remain high on OOS even when MAE improves |

Expect **Test RMSE ≫ Val RMSE** when a few extreme high-beta paths appear in 2025–2026. Prefer MAE for comparing objective upgrades; use RMSE to audit tail risk.

---

## Baseline results (historical reference run)

Source: `coiled_cobra_backtest_trades_2026-07-17.csv` under an **older fixed
date cut** (not the current rolling 26w/26w split) and older hyperparameters.
Treat the numbers as a snapshot, not a live SLA.

Config: 6 features, MAE objectives, `ATR_Pct` sample weights.

### Snapshot metrics (MAE objectives, no Grade)

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

## Walk-forward evaluation (`coiled_cobra_ml_walkforward.py`)

Read-only research harness (never writes served artifacts). Expanding-window folds of 26-week out-of-sample windows walking back from the last signal date; each fold trains the **deployed** config (`make_xgb` / `make_lgb`, ATR weights, mean ensemble) on rows dated before `test_start − 2w embargo`. It compares the ML ranking with raw `Score` per `Signal Date` (the live use is ranking one scan's setups against each other):

- Spearman **rank IC** vs `Forward_Return_2w`, and top-tercile − bottom-tercile mean return (weeks with ≥ `--min-names` setups), plus pooled per-fold versions.
- Paired ML − Score differences with t-stats (effective n = weeks / 2, since 2-bar labels overlap) and a 2-week-block bootstrap 95% CI.

```bash
python -m finance_vibe.coiled_cobra_ml_walkforward [--csv PATH] [--min-names 6] [--out report.json]
```

**Gate for using ML in `Priority`:** the paired IC-difference CI should exclude 0 in ML's favour. Score-only ranking is the baseline to beat.

**First result (v4.0 backfill 2026-09-18, 2,284 rows, 8 folds, 2022-09 → 2026-08):** mean weekly rank IC ML +0.011 vs Score +0.053 (neither significant); paired difference −0.042, CI [−0.160, +0.073]; ML MAE 0.0598 vs 0.0586 for predicting the train median. Same conclusion at `--min-names` 4 and 10. Power is limited (≈104 usable weeks ⇒ only IC ≳ 0.11 is detectable), so read this as *no evidence of edge*, not *proof of none*.

---

## How to use the trained models for decisions

The exported models are intended to be a soft decision aid, not a stand-alone trading system.

1. **Train the baseline**
   - Run the training script to produce model artifacts under the relevant log directory.
   - Artifacts are written to (and read from) the mode's log directory, e.g. `data/logs/weekly/`; `ml_ranker` does not search anywhere else.

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
- Single horizon (`2w`); 5w / 13w / 26w are present in the CSV but not trained in this baseline
- No hyperparameter search, early stopping, or calibration layer
- Inference is **soft only** (`ml_ranker.attach_ml_ranks` writes nullable columns)
- Test RMSE remains sensitive to extreme movers (e.g. high-beta names)

### Sensible next experiments (not implemented)

1. Inverse-vol sample weights (`1 / ATR_Pct`) vs current `ATR_Pct`
2. Huber / Pseudo-Huber objectives (`reg:pseudohubererror`, LightGBM `huber`)
3. Winsorize or clip `y` at train quantiles before fit; still evaluate unclipped OOS
4. Multi-horizon multi-output or separate heads for 5w / 13w / 26w
5. Purged multi-fold walk-forward (Lab 04) on top of the existing rolling cut
6. Add RVOL / Market Gate / coil-width as optional research features (not in `FEATURE_COLS`)

---

## Relationship to other modules

| Module | Relationship |
| ------ | ------------ |
| `coiled_cobra_backtest.py` | **Upstream** — produces the trades CSV (features + forward returns + leakage cols) |
| `coiled_cobra.py` | Live scanner; same geometry fields at signal time; ML does not call it |
| `pipeline_backtest.py` | Separate quality-swing / high_beta path — not an ML input |
| `run_vibe.py` | Does **not** invoke ML training |

Full backtest column definitions and CLI: **[`backtest_and_backfill.md`](backtest_and_backfill.md)**. Theory mapped to this trainer: **[`QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md)**.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `FileNotFoundError` for CSV | Backtest not run / wrong mount | Run `--backtest`; pass `--csv`; in Docker ensure `/app/data` volume has `logs/weekly/` |
| Empty train/val/test | Date range outside CSV | Check `Signal Date` min/max; regenerate backtest with full history |
| Stale script in Docker | Image baked without host edits | `docker cp` the `.py` or `docker compose up -d --build` |
| Import errors for xgboost/lightgbm | Old image / missing deps | Rebuild image; confirm `requirements.txt` installed |
| Huge Test RMSE, OK Val MAE | Tail events in OOS | Expected under heavy tails; compare MAE; consider clipping / Huber experiments |
