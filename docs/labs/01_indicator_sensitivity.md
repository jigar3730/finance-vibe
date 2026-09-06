# Lab 01 — Indicator Sensitivity Analysis

**Objective.** Add or remove one feature from the Coiled Cobra ML matrix and
observe how XGBoost (and LightGBM) gain ranks re-weight. This is the
empirical test of §1 in [`QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md).

## Why this experiment

`FEATURE_COLS` is a 6-column subset of the live coil geometry. `Score` is
collinear with the EMA/Fib distances; `Grade` was already dropped for that
reason. Ablating one column shows whether the trees were using a signal or
just sharing credit inside a correlated cluster.

## Starter pointers

| What | Where |
| ---- | ----- |
| Feature contract | `FEATURE_COLS` in `src/finance_vibe/coiled_cobra_ml_training.py` |
| Inference must match | `ml_ranker.build_feature_frame()` / `FEATURE_COLS` import |
| Indicator construction | `coiled_cobra.add_macro_indicators()` |
| Scorecard pillars **not** in $X$ | `macd_compression_score`, `coil_width_score`, `evaluate_volume_profile_shelf`, `rs_score` |
| Importance artifact | `data/logs/weekly/coiled_cobra_ml_feature_importance.png` |
| Tests that pin the schema | `tests/test_ml_ranker.py` |

## Prerequisites

Complete the [labs README](README.md) data + trainer steps. Record a **baseline**
run before you edit anything:

```bash
export PYTHONPATH=src
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv \
  --artifacts-dir data/logs/weekly/lab01_baseline
```

Copy the ASCII importance tables and Val/Test MAE into your notes.

## Exercise A — drop an existing indicator (fast path)

1. In `coiled_cobra_ml_training.py`, temporarily remove **one** name from
   `FEATURE_COLS`. Recommended first drop: `"Pct_From_Fib786"` (optional
   bonus in the rubric, still in $X$).
2. Retrain into a **different** artifacts directory:

   ```bash
   python src/finance_vibe/coiled_cobra_ml_training.py \
     --csv data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv \
     --artifacts-dir data/logs/weekly/lab01_drop_fib786
   ```

3. Compare, for both XGBoost and LightGBM:
   - new gain rank order,
   - Val MAE / RMSE vs the baseline,
   - whether `Pct_From_Fib618` or `Score` absorbed the dropped column's credit.

4. Restore `FEATURE_COLS` and repeat with `"Score"` dropped. If Val MAE barely
   moves, the trees were already reading geometry directly.

## Exercise B — add a derived coil-width proxy (no re-backtest)

The trades CSV does **not** currently export `coil_width_score`. You can still
add a **research-only** column if you derive it from existing fields, or skip
to Exercise C.

Minimal pattern inside `_load_and_prepare()` (research copy — do not leave this
in production without updating `ml_ranker.FEATURE_COLS`):

```python
# Example: interaction already implied by the scorecard (geometry × vol).
df["Abs_Pct_From_EMA20"] = df["Pct_From_EMA20"].abs()
# Then append "Abs_Pct_From_EMA20" to FEATURE_COLS for this run only.
```

Retrain, then ask:

- Did XGBoost put the new column in the top three?
- Did `Pct_From_EMA20` importance collapse (redundancy)?
- Did Test MAE improve, or only Val (overfit to the new split)?

## Exercise C — add a real indicator (full pipeline)

To pull MACD compression or coil width into $X$ you must:

1. Compute the value in `coiled_cobra.py` at signal time (same causal window
   as `evaluate_coiled_cobra`).
2. Persist it on the setup row in `coiled_cobra_backtest.py` (next to
   `Pct_From_Fib786`).
3. Re-run `--backtest` so the trades CSV grows a new column.
4. Append that column name to `FEATURE_COLS` **and** teach
   `ml_ranker.build_feature_frame()` how to derive or pass it through.
5. Retrain and compare importances.

This is the only path that keeps training and inference aligned.

## What to record

| Run | Features | XGB Val MAE | LGB Val MAE | Top-3 XGB | Top-3 LGB |
| --- | -------- | ----------- | ----------- | --------- | --------- |
| Baseline | 6 cols | | | | |
| Drop Fib786 | 5 cols | | | | |
| Drop Score | 5 cols | | | | |
| Add derived | 7 cols | | | | |

## Expected failure modes

- Editing `FEATURE_COLS` but not `ml_ranker` → inference frame mismatch;
  `build_feature_frame` will omit or NaN the new column.
- Adding a column that exists only after exit (`R Multiple`, `Outcome`) →
  leakage. Check `LEAKAGE_COLS` first.
- Judging success by Test RMSE alone — tails dominate; prefer Val MAE and
  rank stability.

## Restore

Revert `FEATURE_COLS` (and any `_load_and_prepare` experiments) when finished.
Keep the `lab01_*` artifact folders as your lab notebook.
