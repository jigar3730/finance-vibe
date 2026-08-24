# Coiled Cobra MLOps Guide

**Audience:** operators and researchers who run Finance Vibe in Docker and need to train, evaluate, and use the ranking models without treating them as a black box.

**Primary runtime:** container `finance_vibe` (image `finance-vibe-finance-vibe`), working directory `/app`, data volume `/app/data`.

**Companion specs (do not replace this runbook):**

| Document | Role |
| -------- | ---- |
| `Learn.md` / dashboard `/learn` | Curriculum; after **doc edits rebuild** the image (`docker compose up -d --build`) so `/app` markdown updates — compose mounts data only |
| `Coiled Cobra Rubric .MD` | Live scorecard — what a coil *is* |
| `CoiledCobraML.md` | Feature / label / artifact contract (current Win models + historical regression) |
| `docs/CoiledCobraML-Handoff.md` | **Resume here** — decisions, scoreboard, next work (2026-08-24) |
| `CodeReview.MD` | Independent audit + what was fixed vs still open |
| `BacktestAndBackfill.md` | How historical labels are produced |
| This file (`MLOps.md`) | How to **train**, **judge**, and **deploy** models in the container |

`run_vibe.py` never trains. Training is a deliberate offline job. The live scanner only **loads** artifacts if they exist.

---

## 1. What this system is (and is not)

### What you are building

A **multi-horizon supervised ranking layer** for Coiled Cobra setups:

1. Look at a coil **on the signal bar** (26 pillars + EMA/ATR geometry; **not** Score or Fib).
2. Estimate separate probabilities that the **close is up** at 10d / 21d / 42d (`Win_*`). MFE `Hit_*` is research-only.
3. Compare each horizon against Score, **random**, and the **population** on walk-forward OOS (top 10% **inside each fold**).

The probabilities remain separate. A horizon is enabled only when metadata records `production_model: xgb`. That requires beating Score, random, and population on average return, median return, and win rate, with ≥60% of folds passing. **Hit rate is not in the gate.** As of 2026-08-24 no horizon is promoted.

The **rubric Score remains the quality gate and fallback**.

### What you are not building

- Not a live order router
- Not options / LEAPS P&L
- Not a hard “trade / don’t trade” gate
- Not a full ML platform (no MLflow registry, no Airflow DAG, no auto-retrain)

If trading outcomes do not beat Score, metadata records
`production_model: none` and live probabilities remain null.

### Two horizons, two models

Daily is the **project primary**. Weekly is a slower confirmation silo.

| | Daily (default) | Weekly (opt-in) |
| - | --------------- | --------------- |
| Raw data | `/app/data/raw/daily/` (`5y`, `1d`) | `/app/data/raw/weekly/` (`10y`, `1wk`) |
| Logs / artifacts | `/app/data/logs/daily/` | `/app/data/logs/weekly/` |
| Coil window | 30 bars | 8 bars |
| “2w” forward bars | 10 sessions | 2 weeks |
| Live scan default | yes | `python ... coiled_cobra.py weekly` |

A weekly booster **must not** rank a daily scan. `ml_ranker.py` searches only the active mode’s log folder.

---

## 2. Docker environment (memorize this)

Typical `docker ps` on the host (`mediabox`):

```text
CONTAINER ID   IMAGE                     COMMAND                  PORTS                    NAMES
5cbb41f485b7   finance-vibe-finance-vibe "python src/finance_…"   0.0.0.0:5000->5000/tcp   finance_vibe
```

| Fact | Value |
| ---- | ----- |
| Container name | `finance_vibe` |
| Image | `finance-vibe-finance-vibe` (compose project + service) |
| Workdir | `/app` |
| `PYTHONPATH` | `/app/src` (set in the Dockerfile) |
| App code | baked into the image at `/app/src` |
| Persistent data | `/app/data` ← host `/mnt/fast/finance-vibe-data` (`docker-compose.yml`) |
| Default process | Flask dashboard on port **5000** |

**Code vs data.** Source in the image is a snapshot from the last `docker compose build`. Data on the volume survives rebuilds. If you edited Python on the host and training still looks old, rebuild or confirm the container has the new files.

All commands below assume the container is **Up**. Prefix from the **host** (not from inside the container unless noted).

### Open a shell

```bash
docker exec -it finance_vibe bash
# you land in /app
echo $PYTHONPATH    # should be /app/src
ls /app/data/raw/daily | head
```

### One-shot pattern (preferred for jobs)

```bash
docker exec -w /app finance_vibe python src/finance_vibe/<script>.py [args]
```

`-w /app` matches `WORKDIR`. You do not need to export `PYTHONPATH`.

### Confirm Python packages

```bash
docker exec finance_vibe python -c "import xgboost, lightgbm, sklearn, matplotlib; print('ok')"
```

If that fails, rebuild:

```bash
cd /path/to/finance-vibe   # compose project on the host
docker compose up -d --build
```

---

## 3. Mental model: the MLOps loop

```
  [1] Universe          ticker_provider.py
          │
  [2] OHLCV ingest      data_ingestor.py  →  /app/data/raw/daily/*.csv
          │
  [3] Label job         coiled_cobra_backtest.py --backtest
          │               walk-forward coils + Rel_Forward_2w
          ▼
     trades CSV         /app/data/logs/daily/coiled_cobra_backtest_trades_YYYY-MM-DD.csv
          │
  [4] Train             coiled_cobra_ml_training.py
          │               walk-forward, 3× XGBClassifier on Win_*, vs Score/random
          ▼
     artifacts          xgb_{10,21,42}d.json, metadata_*, index, oos_*.csv
          │
  [5] Serve (soft)      coiled_cobra.py  →  ml_ranker.attach_ml_ranks
          │               only if production_model == xgb (currently none)
          ▼
     ML_Prob_Win_*      otherwise null; book stays Score-sorted
```

Steps 1–4 are **batch**. Step 5 is **inference** on the next live run. Nothing is uploaded to a registry; “deploy” means “files exist in the daily log silo.”

---

## 4. Concepts you need before you train

Read this section once. The rest of the guide assumes it.

### 4.1 Supervised learning

You have examples `(X, y)`:

- **X (features)** — numbers known **at signal time** (coil width, RS vs QQQ, …).
- **y (target)** — a number that is only known **later** (forward return vs QQQ).

The model learns a function `f(X) ≈ y`. At live time you compute `f(X_today)` and sort.

If a column is only known after the trade (exit price, fill, R-multiple), it is **leakage**. Using it makes backtest scores look brilliant and live ranks worthless.

### 4.2 Regression vs classification vs ranking

| Task | Question | This project |
| ---- | -------- | ------------ |
| Classification | Will this coil win? (yes/no) | **Not used** — a 51% win rate with huge winners can still be useful |
| Regression | How much relative return over ~2 weeks? | **Training objective** |
| Ranking | Which coil should I look at first? | **How we use the prediction** |

We train regression (`Rel_Forward_2w`) because ranking is then free: sort the predicted numbers. Spearman correlation (below) measures whether that sort is any good.

### 4.3 The target: relative forward return

Absolute return `Forward_Return_2w` is:

```text
(Close[t + H] − Close[t]) / Close[t]
```

Daily primary H is **42 sessions** (`Rel_Forward_42d`); weekly primary H is **13 bars** (`Rel_Forward_13w`). Short `*_2w` labels stay in the CSV.

**Relative** return subtracts QQQ over the same dates:

```text
Rel_Forward_* = stock_return − QQQ_return
```

**Why relative?** A coil that rallies 4% while QQQ rallies 5% did not expand as a *leader*. The rubric is “compressed leaders vs QQQ.” Training on relative return matches that philosophy.

If the preferred relative column is missing, training walks the fallback list then `Forward_Return_2w`. Prefer regenerating the backtest so QQQ is in the mode raw silo.

Rows with a NaN target (not enough future bars at the end of the file) are dropped. That is correct, not data loss.

### 4.4 Why only `Is_New_Coil == True`

A coil can stay valid for several bars. Those extra rows are the **same episode** aging (`Coil_Age_Bars` 2, 3, …). Training on every bar would overweight long-lived coils and leak overlapping forward windows. One row per episode start is the unit of learning.

### 4.5 Features (X)

Score is a **linear mix** of pillars. Grade is a **bin** of Score. Feeding Score+Grade plus pillars teaches the tree to copy the rubric, not to find residual alpha. Score still **filters** rows (hard gates + ≥70) and is the live fallback sort.

`FEATURE_COLS` is pillars **plus raw** measurements: volume contraction ratio, MACD spread/ATR, coil width ATR + percentile, distances to 63/126/252-bar highs, coil OBV slope / up-volume / volume trend, RSI + healthy flag, EMA distances, ATR_Pct, distance to pivot, `MACD_Crossed`. `MACD_Cross` and `Fib_Bonus` are not tree features.

Trees split on these numbers. They do **not** see ticker names, so they cannot memorize “NVDA always wins” unless NVDA’s *geometry* is distinctive.

### 4.6 Leakage columns (never in X)

Dropped if present: `Stock Entry`, `Stock Stop`, `Target 1`, `Target 2`, `Outcome`, `Exit Date`, `Exit Price`, `R Multiple`, `Target_Label`, `Target_R_Mult`.

The Cobra backtest is an **expansion study**, not a fill simulator, so many of those may already be absent. The drop is defensive.

### 4.7 Temporal split and embargo (time-series 101)

Random K-fold **shuffles time**. A 2025 coil can train a model that then “predicts” 2024. That is cheating.

This project cuts on **`Signal Date`**:

```text
max_date  = last signal in the CSV
test      = last 26 weeks
val       = 26 weeks before test
train     = everything before val
```

**Embargo = 2 weeks** (same as the label horizon). A train signal in the last two weeks of the train window has a `y` that is computed from prices **inside** the validation window. Without the embargo, the model would train on labels that peek into val.

```text
train:  Signal Date  <  val_start − 2 weeks
val:    val_start  ≤  Signal Date  <  test_start − 2 weeks
test:   Signal Date  ≥  test_start
```

You need enough history that **train, val, and test are all non-empty**. A 3-month CSV will fail. Full 5y daily ingest is the intended path.

### 4.8 Sample weights: inverse ATR

High-ATR names have noisier 2-week returns. If you weight *by* `ATR_Pct`, the loss is dominated by lottery tickets. This trainer uses:

```text
weight = 1 / ATR_Pct
```

(with non-finite weights replaced by the **train median**). Quiet large-caps still matter. This is **not** inverse-volatility *portfolio* weighting; it only changes how much each row pulls the trees during fit.

### 4.9 Gradient boosting (XGBoost and LightGBM)

Both algorithms build **an ensemble of shallow decision trees**. Each new tree fits the **residual errors** of the previous trees (boosting). Shared knobs:

| Param | Value | Intuition |
| ----- | ----- | --------- |
| `max_depth` | 4 | Shallow trees → less memorization of individual names |
| `learning_rate` | 0.01 | Slow learning; needs more trees |
| `n_estimators` | 400 | Max trees; early stopping usually cuts this |
| `subsample` | 0.8 | Each tree sees 80% of rows (bagging) |
| `colsample_bytree` | 0.8 | Each tree sees 80% of features |
| `bagging_freq` | 1 (LightGBM) | Actually apply row bagging |
| `early_stopping_rounds` | 40 | Stop if val MAE does not improve |
| `random_state` | 42 | Reproducible splits inside the trees |

**Why two models?** If XGB and LGB agree on which features matter and both have similar Spearman, the pattern is less likely to be a library quirk.

**Why MAE (L1) not MSE (L2)?** Squared error explodes on a few +80% names. MAE (`reg:absoluteerror` / `regression_l1`) is the typical 2-week miss in return space and matches how we care about *typical* coils, not one outlier.

Missing feature values stay NaN. Both libraries split around missing values natively. There is no median imputation.

### 4.10 Metrics: MAE, RMSE, Spearman

Printed for **validation** (tuning / early stop) and **test** (honest holdout).

| Metric | Formula idea | Use here |
| ------ | ------------ | -------- |
| **MAE** | average `|pred − y|` | Primary fit; same family as the training loss |
| **RMSE** | sqrt of average squared error | Tail detector; often ugly on test |
| **Spearman** | rank correlation of `pred` vs `y` | **Production metric** — this is what `ML_Rank` cares about |

**Spearman vs Pearson.** Pearson asks “are the *magnitudes* linear?” Spearman asks “did we put the names in the right *order*?” We only sort; we do not size positions from `ML_Pred_Return`. Spearman is the right question.

**How to read Spearman**

| Spearman (test) | Interpretation |
| --------------- | -------------- |
| ≥ 0.10 and stable vs val | Weak but usable ranking edge |
| ~ 0 | Shuffle; ignore ML, use Score |
| Negative | Inverse skill; **do not deploy** — delete or don’t copy artifacts into the live silo |

Test RMSE ≫ val RMSE is common when 2025–2026 contains a handful of violent movers. Compare **MAE and Spearman**, not RMSE alone.

### 4.11 Inference: no averaging

`predict_returns` uses **one** promoted horizon’s XGB probability. It does **not** load LightGBM and does **not** average 10d/21d/42d. If the index says `production_model: none` for every horizon, `ML_Pred_Return` / `ML_Rank` stay null and the scanner keeps Score order.

`resolve_model_paths` still prefers a legacy `coiled_cobra_xgb_21d.json` filename; it is unused by the attach path — delete before anything calls it (`CodeReview.MD` I3).

The planner (`trade_planner.py`) may still multiply `ML_Pred_Return` as if it were an expected return (I4). Do not enable live ML until that is fixed.

---

## 5. Step-by-step: train a daily model (happy path)

Do this from the **host**. Daily is default; omit `weekly`.

### Step 0 — Container is up

```bash
docker ps --filter name=finance_vibe
```

If it is not running:

```bash
docker compose up -d
```

### Step 1 — Universe

```bash
docker exec -w /app finance_vibe python src/finance_vibe/ticker_provider.py
```

Writes `/app/data/active_tickers.csv` (on the host: `/mnt/fast/finance-vibe-data/active_tickers.csv`).

### Step 2 — Daily OHLCV

Prefer ingest **without** `run_vibe.py` so you do not wipe raw files unless you intend to.

```bash
docker exec -w /app finance_vibe python src/finance_vibe/data_ingestor.py
```

Confirm QQQ (required for RS and `Rel_Forward_*`):

```bash
docker exec finance_vibe ls /app/data/raw/daily/QQQ_5y_1d.csv
```

### Step 3 — Optional smoke backtest (minutes, not hours)

Proves the loop before a full-universe run:

```bash
docker exec -w /app finance_vibe python src/finance_vibe/coiled_cobra_backtest.py \
  --backtest --tickers SPY,QQQ,IWM
```

You should see a CSV under `/app/data/logs/daily/`. Smoke files are **too small** for a real temporal split. Use them only to debug paths.

### Step 4 — Full label job (the slow step)

```bash
docker exec -w /app finance_vibe python src/finance_vibe/coiled_cobra_backtest.py --backtest
```

Output:

```text
/app/data/logs/daily/coiled_cobra_backtest_trades_YYYY-MM-DD.csv
```

List it:

```bash
docker exec finance_vibe ls -lt /app/data/logs/daily/coiled_cobra_backtest_trades_*.csv
```

Note the **exact filename** (today’s date). Pass that path in the next step. Do not rely on the old example name `coiled_cobra_backtest_trades_2026-07-17.csv`.

Sanity-check inside the container:

```bash
docker exec -w /app finance_vibe python - <<'PY'
import pandas as pd
from pathlib import Path
p = sorted(Path("/app/data/logs/daily").glob("coiled_cobra_backtest_trades_*.csv"))[-1]
df = pd.read_csv(p)
print(p.name, "rows", len(df))
print("new coils", df["Is_New_Coil"].astype(str).str.lower().isin(["true","1"]).sum() if "Is_New_Coil" in df.columns else "missing")
print("Rel_Forward_42d non-null", df["Rel_Forward_42d"].notna().sum() if "Rel_Forward_42d" in df.columns else "missing")
print("Signal Date", df["Signal Date"].min(), "→", df["Signal Date"].max())
print("pillars ok", all(c in df.columns for c in [
    "Volume_Shelf","MACD_Compression","Structure","RS_Score","Coil_Width","Proximity_Highs",
    "Pct_From_EMA20","Pct_From_EMA50","ATR_Pct","Dist_High_63_Pct"]))
PY
```

You want **hundreds** of new-coil rows with non-null primary Rel_Forward spanning **well over a year**.

### Step 5 — Train (write artifacts next to the CSV)

Replace the date with the file from Step 4. **Always pass `--csv`.** Do not omit it: `SOURCE_FILENAME` still names a missing `..._2026-07-17.csv` and newest-mtime can pick `_Large.csv`.

```bash
docker exec -w /app finance_vibe python -m finance_vibe.coiled_cobra_ml_training \
  --csv /app/data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv \
  --artifacts-dir /app/data/logs/daily
```

**Success on stdout** looks like:

- New-coil rows only (~21k on the 2026-08-24 native CSV)
- Per-horizon `Win_*` class balance (~54–57% positive)
- Walk-forward folds; some skipped as stumps (`best_iteration < 3`)
- Table: ML vs Score vs random vs population, `top_10pct` within fold
- `Promote: NO` and `production_model: none` until the gate actually passes

**Success on disk:**

```bash
docker exec finance_vibe ls -l \
  /app/data/logs/daily/coiled_cobra_xgb_10d.json \
  /app/data/logs/daily/coiled_cobra_xgb_21d.json \
  /app/data/logs/daily/coiled_cobra_xgb_42d.json \
  /app/data/logs/daily/coiled_cobra_ml_model_metadata.json \
  /app/data/logs/daily/coiled_cobra_ml_oos_10d.csv
```

Index must show `feature_columns` length **26**, `prob_column` `ML_Prob_Win_*`, `promoted: false` unless the gate passed. **Do not trust JSON `target_column` until I5 is fixed** (it currently stores the last research Hit name). Resume notes: `docs/CoiledCobraML-Handoff.md`.

### Step 6 — Deploy is already done

Live scan reads `/app/data/logs/daily/`. No copy step if you trained into that folder.

Trigger a scan (ingest optional if raw is fresh):

```bash
docker exec -w /app finance_vibe python src/finance_vibe/coiled_cobra.py
```

Or the full pipeline (this **clears** `data/raw/daily/` unless you add your own keep-raw equivalent from compose):

```bash
docker exec -w /app finance_vibe python src/finance_vibe/run_vibe.py --keep-raw
```

In `/app/data/logs/daily/coiled_cobra_setups_YYYY-MM-DD.csv`, `ML_Prob_Win_*` stay **null** while `production_model` is `none`. That is correct. Non-null `ML_Rank` from a leftover `coiled_cobra_xgb_model.json` is the old regression path — do not treat it as this trainer.

---

## 6. Weekly confirmation model (optional)

Same loop; pass `weekly` and write into the weekly silo.

```bash
docker exec -w /app finance_vibe python src/finance_vibe/data_ingestor.py weekly
docker exec -w /app finance_vibe python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
docker exec -w /app finance_vibe python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv /app/data/logs/weekly/coiled_cobra_backtest_trades_YYYY-MM-DD.csv \
  --artifacts-dir /app/data/logs/weekly
```

Live weekly scan:

```bash
docker exec -w /app finance_vibe python src/finance_vibe/coiled_cobra.py weekly
```

Never copy weekly `*_model.json` into `logs/daily/`.

---

## 7. How to teach yourself from a finished run

1. **Metadata first** — confirm schema and Spearman before looking at plots.
2. **Importance PNG** — which pillars the trees used. If `ATR_Pct` dominates both models, the ranker may be sorting by volatility, not coil quality. That is a research finding, not a bug.
3. **Compare ML_Rank vs Score** on `trade_plan_*.csv`:
   - Same names at the top → model agrees with the rubric.
   - High Score, poor ML rank → historically similar geometry did not beat QQQ over 2 weeks. Investigate regime, RS, width.
   - Low Score, high ML rank → **do not override the hard gates**. The coil already failed compression / structure / RS.
4. **Stability** — if val Spearman is 0.2 and test is −0.05, you overfit the val window. Do not promote.

---

## 8. Promotion checklist (automated + human)

The trainer writes `production_model: xgb` only if the walk-forward gate passes. Do **not** hand-edit that field.

Human checklist before believing a `true`:

- [ ] `--csv` was the **native** 114-column `..._2026-08-24.csv` (or a later full `--backtest`), not `_Large.csv` / smoke tickers
- [ ] `feature_columns` length **26** (not the old Score+Fib six)
- [ ] Fraction cuts used **per-fold** selection (`schema_version` ≥ 3)
- [ ] Gate metrics are avg / median / win rate vs Score **and** random **and** population (no hit_rate)
- [ ] `best_iteration >= 3` on the saved booster
- [ ] Fold pass rate ≥ 0.60
- [ ] You have **not** loosened the gate to force a ship

Do **not** promote because AUC looks good, because Hit rate beat Score, or because a 6-feature importance PNG looks consistent. Current honest result: **Win XGB does not beat random.** See `docs/CoiledCobraML-Handoff.md`.

---

## 9. Retrain cadence

There is no scheduler. Practical policy:

| Event | Action |
| ----- | ------ |
| Rubric or `FEATURE_COLS` change | Mandatory retrain after a new `--backtest` |
| Large universe or ingest period change | Retrain |
| Live ranks look stuck vs Score for weeks | Retrain on latest trades CSV |
| Routine | After a meaningful daily backtest refresh (e.g. weekly or monthly), not every live scan |

Keep the trades CSV that produced the current model next to the artifacts so you can audit `Signal Date` max.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `FileNotFoundError` for trades CSV | Backtest not run, or auto-discover still hunting `...2026-07-17.csv` | Run `--backtest`; pass `--csv` with the dated file under `/app/data/logs/daily/` |
| Empty train/val/test | Smoke ticker set or short history | Full ingest + full `--backtest`; inspect date min/max |
| Missing pillar columns | CSV from pre-v2.1 backtest | Re-run `coiled_cobra_backtest.py --backtest` with current code |
| `import xgboost` fails | Image built before deps were added | `docker compose up -d --build` |
| Training script looks stale | Container has old `/app/src` | Rebuild, or `docker compose` with a `./src` bind mount if you use one |
| Live `ML_*` all null | Models in weekly only, or 6-feature schema skip | Train into `logs/daily/`; check metadata `feature_columns` |
| Huge Test RMSE, OK MAE | Heavy tails | Expected; use Spearman |
| QQQ missing / Rel_Forward null | `QQQ_5y_1d.csv` not ingested | Re-run `data_ingestor.py`; confirm static tickers include QQQ |

Interactive debug:

```bash
docker exec -it finance_vibe bash
ls -l /app/data/logs/daily/
python src/finance_vibe/coiled_cobra_ml_training.py --help
```

---

## 11. Module map

| Module | MLOps role |
| ------ | ---------- |
| `ticker_provider.py` | Universe |
| `data_ingestor.py` | Feature-time OHLCV (and QQQ) |
| `coiled_cobra_backtest.py` | **Label store** — walk-forward coils + forward returns |
| `coiled_cobra_ml_training.py` | **Trainer** — split, fit, metrics, serialize |
| `ml_ranker.py` | **Inference** — load promoted horizon XGB only; never average LGB |
| `coiled_cobra.py` | Live scan; calls `attach_ml_ranks` |
| `trade_planner.py` | Still may treat `ML_Pred_Return` as expected return (I4 — do not enable live ML until fixed) |
| `run_vibe.py` | Orchestrates live path; **does not train** |
| `pipeline_backtest.py` | Quality-swing study — **not** an ML input |

---

## 12. Glossary

| Term | Meaning in this repo |
| ---- | -------------------- |
| **Artifact** | Saved model or metadata file in `data/logs/{mode}/` |
| **Booster** | The fitted tree ensemble (XGB JSON or LGB text) |
| **Early stopping** | Halt adding trees when val MAE stalls for 40 rounds |
| **Embargo** | Gap so train labels do not overlap val/test prices |
| **Episode** | One coil from first valid bar (`Is_New_Coil`) until it fails |
| **Fail-soft** | Missing/stale model → Score sort, pipeline continues |
| **Leakage** | Using information from after the signal in X |
| **OOS** | Out of sample = the test window |
| **Pillar** | One rubric component (e.g. Structure 0–20) |
| **Silo** | Isolated folder: daily vs weekly vs high_beta logs |
| **Soft rank** | Ordering aid, not a gate |
| **Spearman** | Rank correlation; production metric for ML_Rank |
| **Temporal split** | Train/val/test cut by date, not shuffled rows |

---

## 13. Quick reference card

```bash
# 1) Labels (daily, full universe) — slow
docker exec finance_vibe python -m finance_vibe.coiled_cobra_backtest daily --backtest

# 2) Train — pin the native CSV
docker exec finance_vibe python -m finance_vibe.coiled_cobra_ml_training \
  --csv /app/data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv \
  --artifacts-dir /app/data/logs/daily

# 3) Confirm three boosters + index + OOS
docker exec finance_vibe ls /app/data/logs/daily/coiled_cobra_xgb_*d.json \
  /app/data/logs/daily/coiled_cobra_ml_model_metadata.json \
  /app/data/logs/daily/coiled_cobra_ml_oos_*d.csv
```

Training is “done” when those files exist. **Serving** is done only when the index has `production_model: xgb`. Today it does not; live scans correctly stay on Score. Full pickup: `docs/CoiledCobraML-Handoff.md`.
