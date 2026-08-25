# Coiled Cobra MLOps Guide

**Audience:** operators who train, judge, and (eventually) serve ranking models in Docker.

**Primary runtime:** container `finance_vibe`, workdir `/app`, `PYTHONPATH=/app/src`, data volume `/app/data`.

**Companion docs**

| Document | Role |
| -------- | ---- |
| `docs/CoiledCobraML-Handoff.md` | **Resume here** — locked decisions, scoreboard, next work |
| `CodeReview.MD` | Audit: what was fixed 2026-08-24 vs still open |
| `CoiledCobraML.md` | Feature / label / artifact contract |
| `Coiled Cobra Rubric .MD` | Live scorecard — what a coil *is* |
| `BacktestAndBackfill.md` | How historical labels are produced |
| This file | How to **run** the loop in the container |

`run_vibe.py` never trains. Training is an offline job. Live ranks attach **only** when a horizon’s metadata has `production_model: xgb`. As of 2026-08-24 that is **none** for 10d / 21d / 42d — scans correctly stay on rubric **Score**.

---

## Current production state (2026-08-24)

| Item | Value |
| ---- | ----- |
| Task | Three `XGBClassifier`s on `Win_10d` / `Win_21d` / `Win_42d` |
| Features | 26 pillars + geometry; **no** Score, Grade, or Fib |
| Labels CSV | `data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv` (always pass `--csv`) |
| Pool | 80,052 signals / **21,961** new coils / 413 symbols |
| Artifacts | `coiled_cobra_xgb_{10,21,42}d.json` + per-horizon metadata + index + OOS CSVs |
| `schema_version` | 5 |
| Live | `production_model: none` — `ML_Prob_Win_*` stay null |
| Do not | Retrain the same 26 features on Win; loosen the gate; average XGB+LGB; blend horizons |

JSON `target_column` in horizon metadata is **wrong** (last research `Hit_*` name). Models were trained on `Win_*`. See handoff I5.

---

## 1. What this system is (and is not)

### What you are building

A **multi-horizon ranking layer** on valid **new-coil** setups:

1. Features known **on the signal bar** (26 columns).
2. Separate probabilities that **close is up** at 10 / 21 / 42 sessions (`Win_*`). MFE `Hit_*` is research-only.
3. Walk-forward OOS: top 10% **inside each fold**, then pool. A horizon ships only if that cut beats **Score, random, and the population** on average forward return, median forward return, and win rate, and ≥60% of folds pass. **Hit rate is not in the gate.**

The rubric Score remains the quality filter (≥70 + hard gates) and the live sort until a model is promoted.

### What you are not building

- Not a live order router
- Not options / LEAPS P&L
- Not a hard trade / don’t-trade gate
- Not MLflow / Airflow / auto-retrain
- Not the old 6-feature `Forward_Return_2w` XGB+LGB regression (`coiled_cobra_xgb_model.json`)

### Daily vs weekly (do not confuse with 10d/21d/42d)

| | Daily (ML primary) | Weekly |
| - | ------------------ | ------ |
| Raw | `/app/data/raw/daily/` (`10y`, `1d`) e.g. `QQQ_10y_1d.csv` | `/app/data/raw/weekly/` (`10y`, `1wk`) |
| Logs | `/app/data/logs/daily/` | `/app/data/logs/weekly/` |
| Coil window | 30 bars | 8 bars |
| Live scan | default | `python -m finance_vibe.coiled_cobra weekly` |
| This trainer | **Yes** — 10d/21d/42d session models | **No** — `main()` raises if the CSV looks weekly |

A weekly booster must never rank a daily scan. `ml_ranker` only looks in the active mode’s log folder.

---

## 2. Docker environment

Compose (`docker-compose.yml`) mounts **`./data` → `/app/data` only**. Source, tests, and markdown are **baked into the image**. Host edits to `src/` do not appear in the container until you rebuild or `docker cp`.

| Fact | Value |
| ---- | ----- |
| Container | `finance_vibe` |
| Image | `finance-vibe-finance-vibe` |
| Workdir | `/app` |
| `PYTHONPATH` | `/app/src` |
| Data (this machine) | host `C:\Apps\Projects\finance-vibe\data` → `/app/data` |
| Dashboard | port **5000** |
| TZ | `America/New_York` |

Preferred job pattern (`-m` so `from finance_vibe import …` always works):

```bash
docker exec -w /app finance_vibe python -m finance_vibe.<module> [args]
```

After editing Python on the host:

```bash
docker compose up -d --build
# or copy one file:
docker cp src/finance_vibe/coiled_cobra_ml_training.py finance_vibe:/app/src/finance_vibe/coiled_cobra_ml_training.py
```

Doc edits (this file, `Learn.md`, …) also need a rebuild to show on `/learn` inside the image.

Confirm packages:

```bash
docker exec finance_vibe python -c "import xgboost, lightgbm, sklearn, matplotlib; print('ok')"
```

---

## 3. Mental model: the loop

```
  [1] Universe          ticker_provider.py
          │             STATIC_TICKERS + screener → data/active_tickers.csv
          │             (survivorship: today’s names + decade winners — L1)
  [2] OHLCV             data_ingestor.py  →  /app/data/raw/daily/*_10y_1d.csv
          │
  [3] Label job         coiled_cobra_backtest daily --backtest
          │             native Forward_Return_* / Max_Return_* / Win_* / Hit_*
          ▼
     trades CSV         coiled_cobra_backtest_trades_YYYY-MM-DD.csv
          │
  [4] Train             coiled_cobra_ml_training --csv <that file>
          │             walk-forward, 3× XGBClassifier on Win_*
          ▼
     artifacts          xgb_{10,21,42}d.json, metadata_*, index, oos_*.csv
          │
  [5] Serve (soft)      coiled_cobra.py → ml_ranker.attach_ml_ranks
          │             only if production_model == xgb
          ▼
     ML_Prob_Win_*      else null; book stays Score-sorted
```

Steps 1–4 are batch. Step 5 is the next live scan. “Deploy” means files in `logs/daily/` **and** a passing promotion flag — not merely that json exists.

---

## 4. Concepts (current trainer)

### 4.1 Supervised ranking, not predicted dollars

`X` = signal-time features. `y` = `Win_*` (1 if `Forward_Return_Xd > 0`). The booster emits a **probability**. We **sort** on it. We do **not** size positions from it. Trading quality is judged on realized **close-to-close** avg / median / win rate of the selected names — not AUC, not MFE hit rate.

Former regression on `Rel_Forward_2w` / `Forward_Return_2w` is retired. Spearman / MAE / RMSE are not the promotion metrics.

### 4.2 Why `Win_*` and not `Hit_*`

`Hit_*` is MFE vs the signal close (any high ≥ +10/15/25% inside the window). That is nearly a volatility statistic. Hit-trained models raised hit rate while **median and win rate collapsed**. Primary label is now close-up (`Win_*`). `Hit_*` stays in the CSV and as `research_targets`.

Trading columns used in the gate: `Forward_Return_10d` / `_21d` / `_42d` (10d ≡ legacy `Forward_Return_2w` on this daily frame).

### 4.3 Why only `Is_New_Coil == True`

Continuation bars are the same episode. Training every bar overweight long coils and overlaps label windows.

### 4.4 Features (`FEATURE_COLS`, 26)

Pillars: `Volume_Shelf`, `MACD_Compression`, `Structure`, `RS_Score`, `Coil_Width`, `Proximity_Highs`.

Raw: contraction ratio, MACD spread/ATR, width ATR + percentile, dist to 63/126/252 highs (% and ATR), OBV slope, up-volume / volume trend, RSI + healthy, `%` from EMA20/50, `ATR_Pct`, distance to pivot, `MACD_Crossed`.

**Never in X:** Score, Grade, Fib %, ticker, dates, any `Forward_*` / `Win_*` / `Hit_*` / `MAE_*` / `ML_*` / fills / R-multiple.

Score still **filters** the backtest sample and is the live baseline ranker the model must beat.

If `ATR_Pct` dominates gain (~0.21–0.25 today), the trees are mostly sorting volatility. That is a research finding, not a ship signal.

### 4.5 Walk-forward + embargo (not one 2024/2025 cut)

Random K-fold shuffles time. This trainer tiles **expanding** folds: ~26w validation, ~26w test, embargo **2 / 5 / 9 weeks** for 10d / 21d / 42d so train labels do not overlap val/test prices.

Folds with `n_train < 1000` or `best_iteration < 3` are **skipped** (stumps at `learning_rate=0.01` are a constant ≈ base rate).

**C1:** `top_10pct` (and other fraction cuts) are taken **inside each fold**, then pooled. Uncalibrated probabilities from different fold models must not be ranked against each other.

**C2:** per-date cuts are `top 5 / 3 / 1` per signal date. Daily breadth is ~8–9 new coils; “top 10 per date” was almost the population.

### 4.6 Sample weights

**Uniform.** `USE_INVERSE_ATR_WEIGHTS = False`. Inverse-ATR weighting is a future experiment, not current.

### 4.7 Boosting knobs

One `XGBClassifier` per horizon. LightGBM is **not** fitted and **not** loaded at inference.

| Param | Value |
| ----- | ----- |
| `max_depth` | 4 |
| `learning_rate` | 0.01 |
| `n_estimators` | 400 (early stop 40 rounds on **val AUC**) |
| `subsample` / `colsample_bytree` | 0.8 |
| `min_child_weight` | 16 |
| `reg_lambda` | 2.0 |
| `random_state` | 42 |

Logistic regression is an OOS **baseline** only. It often beats this XGB on top-decile **mean** (underfit). Both still fail the trading gate. Do not hunt hyperparameters until features or the label change.

Missing features stay NaN; trees split around them. Live inference should eventually **fail loud** if a pillar is absent (I1 — still open).

### 4.8 What “beats Score” means

Score on this universe is often ≈ **random** and can lose to the **untouched population**. The gate therefore requires the model’s top decile to beat all three baselines on avg, median, and win rate. Fold pass rate ≥ 0.60.

Current Win result: **does not beat random or population on the mean.** Medians stay positive (the Hit lottery is gone). Fold pass is ~0–9%.

### 4.9 Inference

`attach_horizon_probabilities` writes `ML_Prob_Win_*` only for horizons with `production_model == "xgb"`.

`predict_returns` / `ML_Pred_Return` / `ML_Rank` use **one** promoted horizon (priority 10d → 21d → 42d). No LGB average. No horizon blend.

`trade_planner.py` may still treat `ML_Pred_Return` as expected return and boost it (I4). Do not enable live ML until that is fixed.

`resolve_model_paths` still prefers a legacy filename; unused by the attach path — delete before anything calls it (I3).

---

## 5. Train a daily model (happy path)

From the **host**. Daily is default.

### Step 0 — Container up, code current

```bash
docker ps --filter name=finance_vibe
```

If Python on the host changed since the last image: `docker compose up -d --build`.

### Step 1 — Universe

```bash
docker exec -w /app finance_vibe python -m finance_vibe.ticker_provider
```

Writes `/app/data/active_tickers.csv`. Merges manifest + `STATIC_TICKERS` + Yahoo most-actives (cap 1000). Historical studies on this list have **survivorship** (L1).

### Step 2 — Daily OHLCV

Prefer ingest without `run_vibe.py` so you do not wipe raw unless you intend to.

```bash
docker exec -w /app finance_vibe python -m finance_vibe.data_ingestor
```

Confirm QQQ (RS + relative forwards in the backtest):

```bash
docker exec finance_vibe ls /app/data/raw/daily/QQQ_10y_1d.csv
```

(Not `QQQ_5y_1d.csv` — period is **10y**.)

### Step 3 — Optional smoke backtest

```bash
docker exec -w /app finance_vibe python -m finance_vibe.coiled_cobra_backtest daily --backtest --tickers SPY,QQQ,IWM
```

Smoke CSVs are **too small** (`MIN_TRAIN_ROWS=1000`). Use only to debug paths. Do not train production models on them.

### Step 4 — Full label job (slow; ~50+ min)

```bash
docker exec -w /app finance_vibe python -m finance_vibe.coiled_cobra_backtest daily --backtest
```

Output: `/app/data/logs/daily/coiled_cobra_backtest_trades_YYYY-MM-DD.csv` (native 10d/21d/42d forwards, MFE, Win, Hit — 114 columns on the 2026-08-24 run).

```bash
docker exec finance_vibe ls -lt /app/data/logs/daily/coiled_cobra_backtest_trades_*.csv
```

Sanity-check (pin the file you will train on; **do not** take newest mtime if `_Large.csv` / `_small.csv` exist):

```bash
docker exec -w /app finance_vibe python -c "
import pandas as pd
p = '/app/data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv'
df = pd.read_csv(p)
print(p, 'rows', len(df), 'cols', len(df.columns))
print('new coils', (df['Is_New_Coil']==True).sum())
print('Win_10d rate', df.loc[df['Is_New_Coil']==True, 'Win_10d'].mean())
print('dates', df['Signal Date'].min(), '->', df['Signal Date'].max())
print('Win_10d' in df.columns, 'Max_Return_42d' in df.columns)
"
```

You want ~tens of thousands of new-coil rows, native `Win_*` / `Max_Return_*`, multi-year `Signal Date`.

### Step 5 — Train (always `--csv`)

`SOURCE_FILENAME` still names a missing `coiled_cobra_backtest_trades_2026-07-17.csv`. Omitting `--csv` falls through to newest mtime and can pick `_Large.csv` (enriched labels, not the tested backtest path).

```bash
docker exec -w /app finance_vibe python -m finance_vibe.coiled_cobra_ml_training \
  --csv /app/data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv \
  --artifacts-dir /app/data/logs/daily
```

Optional: `--horizons 10d,21d,42d` (default). Training only 42d leaves stale 10d/21d artifacts serving beside it (I2) — avoid until SHA checks exist.

**Stdout success**

- New-coil only (~21k on the pinned CSV)
- `Win_*` balance ~54–57% positive
- Folds skipped for stumps (`best_iteration < 3`)
- Table: ML vs Score vs random vs population, `top_10pct` within fold
- `Promote: NO` until the gate actually passes

**Disk success**

```bash
docker exec finance_vibe ls -l \
  /app/data/logs/daily/coiled_cobra_xgb_10d.json \
  /app/data/logs/daily/coiled_cobra_xgb_21d.json \
  /app/data/logs/daily/coiled_cobra_xgb_42d.json \
  /app/data/logs/daily/coiled_cobra_ml_metadata_10d.json \
  /app/data/logs/daily/coiled_cobra_ml_model_metadata.json \
  /app/data/logs/daily/coiled_cobra_ml_oos_10d.csv
```

Index: `feature_columns` length **26**, `prob_column` `ML_Prob_Win_*`, `promoted: false`. Do not trust per-file `target_column` until I5 is fixed.

### Step 6 — Live scan does not “turn on” ML

Files in `logs/daily/` are visible to the scanner immediately. Probabilities stay **null** while `production_model` is `none`. That is correct.

```bash
docker exec -w /app finance_vibe python -m finance_vibe.coiled_cobra
```

Full pipeline (clears `data/raw/daily/` unless `--keep-raw`):

```bash
docker exec -w /app finance_vibe python -m finance_vibe.run_vibe --keep-raw
```

A leftover `coiled_cobra_xgb_model.json` is the **old regression** stack. Do not treat its `ML_Pred_Return` as this trainer.

### Tests (image must contain current `tests/`)

```bash
docker compose run --rm --no-deps finance-vibe python -m pytest tests/ -q
```

---

## 6. Weekly silo

Weekly **scan** and weekly **backtest** still exist for confirmation vs daily coils.

Weekly **training with this script is unsupported** (`RuntimeError` on a weekly CSV). Do not copy daily `coiled_cobra_xgb_*d.json` into `logs/weekly/` or the reverse.

```bash
docker exec -w /app finance_vibe python -m finance_vibe.data_ingestor weekly
docker exec -w /app finance_vibe python -m finance_vibe.coiled_cobra_backtest weekly --backtest
docker exec -w /app finance_vibe python -m finance_vibe.coiled_cobra weekly
```

---

## 7. How to read a finished run

1. **Handoff + promotion table** — avg / med / wr vs Score, random, population. `Blocked by` and fold pass rate. If ML loses to random, stop.
2. **Index JSON** — `production_model`, 26 features, `do_not_average_with_lightgbm`.
3. **`normalized_xgb_gain` in horizon metadata** — if `ATR_Pct` leads, you are ranking vol. There is no importance PNG in the current trainer.
4. **OOS CSV** `coiled_cobra_ml_oos_{horizon}.csv` — fold id stamped; use this to re-score cuts without retraining.
5. **Live `trade_plan_*.csv`** — with no promotion, ML columns are null. Do not override hard gates if a leftover ranker disagrees with Score.

Do not promote on AUC, Hit rate, or a 6-feature importance chart.

---

## 8. Promotion checklist

The trainer writes `production_model: xgb` only if the gate passes. **Do not hand-edit that field.**

- [ ] `--csv` is the native 114-column full-universe file, not `_Large.csv` / `_small.csv` / 3-ticker smoke
- [ ] Container had the **current** `src/` (rebuild or `docker cp`)
- [ ] 26 features; no Score/Fib
- [ ] `schema_version` ≥ 5; fraction cuts per fold
- [ ] Gate is avg + med + wr vs Score **and** random **and** population
- [ ] Saved `best_iteration >= 3`
- [ ] Fold pass ≥ 0.60
- [ ] Gate was not loosened to force a ship
- [ ] I4 / I1 not ignored if you actually intend to serve ranks

Current honest result: **Win XGB does not beat random.** Next work is new features or a risk-adjusted label — not another identical retrain. See `docs/CoiledCobraML-Handoff.md`.

---

## 9. Retrain cadence

No scheduler.

| Event | Action |
| ----- | ------ |
| `FEATURE_COLS` or rubric change | New `--backtest`, then retrain with `--csv` |
| New label definition (`Win_*` vs Hit vs risk-adjusted) | Retrain; keep the same promotion table |
| Same 26 features, same `Win_*` | **Do not** expect a different gate result |
| New full-universe CSV | Retrain only if you changed labels or features |
| Live scan | Never trains |

Keep the trades CSV that produced the artifacts next to them.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| Trades CSV not found | No backtest, or auto-discover wants `...-07-17.csv` | Run `--backtest`; always `--csv` |
| Picked `_Large.csv` / `_small.csv` | Newest-mtime fallback | Pass the native dated path |
| Training looks like Hit / old 6 features | Stale image | Rebuild or `docker cp` |
| `best_iteration < 3` skip / no save | Stump fold or final refit | Expected on noisy Win labels; C4 refuses shipping stumps |
| Weekly CSV `RuntimeError` | Trainer is daily-only | Use a daily trades file |
| Live `ML_Prob_Win_*` all null | `production_model: none` | **Expected.** Do not force-serve |
| Non-null `ML_Pred_Return` anyway | Legacy `xgb_model.json` | Ignore / quarantine the old files |
| JSON `target_column` is `Hit_*` | I5 loop-variable leak | Trust stdout + `Win_*`; fix the loop then rewrite metadata |
| QQQ missing | Not ingested | `data_ingestor`; file is `QQQ_10y_1d.csv` |
| `import xgboost` fails | Old image | `docker compose up -d --build` |
| pytest missing new tests | `tests/` not in image | Rebuild compose |

```bash
docker exec -it finance_vibe bash
ls -l /app/data/logs/daily/
python -m finance_vibe.coiled_cobra_ml_training --help
```

---

## 11. Module map

| Module | Role |
| ------ | ---- |
| `ticker_provider.py` | Universe |
| `data_ingestor.py` | OHLCV + QQQ |
| `coiled_cobra_backtest.py` | Label store (native forwards / MFE / Win / Hit) |
| `coiled_cobra_ml_training.py` | Walk-forward trainer; writes promotion flag |
| `ml_ranker.py` | Load **promoted** horizon XGB only |
| `coiled_cobra.py` | Live scan; `attach_ml_ranks` |
| `trade_planner.py` | Plan CSV; still may misuse `ML_Pred_Return` (I4) |
| `run_vibe.py` | Live orchestrator; **does not train** |
| `pipeline_backtest.py` | Quality-swing study — not an ML input |

---

## 12. Glossary

| Term | Meaning here |
| ---- | ------------ |
| **Artifact** | Model or metadata in `data/logs/daily/` |
| **Booster** | Saved XGB JSON (`coiled_cobra_xgb_{horizon}d.json`) |
| **Embargo** | Date gap so labels do not overlap the next split (2w/5w/9w) |
| **Episode** | One coil from `Is_New_Coil` until it fails |
| **Fail-soft** | No promoted model → Score sort, pipeline continues |
| **Fold** | One walk-forward train/val/test tile; has its own booster |
| **Hit** | MFE threshold — research label, not the gate |
| **Leakage** | After-signal information in X |
| **OOS** | Out-of-sample test windows, pooled after per-fold cuts |
| **Population** | All labeled new coils in that OOS pool (no ranker) |
| **Promote** | `production_model: xgb` written by the trainer, not by hand |
| **Win** | `Forward_Return_Xd > 0` — current train label |
| **Silo** | `daily` vs `weekly` vs `high_beta` folders |

---

## 13. Quick reference

```bash
# Labels (slow)
docker exec finance_vibe python -m finance_vibe.coiled_cobra_backtest daily --backtest

# Train — pin the native CSV
docker exec finance_vibe python -m finance_vibe.coiled_cobra_ml_training \
  --csv /app/data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv \
  --artifacts-dir /app/data/logs/daily

# Artifacts
docker exec finance_vibe ls /app/data/logs/daily/coiled_cobra_xgb_*d.json \
  /app/data/logs/daily/coiled_cobra_ml_model_metadata.json \
  /app/data/logs/daily/coiled_cobra_ml_oos_*d.csv

# Live scan (ML stays null until promoted)
docker exec finance_vibe python -m finance_vibe.coiled_cobra
```

Training is done when those files exist. **Serving** is done only when the index has `production_model: xgb`. Today it does not. Pickup: `docs/CoiledCobraML-Handoff.md`.
