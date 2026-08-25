# Coiled Cobra ML — resume here (2026-08-24)

Read this first after a break. Audit detail lives in **`CodeReview.MD`**. Feature contract: **`CoiledCobraML.md`**. Docker runbook: **`MLOps.md`**.

## Status in one line

**Do not promote** (OOS gate still fails). Live scans **do** sort by ML when `SERVE_ML_RANKER` is True. Three independent XGB classifiers exist (`schema_version` 5, `Win_*` labels). Index `production_model` is `"none"` for 10d / 21d / 42d — that flag is the gate, not the serve switch. Retraining the same 26 features on `Win_*` will not create an OOS edge.

## What we decided (do not reverse without new evidence)

| Decision | Why it is locked | Do not |
| -------- | ---------------- | ------ |
| Train only `Is_New_Coil=True` | Continuation bars are the same coil | Mix aged bars into X |
| 26 pillars; no `Score`, `Grade`, Fib % | Score is the baseline we must beat; Fib is off the live rubric | Put Score or Fib back into `FEATURE_COLS` |
| Separate 10d / 21d / 42d models | Horizons are different problems | Average horizons or XGB+LGB |
| Walk-forward + embargo (2w / 5w / 9w) | Labels overlap the next window | Random K-fold |
| Rank top-% **inside each fold** then pool | Uncalibrated probs are not comparable across folds (C1) | Pooled top-decile on concatenated OOS |
| Per-date cuts `(5, 3, 1)` | Daily breadth ~8–9 coils; top-10/20 was almost the population (C2) | `top_20_per_date` as a headline |
| `MIN_TRAIN_ROWS=1000`, `MIN_BEST_ITERATION=3` | Stump folds polluted OOS and shipped artifacts (C3/C4) | Save `best_iteration < 3` |
| Promotion: beat **Score and random and population** on **avg, median, win rate**; ≥60% folds | Score ≈ random; Hit-rate is the model’s own objective (E1/E2/T2) | Promote on AUC, hit rate, or mean alone |
| Primary label = `Win_*` (`Forward_Return_Xd > 0`) | `Hit_*` (MFE) is a vol proxy: tighter cuts → higher hit, **negative median** (T1) | Retrain Hit as the production label |
| `Hit_*` stays research-only | Still useful as a diagnostic, not a gate | Put hit_rate back in `PROMOTION_METRICS` |
| Native backtest CSV only | Old `_Large.csv` synthesized MFE via enrich (T3/T4) | Newest-by-mtime without `--csv` |
| Judge trading outcomes, not AUC | Ranking for a book, not a classifier leaderboard | Promote on ROC |

## Artifacts on disk (daily silo)

Pinned labels: **`data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv`**
(80,052 signals / **21,961 new coils** / 413 symbols / 114 columns). Always pass `--csv` to this file until `SOURCE_FILENAME` is updated.

| File | Role |
| ---- | ---- |
| `coiled_cobra_xgb_{10,21,42}d.json` | Boosters (saved trees ~56 / 18 / 113 on the Win run) |
| `coiled_cobra_ml_metadata_{10,21,42}d.json` | Walk-forward + promotion; `schema_version` 5 |
| `coiled_cobra_ml_model_metadata.json` | Index: all `promoted: false`, `production_model: none`, probs `ML_Prob_Win_*` |
| `coiled_cobra_ml_oos_{10,21,42}d.csv` | Fold-stamped OOS predictions for re-scoring |

**Known metadata bug (do not trust `target_column` in those JSON files):** `evaluate_research_targets` loops `for target in research_targets`, which **rebinds** the outer `target` in `train_horizon`. On disk:

| Horizon | JSON `target_column` (wrong) | Actual train label | `prob_column` |
| ------- | ---------------------------- | ------------------ | ------------- |
| 10d | `Hit_15Pct_10d` | `Win_10d` | `ML_Prob_Win_10d` |
| 21d | `Hit_20Pct_21d` | `Win_21d` | `ML_Prob_Win_21d` |
| 42d | `Hit_50Pct_42d` | `Win_42d` | `ML_Prob_Win_42d` |

Use **training stdout**, `label_col(spec)`, and the OOS `Win_*` columns as truth. Fix: rename the loop variable in `evaluate_research_targets` and rewrite metadata. Until then, `ml_ranker` may also read the wrong `target_column` for schema checks.

Stale files that can hijack auto-discover: `*_Large.csv`, `*_small.csv`, leftover `coiled_cobra_xgb_model.json` (6-feature regression). `SOURCE_FILENAME` still says `..._2026-07-17.csv` (missing → newest mtime).

## Latest Win scoreboard (top 10% **within fold**)

| Horizon | ML avg | Score | Rand | Pop | ML med | Score med | ML wr | Score wr | Pop wr | Fold pass | Promote | Blocked |
| ------- | -----: | ----: | ---: | --: | -----: | --------: | ----: | -------: | -----: | --------: | ------- | ------- |
| 10d | +0.38% | +0.54% | +0.68% | +0.77% | +0.36% | +0.61% | 0.541 | 0.544 | 0.527 | 9.1% | NO | avg, med, wr |
| 21d | +1.40% | +1.75% | +2.25% | +1.78% | +1.23% | +1.17% | 0.583 | 0.555 | 0.559 | 9.1% | NO | avg, med |
| 42d | +1.14% | +4.84% | +5.52% | +5.37% | +0.97% | +2.56% | 0.551 | 0.566 | 0.573 | 0.0% | NO | avg, med, wr |

Hit-trained models had a **mean** edge and a **negative median**. That edge was volatility (`ATR_Pct` top gain). Win training killed the lottery; it did **not** beat random or population on the mean. `ATR_Pct` is still ~0.21–0.25 gain.

## Old “production” 6-feature chart vs this stack

The PNG titled **Forward_Return_2w** (XGB + LightGBM, 6 bars) is a **different product**, not a prior checkpoint of this trainer.

| | Legacy chart / `xgb_model.json` era | Current (Batch D) |
| - | ----------------------------------- | ----------------- |
| Task | Regression on raw `Forward_Return_2w` | Binary `Win_*` per horizon |
| Features | EMA50, ATR, EMA20, Fib 786, Fib 618, **Score** | 26 pillars; Score/Fib forbidden |
| Models | XGB **and** LGB | XGB only; never average |
| OOS vs Score/random/pop | Not in that chart | Walk-forward; **fails** |
| Live | `ML_Pred_Return` as predicted return | `SERVE_ML_RANKER`: rank by 21d P(win); gate flag stays `none` |

Shared DNA: both lean on **ATR + EMA distance**. That is not evidence of a selector.

A fair bake-off was **not run**. To do it: score the old booster on `coiled_cobra_ml_oos_10d.csv` with the same per-fold `top_10pct` protocol and `Forward_Return_10d` / `Win_10d`. Only if `coiled_cobra_xgb_model.json` still exists.

## Docker (easy to forget)

Compose mounts **`./data` only**. Host edits to `src/` and `tests/` are **not** in the running container until `docker cp` or `docker compose build` + recreate.

```bash
# Labels (slow; ~50+ min)
docker exec finance_vibe python -m finance_vibe.coiled_cobra_backtest daily --backtest

# Train — always pin --csv
docker exec finance_vibe python -m finance_vibe.coiled_cobra_ml_training \
  --csv /app/data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv

# Tests (image must contain current tests/)
docker compose run --rm --no-deps finance-vibe python -m pytest tests/ -q
```

Last test count after Batch D: **140 passed**.

## Next work (priority)

Do **not** loosen the promotion gate. Do **not** retrain the same 26 columns on Win hoping hyperparameters save it.

1. **New signal-time features, one group at a time** (slopes, RS vs QQQ path, ATR *contraction*, volume dry-up, stop distance). Drop the group if top-decile med/wr does not beat random **and** population. Collapse redundant pairs (`Volume_Shelf` ≡ `Volume_Contraction_Ratio`, dist-to-high Pct vs ATR).
2. **Risk-adjusted label** if Win is too noisy: e.g. `Max_Return / ATR_Pct`, Hit∧Win, or Win with an MAE cap. Keep the **same** promotion metrics.
3. **L1 universe:** point-in-time names; drop `STATIC_TICKERS` from historical study; report lift vs population.
4. **Hygiene before any live `xgb`:** I1 fail-loud missing live features; I2 shared `source_csv_sha256`; pin `SOURCE_FILENAME`; I3 delete unused `resolve_model_paths`; I4 do not treat `ML_Pred_Return` as expected return; V1 date-block SEs; V2 embargo on `final_train`; **fix research-loop `target` clobber**.
5. Score legacy 6-col booster on the OOS CSVs if you still need that comparison.

Hyperparameter search (`lr=0.01` underfit vs logistic — E3) only **after** a real feature or label signal exists.

## Batches already shipped (2026-08-24)

| Batch | What |
| ----- | ---- |
| 0 + A | Persist OOS CSV; C1 per-fold cuts; C2 top N = 5/3/1; E1/E2/T2 promotion gate; `schema_version` 3 |
| B | Native `--backtest` CSV; T3/T4 no close-based Hit fallback; strict enrich |
| C | C3/C4 train/save/load floors; `schema_version` 4 |
| D | Primary label `Win_*`; live `ML_Prob_Win_*`; gate unchanged; `schema_version` 5 |

## Code map

| File | Role |
| ---- | ---- |
| `src/finance_vibe/coiled_cobra_ml_training.py` | Trainer, `HORIZON_SPECS`, promotion, artifacts |
| `src/finance_vibe/ml_ranker.py` | Inference; `SERVE_ML_RANKER` ranks even when `production_model` is `none` |
| `src/finance_vibe/coiled_cobra_backtest.py` | Native `Forward_Return_*` / `Max_Return_*` / `Win_*` / `Hit_*` |
| `src/finance_vibe/config.py` | Universe including `STATIC_TICKERS` (L1) |
| `tests/test_coiled_cobra_ml.py` / `test_ml_ranker.py` / `test_coiled_cobra_backtest.py` | Contract |

## Open issues still in `CodeReview.MD`

L1 survivorship, V1 no date-block SE, V2 final-train embargo, V3 silent fold skips, I1–I4, E3 underfit, E4 0.5-threshold P/R, `SOURCE_FILENAME` pin, metadata `target_column` clobber.
