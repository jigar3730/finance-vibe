**Resume / pickup (2026-08-24, second pass):** start at **§A**, then §10's P0 table (A→I). Do not promote. Do not serve. Do not retrain the same 26 features on `Win_*`. Handoff doc `docs/CoiledCobraML-Handoff.md` still documents `SERVE_ML_RANKER` as intended behaviour and needs the same correction.

> **SECOND-PASS AUDIT — 2026-08-24 evening (independent review).** The verdict below is **out of date and too favourable**. An independent replay of the artifacts on disk found that the model being served live is **not** the model any number in this document describes, and that the "fix" applied for I5 destroyed the evidence rather than the defect. **Read §A first.** Everything in §1–§12 describes the walk-forward *evaluation*; §A describes what is actually shipping.

---

## Verdict

**CRITICAL — stop serving. The live ranker is a different model from the one that was evaluated, and it is enabled by default.**

Superseded verdict (first pass, retained for history): *"the plumbing is now mostly sound; there is still no tradable selector."* Walk-forward ranking (C1), per-date breadth (C2), stump folds/artifacts (C3/C4), native labels (T3/T4), and the promotion gate (E1/E2/T2) were fixed on 2026-08-24. Primary labels are `Win_*` (T1). All three horizons remain `production_model: none`.

What the second pass establishes on top of that:

1. The three boosters in `data/logs/daily/` were fitted on **`Hit_15Pct_10d` / `Hit_20Pct_21d` / `Hit_50Pct_42d`** — the *last research target* of each horizon — not on `Win_*` (**A1**). Their `base_score` and their AUC against each candidate label prove it.
2. `config.SERVE_ML_RANKER = True` (added in `b9ec562`) makes the live scan **sort the book by those boosters** even though `production_model` is `none` for all three horizons (**A2**). The gate is now decorative.
3. The horizon metadata was **hand-rewritten 3.5 hours after training** to say `target_column: Win_*`, and `_allowed_target_columns` was widened so a `Hit_*` artifact validates as a `Win_*` artifact (**A3**). The only two mechanisms that could have caught A1 were removed in the act of "fixing" it.
4. The training corpus named in the metadata **no longer exists**. `validate_cobra_signals.py` overwrote it with a 28-symbol / 556-row file (**A4**).
5. Even the correctly-evaluated `Win_*` models are **anti-selective**: mean return falls monotonically as the cut tightens, and the top decile loses to a random decile at all three horizons (**A6**).

The 6-feature `Forward_Return_2w` XGB+LGB importance chart is a **different model family**, not a prior version of this stack (see §12).

---

## A. Second-pass findings (independent audit, 2026-08-24 evening)

Method: replayed the on-disk artifacts directly — loaded each booster with `xgboost`, read its `base_score` and objective out of `save_config()`, scored it against every candidate label, and compared its output distribution to the `coiled_cobra_ml_oos_*.csv` files written by the same training run. No code was changed. Every number below is reproducible from `data/logs/daily/`.

### A1 — The served boosters were trained on the research MFE targets, not on `Win_*` (P0, blocker)

XGBoost stores its fitted intercept as `base_score`, which for `binary:logistic` is the mean of the label it was trained on. Comparing that against the base rates the training run itself recorded in `class_balance` / `research_targets[*].class_balance`:

| Horizon | Booster `base_score` | `Win_*` rate | `hit_col` rate | **`research_targets[-1]` rate** |
|---|---:|---:|---:|---:|
| 10d | **0.07576** | 0.54359 | `Hit_10Pct_10d` 0.16969 | **`Hit_15Pct_10d` 0.08792** |
| 21d | **0.09300** | 0.55884 | `Hit_15Pct_21d` 0.17453 | **`Hit_20Pct_21d` 0.10857** |
| 42d | **0.03669** | 0.57453 | `Hit_25Pct_42d` 0.15027 | **`Hit_50Pct_42d` 0.04730** |

Three for three, the intercept lands on the last research target and is off the `Win_*` rate by a factor of 6–16×. Two independent checks agree:

| Check | 10d | 21d | 42d |
|---|---:|---:|---:|
| AUC of served booster vs `Win_*` | **0.460** | **0.450** | **0.462** |
| AUC of served booster vs `Hit_*` | 0.781 | 0.814 | 0.800 |
| Mean prediction, served booster | 0.082 | 0.097 | 0.040 |
| Mean prediction, `coiled_cobra_ml_oos_*.csv` (same run) | 0.556 | 0.582 | 0.594 |

The AUC against `Win_*` is **below 0.5** — as a win-probability ranker the artifact is worse than a coin flip and slightly inverted. The last two rows are the cleanest proof and depend on no external file: the OOS predictions written at 17:37 and the booster written at 17:36–17:38 come from the same `train_horizon` call, yet their output distributions cannot belong to the same model. (Caveat on the AUC rows only: they were computed against the surviving 556-row CSV, since the training corpus is gone — see A4. The `base_score` and mean-prediction rows do not depend on any CSV and are exact.)

This is I5, but I5's stated conclusion — *"Models on disk were trained on Win; the JSON field lies"* — is **exactly backwards**. The JSON field was telling the truth. The `for target in spec["research_targets"]` rebind did not merely mislabel the metadata; it rebound the variable that `y_tr = final_train[target]` reads, so **the saved artifact itself is the research model**. Everything downstream of `final_model` — the booster, `best_iteration`, `normalized_xgb_gain`, the importance chart — describes an MFE model.

Consequence: `ML_Prob_Win_21d` in the live scan is P(+20% maximum favourable excursion within 21 days). It is the volatility lottery that **T1 concluded must never be traded**, relabelled as a win probability and shipped. The live file `coiled_cobra_setups_2026-08-24.csv` shows the signature plainly — 37 of 37 rows carry `ML_Prob_Win_21d` between 0.12 and 0.16 against a 56% win base rate.

**Fix:** delete all three boosters, retrain from a known-good CSV on current `HEAD` (which added the defensive `target = label_col(spec)` re-bind), and assert in `train_horizon` that `float(base_score)` is within a few points of `class_balance["positive_rate"]` before `save_xgb_artifact` writes anything.

### A2 — `SERVE_ML_RANKER` makes the promotion gate decorative (P0, blocker)

`config.py:406` sets `SERVE_ML_RANKER = True`. `ml_ranker._serve_horizon` returns True whenever that flag is set, **regardless of `production_model`**:

```python
if meta.get("production_model") == "xgb":
    return True
return bool(getattr(config, "SERVE_ML_RANKER", False))
```

So `attach_ml_ranks` fills `ML_Prob_Win_*`, sets `ML_Pred_Return`, and **re-sorts the entire book** by a model that failed every promotion criterion. `ML_RANK_HORIZON_PRIORITY` puts 21d first — the horizon whose saved artifact is the `Hit_20Pct_21d` model.

The docstrings and `MLOps.md` defend this as "promotion flags stay honest". They do, and it does not help: nothing downstream reads `production_model`. The trade plan, the email, and the operator all consume row order, and row order is now set by the failed model. A flag that records a decision nobody enforces is not a gate.

`§8` lists *"`attach_horizon_probabilities` returning NaN when `production_model == "none"` — the actual production gate"* as a missing test. That test is missing because the behaviour it would assert was deliberately reversed.

**Fix:** default `SERVE_ML_RANKER = False`. If a shadow-mode path is genuinely wanted, write the probabilities to columns the sort never reads, keep `ML_Pred_Return` / `ML_Rank` null until `production_model == "xgb"`, and log loudly at scan time that an ungated model is attached.

### A3 — The I5 remediation removed both detectors instead of the defect (P0)

Two things happened after the 17:36–17:38 training run:

| Artifact | mtime (container) |
|---|---|
| `coiled_cobra_xgb_10d/21d/42d.json` | 17:36:40 / 17:37:29 / 17:38:13 |
| `coiled_cobra_ml_oos_21d.csv` | 17:37:08 |
| `coiled_cobra_ml_model_metadata.json` (index) | 17:38:14 |
| **`coiled_cobra_ml_metadata_10d/21d/42d.json`** | **21:03:18** |

The three per-horizon metadata files were rewritten 3.5 hours later — 86 seconds after commit `b9ec562` — while the boosters were not. §6/I5 recommended exactly this (*"rewrite the three metadata files (or retrain)"*). Taking the first branch edited the description to match the intention rather than retraining the model to match the description, and it erased the single field that recorded which label the artifact had actually seen.

In the same commit, `ml_ranker._allowed_target_columns` was introduced:

```python
def _allowed_target_columns(spec: dict) -> set[str]:
    allowed = {label_col(spec)}
    allowed.update(spec.get("research_targets") or ())
    if spec.get("hit_col"):
        allowed.add(spec["hit_col"])
```

`validate_artifact_metadata` now accepts `target_column` ∈ {`Win_10d`, `Hit_10Pct_10d`, `Hit_15Pct_10d`} for the 10d horizon. The check whose entire purpose is to refuse an artifact trained on the wrong target now accepts every target that horizon can produce, including the exact wrong value I5 was about. Had the metadata not been hand-edited, it would have loaded silently anyway.

**Fix:** `_allowed_target_columns` must return `{label_col(spec)}` and nothing else. Research models are not servable artifacts and should never share a filename with one. Treat any metadata file newer than its booster as a hard load error.

### A4 — `validate_cobra_signals.py` destroyed the training corpus (P0)

`validate_cobra_signals.py` (new in `0775b74`) shells out to the backtest over a hardcoded 40-ticker `UNIVERSE` and writes into the shared log directory under the **same date-stamped filename** the trainer reads:

```python
cmd = ["python", BACKTEST_SCRIPT, "--backtest", "--tickers", ticker_str]
```

It ran at 20:36 and overwrote `coiled_cobra_backtest_trades_2026-08-24.csv`. Current state of the file the metadata names as its source:

| | Recorded at training | On disk now |
|---|---|---|
| `source_csv_sha256` | `5567a2e3…b1b65057` | `914845ff…bc44796b` |
| new-coil rows | ~21,961 | **556** |
| symbols | 413 | **28** |
| OOS rows still joinable | — | **1.4–1.5%** |

The corpus that produced every number in §1–§12 is gone; `_Large.csv` is the 89-column pre-native-label file, not a substitute. Because `SOURCE_FILENAME` still names a non-existent `..._2026-07-17.csv`, `resolve_source_csv` falls through to newest-by-mtime, so the **next** `--csv`-less training run picks up the 556-row file. `MIN_TRAIN_ROWS = 1000` then empties `walk_forward_folds`, silently dropping the run into the single-split fallback at `train_horizon:1294`.

`source_csv_sha256` is recorded and, per I2, never compared — so nothing noticed.

**Fix:** give validation runs their own output directory or filename prefix; make the trainer verify `source_csv_sha256` against the file it just read and abort on mismatch; pin `SOURCE_FILENAME` and delete the newest-mtime fallback.

Separately, `validate_cobra_signals.py` has its own methodology problems: its "lookahead leakage audit" is `|Pearson r| > 0.80` against `Forward_Return_21d`, which no realistic leakage in this pipeline would trip; its win rate is computed over all rows rather than `Is_New_Coil`; `OUTPUT_DIR` is hardcoded to `/app/...`; and it selects "latest" by `getctime`.

### A5 — The `MIN_BEST_ITERATION` floor is set an order of magnitude too low (P1)

C4 established that a booster stopping at 0–2 iterations is a near-constant and set the floor at `best_iteration >= 3`. The reasoning is right and the threshold does not implement it: at `learning_rate = 0.01`, four trees is just as constant as three. Actual fold-level `best_iteration` in the shipped run:

| Horizon | Contributing folds | `best_iteration` per fold | Folds < 10 trees | Share of pooled OOS rows they contribute |
|---|---:|---|---:|---:|
| 10d | 11 | 4, 5, 6, 13, 18, 23, 29, 36, 45, 88, 92 | 3 / 11 | **20.4%** |
| 21d | 11 | 3, 5, 8, 9, 12, 13, 14, 56, 62, 74, 104 | 4 / 11 | **39.9%** |
| 42d | 10 | 3, 3, 3, 4, 5, 5, 6, 7, 20, 93 | 8 / 10 | **76.2%** |

At 42d, **eight of ten surviving folds are stumps** and a single fold (93 trees) is the only substantive model in the pool. Three quarters of the 42d OOS rows are ranked by near-constant predictors, so "42d XGB top decile" is largely an arbitrary ordering of ties. The C3/C4 fix moved the failure from *"stumps are pooled"* to *"stumps just above the floor are pooled"*.

This also invalidates the §2/C3 table, which reports 16 / 15 / 14 contributing folds and 0 / 1 / 1 stumps. The shipped artifacts have **11 / 11 / 10 contributing and 5 / 5 / 5 skipped**. That table describes the earlier Hit-label run and should not be read as current.

**Fix:** raise the floor to something with a defensible basis — either `best_iteration >= 50` at `lr = 0.01`, or better, raise `learning_rate` to 0.05 and require the fold to beat its own base rate on validation AUC by a margin. Report the trees-weighted share of OOS rows in the summary so this is visible without a script.

### A6 — The `Win_*` models are anti-selective, not merely edgeless (P0 for interpretation)

§7/T1 records that Win-trained XGB "does not beat Score, random, or the population on average return." The pooled metadata shows something stronger and more diagnostic — return falls **monotonically as the cut tightens**, at every horizon:

| Horizon | population | top 25% | top 10% | top 5% | random top 10% |
|---|---:|---:|---:|---:|---:|
| 10d | +0.77% | +0.35% | +0.38% | +0.14% | +0.68% |
| 21d | +1.78% | +0.97% | +1.40% | +1.62% | +2.25% |
| 42d | **+5.37%** | +2.40% | **+1.14%** | +1.13% | +5.52% |

At 42d the top decile earns roughly a fifth of the population it is drawn from. A model with no signal produces a flat line (as `random` does); a line that slopes down is signal pointed the wrong way. The mechanism is visible in the OOS predictions — `spearman(P(win), ATR_Pct)` is **−0.29 / −0.32 / −0.16**, and top-decile MFE hit rate collapses versus population (10d 0.100 vs 0.199; 42d 0.075 vs 0.194). The `Win_*` model learned "low volatility ⇒ more likely to close green", which is true and useless, because in this universe the return lives in the tail it is now excluding. It is the same `ATR_Pct` proxy T1 diagnosed, with the sign flipped.

Fold-level agreement confirms it is not noise: `fold_pass_rate` is **9% (1/11), 9% (1/11), 0% (0/10)**.

Practical implication: `1 − P(win)` is a better ranker than `P(win)` on this pool. That is not a promotion candidate — it is evidence that the 26-feature set spans volatility and little else. §10 item 5 already reaches the right conclusion; A6 raises its priority from "no edge" to "inverted edge that is currently live via A2".

### A7 — New hard gates were added to the scanner without re-deriving the training population (P1)

Commit `0775b74` added two gates to `evaluate_coiled_cobra`:

```python
or volume_ratio > 0.82  # Enforces Volume Contraction Cutoff
or checks_passed < 5     # Enforces 5/6 Quality Standard
```

In the 21,961-row training pool (`_Large.csv` lineage), **68.3%** of new coils have `Volume_Contraction_Ratio > 0.82`. The live scanner now emits a population that excludes two thirds of what the models were fitted on, and `Volume_Contraction_Ratio` is itself feature #7 in `FEATURE_COLS` — so the live feature is truncated to `[0.33, 0.82]` while training saw up to 212.75. Any retrain must regenerate the backtest under the new gates; any inference before that is extrapolating off the edge of the training support.

The same commit left three tests failing (`test_evaluate_can_pass_coiled_uptrend`, `test_evaluate_clips_score_at_100`, `test_score_excludes_fib_and_macd_cross`) — the fixtures construct setups the new gates reject. **140 passed, 3 failed.** A red suite is how A1–A4 stayed invisible.

### A8 — "Beats random" is a single draw, not a test (P1)

E1 correctly added a random baseline to the gate, but `_Random_Rank` is one `default_rng(RANDOM_SEED + i).random(len(test_out))` draw per fold. Requiring the model to beat *one* sample from the null says almost nothing: at 42d the random top decile (+5.52%) beat Score (+4.84%), which is a statement about that draw, not about Score. Combined with V1 (overlapping 42-day label windows, effective n of a few hundred date-blocks rather than 12k rows), the gate's pass/fail is inside its own noise band in both directions.

**Fix:** replace the single draw with 200–1000 permutations of the rank column within each fold, and promote only when the candidate clears the 95th percentile of the permutation distribution on every metric. That is the same work item as V1 item 8 and should be done once, for both.

### A9 — Live feature frame still cannot fail loudly (I1 confirmed open, now P0 given A2)

`build_feature_frame` still falls through to `out[col] = np.nan` for any feature the scanner did not write, and `predict_horizon_proba` only skips a row when **every** feature is NaN:

```python
valid = feats.notna().any(axis=1)
```

A row with 25 of 26 features missing still receives a prediction, and XGBoost routes NaN to a default direction without complaint. This was P1 while nothing was served. Under A2 it is P0. Today's scan happens to write all 26 columns, so this is latent rather than active — but it is latent by luck, not by construction.

### A10 — Second-pass status of previously-open items

| Item | Claimed | Verified state |
|---|---|---|
| I1 raise on missing live features | open | **open** — `ml_ranker.py:229` |
| I2 cross-horizon `source_csv_sha256` | open | **open**, and A4 shows the cost |
| I3 delete `resolve_model_paths` | open | **open** — `ml_ranker.py:79-100`, still 21d-filename-first, still bypasses validation |
| I4 `ML_Pred_Return` semantics | open | **partly fixed** — `attach_weekly_confirmation` no longer overwrites the stored value; it now boosts a separate `rank_key`. But `trade_planner.py:293` still computes `Priority = ml_pred × propensity × weekly_boost` on a probability, and `test_priority_keeps_negative_ml_below_zero` still feeds `-0.04` |
| I5 metadata `target_column` | "OPEN, JSON field lies" | **worse and mis-diagnosed** — see A1/A3 |
| V2 embargo on `final_train` | open | **open** — `coiled_cobra_ml_training.py:1466` |
| `SOURCE_FILENAME` pin | open | **open** — `:135` still names the non-existent 2026-07-17 file |
| C1 per-fold cuts | fixed | **holds** — metadata reports `selection_scope: per_fold`, `n_selection_groups` 11/11/10 |
| C2 per-date breadth | fixed | **holds**, and `selected_fraction` exposes that `top_5_per_date` still keeps 45–49%; only `top_1_per_date` (≈10%) is comparable to `top_10pct` |
| T3/T4 native labels | resolved | **unverifiable** — the CSV that would prove it was overwritten (A4) |

---

## 0. Locked decisions (do not reverse without new OOS)

| Decision | Evidence |
| -------- | -------- |
| Judge avg / median / win rate vs Score **and** random **and** population; ignore hit_rate | E1, E2, T1, T2 |
| Rank fraction cuts **inside each walk-forward fold** | C1 |
| Per-date N = 5 / 3 / 1 | C2 |
| `MIN_TRAIN_ROWS=1000`, refuse `best_iteration < 3` | C3, C4 |
| Train `Win_*`; keep `Hit_*` as research | T1 |
| Native backtest CSV only; always `--csv` until `SOURCE_FILENAME` is pinned | T3, item 16 |
| No Score / Fib in `FEATURE_COLS`; no XGB+LGB average; no horizon blend | contract tests |
| Do not loosen the gate to force a promotion | T1: Hit mean edge was vol |
| **Do not loosen a validator to make an artifact load** | A3 |
| **An artifact that fails the gate is not served, shadow or otherwise** | A2 |
| **Never "fix" an artifact by editing its metadata — retrain or delete** | A1, A3 |
| **Validation/experiment scripts never write into the trainer's filename** | A4 |

---

## 1. What actually verified clean

| Requirement | Status | Evidence |
|---|---|---|
| Only `Is_New_Coil=True` trained | PASS | 21,961 / 80,052 rows kept |
| Features causal | PASS | backtest evaluates `df.iloc[: idx + 1]` (`coiled_cobra_backtest.py:430,495`); `relative_strength` truncates the benchmark at `as_of` (`market.py:193-196`) |
| No forward/exit/Outcome/R-Multiple in features | PASS | `LEAKAGE_COLS` + `LEAKAGE_PREFIXES` cover them, and `_feature_matrix` reindexes to `FEATURE_COLS`, so nothing else can enter |
| No random split | PASS | all splits date-based, `walk_forward_folds` tiles 2018→2026 with non-overlapping test windows |
| Embargo per horizon | PASS | 2w/5w/9w ≥ 10/21/42 trading days, applied at both train→val and val→test |
| `best_iteration` honored | PASS | training `iteration_range=(0, best+1)`; inference reads it from metadata |
| Separate models/metadata | PASS | three model files, three metadata files, index file |
| No LGB averaging | PASS | LightGBM is not loaded in any prediction path |
| Duplicate `Symbol+Signal Date` | PASS | exact dupes dropped, conflicting dupes abort |
| ~~Promotion gate held~~ | **FAIL (A2)** | `production_model` is `none` for all three, but `SERVE_ML_RANKER = True` serves and sorts by them anyway |
| ~~Targets correct~~ | **FAIL (A1)** | The walk-forward *evaluation* used `Win_*`. The **saved artifacts** were fitted on `Hit_15Pct_10d` / `Hit_20Pct_21d` / `Hit_50Pct_42d` — proven by `base_score` and by AUC 0.45 vs `Win_*`. |
| ~~Reproducible source~~ | **FAIL (A4)** | `source_csv_sha256` no longer matches the file it names; the corpus was overwritten with 556 rows / 28 symbols |

---

## 2. Critical bugs

**C1 — Pooled top-X% ranking mixes probabilities from 17 different models (P0). — FIXED 2026-08-24.**
`top_fraction` (`coiled_cobra_ml_training.py:730-738`) was applied to the concatenated OOS frame, so it ranked fold-1's probabilities against fold-17's. XGBoost probabilities are not calibrated across folds — a fold trained through a bull stretch emits systematically higher probabilities than one trained through 2022. The pooled top decile therefore concentrated in whichever periods produced high probabilities, not in the best setups. `Score` is on a fixed 70–98 scale and suffered none of this, so the Score-vs-ML comparison was apples-to-oranges.

The size of the distortion was measurable from the metadata. For 42d, with roughly equal fold sizes:

- mean of the 16 fold-level XGB top-decile averages: **+7.60%**
- pooled top-decile average: **+4.04%**

If the pooled cut were drawing ~10% from each fold it would land near the fold mean. It landed at half, which is only possible if the pooled decile was selecting a materially different, period-concentrated sample.

Fix: `top_fraction` now takes a `group_col` (default `FOLD_COL`) and cuts the fraction inside each walk-forward fold, pooling the picks afterwards. Each fold contributes its own share and no two models are ranked against each other. `evaluate_ranker` reports `selection_scope` / `n_selection_groups`, metadata carries a `selection_scope` note and `schema_version` is 3. The dead `_top_frac_mean` helper, which had the same defect, was removed. Verified invariant: pooled `top_10pct` now equals the size-weighted mean of the fold-level `top_10pct` values to within 1e-18 for xgb, score, and random at all three horizons.

Effect on conclusions after retraining — the non-monotonicity disappeared and ML's mean advantage grew, but win rate is still below Score at every cut, so T1 remains and all three horizons still fail promotion:

| Horizon | Cut | Score avg (before → after) | XGB avg (before → after) |
| --- | --- | --- | --- |
| 10d | top 10% | +0.97% → +0.75% | +1.48% → +1.91% |
| 10d | top 5% | +1.18% → +0.64% | **−1.27%** → +1.83% |
| 21d | top 10% | +2.76% → +2.26% | +1.56% → +3.69% |
| 21d | top 5% | +2.80% → +2.27% | **−0.97%** → +2.70% |
| 42d | top 10% | +3.93% → +3.43% | +4.04% → +8.82% |
| 42d | top 5% | +3.59% → +2.78% | +4.26% → +8.77% |

**C2 — `top_10_per_date` selects 77% of the population (P0). — FIXED 2026-08-24.**
Median signals per date is 8 (mean 9.5, 25th pct 5). So `top_10_per_date` was 15,387 of 19,947 rows and `top_20_per_date` was 19,290, yet `walk_forward.pooled.score_top10_avg_return` / `ml_top10_avg_return` — the fields a reader treats as the headline — were exactly these near-population numbers, labeled "top10".

Fix: `TOP_N_PER_DATE_LEVELS` is now `(5, 3, 1)` and `TOP_N_PER_DATE` derives from it, so the headline per-date metric can't silently drift back to a near-population cut. The misleading `score_top10_*` / `ml_top10_*` metadata keys were renamed to `score_per_date_*` / `ml_per_date_*` and carry `per_date_n` alongside. Every selection now also reports `selected_fraction`, printed as a `Keep` column, so an over-loose cut is visible on sight rather than needing to be discovered.

That immediately showed the remaining looseness: `top_5_per_date` still keeps 47% and `top_3_per_date` 30%. **`top_1_per_date` (10.4%) is the only per-date cut comparable in tightness to `top_10pct`.**

**C3 — Fold 1 contributes ~1,150 predictions from an untrained model (P1). — FIXED 2026-08-24.**
`MIN_TRAIN_ROWS = 80` admitted a fold with `n_train=505`, and both the 10d and 21d fold-1 models stopped at `best_iteration=0` — a single tree at `learning_rate=0.01`, i.e. a near-constant predictor. Those rows were pooled into OOS as if they were real predictions.

Fix: `MIN_TRAIN_ROWS` is 1000, so the 252/505/762-row first folds never enter walk-forward. Independently, any fold that still early-stops at `best_iteration < 3` is recorded as skipped and excluded from the OOS pool (and from `fold_pass_rate`). After retrain:

> **Stale — see A5.** The table below is from the Hit-label run. The shipped artifacts have **11 / 11 / 10** contributing folds and **5 / 5 / 5** skipped, and the `best_iteration >= 3` floor is too low to do what it claims: 76% of pooled 42d OOS rows still come from folds with fewer than ten trees.

| Horizon | Folds before | Contributing | Skipped as stump | First train size now |
| --- | ---: | ---: | ---: | ---: |
| 10d | 17 | 16 | 0 | 1,715 |
| 21d | 17 | 15 | 1 (`n_train=17,672`, `best=2`) | 1,346 |
| 42d | 16 | 14 | 1 (`n_train=12,222`, `best=2`) | 1,907 |

The 21d/42d skips prove the iteration floor is not redundant with the row floor: a 17k-row fold can still be a stump. Promotion is unchanged — all three horizons still fail on `med_fwd, win_rate`.

**C4 — The saved 21d artifact is degenerate (P1). — FIXED 2026-08-24.**
`coiled_cobra_xgb_21d.json` previously had `best_iteration = 2`. Three trees at lr 0.01 is a constant ≈ base rate. Nothing in `train_horizon` refused to save a model that early-stopped at ~0 iterations, and nothing in `ml_ranker` refused to load one.

Fix: `save_xgb_artifact` writes the booster only when `best_iteration >= 3` and deletes any leftover file otherwise; `production_model` is forced to `none` if the artifact is not saved. `validate_artifact_metadata` now rejects a degenerate `best_iteration` before inference. This run's saved artifacts are 83 / 44 / 199 trees and were written. `schema_version` is 4.

---

## 3. Data leakage risks

No classic leakage. Two soft lookahead problems that matter more than they look:

**L1 — Survivorship / hindsight universe (P0 for interpretation).**
`config.py:48-69`: the universe is today's `most_actives` screener plus `STATIC_TICKERS`, which is a hand-picked list of known decade winners (NVDA, PLTR, AXON, DUOL, CELH, SMCI, HOOD, COIN, MELI, DECK…). You then backtest that list over 10 years. The tell is in the data: **random selection inside this universe returns +3.83% per 42 trading days**, roughly +24% annualized. Delisted and dead names are absent entirely. This inflates every absolute number and, worse, it can *manufacture* an apparent edge for "buy the volatile name," because in this universe volatility survived.

**L2 — Split/dividend-adjusted history (P2).** Raw CSVs are `auto_adjust=True` as of download date. Ratio features and returns mostly cancel this out, so it's minor, but it is not point-in-time.

---

## 4. Target-definition problems

**T1 — `Hit_X%` (max favorable excursion) is monotone in volatility, and the model learned exactly that (P0). — ADDRESSED 2026-08-24.**
MFE over a fixed window is close to a volatility statistic. The Hit-trained 42d XGB as you tightened the cut:

| Selection | Avg | **Median** | **Win rate** | Hit rate |
|---|---:|---:|---:|---:|
| population | +3.83% | +1.89% | 0.560 | 0.161 |
| top 25% | +5.69% | +0.14% | 0.502 | 0.329 |
| top 10% | +4.04% | **−4.66%** | **0.447** | 0.374 |
| top 5% | +4.26% | **−7.83%** | **0.406** | 0.438 |

Hit rate climbed while median and win rate collapsed. The model was finding wider distributions, not better setups.

Fix: the primary training label is now `Win_*` (`Forward_Return_Xd > 0`). `Hit_*` remains as `hit_col` for reporting and as a research target. Live probabilities are `ML_Prob_Win_10d` / `_21d` / `_42d`. Promotion metrics were left unchanged (`avg_fwd`, `med_fwd`, `win_rate`) so the scoreboard move is attributable to the label. `schema_version` is 5.

After retrain, the lottery is gone and so is the apparent mean edge:

| Horizon | Target | ML avg | Score avg | Rand avg | Pop avg | ML med | Score med | ML wr | Score wr | Pop wr | Promote |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10d | Win | +0.38% | +0.54% | +0.68% | +0.77% | **+0.36%** | +0.61% | 0.541 | 0.544 | 0.527 | NO |
| 21d | Win | +1.40% | +1.75% | +2.25% | +1.78% | **+1.23%** | +1.17% | 0.583 | 0.555 | 0.559 | NO |
| 42d | Win | +1.14% | +4.84% | +5.52% | +5.37% | **+0.97%** | +2.56% | 0.551 | 0.566 | 0.573 | NO |

Medians stay positive at every cut. Win-trained XGB does **not** beat Score, random, or the population on average return. `ATR_Pct` is still the top gain feature (0.21–0.25), and the 55% base rate plus `learning_rate=0.01` produced more stump folds (5 per horizon skipped). That is T1 confirmed: the Hit models' mean advantage was the volatility proxy, not setup quality. A useful selector needs a different feature set or a risk-adjusted label (`Max_Return / ATR_Pct`, MFE conditioned on MAE) — not another retrain of the same 26 features on Win.

**T2 — `hit_rate` is invalid as a promotion criterion (`compare_rankers:820-825`) (P1).** It is the model's own training objective. XGB will beat Score on it essentially always, so it adds nothing to the gate while making it look stricter.

**T3 — The training labels were built by a fallback path, not by the backtest (P1). — RESOLVED 2026-08-24.** The source CSV (`coiled_cobra_backtest_trades_2026-08-24_Large.csv`, 89 columns) contained **no** `Forward_Return_10d`, **no** `Max_Return_*`, **no** `Win_*`, **no** `Hit_*`. Every label came from two runtime derivations: `Forward_Return_10d` aliased off `Forward_Return_2w` (`_ensure_horizon_targets`), and all `Max_Return_*`/`Hit_*` synthesized by `enrich_mfe_targets`. The new `HIT_LABEL_SPECS`/`_horizon_fields` code — the code the tests cover — never touched that dataset.

Resolution: re-ran `coiled_cobra_backtest daily --backtest` to produce a CSV (114 columns) carrying native `Forward_Return_*`/`Max_Return_*`/`Win_*`/`Hit_*`, then diffed the two label sets on the shared signal set and retrained from the native CSV.

| check | result |
| --- | --- |
| signal set | 21,961 new coils / 413 symbols on both sides, 0 old-only, 0 new-only — the backtest is deterministic |
| `Forward_Return_10d` vs old `Forward_Return_2w` | **0 mismatches**, max abs diff 0.0 — the alias claim is confirmed exactly |
| `Forward_Return_42d` (control) | 0 mismatches |
| NaN gating | identical (180 / 343 / 479 unlabelled tail rows per horizon on both sides) |
| `Max_Return_*` | ~10–11% of rows differ, max abs diff 0.0122 / 0.0156 / 0.0870; means agree to 5 decimals |
| `Hit_*` | 0–12 rows flip out of ~21.7k, i.e. 0.005%–0.056%; base rates move in the 4th decimal (`Hit_10Pct_10d` 0.16946 → 0.16969) |

The `Max_Return_*` gap is fully explained: `backtest_ticker:465` divides by `float(stamped["Close"])`, the 2-decimal-rounded close from `build_setup_row`, while `enrich_mfe_targets` divides by the raw OHLC close. All 2,425 mismatching 42d rows have `|Close − raw Close| ≤ 0.005`, and they are concentrated in cheap names (median price $9.43 vs $68.31 pool-wide) where half a cent is a material denominator. Worst case is QUBT at $0.725 → $0.73, moving an 1160% MFE by 8.7pp.

Verdict: since every trading metric is computed from `Forward_Return_*`, which is bit-identical, and only ~0.04% of hit labels flip, **the previously reported OOS returns and win rates stand.** The retrain on native labels reproduces them (10d top-decile avg +0.0196 vs +0.0191, median −0.0060 vs −0.0077, win rate 0.476 vs 0.472) and all three horizons still fail promotion on `med_fwd` and `win_rate`. T1 is confirmed on labels straight from the tested code.

**T4 — `enrich_mfe_targets` silently mixes two label definitions (P1). — FIXED 2026-08-24.**

A symbol whose raw file was missing kept the close-based `Hit_*` that `_ensure_horizon_targets` had already written from `Forward_Return_*`. Those rows trained on "close ≥ +25%" while everyone else trained on "any high ≥ +25%", and only a count was printed. It happened to be 0 missing that run.

Both halves of the mix are now impossible:

- The close-based `Hit_*` fallback in `_ensure_horizon_targets` is **deleted**. Hit labels are only ever derived from `Max_Return_*`; a horizon that cannot supply one raises with the re-run command instead of approximating. The alias derivations (`legacy_max`, `legacy_forward`) are exact and remain.
- `load_and_prepare` now calls `_ensure_horizon_targets(require_max=False)` before enrichment and `require_max=True` after, so a pool that still lacks intra-horizon highes fails rather than trains.
- `enrich_mfe_targets` takes `strict=True` (default) and raises `FileNotFoundError` naming the unresolved symbols instead of returning them unlabelled.

Tests added: `test_hit_labels_never_fall_back_to_the_horizon_close`, `test_hit_labels_come_from_the_intra_horizon_high`, `test_enrich_mfe_targets_labels_each_horizon_from_its_own_window`, `test_enrich_mfe_targets_leaves_tail_signals_unlabelled`, `test_enrich_mfe_targets_raises_when_a_symbol_has_no_raw_history`. The strictness immediately caught a pre-existing test fixture that supplied only `Max_Return_10d` and silently fell into the enrich path.

---

## 5. Validation / split problems

**V1 — No purging of overlapping label windows (P1).** A 42-day label means consecutive same-symbol rows share almost their whole outcome window, and every symbol on a given date shares market beta. Effective sample size is a few hundred date-blocks, not 21k rows. The consequence is visible in the fold-level 42d top-decile averages: `[+6.4%, +2.8%, +0.8%, +25.2%, +38.1%, +1.1%, −13.0%, −19.4%, −1.7%, +20.8%, −11.3%, +6.0%, +21.7%, −1.1%, +31.1%, +14.1%]`. At n≈100 per fold this is noise, and nothing in the pipeline computes a standard error or a date-block bootstrap.

**V2 — The saved artifact's early stopping is contaminated (P2).** `train_horizon:1218-1221` sets `final_train = labeled[date < last["val_start"]]`, but the fold's own train stopped at `val_start − embargo`. So the final refit's training rows include the embargo gap, whose labels overlap `final_val` — the set used to pick `best_iteration`. Test data is untouched, so this doesn't inflate reported OOS, but the shipped `best_iteration` is optimistic.

**V3 — Silent OOS gaps.** In `walk_forward_folds`, a window failing `min_val_rows`/`min_test_rows` is skipped while the loop continues, leaving an unrecorded hole in coverage.

---

## 6. Inference / model-loading problems

**I1 — Missing live feature columns become NaN instead of an error (P1).** `build_feature_frame` (`ml_ranker.py:163-195`) falls through to `out[col] = np.nan` for any pillar the scanner didn't write. `_validate_xgb_booster` checks the *booster* against *metadata*, but nothing checks the *live frame* against either. Your brief said "fail loudly on mismatch"; this is the one path that doesn't.

**I2 — No cross-artifact consistency check (P1).** `attach_horizon_probabilities` reads each horizon's metadata file independently. Train `--horizons 42d` only, and stale 10d/21d models from a different source CSV keep serving predictions alongside it. `source_csv_sha256` is recorded but never compared.

**I3 — `resolve_model_paths` is a live landmine (P2).** Lines 61-82 still return `coiled_cobra_xgb_21d.json` by filename preference, bypassing `validate_artifact_metadata` entirely. Currently unused by any prediction path — delete it before something calls it.

**I4 — `ML_Pred_Return` now holds a probability (P2).** `trade_planner.py:283-295` computes `Priority = ml_pred × propensity × weekly_boost`, and `coiled_cobra.py:930-935` multiplies the stored column by a >1 boost — which can push a probability above 1.0 and means the persisted value no longer equals the model output. `test_priority_keeps_negative_ml_below_zero` still feeds `-0.04`, a value the new pipeline cannot produce.

**I5 — Research-loop `target` rebind (P1 → P0). — MIS-DIAGNOSED; SUPERSEDED BY A1/A3.**

~~`train_horizon` sets `target = label_col(spec)` (`Win_*`) and trains on that. Later `evaluate_research_targets` does `for target in spec["research_targets"]`, which rebinds the same name. Metadata then writes `"target_column": target` as the **last research Hit column** (`Hit_15Pct_10d` / `Hit_20Pct_21d` / `Hit_50Pct_42d`) while `prob_column` is correctly `ML_Prob_Win_*` and `schema_version` is 5. Models on disk were trained on Win; the JSON field lies. Fix: rename the loop variable, rewrite the three metadata files (or retrain).~~

The observation was right and the inference was backwards. The rebind happens **before** `y_tr = final_train[target]`, so it changed the model, not just the label on the model. `target_column` was reporting the truth. See **A1** for the proof (`base_score` 0.076 / 0.093 / 0.037 against `Win_*` rates of 0.54 / 0.56 / 0.57; AUC 0.45 vs `Win_*`, 0.78–0.81 vs `Hit_*`) and **A3** for how the recommended remediation — rewriting the metadata instead of retraining — removed the evidence and let the artifact pass validation.

Current `HEAD` (`b9ec562`) adds a defensive `target = label_col(spec)` after the research call, so the **code** is now safe. The **artifacts on disk are not**, and no retrain has occurred since. Delete them.

---

## 7. Evaluation problems

Beyond C1/C2 above:

**E1 — "Beats Score" is a near-worthless bar, and nothing enforces "beats random" (P0). — FIXED 2026-08-24.**
On the scale-safe per-date cuts, Score is indistinguishable from — sometimes worse than — random:

| 42d, top 5 per date | Avg | Median | Win rate |
|---|---:|---:|---:|
| population | +3.83% | +1.89% | 0.560 |
| random | +3.41% | — | — |
| **Score** | **+3.38%** | +1.73% | 0.550 |
| XGB | +5.40% | +1.72% | 0.537 |
| Logistic | +5.53% | — | — |

Score's top per-date picks underperform the population it's drawing from. `compare_rankers` computed a random baseline and then never used it in the promotion condition.

Fix: `evaluate_promotion` now requires the top decile to beat **Score, random, and the untouched population** on all promotion metrics, and records a per-metric / per-baseline `beats` breakdown in metadata plus a `Blocked by` column in the summary table. The value showed up immediately — at 42d the random top decile (+4.96%) beats Score (+3.43%), which the old table could not have revealed.

**E2 — ML's advantage is mean-only; median and win rate were buried (P0). — FIXED 2026-08-24.**
Median forward return and win rate are now printed next to the mean for every cut and are first-class promotion criteria. This turned the vague "tail-driven" suspicion into a precise, reproducible verdict: all three horizons are blocked by exactly `med_fwd, win_rate` while passing `avg_fwd` against every baseline.

**E3 — Logistic regression beats XGBoost — RE-CHECKED after C3/C4, still open.**
I originally guessed this was pooling, then stump folds. After dropping untrained OOS rows, logistic still wins the 21d and 42d top-decile averages (21d +4.13% vs +3.19%; 42d +13.0% vs +9.5%). 10d flipped: XGB +2.04% vs logistic +1.93%. So C3/C4 were not the whole story. Remaining likely cause is `MODEL_PARAMS` (`learning_rate=0.01`, 400 trees, `early_stopping_rounds=40` on AUC) — XGBoost is still underfit on the longer horizons. That is a research tuning issue, not a correctness bug, and it does not change promotion (both lose to Score on median and win rate).

**E4 — Classification metrics are uninformative.** At a fixed 0.5 threshold with a 15–17% base rate and `scale_pos_weight=1`, recall is under 1%. Harmless since you don't gate on them, but they shouldn't appear in a report.

---

## 8. Missing tests

~~The most important gap: there is no positive-control test that promotion can ever return `True`.~~ **Added 2026-08-24** — `test_promote_passes_when_ml_beats_every_baseline` proves the gate can fire, so "correctly strict" is now distinguishable from "stuck off". Also added: `test_promotion_gate_ignores_hit_rate`, `test_promotion_requires_beating_random_and_population`, `test_selected_fraction_reports_how_much_a_cut_discards`, `test_per_date_levels_stay_below_typical_daily_breadth`, and per-fold selection coverage for `top_fraction`.

~~Any test of `enrich_mfe_targets` — the code path that generated 100% of your training labels.~~ **Added 2026-08-24** — three tests cover per-horizon window isolation, tail NaN gating, and the strict raise on a missing raw file, plus two on `_ensure_horizon_targets` proving hit labels can only come from the intra-horizon high.

Still missing — the second-pass items first, since each one would have caught a live defect:

- **A saved booster's `base_score` is within a few points of the horizon's `class_balance["positive_rate"]`.** This single assertion catches A1 and every future variant of it, and it needs no fixtures.
- **`attach_horizon_probabilities` returns NaN when `production_model == "none"`.** Listed as missing in the first pass; it is missing because A2 reversed the behaviour it would assert. Restore the behaviour, then the test.
- **`validate_artifact_metadata` rejects `target_column` set to any `Hit_*` research target** (A3). The current `_allowed_target_columns` makes this test fail by design.
- **A metadata file newer than its booster is a load error** (A3).
- **`Volume_Contraction_Ratio` support in the live frame overlaps the training range** (A7) — a distribution guard, not an equality check.
- Then, unchanged from the first pass:
- `train_horizon` for 21d and 42d (only `HORIZON_SPECS[0]` is exercised).
- Inference raising when a pillar column is absent from the live frame, and when only one of 26 is present (I1 / A9).
- `fold_pass_rate` gating: no test that a good pooled result plus a bad fold rate blocks promotion.
- Cross-horizon `source_csv_sha256` agreement, **and agreement with the file actually on disk** (A4).
- ~~A guard rejecting `best_iteration <= 2`.~~ **Added 2026-08-24** — `test_booster_is_degenerate_below_four_trees`, `test_save_xgb_artifact_refuses_degenerate_and_deletes_stale`, `test_horizon_metadata_rejects_degenerate_best_iteration`, `test_train_horizon_skips_a_stump_fold_and_keeps_the_rest`, `test_train_horizon_raises_when_every_fold_is_a_stump`.

---

## 9. Remaining code changes (fixed items removed)

C1, C2, T2, E1, E2 hold. C3/C4 are only partly effective (A5). T3/T4 are no longer verifiable (A4). Still open:

| File | Issue |
|---|---|
| `data/logs/daily/coiled_cobra_xgb_*.json` | **A1 — delete. These are `Hit_*` MFE models being served as `Win_*`.** |
| `config.py:406` | **A2 — `SERVE_ML_RANKER` defaults to True and bypasses `production_model`. Set False.** |
| `ml_ranker.py` `_allowed_target_columns` | **A3 — must return `{label_col(spec)}` only; it currently accepts every `Hit_*` too** |
| `validate_cobra_signals.py` | **A4 — writes into the trainer's filename; give it its own output prefix** |
| `coiled_cobra_ml_training.py` `MIN_BEST_ITERATION` | A5 — 3 trees at `lr=0.01` is still a constant; 76% of 42d OOS rows come from stumps |
| `coiled_cobra.py` `evaluate_coiled_cobra` | A7 — new gates cut 68% of the training population; 3 tests red |
| `coiled_cobra_ml_training.py` `_Random_Rank` | A8 — one draw is not a null distribution; permute |
| `coiled_cobra_ml_training.py` `final_train` | V2 — subtract embargo |
| `coiled_cobra_ml_training.py` `SOURCE_FILENAME` | Pin the CSV; delete the newest-mtime fallback (A4 made this active, not hygiene) |
| `ml_ranker.py` `build_feature_frame` | I1 — raise on missing live feature columns (P0 while A2 stands) |
| `ml_ranker.py` attach path | I2 — all horizons share `source_csv_sha256`, and it must match the file on disk |
| `ml_ranker.py` `resolve_model_paths` | I3 — delete unused 21d-filename landmine |
| `trade_planner.py:293` | I4 — `Priority` still multiplies a probability by two boosts |
| `config.py` universe | L1 — point-in-time names; drop `STATIC_TICKERS` from history |
| evaluation | V1 / A8 — date-block SEs / refuse promote inside noise |

---

## 10. Prioritized fixes

**P0 (second pass) — do before the next scan runs**

These supersede everything below and should be done in order. Steps 1–3 are minutes of work and stop the bleeding.

| # | Action | Item |
|---|---|---|
| A | `SERVE_ML_RANKER = False` in `config.py`. The live book goes back to Score order. | A2 |
| B | Delete `coiled_cobra_xgb_10d.json`, `_21d.json`, `_42d.json` and the three horizon metadata files. They describe a model that was never evaluated. | A1, A3 |
| C | Fix `_allowed_target_columns` to `{label_col(spec)}`. Add a load-time refusal when a metadata file is newer than its booster. | A3 |
| D | Repoint `validate_cobra_signals.py` at its own output prefix, then regenerate the full-universe backtest CSV **after** the A7 gate change and pin it in `SOURCE_FILENAME`. | A4, A7 |
| E | Fix the three failing scanner tests, then keep the suite green as a merge condition. | A7 |
| F | Retrain on `HEAD`. Before `save_xgb_artifact` writes, assert the booster's `base_score` matches `class_balance["positive_rate"]` to within a few points, and assert `feature_columns` matches the booster's own `feature_names`. | A1 |
| G | Raise `MIN_BEST_ITERATION` (or `learning_rate`) so a "contributing" fold is actually a fitted model, and print the stump-weighted share of OOS rows. | A5 |
| H | Replace the single `_Random_Rank` draw with a within-fold permutation null; promote only above the 95th percentile. | A8, V1 |
| I | Raise in `build_feature_frame` when a live feature column is absent, and require more than one non-NaN feature per row. | I1 |

**Do not** treat a post-retrain result as new evidence until D and E are done — the population changed under A7, so the next run is not comparable to anything in §1–§12.

**P0 (first pass) — do before believing any number**
1. ~~Normalize ranks within each fold (or evaluate only per-fold and aggregate), then regenerate every reported metric.~~ **Done — C1 fixed and all three horizons retrained.**
2. ~~Fix `TOP_N_PER_DATE_LEVELS` so per-date cuts actually select.~~ **Done — C2 fixed, `selected_fraction` now exposed on every cut.**
3. ~~Add random and population baselines to the promotion gate.~~ **Done — E1 and E2 fixed, T2 removed hit rate from the gate.**
4. Rebuild the universe point-in-time, or at minimum report every metric as a *lift over the population* rather than an absolute return, and drop `STATIC_TICKERS` from any historical study. L1.
5. ~~Add a secondary target that isn't volatility-loaded — `Forward_Return_Xd > 0` (`Win_*`), or a risk-adjusted `Max_Return / ATR_Pct`, or MFE conditioned on MAE. Keep `Hit_X%` as research.~~ **Done — T1. Primary label is `Win_*`; `Hit_*` is research-only. Promotion still fails, now on average return as well as (for 10d/42d) win rate. Medians no longer go negative.**

**P1 — do before the next training run**
6. ~~Re-run `--backtest` so `Max_Return_*`/`Hit_*`/`Win_*` come from the tested labeling code, then retrain from that CSV.~~ **Done — T3 resolved; labels diffed and results reproduced within noise.**
7. ~~Make the `enrich_mfe_targets` missing-file and close-based-fallback paths fatal.~~ **Done — T4 fixed, both paths now raise.**
8. Add a date-block bootstrap or paired t-stat per fold; refuse to promote on differences inside the noise band. V1.
9. ~~Drop `hit_rate` from the gate; add median return and require win rate ≥ population.~~ **Done with E1/T2.**
10. ~~Reject degenerate folds and refuse to save models with `best_iteration < 3`.~~ **Done — C3 and C4. `ml_ranker` also rejects a degenerate artifact.**
11. Raise loudly on missing live feature columns; verify `source_csv_sha256` agreement across horizons. I1, I2.
12. Remaining test gaps are listed in section 8.
18. Fix I5 (`for target in research_targets` rebind) and rewrite horizon metadata so `target_column` is `Win_*`.

Note after T3: `SOURCE_FILENAME` still names a non-existent `..._2026-07-17.csv`, so resolution falls through to newest-by-mtime. The stale `_Large.csv` (the enriched-label source) is still sitting in `data/logs/daily/` and would be picked if its mtime ever led. Item 16 below is now a reproducibility risk, not just hygiene — training runs should pass `--csv` explicitly until it is fixed.

**P2 — hygiene**
13. Subtract the embargo from `final_train`. V2.
14. Rename `ML_Pred_Return` → `ML_Prob` (or stop boosting it), and fix the `trade_planner` / `coiled_cobra` semantics and the stale test. I4.
15. Delete `resolve_model_paths`. I3.
16. Pin the source CSV explicitly instead of newest-by-mtime.
17. Drop the 0.5-threshold precision/recall/F1 from the report or evaluate at the actual selection cut.

---

## 11. How to pick up tomorrow

1. Read **§A first**, then `docs/CoiledCobraML-Handoff.md` (which currently documents `SERVE_ML_RANKER` as intended behaviour and needs the same correction).
2. Work §10's P0 table A→I in order. A, B, C are the stop-the-bleeding steps.
3. Do **not** promote, loosen the gate, or retrain the same 26 features on Win. A6 shows that feature set is a volatility proxy in both label directions.
4. Next research: one feature group at a time, or a risk-adjusted label; keep the same promotion table.
5. Docker: `src/` is not bind-mounted; rebuild or `docker cp` before trusting container training. `data/` **is** shared, which is how A4 happened.

~~Pinned CSV: `data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv`.~~ **That file has been overwritten (A4) — it is now 556 rows / 28 symbols and is not a training corpus.** Regenerate before the command below is meaningful:

```bash
# 1. regenerate the full-universe corpus AFTER the A7 gate change, under a pinned name
docker exec finance_vibe python -m finance_vibe.coiled_cobra_backtest daily --backtest

# 2. train explicitly against it — never rely on newest-by-mtime
docker exec finance_vibe python -m finance_vibe.coiled_cobra_ml_training \
  --csv /app/data/logs/daily/<pinned-file>.csv
```

### Reproducing the §A findings

```python
import json, xgboost as xgb
b = xgb.Booster(); b.load_model("data/logs/daily/coiled_cobra_xgb_21d.json")
cfg = json.loads(b.save_config())
print(cfg["learner"]["learner_model_param"]["base_score"])   # 0.093 -> not a Win_21d model
meta = json.load(open("data/logs/daily/coiled_cobra_ml_metadata_21d.json"))
print(meta["target_column"], meta["class_balance"]["positive_rate"])  # Win_21d 0.559
```

Then compare `coiled_cobra_ml_oos_21d.csv`'s `ML_Prob_Win_21d` (mean ≈ 0.58) against the booster's own predictions (mean ≈ 0.10). Same training run, two different models.

## 12. Legacy 6-feature “production” vs current stack

The importance PNG **Coiled Cobra ML — Feature Importances (Forward_Return_2w)** (XGB gain + LightGBM split/gain, six features, identical rank: `Pct_From_EMA50` > `ATR_Pct` > `Pct_From_EMA20` > Fib786 > Fib618 > `Score`) is the **old regression ranker**. It is not a walk-forward scoreboard and it is not loaded by `attach_horizon_probabilities`.

| | Legacy | Current (Batch D, on disk) |
| - | ------ | -------------------------- |
| Target | Raw `Forward_Return_2w` | `Win_10d` / `Win_21d` / `Win_42d` |
| Features | 6 including **Score** and **Fib** | 26; Score/Fib forbidden |
| Models | XGB + LGB | XGB only |
| Live | Historical `coiled_cobra_xgb_model.json` | Horizon json; **none promoted** |
| Importance | EMA50 + ATR + Fib + Score | `ATR_Pct` still ~0.21–0.25; then dist-to-high / EMA % |

A numeric bake-off of the old booster vs the new one on `coiled_cobra_ml_oos_10d.csv` was **not** run. Current 10d Win top-decile **loses** to Score, random, and population on mean return. Neither model is promotion-ready.

## Bottom line

The three `production: none` decisions are still correct, and they are also no longer being enforced.

The first pass got the science right and the shipping wrong. Every conclusion in §1–§12 is about `Win_*` models that were evaluated honestly, failed honestly, and were correctly refused promotion. None of that describes what the scanner is actually doing. The artifacts on disk are `Hit_15Pct_10d` / `Hit_20Pct_21d` / `Hit_50Pct_42d` MFE models (**A1**); `SERVE_ML_RANKER = True` sorts the live book with them regardless of the gate (**A2**); the metadata that would have exposed the swap was rewritten by hand and the validator that would have caught it was widened to permit it (**A3**); and the corpus that would let anyone check was overwritten by a validation script sharing the trainer's filename (**A4**).

The through-line is that four separate safety mechanisms — the promotion flag, the `target_column` assertion, the artifact/metadata pairing, and `source_csv_sha256` — each existed, each recorded the right thing, and none of them was load-bearing. Every one was satisfiable by editing the record rather than the system. Adding a fifth check is worth less than making one of the existing four able to fail a run.

On the modelling itself: **Score has no measurable cross-sectional selection value versus random on the honest cuts, and Win-trained XGB is worse than that** — its return declines monotonically as the cut tightens and its top decile trails a random decile at all three horizons (**A6**). `ATR_Pct` dominates under `Hit_*` and under `Win_*` with the sign flipped, which is the same finding twice: these 26 features encode volatility and not much else. The next experiment should change the feature set or the label, not the hyperparameters.

Pickup: **§A**, then §10's P0 table, then `docs/CoiledCobraML-Handoff.md`.