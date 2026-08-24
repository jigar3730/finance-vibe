**Resume / pickup (2026-08-24):** full handoff in **`docs/CoiledCobraML-Handoff.md`**. This file is the audit log. Do not promote. Do not retrain the same 26 features on `Win_*`. Next work is new features or a risk-adjusted label, plus hygiene (I1–I4, L1, V1/V2, metadata `target_column` clobber).

---

## Verdict

**WARNING — the plumbing is now mostly sound; there is still no tradable selector.**

Walk-forward ranking (C1), per-date breadth (C2), stump folds/artifacts (C3/C4), native labels (T3/T4), and the promotion gate (E1/E2/T2) were fixed on 2026-08-24. Primary labels are `Win_*` (T1). All three horizons remain **`production_model: none`**. Win-trained XGB does not beat Score, random, or the population on average return. Hit-trained mean edge was a volatility lottery.

The 6-feature `Forward_Return_2w` XGB+LGB importance chart is a **different model family**, not a prior version of this stack (see §12).

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
| Promotion gate held | PASS | all three horizons = `none` |
| Targets correct | PASS (Win primary) | Train `Win_10d` / `Win_21d` / `Win_42d`; `Hit_*` is `hit_col` + research only. **JSON `target_column` is currently wrong — see I5.** |

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

**I5 — Horizon metadata `target_column` is clobbered by the research loop (P1). — OPEN 2026-08-24.**
`train_horizon` sets `target = label_col(spec)` (`Win_*`) and trains on that. Later `evaluate_research_targets` does `for target in spec["research_targets"]`, which rebinds the same name. Metadata then writes `"target_column": target` as the **last research Hit column** (`Hit_15Pct_10d` / `Hit_20Pct_21d` / `Hit_50Pct_42d`) while `prob_column` is correctly `ML_Prob_Win_*` and `schema_version` is 5. Models on disk were trained on Win; the JSON field lies. Fix: rename the loop variable, rewrite the three metadata files (or retrain). Until then treat stdout + `HORIZON_SPECS` + OOS CSVs as truth.

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

Still missing:
- `train_horizon` for 21d and 42d (only `HORIZON_SPECS[0]` is exercised).
- `attach_horizon_probabilities` returning NaN when `production_model == "none"` — the actual production gate.
- Inference raising when a pillar column is absent from the live frame (I1).
- Metadata `target_column` equals `label_col(spec)` after research-target evaluation (I5).
- `fold_pass_rate` gating: no test that a good pooled result plus a bad fold rate blocks promotion.
- Cross-horizon `source_csv_sha256` agreement.
- ~~A guard rejecting `best_iteration <= 2`.~~ **Added 2026-08-24** — `test_booster_is_degenerate_below_four_trees`, `test_save_xgb_artifact_refuses_degenerate_and_deletes_stale`, `test_horizon_metadata_rejects_degenerate_best_iteration`, `test_train_horizon_skips_a_stump_fold_and_keeps_the_rest`, `test_train_horizon_raises_when_every_fold_is_a_stump`.

---

## 9. Remaining code changes (fixed items removed)

C1, C2, C3, C4, T2, T3, T4, E1, E2 are done. Still open:

| File | Issue |
|---|---|
| `coiled_cobra_ml_training.py` `evaluate_research_targets` | I5 — rename loop var so metadata `target_column` stays `Win_*` |
| `coiled_cobra_ml_training.py` `final_train` | V2 — subtract embargo |
| `coiled_cobra_ml_training.py` `SOURCE_FILENAME` | Pin `coiled_cobra_backtest_trades_2026-08-24.csv`; do not newest-mtime |
| `ml_ranker.py` `build_feature_frame` | I1 — raise on missing live feature columns |
| `ml_ranker.py` attach path | I2 — all horizons share `source_csv_sha256` |
| `ml_ranker.py` `resolve_model_paths` | I3 — delete unused 21d-filename landmine |
| `trade_planner.py` / `coiled_cobra.py` | I4 — do not treat/boost `ML_Pred_Return` as expected return |
| `config.py` universe | L1 — point-in-time names; drop `STATIC_TICKERS` from history |
| evaluation | V1 — date-block SEs / refuse promote inside noise |

---

## 10. Prioritized fixes

**P0 — do before believing any number**
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

1. Read **`docs/CoiledCobraML-Handoff.md`**.
2. Do **not** promote, loosen the gate, or retrain the same 26 features on Win.
3. Next research: one feature group at a time, or a risk-adjusted label; keep the same promotion table.
4. Hygiene: I5 (metadata), pin `--csv` / `SOURCE_FILENAME`, then I1–I4, V1/V2, L1.
5. Docker: `src/` is not bind-mounted; rebuild or `docker cp` before trusting container training.

Pinned CSV: `data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv`.

```bash
docker exec finance_vibe python -m finance_vibe.coiled_cobra_ml_training \
  --csv /app/data/logs/daily/coiled_cobra_backtest_trades_2026-08-24.csv
```

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

The three `production: none` decisions are still correct. Training on `Win_*` removed the Hit-model's mean-vs-median pathology and showed that advantage was the volatility proxy. **Score has no measurable cross-sectional selection value vs random on the honest cuts, and neither does Win-trained XGB.** Pickup: `docs/CoiledCobraML-Handoff.md`.