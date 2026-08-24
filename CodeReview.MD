## Verdict

**WARNING — the plumbing is sound, the headline numbers are not.**

I could not find true feature leakage, a random split, an embargo error, a blind XGB+LGB average, or a `best_iteration` mismatch. Those parts of the brief were implemented correctly. But the **evaluation layer that produces "ML beats Score" is broken in two independent ways**, and the **target itself is volatility-biased**, so the reported top-decile returns cannot be used for a promotion decision. Separately, the models on disk were trained on labels that were never produced by the tested backtest labeling code.

Direct answer to your main question: **No — not the `top_X%` numbers.** The only selection metrics in the current metadata that are methodologically valid are the `top_N_per_date` ones, and those show ML's edge is entirely in the *mean* with a flat median and a slightly *worse* win rate.

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
| Targets correct | PASS (with caveat below) | `Hit_10Pct_10d` / `Hit_15Pct_21d` / `Hit_25Pct_42d` wired to the right forward columns and thresholds |

---

## 2. Critical bugs

**C1 — Pooled top-X% ranking mixes probabilities from 17 different models (P0).**
`top_fraction` (`coiled_cobra_ml_training.py:730-738`) is applied to the concatenated OOS frame, so it ranks fold-1's probabilities against fold-17's. XGBoost probabilities are not calibrated across folds — a fold trained through a bull stretch emits systematically higher probabilities than one trained through 2022. The pooled top decile therefore concentrates in whichever periods produced high probabilities, not in the best setups. `Score` is on a fixed 70–98 scale and suffers none of this, so **the Score-vs-ML comparison is apples-to-oranges**.

The size of the distortion is measurable from your own metadata. For 42d, with roughly equal fold sizes:

- mean of the 16 fold-level XGB top-decile averages: **+7.60%**
- pooled top-decile average: **+4.04%**

If the pooled cut were drawing ~10% from each fold it would land near the fold mean. It lands at half, which is only possible if the pooled decile is selecting a materially different, period-concentrated sample.

**C2 — `top_10_per_date` selects 77% of the population (P0).**
Median signals per date is 8 (mean 9.5, 25th pct 5). So `top_10_per_date` = 15,387 of 19,947 rows, and `top_20_per_date` = 19,290. Yet `walk_forward.pooled.score_top10_avg_return` / `ml_top10_avg_return` — the fields a reader will treat as the headline — are exactly these near-population numbers, labeled "top10". `TOP_N_PER_DATE_LEVELS = (20, 10, 5)` at line 86 needs to be something like `(5, 3, 1)` for this universe.

**C3 — Fold 1 contributes ~1,150 predictions from an untrained model (P1).**
`MIN_TRAIN_ROWS = 80` (line 81) admits a fold with `n_train=505`, and both the 10d and 21d fold-1 models stopped at `best_iteration=0` — a single tree at `learning_rate=0.01`, i.e. a near-constant predictor. Those rows are pooled into OOS as if they were real predictions.

**C4 — The saved 21d artifact is degenerate (P1).**
`coiled_cobra_xgb_21d.json` has `best_iteration = 2`. Three trees at lr 0.01 is a constant ≈ base rate. Nothing in `train_horizon` refuses to save a model that early-stopped at ~0 iterations, and nothing in `ml_ranker` refuses to load one. It's harmless today only because `production_model = "none"`.

---

## 3. Data leakage risks

No classic leakage. Two soft lookahead problems that matter more than they look:

**L1 — Survivorship / hindsight universe (P0 for interpretation).**
`config.py:48-69`: the universe is today's `most_actives` screener plus `STATIC_TICKERS`, which is a hand-picked list of known decade winners (NVDA, PLTR, AXON, DUOL, CELH, SMCI, HOOD, COIN, MELI, DECK…). You then backtest that list over 10 years. The tell is in the data: **random selection inside this universe returns +3.83% per 42 trading days**, roughly +24% annualized. Delisted and dead names are absent entirely. This inflates every absolute number and, worse, it can *manufacture* an apparent edge for "buy the volatile name," because in this universe volatility survived.

**L2 — Split/dividend-adjusted history (P2).** Raw CSVs are `auto_adjust=True` as of download date. Ratio features and returns mostly cancel this out, so it's minor, but it is not point-in-time.

---

## 4. Target-definition problems

**T1 — `Hit_X%` (max favorable excursion) is monotone in volatility, and the model learned exactly that (P0).**
MFE over a fixed window is close to a volatility statistic. Watch what happens to the 42d XGB as you tighten the cut:

| Selection | Avg | **Median** | **Win rate** | Hit rate |
|---|---:|---:|---:|---:|
| population | +3.83% | +1.89% | 0.560 | 0.161 |
| top 25% | +5.69% | +0.14% | 0.502 | 0.329 |
| top 10% | +4.04% | **−4.66%** | **0.447** | 0.374 |
| top 5% | +4.26% | **−7.83%** | **0.406** | 0.438 |

Hit rate climbs from 0.161 to 0.438 — the model is very good at its label. Median return goes to −7.8% and win rate falls below the population. 10d and 21d show the same shape (10d top-5% avg is *negative*, −1.27%). The model is not finding better setups; it is finding wider distributions. Your stated success criteria were higher average **and** higher win rate, and this target structurally trades the second away for the first.

**T2 — `hit_rate` is invalid as a promotion criterion (`compare_rankers:820-825`) (P1).** It is the model's own training objective. XGB will beat Score on it essentially always, so it adds nothing to the gate while making it look stricter.

**T3 — The training labels were built by a fallback path, not by the backtest (P1).** The source CSV (`coiled_cobra_backtest_trades_2026-08-24_Large.csv`, 89 columns) contains **no** `Forward_Return_10d`, **no** `Max_Return_*`, **no** `Win_*`, **no** `Hit_*`. So every label came from two runtime derivations: `Forward_Return_10d` aliased off `Forward_Return_2w` (`_ensure_horizon_targets`), and all `Max_Return_*`/`Hit_*` synthesized by `enrich_mfe_targets`. The alias happens to be correct (legacy daily `2w` = 10 bars), but the new `HIT_LABEL_SPECS`/`_horizon_fields` code in `coiled_cobra_backtest.py:37-58,298-329` — the code your tests actually cover — never touched this dataset.

**T4 — `enrich_mfe_targets` silently mixes two label definitions (P1).**

```211:254:src/finance_vibe/coiled_cobra_backtest.py
        if not path or not os.path.isfile(path):
            n_miss += 1
            pieces.append(g)
            continue
```

A symbol whose raw file is missing keeps the close-based `Hit_*` that `_ensure_horizon_targets` already wrote from `Forward_Return_*`. Those rows train on "close ≥ +25%" while everyone else trains on "any high ≥ +25%". Only a count is printed. It happened to be 0 missing this run.

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

---

## 7. Evaluation problems

Beyond C1/C2 above:

**E1 — "Beats Score" is a near-worthless bar, and nothing enforces "beats random" (P0).** On the scale-safe per-date cuts, Score is indistinguishable from — sometimes worse than — random:

| 42d, top 5 per date | Avg | Median | Win rate |
|---|---:|---:|---:|
| population | +3.83% | +1.89% | 0.560 |
| random | +3.41% | — | — |
| **Score** | **+3.38%** | +1.73% | 0.550 |
| XGB | +5.40% | +1.72% | 0.537 |
| Logistic | +5.53% | — | — |

Score's top per-date picks underperform the population it's drawing from. `compare_rankers` computes a random baseline and then never uses it in the promotion condition (lines 820-825).

**E2 — The scale-safe metrics say the edge is tail-only.** Across all three horizons and every per-date cut, XGB's mean beats Score by 30–60% while the **median is identical to two decimals** and the **win rate is slightly lower**. That is the signature of a few large right-tail winners, not of better setup selection. In a 413-name hindsight universe, that tail is exactly what L1 would fabricate.

**E3 — Logistic regression beats XGBoost on every pooled cut** (42d top-5%: 14.42% vs 4.26%). Either the tuned gradient booster is worse than a linear model on 26 features, or — far more likely — this is C1 again, which is itself the cleanest available proof that the pooled cuts are artifacts.

**E4 — Classification metrics are uninformative.** At a fixed 0.5 threshold with a 15–17% base rate and `scale_pos_weight=1`, recall is under 1%. Harmless since you don't gate on them, but they shouldn't appear in a report.

---

## 8. Missing tests

The most important gap: **there is no positive-control test that promotion can ever return `True`.** `test_promote_requires_higher_return_and_win_rate` only asserts `False`. Every horizon returned "none" and no test distinguishes "the gate is correctly strict" from "the gate is stuck off."

Also missing:
- Any test of `enrich_mfe_targets` — the code path that generated 100% of your training labels.
- Any test that `top_fraction` / `top_n_per_date` return the expected row counts, or that per-fold normalization is applied.
- `train_horizon` for 21d and 42d (only `HORIZON_SPECS[0]` is exercised).
- `attach_horizon_probabilities` returning NaN when `production_model == "none"` — the actual production gate.
- Inference raising when a pillar column is absent from the live frame (I1).
- `fold_pass_rate` gating: no test that a good pooled result plus a bad fold rate blocks promotion.
- Cross-horizon `source_csv_sha256` agreement.
- A guard rejecting `best_iteration <= 2`.

---

## 9. Exact files / lines needing change

| File | Lines | Issue |
|---|---|---|
| `coiled_cobra_ml_training.py` | 730-738, 771-801 | C1 — rank within fold, or z-score/percentile-normalize per fold before pooling |
| `coiled_cobra_ml_training.py` | 86 | C2 — `TOP_N_PER_DATE_LEVELS` → `(5, 3, 1)` |
| `coiled_cobra_ml_training.py` | 1313-1327 | C2 — `score_top10_*` / `ml_top10_*` are near-population; rename or repoint |
| `coiled_cobra_ml_training.py` | 820-825, 1181-1187 | T2, E1 — drop `hit_rate`, add median + beats-random + beats-population |
| `coiled_cobra_ml_training.py` | 81 | C3 — raise `MIN_TRAIN_ROWS`, skip folds with `best_iteration < 3` |
| `coiled_cobra_ml_training.py` | 1218-1221 | V2 — subtract the embargo from `final_train` |
| `coiled_cobra_ml_training.py` | 400-438 | T4 — make the close-based `Hit_*` fallback fatal, not a warning |
| `coiled_cobra_ml_training.py` | 115, 269-289 | reproducibility — hardcoded date + mtime-based CSV fallback can silently pick `_small.csv` |
| `coiled_cobra_backtest.py` | 211-254 | T4 — raise on `n_miss > 0` |
| `ml_ranker.py` | 163-195 | I1 — raise on missing live feature columns |
| `ml_ranker.py` | 307-318 | I2 — verify all horizons share `source_csv_sha256` |
| `ml_ranker.py` | 61-82 | I3 — delete `resolve_model_paths` |
| `trade_planner.py` | 283-295 | I4 — probability treated as expected return |
| `coiled_cobra.py` | 930-935 | I4 — boost mutates the stored prediction, can exceed 1.0 |

---

## 10. Prioritized fixes

**P0 — do before believing any number**
1. Normalize ranks within each fold (or evaluate only per-fold and aggregate), then regenerate every reported metric. C1.
2. Fix `TOP_N_PER_DATE_LEVELS` so per-date cuts actually select. C2.
3. Add random and population baselines to the promotion gate; ML must beat Score **and** random **and** the population it draws from. E1.
4. Rebuild the universe point-in-time, or at minimum report every metric as a *lift over the population* rather than an absolute return, and drop `STATIC_TICKERS` from any historical study. L1.
5. Add a secondary target that isn't volatility-loaded — `Forward_Return_Xd > 0` (`Win_*`), or a risk-adjusted `Max_Return / ATR_Pct`, or MFE conditioned on MAE. Keep `Hit_X%` as research. T1.

**P1 — do before the next training run**
6. Re-run `--backtest` so `Max_Return_*`/`Hit_*`/`Win_*` come from the tested labeling code, then retrain from that CSV. T3.
7. Make the `enrich_mfe_targets` missing-file and close-based-fallback paths fatal. T4.
8. Add a date-block bootstrap or paired t-stat per fold; refuse to promote on differences inside the noise band. V1.
9. Drop `hit_rate` from the gate; add median return and require win rate ≥ population. T2.
10. Reject degenerate folds and refuse to save models with `best_iteration < 3`. C3, C4.
11. Raise loudly on missing live feature columns; verify `source_csv_sha256` agreement across horizons. I1, I2.
12. Add the missing tests, starting with the positive-control promotion test and `enrich_mfe_targets`.

**P2 — hygiene**
13. Subtract the embargo from `final_train`. V2.
14. Rename `ML_Pred_Return` → `ML_Prob` (or stop boosting it), and fix the `trade_planner` / `coiled_cobra` semantics and the stale test. I4.
15. Delete `resolve_model_paths`. I3.
16. Pin the source CSV explicitly instead of newest-by-mtime.
17. Drop the 0.5-threshold precision/recall/F1 from the report or evaluate at the actual selection cut.

---

## Bottom line

The three "production: none" decisions are correct, and the fold pass rates (5.9% / 11.8% / 12.5% against a 60% requirement) are the honest signal here. But they're correct partly by accident — the pooled metrics feeding the first half of that gate are artifacts, and the reason ML fails the win-rate test is that the `Hit_X%` target is a volatility proxy.

The one conclusion I'd actually stand behind from this run: **on the scale-safe per-date cuts, the existing `Score` has no measurable cross-sectional selection value at all** (42d top-5-per-date: Score +3.38% vs random +3.41% vs population +3.83%). That is a more actionable finding than anything about the ML models, and it means "beat Score" was never the right bar. Fix C1, C2 and E1, re-run, and compare against random and the population instead.