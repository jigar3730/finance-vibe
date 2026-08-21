# Machine learning primer (this repo)

You do not need a graduate course to understand what Finance Vibe trains. You do need to know **what is allowed at signal time**, **why time cannot be shuffled**, and **what Spearman is for**. Train/deploy commands live in [MLOps.md](/docs/mlops). The exact 10 columns live in [CoiledCobraML.md](/docs/cobra-ml).

---

## 1. Supervised learning in one paragraph

You collect examples `(X, y)`:

- **X** — numbers known **when the coil prints** (pillars, EMA distances, ATR_Pct).
- **y** — a number known only **later** (forward return vs QQQ over ~2 weeks).

A model learns `f(X) ≈ y`. On a live scan you compute `f(X_today)` and **sort**. That sort is `ML_Rank`. The rubric **Score** already decided the coil was legal.

---

## 2. Regression used as ranking

| Task | Question | Here |
| ---- | -------- | ---- |
| Classification | Will it win? yes/no | Not used |
| Regression | How much Rel_Forward_2w? | **Training loss** |
| Ranking | Which name first? | **How we use `f(X)`** |

A 2-week **relative** return can be small and still useful if the **order** of names is right. We do not size positions from `ML_Pred_Return`.

`Rel_Forward_2w` = stock return over H bars minus QQQ over the same dates. Daily H = 10 sessions; weekly H = 2 bars. Absolute `Forward_Return_2w` is the fallback if relative is missing (prefer fixing QQQ ingest).

---

## 3. Leakage (the silent cheat)

If a column is only known after the signal (exit, fill, R-multiple), putting it in X makes history look smart and live ranks worthless. Training **drops** execution-style columns. **Score** and **Grade** are also out of X: Score is a mix of the pillars; Grade is a bin of Score. Trees would copy the rubric instead of finding residual pattern.

**Episode leakage:** a coil that lasts five bars is one story. Training keeps **`Is_New_Coil == True`** only (age = 1). Continuation bars share overlapping forward windows.

---

## 4. Why not random K-fold?

K-fold shuffles rows. A 2026 coil can train a model that then “predicts” 2024. In markets that is peeking at the future.

This project **cuts on `Signal Date`**: last 26 weeks = test, 26 weeks before = val, earlier = train.

**Embargo = 2 weeks** (same as the label). A train signal near the val cut has a `y` computed from prices **inside** val. The embargo leaves a gap so train labels do not overlap val/test prices.

If any split is empty, you do not have enough history. A 3-ticker smoke backtest is not a training set.

---

## 5. Sample weights

Violent names have noisier 2-week returns. Weighting **by** ATR_Pct would train mostly on lottery tickets. The code uses **`1 / ATR_Pct`** (non-finite weights → train median). That is fit-time only, not portfolio risk parity.

---

## 6. Gradient boosting (XGBoost and LightGBM)

Both build **many shallow trees**. Each tree fits the **errors** of the previous ones (boosting). Shared knobs: depth 4, slow learning rate 0.01, up to 400 trees, 80% row/feature bagging, **early stopping** on val MAE (40 rounds).

**MAE (L1)** is the training objective: typical absolute miss in return space. **MSE/RMSE** squashes a few +80% names into the loss; RMSE on test is often ugly. Compare **MAE and Spearman**, not RMSE alone.

Two libraries: if they agree on Spearman and importances, the pattern is less likely a quirk of one package.

Missing X stays NaN; trees split around missing values. No median fill.

---

## 7. Spearman is the production metric

| Metric | Asks | Use |
| ------ | ---- | --- |
| MAE | How far off on average? | Fit quality (matches L1) |
| RMSE | How bad are tails? | Audit, not promotion |
| **Spearman** | Did we put names in the **right order**? | **`ML_Rank`** |

Pearson cares about linear *magnitudes*. We only sort, so Spearman is the right question.

Test Spearman **> 0** (and not collapsing vs val) → weak ranking edge. **~0** → ignore ML, use Score. **Negative** → do not leave those files in `logs/daily/`.

---

## 8. Inference (fail-soft)

`ml_ranker.py` looks only in the **active mode** silo (`data/logs/daily/` for the default scan). It loads XGB JSON and LGB text, **skips** a booster whose feature names are not the current 10 columns (old Score+Fib models), then **averages** predictions. Rank 1 = highest predicted relative return. No file or all NaN → Score sort; the pipeline still runs.

The helper may multiply a **1.25×** boost when coil width ≤ 4 ATR or risk ≤ 3% of close. That is a business rule on top of `f(X)`, not part of training.

---

## 9. What “deploy” means here

There is no MLflow. Deploy = artifacts sit in `/app/data/logs/daily/`:

- `coiled_cobra_xgb_model.json`
- `coiled_cobra_lgb_model.txt`
- `coiled_cobra_ml_model_metadata.json`

The next `coiled_cobra.py` process loads them. Dashboard **ML daily** chip is found vs missing.

**Train:** [MLOps.md §5](/docs/mlops). **Contract:** [CoiledCobraML.md](/docs/cobra-ml). **Curriculum:** [Learn.md](/docs/learn) Track C.
