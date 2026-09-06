# Quantitative ML Reference Manual

A theory-to-code map for Finance Vibe. Each section states the finance /
data-science idea first, then points at the module that implements (or
deliberately does **not** yet implement) it.

Companion specs:

- ML pipeline contract: [`../architecture/coiled_cobra_ml.md`](../architecture/coiled_cobra_ml.md)
- Cobra scorecard: [`coiled_cobra_rubric.md`](coiled_cobra_rubric.md)
- Macro Vibe Score: [`scoring_logic.md`](scoring_logic.md)
- Walk-forward infra: [`../architecture/backtest_and_backfill.md`](../architecture/backtest_and_backfill.md)
- Experiments: [`../labs/README.md`](../labs/README.md)

**Code of record for the ML baseline** (as of this writing):

| Contract | Value | Location |
| -------- | ----- | -------- |
| Task | Regression on continuous forward return | `coiled_cobra_ml_training.py` |
| Target $Y$ | `Forward_Return_2w` | `TARGET_COL` |
| Features $X$ | 6 pre-signal columns | `FEATURE_COLS` |
| Leakage drop | Execution / outcome fields | `LEAKAGE_COLS` |
| Split | Rolling 26w test / 26w val / rest train | `_temporal_split()` |
| Models | `XGBRegressor` + `LGBMRegressor` (MAE / L1) | `_train_and_report()` |
| Inference | Soft rank only | `ml_ranker.py` |

CatBoost, Optuna, and SHAP are **not** production dependencies. They appear
only in labs 03, 05, and 06.

---

## 1. Feature Engineering & Technical Analysis

### 1.1 Why these indicators exist (finance)

Coiled Cobra is a **compression-before-expansion** scanner, not a mean-reversion
discount hunter. The live rubric in `coiled_cobra.py` scores:

| Pillar | Function | Finance rationale |
| ------ | -------- | ----------------- |
| Volume profile shelf | `evaluate_volume_profile_shelf` | Auction-market accumulation (high-volume node / POC) |
| MACD compression | `macd_compression_score` | Tight $\lvert\mathrm{MACD}-\mathrm{Signal}\rvert / \mathrm{ATR}$ = coiled energy |
| Structure | `structure_score` | Close above a rising EMA50, ideally EMA50 > EMA100 |
| Relative strength vs QQQ | `rs_score` | Prefer names already leading the tape |
| Coil width | `coil_width_score` | $N$-bar range / ATR — tight base before expansion |
| MACD cross (optional trigger) | rubric add-on | Early expansion tell |
| Fib bonus | `fibonacci_score` | Context only; demoted from a hard gate |

`add_macro_indicators()` builds the raw series the scorecard reads: EMA20/50/100,
MACD 12/26/9, RSI 14, ATR 14, rolling Fib 61.8 / 78.6.

The **ML feature set is a strict subset** of that geometry. Trees never see the
volume histogram, RS, or coil-width score directly — only the six numeric
distances exported on each backtest row:

```python
FEATURE_COLS = [
    "Score",            # 100-pt rubric at the signal bar
    "Pct_From_EMA20",   # (Close − EMA20) / EMA20
    "Pct_From_EMA50",
    "Pct_From_Fib618",
    "Pct_From_Fib786",
    "ATR_Pct",          # ATR / Close
]
```

`Grade` is excluded on purpose: it is a binned copy of `Score` (A ≥ 85, B ≥ 70).
Including both wastes split budget and injects collinearity. Inference rebuilds
the same frame in `ml_ranker.build_feature_frame()` from either precomputed
`Pct_From_*` columns or raw Close / EMA / Fib / ATR fields.

Macro Vibe Score (`analysis_engine.build_features`) is a **separate** SMA-based
layer: trend alignment, MACD/RSI momentum, pullback timing, CCI MAD, RSI caps.
It is used for walk-forward gating in `pipeline_backtest.py`, not as an ML
input today.

### 1.2 Data-science properties of this feature space

**Stationarity.** Raw prices are integrated. Every ML column is already a
*relative* transform — percent distance or ATR scaled by Close — so a \$20 name
and a \$400 name share a comparable scale. That is why the trainer does not
apply StandardScaler or RobustScaler: tree splits are monotone-invariant, and
the features are already dimensionless.

**Scaling and sample weights.** Absolute weekly returns still fan out with
volatility. The trainer passes `sample_weight=ATR_Pct` so high-beta paths do
not dominate MAE purely via scale. Non-finite / non-positive weights fall back
to the **train** median of `ATR_Pct`. Inverse-vol weights (`1 / ATR_Pct`) are a
research alternative, not the baseline — see Lab 01 / Lab 07.

**Collinearity.** `Score` is an additive function of geometry the other five
columns partially encode (EMA distances, Fib proximity, ATR). Trees tolerate
this better than linear models, but gain importance will *share* credit across
the cluster. Dropping `Grade` was the first collinearity cut. Lab 01 is the
place to drop `Pct_From_Fib786` or `Score` and watch importance re-weight.

**Signal-to-noise.** Weekly coil features predict a noisy continuous return.
Expect $R^2$ near zero and MAE on the order of the typical $|Y|$ — the model
is a **ranker**, not a price forecast. `ml_ranker` treats `ML_Pred_Return` as
a soft confirmation after the rubric, structure, and risk gates.

### 1.3 Lab hook

[Lab 01 — Indicator Sensitivity](../labs/01_indicator_sensitivity.md): add or
remove one column from `FEATURE_COLS` and compare XGBoost gain ranks.

---

## 2. Target Engineering & Horizon Shifts

### 2.1 Constructing $Y$

`coiled_cobra_backtest.py` writes four close-to-close forward returns at the
**signal bar** index `idx`, independent of whether the simulated order filled:

$$
Y_h = \frac{\mathrm{Close}[t+h] - \mathrm{Close}[t]}{\mathrm{Close}[t]}
$$

| Column | $h$ (weekly bars) | Typical use |
| ------ | ----------------: | ----------- |
| `Forward_Return_2w` | 2 | **Baseline $Y$** — short tactical rank |
| `Forward_Return_5w` | 5 | ~1-month swing |
| `Forward_Return_13w` | 13 | ~1-quarter expansion |
| `Forward_Return_26w` | 26 | ~half-year hold |

If $t+h$ does not exist, the cell is `None` / NaN. `_load_and_prepare()` drops
rows with a missing **active** target. Longer horizons therefore shrink the
sample at the right edge of every ticker — a mechanical bias, not a market
regime.

`no_fill` rows are **kept**. The continuous target does not require a fill;
dropping them would condition on a future path (limit-touch) and leak.

### 2.2 Classification vs regression, imbalance, noise

The baseline is **regression** (`reg:absoluteerror` / `regression_l1`). A
binary "hit T1" label (`Target_Label`) exists on the CSV but is post-trade
and sits in `LEAKAGE_COLS` — do not train on it.

If you binarize $Y$ (e.g. $Y_{13w} > 0$), class balance will shift with
horizon and with the bullish bias of a long-only coil scanner. Longer $h$
usually raises the hit rate *and* the variance: more time for drift, fatter
tails, lower signal-to-noise for a 6-feature coil snapshot. MAE can look
"worse" in absolute terms simply because $|Y|$ is larger; always compare
MAE / $\mathrm{std}(Y)$ or Spearman rank vs realized $Y$.

### 2.3 Leakage rules for targets

| Allowed | Forbidden |
| ------- | --------- |
| Forward close-to-close from the signal bar | `Outcome`, `Exit Price`, `R Multiple` |
| Features known at $t$ | `Stock Entry` / Stop / Targets (planned from $t$ but unused as $X$ by contract) |
| `no_fill` rows with a defined $Y_h$ | Random shuffle of rows before a time split |

Lookahead in **feature** space is a different bug: any indicator that uses
bars $> t$ on the signal row is leakage. The backtest evaluates
`evaluate_coiled_cobra()` on `df.iloc[:i+1]` (causal window). Do not "fix"
NaNs in $Y$ by forward-filling — that invents future prices.

### 2.4 Lab hook

[Lab 02 — Target Horizon Shift](../labs/02_target_horizon_shift.md): change
`TARGET_COL` from `Forward_Return_2w` to `Forward_Return_13w` and compare
label distribution, NaN attrition, and MAE / RMSE.

---

## 3. Model Architectures & Ensembling

### 3.1 Why gradient-boosted trees

Financial $X$ here is low-dimensional, mixed-scale, and weakly nonlinear
(thresholds like "inside 0.5 ATR of Fib 78.6"). GBDTs:

- split on thresholds without scaling,
- ignore monotone transforms,
- tolerate NaNs natively (no median imputation in this trainer),
- overfit less violently than deep nets on a few thousand weekly coils.

The baseline trains **two** libraries side by side so a conclusion is not an
XGBoost artifact.

### 3.2 What is implemented

| Library | Class | Objective | Regularization in `MODEL_PARAMS` |
| ------- | ----- | --------- | -------------------------------- |
| XGBoost | `XGBRegressor` | `reg:absoluteerror` | `max_depth=4`, `learning_rate=0.01`, `n_estimators=400`, `subsample=0.8`, `colsample_bytree=0.8` |
| LightGBM | `LGBMRegressor` | `regression_l1` | Same depth / rate / bagging |

MAE / L1 is intentional: weekly equity returns are heavy-tailed; squared
error lets a few high-beta paths dominate. `random_state=42`,
`tree_method="hist"` (XGB).

Shallow trees + slow learning + row/column bagging are the current $L2$-like
complexity control. There is no explicit `reg_alpha` / `reg_lambda` sweep in
production — that is Lab 05.

### 3.3 What is not implemented

| Topic | Status |
| ----- | ------ |
| CatBoost / ordered boosting / symmetric trees | Lab-optional ([Lab 03](../labs/03_gbdt_showdown.md)) |
| Native categoricals (`Grade`) | Excluded; CatBoost would be the natural home if re-introduced |
| Stacking / blending the two boosters | Inference uses whichever artifacts `ml_ranker` finds; not a stacked ensemble |
| Early stopping on the val window | Not wired; `n_estimators` is fixed |

`ml_ranker.attach_ml_ranks()` writes `ML_Pred_Return` and dense `ML_Rank`
(1 = highest predicted return). Missing artifacts fail **soft** (null
columns) so the live scan still sorts on rubric `Score`.

### 3.4 Lab hook

[Lab 03 — GBDT Showdown](../labs/03_gbdt_showdown.md): XGBoost vs LightGBM vs
optional CatBoost on fit time, MAE/RMSE, and `Grade` as a categorical.

---

## 4. Validation Integrity & Backtesting

### 4.1 Why standard $K$-fold fails

IID $K$-fold assumes exchangeable rows. A weekly coil in May 2024 shares
overlapping 13-week outcomes with a coil in June 2024 on the same name, and
shares a market factor with every other name that week. Random folds put
"future" structure into the training set:

- **Embargo / overlap leakage:** $Y$ for train row $t$ uses prices through
  $t+h$; a val row at $t+h-1$ has features that sit inside that window.
- **Regime leakage:** 2020-crash and 2023-AI-rally bars get mixed into both
  sides; metrics look stable and then collapse on a true holdout.

The trainer therefore **forbids** `sklearn.model_selection.KFold` on this
table. `_temporal_split()` cuts on `Signal Date` only.

### 4.2 What the repo actually does

**ML trainer** — dynamic rolling cut from `max(Signal Date)`:

- Test: last 26 weeks
- Val: the 26 weeks before that
- Train: everything earlier

Empty partitions raise `RuntimeError`. This is walk-forward in spirit (a
single expanding train + trailing holdout), not a multi-fold purged CV.

**Cobra trade sim** (`coiled_cobra_backtest.py`) — bar $i$ sees only
`df.iloc[:i+1]`, then simulates fills forward. Legacy full-exit simulator
(no scale-out).

**Swing / high_beta sim** (`pipeline_backtest.py`) — same causal detect +
`simulate_scaled_trade` (gap/slippage, 50% at 1R, runner to 2R). Optional
macro gate via `analysis_engine.score_last_row`.

Neither backtest is inside `run_vibe.py`.

### 4.3 Purged walk-forward (specified, not shipped)

A production-grade purge would:

1. Sort unique event dates.
2. For each fold, train on $[t_0, t_{\mathrm{cut}}]$, validate on
   $[t_{\mathrm{cut}} + h + \epsilon, t_{\mathrm{end}}]$ where $h$ is the
   target horizon in bars and $\epsilon$ is an extra embargo.
3. Drop any train row whose $Y$ window overlaps the val feature window.

That logic is **not** in `coiled_cobra_ml_training.py`. Lab 04 implements a
minimal purge next to a leaking `KFold` so the metric gap is visible.

### 4.4 Lab hook

[Lab 04 — Leakage & Validation](../labs/04_validation_integrity.md): random
$K$-fold vs the current temporal split vs a purged embargo.

---

## 5. Model Interpretability & Optimization

### 5.1 What ships today

After fit, the trainer prints ASCII gain/split bars and writes
`coiled_cobra_ml_feature_importance.png` (XGB vs LGB side by side). Typical
pattern (not a frozen ranking):

- LightGBM often leads with `ATR_Pct` (volatility scale + sample weight)
- XGBoost (MAE) spreads credit across Fib / EMA distances
- `Score` remains contributory but is not automatically #1

Gain importance is **not** causal. Correlated EMA/Fib distances steal splits
from each other. Use it to sanity-check "did the model ignore geometry?" not
to rewrite the 100-pt rubric.

### 5.2 SHAP (lab-only)

SHAP values decompose a prediction into per-feature contributions that sum
to $\hat{y} - \mathbb{E}[\hat{y}]$. For trees, `shap.TreeExplainer` is the
right tool. Summary plots show global attribution; dependence plots show
interactions (e.g. `Pct_From_Fib786` vs `ATR_Pct`).

Not installed by default. See [Lab 06](../labs/06_shap_explainability.md).

### 5.3 Optuna (lab-only)

`MODEL_PARAMS` is a fixed regularized default. A time-series-safe study
would optimize `max_depth`, `learning_rate`, `n_estimators`, `subsample`,
`colsample_bytree`, and optionally `reg_alpha` / `reg_lambda`, scoring each
trial on the **val** partition from `_temporal_split()` (or a purged CV),
never on shuffled K-fold.

Not installed by default. See [Lab 05](../labs/05_hyperparameter_optuna.md).

### 5.4 Inference discipline

`ml_ranker` is a **tie-breaker**:

1. Rubric / structure / RS / risk must already pass.
2. Attach `ML_Pred_Return` / `ML_Rank`.
3. Prefer names that are strong on both.
4. Never size or gate solely on the booster.

---

## 6. Market regimes (preview of Lab 07)

High-beta swing logic already gates on QQQ regime and 63-day relative
strength (`analysis_engine.market_regime_ok` / `relative_strength`). The ML
baseline does **not** interact those flags. `ATR_Pct` is the only explicit
volatility channel in $X$ and in `sample_weight`.

A regime-robust workflow:

- slice OOS MAE by `ATR_Pct` terciles,
- compare a single frozen model vs periodic retrain (the current 26w test
  window is already a crude rolling holdout),
- do not retune on the same names used as the high_beta promotion basket.

[Lab 07 — Regime Shifts](../labs/07_regime_shifts.md).

---

## 7. End-to-end research loop

```
data_ingestor.py weekly
        │
        ▼
coiled_cobra_backtest.py weekly --backtest
        │  coiled_cobra_backtest_trades_*.csv
        ▼
coiled_cobra_ml_training.py --csv … --artifacts-dir …
        │  xgb/lgb artifacts + importance PNG + metadata JSON
        ▼
ml_ranker.attach_ml_ranks(setups_df)   # soft rank on new scans
```

```bash
export PYTHONPATH=src
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest --tickers SPY,QQQ
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_YYYY-MM-DD.csv
```

---

## 8. Lab index

| Lab | Question this manual raises |
| --- | --------------------------- |
| [01](../labs/01_indicator_sensitivity.md) | Which geometry columns actually move gain? |
| [02](../labs/02_target_horizon_shift.md) | Does 13w $Y$ change SNR and sample size? |
| [03](../labs/03_gbdt_showdown.md) | Is the ranking library-specific? |
| [04](../labs/04_validation_integrity.md) | How much does leaking K-fold flatter you? |
| [05](../labs/05_hyperparameter_optuna.md) | Can val MAE improve without breaking the time split? |
| [06](../labs/06_shap_explainability.md) | Where do individual predictions come from? |
| [07](../labs/07_regime_shifts.md) | Does the model survive vol-regime changes? |
