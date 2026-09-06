# Lab 02 — Target Horizon Shift

**Objective.** Move the training target $Y$ from the 2-week forward return
to the 13-week forward return and measure how the label distribution and
signal-to-noise change.

Code of record today: `TARGET_COL = "Forward_Return_2w"` and
`TARGET_HORIZON_WEEKS = 2` in `src/finance_vibe/coiled_cobra_ml_training.py`.
Older docs sometimes said 13w; the trainer does **not**.

## Why this experiment

$$
Y_h = \frac{\mathrm{Close}[t+h]-\mathrm{Close}[t]}{\mathrm{Close}[t]}
$$

is computed in `coiled_cobra_backtest.py` for $h \in \{2,5,13,26\}$. A 13-week
$Y$ matches the coil → expansion story (multi-month leader runs) but:

- more rows lose $Y$ at the right edge (`None` when `idx + h` is past the file),
- $|Y|$ and tail mass grow, so raw MAE is not comparable across horizons,
- a 6-feature *snapshot* at $t$ has more time to be swamped by regime drift.

## Starter pointers

| What | Where |
| ---- | ----- |
| Active target | `TARGET_COL`, `TARGET_HORIZON_WEEKS` in `coiled_cobra_ml_training.py` |
| Row drop on NaN $Y$ | `_load_and_prepare()` |
| Horizon construction | `_forward_return(horizon)` in `coiled_cobra_backtest.py` (~lines 212–261) |
| Leakage columns | `LEAKAGE_COLS` — do not switch $Y$ to `R Multiple` or `Target_Label` |
| Metadata written at train time | `coiled_cobra_ml_model_metadata.json` → `target_column` |

## Prerequisites

A trades CSV that already contains `Forward_Return_2w` **and**
`Forward_Return_13w` (any recent `--backtest` export).

## Step 1 — inspect both labels (no training yet)

```bash
export PYTHONPATH=src
python - <<'PY'
import pandas as pd
from pathlib import Path

p = sorted(Path("data/logs/weekly").glob("coiled_cobra_backtest_trades_*.csv"))[-1]
df = pd.read_csv(p)
print("source:", p)
for col in ["Forward_Return_2w", "Forward_Return_5w", "Forward_Return_13w", "Forward_Return_26w"]:
    s = pd.to_numeric(df[col], errors="coerce")
    print(f"{col:22} n={s.notna().sum():5}  nan={s.isna().sum():5}  "
          f"mean={s.mean():7.4f}  std={s.std():7.4f}  "
          f"p_pos={(s.dropna()>0).mean():5.3f}  p99={s.quantile(0.99):7.4f}")
PY
```

Record: sample attrition as $h$ grows, mean/std, fraction of positive returns,
and the 99th percentile (tail).

## Step 2 — baseline train on 2w

```bash
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv \
  --artifacts-dir data/logs/weekly/lab02_2w
```

Note train/val/test **row counts** (printed as temporal-split bounds) and
Val/Test MAE, RMSE.

## Step 3 — switch $Y$ to 13w

In `coiled_cobra_ml_training.py` change only:

```python
TARGET_COL = "Forward_Return_13w"
TARGET_HORIZON_WEEKS = 13
```

Retrain:

```bash
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_<date>.csv \
  --artifacts-dir data/logs/weekly/lab02_13w
```

`_load_and_prepare()` will now drop rows missing `Forward_Return_13w`, so
the printed "Dropped … NaN" count and the split sizes **must** change.

## Step 4 — compare apples to apples

Raw MAE($Y_{13}$) > MAE($Y_2$) is expected. Normalize:

$$
\widetilde{\mathrm{MAE}} = \frac{\mathrm{MAE}}{\mathrm{std}(Y_{\mathrm{val}})}
$$

and, if you want a rank metric, Spearman correlation of predictions vs $Y$
on the test partition (optional `scipy.stats.spearmanr`).

Also compare:

- XGB vs LGB feature-importance **order** (does `ATR_Pct` rise at 13w?).
- Whether Test RMSE / MAE blows up (tail events over a quarter).

## What to record

| Horizon | Rows after NaN drop | Train / Val / Test | Val MAE | Val MAE / std($Y$) | Test MAE | Top feature (XGB) |
| ------- | ------------------- | ------------------ | ------- | ------------------ | -------- | ----------------- |
| 2w | | | | | | |
| 13w | | | | | | |

## Expected failure modes

- Leaving `TARGET_HORIZON_WEEKS = 2` while changing `TARGET_COL` → metadata
  JSON lies; `ml_ranker` does not read the horizon, but your notes will.
- Interpreting a higher 13w hit rate ($P(Y>0)$) as a better *model*. The
  scanner is long-biased; drift alone lifts the positive rate.
- Training on `Target_Label` (filled-trade binary) — that is post-outcome
  leakage, not a horizon shift.

## Restore

Set `TARGET_COL` and `TARGET_HORIZON_WEEKS` back to `Forward_Return_2w` / `2`
unless you are deliberately changing the production contract. Update
[`../architecture/coiled_cobra_ml.md`](../architecture/coiled_cobra_ml.md) if
you promote 13w.
