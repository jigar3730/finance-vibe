"""Expanding-window walk-forward evaluation of the Coiled Cobra ML ranker.

Standalone, read-only research harness: it never writes the served model
artifacts. For each fold it trains the *deployed* XGB + LGB configuration
(``make_xgb`` / ``make_lgb``, ATR_Pct weights, mean ensemble -- exactly what
``ml_ranker`` serves) on every signal dated before the fold's test window minus
an embargo equal to the forward-return horizon, then scores the out-of-sample
window and compares the ML ranking with the raw rubric ``Score`` ranking.

Metrics are computed cross-sectionally (per ``Signal Date``), because the live
use is ranking the setups of one scan against each other:

* Spearman rank IC between the ranking variable and ``Forward_Return_2w``
* top-tercile minus bottom-tercile mean forward return

plus pooled (whole-fold) versions and paired ML-minus-Score comparisons with
block-bootstrap confidence intervals. Overlapping 2-bar labels make adjacent
weeks dependent, so t-stats use an effective sample size of ``n_weeks / horizon``
and the bootstrap resamples blocks of ``horizon`` consecutive weeks.

    python -m finance_vibe.coiled_cobra_ml_walkforward [--csv PATH] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from finance_vibe import config
    from finance_vibe import coiled_cobra_ml_training as trn
except ImportError:  # pragma: no cover - local direct execution
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe import coiled_cobra_ml_training as trn

DATE_COL = trn.DATE_COL
TARGET_COL = trn.TARGET_COL
HORIZON = trn.TARGET_HORIZON_WEEKS
RANKERS = ("ML", "XGB", "LGB", "Score")


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def make_folds(
    df: pd.DataFrame,
    *,
    test_weeks: int = 26,
    embargo_weeks: int = trn.EMBARGO_WEEKS,
    min_train_weeks: int = 104,
    min_train_rows: int = 300,
    min_test_rows: int = 30,
    max_folds: int = 8,
) -> list[dict]:
    """Contiguous, non-overlapping test windows walking back from the last date.

    Fold ``k`` tests ``[max_date + 1d - (k+1)*W, max_date + 1d - k*W)`` and trains
    on all rows dated before ``test_start - embargo`` (expanding window). Rows
    inside the embargo are dropped from training. Returned oldest fold first.
    """
    if test_weeks <= 0 or embargo_weeks < 0:
        raise ValueError("test_weeks must be > 0 and embargo_weeks >= 0")
    first_date, max_date = df[DATE_COL].min(), df[DATE_COL].max()
    window = pd.Timedelta(weeks=test_weeks)
    embargo = pd.Timedelta(weeks=embargo_weeks)
    end_all = max_date + pd.Timedelta(days=1)

    folds: list[dict] = []
    for k in range(max_folds):
        test_end = end_all - k * window
        test_start = test_end - window
        train_end = test_start - embargo
        if (train_end - first_date) < pd.Timedelta(weeks=min_train_weeks):
            break
        train_idx = df.index[df[DATE_COL] < train_end]
        test_idx = df.index[(df[DATE_COL] >= test_start) & (df[DATE_COL] < test_end)]
        if len(train_idx) < min_train_rows or len(test_idx) < min_test_rows:
            break
        folds.append({
            "fold": 0,  # renumbered below (oldest = 1)
            "train_end": train_end, "test_start": test_start, "test_end": test_end,
            "train_idx": train_idx, "test_idx": test_idx,
        })
    folds.reverse()
    for i, f in enumerate(folds, start=1):
        f["fold"] = i
    return folds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation (average ranks for ties); NaN if undefined."""
    if len(a) < 3:
        return float("nan")
    ra, rb = a.rank(), b.rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return float("nan")
    return float(ra.corr(rb))


def _tercile_spread(pred: pd.Series, ret: pd.Series, seed: int) -> float:
    """Mean return of the top third minus the bottom third by ``pred``.

    Ties are broken by a seeded shuffle so a heavily tied ranking (the rounded
    rubric Score) is not favoured or penalised by row order.
    """
    n = len(pred)
    k = n // 3
    if k < 1:
        return float("nan")
    perm = np.random.default_rng(seed).permutation(n)
    p = pred.to_numpy()[perm]
    r = ret.to_numpy()[perm]
    order = pd.Series(p).rank(method="first", ascending=False).to_numpy()
    return float(r[order <= k].mean() - r[order > n - k].mean())


def weekly_metrics(
    frame: pd.DataFrame, rank_col: str, min_names: int, ret_col: str = TARGET_COL
) -> pd.DataFrame:
    """Per-``Signal Date`` rank IC and tercile spread (weeks with >= min_names)."""
    rows = []
    for date, g in frame.groupby(DATE_COL):
        if len(g) < min_names:
            continue
        rows.append({
            DATE_COL: date,
            "n": len(g),
            "ic": spearman(g[rank_col], g[ret_col]),
            "spread": _tercile_spread(g[rank_col], g[ret_col], int(date.value % (2**31))),
        })
    return pd.DataFrame(rows, columns=[DATE_COL, "n", "ic", "spread"])


def _mean_t(x: pd.Series, horizon: int = HORIZON) -> tuple[float, float, int]:
    """Mean, t-stat with n_eff = n / horizon (overlapping labels), n."""
    x = x.dropna()
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    sd = float(x.std(ddof=1))
    n_eff = max(n / horizon, 1.0)
    t = float(x.mean() / (sd / math.sqrt(n_eff))) if sd > 0 else float("nan")
    return float(x.mean()), t, n


def block_bootstrap_ci(
    x: np.ndarray, block: int = HORIZON, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of the mean via a circular block bootstrap (time-ordered input)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        means[i] = x[idx.ravel()[:n]].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------

def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Train the deployed XGB + LGB on ``train``; return per-model + ML preds."""
    # _build_matrices applies the same X / y / ATR-weight preprocessing as the
    # deployed training run (train-median weight fill); the 'val' slot is used
    # here as the scoring set.
    parts = trn._build_matrices(train, test, test)
    tr, te = parts["train"], parts["val"]

    xgb = trn.make_xgb().fit(tr["X"], tr["y"], sample_weight=tr["w"])
    lgb = trn.make_lgb().fit(tr["X"], tr["y"], sample_weight=tr["w"])
    p_xgb = np.asarray(xgb.predict(te["X"]), dtype=float)
    p_lgb = np.asarray(lgb.predict(te["X"]), dtype=float)

    out = test[[DATE_COL, TARGET_COL, "Score"]].copy()
    out["XGB"], out["LGB"] = p_xgb, p_lgb
    out["ML"] = (p_xgb + p_lgb) / 2.0          # same mean ensemble as ml_ranker
    out["train_median_y"] = float(np.median(tr["y"]))
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _safe_mean(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else float("nan")


def evaluate(
    df: pd.DataFrame,
    folds: list[dict],
    *,
    min_names: int = 6,
    n_boot: int = 2000,
) -> dict:
    """Run every fold; return per-fold rows, weekly series, and the summary."""
    fold_rows: list[dict] = []
    weekly: dict[str, list[pd.DataFrame]] = {r: [] for r in RANKERS}

    for f in folds:
        train, test = df.loc[f["train_idx"]], df.loc[f["test_idx"]]
        pred = fit_predict(train, test)
        # Scoring column must be numeric for ranking.
        pred["Score"] = pd.to_numeric(pred["Score"], errors="coerce")

        row = {
            "fold": f["fold"],
            "train_end": f["train_end"].strftime("%Y-%m-%d"),
            "test_start": f["test_start"].strftime("%Y-%m-%d"),
            "test_end": (f["test_end"] - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "n_train": len(train), "n_test": len(test),
            "mae_ml": float(np.mean(np.abs(pred["ML"] - pred[TARGET_COL]))),
            "mae_median_baseline": float(np.mean(np.abs(pred["train_median_y"] - pred[TARGET_COL]))),
        }
        for r in RANKERS:
            w = weekly_metrics(pred, r, min_names)
            w["fold"] = f["fold"]
            weekly[r].append(w)
            row[f"ic_{r}"] = _safe_mean(w["ic"])
            row[f"spread_{r}"] = _safe_mean(w["spread"])
            row[f"pooled_ic_{r}"] = spearman(pred[r], pred[TARGET_COL])
            row[f"pooled_spread_{r}"] = _tercile_spread(
                pred[r], pred[TARGET_COL], seed=f["fold"]
            )
        row["weeks_used"] = int(len(weekly["ML"][-1]))
        row["weeks_total"] = int(pred[DATE_COL].nunique())
        fold_rows.append(row)

    series = {r: pd.concat(weekly[r], ignore_index=True).sort_values(DATE_COL) for r in RANKERS}
    return {"folds": fold_rows, "weekly": series,
            "summary": summarize(fold_rows, series, n_boot=n_boot)}


def summarize(fold_rows: list[dict], series: dict[str, pd.DataFrame], n_boot: int = 2000) -> dict:
    summary: dict = {"n_folds": len(fold_rows), "rankers": {}, "ml_vs_score": {}}
    for r in RANKERS:
        ic_m, ic_t, n = _mean_t(series[r]["ic"])
        sp_m, sp_t, _ = _mean_t(series[r]["spread"])
        summary["rankers"][r] = {
            "weeks": n, "mean_ic": ic_m, "ic_t_neff": ic_t,
            "mean_spread": sp_m, "spread_t_neff": sp_t,
            "folds_ic_positive": int(sum(1 for f in fold_rows if f[f"ic_{r}"] > 0)),
        }

    # Paired ML - Score on the weeks both have defined (same weeks by construction).
    ml, sc = series["ML"].set_index(DATE_COL), series["Score"].set_index(DATE_COL)
    joined = ml[["ic", "spread"]].join(sc[["ic", "spread"]], lsuffix="_ml", rsuffix="_sc").dropna()
    for metric in ("ic", "spread"):
        d = (joined[f"{metric}_ml"] - joined[f"{metric}_sc"]).sort_index()
        mean, t, n = _mean_t(d)
        lo, hi = block_bootstrap_ci(d.to_numpy(), n_boot=n_boot)
        summary["ml_vs_score"][metric] = {
            "weeks": n, "mean_diff": mean, "t_neff": t, "ci95": [lo, hi],
            "folds_ml_better": int(sum(
                1 for f in fold_rows
                if np.isfinite(f[f"{metric}_ML"]) and np.isfinite(f[f"{metric}_Score"])
                and f[f"{metric}_ML"] > f[f"{metric}_Score"]
            )),
        }
    summary["mae"] = {
        "ml": float(np.mean([f["mae_ml"] for f in fold_rows])),
        "train_median_baseline": float(np.mean([f["mae_median_baseline"] for f in fold_rows])),
    }
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _f(x, nd=3, pct=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  n/a"
    return f"{x * 100:+.2f}%" if pct else f"{x:+.{nd}f}"


def format_report(result: dict, meta: dict) -> str:
    L: list[str] = []
    L.append("=" * 100)
    L.append("Coiled Cobra ML - expanding-window walk-forward (out-of-sample)")
    L.append("=" * 100)
    for k, v in meta.items():
        L.append(f"  {k}: {v}")
    L.append("")
    L.append("Per fold (per-week cross-sectional means; ML = mean of XGB+LGB)")
    L.append(f"{'fold':>4} {'test window':<23} {'ntrain':>6} {'ntest':>5} {'wks':>7} "
             f"{'IC ML':>7} {'IC Scr':>7} {'Sprd ML':>8} {'Sprd Scr':>8} "
             f"{'pIC ML':>7} {'pIC Scr':>7} {'MAE ML':>7} {'MAE med':>7}")
    for f in result["folds"]:
        L.append(
            f"{f['fold']:>4} {f['test_start']}..{f['test_end'][5:]:<8} {f['n_train']:>6} {f['n_test']:>5} "
            f"{f['weeks_used']:>3}/{f['weeks_total']:<3} "
            f"{_f(f['ic_ML']):>7} {_f(f['ic_Score']):>7} "
            f"{_f(f['spread_ML'], pct=True):>8} {_f(f['spread_Score'], pct=True):>8} "
            f"{_f(f['pooled_ic_ML']):>7} {_f(f['pooled_ic_Score']):>7} "
            f"{f['mae_ml']:.4f} {f['mae_median_baseline']:.4f}"
        )
    s = result["summary"]
    L.append("")
    L.append("All OOS weeks pooled across folds (t-stats use n_eff = weeks / horizon)")
    L.append(f"{'ranker':<7} {'weeks':>5} {'mean IC':>8} {'t':>6} {'folds IC>0':>10} "
             f"{'mean spread':>12} {'t':>6}")
    for r in RANKERS:
        x = s["rankers"][r]
        L.append(
            f"{r:<7} {x['weeks']:>5} {_f(x['mean_ic']):>8} {_f(x['ic_t_neff'], 2):>6} "
            f"{x['folds_ic_positive']:>4}/{s['n_folds']:<5} "
            f"{_f(x['mean_spread'], pct=True):>12} {_f(x['spread_t_neff'], 2):>6}"
        )
    L.append("")
    L.append("Paired ML minus Score (same weeks)")
    for metric, label, pct in (("ic", "rank IC", False), ("spread", "tercile spread", True)):
        x = s["ml_vs_score"][metric]
        lo, hi = x["ci95"]
        L.append(
            f"  {label:<15} mean diff {_f(x['mean_diff'], pct=pct)}  t={_f(x['t_neff'], 2)}  "
            f"95% block-bootstrap CI [{_f(lo, pct=pct)}, {_f(hi, pct=pct)}]  "
            f"ML better in {x['folds_ml_better']}/{s['n_folds']} folds  (n={x['weeks']} wks)"
        )
    L.append(f"  MAE: ML {s['mae']['ml']:.4f} vs train-median baseline "
             f"{s['mae']['train_median_baseline']:.4f}")
    L.append("")
    L.append(verdict(s))
    return "\n".join(L)


def verdict(summary: dict) -> str:
    """Plain-language read-out; deliberately conservative (CI must exclude 0)."""
    ic = summary["ml_vs_score"]["ic"]
    ml = summary["rankers"]["ML"]
    lo, hi = ic["ci95"]
    if not np.isfinite(lo):
        return "VERDICT: insufficient weeks for a confidence interval."
    if lo > 0:
        return (f"VERDICT: ML rank IC beats Score with the 95% CI excluding 0 "
                f"(mean IC {_f(ml['mean_ic'])}).")
    if hi < 0:
        return "VERDICT: ML rank IC is significantly WORSE than raw Score - do not use it."
    return ("VERDICT: ML is not distinguishable from raw Score out-of-sample "
            "(paired IC-difference CI includes 0) - no evidence it should drive Priority.")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None, help="Trades CSV (default: newest in the weekly log dir)")
    ap.add_argument("--test-weeks", type=int, default=26)
    ap.add_argument("--embargo-weeks", type=int, default=trn.EMBARGO_WEEKS)
    ap.add_argument("--min-train-weeks", type=int, default=104)
    ap.add_argument("--min-train-rows", type=int, default=300)
    ap.add_argument("--max-folds", type=int, default=8)
    ap.add_argument("--min-names", type=int, default=6,
                    help="Min setups on a Signal Date for it to count in per-week metrics")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="Write the full result (folds, summary) as JSON")
    ap.add_argument("--allow-rubric-mismatch", action="store_true")
    args = ap.parse_args(argv)

    csv_path = trn._resolve_source_csv(args.csv, trn.TRAIN_MODE)
    df = trn._load_and_prepare(csv_path)
    rubric = trn._validate_rubric_version(df, csv_path, args.allow_rubric_mismatch)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

    folds = make_folds(
        df, test_weeks=args.test_weeks, embargo_weeks=args.embargo_weeks,
        min_train_weeks=args.min_train_weeks, min_train_rows=args.min_train_rows,
        max_folds=args.max_folds,
    )
    if not folds:
        raise RuntimeError("Not enough history for even one walk-forward fold.")

    result = evaluate(df, folds, min_names=args.min_names, n_boot=args.n_boot)
    meta = {
        "source_csv": csv_path.name, "rubric_version": rubric, "rows": len(df),
        "signal_dates": f"{df[DATE_COL].min():%Y-%m-%d} .. {df[DATE_COL].max():%Y-%m-%d}",
        "folds": f"{len(folds)} x {args.test_weeks}w test, expanding train, "
                 f"{args.embargo_weeks}w embargo",
        "min_names_per_week": args.min_names,
    }
    print(format_report(result, meta))

    if args.out:
        payload = {"meta": meta, "folds": result["folds"], "summary": result["summary"]}
        Path(args.out).write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
