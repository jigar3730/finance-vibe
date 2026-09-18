"""Pre-registered ML-vs-Score experiment for the Coiled Cobra ranker (research only).

Question: can any model built from richer features / better targets rank a
week's setups better than the raw rubric ``Score``?  This harness never writes
served artifacts.  The protocol is fixed in code *before* looking at results:

* **Primary outcome** -- realised ``R Multiple`` of *filled* trades, ranked
  within each ``Signal Date`` (weeks with >= ``--min-names`` filled trades).
  Secondary (reported, never used for selection): ``Excess_Return_2w`` (vs QQQ)
  and ``Forward_Return_2w`` over all rows.
* **Rankers** -- the raw Score baseline and five fixed variants (``VARIANTS``),
  including ``C0_shuffled_control`` (labels shuffled within week in training),
  which must show no edge; if it "wins", the harness is broken.
* **Lockbox** -- the last ``--lockbox-weeks`` are never used for development or
  selection.  A variant *qualifies* only if, on the development walk-forward
  folds, its paired (variant - Score) rank-IC difference has a 95% block-bootstrap
  CI above 0.  Only then is the single best qualifier evaluated **once** on the
  lockbox, where it passes if the paired difference is positive with t > 1.645.
  If nothing qualifies the lockbox is left unspent.
* All train/test splits use the same 2-week embargo as the deployed pipeline.

    python -m finance_vibe.coiled_cobra_ml_experiment --csv <trades.csv> [--out r.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

try:
    from finance_vibe import coiled_cobra_ml_training as trn
    from finance_vibe import coiled_cobra_ml_walkforward as wf
except ImportError:  # pragma: no cover - local direct execution
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import coiled_cobra_ml_training as trn
    from finance_vibe import coiled_cobra_ml_walkforward as wf

DATE = trn.DATE_COL
PRIMARY = "R Multiple"
SECONDARY = ("Excess_Return_2w", "Forward_Return_2w")

BASE_FEATURES = list(trn.FEATURE_COLS)
EXT_FEATURES = BASE_FEATURES + [
    "RVOL", "BBWidth_Pctile", "RS_63d", "Checks_N",
    "Part_vol_contraction", "Part_relative_strength", "Part_structure",
    "Part_volume_shelf", "Part_overhead_clearance", "Part_rvol_trigger",
    "QQQ_Pct_From_EMA50", "QQQ_Ret_13w",
]

# name -> spec.  ``deployed`` = the shipped XGB+LGB MAE config with ATR weights.
VARIANTS: dict[str, dict] = {
    "V1_deployed": dict(model="deployed", features=BASE_FEATURES, target="Forward_Return_2w",
                        demean=False, filled_only=False),
    "V2_xgb_ext_excess": dict(model="xgb", features=EXT_FEATURES, target="Excess_Return_2w",
                              demean=True, filled_only=False),
    "V3_ridge_ext_excess": dict(model="ridge", features=EXT_FEATURES, target="Excess_Return_2w",
                                demean=True, filled_only=False),
    "V4_xgb_ext_rmult": dict(model="xgb", features=EXT_FEATURES, target=PRIMARY,
                             demean=True, filled_only=True),
    "C0_shuffled_control": dict(model="xgb", features=EXT_FEATURES, target="Excess_Return_2w",
                                demean=True, filled_only=False, shuffle=True),
}


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------

def _demean_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """Cross-sectional demeaning: label minus that date's mean label."""
    return df[col] - df.groupby(DATE)[col].transform("mean")


def _prepare_target(train: pd.DataFrame, spec: dict, seed: int) -> pd.DataFrame:
    """Rows usable for training + a ``y`` column (demeaned/winsorised/shuffled)."""
    t = train.copy()
    if spec["filled_only"]:
        t = t[t["Outcome"] != "no_fill"]
    t = t[t[spec["target"]].notna()].copy()
    if spec["demean"]:
        # need >= 3 names on a date for a meaningful cross-sectional mean
        t = t[t.groupby(DATE)[spec["target"]].transform("size") >= 3].copy()
        t["y"] = _demean_by_date(t, spec["target"])
    else:
        t["y"] = t[spec["target"]].astype(float)
    if len(t) and spec["model"] != "deployed":       # V1 must stay exactly as shipped
        lo, hi = t["y"].quantile([0.01, 0.99])
        t["y"] = t["y"].clip(lo, hi)
    if spec.get("shuffle"):
        rng = np.random.default_rng(seed)
        t["y"] = t.groupby(DATE)["y"].transform(lambda s: rng.permutation(s.to_numpy()))
    return t


def _xgb_reg() -> XGBRegressor:
    # One fixed, deliberately small/regularised config for the extended variants.
    return XGBRegressor(
        max_depth=3, learning_rate=0.03, n_estimators=200, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
        objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=42,
    )


def fit_predict_variant(train: pd.DataFrame, test: pd.DataFrame, spec: dict, seed: int = 0) -> np.ndarray:
    """Predictions for ``test`` from a model fit on ``train`` only."""
    feats = spec["features"]
    t = _prepare_target(train, spec, seed)
    if len(t) < 100:
        return np.full(len(test), np.nan)

    if spec["model"] == "deployed":
        # Exactly the shipped configuration (XGB+LGB MAE, ATR weights, mean ensemble).
        parts = trn._build_matrices(t.assign(**{trn.TARGET_COL: t["y"]}), test, test)
        tr, te = parts["train"], parts["val"]
        xgb = trn.make_xgb().fit(tr["X"], tr["y"], sample_weight=tr["w"])
        lgb = trn.make_lgb().fit(tr["X"], tr["y"], sample_weight=tr["w"])
        return (np.asarray(xgb.predict(te["X"])) + np.asarray(lgb.predict(te["X"]))) / 2.0

    X_tr = t[feats].apply(pd.to_numeric, errors="coerce")
    X_te = test[feats].apply(pd.to_numeric, errors="coerce")
    if spec["model"] == "xgb":
        return np.asarray(_xgb_reg().fit(X_tr, t["y"].to_numpy()).predict(X_te))

    if spec["model"] == "ridge":
        med = X_tr.median().fillna(0.0)
        mu, sd = X_tr.fillna(med).mean(), X_tr.fillna(med).std().replace(0, 1.0).fillna(1.0)
        z = lambda X: ((X.fillna(med) - mu) / sd).to_numpy()
        return Ridge(alpha=10.0).fit(z(X_tr), t["y"].to_numpy()).predict(z(X_te))

    raise ValueError(spec["model"])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval_frame(test: pd.DataFrame, outcome: str) -> pd.DataFrame:
    if outcome == PRIMARY:
        return test[(test["Outcome"] != "no_fill") & test[PRIMARY].notna()]
    return test[test[outcome].notna()]


def weekly_series(pred_frame: pd.DataFrame, rankers: list[str], outcome: str,
                  min_names: int) -> dict[str, pd.DataFrame]:
    """Per-week IC/spread for each ranker on the outcome-eligible rows."""
    ev = _eval_frame(pred_frame, outcome)
    return {r: wf.weekly_metrics(ev, r, min_names, ret_col=outcome) for r in rankers}


def _ic_stats(w: pd.DataFrame) -> dict:
    mean_ic, ic_t, n = wf._mean_t(w["ic"])
    return {"weeks": n, "mean_ic": mean_ic, "ic_t": ic_t}


def _paired(series: dict[str, pd.DataFrame], ranker: str, n_boot: int, seed: int = 0) -> dict:
    a = series[ranker].set_index(DATE)["ic"]
    b = series["Score"].set_index(DATE)["ic"]
    d = (a - b).dropna().sort_index()
    mean, t, n = wf._mean_t(d)
    lo, hi = wf.block_bootstrap_ci(d.to_numpy(), n_boot=n_boot, seed=seed)
    mean_ic, ic_t, n_ic = wf._mean_t(a)
    return {"weeks": n, "mean_ic": mean_ic, "ic_t": ic_t, "mean_diff": mean, "t": t, "ci95": [lo, hi]}


def run_folds(df: pd.DataFrame, folds: list[dict], variants: dict, min_names: int) -> dict:
    """Train every variant per fold; return weekly series per outcome."""
    rankers = list(variants) + ["Score"]
    acc: dict[str, dict[str, list]] = {o: {r: [] for r in rankers} for o in (PRIMARY, *SECONDARY)}
    fold_log = []
    for f in folds:
        train, test = df.loc[f["train_idx"]], df.loc[f["test_idx"]].copy()
        test["Score"] = pd.to_numeric(test["Score"], errors="coerce")
        for name, spec in variants.items():
            test[name] = fit_predict_variant(train, test, spec, seed=f["fold"])
        for outcome in acc:
            for r, w in weekly_series(test, rankers, outcome, min_names).items():
                w["fold"] = f["fold"]
                acc[outcome][r].append(w)
        fold_log.append({"fold": f["fold"], "n_train": len(train), "n_test": len(test),
                         "n_test_filled": int((test["Outcome"] != "no_fill").sum()),
                         "test_start": f["test_start"].strftime("%Y-%m-%d"),
                         "test_end": (f["test_end"] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")})
    series = {o: {r: pd.concat(v, ignore_index=True).sort_values(DATE) for r, v in d.items()}
              for o, d in acc.items()}
    return {"series": series, "folds": fold_log}


def split_dev_lockbox(df: pd.DataFrame, lockbox_weeks: int, embargo_weeks: int):
    """Dev = dates < lockbox_start - embargo (no label reaches the lockbox)."""
    lock_start = df[DATE].max() - pd.Timedelta(weeks=lockbox_weeks) + pd.Timedelta(days=1)
    dev = df[df[DATE] < lock_start - pd.Timedelta(weeks=embargo_weeks)].reset_index(drop=True)
    lock = df[df[DATE] >= lock_start]
    return dev, lock, lock_start


def qualifies(paired: dict) -> bool:
    lo = paired["ci95"][0]
    return bool(np.isfinite(lo) and lo > 0)


def run_experiment(df: pd.DataFrame, *, lockbox_weeks: int = 52, embargo_weeks: int = trn.EMBARGO_WEEKS,
                   test_weeks: int = 26, min_names: int = 8, max_folds: int = 10,
                   n_boot: int = 2000, variants: dict | None = None) -> dict:
    variants = variants or VARIANTS
    dev, lock, lock_start = split_dev_lockbox(df, lockbox_weeks, embargo_weeks)
    folds = wf.make_folds(dev, test_weeks=test_weeks, embargo_weeks=embargo_weeks,
                          min_train_rows=300, min_test_rows=50, max_folds=max_folds)
    if not folds:
        raise RuntimeError("Not enough development history for a walk-forward fold.")

    dev_res = run_folds(dev, folds, variants, min_names)
    table = {}
    for outcome, series in dev_res["series"].items():
        table[outcome] = {r: _paired(series, r, n_boot) for r in variants}
        table[outcome]["Score"] = _ic_stats(series["Score"])
    qual = [v for v in variants if not v.startswith("C0") and qualifies(table[PRIMARY][v])]
    qual.sort(key=lambda v: table[PRIMARY][v]["mean_diff"], reverse=True)

    result = {"lockbox_start": lock_start.strftime("%Y-%m-%d"), "n_dev_rows": len(dev),
              "n_lockbox_rows": len(lock), "folds": dev_res["folds"], "dev": table,
              "control_qualifies": qualifies(table[PRIMARY].get("C0_shuffled_control", {"ci95": [np.nan, 0]})),
              "qualifiers": qual, "lockbox": None}

    if qual:
        best = qual[0]
        train = dev
        test = lock.copy()
        test["Score"] = pd.to_numeric(test["Score"], errors="coerce")
        test[best] = fit_predict_variant(train, test, variants[best], seed=999)
        ser = weekly_series(test, [best, "Score"], PRIMARY, min_names)
        p = _paired(ser, best, n_boot)
        result["lockbox"] = {"variant": best, **p,
                             "passed": bool(p["mean_diff"] > 0 and np.isfinite(p["t"]) and p["t"] > 1.645)}
    return result


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------

def _f(x, nd=3):
    return "  n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def format_report(res: dict, meta: dict) -> str:
    L = ["=" * 104, "Coiled Cobra ML experiment - pre-registered, lockbox-protected", "=" * 104]
    L += [f"  {k}: {v}" for k, v in meta.items()]
    L.append(f"  lockbox: dates >= {res['lockbox_start']} ({res['n_lockbox_rows']} rows, unspent unless a variant qualifies)")
    L.append(f"  development: {res['n_dev_rows']} rows, {len(res['folds'])} expanding folds")
    for outcome, primary in ((PRIMARY, True), *((o, False) for o in SECONDARY)):
        tag = "PRIMARY (selection)" if primary else "secondary (not used for selection)"
        L += ["", f"Outcome: {outcome}  -- {tag}",
              f"{'ranker':<22}{'weeks':>6}{'mean IC':>9}{'IC t':>7}{'vs Score':>10}{'t':>7}   95% block-bootstrap CI"]
        for r, p in res["dev"][outcome].items():
            if r == "Score":
                L.append(f"{'Score (baseline)':<22}{p['weeks']:>6}{_f(p['mean_ic']):>9}{_f(p['ic_t'], 2):>7}")
                continue
            lo, hi = p["ci95"]
            L.append(f"{r:<22}{p['weeks']:>6}{_f(p['mean_ic']):>9}{_f(p['ic_t'], 2):>7}"
                     f"{_f(p['mean_diff']):>10}{_f(p['t'], 2):>7}   [{_f(lo)}, {_f(hi)}]")
    L.append("")
    if res["control_qualifies"]:
        L.append("!! CONTROL QUALIFIED: the shuffled-label control shows an 'edge' -> harness/data problem, ignore results.")
    if res["qualifiers"]:
        lb = res["lockbox"]
        L.append(f"QUALIFIED on development folds: {res['qualifiers']}")
        L.append(f"LOCKBOX (single evaluation) for {lb['variant']}: mean IC {_f(lb['mean_ic'])}, "
                 f"vs Score {_f(lb['mean_diff'])} (t={_f(lb['t'], 2)}, {lb['weeks']} weeks) -> "
                 + ("PASSED" if lb["passed"] else "FAILED"))
    else:
        L.append("VERDICT: no variant beats raw Score with a 95% CI above 0 on the development folds; "
                 "lockbox left unspent. Do not enable ML ranking.")
    return "\n".join(L)


def _json(o):
    return wf._jsonable(o)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--lockbox-weeks", type=int, default=52)
    ap.add_argument("--test-weeks", type=int, default=26)
    ap.add_argument("--min-names", type=int, default=8)
    ap.add_argument("--max-folds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    csv_path = trn._resolve_source_csv(args.csv, trn.TRAIN_MODE)
    df = pd.read_csv(csv_path, dtype={trn.config.RUBRIC_VERSION_COL: str})
    rubric = trn._validate_rubric_version(df, csv_path)
    df[DATE] = pd.to_datetime(df[DATE])
    df = df.sort_values(DATE).reset_index(drop=True)
    need = set(EXT_FEATURES) | {PRIMARY, "Outcome", "Excess_Return_2w", trn.TARGET_COL}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"CSV lacks experiment columns {missing}; regenerate with the current backtest.")

    res = run_experiment(df, lockbox_weeks=args.lockbox_weeks, test_weeks=args.test_weeks,
                         min_names=args.min_names, max_folds=args.max_folds, n_boot=args.n_boot)
    meta = {"source_csv": csv_path.name, "rubric_version": rubric, "rows": len(df),
            "symbols": int(df["Symbol"].nunique()),
            "signal_dates": f"{df[DATE].min():%Y-%m-%d} .. {df[DATE].max():%Y-%m-%d}",
            "min_names_per_week": args.min_names, "embargo_weeks": trn.EMBARGO_WEEKS}
    print(format_report(res, meta))
    if args.out:
        Path(args.out).write_text(json.dumps(_json({"meta": meta, **res}), indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
