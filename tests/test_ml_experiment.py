"""Tests for the pre-registered ML experiment harness (coiled_cobra_ml_experiment).

Synthetic data with known ground truth: a planted edge in an *extended* feature
(``Part_structure``) must be found by the extended variants but not by the
base-feature model, the shuffled-label control and the null dataset must not
qualify, and the lockbox must never leak into development.
"""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")
pytest.importorskip("matplotlib")

from finance_vibe import config
from finance_vibe import coiled_cobra_ml_experiment as ex
from finance_vibe import coiled_cobra_ml_training as trn
from finance_vibe import coiled_cobra_ml_walkforward as wf

DATE = trn.DATE_COL


def _synthetic(edge: float, n_weeks=200, per_week=30, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_weeks, freq="7D")
    df = pd.DataFrame({DATE: np.repeat(dates, per_week)})
    n = len(df)
    df["Symbol"] = [f"T{i % per_week}" for i in range(n)]
    df["Score"] = rng.uniform(70, 95, n).round(2)             # uninformative by construction
    for c in ("Pct_From_EMA20", "Pct_From_EMA50", "Pct_From_Fib618", "Pct_From_Fib786"):
        df[c] = rng.normal(0.02, 0.05, n)
    df["ATR_Pct"] = rng.uniform(0.02, 0.08, n)
    for c in ("RVOL", "BBWidth_Pctile", "RS_63d", "Checks_N", "Part_vol_contraction",
              "Part_relative_strength", "Part_volume_shelf", "Part_overhead_clearance",
              "Part_rvol_trigger", "QQQ_Pct_From_EMA50", "QQQ_Ret_13w"):
        df[c] = rng.normal(0, 1, n)
    df["Part_structure"] = rng.normal(0, 1, n)                 # the planted driver
    market = np.repeat(rng.normal(0, 0.02, n_weeks), per_week)
    noise = rng.normal(0, 0.05, n)
    df["Excess_Return_2w"] = edge * 0.02 * df["Part_structure"] + noise
    df["Forward_Return_2w"] = df["Excess_Return_2w"] + market
    filled = rng.uniform(size=n) > 0.15
    df["Outcome"] = np.where(filled, "stopped", "no_fill")
    df["R Multiple"] = np.where(filled, edge * 0.5 * df["Part_structure"] + rng.normal(0, 1.2, n), np.nan)
    return df


def _run(df, **kw):
    kw.setdefault("max_folds", 3)
    kw.setdefault("n_boot", 300)
    kw.setdefault("lockbox_weeks", 40)
    return ex.run_experiment(df, **kw)


# ---------------------------------------------------------------------------
# Training-target preparation
# ---------------------------------------------------------------------------

def test_prepare_target_demeans_within_date_and_filters_filled():
    df = _synthetic(edge=1.0, n_weeks=20)
    spec = ex.VARIANTS["V4_xgb_ext_rmult"]
    t = ex._prepare_target(df, spec, seed=0)
    assert (t["Outcome"] != "no_fill").all() and t["R Multiple"].notna().all()
    raw = t["R Multiple"] - t.groupby(DATE)["R Multiple"].transform("mean")
    lo, hi = raw.quantile([0.01, 0.99])
    assert t["y"].between(lo - 1e-9, hi + 1e-9).all()                       # winsorised
    assert t.groupby(DATE)["y"].mean().abs().max() < 0.5                    # ~demeaned (post-clip)


def test_deployed_variant_target_is_not_winsorised_or_demeaned():
    df = _synthetic(edge=1.0, n_weeks=20)
    df.loc[0, "Forward_Return_2w"] = 5.0                                    # outlier must survive
    t = ex._prepare_target(df, ex.VARIANTS["V1_deployed"], seed=0)
    assert t["y"].max() == pytest.approx(5.0)
    assert t["y"].equals(t["Forward_Return_2w"].astype(float))


def test_shuffle_control_keeps_per_week_labels_but_breaks_association():
    df = _synthetic(edge=1.0, n_weeks=30)
    spec = ex.VARIANTS["C0_shuffled_control"]
    real = ex._prepare_target(df, {**spec, "shuffle": False}, seed=1)
    shuf = ex._prepare_target(df, spec, seed=1)
    for d, g in real.groupby(DATE):                                          # same multiset per week
        assert np.allclose(np.sort(g["y"]), np.sort(shuf.loc[g.index, "y"]))
    assert abs(np.corrcoef(shuf["y"], shuf["Part_structure"])[0, 1]) < 0.08
    assert np.corrcoef(real["y"], real["Part_structure"])[0, 1] > 0.3


def test_primary_outcome_only_scores_filled_trades():
    df = _synthetic(edge=1.0, n_weeks=10)
    ev = ex._eval_frame(df, ex.PRIMARY)
    assert (ev["Outcome"] != "no_fill").all() and ev[ex.PRIMARY].notna().all()
    assert len(ev) < len(df)
    assert len(ex._eval_frame(df, "Excess_Return_2w")) == len(df)          # secondary uses all rows


def test_tiny_training_set_returns_nan_predictions():
    df = _synthetic(edge=1.0, n_weeks=2, per_week=5)
    out = ex.fit_predict_variant(df, df, ex.VARIANTS["V2_xgb_ext_excess"])
    assert np.isnan(out).all() and len(out) == len(df)


def test_weekly_metrics_accepts_alternative_outcome_column():
    df = pd.DataFrame({DATE: [pd.Timestamp("2024-01-01")] * 8, "r": range(8),
                       "Forward_Return_2w": [0.0] * 8, "R Multiple": range(8)})
    assert wf.weekly_metrics(df, "r", 6, ret_col="R Multiple").iloc[0]["ic"] == pytest.approx(1.0)
    assert np.isnan(wf.weekly_metrics(df, "r", 6).iloc[0]["ic"])            # default outcome is constant


# ---------------------------------------------------------------------------
# Lockbox discipline
# ---------------------------------------------------------------------------

def test_split_dev_lockbox_leaves_an_embargo_gap():
    df = _synthetic(edge=0.0, n_weeks=120)
    dev, lock, lock_start = ex.split_dev_lockbox(df, lockbox_weeks=26, embargo_weeks=2)
    assert lock[DATE].min() >= lock_start and lock[DATE].max() == df[DATE].max()
    assert dev[DATE].max() + pd.Timedelta(weeks=2) < lock[DATE].min()      # no dev label reaches the lockbox
    assert len(dev) + len(lock) < len(df)                                   # embargo rows dropped


def test_development_never_touches_the_lockbox_and_null_leaves_it_unspent(monkeypatch):
    df = _synthetic(edge=0.0)
    calls = []
    real = ex.fit_predict_variant

    def spy(train, test, spec, seed=0):
        calls.append((train[DATE].max(), test[DATE].min(), test[DATE].max()))
        return real(train, test, spec, seed)

    monkeypatch.setattr(ex, "fit_predict_variant", spy)
    res = _run(df)
    lock_start = pd.Timestamp(res["lockbox_start"])

    assert calls, "development folds should have trained models"
    assert all(test_max < lock_start for _, _, test_max in calls)           # nothing scored in lockbox
    assert all(train_max < lock_start - pd.Timedelta(weeks=trn.EMBARGO_WEEKS) for train_max, _, _ in calls)
    assert res["qualifiers"] == [] and res["lockbox"] is None              # unspent


# ---------------------------------------------------------------------------
# Detection on known ground truth
# ---------------------------------------------------------------------------

def test_planted_extended_feature_edge_is_found_and_confirmed_on_lockbox():
    res = _run(_synthetic(edge=1.0))
    prim = res["dev"][ex.PRIMARY]

    for v in ("V2_xgb_ext_excess", "V3_ridge_ext_excess", "V4_xgb_ext_rmult"):
        assert prim[v]["mean_diff"] > 0.1 and prim[v]["ci95"][0] > 0, v
    assert "V1_deployed" not in res["qualifiers"]                           # base features can't see it
    assert abs(prim["Score"]["mean_ic"]) < 0.06                             # baseline uninformative
    assert not res["control_qualifies"]
    assert abs(prim["C0_shuffled_control"]["mean_ic"]) < 0.06               # control ~ 0

    lb = res["lockbox"]
    assert lb["variant"] in res["qualifiers"] and lb["passed"] and lb["mean_diff"] > 0.1


def test_null_data_produces_no_qualifier_and_control_stays_quiet():
    res = _run(_synthetic(edge=0.0, seed=3))
    assert res["qualifiers"] == [] and res["lockbox"] is None
    assert not res["control_qualifies"]
    for v in ex.VARIANTS:
        assert res["dev"][ex.PRIMARY][v]["ci95"][0] <= 0 or not np.isfinite(res["dev"][ex.PRIMARY][v]["ci95"][0])


def test_report_and_json_render_for_both_outcomes():
    res = _run(_synthetic(edge=1.0, n_weeks=190), max_folds=2)
    text = ex.format_report(res, {"source_csv": "x.csv"})
    assert "PRIMARY (selection)" in text and "secondary" in text and "LOCKBOX" in text
    json.dumps(ex._json(res))
    null_text = ex.format_report(_run(_synthetic(edge=0.0, n_weeks=190), max_folds=2), {})
    assert "lockbox left unspent" in null_text


def test_main_requires_the_new_backtest_columns(tmp_path):
    df = _synthetic(edge=0.0, n_weeks=60).drop(columns=["Part_structure", "QQQ_Ret_13w"])
    df[config.RUBRIC_VERSION_COL] = config.RUBRIC_VERSION
    csv = tmp_path / "coiled_cobra_backtest_trades_2026-09-19.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Part_structure"):
        ex.main(["--csv", str(csv)])
