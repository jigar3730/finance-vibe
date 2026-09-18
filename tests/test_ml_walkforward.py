"""Tests for the expanding-window walk-forward harness (coiled_cobra_ml_walkforward).

Synthetic data with a *known* ground truth validates the harness itself:
an ML-learnable edge must be detected, a Score-driven edge must be credited to
Score, and pure noise must not produce an edge.
"""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")
pytest.importorskip("matplotlib")

from finance_vibe import config
from finance_vibe import coiled_cobra_ml_training as trn
from finance_vibe import coiled_cobra_ml_walkforward as wf

DATE, TARGET = trn.DATE_COL, trn.TARGET_COL


def _synthetic(signal: str, n_weeks=300, per_week=8, seed=0) -> pd.DataFrame:
    """Monday-labelled weekly signals; ``signal`` picks what drives the label."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_weeks, freq="7D")
    df = pd.DataFrame({DATE: np.repeat(dates, per_week)})
    n = len(df)
    df["Symbol"] = [f"T{i % per_week}" for i in range(n)]
    df["Score"] = rng.uniform(70, 95, n).round(2)
    df["Pct_From_EMA20"] = rng.normal(0.02, 0.05, n)
    df["Pct_From_EMA50"] = rng.normal(0.04, 0.08, n)
    df["Pct_From_Fib618"] = rng.normal(0.05, 0.08, n)
    df["Pct_From_Fib786"] = rng.normal(0.08, 0.10, n)
    df["ATR_Pct"] = rng.uniform(0.02, 0.08, n)
    noise = rng.normal(0, 0.04, n)
    if signal == "feature":      # learnable from FEATURE_COLS, Score is uninformative
        z = (df["Pct_From_EMA20"] - 0.02) / 0.05
        df[TARGET] = 0.02 * z + noise
    elif signal == "score":      # label driven by Score itself
        z = (df["Score"] - 82.5) / 7.2
        df[TARGET] = 0.02 * z + noise
    elif signal == "none":
        df[TARGET] = noise
    else:
        raise ValueError(signal)
    return df


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def test_folds_are_expanding_disjoint_and_embargoed():
    df = _synthetic("none")
    folds = wf.make_folds(df, test_weeks=26, embargo_weeks=2, max_folds=5)
    assert len(folds) == 5
    horizon = pd.Timedelta(weeks=trn.TARGET_HORIZON_WEEKS)

    assert [f["fold"] for f in folds] == [1, 2, 3, 4, 5]           # oldest first
    for f in folds:
        train_dates = df.loc[f["train_idx"], DATE]
        test_dates = df.loc[f["test_idx"], DATE]
        # no train/test row overlap, and no train label realised inside the test window
        assert set(f["train_idx"]).isdisjoint(f["test_idx"])
        assert train_dates.max() + horizon < test_dates.min()
        assert test_dates.min() >= f["test_start"] and test_dates.max() < f["test_end"]

    for a, b in zip(folds, folds[1:]):
        assert a["test_end"] == b["test_start"]                     # contiguous, no gaps/overlap
        assert len(b["train_idx"]) > len(a["train_idx"])            # expanding window
    assert folds[-1]["test_end"] == df[DATE].max() + pd.Timedelta(days=1)
    assert df.loc[folds[-1]["test_idx"], DATE].max() == df[DATE].max()  # last date is tested


def test_folds_stop_when_training_history_is_too_short():
    df = _synthetic("none", n_weeks=140)
    folds = wf.make_folds(df, test_weeks=26, min_train_weeks=104, max_folds=8)
    assert 0 < len(folds) < 8
    for f in folds:
        assert (f["train_end"] - df[DATE].min()) >= pd.Timedelta(weeks=104)


def test_make_folds_rejects_bad_args():
    with pytest.raises(ValueError):
        wf.make_folds(_synthetic("none"), test_weeks=0)
    with pytest.raises(ValueError):
        wf.make_folds(_synthetic("none"), embargo_weeks=-1)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def test_spearman_matches_known_values():
    a = pd.Series([1, 2, 3, 4, 5.0])
    assert wf.spearman(a, a * 10) == pytest.approx(1.0)
    assert wf.spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(wf.spearman(a, pd.Series([1.0] * 5)))           # constant -> undefined
    assert np.isnan(wf.spearman(a.iloc[:2], a.iloc[:2]))            # too few
    # ties: average ranks (matches scipy)
    scipy_stats = pytest.importorskip("scipy.stats")
    x, y = pd.Series([1, 1, 2, 3, 3, 4.0]), pd.Series([2, 1, 1, 5, 4, 6.0])
    assert wf.spearman(x, y) == pytest.approx(scipy_stats.spearmanr(x, y)[0])


def test_tercile_spread_sign_and_value():
    pred = pd.Series(np.arange(9.0))
    ret = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    assert wf._tercile_spread(pred, ret, seed=0) == pytest.approx(2.0)
    assert wf._tercile_spread(-pred, ret, seed=0) == pytest.approx(-2.0)
    assert np.isnan(wf._tercile_spread(pred.iloc[:2], ret.iloc[:2], seed=0))


def test_tercile_spread_ties_are_not_order_biased():
    # All-tied ranking carries no information: spread must be ~0 on average
    # regardless of how rows happen to be ordered.
    ret = pd.Series(np.arange(30.0))
    tied = pd.Series([1.0] * 30)
    spreads = [wf._tercile_spread(tied, ret, seed=s) for s in range(300)]
    assert abs(np.mean(spreads)) < 1.5      # unbiased; an order-based tie-break gives +-19.5


def test_weekly_metrics_respects_min_names():
    df = pd.DataFrame({
        DATE: [pd.Timestamp("2024-01-01")] * 8 + [pd.Timestamp("2024-01-08")] * 3,
        "r": list(range(8)) + [1, 2, 3],
        TARGET: list(range(8)) + [1, 2, 3],
    })
    w = wf.weekly_metrics(df, "r", min_names=6)
    assert len(w) == 1 and w.iloc[0]["ic"] == pytest.approx(1.0)


def test_mean_t_uses_effective_sample_size():
    x = pd.Series(np.r_[np.ones(50) * 0.1, np.ones(50) * -0.1] + np.tile([0.0, 0.05], 50) + 0.02)
    m, t, n = wf._mean_t(x, horizon=2)
    naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert n == 100 and t == pytest.approx(naive / np.sqrt(2))


def test_block_bootstrap_ci_brackets_mean_and_is_seeded():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 200)
    lo, hi = wf.block_bootstrap_ci(x, n_boot=500, seed=1)
    assert lo < x.mean() < hi and lo > 0
    assert (lo, hi) == wf.block_bootstrap_ci(x, n_boot=500, seed=1)
    assert np.isnan(wf.block_bootstrap_ci(np.ones(3))[0])


# ---------------------------------------------------------------------------
# End-to-end harness behaviour on known ground truth
# ---------------------------------------------------------------------------

def _run(signal, max_folds=3, seed=0):
    df = _synthetic(signal, seed=seed)
    folds = wf.make_folds(df, max_folds=max_folds)
    return df, wf.evaluate(df, folds, n_boot=300)


def test_harness_detects_a_learnable_feature_edge_and_credits_ml_not_score():
    _, res = _run("feature")
    s = res["summary"]
    assert s["rankers"]["ML"]["mean_ic"] > 0.15
    assert abs(s["rankers"]["Score"]["mean_ic"]) < 0.08
    assert s["ml_vs_score"]["ic"]["ci95"][0] > 0                    # CI excludes 0
    assert s["ml_vs_score"]["spread"]["mean_diff"] > 0
    assert s["rankers"]["ML"]["folds_ic_positive"] == s["n_folds"]
    assert "beats Score" in wf.verdict(s)


def test_harness_credits_score_when_label_is_driven_by_score():
    _, res = _run("score")
    s = res["summary"]
    assert s["rankers"]["Score"]["mean_ic"] > 0.15
    assert s["rankers"]["Score"]["folds_ic_positive"] == s["n_folds"]
    # ML sees Score as a feature, so it may match it, but must not be "significantly worse".
    assert "WORSE" not in wf.verdict(s)


def test_harness_finds_no_edge_in_pure_noise():
    _, res = _run("none")
    s = res["summary"]
    assert abs(s["rankers"]["ML"]["mean_ic"]) < 0.08
    assert abs(s["rankers"]["Score"]["mean_ic"]) < 0.08
    assert "beats Score" not in wf.verdict(s)


def test_harness_uses_no_future_information():
    # If the label depends on a feature that only exists AFTER the test window
    # in time order (we permute features across dates), nothing may be learned:
    # shuffling the feature within each week keeps cross-sectional structure, but
    # permuting whole-date blocks must destroy an edge that requires alignment.
    df = _synthetic("feature")
    rng = np.random.default_rng(3)
    dates = df[DATE].unique()
    shuffled = dict(zip(dates, rng.permutation(dates)))
    donor = df.set_index(DATE)["Pct_From_EMA20"]
    misaligned = df.copy()
    misaligned["Pct_From_EMA20"] = [
        donor.loc[shuffled[d]].to_numpy()[i % 8] for i, d in enumerate(df[DATE])
    ]
    folds = wf.make_folds(misaligned, max_folds=3)
    s = wf.evaluate(misaligned, folds, n_boot=200)["summary"]
    assert abs(s["rankers"]["ML"]["mean_ic"]) < 0.08


def test_evaluate_shapes_and_folds_report_purged_train_sizes():
    df, res = _run("none", max_folds=3)
    assert [f["fold"] for f in res["folds"]] == [1, 2, 3]
    for f in res["folds"]:
        assert f["n_train"] > 0 and f["n_test"] > 0
        assert f["weeks_used"] <= f["weeks_total"]
    assert res["folds"][0]["n_train"] < res["folds"][-1]["n_train"]


def test_report_formats_and_is_json_serialisable():
    df, res = _run("feature", max_folds=2)
    text = wf.format_report(res, {"source_csv": "x.csv"})
    assert "walk-forward" in text and "VERDICT" in text and "Paired ML minus Score" in text
    payload = {"folds": res["folds"], "summary": res["summary"]}
    json.dumps(wf._jsonable(payload))


def test_main_end_to_end_writes_json(tmp_path, capsys):
    df = _synthetic("feature")
    df["Outcome"] = "stopped"                       # leakage col that must be dropped
    df[config.RUBRIC_VERSION_COL] = config.RUBRIC_VERSION
    csv = tmp_path / "coiled_cobra_backtest_trades_2026-09-18.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "report.json"

    assert wf.main(["--csv", str(csv), "--max-folds", "2", "--n-boot", "200", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["meta"]["rubric_version"] == config.RUBRIC_VERSION
    assert payload["summary"]["n_folds"] == 2
    assert "VERDICT" in capsys.readouterr().out


def test_main_refuses_unversioned_csv(tmp_path):
    df = _synthetic("none")
    csv = tmp_path / "coiled_cobra_backtest_trades_2026-07-17.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="rubric version"):
        wf.main(["--csv", str(csv)])
