"""RESEARCH.md §13 ranker on synthetic data only: labels, seal, gap, pickers, decision rule."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import prebreakout_fixture as pf
from finance_vibe import config
from finance_vibe.research import ranker as rk
from finance_vibe.research import ranker_features as rf
from test_research_ranker_features import _write_snapshot

CAL = pd.bdate_range("2017-12-01", "2026-05-29")
TRAIN = rf.session_grid(CAL, config.RESEARCH_RANKER_TRAIN_STRIDE)
SCORE = rf.session_grid(CAL, config.RESEARCH_RANKER_SCORE_STRIDE)
N_STOCKS = 60


def _panel(signal: float, seed: int = 0, momentum_signal: float = 0.0) -> pd.DataFrame:
    """A ranked, labelled panel: R20 = signal × (RS_QQQ_63 − 0.5) + noise.

    ``momentum_signal`` puts the same kind of signal on MOM_12_1, the baseline.
    """
    rng = np.random.default_rng(seed)
    n = len(TRAIN) * N_STOCKS
    frame = pd.DataFrame(
        {
            "Date": np.repeat(TRAIN, N_STOCKS),
            "Symbol": np.tile([f"S{i:02d}" for i in range(N_STOCKS)], len(TRAIN)),
        }
    )
    for name in rf.FEATURES:
        frame[name] = rng.random(n)
    frame[rk.LABEL] = (
        signal * (frame["RS_QQQ_63"] - 0.5)
        + momentum_signal * (frame["MOM_12_1"] - 0.5)
        + rng.normal(0, 0.05, n)
    )
    pos = np.searchsorted(CAL, frame["Date"])
    frame["exit_date"] = CAL[np.minimum(pos + config.RESEARCH_RANKER_HORIZON, len(CAL) - 1)]
    frame["label_ok"] = True
    frame["split_suspect"] = False
    frame[f"SPY_{rk.LABEL}"] = 0.01
    return frame


def _bars(seed: int, days: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    open_ = close * (1 + rng.normal(0, 0.003, len(days)))
    return pd.DataFrame(
        {
            "Date": days,
            "Open": open_,
            "High": np.maximum(open_, close) * 1.01,
            "Low": np.minimum(open_, close) * 0.99,
            "Close": close,
            "Volume": 1e6,
        }
    )


# --- labels and the seal ----------------------------------------------------


def test_label_is_r20_from_next_open_to_the_20th_close():
    days = pd.bdate_range("2018-01-02", periods=80)
    bars = {"AAA": _bars(1, days), "SPY": _bars(2, days)}
    keys = pd.DataFrame({"Symbol": ["AAA", "AAA"], "Date": [days[10], days[70]]})
    out = rk.label_panel(keys, bars)
    a = bars["AAA"]
    assert out.loc[0, rk.LABEL] == pytest.approx(a["Close"][30] / a["Open"][11] - 1)
    assert out.loc[0, "exit_date"] == days[30] and out.loc[0, "label_ok"]
    assert not out.loc[1, "label_ok"]  # only 9 bars left: short window


def test_labels_refuse_the_sealed_holdout():
    days = pd.bdate_range("2026-03-02", periods=60)
    bars = {"AAA": _bars(1, days), "SPY": _bars(2, days)}
    keys = pd.DataFrame({"Symbol": ["AAA"], "Date": [pd.Timestamp("2026-04-01")]})
    with pytest.raises(ValueError, match="holdout is sealed"):
        rk.label_panel(keys, bars)


def test_grades_are_within_date_quintiles():
    frame = pd.DataFrame(
        {"Date": pd.Timestamp("2020-01-02"), "Symbol": list("ABCDEFGHIJ"), rk.LABEL: range(10)}
    )
    assert list(rk.grades(frame)) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


# --- walk-forward gap -------------------------------------------------------


def test_no_training_label_reaches_the_test_fold():
    frame = _panel(signal=0.0)
    for fold in config.RESEARCH_RANKER_FOLDS:
        train, test = rk.fold_rows(frame, SCORE, fold)
        first = test["Date"].min()
        assert train["exit_date"].max() < first
        gap = np.searchsorted(CAL, first) - np.searchsorted(CAL, train["Date"].max())
        assert gap >= config.RESEARCH_RANKER_HORIZON + 1
        assert set(test["Date"]) <= set(SCORE)
        assert test["Date"].between(pd.Timestamp(fold[0]), pd.Timestamp(fold[1])).all()


def test_unusable_rows_are_left_out_of_both_sides():
    frame = _panel(signal=0.0)
    frame.loc[frame["Symbol"] == "S00", "label_ok"] = False
    train, test = rk.fold_rows(frame, SCORE, config.RESEARCH_RANKER_FOLDS[0])
    assert "S00" not in set(train["Symbol"]) | set(test["Symbol"])


# --- pickers ----------------------------------------------------------------


def test_top_k_rounds_up():
    assert rk.top_k(497) == 50 and rk.top_k(10) == 1 and rk.top_k(1) == 1 and rk.top_k(11) == 2


def test_date_table_pickers_and_ties():
    day = pd.DataFrame(
        {
            "Date": pd.Timestamp("2020-02-03"),
            "Symbol": [f"S{i:02d}" for i in range(20)],
            rk.LABEL: np.arange(20) / 100.0,
            rk.SCORE: np.arange(20.0),  # model ranks by return: perfect
            "MOM_12_1": -np.arange(20.0),  # momentum ranks backwards
            "fold": 1,
            f"SPY_{rk.LABEL}": 0.0,
        }
    )
    t = rk.date_table(day).iloc[0]
    assert t["k"] == 2
    assert t["model"] == pytest.approx((0.19 + 0.18) / 2)
    assert t["momentum"] == pytest.approx((0.00 + 0.01) / 2)
    assert t["eligible_mean"] == pytest.approx(0.095)
    assert t["model_top3"] == pytest.approx(0.18)
    assert t["rank_ic"] == pytest.approx(1.0)
    assert t["A"] == pytest.approx(t["model"] - t["eligible_mean"])
    assert t["B"] == pytest.approx(t["model"] - t["momentum"])
    again = rk.date_table(day.sample(frac=1, random_state=3)).iloc[0]
    assert again["random"] == t["random"]  # seeded, row order does not matter


def test_random_picker_lands_near_the_eligible_mean():
    frame = _panel(signal=0.0, seed=5)
    test = frame[frame["Date"].isin(SCORE)].assign(fold=1)
    test[rk.SCORE] = np.arange(len(test), dtype="float64")
    table = rk.date_table(test)
    assert abs((table["random"] - table["eligible_mean"]).mean()) < 0.01


# --- decision rule ----------------------------------------------------------


def _table(a: list[float], b: list[float], folds: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Date": pd.bdate_range("2020-01-01", periods=len(a)), "A": a, "B": b, "fold": folds}
    )


def test_decide_needs_both_tests_and_the_fold_rule():
    folds = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    good = [0.02, 0.03, 0.025, 0.02, 0.03, 0.02, 0.025, 0.03, 0.02, 0.03, 0.025, 0.02]
    assert rk.decide(_table(good, good, folds)).passed
    noise = [0.02, -0.03, 0.01, -0.02, 0.0, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.0]
    assert not rk.decide(_table(good, noise, folds)).passed  # fails vs momentum
    two_folds = [0.05, 0.05, 0.05, 0.05, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001]
    two_folds += [-0.001, -0.001]
    d = rk.decide(_table(good, two_folds, folds))
    assert d.folds_b_positive == 2 and not d.passed  # B positive in only 2 of 6 folds


def test_decide_never_passes_an_undefined_test():
    d = rk.decide(_table([0.01, 0.01, 0.01], [0.02, 0.03, 0.04], [1, 2, 3]))
    assert math.isnan(d.p_a) and math.isnan(d.p) and not d.passed  # zero variance in A


# --- end to end on synthetic data ------------------------------------------


@pytest.fixture(scope="module")
def planted():
    pytest.importorskip("lightgbm")
    scored, importance = rk.walk_forward(_panel(signal=0.4, seed=1), SCORE)
    return scored, importance


def test_walk_forward_finds_a_planted_signal(planted):
    scored, importance = planted
    d = rk.decide(rk.date_table(scored))
    assert d.passed and d.mean_a > 0 and d.mean_b > 0
    lo, hi = config.RESEARCH_RANKER_FOLDS[0][0], config.RESEARCH_RANKER_FOLDS[-1][1]
    assert d.n_dates == len(SCORE[(SCORE >= lo) & (SCORE <= hi)]) and d.n_folds == 6
    top = importance.groupby("feature")["gain"].sum().idxmax()
    assert top == "RS_QQQ_63"
    assert (importance["train_last_exit"] < importance["test_first_date"]).all()


def test_walk_forward_rejects_pure_noise():
    pytest.importorskip("lightgbm")
    scored, _ = rk.walk_forward(_panel(signal=0.0, seed=2), SCORE)
    assert not rk.decide(rk.date_table(scored)).passed


def test_walk_forward_does_not_beat_a_momentum_baseline_that_holds_the_signal():
    pytest.importorskip("lightgbm")
    scored, _ = rk.walk_forward(_panel(signal=0.0, seed=3, momentum_signal=0.4), SCORE)
    d = rk.decide(rk.date_table(scored))
    assert d.mean_a > 0  # it learns momentum and beats random ...
    assert not d.passed  # ... but adds nothing over the baseline itself


# --- the run-once report script, on a tiny synthetic snapshot ---------------


def test_insample_report_runs_end_to_end_on_synthetic_data(tmp_path, capsys):
    pytest.importorskip("lightgbm")
    snap, history, features_script = _write_snapshot(tmp_path)
    out = tmp_path / "out"
    base = ["--out-dir", str(out), "--stamp", "x", "--commit", "abc"]
    assert features_script.main(["--snapshot", str(snap), "--history", str(history), *base]) == 0
    features = out / "ranker_features_x"
    capsys.readouterr()

    script = pf.load_script("ranker_insample_report")
    assert script.main(["--features", str(features), "--snapshot", str(snap), *base]) == 0
    run = out / "ranker_insample_x"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["features_commit"] == "abc"
    assert manifest["decision"]["n_folds"] == 6
    dates = pd.read_csv(run / "dates.csv", parse_dates=["Date"])
    assert dates["Date"].max() <= pd.Timestamp(config.RESEARCH_IN_SAMPLE_END)
    assert "Decision:" in capsys.readouterr().out

    # a feature file edited after the blind run is refused
    (features / "features.csv.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed since"):
        script.main(
            [
                "--features",
                str(features),
                "--snapshot",
                str(snap),
                *base[:2],
                "--stamp",
                "y",
                "--commit",
                "abc",
            ]
        )
