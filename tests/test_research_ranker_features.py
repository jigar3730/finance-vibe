"""RESEARCH.md §13 features: exact definitions, no lookahead, ranks, eligibility, grids."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import prebreakout_fixture as pf
from finance_vibe import config, indicators
from finance_vibe.research import ranker_features as rf
from finance_vibe.research import snapshot as sn

N_BARS = 400


def _bars(seed: int, n: int = N_BARS, start: str = "2017-01-02", price: float = 50.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = price * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(500_000, 2_000_000, n).astype("float64")
    return pd.DataFrame(
        {"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def _levels(bars: pd.DataFrame, symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": bars["Date"],
            "Symbol": symbol,
            "Resistance": bars["High"].rolling(20, min_periods=1).max(),
            "Support": bars["Low"].rolling(20, min_periods=1).min(),
        }
    )


@pytest.fixture
def market():
    return {"QQQ": _bars(1, price=300.0), "SPY": _bars(2, price=400.0)}


def test_features_match_the_frozen_definitions(market):
    b = _bars(3)
    lv = _levels(b)
    f = rf.ticker_features(b, market, lv)
    t = 300
    c, h, v = b["Close"], b["High"], b["Volume"]
    q, s = market["QQQ"]["Close"], market["SPY"]["Close"]
    atr = indicators.atr(b["High"], b["Low"], c, length=14)
    obv = indicators.obv(c, v)
    expected = {
        "RS_QQQ_63": (c[t] / c[t - 63]) / (q[t] / q[t - 63]) - 1,
        "RS_QQQ_126": (c[t] / c[t - 126]) / (q[t] / q[t - 126]) - 1,
        "RS_SPY_63": (c[t] / c[t - 63]) / (s[t] / s[t - 63]) - 1,
        "RS_SPY_126": (c[t] / c[t - 126]) / (s[t] / s[t - 126]) - 1,
        "MOM_12_1": c[t - 21] / c[t - 252] - 1,
        "REV_1M": c[t] / c[t - 21] - 1,
        "HIGH_52W": c[t] / h[t - 251 : t + 1].max() - 1,
        "ATR_PCT": atr[t] / c[t],
        "RES_ATR": (lv["Resistance"][t] - c[t]) / atr[t],
        "SUP_ATR": (c[t] - lv["Support"][t]) / atr[t],
        "OBV_SLOPE_20": (obv[t] - obv[t - 20]) / v[t - 19 : t + 1].sum(),
        "VOL_RATIO": v[t] / v[t - 20 : t].mean(),
    }
    for name, value in expected.items():
        assert f.loc[t, name] == pytest.approx(value, rel=1e-12), name


def test_features_need_full_lookback(market):
    f = rf.ticker_features(_bars(3), market, _levels(_bars(3)))
    assert f.loc[250, "MOM_12_1"] != f.loc[250, "MOM_12_1"]  # NaN: needs bar t-252
    assert f.loc[250, "HIGH_52W"] != f.loc[250, "HIGH_52W"]  # NaN: needs 252 highs
    assert f.loc[252, rf.FEATURES].notna().all()


def test_no_lookahead_cutting_later_bars_changes_nothing(market):
    b = _bars(4)
    lv = _levels(b)
    full = rf.ticker_features(b, market, lv)
    t = 320
    cut = b["Date"][t]
    short = rf.ticker_features(
        b[b["Date"] <= cut],
        {k: m[m["Date"] <= cut] for k, m in market.items()},
        lv[lv["Date"] <= cut],
    )
    pd.testing.assert_series_equal(
        full.loc[t, rf.FEATURES], short.loc[t, rf.FEATURES], check_names=False
    )


def test_benchmark_is_taken_on_the_tickers_own_dates(market):
    b = _bars(5).drop(index=[200, 201]).reset_index(drop=True)  # two missing sessions
    f = rf.ticker_features(b, market, None)
    t = 300
    q = market["QQQ"].set_index("Date")["Close"]
    c = b["Close"]
    expected = (c[t] / c[t - 63]) / (q[b["Date"][t]] / q[b["Date"][t - 63]]) - 1
    assert f.loc[t, "RS_QQQ_63"] == pytest.approx(expected, rel=1e-12)
    assert f[["RES_ATR", "SUP_ATR"]].isna().all().all()  # no levels given


def test_zero_volume_never_gives_infinite_features(market):
    b = _bars(6)
    b.loc[100:130, "Volume"] = 0.0
    f = rf.ticker_features(b, market, _levels(b))
    assert np.isfinite(f[rf.FEATURES].to_numpy(dtype="float64")[~f[rf.FEATURES].isna()]).all()


def _universe(n: int = 4):
    bars = {f"S{i}": _bars(10 + i, price=40.0 + 10 * i) for i in range(n)}
    bars["S0"]["Volume"] = 1_000.0  # never reaches the $20M dollar-volume check
    levels = pd.concat([_levels(b, s) for s, b in bars.items()], ignore_index=True)
    bars |= {"QQQ": _bars(1, price=300.0), "SPY": _bars(2, price=400.0)}
    return bars, levels


def test_panel_eligibility_needs_membership_and_every_feature():
    bars, levels = _universe()
    panel = rf.build_panel(bars, ["S0", "S1", "S2", "S3"], levels)
    assert list(panel.columns) == rf.PANEL_COLUMNS
    assert not panel.loc[panel["Symbol"] == "S0", "eligible"].any()
    early = panel["Date"] < bars["S1"]["Date"][252]
    assert not panel.loc[early, "eligible"].any()  # MOM_12_1 / HIGH_52W undefined
    late = panel[(panel["Symbol"] != "S0") & ~early]
    assert late["eligible"].all()


def test_panel_keeps_only_requested_dates_and_rejects_missing_bars():
    bars, levels = _universe()
    grid = bars["SPY"]["Date"][260::5]
    panel = rf.build_panel(bars, ["S1", "S2"], levels, dates=grid)
    assert set(panel["Date"]) == set(grid)
    with pytest.raises(ValueError, match="no bars"):
        rf.build_panel({k: v for k, v in bars.items() if k != "QQQ"}, ["S1"], levels)


def test_rank_features_are_within_date_percentiles_of_eligible_rows():
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 2),
            "Symbol": ["A", "B", "C", "A", "B"],
            "in_universe": True,
            "eligible": [True, True, False, True, True],
        }
    )
    for i, name in enumerate(rf.FEATURES):
        panel[name] = [1.0 + i, 3.0, 99.0, 5.0, 2.0]
    ranked = rf.rank_features(panel)
    assert list(ranked["Symbol"]) == ["A", "B", "A", "B"]  # C not eligible
    first = ranked[ranked["Date"] == "2020-01-02"]
    assert list(first["RS_QQQ_63"]) == [0.5, 1.0]  # A=1 < B=3
    assert list(first["RS_SPY_63"]) == [0.75, 0.75]  # tie at 3
    assert ranked[rf.FEATURES].min().min() > 0 and ranked[rf.FEATURES].max().max() <= 1
    assert list(ranked.loc[ranked["Date"] == "2020-01-03", "VOL_RATIO"]) == [1.0, 0.5]


def test_session_grid_is_anchored_and_sealed():
    cal = pd.bdate_range("2017-12-20", "2026-06-30")
    g5 = rf.session_grid(cal, 5)
    g20 = rf.session_grid(cal, 20)
    assert g5[0] == pd.Timestamp(config.RESEARCH_RANKER_IN_SAMPLE_START)
    assert g20[0] == g5[0] and set(g20) <= set(g5)  # the score grid sits on the training grid
    assert g20.max() <= pd.Timestamp(config.RESEARCH_IN_SAMPLE_END)
    later = rf.session_grid(cal, 20, end="2026-06-30")
    assert list(later[: len(g20)]) == list(g20)  # the holdout continues the same grid
    with pytest.raises(ValueError):
        rf.session_grid(cal, 0)


# --- the blind run script ---------------------------------------------------


def _write_snapshot(tmp_path):
    """A verified 3-ticker snapshot (plus QQQ, SPY) that runs past the 2026-03-31 seal."""
    days = pd.bdate_range("2017-01-02", "2026-05-29")
    frames = {s: _bars(i, n=len(days)) for i, s in enumerate(["AAA", "BBB", "CCC"], start=20)}
    frames |= {"QQQ": _bars(1, n=len(days), price=300.0), "SPY": _bars(2, n=len(days), price=400.0)}
    snap = tmp_path / "snap"
    (snap / "daily").mkdir(parents=True)
    names = []
    for s, b in frames.items():
        name = f"daily/{s}_10y_1d.csv"
        b.to_csv(snap / name, index=False)
        names.append(name)
    (snap / sn.MANIFEST).write_text(
        json.dumps({"stamp": "2026-11-09", "files": {n: sn._sha256(snap / n) for n in names}})
    )
    pd.DataFrame(
        {"Symbol": ["AAA", "BBB", "CCC", "QQQ", "SPY"], "In Universe": [True] * 3 + [False] * 2}
    ).to_csv(snap / sn.UNIVERSE, index=False)
    hist = tmp_path / "hist" / f"rubric-{config.BREAKOUT_RUBRIC_VERSION}"
    hist.mkdir(parents=True)
    levels = pd.concat(
        [_levels(frames[s], s) for s in ["AAA", "BBB", "CCC"]], ignore_index=True
    ).assign(**{"Rubric Version": config.BREAKOUT_RUBRIC_VERSION})
    levels.to_csv(hist / "breakout_daily_history.csv.gz", index=False)
    return snap, tmp_path / "hist", pf.load_script("research_ranker_features")


def test_blind_run_writes_features_and_counts_only(tmp_path, capsys):
    snap, history, script = _write_snapshot(tmp_path)
    out = tmp_path / "out"
    argv = ["--snapshot", str(snap), "--history", str(history), "--out-dir", str(out)]
    assert script.main([*argv, "--stamp", "x", "--commit", "abc"]) == 0
    run = out / "ranker_features_x"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["commit"] == "abc" and manifest["universe_tickers"] == 3
    assert manifest["first_date"] == config.RESEARCH_RANKER_IN_SAMPLE_START
    assert manifest["last_date"] <= config.RESEARCH_IN_SAMPLE_END  # holdout sealed
    assert set(manifest["counts"]["score_dates_by_fold"]) == {"0", "1", "2", "3", "4", "5", "6"}
    panel = pd.read_csv(run / "features.csv.gz", parse_dates=["Date"])
    assert list(panel.columns) == rf.PANEL_COLUMNS
    assert panel["Date"].max() <= pd.Timestamp(config.RESEARCH_IN_SAMPLE_END)
    # every grid date is past the 252-bar lookback, so only membership removes rows
    assert (panel["eligible"] == panel["in_universe"]).all() and panel["eligible"].any()
    printed = capsys.readouterr().out
    assert "R20" not in printed and "return" not in printed.lower()
    with pytest.raises(FileExistsError):
        script.main([*argv, "--stamp", "x", "--commit", "abc"])


def test_blind_run_requires_a_commit(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    with pytest.raises(SystemExit):
        pf.load_script("research_ranker_features").main(["--out-dir", str(tmp_path)])
