"""Breakout Readiness experiment: predicates, causality, IC, protocol plumbing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from finance_vibe import breakout_experiment as bx
from finance_vibe import breakout_scanner as bs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _weekly_csv(path, n=140, seed=7, bump_after=None):
    """Synthetic weekly OHLCV; optionally distort every bar after ``bump_after``."""
    rng = np.random.default_rng(seed)
    close = 50 * np.cumprod(1 + rng.normal(0.004, 0.03, n))
    dates = pd.date_range("2016-01-04", periods=n, freq="W-MON")
    df = pd.DataFrame({
        "Date": dates, "Open": close * 0.99, "High": close * 1.03,
        "Low": close * 0.97, "Close": close, "Volume": rng.integers(1e5, 1e6, n).astype(float),
    })
    if bump_after is not None:
        cols = ["Open", "High", "Low", "Close"]
        df.loc[bump_after + 1:, cols] *= 3.0
        df.loc[bump_after + 1:, "Volume"] *= 10
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# variant predicates
# ---------------------------------------------------------------------------

def test_status_variants():
    pre = bx.variant_flags({"Status": bs.STATUS_PRE, "Breakout Readiness": 40})
    assert pre["BK_PRE"] and pre["BK_PRE_OR_CONF"] and pre["BK_DISPLAY"]
    assert not pre["BK_CONFIRMED"] and not pre["BK_READY70"] and not pre[bx.NEGATIVE]
    conf = bx.variant_flags({"Status": bs.STATUS_CONFIRMED, "Breakout Readiness": 90})
    assert conf["BK_CONFIRMED"] and conf["BK_PRE_OR_CONF"] and conf["BK_READY70"]


def test_display_floors_match_dashboard_rule():
    assert bx.variant_flags({"Status": bs.STATUS_WATCH, "Breakout Readiness": 55})["BK_DISPLAY"]
    assert not bx.variant_flags({"Status": bs.STATUS_WATCH, "Breakout Readiness": 54})["BK_DISPLAY"]
    assert bx.variant_flags({"Status": bs.STATUS_DEV, "Breakout Readiness": 50})["BK_DISPLAY"]
    assert not bx.variant_flags({"Status": bs.STATUS_DEV, "Breakout Readiness": 49})["BK_DISPLAY"]


def test_failed_is_negative_only():
    f = bx.variant_flags({"Status": bs.STATUS_FAILED, "Breakout Readiness": 95})
    assert f[bx.NEGATIVE]
    assert not any(f[v] for v in bx.QUALIFYING)


def test_ready70_boundary_and_missing_score():
    assert bx.variant_flags({"Status": bs.STATUS_WATCH, "Breakout Readiness": 70})["BK_READY70"]
    assert not bx.variant_flags({"Status": bs.STATUS_WATCH, "Breakout Readiness": 69})["BK_READY70"]
    assert not bx.variant_flags({"Status": bs.STATUS_WATCH, "Breakout Readiness": None})["BK_READY70"]


# ---------------------------------------------------------------------------
# causality: a row only depends on bars up to its own date
# ---------------------------------------------------------------------------

_SIGNAL_COLS = ["status", "ready", "penalty", *bx.PILLARS, "trend", "volatility", "momentum",
                "structure", "mtf", "dist_res_atr", "rvol20", "atr"]


def test_rows_ignore_future_bars(tmp_path):
    cut = 110
    _, base, _ = bx._ticker_pass(str(_weekly_csv(tmp_path / "AAA_10y_1wk.csv")))
    _, bumped, _ = bx._ticker_pass(str(_weekly_csv(tmp_path / "BBB_10y_1wk.csv", bump_after=cut)))
    a = pd.DataFrame(base).set_index("idx")
    b = pd.DataFrame(bumped).set_index("idx")
    assert a.index.min() == bs.MIN_PRIMARY_BARS - 1
    pd.testing.assert_frame_equal(a.loc[:cut, _SIGNAL_COLS], b.loc[:cut, _SIGNAL_COLS])
    # forward outcomes *should* see the future
    assert not np.allclose(a.loc[cut - 5, "mu13"], b.loc[cut - 5, "mu13"])


def test_rows_match_live_scanner_on_cut_data(tmp_path):
    path = _weekly_csv(tmp_path / "CCC_10y_1wk.csv")
    _, rows, _ = bx._ticker_pass(str(path))
    df = bs.normalize_ohlcv(pd.read_csv(path))
    for rec in rows[::15]:
        live = bs.evaluate_ticker(df.iloc[: rec["idx"] + 1], "CCC", "weekly", "weekly")
        assert live["Status"] == rec["status"]
        assert live["Breakout Readiness"] == rec["ready"]


def test_random_control_is_deterministic(tmp_path):
    path = str(_weekly_csv(tmp_path / "DDD_10y_1wk.csv"))
    a = [r["f_C0_random"] for r in bx._ticker_pass(path)[1]]
    b = [r["f_C0_random"] for r in bx._ticker_pass(path)[1]]
    assert a == b and any(a)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def _ic_frame(sign: float, weeks=30, names=12):
    rows = []
    for w in range(weeks):
        d = pd.Timestamp("2020-01-06") + pd.Timedelta(weeks=w)
        for k in range(names):
            rows.append({"date": d, "ready": k, "fr13": sign * k * 0.01})
    return pd.DataFrame(rows)


def test_weekly_ic_perfect_and_inverse():
    assert bx.weekly_ic(_ic_frame(1.0), "ready", 200)["mean_ic"] == 1.0
    neg = bx.weekly_ic(_ic_frame(-1.0), "ready", 200)
    assert neg["mean_ic"] == -1.0 and neg["hi"] == -1.0


def test_weekly_ic_skips_thin_weeks():
    assert bx.weekly_ic(_ic_frame(1.0, names=5), "ready", 50)["weeks"] == 0


def _bars_table(n_weeks=300, names=8, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(names):
        for w in range(n_weeks):
            rec = {"sym": f"S{s}", "idx": w, "date": pd.Timestamp("2016-01-04") + pd.Timedelta(weeks=w),
                   "close": 10.0, "atr": 1.0, "status": bs.STATUS_WATCH, "ready": float(rng.integers(0, 100)),
                   "penalty": 0, **{p: float(rng.integers(0, 15)) for p in bx.PILLARS},
                   "mu13": 0.1, "dd13": -0.05, "mu26": 0.2, "dd26": -0.1, "fr13": rng.normal(0, 0.1),
                   "tr_r": rng.normal(0.2, 1.0), "tr_censored": False, "tr_mfe_r": 1.0}
            for v in bx.VARIANTS:
                rec[f"f_{v}"] = False
            rec["f_C0_random"] = bool(rng.random() < 0.3)
            rec["f_BK_DISPLAY"] = bool(rng.random() < 0.3)
            rows.append(rec)
    return pd.DataFrame(rows)


def test_run_analysis_structure_and_unspent_lockbox():
    bars = _bars_table()
    last = bars["date"].max() + pd.Timedelta(weeks=1)
    res = bx.run_analysis(bars, last, n_boot=100)
    assert set(res["qualification"]) == set(bx.QUALIFYING)
    assert res["dev"]["BK_PRE"]["episodes"] == 0
    assert not res["qualification"]["BK_PRE"]["qualified"]
    assert res["report_only"]["cobra"] == {"available": False}
    assert "ready" in res["report_only"]["ic_dev"]
    assert bx.format_report(res)          # renders without error
    # lockbox is only spent on a qualifier, and only once
    assert res["lockbox"]["spent"] == any(q["qualified"] for q in res["qualification"].values())


def test_dev_period_excludes_lockbox_and_live():
    bars = _bars_table()
    last = bars["date"].max() + pd.Timedelta(weeks=1)
    res = bx.run_analysis(bars, last, n_boot=50)
    lock_weeks = bx.lx.LOCKBOX_WEEKS + bx.lx.CENSOR_WEEKS
    expected_dev = int((bars["date"] < last - pd.Timedelta(weeks=lock_weeks)).sum())
    assert res["dev"]["all_bars"]["bars"] == expected_dev
