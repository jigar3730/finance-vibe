"""Leader-Expansion vs Coiled Cobra experiment: predicates, exit model, stats, protocol."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_vibe import coiled_cobra_leader_experiment as lx


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rec(**over):
    """A bar profile that passes the baseline and L1-L4 unless overridden."""
    base = dict(fails=[], score=75.0, rs=20, struct=15, ovhd=10, rvol=8,
                a_close_gt_slow=True, a_c3_slow_rising=True)
    base.update(over)
    return base


def _ohlc(rows):
    """rows of (open, high, low, close) -> frame."""
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])


# ---------------------------------------------------------------------------
# variant predicates
# ---------------------------------------------------------------------------

def test_baseline_needs_no_gate_fail_and_score_70():
    assert lx.variant_flags(_rec(score=70.0))[lx.BASELINE]
    assert not lx.variant_flags(_rec(score=69.99))[lx.BASELINE]
    assert not lx.variant_flags(_rec(fails=["D"]))[lx.BASELINE]


def test_leaders_ignore_coil_gates_but_require_trend_and_market_gates():
    coil_fail = lx.variant_flags(_rec(fails=["C", "D"], score=40.0))
    assert not coil_fail[lx.BASELINE]
    assert all(coil_fail[v] for v in lx.LEADERS)
    for gate in ("A", "B"):
        assert not any(lx.variant_flags(_rec(fails=[gate]))[v] for v in lx.LEADERS)


def test_l4_rs_and_open_sky_boundaries():
    assert lx.variant_flags(_rec(rs=18, ovhd=8))["L4_trend_rs18_sky8"]
    assert not lx.variant_flags(_rec(rs=17, ovhd=8))["L4_trend_rs18_sky8"]
    assert not lx.variant_flags(_rec(rs=18, ovhd=7))["L4_trend_rs18_sky8"]


def test_l1_l2_l3_boundaries():
    assert lx.variant_flags(_rec(rs=14, struct=8, ovhd=5))["L1_trend_rs_struct_sky"]
    assert not lx.variant_flags(_rec(rs=13, struct=8, ovhd=5))["L1_trend_rs_struct_sky"]
    assert not lx.variant_flags(_rec(rs=14, struct=7, ovhd=5))["L1_trend_rs_struct_sky"]
    assert not lx.variant_flags(_rec(rs=14, struct=8, ovhd=4))["L1_trend_rs_struct_sky"]
    assert lx.variant_flags(_rec(rvol=6))["L2_L1_plus_rvol"]
    assert not lx.variant_flags(_rec(rvol=5))["L2_L1_plus_rvol"]
    assert lx.variant_flags(_rec(struct=0, ovhd=0, rvol=6))["L3_trend_rs_rvol"]
    assert not lx.variant_flags(_rec(struct=0, ovhd=0, rvol=5))["L3_trend_rs_rvol"]


def test_r1_relaxes_gate_a_c_and_score_floor():
    # Gate A failed only on fast>slow; close>slow and slow rising -> relaxed pass.
    assert lx.variant_flags(_rec(fails=["A"], score=60.0))["R1_relaxed_gates"]
    assert not lx.variant_flags(_rec(fails=["A"], a_close_gt_slow=False))["R1_relaxed_gates"]
    assert not lx.variant_flags(_rec(fails=["A"], a_c3_slow_rising=False))["R1_relaxed_gates"]
    # Gate C needs structure only (>= 8).
    assert lx.variant_flags(_rec(fails=["C"], struct=8))["R1_relaxed_gates"]
    assert not lx.variant_flags(_rec(fails=["C"], struct=7))["R1_relaxed_gates"]
    # Gates B and D are NOT relaxed; score floor is 60.
    assert not lx.variant_flags(_rec(fails=["B"]))["R1_relaxed_gates"]
    assert not lx.variant_flags(_rec(fails=["D"]))["R1_relaxed_gates"]
    assert not lx.variant_flags(_rec(score=59.99))["R1_relaxed_gates"]


# ---------------------------------------------------------------------------
# trail exit model
# ---------------------------------------------------------------------------

def test_trail_stop_gap_through_fills_at_open():
    df = _ohlc([(100, 101, 99, 100),        # signal bar
                (100, 101, 99, 100),        # entry bar (open 100)
                (90, 91, 88, 89)])          # gaps below the initial stop
    tr = lx.simulate_trail_trade(df, 1, atr=2.0, slippage=0.0)
    assert tr["outcome"] == "stopped" and tr["exit_idx"] == 2 and not tr["censored"]
    assert tr["exit_price"] == pytest.approx(90.0)          # open, not the (higher) stop
    assert tr["r"] < -1.0                                    # gap loss exceeds 1R


def test_trail_ratchets_off_highest_close_then_trails_out():
    # entry 100, ATR 2 -> initial stop 95, trail = highest close - 6.
    df = _ohlc([(100, 101, 99, 100),
                (100, 111, 100, 110),       # entry bar; close 110 -> stop 104
                (110, 121, 109, 120),       # close 120 -> stop 114
                (119, 120, 113, 115)])      # low 113 <= 114 -> trailed out at 114
    tr = lx.simulate_trail_trade(df, 1, atr=2.0, slippage=0.0)
    assert tr["outcome"] == "trailed"
    assert tr["exit_price"] == pytest.approx(114.0)
    assert tr["r"] == pytest.approx((114 - 100) / 5.0)


def test_trail_same_bar_high_cannot_rescue_a_stop_out():
    # Entry bar trades down through the stop on the same bar it spikes to a huge high.
    df = _ohlc([(100, 101, 99, 100),
                (100, 150, 94, 149)])
    tr = lx.simulate_trail_trade(df, 1, atr=2.0, slippage=0.0)
    assert tr["outcome"] == "stopped"
    assert tr["exit_price"] == pytest.approx(95.0)
    assert tr["r"] == pytest.approx(-1.0)


def test_trail_time_exit_and_censoring():
    flat = [(100, 101, 99, 100)] * 6
    expired = lx.simulate_trail_trade(_ohlc(flat), 1, atr=2.0, slippage=0.0, max_hold=3)
    assert expired["outcome"] == "expired" and not expired["censored"] and expired["exit_idx"] == 3
    open_ = lx.simulate_trail_trade(_ohlc(flat[:3]), 1, atr=2.0, slippage=0.0, max_hold=10)
    assert open_["outcome"] == "open" and open_["censored"]


def test_trail_rejects_degenerate_inputs():
    df = _ohlc([(100, 101, 99, 100)] * 3)
    assert lx.simulate_trail_trade(df, 3, atr=2.0) is None          # no bar to enter on
    assert lx.simulate_trail_trade(df, 1, atr=0.0) is None
    assert lx.simulate_trail_trade(df, 1, atr=60.0) is None          # stop would be <= 0


# ---------------------------------------------------------------------------
# episodes, runs, bootstrap
# ---------------------------------------------------------------------------

def test_dedup_keeps_first_bar_of_each_consecutive_run_per_ticker():
    f = pd.DataFrame({"sym": ["A", "A", "A", "A", "B", "B"], "idx": [10, 11, 12, 20, 11, 12]})
    kept = lx.dedup_episodes(f)
    assert list(zip(kept["sym"], kept["idx"])) == [("A", 10), ("A", 20), ("B", 11)]


def test_find_runs_finds_the_double():
    lows = np.array([10.0] * 5 + [10, 12, 15, 18, 22, 25] + [25.0] * 5)
    highs = lows * 1.02
    runs = lx.find_runs(highs, lows)
    assert len(runs) == 1
    gain, i, j = runs[0]
    assert gain >= 1.0 and i < j and lows[i] == 10.0
    assert lx.find_runs(highs * 0 + 11, lows * 0 + 10) == []         # +10% is not a monster


def test_cluster_stats_deterministic_and_brackets_mean():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"date": pd.date_range("2020-01-06", periods=200, freq="W").repeat(2),
                      "x": rng.normal(0.5, 1.0, 400)})
    a, b = lx.cluster_stats(d, "x", 500, seed=1), lx.cluster_stats(d, "x", 500, seed=1)
    assert a == b
    assert a["n"] == 400 and a["lo"] < a["mean"] < a["hi"] and a["lo"] > 0 and a["t"] > 1.645
    assert lx.cluster_stats(d.iloc[0:0], "x", 10)["n"] == 0


def test_paired_diff_sign_and_ci():
    weeks = pd.date_range("2020-01-06", periods=150, freq="W")
    rng = np.random.default_rng(1)
    a = pd.DataFrame({"date": weeks, "x": rng.normal(1.0, 0.5, 150)})
    b = pd.DataFrame({"date": weeks, "x": rng.normal(0.0, 0.5, 150)})
    d = lx.paired_diff(a, b, "x", 500, seed=2)
    assert d["diff"] > 0.7 and d["lo"] > 0
    assert np.isnan(lx.paired_diff(a.iloc[0:0], b, "x", 10)["diff"])


# ---------------------------------------------------------------------------
# qualification protocol
# ---------------------------------------------------------------------------

def _res(episodes=150, lo=0.1, useful=0.6, dd=0.3):
    return {"episodes": episodes, "trail": {"lo": lo}, "capture": {"useful_rate": useful},
            "forward": {"dd26_worse25": dd}}


def test_qualifies_requires_every_check():
    base = _res(useful=0.3, dd=0.25)
    good = lx.qualifies("L4", _res(useful=0.5, dd=0.30), base, {"lo": 0.05})
    assert good["qualified"]
    # each check individually breaks it
    assert not lx.qualifies("v", _res(episodes=99, useful=0.5, dd=0.3), base, {"lo": 0.05})["qualified"]
    assert not lx.qualifies("v", _res(lo=0.0, useful=0.5, dd=0.3), base, {"lo": 0.05})["qualified"]
    assert not lx.qualifies("v", _res(useful=0.44, dd=0.3), base, {"lo": 0.05})["qualified"]      # < base + 0.15
    assert not lx.qualifies("v", _res(useful=0.5, dd=0.36), base, {"lo": 0.05})["qualified"]      # > base + 0.10
    assert not lx.qualifies("v", _res(useful=0.5, dd=0.3), base, {"lo": -0.01})["qualified"]


def test_split_periods_are_disjoint_and_ordered():
    last = pd.Timestamp("2026-09-07")
    dates = [last - pd.Timedelta(weeks=w) for w in (200, 105, 103, 60, 53, 51, 1)]
    per = lx.split_periods(pd.DataFrame({"date": dates}), last)
    assert list(per) == ["dev", "dev", "lockbox", "lockbox", "lockbox", "live", "live"]


# ---------------------------------------------------------------------------
# end-to-end analysis on synthetic bars
# ---------------------------------------------------------------------------

def _synthetic_bars(seed=0, n_weeks=420, edge=0.0, probs=None):
    """Random bars; ``probs`` overrides per-variant flag probability, ``edge`` is L4's mean trail-R."""
    rng = np.random.default_rng(seed)
    probs = probs or {}
    weeks = pd.date_range("2018-01-01", periods=n_weeks, freq="W")
    rows = []
    for sym in ("AAA", "BBB", "CCC", "MU"):                      # MU is a discovery ticker
        for k, d in enumerate(weeks):
            flags = {f"f_{v}": bool(rng.random() < probs.get(v, 0.05)) for v in lx.VARIANTS}
            r = rng.normal(edge if flags["f_L4_trend_rs18_sky8"] else 0.0, 1.0)
            rows.append({"sym": sym, "idx": 160 + k, "date": d, "close": 100.0,
                         "mu13": rng.uniform(0, 0.5), "dd13": -rng.uniform(0, 0.2),
                         "mu26": rng.uniform(0, 1.2), "dd26": -rng.uniform(0, 0.2),
                         "pl_outcome": "stopped", "pl_r": -1.0, "tr_outcome": "expired",
                         "tr_r": r, "tr_censored": False, "tr_mfe_r": abs(r), "tr_bars": 10, **flags})
    return pd.DataFrame(rows), weeks[-1] + pd.Timedelta(weeks=1)


def _synthetic_runs(weeks_start="2018-01-01"):
    d = pd.Timestamp(weeks_start)
    return pd.DataFrame([{"sym": "AAA", "low_idx": 200, "peak_idx": 230, "gain": 2.0, "low": 10.0, "high": 300.0,
                          "low_date": d + pd.Timedelta(weeks=40), "peak_date": d + pd.Timedelta(weeks=70)}])


def test_run_analysis_random_data_leaves_lockbox_unspent():
    bars, last = _synthetic_bars()
    res = lx.run_analysis(bars, _synthetic_runs(), last, n_boot=200)
    assert res["tickers"] == {"holdout": 3, "discovery": 1}
    assert res["lockbox"]["spent"] is False
    assert not any(q["qualified"] for q in res["qualification"].values())
    assert set(res["qualification"]) == set(lx.QUALIFYING)
    text = lx.format_report(res)
    assert "LOCKBOX left unspent" in text and "HOLDOUT" in text


def test_discovery_tickers_are_never_used_for_qualification():
    bars, last = _synthetic_bars()
    only_disc = bars.copy()
    only_disc["sym"] = "MU"
    res = lx.run_analysis(only_disc, _synthetic_runs(), last, n_boot=100)
    assert res["tickers"]["holdout"] == 0
    assert res["holdout_dev"][lx.BASELINE]["episodes"] == 0
    assert not any(q["qualified"] for q in res["qualification"].values())


def test_run_analysis_spends_lockbox_once_when_a_variant_has_real_edge():
    bars, last = _synthetic_bars(seed=3, n_weeks=520, edge=1.5,
                                 probs={"L4_trend_rs18_sky8": 0.5, lx.BASELINE: 0.02})
    # keep the baseline out of the monster run window so L4's capture margin (Q2) is real
    win = (bars["sym"] == "AAA") & bars["idx"].between(196, 230)
    bars.loc[win, f"f_{lx.BASELINE}"] = False
    bars.loc[win, "f_L4_trend_rs18_sky8"] = True
    res = lx.run_analysis(bars, _synthetic_runs(), last, n_boot=300)
    q = res["qualification"]["L4_trend_rs18_sky8"]
    assert q["qualified"], q["checks"]
    assert res["lockbox"]["spent"] is True
    assert res["lockbox"]["best"] == "L4_trend_rs18_sky8"
    assert res["lockbox"]["passed"] is True
    assert "LOCKBOX spent once on L4_trend_rs18_sky8" in lx.format_report(res)
