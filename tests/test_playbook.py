"""Home page sections by evidence, chart lines and Market rows."""

from __future__ import annotations

import pandas as pd
import pytest

from finance_vibe.web import playbook as pb


def _bo(symbol, status, readiness, **kw):
    row = {
        "Symbol": symbol,
        "Status": status,
        "Breakout Readiness": readiness,
        "Close": 48.68,
        "Resistance": 50.31,
        "Support": 44.62,
        "ATR": 1.70,
        "SMA20": 47.0,
        "RVOL20": 1.0,
    }
    row.update(kw)
    return row


def _cobra(symbol, score, swing_low=45.65):
    return {"Symbol": symbol, "Score": score, "Grade": "A - Watch", "Swing Low": swing_low}


def test_weekly_confirmed_only_and_marks_continuations():
    now = pd.DataFrame(
        [
            _bo("zeta ", "BREAKOUT_CONFIRMED", 10),
            _bo("ALFA", "BREAKOUT_CONFIRMED", 90),
            _bo("PRE", "PRE_BREAKOUT", 99),
            _bo("TRIG", "BREAKOUT_TRIGGERED", 99),
            _bo("FAIL", "FAILED_BREAKOUT", 99),
        ]
    )
    prev = pd.DataFrame([_bo("ZETA", "BREAKOUT_CONFIRMED", 50), _bo("ALFA", "WATCH", 50)])
    got = pb.weekly_confirmed(now, prev)
    assert list(got["Symbol"]) == ["ALFA", "ZETA"]  # by symbol, not by any score
    assert list(got["Confirmed Last Run"]) == [False, True]
    assert "Rank Score" not in got.columns


def test_weekly_confirmed_without_previous_run_or_status():
    now = pd.DataFrame([_bo("ALFA", "BREAKOUT_CONFIRMED", 90)])
    assert list(pb.weekly_confirmed(now, pd.DataFrame())["Confirmed Last Run"]) == [False]
    assert pb.weekly_confirmed(now.drop(columns=["Status"]), pd.DataFrame()).empty
    assert pb.weekly_confirmed(pd.DataFrame(), pd.DataFrame()).empty


def test_untested_cobra_carries_breakout_status():
    cobra = pd.DataFrame([_cobra("VRNS", 85), _cobra("LOST", 95)])
    breakout = pd.DataFrame([_bo("VRNS", "PRE_BREAKOUT", 10)])
    got = pb.untested_cobra(cobra, breakout).set_index("Symbol")
    assert list(got.index) == ["LOST", "VRNS"]
    assert got.loc["VRNS", "Breakout Status"] == "PRE_BREAKOUT"
    assert pd.isna(got.loc["LOST", "Breakout Status"])  # no breakout row: still listed
    assert pb.untested_cobra(pd.DataFrame(), breakout).empty


def test_failed_counts_per_mode():
    statuses = ["PRE_BREAKOUT", "BREAKOUT_TRIGGERED", "BREAKOUT_TRIGGERED", "BREAKOUT_CONFIRMED"]
    bo = pd.DataFrame([_bo(f"S{i}", s, 0) for i, s in enumerate([*statuses, "WATCH"])])
    assert pb.failed_counts(bo, "daily") == {
        "PRE_BREAKOUT": 1,
        "BREAKOUT_TRIGGERED": 2,
        "BREAKOUT_CONFIRMED": 1,
    }
    # Weekly CONFIRMED passed in-sample, so it is not a failed count.
    assert pb.failed_counts(bo, "weekly") == {"PRE_BREAKOUT": 1, "BREAKOUT_TRIGGERED": 2}
    assert pb.failed_counts(pd.DataFrame(), "weekly") == {
        "PRE_BREAKOUT": 0,
        "BREAKOUT_TRIGGERED": 0,
    }


def test_load_previous_breakout(tmp_path):
    for day, sym in (("2026-10-16", "OLD"), ("2026-10-23", "PREV"), ("2026-10-30", "NOW")):
        pd.DataFrame([_bo(sym, "BREAKOUT_CONFIRMED", 0)]).to_csv(
            tmp_path / f"breakout_setups_{day}.csv", index=False
        )
    assert list(pb.load_previous_breakout(str(tmp_path), "2026-10-30")["Symbol"]) == ["PREV"]
    assert pb.load_previous_breakout(str(tmp_path), "2026-10-16").empty
    assert pb.load_previous_breakout(str(tmp_path), None).empty


def test_load_latest_picks_the_newest_breakout_run(tmp_path):
    pd.DataFrame([_bo("A", "PRE_BREAKOUT", 10)]).to_csv(
        tmp_path / "breakout_setups_2026-10-19.csv", index=False
    )
    pd.DataFrame([_bo("B", "PRE_BREAKOUT", 10)]).to_csv(
        tmp_path / "breakout_setups_2026-10-20.csv", index=False
    )
    pd.DataFrame([_cobra("B", 80)]).to_csv(
        tmp_path / "coiled_cobra_setups_2026-10-20.csv", index=False
    )
    date, cobra, breakout = pb.load_latest(str(tmp_path))
    assert date == "2026-10-20" and list(breakout["Symbol"]) == ["B"] and len(cobra) == 1
    assert pb.load_latest(str(tmp_path / "missing"))[0] is None


def test_ticker_history_one_row_per_run_newest_first(tmp_path):
    pd.DataFrame(
        [_bo(" aaa", "WATCH", 40, **{"Breakout Triggered": True}), _bo("B", "WATCH", 5)]
    ).to_csv(tmp_path / "breakout_setups_2026-10-19.csv", index=False)
    pd.DataFrame([_bo("B", "PRE_BREAKOUT", 10)]).to_csv(
        tmp_path / "breakout_setups_2026-10-20.csv", index=False
    )
    pd.DataFrame([_bo("AAA", "PRE_BREAKOUT", 70, Close=51.5, Trend="BULLISH")]).to_csv(
        tmp_path / "breakout_setups_2026-10-21.csv", index=False
    )
    pd.DataFrame([_cobra("AAA", 82)]).to_csv(
        tmp_path / "coiled_cobra_setups_2026-10-21.csv", index=False
    )
    hist = pb.ticker_history(str(tmp_path), "aaa ")
    assert list(hist.columns) == pb.TICKER_HISTORY_COLUMNS
    assert list(hist["Date"]) == ["2026-10-21", "2026-10-19"]  # 10-20 has no AAA row
    newest, oldest = hist.iloc[0], hist.iloc[1]
    assert newest["Close"] == 51.5 and newest["Trend"] == "BULLISH"
    assert newest["Cobra Score"] == 82 and newest["Cobra Grade"] == "A - Watch"
    assert newest["in_cobra"] and newest["in_breakout"]
    assert oldest["Status"] == "BREAKOUT_TRIGGERED"  # old WATCH + trigger flag relabelled
    assert pd.isna(oldest["Cobra Score"]) and not oldest["in_cobra"]


def test_ticker_history_unknown_ticker_or_folder_is_empty(tmp_path):
    pd.DataFrame([_bo("B", "PRE_BREAKOUT", 10)]).to_csv(
        tmp_path / "breakout_setups_2026-10-20.csv", index=False
    )
    assert pb.ticker_history(str(tmp_path), "ZZZ").empty
    assert pb.ticker_history(str(tmp_path / "missing"), "B").empty


def _hrow(date, status="PRE_BREAKOUT", readiness=70, cobra=float("nan")):
    return {"Date": date, "Day": f"Day {date}", "Status": status,
            "Breakout Readiness": readiness, "Cobra Score": cobra}  # fmt: skip


def test_timeframe_compare_same_status_and_readiness_difference():
    weekly = [_hrow("2026-09-04", readiness=72, cobra=82), _hrow("2026-08-28", "WATCH")]
    daily = [_hrow("2026-09-05", readiness=65)]
    out = pb.timeframe_compare(
        weekly, daily, latest_runs={"weekly": "2026-09-04", "daily": "2026-09-05"}
    )
    assert out["status_same"] is True
    assert out["readiness_diff"] == 7.0
    assert out["cobra"] == "weekly only"
    assert out["weekly"]["cobra"] == 82.0 and out["daily"]["cobra"] is None
    assert out["weekly"]["day"] == "Day 2026-09-04"
    assert not out["weekly"]["stale"] and not out["daily"]["stale"]


def test_timeframe_compare_exact_names_not_groups():
    out = pb.timeframe_compare(
        [_hrow("2026-09-04", "BREAKOUT_CONFIRMED", cobra=80)],
        [_hrow("2026-09-05", "BREAKOUT_TRIGGERED", cobra=75)],
    )
    assert out["status_same"] is False and out["cobra"] == "both"


def test_timeframe_compare_stale_and_missing_sides():
    out = pb.timeframe_compare(
        [_hrow("2026-08-28")], [], latest_runs={"weekly": "2026-09-04", "daily": "2026-09-05"}
    )
    assert out["weekly"]["stale"] and out["daily"] is None
    assert out["status_same"] is None and out["readiness_diff"] is None
    assert out["cobra"] == "neither"
    only_daily = pb.timeframe_compare([], [_hrow("2026-09-05", cobra=90)])
    assert only_daily["weekly"] is None and only_daily["cobra"] == "daily only"
    assert pb.timeframe_compare([], []) is None


def test_timeframe_compare_cobra_only_row_has_no_status():
    out = pb.timeframe_compare(
        [_hrow("2026-09-04", status=float("nan"), readiness=float("nan"), cobra=88)],
        [_hrow("2026-09-05")],
    )
    assert out["weekly"]["status"] is None and out["status_same"] is None
    assert out["readiness_diff"] is None and out["cobra"] == "weekly only"


def test_known_symbols_from_the_newest_breakout_run(tmp_path):
    pd.DataFrame([_bo("old", "PRE_BREAKOUT", 10)]).to_csv(
        tmp_path / "breakout_setups_2026-10-19.csv", index=False
    )
    pd.DataFrame([_bo(" zz", "PRE_BREAKOUT", 10), _bo("AB", "WATCH", 5)]).to_csv(
        tmp_path / "breakout_setups_2026-10-20.csv", index=False
    )
    assert pb.known_symbols(str(tmp_path)) == ["AB", "ZZ"]
    assert pb.known_symbols(str(tmp_path / "missing")) == []


# ---------------------------------------------------------------------------
# Market fixtures
# ---------------------------------------------------------------------------


def _spy(**kw):
    row = {
        "Symbol": "SPY",
        "Close": 769.05,
        "Resistance": 781.62,
        "Support": 758.79,
        "ATR": 7.08,
        "SMA20": 770.62,
        "SMA50": 767.26,
        "Trend": "NEUTRAL",
        "MTF": "PARTIAL",
    }
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# Market facts (no bias rule)
# ---------------------------------------------------------------------------


def test_market_facts_are_the_states_and_the_50_day_side():
    facts = pb.market_facts(_spy(Close=760.0), {"Trend": "BEARISH", "MTF": "PARTIAL"})
    assert facts == [
        "weekly trend BEARISH",
        "weekly MTF PARTIAL",
        "daily close below its 50-day average (767.26)",
    ]
    assert pb.market_facts(_spy(), {"Trend": "BULLISH", "MTF": "ALIGNED"})[-1].startswith(
        "daily close above"
    )


def test_market_facts_say_what_is_missing():
    assert pb.market_facts(None, None) == ["no weekly scan row", "no daily scan row"]
    assert pb.market_facts(_spy(SMA50=None), {"Trend": "BULLISH", "MTF": "ALIGNED"})[-1] == (
        "daily 50-day average unavailable"
    )
    assert not hasattr(pb, "market_bias")  # the LONG / SHORT / NEUTRAL rule is gone


def test_market_rows_carry_facts_price_and_support_resistance_lines():
    daily = pd.DataFrame([_spy(), _spy(Symbol="QQQ")])
    weekly = pd.DataFrame([_spy(Trend="BULLISH", MTF="ALIGNED")])
    rows = {r["symbol"]: r for r in pb.market_rows(daily, weekly, {"SPY": 790.0, "QQQ": "N/A"})}
    assert list(rows) == ["SPY", "QQQ", "SCHG", "SCHD"]
    assert rows["SPY"]["facts"][:2] == ["weekly trend BULLISH", "weekly MTF ALIGNED"]
    assert "bias" not in rows["SPY"] and "plans" not in rows["SPY"]
    assert rows["SPY"]["price"] == 790.0 and rows["SPY"]["price_is_live"]
    assert rows["QQQ"]["facts"][0] == "no weekly scan row"
    assert rows["QQQ"]["price"] == 769.05 and not rows["QQQ"]["price_is_live"]
    assert [(line["label"], line["price"]) for line in rows["SPY"]["levels"]] == [
        ("Resistance", 781.62),
        ("Support", 758.79),
    ]
    assert rows["SCHG"]["levels"] is None and rows["SCHD"]["price"] is None


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "N/A", None, True])
def test_unusable_live_price_is_none(bad):
    assert pb.live_price_or_none(bad) is None
    assert pb.live_price_or_none(48.68) == 48.68


@pytest.mark.parametrize(
    ("close", "pct", "outside"),
    [
        (100.0, 50.0, None),
        (85.5, 0.0, "below support by 5.00%"),  # pinned at the left end
        (115.5, 100.0, "above resistance by 5.00%"),
    ],
)
def test_position_bar_between_support_and_resistance(close, pct, outside):
    bar = pb.position_bar({"Close": close, "Support": 90.0, "Resistance": 110.0})
    assert bar["pct"] == pytest.approx(pct) and bar["outside"] == outside
    assert (bar["support"], bar["resistance"], bar["close"]) == (90.0, 110.0, close)
    assert bar["side"] == {85.5: "below", 115.5: "above"}.get(close)


def test_position_bar_needs_all_three_levels():
    assert pb.position_bar({"Close": 1.0, "Support": 2.0}) is None
    assert pb.position_bar({"Close": 1.0, "Support": 2.0, "Resistance": 2.0}) is None


def _closes(dates, closes):
    return pd.DataFrame({"Date": pd.to_datetime(dates), "Close": [float(c) for c in closes]})


def test_recent_closes_last_n_up_to_the_run_date():
    days = pd.bdate_range("2026-10-01", periods=15)
    bars = _closes(days, range(1, 16))
    assert pb.recent_closes(bars, str(days[11].date()), weekly=False) == [
        float(c) for c in range(3, 13)
    ]  # 10 closes ending on the run date, nothing after it
    assert pb.recent_closes(bars, str(days[2].date()), weekly=False) == [1.0, 2.0, 3.0]
    assert pb.recent_closes(None, "2026-10-10", weekly=False) == []


def test_recent_closes_weekly_skips_the_week_in_progress():
    mondays = pd.date_range("2026-10-05", periods=4, freq="W-MON")
    bars = _closes(mondays, [10, 11, 12, 13])
    # Tuesday 2026-10-27: the week of 10-26 is not complete yet.
    assert pb.recent_closes(bars, "2026-10-27", weekly=True) == [10.0, 11.0, 12.0]
    assert pb.recent_closes(bars, "2026-10-30", weekly=True) == [10.0, 11.0, 12.0, 13.0]


def test_sparkline_geometry_against_the_level():
    s = pb.sparkline([10.0, 12.0, 11.0], 14.0)
    assert s is not None and s["count"] == 3
    top, bottom = pb.SPARK_PAD, pb.SPARK_HEIGHT - pb.SPARK_PAD
    assert s["level_y"] == top  # the level is the highest value: top of the box
    xs, ys = zip(*(map(float, p.split(",")) for p in s["points"].split()), strict=True)
    assert xs[0] == pb.SPARK_PAD and xs[-1] == pb.SPARK_WIDTH - pb.SPARK_PAD
    assert ys[0] == bottom  # 10 is the lowest value
    assert s["above"] is False and s["vs_level_pct"] == pytest.approx(100 * (11 / 14 - 1))
    assert (s["last_x"], s["last_y"]) == (xs[-1], ys[-1])


def test_sparkline_above_flat_no_level_and_too_short():
    assert pb.sparkline([10.0, 13.0], 12.0)["above"] is True
    assert pb.sparkline([10.0, 12.0], 12.0)["above"] is False  # at the level is not through
    flat = pb.sparkline([5.0, 5.0, 5.0], 5.0)
    assert flat["level_y"] == pb.SPARK_HEIGHT / 2 and flat["last_y"] == pb.SPARK_HEIGHT / 2
    none = pb.sparkline([1.0, 2.0], float("nan"))
    assert none["level_y"] is None and none["above"] is None and none["vs_level_pct"] is None
    assert pb.sparkline([1.0], 2.0) is None and pb.sparkline([], 2.0) is None
