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
