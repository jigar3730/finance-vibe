"""Home page sections by evidence, rule-based levels and plan status, Market rows."""

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


def _row(**kw):
    row = _bo("VRNS", "WATCH", 72)
    row["Cobra Swing Low"] = 45.65
    row.update(kw)
    return pd.Series(row)


def test_plan_levels_from_resistance_support_and_swing_low():
    plan = pb.action_plan(_row())
    stop = max(44.62, 45.65) - 0.25 * 1.70
    risk = 50.31 - stop
    assert plan["trigger"] == 50.31
    assert plan["support"] == 44.62
    assert plan["stop"] == pytest.approx(stop, abs=0.005)
    assert plan["risk"] == pytest.approx(risk, abs=0.005)
    assert "targets" not in plan
    assert plan["risk_pct"] == pytest.approx(100 * risk / 50.31, abs=0.05)
    assert plan["wide_stop"]  # ~10% > 8%


def test_swing_low_above_the_trigger_is_ignored():
    plan = pb.action_plan(_row(**{"Cobra Swing Low": 60.0}))
    assert plan["stop"] == pytest.approx(44.62 - 0.25 * 1.70, abs=0.01)


@pytest.mark.parametrize(
    ("price", "status"),
    [(48.68, "WAITING"), (50.5, "TRIGGERED"), (52.5, "EXTENDED"), (44.0, "INVALIDATED")],
)
def test_status_follows_the_price(price, status):
    plan = pb.action_plan(_row(), live_price=price)
    assert plan["status"] == status and plan["price_is_live"]


def test_status_uses_close_without_a_live_price():
    plan = pb.action_plan(_row())
    assert plan["status"] == "WAITING" and plan["price"] == 48.68 and not plan["price_is_live"]
    assert plan["note"].startswith("1.0 ATR below")


def test_confirmed_breakout_note():
    plan = pb.action_plan(_row(Status="BREAKOUT_CONFIRMED"), live_price=50.5)
    assert plan["status"] == "TRIGGERED" and "confirmed" in plan["note"]


def test_plan_needs_levels():
    with pytest.raises(ValueError):
        pb.action_plan(_row(ATR=0))
    with pytest.raises(ValueError):
        pb.action_plan(_row(Support=None, **{"Cobra Swing Low": None}))


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
# Short side and chart lines
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


def test_short_plan_mirrors_the_long_plan():
    plan = pb.action_plan(_spy(), side="short")
    stop = min(781.62, 770.62) + 0.25 * 7.08
    risk = stop - 758.79
    assert plan["side"] == "short" and plan["trigger"] == 758.79
    assert plan["stop"] == pytest.approx(stop, abs=0.01)
    assert plan["risk"] == pytest.approx(risk, abs=0.01)
    assert plan["condition"].startswith("Close below 758.79")
    long = pb.action_plan(_spy())
    assert long["trigger"] == 781.62 and long["stop"] == pytest.approx(
        758.79 - 0.25 * 7.08, abs=0.01
    )


@pytest.mark.parametrize(
    ("price", "status"),
    [(769.05, "WAITING"), (755.0, "TRIGGERED"), (745.0, "EXTENDED"), (775.0, "INVALIDATED")],
)
def test_short_status_follows_the_price(price, status):
    assert pb.action_plan(_spy(), live_price=price, side="short")["status"] == status


def test_bad_side_is_rejected():
    with pytest.raises(ValueError):
        pb.action_plan(_spy(), side="sideways")


def test_chart_levels_long_with_support():
    lines = pb.chart_levels(pb.action_plan(_row()), with_support=True)
    assert [line["label"] for line in lines] == ["Trigger", "Stop", "Support"]
    assert lines[0]["price"] == 50.31 and lines[-1]["price"] == 44.62


# ---------------------------------------------------------------------------
# Market bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("weekly", "daily_close", "bias"),
    [
        ({"Trend": "BULLISH", "MTF": "ALIGNED"}, 769.05, "LONG"),
        ({"Trend": "BULLISH", "MTF": "PARTIAL"}, 760.0, "NEUTRAL"),  # below SMA50
        ({"Trend": "BULLISH", "MTF": "DIVERGENT"}, 769.05, "NEUTRAL"),
        ({"Trend": "BEARISH", "MTF": "PARTIAL"}, 760.0, "SHORT"),
        ({"Trend": "NEUTRAL", "MTF": "BEARISH"}, 760.0, "SHORT"),
        ({"Trend": "BEARISH", "MTF": "BEARISH"}, 769.05, "NEUTRAL"),  # above SMA50
        ({"Trend": "NEUTRAL", "MTF": "PARTIAL"}, 769.05, "NEUTRAL"),
    ],
)
def test_market_bias_rule(weekly, daily_close, bias):
    got, reasons = pb.market_bias(_spy(Close=daily_close), weekly)
    assert got == bias
    assert reasons[0] == f"weekly trend {weekly['Trend']}"


def test_market_bias_is_neutral_without_rows():
    assert pb.market_bias(None, {"Trend": "BULLISH"})[0] == "NEUTRAL"
    assert pb.market_bias(_spy(SMA50=None), {"Trend": "BULLISH", "MTF": "ALIGNED"})[0] == "NEUTRAL"


def test_market_rows_carry_bias_price_and_support_resistance_lines():
    daily = pd.DataFrame([_spy(), _spy(Symbol="QQQ")])
    weekly = pd.DataFrame([_spy(Trend="BULLISH", MTF="ALIGNED")])
    rows = {r["symbol"]: r for r in pb.market_rows(daily, weekly, {"SPY": 790.0, "QQQ": "N/A"})}
    assert list(rows) == ["SPY", "QQQ", "SCHG", "SCHD"]
    assert rows["SPY"]["bias"] == "LONG" and "plans" not in rows["SPY"]
    assert rows["SPY"]["price"] == 790.0 and rows["SPY"]["price_is_live"]
    assert rows["QQQ"]["bias"] == "NEUTRAL"  # no weekly row
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
def test_unusable_live_price_falls_back_to_close(bad):
    plan = pb.action_plan(_row(), live_price=bad)
    assert plan["status"] == "WAITING" and plan["price"] == 48.68 and not plan["price_is_live"]


def test_extended_note_uses_the_threshold(monkeypatch):
    monkeypatch.setattr(pb, "EXTENDED_ATR", 1.5)
    plan = pb.action_plan(_row(), live_price=50.31 + 2 * 1.70)
    assert plan["status"] == "EXTENDED" and "More than 1.5 ATR" in plan["note"]


def test_short_side_says_it_is_not_a_scanner_signal():
    plan = pb.action_plan(_spy(), live_price=755.0, side="short")
    assert "not a scanner signal" in plan["condition"]
    assert "does not confirm breakdowns" in plan["note"]


def test_rvol_threshold_comes_from_config():
    from finance_vibe import config
    from finance_vibe.scanners import breakout as bs

    assert pb.RVOL_CONFIRM == bs.RVOL_CONFIRM == config.BREAKOUT_RVOL_CONFIRM
