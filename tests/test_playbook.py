"""Top setups: ranking across both scanners, rule-based levels and plan status."""

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


def test_ranking_blends_both_scanners_and_rewards_agreement():
    breakout = pd.DataFrame(
        [
            _bo("COBRA", "WATCH", 40),  # Cobra-only: rank = Cobra Score 85
            _bo("BOTH", "BREAKOUT_TRIGGERED", 50),  # max(80, 50 + 10) + 10 = 90
            _bo("CONF", "BREAKOUT_CONFIRMED", 70),  # 70 + 20 = 90, no Cobra -> after BOTH
            _bo("PRE", "PRE_BREAKOUT", 50),  # 50 + 15 = 65
            _bo("QUIET", "WATCH", 99),  # not an event, not a Cobra setup
            _bo("FAIL", "FAILED_BREAKOUT", 80),  # failed: dropped even with a Cobra row
        ]
    )
    cobra = pd.DataFrame([_cobra("COBRA", 85), _cobra("BOTH", 80), _cobra("FAIL", 95)])
    ranked = pb.candidates(cobra, breakout)
    assert list(ranked["Symbol"]) == ["BOTH", "CONF", "COBRA", "PRE"]
    assert list(ranked["Rank Score"]) == [90, 90, 85, 65]
    assert ranked.set_index("Symbol").loc["BOTH", "Sources"] == ("cobra", "breakout")
    assert list(pb.top_setups(cobra, breakout, n=3)["Symbol"]) == ["BOTH", "CONF", "COBRA"]


def test_rows_without_levels_are_dropped():
    breakout = pd.DataFrame(
        [_bo("NOATR", "BREAKOUT_CONFIRMED", 60, ATR=None), _bo("OK", "PRE_BREAKOUT", 10)]
    )
    assert list(pb.candidates(pd.DataFrame(), breakout)["Symbol"]) == ["OK"]


def test_empty_inputs_give_no_candidates():
    assert pb.candidates(pd.DataFrame(), pd.DataFrame()).empty


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
    assert plan["targets"] == pytest.approx([50.31 + 2 * risk, 50.31 + 3 * risk], abs=0.005)
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
    assert plan["targets"] == pytest.approx([758.79 - 2 * risk, 758.79 - 3 * risk], abs=0.01)
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
    assert [line["label"] for line in lines] == ["Trigger", "Stop", "T1", "T2", "Support"]
    assert lines[0]["price"] == 50.31 and lines[-1]["price"] == 44.62


@pytest.mark.parametrize(
    ("bias", "labels"),
    [
        ("LONG", ["Trigger", "Stop", "T1", "T2", "Short trigger"]),
        ("SHORT", ["Short Trigger", "Short Stop", "Short T1", "Short T2", "Long trigger"]),
        ("NEUTRAL", ["Break up", "Break down", "Long stop", "Short stop"]),
    ],
)
def test_market_chart_levels(bias, labels):
    long, short = pb.action_plan(_spy()), pb.action_plan(_spy(), side="short")
    assert [line["label"] for line in pb.market_chart_levels(bias, long, short)] == labels


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


def test_market_rows_build_both_plans_per_symbol():
    daily = pd.DataFrame([_spy(), _spy(Symbol="QQQ")])
    weekly = pd.DataFrame([_spy(Trend="BULLISH", MTF="ALIGNED")])
    rows = {r["symbol"]: r for r in pb.market_rows(daily, weekly, {"SPY": 790.0})}
    assert list(rows) == ["SPY", "QQQ", "SCHG"]
    assert rows["SPY"]["bias"] == "LONG" and rows["SPY"]["plans"]["long"]["status"] == "EXTENDED"
    assert rows["QQQ"]["bias"] == "NEUTRAL" and rows["QQQ"]["plans"] is not None  # no weekly row
    assert rows["SCHG"]["plans"] is None and rows["SCHG"]["levels"] is None


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "N/A", None, True])
def test_unusable_live_price_falls_back_to_close(bad):
    plan = pb.action_plan(_row(), live_price=bad)
    assert plan["status"] == "WAITING" and plan["price"] == 48.68 and not plan["price_is_live"]


def test_breakout_csv_without_status_column_still_ranks_cobra_setups():
    breakout = pd.DataFrame([_bo("COBRA", "WATCH", 40)]).drop(columns=["Status"])
    ranked = pb.candidates(pd.DataFrame([_cobra("COBRA", 85)]), breakout)
    assert list(ranked["Symbol"]) == ["COBRA"] and list(ranked["Rank Score"]) == [85]


def test_cobra_setup_without_breakout_row_is_logged(caplog):
    caplog.set_level("INFO")
    breakout = pd.DataFrame([_bo("OTHER", "PRE_BREAKOUT", 10)])
    ranked = pb.candidates(pd.DataFrame([_cobra("LOST", 95)]), breakout)
    assert "LOST" not in set(ranked["Symbol"])
    assert "no breakout row" in caplog.text and "LOST" in caplog.text


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
