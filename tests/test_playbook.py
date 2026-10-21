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
