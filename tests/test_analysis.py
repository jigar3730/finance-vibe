"""Standard analysis queries: runs measured correctly, one snapshot, descriptive counts."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from finance_vibe.data import analysis as an

DATES = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]


def _bars(symbol: str, states: list) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbol, "date": DATES[: len(states)], "state": states})


def _cobra(symbol, day, passed, grade="B - Watch", snapshot="2026-11-09", **points):
    return {
        "snapshot": snapshot,
        "rubric_version": 4.0,
        "mode": "daily",
        "symbol": symbol,
        "date": day,
        "passed": passed,
        "grade": grade,
        "vol_contraction_points": points.get("vol", 20),
        "structure_points": points.get("struct", 20),
        "relative_strength_points": 10,
        "volume_shelf_points": 10,
        "overhead_clearance_points": 10,
        "rvol_trigger_points": 5,
    }


def _breakout(symbol, day, status, snapshot="2026-11-09"):
    return {
        "snapshot": snapshot,
        "rubric_version": 1.0,
        "mode": "daily",
        "symbol": symbol,
        "date": day,
        "status": status,
    }


def _live_cobra(day, symbol, score=80):
    return {
        "run_date": day,
        "mode": "daily",
        "symbol": symbol,
        "score": score,
        "grade": "B - Watch",
    }


def _live_breakout(day, symbol, status):
    row = {"run_date": day, "mode": "daily", "symbol": symbol, "status": status}
    return {**row, "breakout_readiness": 50}


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    gate = "Rejected - Gate Fail"
    cobra = [
        _cobra("AAA", DATES[0], 0, f"{gate} (C)", vol=5),
        _cobra("AAA", DATES[1], 1),
        _cobra("AAA", DATES[2], 1),
        _cobra("AAA", DATES[3], 0, f"{gate} (A/C)", struct=2),
        _cobra("AAA", DATES[4], 1),
        _cobra("AAA", DATES[5], 0, "Rejected - Below Threshold"),
        _cobra("OLD", DATES[0], 1, snapshot="2025-01-01"),
    ]
    breakout = [
        _breakout("AAA", DATES[0], "WATCH"),
        _breakout("AAA", DATES[1], "PRE_BREAKOUT"),
        _breakout("AAA", DATES[2], "PRE_BREAKOUT"),
        _breakout("AAA", DATES[3], "BREAKOUT_CONFIRMED"),
        _breakout("AAA", DATES[4], "WATCH"),
        _breakout("AAA", DATES[5], "WATCH"),
    ]
    pd.DataFrame(cobra).to_sql(an.COBRA, connection, index=False)
    pd.DataFrame(breakout).to_sql(an.BREAKOUT, connection, index=False)
    pd.DataFrame([_live_cobra("2026-01-05", "AAA"), _live_cobra("2026-01-06", "BBB", 90)]).to_sql(
        "live_cobra", connection, index=False
    )
    pd.DataFrame(
        [
            _live_breakout("2026-01-05", "AAA", "WATCH"),
            _live_breakout("2026-01-05", "BBB", "WATCH"),
            _live_breakout("2026-01-06", "AAA", "BREAKOUT_TRIGGERED"),
            _live_breakout("2026-01-06", "BBB", "WATCH"),
        ]
    ).to_sql("live_breakout", connection, index=False)
    yield connection
    connection.close()


def test_runs_split_on_state_and_ticker():
    bars = pd.concat([_bars("AAA", [1, 1, 0, 1]), _bars("BBB", [1, 1])], ignore_index=True)
    out = an.runs(bars)
    assert list(out["bars"]) == [2, 1, 1, 2]
    assert list(out["symbol"]) == ["AAA", "AAA", "AAA", "BBB"]


def test_runs_touching_either_end_are_open():
    out = an.runs(_bars("AAA", [1, 0, 0, 1, 1, 0]))
    assert list(out["bars"]) == [1, 2, 2, 1]
    assert list(out["open"]) == [True, False, False, True]


def test_length_figures_leave_out_open_runs(conn):
    row = an.qualifying_runs(conn, "daily").iloc[0]
    assert row["Runs"] == 2
    assert row["Open runs"] == 0
    assert row["Longest bars"] == 2
    assert row["One-bar runs"] == 1


def test_state_durations(conn):
    out = an.state_durations(conn, "daily").set_index("Status")
    assert out.loc["PRE_BREAKOUT", "Median bars"] == 2
    assert out.loc["WATCH", "Runs"] == 0
    assert out.loc["WATCH", "Open runs"] == 2


def test_blockers_count_every_gate_and_sole_blockers(conn):
    out = an.blockers(conn, "daily").set_index("Blocker")
    assert out.loc[an.GATE_NAMES["C"], "Rejected bars"] == 2
    assert out.loc[an.GATE_NAMES["C"], "Only blocker"] == 1
    assert out.loc[an.GATE_NAMES["A"], "Only blocker"] == 0
    assert out.loc[an.BELOW_THRESHOLD, "Rejected bars"] == 1
    assert out.loc[an.GATE_NAMES["C"], "Pct of rejected"] == pytest.approx(66.7)


def test_gate_c_detail(conn):
    out = an.gate_c_detail(conn, "daily").set_index("Gate C miss")["Bars"]
    assert out["Volatility contraction only"] == 1
    assert out["Structure only"] == 1
    assert out["Both"] == 0


def test_older_snapshot_is_ignored(conn):
    assert "OLD" not in set(an.ticker_counts(conn, "daily")["Symbol"])


def test_per_day_and_year(conn):
    daily = an.per_day(conn, "daily")
    assert list(daily["cobra_passed"]) == [0, 1, 1, 0, 1, 0]
    year = an.per_year(daily).iloc[0]
    assert (year["Year"], year["Cobra bars"], year["Cobra passes"]) == ("2026", 6, 3)


def test_overlaps(conn):
    out = an.overlaps(conn, "daily").set_index("Status")
    assert out.loc["PRE_BREAKOUT", "Cobra passes"] == 2
    assert out.loc["WATCH", "All bars"] == 3
    assert out["Cobra passes"].sum() == 3


def test_current_overlaps_use_the_newest_run(conn):
    out = an.current_overlaps(conn, "daily")
    assert list(out["symbol"]) == ["BBB"]
    assert list(out["status"]) == ["WATCH"]


def test_changes_between_runs(conn):
    row = an.changes_between_runs(conn, "daily").iloc[0]
    assert (row["Entered"], row["Left"], row["Status changed"]) == (1, 1, 1)


def test_status_transitions(conn):
    out = an.status_transitions(conn, "daily")
    assert out.to_dict("records") == [{"From": "WATCH", "To": "BREAKOUT_TRIGGERED", "Tickers": 1}]


def test_ticker_counts(conn):
    row = an.ticker_counts(conn, "daily").iloc[0]
    assert row["Symbol"] == "AAA"
    assert row["Cobra bars passed"] == 3
    assert row["Cobra stretches"] == 2
    assert row["Breakout confirmed stretches"] == 1
    assert row["Pct of bars passed"] == 50.0


def test_empty_table_is_rejected():
    connection = sqlite3.connect(":memory:")
    connection.execute(f"CREATE TABLE {an.COBRA} (snapshot, rubric_version)")
    with pytest.raises(ValueError, match="empty"):
        an.qualifying_runs(connection, "daily")
    connection.close()


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "none.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        an.connect(path)
    assert not path.exists()
