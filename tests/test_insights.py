"""web.insights: ETF freshness and bar-quality lookups for the dashboard."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.web import insights

EXPECTED = {"daily": date(2026, 10, 30), "weekly": date(2026, 10, 26)}


@pytest.fixture
def raw_silo(tmp_path, monkeypatch):
    """Point both modes' raw dirs at tmp; returns a writer for one ticker's bars."""
    for mode in ("daily", "weekly"):
        (tmp_path / mode).mkdir()
        monkeypatch.setitem(config.TIMEFRAME_PROFILES[mode], "raw_dir", str(tmp_path / mode))

    def write(symbol: str, mode: str, last: date, *, bad_last: bool = False) -> None:
        freq = "W-MON" if mode == "weekly" else "B"
        dates = pd.date_range(end=pd.Timestamp(last), periods=30, freq=freq)
        close = np.linspace(30.0, 35.0, len(dates))
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": close,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": 1e6,
            }
        )
        if bad_last:
            df.loc[df.index[-1], "High"] = df["Open"].iloc[-1] - 0.5
        profile = config.TIMEFRAME_PROFILES[mode]
        df.to_csv(config.get_raw_path(symbol, profile), index=False)

    return write


def test_fresh_rows_and_clean_raw_are_ok(raw_silo):
    raw_silo("SCHD", "daily", EXPECTED["daily"])
    raw_silo("SCHD", "weekly", EXPECTED["weekly"])
    rows = {"daily": {"AsOf Date": "2026-10-30"}, "weekly": {"AsOf Date": "2026-10-26"}}
    got = insights.etf_freshness("SCHD", rows, EXPECTED)
    assert got["daily"] == {
        "as_of": "2026-10-30",
        "expected": "2026-10-30",
        "ok": True,
        "flags": [],
        "issues": [],
    }
    assert got["weekly"]["ok"]


def test_stale_row_missing_row_and_bad_raw_bar_are_flagged(raw_silo):
    raw_silo("SCHD", "daily", date(2026, 10, 29), bad_last=True)
    rows = {"daily": {"AsOf Date": "2026-10-29"}, "weekly": None}
    got = insights.etf_freshness("SCHD", rows, EXPECTED)
    # The raw file's own stale flag isn't repeated next to the row's.
    assert got["daily"]["flags"] == ["stale", "ohlc_incoherent"] and not got["daily"]["ok"]
    assert got["weekly"]["flags"] == ["missing"] and got["weekly"]["as_of"] is None


def test_row_without_a_date_is_stale(raw_silo):
    got = insights.etf_freshness("SCHD", {"daily": {"Close": 1.0}, "weekly": {}}, EXPECTED)
    assert got["daily"]["flags"] == ["stale"] and got["weekly"]["flags"] == ["stale"]


def test_quality_errors_keeps_errors_only(tmp_path):
    pd.DataFrame(
        [
            {
                "Symbol": "AAA",
                "Check": "stale",
                "Severity": "error",
                "Date": "2026-10-30",
                "Detail": "",
            },
            {
                "Symbol": "AAA",
                "Check": "ohlc_incoherent",
                "Severity": "error",
                "Date": "2026-10-30",
                "Detail": "",
            },
            {
                "Symbol": "BBB",
                "Check": "big_move",
                "Severity": "warn",
                "Date": "2026-10-30",
                "Detail": "",
            },
        ]
    ).to_csv(tmp_path / "data_quality_2026-10-30.csv", index=False)
    pd.DataFrame([{"Symbol": "OLD", "Check": "stale", "Severity": "error"}]).to_csv(
        tmp_path / "data_quality_2026-10-29.csv", index=False
    )
    assert insights.quality_errors(str(tmp_path)) == {"AAA": ["ohlc_incoherent", "stale"]}
    assert insights.quality_errors(str(tmp_path / "none")) == {}


# ---------------------------------------------------------------------------
# Run changes
# ---------------------------------------------------------------------------


def _bo(rows):
    return pd.DataFrame([{"Symbol": s, "Status": st, "Breakout Readiness": r} for s, st, r in rows])


def test_status_changes_buckets_and_skips_universe_churn():
    prev = _bo(
        [
            ("UP", "WATCH", 40),
            ("FAIL", "BREAKOUT_TRIGGERED", 60),
            ("SIDE", "PRE_BREAKOUT", 55),
            ("SAME", "WATCH", 30),
            ("GONE", "WATCH", 10),
        ]
    )
    now = _bo(
        [
            ("UP", "BREAKOUT_CONFIRMED", 70),
            ("FAIL", "FAILED_BREAKOUT", 20),
            ("SIDE", "WATCH", 45),
            ("SAME", "WATCH", 35),
            ("NEW", "PRE_BREAKOUT", 80),
        ]
    )
    got = insights.status_changes(now, prev)
    assert [r["Symbol"] for r in got["upgrades"]] == ["UP"]
    assert (
        got["upgrades"][0]["Status Prev"] == "WATCH"
        and got["upgrades"][0]["Readiness Change"] == 30
    )
    assert [r["Symbol"] for r in got["failures"]] == ["FAIL"]
    assert [r["Symbol"] for r in got["other"]] == ["SIDE"]


def test_step_back_between_events_is_not_an_upgrade():
    prev = _bo([("BACK", "BREAKOUT_TRIGGERED", 60), ("FWD", "BREAKOUT_TRIGGERED", 60)])
    now = _bo([("BACK", "PRE_BREAKOUT", 55), ("FWD", "BREAKOUT_CONFIRMED", 70)])
    got = insights.status_changes(now, prev)
    assert [r["Symbol"] for r in got["upgrades"]] == ["FWD"]
    assert [r["Symbol"] for r in got["other"]] == ["BACK"]


def test_status_changes_without_status_column_is_empty():
    assert insights.status_changes(pd.DataFrame({"Symbol": ["A"]}), _bo([])) == {
        "upgrades": [],
        "failures": [],
        "other": [],
    }


def test_cobra_changes_new_dropped_kept():
    now = pd.DataFrame([{"Symbol": "KEEP", "Score": 80}, {"Symbol": "NEW", "Score": 90}])
    prev = pd.DataFrame([{"Symbol": "KEEP", "Score": 75}, {"Symbol": "OLD", "Score": 70}])
    got = insights.cobra_changes(now, prev)
    assert [r["Symbol"] for r in got["new"]] == ["NEW"]
    assert [r["Symbol"] for r in got["dropped"]] == ["OLD"]
    assert got["kept"][0]["Symbol"] == "KEEP" and got["kept"][0]["Score Change"] == 5


def test_cobra_changes_when_one_run_had_no_cobra_csv():
    got = insights.cobra_changes(pd.DataFrame(), pd.DataFrame([{"Symbol": "OLD", "Score": 70}]))
    assert [r["Symbol"] for r in got["dropped"]] == ["OLD"] and got["new"] == []


def test_run_changes_compares_the_two_newest_runs(tmp_path):
    for day, status in (
        ("2026-10-28", "WATCH"),
        ("2026-10-29", "WATCH"),
        ("2026-10-30", "PRE_BREAKOUT"),
    ):
        _bo([("AAA", status, 50)]).to_csv(tmp_path / f"breakout_setups_{day}.csv", index=False)
    got = insights.run_changes(str(tmp_path))
    assert (got["prev"], got["now"]) == ("2026-10-29", "2026-10-30") and not got["cobra_ran"]
    assert [r["Symbol"] for r in got["status"]["upgrades"]] == ["AAA"]


def test_run_changes_needs_two_runs(tmp_path):
    _bo([("AAA", "WATCH", 50)]).to_csv(tmp_path / "breakout_setups_2026-10-30.csv", index=False)
    got = insights.run_changes(str(tmp_path))
    assert got["now"] == "2026-10-30" and got["prev"] is None
    assert got["status"]["upgrades"] == [] and got["cobra"]["new"] == []


def test_bias_changes_pairs_each_day_with_its_weekly_run(tmp_path):
    daily, weekly = tmp_path / "d", tmp_path / "w"
    daily.mkdir()
    weekly.mkdir()
    row = {"Symbol": "SPY", "Close": 110.0, "SMA50": 100.0}
    for day in ("2026-10-29", "2026-10-30"):
        pd.DataFrame([row]).to_csv(daily / f"breakout_setups_{day}.csv", index=False)
    # Only the 10-30 daily run has a weekly run on or before it that is bullish.
    pd.DataFrame([{"Symbol": "SPY", "Trend": "BULLISH", "MTF": "ALIGNED"}]).to_csv(
        weekly / "breakout_setups_2026-10-30.csv", index=False
    )
    got = {b["symbol"]: b for b in insights.bias_changes(str(daily), str(weekly))}
    assert got["SPY"] == {"symbol": "SPY", "bias": "LONG", "prev": "NEUTRAL", "changed": True}
    assert got["SCHD"]["changed"] is False
