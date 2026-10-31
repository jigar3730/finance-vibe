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

    def write(
        symbol: str,
        mode: str,
        last: date,
        *,
        bad_last: bool = False,
        periods: int = 30,
        growth: bool = False,
    ) -> None:
        freq = "W-MON" if mode == "weekly" else "B"
        dates = pd.date_range(end=pd.Timestamp(last), periods=periods, freq=freq)
        close = np.linspace(30.0, 35.0 if not growth else 60.0, len(dates))
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


# ---------------------------------------------------------------------------
# ETF relative view and growth-vs-dividend rotation
# ---------------------------------------------------------------------------


def _closes(values) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {"Date": pd.bdate_range(end="2026-10-30", periods=len(values)), "Close": values}
    )


FLAT = _closes(np.full(150, 10.0))


def test_rotation_growth_leading():
    got = insights.rotation(_closes(np.linspace(10, 20, 150)), FLAT)
    assert got["label"] == insights.GROWTH_LEADING
    assert got["ratio"] > got["ma"] and got["spread"] > 0 and got["as_of"] == "2026-10-30"


def test_rotation_dividend_leading_is_the_mirror():
    got = insights.rotation(FLAT, _closes(np.linspace(10, 20, 150)))
    assert got["label"] == insights.DIVIDEND_LEADING
    assert got["ratio"] < got["ma"] and got["spread"] < 0


def test_rotation_is_mixed_when_ratio_and_spread_disagree():
    # A long slide in the ratio, then a bounce: above its 50-day SMA, still down over 63 days.
    ratio = np.concatenate([np.linspace(2.0, 1.0, 140), np.linspace(1.02, 1.2, 10)])
    got = insights.rotation(_closes(ratio * 10), FLAT)
    assert got["ratio"] > got["ma"] and got["spread"] < 0
    assert got["label"] == insights.MIXED


def test_rotation_without_enough_history_is_mixed():
    got = insights.rotation(_closes(np.linspace(10, 20, 40)), _closes(np.full(40, 10.0)))
    assert got["label"] == insights.MIXED and got["ratio"] is None
    assert insights.rotation(None, FLAT)["label"] == insights.MIXED


def test_rotation_aligns_on_dates():
    growth = _closes(np.linspace(10, 20, 150))
    dividend = FLAT.iloc[5:]  # starts later: only shared dates count
    got = insights.rotation(growth, dividend)
    assert got["label"] == insights.GROWTH_LEADING and got["ratio"] == pytest.approx(2.0)


def test_etf_relative_returns_and_rs(raw_silo):
    for sym in ("SPY", "QQQ", "SCHG", "SCHD"):
        raw_silo(sym, "daily", EXPECTED["daily"], periods=150, growth=sym == "SCHG")
    got = insights.etf_relative()
    rows = {r["symbol"]: r for r in got["rows"]}
    assert list(rows) == ["SPY", "QQQ", "SCHG", "SCHD"] and got["windows"] == [
        "1W",
        "1M",
        "3M",
        "6M",
    ]
    assert rows["SPY"]["rs"] is None and rows["SPY"]["as_of"] == "2026-10-30"
    assert rows["SCHG"]["rs"] > 0 and rows["SCHG"]["rs_ok"]
    assert rows["SCHD"]["returns"]["6M"] == pytest.approx(rows["SPY"]["returns"]["6M"])
    assert rows["SCHG"]["returns"]["6M"] is not None and rows["SCHG"]["returns"]["1W"] > 0
    assert got["rotation"]["label"] == insights.GROWTH_LEADING


def test_etf_relative_without_raw_files(raw_silo):
    got = insights.etf_relative()
    assert all(r["returns"] == {} and r["rs"] is None for r in got["rows"])
    assert got["rotation"]["label"] == insights.MIXED


# ---------------------------------------------------------------------------
# Market strip
# ---------------------------------------------------------------------------


def _strip_row(symbol, **kw):
    row = {"Symbol": symbol, "Close": 100.0, "SMA50": 95.0, "SMA200": 105.0}
    row |= {"Support": 98.0, "RVOL20": 1.0}
    return row | kw


def test_market_strip_facts(raw_silo):
    for sym in ("SPY", "QQQ", "SCHG", "SCHD"):
        raw_silo(sym, "daily", EXPECTED["daily"], periods=150, growth=sym == "SCHG")
    daily = pd.DataFrame(
        [
            _strip_row("SPY"),
            _strip_row("QQQ", Close=97.0, RVOL20=config.BREAKOUT_RVOL_CONFIRM),
            _strip_row("SCHG", Close=97.0, RVOL20=1.49),  # below support, light volume
        ]
    )
    got = insights.market_strip(daily)
    rows = {r["symbol"]: r for r in got["rows"]}
    assert list(rows) == ["SPY", "QQQ", "SCHG", "SCHD"]
    closes = insights.daily_closes("SPY")["Close"]
    assert rows["SPY"]["change"] == pytest.approx(closes.iloc[-1] / closes.iloc[-2] - 1)
    assert rows["SPY"]["as_of"] == "2026-10-30"
    assert rows["SPY"]["above_sma50"] is True and rows["SPY"]["above_sma200"] is False
    assert not rows["SPY"]["heavy_break"]
    assert rows["QQQ"]["heavy_break"] and not rows["SCHG"]["heavy_break"]
    # No scan row: the % change still shows, the scan facts are unknown.
    assert rows["SCHD"]["change"] is not None and rows["SCHD"]["above_sma50"] is None
    assert not rows["SCHD"]["heavy_break"]
    assert got["rotation"] == insights.GROWTH_LEADING


def test_market_strip_without_data(raw_silo):
    got = insights.market_strip(pd.DataFrame())
    assert all(
        r["change"] is None and r["above_sma50"] is None and not r["heavy_break"]
        for r in got["rows"]
    )
    assert got["rotation"] == insights.MIXED
