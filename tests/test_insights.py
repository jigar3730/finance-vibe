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
