"""Daily ingest: today's bar is final only after the close (ET); NYSE session calendar."""

from __future__ import annotations

import sys
from datetime import date, datetime

import pandas as pd
import pytest

from finance_vibe import config, run_vibe
from finance_vibe import daily_ingest as di

ET = di.MARKET_TZ
DAY = pd.Timestamp("2026-09-16")  # a Wednesday session


@pytest.mark.parametrize(
    "now, complete",
    [
        (datetime(2026, 9, 16, 9, 0, tzinfo=ET), False),  # before the open
        (datetime(2026, 9, 16, 12, 0, tzinfo=ET), False),  # mid-session: partial bar
        (datetime(2026, 9, 16, 16, 30, tzinfo=ET), False),  # inside the publish buffer
        (datetime(2026, 9, 16, 17, 0, tzinfo=ET), True),
        (datetime(2026, 9, 16, 17, 30, tzinfo=ET), True),  # the scheduled 5:30 PM run
        (datetime(2026, 9, 17, 9, 0, tzinfo=ET), True),  # next morning
    ],
)
def test_daily_bar_completes_after_close(now, complete):
    assert di.daily_bar_is_complete(DAY, now=now) is complete


def test_utc_clock_is_converted_to_market_time():
    # 21:30 UTC is 17:30 EDT -> complete; 20:30 UTC is 16:30 EDT -> not yet.
    assert di.daily_bar_is_complete(
        DAY, now=pd.Timestamp("2026-09-16 21:30", tz="UTC").to_pydatetime()
    )
    assert not di.daily_bar_is_complete(
        DAY, now=pd.Timestamp("2026-09-16 20:30", tz="UTC").to_pydatetime()
    )


def test_tz_aware_and_string_bar_dates():
    bar = pd.Timestamp("2026-09-16 00:00", tz="America/New_York")
    assert not di.daily_bar_is_complete(bar, now=datetime(2026, 9, 16, 14, 0, tzinfo=ET))
    assert di.daily_bar_is_complete("2026-09-16", now=datetime(2026, 9, 16, 18, 0, tzinfo=ET))


# ---------------------------------------------------------------------------
# drop_incomplete_daily_bars
# ---------------------------------------------------------------------------


def _write_raw(path, last_day: str, rows: int):
    dates = pd.bdate_range(end=last_day, periods=rows)
    pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": 1.0,
            "High": 2.0,
            "Low": 0.5,
            "Close": 1.5,
            "Volume": 100,
        }
    ).to_csv(path, index=False)


def test_mid_session_run_drops_todays_bar_only(tmp_path):
    raw, logs = tmp_path / "raw", tmp_path / "logs"
    raw.mkdir()
    logs.mkdir()
    _write_raw(raw / "AAA_5y_1d.csv", "2026-09-16", 100)  # has today's partial bar
    _write_raw(raw / "BBB_5y_1d.csv", "2026-09-15", 100)  # already complete
    trimmed = di.drop_incomplete_daily_bars(
        str(raw), str(logs), now=datetime(2026, 9, 16, 12, 0, tzinfo=ET)
    )

    assert trimmed == 1
    a = pd.read_csv(raw / "AAA_5y_1d.csv")
    assert len(a) == 99 and a["Date"].iloc[-1] == "2026-09-15"
    assert len(pd.read_csv(raw / "BBB_5y_1d.csv")) == 100


def test_after_close_run_keeps_todays_bar(tmp_path):
    _write_raw(tmp_path / "AAA_5y_1d.csv", "2026-09-16", 100)
    assert (
        di.drop_incomplete_daily_bars(
            str(tmp_path), str(tmp_path), now=datetime(2026, 9, 16, 17, 30, tzinfo=ET)
        )
        == 0
    )
    assert len(pd.read_csv(tmp_path / "AAA_5y_1d.csv")) == 100


def test_file_below_min_rows_after_trim_is_removed_and_logged(tmp_path):
    raw, logs = tmp_path / "raw", tmp_path / "logs"
    raw.mkdir()
    logs.mkdir()
    _write_raw(raw / "NEW_5y_1d.csv", "2026-09-16", config.MIN_SAVE_ROWS)
    di.drop_incomplete_daily_bars(str(raw), str(logs), now=datetime(2026, 9, 16, 12, 0, tzinfo=ET))

    assert not (raw / "NEW_5y_1d.csv").exists()
    errors = pd.concat(pd.read_csv(p) for p in logs.glob("ingest_errors_*.csv"))
    assert errors["Ticker"].tolist() == ["NEW"]
    assert errors["Error"].iloc[0].startswith("insufficient_rows")


# ---------------------------------------------------------------------------
# NYSE session calendar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "now, expected",
    [
        (datetime(2026, 9, 16, 17, 30, tzinfo=ET), date(2026, 9, 16)),  # after close: today
        (datetime(2026, 9, 16, 12, 0, tzinfo=ET), date(2026, 9, 15)),  # mid-session: yesterday
        (datetime(2026, 9, 21, 9, 0, tzinfo=ET), date(2026, 9, 18)),  # Monday morning: Friday
        (datetime(2026, 9, 19, 12, 0, tzinfo=ET), date(2026, 9, 18)),  # Saturday: Friday
        (datetime(2026, 9, 7, 17, 30, tzinfo=ET), date(2026, 9, 4)),  # Labor Day: Friday before
        (datetime(2026, 4, 3, 17, 30, tzinfo=ET), date(2026, 4, 2)),  # Good Friday
        (datetime(2026, 6, 19, 17, 30, tzinfo=ET), date(2026, 6, 18)),  # Juneteenth
        (datetime(2026, 7, 3, 17, 30, tzinfo=ET), date(2026, 7, 2)),  # July 4 (Sat) observed Fri
        (
            datetime(2026, 11, 27, 17, 30, tzinfo=ET),
            date(2026, 11, 27),
        ),  # day after Thanksgiving: half day, open
        (datetime(2026, 11, 26, 17, 30, tzinfo=ET), date(2026, 11, 25)),  # Thanksgiving
        (datetime(2027, 1, 1, 17, 30, tzinfo=ET), date(2026, 12, 31)),  # New Year's Day
        (
            datetime(2027, 12, 31, 17, 30, tzinfo=ET),
            date(2027, 12, 31),
        ),  # Sat Jan 1 is not moved to Fri
    ],
)
def test_last_complete_session(now, expected):
    assert di.last_complete_session(now) == expected


# ---------------------------------------------------------------------------
# orchestrator routing
# ---------------------------------------------------------------------------


def _stage_cmds(monkeypatch, argv):
    cmds: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["run_vibe.py", *argv])
    monkeypatch.setattr(run_vibe.subprocess, "run", lambda cmd, **kw: cmds.append(cmd))
    monkeypatch.setattr(run_vibe, "clean_raw_folder", lambda *a, **k: None)
    run_vibe.run_workflow()
    return cmds


def test_daily_mode_runs_daily_ingest_without_a_mode_argument(monkeypatch):
    cmds = _stage_cmds(monkeypatch, ["--mode", "daily"])
    ingest = cmds[1]
    assert ingest[1].endswith("src/finance_vibe/daily_ingest.py") and len(ingest) == 2
    assert not any(c[1].endswith("data_ingestor.py") for c in cmds)


def test_weekly_mode_still_runs_data_ingestor(monkeypatch):
    cmds = _stage_cmds(monkeypatch, ["--mode", "weekly"])
    assert cmds[1][1].endswith("src/finance_vibe/data_ingestor.py") and cmds[1][2] == "weekly"
    assert not any(c[1].endswith("daily_ingest.py") for c in cmds)


def test_daily_reuse_raw_skips_daily_ingest(monkeypatch):
    cmds = _stage_cmds(monkeypatch, ["--mode", "daily", "--reuse-raw"])
    assert not any(c[1].endswith(("daily_ingest.py", "data_ingestor.py")) for c in cmds)
