"""Bar-level quality flags (data.quality) and the expected-newest-bar helpers."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.data import ingest, quality

LAST = date(2026, 10, 30)


def _bars(n: int = 40, last: date = LAST) -> pd.DataFrame:
    close = np.linspace(50.0, 60.0, n)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range(end=pd.Timestamp(last), periods=n),
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        }
    )


def _checks(df: pd.DataFrame, expected: date = LAST) -> dict[str, dict[str, str]]:
    return {i["Check"]: i for i in quality.bar_issues(df, "TST", expected_last=expected)}


def test_clean_bars_have_no_issues():
    assert quality.bar_issues(_bars(), "TST", expected_last=LAST) == []


def test_stale_newest_bar_is_an_error():
    got = _checks(_bars(last=date(2026, 10, 28)))
    assert set(got) == {"stale"}
    assert got["stale"]["Severity"] == "error" and got["stale"]["Date"] == "2026-10-28"


def test_provisional_bar_with_high_below_open_is_incoherent():
    df = _bars()
    df.loc[df.index[-1], ["Open", "High"]] = [61.0, 60.5]  # like TGB / SCHW / BAX on 2026-10-30
    got = _checks(df)
    assert set(got) == {"ohlc_incoherent"}
    assert (
        got["ohlc_incoherent"]["Date"] == "2026-10-30"
        and got["ohlc_incoherent"]["Severity"] == "error"
    )


def test_float_noise_is_not_incoherent():
    df = _bars()
    df["High"] = df[["Open", "Close"]].max(axis=1) * (1 - 1e-9)
    assert _checks(df) == {}


def test_incoherent_bar_outside_the_recent_window_is_ignored():
    df = _bars()
    df.loc[0, "Low"] = df.loc[0, "Close"] + 1
    assert _checks(df) == {}


def test_non_positive_price_is_incoherent():
    df = _bars()
    df.loc[df.index[-3], ["Low"]] = 0.0
    assert "ohlc_incoherent" in _checks(df)


def test_zero_volume_is_a_warning():
    df = _bars()
    df.loc[df.index[-2], "Volume"] = 0
    got = _checks(df)
    assert set(got) == {"zero_volume"} and got["zero_volume"]["Severity"] == "warn"


def test_zero_volume_check_can_be_skipped_for_an_index():
    df = _bars()
    df["Volume"] = 0.0
    issues = quality.bar_issues(df, "VIX", expected_last=LAST, check_volume=False)
    assert "zero_volume" not in {i["Check"] for i in issues}


def test_flat_close_run_is_flagged_at_the_threshold():
    df = _bars()
    run = config.DQ_FLAT_RUN
    df.loc[df.index[-run:], ["Open", "High", "Low", "Close"]] = 55.0
    assert "flat_closes" in _checks(df)
    df.loc[df.index[-run], "Close"] = 54.9  # one bar shorter
    assert "flat_closes" not in _checks(df)


def test_big_move_is_a_warning():
    df = _bars()
    df.loc[df.index[-1], ["Open", "High", "Low", "Close"]] = [90.0, 92.0, 88.0, 90.0]
    got = _checks(df)
    assert set(got) == {"big_move"} and "+5" in got["big_move"]["Detail"]


def test_empty_frame_is_stale():
    assert _checks(_bars().iloc[:0])["stale"]["Detail"] == "no bars"


def _silo(tmp_path, monkeypatch, frames: dict[str, pd.DataFrame]):
    raw_dir, logs_dir = tmp_path / "raw", tmp_path / "logs"
    raw_dir.mkdir()
    for sym, df in frames.items():
        df.to_csv(raw_dir / f"{sym}_5y_1d.csv", index=False)
    monkeypatch.setitem(config.TIMEFRAME_PROFILES["daily"], "raw_dir", str(raw_dir))
    monkeypatch.setitem(config.TIMEFRAME_PROFILES["daily"], "logs_dir", str(logs_dir))
    return logs_dir


def test_write_report_is_header_only_when_clean(tmp_path, monkeypatch):
    logs = _silo(tmp_path, monkeypatch, {"AAA": _bars()})
    path = quality.write_report("daily", expected_last=LAST, now=datetime(2026, 10, 30, 18))
    assert path == str(logs / "data_quality_2026-10-30.csv")
    report = pd.read_csv(path)
    assert list(report.columns) == quality.REPORT_COLUMNS and report.empty


def test_write_report_rows_and_latest_report(tmp_path, monkeypatch, caplog):
    stale = _bars(last=date(2026, 10, 27))
    logs = _silo(tmp_path, monkeypatch, {"AAA": _bars(), "BBB": stale})
    caplog.set_level("WARNING")
    quality.write_report("daily", expected_last=LAST, now=datetime(2026, 10, 30, 18))
    assert "BBB: data quality stale" in caplog.text
    latest = quality.latest_report(str(logs))
    assert latest[["Symbol", "Check"]].to_numpy().tolist() == [["BBB", "stale"]]


def test_latest_report_without_a_report_is_empty(tmp_path):
    assert quality.latest_report(str(tmp_path / "none")).empty


@pytest.mark.parametrize(
    ("now", "want"),
    [
        (
            datetime(2026, 10, 30, 18, tzinfo=ingest.MARKET_TZ),
            date(2026, 10, 26),
        ),  # Fri after close
        (
            datetime(2026, 10, 30, 12, tzinfo=ingest.MARKET_TZ),
            date(2026, 10, 19),
        ),  # Fri mid-session
        (datetime(2026, 11, 1, 9, tzinfo=ingest.MARKET_TZ), date(2026, 10, 26)),  # Sunday
        (datetime(2026, 11, 2, 18), date(2026, 10, 26)),  # Monday, naive = ET
    ],
)
def test_last_complete_week(now, want):
    assert ingest.last_complete_week(now) == want
