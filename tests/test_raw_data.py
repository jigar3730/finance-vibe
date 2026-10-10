from __future__ import annotations

import pandas as pd
import pytest

from finance_vibe import raw_data

HEADER = "Date,Open,High,Low,Close,Volume\n"


def _write(tmp_path, name, body, header=HEADER):
    path = tmp_path / name
    path.write_text(header + body)
    return str(path)


def test_raw_files_sorted_with_tickers(tmp_path):
    for name in ("MSFT_10y_1wk.csv", "aapl_10y_1wk.csv", "notes.txt"):
        (tmp_path / name).write_text(HEADER)
    files = raw_data.raw_files(str(tmp_path))
    assert [f.ticker for f in files] == ["MSFT", "AAPL"]  # name order: uppercase sorts first
    assert all(f.path.startswith(str(tmp_path)) for f in files)


def test_raw_files_missing_dir_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        raw_data.raw_files(str(tmp_path / "nope"))


def test_load_raw_sorts_dedupes_and_parses_dates(tmp_path):
    path = _write(
        tmp_path,
        "T_5y_1d.csv",
        "2026-10-06,2,2,2,2,20\n"
        "2026-10-05,1,1,1,1,10\n"
        "2026-10-06,3,3,3,3,30\n"  # duplicate date: last one wins
        "2026-10-07,x,4,4,4,40\n",  # non-numeric Open: row dropped
    )
    df = raw_data.load_raw(path)
    assert list(df["Date"]) == [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06")]
    assert list(df["Close"]) == [1.0, 3.0]
    assert df["Date"].dt.tz is None
    assert df["Close"].dtype == "float64"


def test_load_raw_normalizes_tz_aware_dates(tmp_path):
    path = _write(tmp_path, "T_5y_1d.csv", "2026-10-05 00:00:00+00:00,1,1,1,1,10\n")
    assert raw_data.load_raw(path)["Date"].iloc[0] == pd.Timestamp("2026-10-05")


def test_load_raw_rejects_contract_violations(tmp_path):
    path = _write(
        tmp_path, "T_5y_1d.csv", "2026-10-05,1,1,1,1\n", header="Date,Open,High,Low,Close\n"
    )
    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        raw_data.load_raw(path)
    empty = tmp_path / "E_5y_1d.csv"
    empty.write_text("")
    with pytest.raises(ValueError):
        raw_data.load_raw(str(empty))
    with pytest.raises(OSError):
        raw_data.load_raw(str(tmp_path / "missing.csv"))


def test_load_raw_weekly_as_of_excludes_week_in_progress(tmp_path):
    # Monday-dated weekly bars; the 2026-10-05 week ends Friday 2026-10-09.
    path = _write(tmp_path, "T_10y_1wk.csv", "2026-09-28,1,1,1,1,10\n2026-10-05,2,2,2,2,20\n")
    mid_week = raw_data.load_raw(path, as_of="2026-10-07", weekly=True)
    assert list(mid_week["Close"]) == [1.0]
    friday = raw_data.load_raw(path, as_of="2026-10-09", weekly=True)
    assert list(friday["Close"]) == [1.0, 2.0]


def test_load_raw_daily_as_of_keeps_that_day(tmp_path):
    path = _write(tmp_path, "T_5y_1d.csv", "2026-10-05,1,1,1,1,10\n2026-10-06,2,2,2,2,20\n")
    assert list(raw_data.load_raw(path, as_of="2026-10-05", weekly=False)["Close"]) == [1.0]


def test_load_raw_as_of_requires_timeframe(tmp_path):
    path = _write(tmp_path, "T_5y_1d.csv", "2026-10-05,1,1,1,1,10\n")
    with pytest.raises(ValueError, match="weekly"):
        raw_data.load_raw(path, as_of="2026-10-05")
