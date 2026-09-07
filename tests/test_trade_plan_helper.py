"""Finance-Vibe pipeline path resolution tests."""

import os
from pathlib import Path

import pandas as pd

from finance_vibe.trade_plan_helper import (
    CLEAN_EXPORT_COLUMNS,
    main,
    process_trade_plan,
    resolve_trade_plan_path,
)


def test_resolve_trade_plan_path_finds_daily_subdirectory(tmp_path):
    unique_day = "2099-01-01"
    mode_dir = tmp_path / "data" / "logs" / "daily"
    mode_dir.mkdir(parents=True)
    plan = mode_dir / f"trade_plan_{unique_day}.csv"
    plan.write_text("Symbol,Stock Entry\nAAPL,100\n")

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        found_dir, found_path = resolve_trade_plan_path("daily", today=unique_day)
        resolved_path = found_path.resolve()
        resolved_dir = found_dir.resolve()
    finally:
        os.chdir(cwd)

    assert resolved_path == plan.resolve()
    assert resolved_dir.name == "daily"


def test_process_trade_plan_empty_zero_byte_file_exits_cleanly(tmp_path, monkeypatch):
    date_str = "2099-03-01"
    mode_dir = tmp_path / "data" / "logs" / "weekly"
    mode_dir.mkdir(parents=True)
    (mode_dir / f"trade_plan_{date_str}.csv").write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    out_path = process_trade_plan("weekly", today=date_str)
    assert out_path.name == f"trade_plan_clean_{date_str}.csv"
    assert out_path.exists()
    clean = pd.read_csv(out_path)
    assert clean.empty
    assert list(clean.columns) == CLEAN_EXPORT_COLUMNS


def test_process_trade_plan_header_only_file_exits_cleanly(tmp_path, monkeypatch):
    date_str = "2099-03-02"
    mode_dir = tmp_path / "data" / "logs" / "weekly"
    mode_dir.mkdir(parents=True)
    pd.DataFrame(columns=["Symbol", "Stock Entry", "Stock Stop"]).to_csv(
        mode_dir / f"trade_plan_{date_str}.csv", index=False
    )
    monkeypatch.chdir(tmp_path)

    out_path = process_trade_plan("weekly", today=date_str)
    clean = pd.read_csv(out_path)
    assert clean.empty


def test_helper_main_returns_zero_on_empty_plan(tmp_path, monkeypatch):
    from datetime import datetime as real_datetime

    date_str = "2099-03-03"
    mode_dir = tmp_path / "data" / "logs" / "daily"
    mode_dir.mkdir(parents=True)
    (mode_dir / f"trade_plan_{date_str}.csv").write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    class _FixedDate(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2099, 3, 3)

    monkeypatch.setattr("finance_vibe.trade_plan_helper.datetime", _FixedDate)
    assert main(["daily"]) == 0
    assert (mode_dir / f"trade_plan_clean_{date_str}.csv").exists()
