"""Finance-Vibe pipeline path resolution tests."""

import os
from pathlib import Path

from finance_vibe.trade_plan_helper import resolve_trade_plan_path


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
