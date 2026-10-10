"""Stage entry points: no CLI parsing at import, explicit mode selection, main(argv)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from finance_vibe import analysis_engine as ae
from finance_vibe import breakout_scanner as bs
from finance_vibe import coiled_cobra as cc
from finance_vibe import config
from finance_vibe import trade_planner as tp


@pytest.fixture
def restore_modes():
    yield
    cc.apply_timeframe("weekly")
    bs.set_mode("weekly")
    tp.set_mode("weekly")


def test_import_ignores_sys_argv():
    code = (
        "import sys; sys.argv = ['x', 'daily']\n"
        "from finance_vibe import coiled_cobra as cc, breakout_scanner as bs, trade_planner as tp\n"
        "print(cc.mode, cc.LOOKBACK, bs.mode, tp.mode)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert out.stdout.split() == ["weekly", "52", "weekly", "weekly"]
    assert "Unknown mode" not in out.stderr


def test_timeframe_calibration():
    weekly, daily = cc.Timeframe.for_mode("weekly"), cc.Timeframe.for_mode("daily")
    assert (weekly.lookback, weekly.coil_bars, weekly.rs_lookback, weekly.wk_to_bar) == (
        52,
        8,
        13,
        1,
    )
    assert (daily.lookback, daily.coil_bars, daily.rs_lookback, daily.wk_to_bar) == (252, 30, 63, 5)
    assert cc.Timeframe.for_mode("bogus") == weekly
    with pytest.raises(AttributeError):  # frozen
        weekly.mode = "daily"  # type: ignore[misc]


def test_apply_timeframe_sets_constants_and_paths(restore_modes):
    assert cc.apply_timeframe("daily") == "daily"
    assert cc.TIMEFRAME.is_daily_bars and cc._is_daily_bars
    assert (cc.LOOKBACK, cc.TT_EMA_SLOW, cc.MIN_BARS_FULL_SCORE) == (252, 200, 800)
    assert cc.MIN_BARS_TO_EVALUATE == max(cc.COIL_BARS + 2, 300)
    assert cc.RAW_DATA_DIR.endswith(os.path.join("data", "raw", "daily"))
    assert cc.LOG_DIR == os.path.join(config.PROJECT_ROOT, "data", "logs", "daily")
    cc.apply_timeframe("weekly")
    assert (cc.LOOKBACK, cc.TT_EMA_SLOW, cc.MIN_BARS_FULL_SCORE, cc._is_daily_bars) == (
        52,
        40,
        160,
        False,
    )


def test_set_mode_paths(restore_modes):
    assert bs.set_mode("daily") == "daily"
    assert bs.RAW_DATA_DIR.endswith(os.path.join("data", "raw", "daily"))
    assert tp.set_mode("daily") == "daily"
    assert tp.SCANNER_DIR.parts[-2:] == ("logs", "daily")
    assert tp.set_mode("bogus") == "weekly"


@pytest.mark.parametrize("module", [cc, bs, tp, ae])
def test_main_rejects_bad_as_of(module, restore_modes):
    assert module.main(["weekly", "--as-of", "not-a-date"]) == 2
