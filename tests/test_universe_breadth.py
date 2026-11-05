"""Breadth across the scanned tickers (data.universe_breadth) and the levels it sits beside."""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.data import market_snapshot, universe_breadth
from finance_vibe.scanners import breakout
from finance_vibe.web import health, insights


def _frame(closes, last="2026-10-30"):
    dates = pd.bdate_range(end=last, periods=len(closes))
    return pd.DataFrame({"Date": dates, "Close": [float(c) for c in closes]})


def test_summarize_counts_averages_and_extremes():
    up = list(range(100, 400))  # above both averages, at a new high
    down = list(range(400, 100, -1))  # below both, at a new low
    got = universe_breadth.summarize([_frame(up), _frame(down)])
    assert got["date"] == "2026-10-30" and got["previous_date"] == "2026-10-29"
    assert got["tickers"] == 2
    assert got["current"] == {
        "above_50": 1,
        "counted_50": 2,
        "above_200": 1,
        "counted_200": 2,
        "new_highs": 1,
        "new_lows": 1,
    }


def test_summarize_previous_session_excludes_the_newest_bar():
    # At its high the bar before, then a sharp fall through both averages.
    closes = [*range(100, 400), 50]
    got = universe_breadth.summarize([_frame(closes)])
    assert got["current"]["above_50"] == 0 and got["current"]["new_lows"] == 1
    assert got["previous"]["above_50"] == 1 and got["previous"]["new_highs"] == 1


def test_summarize_leaves_out_short_history_and_old_tickers():
    full = list(range(100, 400))
    got = universe_breadth.summarize(
        [_frame(full), _frame([10.0] * 60), _frame(full, last="2026-10-20")]
    )
    assert got["tickers"] == 2  # the ticker that stopped on 10-20 is not counted
    assert got["current"]["counted_50"] == 2
    assert got["current"]["counted_200"] == 1
    assert got["current"]["new_highs"] == 1 and got["current"]["new_lows"] == 0


def test_summarize_without_bars():
    assert universe_breadth.summarize([]) is None
    assert universe_breadth.summarize([pd.DataFrame({"Date": [], "Close": []})]) is None


def test_main_writes_the_file_and_keeps_it_without_raw_files(tmp_path, monkeypatch):
    monkeypatch.setitem(config.TIMEFRAME_PROFILES["daily"], "raw_dir", str(tmp_path / "raw"))
    os.makedirs(tmp_path / "raw")
    assert universe_breadth.main([]) == 0
    assert not os.path.exists(config.UNIVERSE_BREADTH_PATH)
    closes = [100.0 + i for i in range(60)]
    frame = _frame(closes).assign(Open=closes, High=closes, Low=closes, Volume=1)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    frame.to_csv(tmp_path / "raw" / "AAA_5y_1d.csv", index=False)
    (tmp_path / "raw" / "BAD_5y_1d.csv").write_text("not,a,valid,file\n1,2,3,4\n")
    assert universe_breadth.main([]) == 0
    with open(config.UNIVERSE_BREADTH_PATH) as handle:
        got = json.load(handle)
    assert got["tickers"] == 1 and got["current"]["above_50"] == 1


@pytest.mark.parametrize("content", ["not json", "{}", '{"date": "x"}', "[]"])
def test_scanned_breadth_survives_a_bad_file(content):
    os.makedirs(os.path.dirname(config.UNIVERSE_BREADTH_PATH))
    with open(config.UNIVERSE_BREADTH_PATH, "w") as handle:
        handle.write(content)
    assert (
        health.scanned_breadth(datetime(2026, 11, 2, 12, tzinfo=market_snapshot.MARKET_TZ)) is None
    )


@pytest.mark.parametrize(
    ("closes", "last", "live", "want"),
    [
        ([*([100.0] * 59), 120.0], 90.0, True, "below"),
        ([100.0] * 60, 110.0, True, "above"),  # a close equal to its average is not above it
        ([100.0] * 59 + [90.0], 110.0, True, "above"),
        ([100.0] * 60, 100.0, False, None),
        ([*([100.0] * 59), 120.0], 120.0, False, "above"),
    ],
)
def test_crossed_ma(closes, last, live, want):
    assert insights.crossed_ma(pd.Series(closes), last, live, 50) == want


def test_crossed_ma_needs_enough_history():
    assert insights.crossed_ma(pd.Series([100.0] * 10), 90.0, True, 50) is None
    assert insights.crossed_ma(pd.Series([100.0]), 100.0, False, 50) is None


def test_levels_match_the_breakout_scan(tmp_path, monkeypatch):
    profile = dict(config.MARKET_CONTEXT_PROFILE) | {"raw_dir": str(tmp_path)}
    monkeypatch.setattr(config, "MARKET_CONTEXT_PROFILE", profile)
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 260))
    bars = pd.DataFrame(
        {
            "Date": pd.bdate_range(end="2026-10-30", periods=260).strftime("%Y-%m-%d"),
            "Open": close,
            "High": close + rng.uniform(0.1, 2, 260),
            "Low": close - rng.uniform(0.1, 2, 260),
            "Close": close,
            "Volume": 1_000_000.0,
        }
    )
    bars.to_csv(config.get_raw_path("SPY", profile), index=False)
    row = next(r for r in insights.levels_table() if r["symbol"] == "SPY")
    scan = breakout.add_indicators(breakout.normalize_ohlcv(bars), pctl_window=100).iloc[-1]
    assert row["levels"]["resistance"] == pytest.approx(scan["Resistance"])
    assert row["levels"]["support"] == pytest.approx(scan["Support"])
    assert row["levels"]["prior_high"] == pytest.approx(bars["High"].iloc[-1])
    assert row["levels"]["prior_low"] == pytest.approx(bars["Low"].iloc[-1])
    assert row["levels"]["sma50"] == pytest.approx(close[-50:].mean())
    assert row["last"] == pytest.approx(close[-1])


def test_levels_table_without_data(tmp_path, monkeypatch):
    profile = dict(config.MARKET_CONTEXT_PROFILE) | {"raw_dir": str(tmp_path)}
    monkeypatch.setattr(config, "MARKET_CONTEXT_PROFILE", profile)
    monkeypatch.setitem(config.TIMEFRAME_PROFILES["daily"], "raw_dir", str(tmp_path / "none"))
    assert all(r["last"] is None and r["levels"] == {} for r in insights.levels_table())
