"""Breakout scanner labels: failed breakouts and multi-timeframe (MTF) agreement."""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from finance_vibe import breakout_scanner as bs

# ---------------------------------------------------------------------------
# Failed Breakout: judged against the level that was actually broken
# ---------------------------------------------------------------------------

def _bars(tail: list[tuple[float, float]]) -> pd.DataFrame:
    """30 bars ranging 9.0-10.0, then ``tail`` as (high, close) bars."""
    rows = [(10.0, 9.5)] * 30 + tail
    return pd.DataFrame({
        "Date": pd.bdate_range("2026-01-01", periods=len(rows)).strftime("%Y-%m-%d"),
        "Open": [c for _, c in rows],
        "High": [h for h, _ in rows],
        "Low": [min(9.0, c - 0.1) for _, c in rows],
        "Close": [c for _, c in rows],
        "Volume": 1000,
    })


def _last(tail):
    out = bs.add_indicators(bs.normalize_ohlcv(_bars(tail)), pctl_window=252)
    return out.iloc[-1]


def test_breakout_that_holds_above_the_broken_level_is_not_failed():
    # Breaks 10.0 on a 10.6 high; next day closes 10.3 -- under the new 20-bar
    # high (10.6) but still above the level it broke.
    last = _last([(10.6, 10.5), (10.4, 10.3)])
    assert not last["Breakout Triggered"]
    assert last["Breakout Level"] == pytest.approx(10.0)
    assert not last["Failed Breakout"]


def test_close_back_below_the_broken_level_is_failed():
    last = _last([(10.6, 10.5), (10.4, 9.8)])
    assert last["Failed Breakout"]


def test_multi_day_run_is_judged_against_the_level_that_started_it():
    # Day 2 extends the run over 10.6; day 3 falls under 10.6 but holds 10.0.
    held = _last([(10.6, 10.5), (10.9, 10.8), (10.5, 10.4)])
    assert held["Breakout Level"] == pytest.approx(10.0)
    assert not held["Failed Breakout"]
    lost = _last([(10.6, 10.5), (10.9, 10.8), (10.5, 9.9)])
    assert lost["Failed Breakout"]


def test_failure_window_expires_after_fakeout_lookback():
    # Closes below the broken level, but the breakout bar is now 6 bars back.
    tail = [(10.6, 10.5)] + [(10.4, 10.3)] * bs.FAKEOUT_LOOKBACK + [(10.0, 9.8)]
    assert not _last(tail)["Failed Breakout"]


# ---------------------------------------------------------------------------
# MTF label
# ---------------------------------------------------------------------------

_BASE = bs.BreakoutFeatures(**{
    f.name: None for f in dataclasses.fields(bs.BreakoutFeatures) if f.name != "has_daily"
}, has_daily=True)


def _feat(daily, weekly, monthly, *, bears=(False, False, False)):
    return dataclasses.replace(
        _BASE,
        daily_trend_bull=daily, weekly_trend_bull=weekly, monthly_trend_bull=monthly,
        daily_trend_bear=bears[0], weekly_trend_bear=bears[1], monthly_trend_bear=bears[2],
    )


@pytest.mark.parametrize("feat, label", [
    (_feat(True, True, True), "ALIGNED"),
    (_feat(True, False, True), "PARTIAL"),
    (_feat(True, False, False, bears=(False, True, False)), "DIVERGENT"),
    (_feat(False, False, False, bears=(True, True, False)), "BEARISH"),
    (_feat(False, False, False), "NEUTRAL"),
    (_feat(None, None, True), "INSUFFICIENT"),
])
def test_mtf_label(feat, label):
    assert bs._mtf_label(feat) == label


@pytest.mark.parametrize("label", ["DIVERGENT", "BEARISH", "NEUTRAL"])
def test_mtf_penalty_unchanged_for_names_formerly_labelled_divergent(label):
    states = {"MTF": label, "Trend": "NEUTRAL"}
    assert bs.score_readiness(_BASE, states)["Penalty"] == 10


@pytest.mark.parametrize("label", ["ALIGNED", "PARTIAL", "INSUFFICIENT"])
def test_no_mtf_penalty_otherwise(label):
    states = {"MTF": label, "Trend": "NEUTRAL"}
    assert bs.score_readiness(_BASE, states)["Penalty"] == 0
