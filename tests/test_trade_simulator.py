"""Tests for trade_simulator's generic OHLC-bar trade simulation helpers."""

import pandas as pd

from finance_vibe.trade_simulator import (
    simulate_trade,
)


def _bars(rows):
    """Build an OHLC frame from (open, high, low, close) tuples with a leading pad bar."""
    rows = [(100.0, 100.0, 100.0, 100.0)] + rows
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=len(rows), freq="D"),
            "Open": [r[0] for r in rows],
            "High": [r[1] for r in rows],
            "Low": [r[2] for r in rows],
            "Close": [r[3] for r in rows],
        }
    )


def test_simulate_trade_long_stopped():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=4, freq="W"),
            "High": [110.0, 101.0, 99.0, 100.0],
            "Low": [100.0, 100.0, 95.0, 96.0],
            "Close": [105.0, 100.0, 97.0, 98.0],
        }
    )
    outcome, exit_date, exit_price, r_mult = simulate_trade(
        df,
        start_idx=1,
        is_long=True,
        entry=100.0,
        stop=96.0,
        target1=104.0,
        target2=108.0,
        entry_valid_bars=2,
        max_hold_bars=4,
    )
    assert outcome == "stopped"
    assert exit_price == 96.0
    assert r_mult == -1.0
    assert exit_date is not None


def test_simulate_trade_long_target1():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=4, freq="W"),
            "High": [110.0, 101.0, 105.0, 106.0],
            "Low": [100.0, 99.0, 100.0, 101.0],
            "Close": [105.0, 100.0, 103.0, 104.0],
        }
    )
    outcome, _, exit_price, r_mult = simulate_trade(
        df,
        start_idx=1,
        is_long=True,
        entry=100.0,
        stop=96.0,
        target1=104.0,
        target2=108.0,
        entry_valid_bars=2,
        max_hold_bars=4,
    )
    assert outcome == "target1"
    assert exit_price == 104.0
    assert r_mult == 1.0


def test_simulate_trade_short_r_multiple():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=4, freq="W"),
            "High": [100.0, 101.0, 102.0, 103.0],
            "Low": [90.0, 99.0, 94.0, 95.0],
            "Close": [95.0, 100.0, 96.0, 97.0],
        }
    )
    outcome, _, exit_price, r_mult = simulate_trade(
        df,
        start_idx=1,
        is_long=False,
        entry=100.0,
        stop=104.0,
        target1=96.0,
        target2=92.0,
        entry_valid_bars=2,
        max_hold_bars=4,
    )
    assert outcome == "target1"
    assert exit_price == 96.0
    assert r_mult == 1.0
