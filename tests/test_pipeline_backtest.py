"""Tests for pipeline_backtest walk-forward simulation helpers."""

import pandas as pd

from finance_vibe.pipeline_backtest import passes_macro_gate, simulate_trade


def test_passes_macro_gate_long():
    assert passes_macro_gate("SETUP_LONG", 7, 7, -2) is True
    assert passes_macro_gate("SETUP_LONG", 5, 7, -2) is False


def test_passes_macro_gate_short():
    assert passes_macro_gate("SETUP_SHORT", -2, 7, -2) is True
    assert passes_macro_gate("SETUP_SHORT", 0, 7, -2) is False


def test_simulate_trade_long_stopped():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=4, freq="W"),
        "High": [110.0, 101.0, 99.0, 100.0],
        "Low": [100.0, 100.0, 95.0, 96.0],
        "Close": [105.0, 100.0, 97.0, 98.0],
    })
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
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=4, freq="W"),
        "High": [110.0, 101.0, 105.0, 106.0],
        "Low": [100.0, 99.0, 100.0, 101.0],
        "Close": [105.0, 100.0, 103.0, 104.0],
    })
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
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=4, freq="W"),
        "High": [100.0, 101.0, 102.0, 103.0],
        "Low": [90.0, 99.0, 94.0, 95.0],
        "Close": [95.0, 100.0, 96.0, 97.0],
    })
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
