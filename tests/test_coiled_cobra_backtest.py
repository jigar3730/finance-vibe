import os
from pathlib import Path

import pandas as pd
import pytest

from finance_vibe import trade_planner
from finance_vibe.coiled_cobra_backtest import backtest_ticker


def test_trade_planner_accepts_cobra_source():
    row = {
        "Source": "Cobra",
        "Setup Type": "SETUP_LONG",
        "Close": 100.0,
        "EMA20": 98.0,
        "EMA50": 95.0,
        "ATR": 4.0,
        "Fib 78.6%": 96.0,
        "Swing Low": 94.0,
    }

    entry, stop, target1, target2, option_type, delta_range = trade_planner.calculate_stock_levels(row)

    assert option_type == "CALL"
    assert stop < entry
    assert target1 > entry
    assert target2 > target1
    risk = entry - stop
    assert target1 == pytest.approx(entry + 2.0 * risk)
    assert target2 == pytest.approx(entry + 3.0 * risk)


def test_coiled_cobra_backtest_ticker_records_trade(tmp_path, monkeypatch):
    symbol = "TEST"
    path = tmp_path / f"{symbol}_10y_1wk.csv"

    dates = pd.date_range("2024-01-01", periods=70, freq="W")
    df = pd.DataFrame({
        "Date": dates,
        "Open": [100.0] * 70,
        "High": [110.0] * 70,
        "Low": [90.0] * 70,
        "Close": [100.0] * 70,
        "Volume": [100000] * 70,
    })
    df.to_csv(path, index=False)

    def fake_load_ohlc_csv(file_path):
        assert str(file_path) == str(path)
        return pd.read_csv(file_path)

    def fake_detect_cobra_setup_at_bar(window, ticker, benchmark_df=None):
        if len(window) == 68:
            return {
                "Symbol": symbol,
                "Setup Type": "SETUP_LONG",
                "Close": 100.0,
                "EMA20": 98.0,
                "EMA50": 95.0,
                "ATR": 4.0,
                "Swing Low": 94.0,
                "Fib 78.6%": 96.0,
                "Score": 85,
                "Grade": "A - Coil Ready",
                "Checks Met": "5/6",
                "Source": "coiled_cobra",
                "Pct_From_EMA20": 0.02,
                "Pct_From_EMA50": 0.05,
                "Pct_From_Fib618": 0.01,
                "Pct_From_Fib786": 0.04,
                "ATR_Pct": 0.04,
            }
        return None

    import finance_vibe.coiled_cobra_backtest as module

    monkeypatch.setattr(module, "load_ohlc_csv", fake_load_ohlc_csv)
    monkeypatch.setattr(module, "detect_cobra_setup_at_bar", fake_detect_cobra_setup_at_bar)

    trades, counts = backtest_ticker(str(path), entry_valid=2, max_hold=4)

    assert counts["signals"] == 1
    assert counts["filled"] >= 0
    assert isinstance(trades, list)
    assert trades[0]["Symbol"] == symbol
    assert trades[0]["Setup Type"] == "SETUP_LONG"
    assert trades[0]["Score"] == 85
