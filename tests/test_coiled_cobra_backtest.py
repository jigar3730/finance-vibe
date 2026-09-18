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

    def fake_detect_cobra_setup_at_bar(window, ticker, benchmark_df=None, spy_df=None):
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


def _bench_backtest(tmp_path, monkeypatch, bench_close):
    """Run backtest_ticker once with a fake setup carrying pillar parts."""
    symbol, n = "TEST", 70
    dates = pd.date_range("2024-01-01", periods=n, freq="W")
    price = pd.DataFrame({
        "Date": dates, "Open": 100.0, "High": [200.0 + i for i in range(n)],
        "Low": [50.0] * n, "Close": [100.0 + i for i in range(n)], "Volume": 100000,
    })
    path = tmp_path / f"{symbol}_10y_1wk.csv"
    price.to_csv(path, index=False)
    bench = (
        None if bench_close is None
        else pd.DataFrame({"Date": dates, "Close": bench_close, "EMA50": 190.0})
    )

    def fake_detect(window, ticker, benchmark_df=None, spy_df=None):
        if len(window) != 68:
            return None
        return {
            "Symbol": symbol, "Setup Type": "SETUP_LONG", "Close": 167.0, "EMA20": 165.0,
            "EMA50": 160.0, "ATR": 4.0, "Swing Low": 158.0, "Fib 78.6%": 163.0,
            "Score": 82, "Grade": "B - Valid Coil", "Tier": "Watchlist", "Checks Met": "5/6",
            "Source": "coiled_cobra", "Pct_From_EMA20": 0.01, "Pct_From_EMA50": 0.04,
            "Pct_From_Fib618": 0.02, "Pct_From_Fib786": 0.03, "ATR_Pct": 0.02,
            "RS 63d": 0.11, "BBWidth Pctile": 12.5,
            "Part_vol_contraction": 25, "Part_structure": 16, "Part_relative_strength": 18,
        }

    import finance_vibe.coiled_cobra_backtest as module
    monkeypatch.setattr(module, "load_ohlc_csv", lambda p: pd.read_csv(p, parse_dates=["Date"]))
    monkeypatch.setattr(module, "detect_cobra_setup_at_bar", fake_detect)
    trades, _ = backtest_ticker(str(path), entry_valid=2, max_hold=4, benchmark_df=bench)
    return trades[0]


def test_backtest_rows_carry_pillars_and_benchmark_relative_label(tmp_path, monkeypatch):
    bench_close = [200.0 + 2 * i for i in range(70)]
    t = _bench_backtest(tmp_path, monkeypatch, bench_close)

    assert (t["Tier"], t["Checks_N"], t["RS_63d"], t["BBWidth_Pctile"]) == ("Watchlist", 5, 0.11, 12.5)
    assert (t["Part_vol_contraction"], t["Part_structure"], t["Part_relative_strength"]) == (25, 16, 18)

    idx = 67
    stock_fwd = (100.0 + idx + 2) / (100.0 + idx) - 1
    bench_fwd = bench_close[idx + 2] / bench_close[idx] - 1
    assert t["Forward_Return_2w"] == pytest.approx(round(stock_fwd, 4))
    assert t["Excess_Return_2w"] == pytest.approx(round(round(stock_fwd, 4) - bench_fwd, 4), abs=1e-4)
    assert t["QQQ_Pct_From_EMA50"] == pytest.approx(bench_close[idx] / 190.0 - 1)
    assert t["QQQ_Ret_13w"] == pytest.approx(bench_close[idx] / bench_close[idx - 13] - 1)


def test_backtest_regime_features_are_causal_but_label_uses_future(tmp_path, monkeypatch):
    base = [200.0 + 2 * i for i in range(70)]
    scrambled = base[:68] + [900.0, 900.0]            # only bars AFTER the signal bar (idx 67)
    a = _bench_backtest(tmp_path, monkeypatch, base)
    b = _bench_backtest(tmp_path, monkeypatch, scrambled)

    # Features at the signal bar cannot see the future...
    assert a["QQQ_Pct_From_EMA50"] == b["QQQ_Pct_From_EMA50"]
    assert a["QQQ_Ret_13w"] == b["QQQ_Ret_13w"]
    # ...while the excess-return *label* legitimately does.
    assert a["Excess_Return_2w"] != b["Excess_Return_2w"]


def test_backtest_rows_without_benchmark_have_null_context(tmp_path, monkeypatch):
    t = _bench_backtest(tmp_path, monkeypatch, None)
    assert t["Excess_Return_2w"] is None
    assert t["QQQ_Pct_From_EMA50"] is None and t["QQQ_Ret_13w"] is None
    assert t["Forward_Return_2w"] is not None            # the plain label is unaffected
