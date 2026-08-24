import pandas as pd
import pytest

from finance_vibe import trade_planner
from finance_vibe.coiled_cobra_backtest import (
    backtest_ticker,
    forward_horizon_bars,
    rel_forward_at,
    stamp_episode,
)


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


def test_stamp_episode_starts_and_ages():
    row = {"Score": 80}
    first, valid, age = stamp_episode(row, False, 0)
    assert first["Is_New_Coil"] is True
    assert first["Coil_Age_Bars"] == 1
    assert valid is True and age == 1

    second, valid, age = stamp_episode(row, valid, age)
    assert second["Is_New_Coil"] is False
    assert second["Coil_Age_Bars"] == 2

    none, valid, age = stamp_episode(None, valid, age)
    assert none is None
    assert valid is False and age == 0


def test_rel_forward_subtracts_qqq_on_or_before_date():
    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    stock = pd.DataFrame({"Date": dates, "Close": [100.0] * 8 + [110.0, 110.0]})
    flat_qqq = pd.DataFrame({"Date": dates, "Close": [50.0] * 10})
    up_qqq = pd.DataFrame({"Date": dates, "Close": [50.0] * 8 + [55.0, 55.0]})

    assert rel_forward_at(stock, flat_qqq, 7, 2, 100.0) == 0.10
    assert rel_forward_at(stock, up_qqq, 7, 2, 100.0) == 0.0
    assert rel_forward_at(stock, None, 7, 2, 100.0) is None


def test_forward_horizon_bars_are_calendar_equivalent():
    assert forward_horizon_bars("weekly") == (2, 4, 5, 8, 13, 26)
    assert forward_horizon_bars("daily") == (10, 21, 25, 42, 63, 126)
    from finance_vibe.coiled_cobra_backtest import (
        forward_label_specs,
        held_coil_low_at,
        mae_at,
        max_return_at,
        future_max_return_series,
    )

    daily_suffixes = [s for _, s in forward_label_specs("daily")]
    assert "10d" in daily_suffixes and "21d" in daily_suffixes and "42d" in daily_suffixes
    weekly_suffixes = [s for _, s in forward_label_specs("weekly")]
    assert "4w" in weekly_suffixes and "8w" in weekly_suffixes

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "Date": dates,
        "Close": [100.0] * 10,
        "High": [100.0, 100.0, 115.0, 110.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "Low": [100.0, 100.0, 90.0, 95.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    })
    assert mae_at(df, 0, 3, 100.0) == pytest.approx(0.10)
    assert max_return_at(df, 0, 3, 100.0) == pytest.approx(0.15)
    assert max_return_at(df, 0, 20, 100.0) is None
    assert mae_at(df, 0, 20, 100.0) is None
    mfe = future_max_return_series(df, 3)
    assert mfe.iloc[0] == pytest.approx(0.15)
    df["Close"] = [100.0, 101.0, 102.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    assert held_coil_low_at(df, 0, 3, 100.0) == 0
    assert held_coil_low_at(df, 0, 2, 100.0) == 1
    assert held_coil_low_at(df, 0, 20, 100.0) is None


def _fake_setup(symbol: str) -> dict:
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
        "Volume_Shelf": 16.0,
        "MACD_Compression": 18.5,
        "Structure": 17.0,
        "RS_Score": 13.5,
        "Coil_Width": 12.0,
        "Proximity_Highs": 8.0,
        "MACD_Cross": 0.0,
        "Fib_Bonus": 1.2,
        "Coil_Low": 94.0,
    }


def test_coiled_cobra_backtest_ticker_records_expansion(tmp_path, monkeypatch):
    from finance_vibe import coiled_cobra as cobra
    from finance_vibe import config

    cobra.configure_mode("daily")
    try:
        symbol = "TEST"
        path = tmp_path / f"{symbol}_{config.TIMEFRAME_PROFILES['daily']['period']}_{config.TIMEFRAME_PROFILES['daily']['interval']}.csv"

        dates = pd.date_range("2024-01-01", periods=180, freq="B")
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0] * 180,
            "High": [110.0] * 180,
            "Low": [90.0] * 180,
            "Close": [100.0] * 180,
            "Volume": [100000] * 180,
        })
        df.to_csv(path, index=False)

        def fake_load_ohlc_csv(file_path):
            assert str(file_path) == str(path)
            return pd.read_csv(file_path)

        # Daily LOOKBACK=252 → min_bars = 141; window length at idx is idx+1.
        def fake_detect_cobra_setup_at_bar(window, ticker, benchmark_df=None):
            if len(window) in (142, 143):
                return _fake_setup(symbol)
            return None

        import finance_vibe.coiled_cobra_backtest as module

        monkeypatch.setattr(module, "load_ohlc_csv", fake_load_ohlc_csv)
        monkeypatch.setattr(module, "detect_cobra_setup_at_bar", fake_detect_cobra_setup_at_bar)

        trades, counts = backtest_ticker(str(path), entry_valid=2, max_hold=4)

        assert counts["signals"] == 2
        assert counts["new_coils"] == 1
        assert "filled" not in counts
        assert "Outcome" not in trades[0]
        assert trades[0]["Score"] == 85
        assert trades[0]["Is_New_Coil"] is True
        assert trades[0]["Coil_Age_Bars"] == 1
        assert trades[0]["Forward_Return_2w"] == 0.0
        assert trades[0]["Forward_Return_10d"] == 0.0
        assert trades[0]["Max_Return_10d"] == pytest.approx(0.10)
        assert trades[0]["Hit_10Pct_10d"] == 1
        assert trades[0]["Hit_15Pct_10d"] == 0
        assert trades[0]["Win_10d"] == 0
        assert trades[0]["Hit_10Pct_21d"] == 1
        assert trades[0]["Hit_15Pct_21d"] == 0
        assert trades[0]["Hit_20Pct_21d"] == 0
        assert trades[0]["Hit_10Pct_42d"] is None
        assert trades[0]["Hit_25Pct_42d"] is None
        assert trades[0]["Hit_50Pct_42d"] is None
        assert "Rel_Forward_2w" in trades[0]
        assert "Rel_Forward_42d" in trades[0]
        assert "MAE_2w" in trades[0]
        assert "Held_Coil_Low_2w" in trades[0]
        assert trades[0]["MACD_Compression"] == 18.5

        assert trades[1]["Is_New_Coil"] is False
        assert trades[1]["Coil_Age_Bars"] == 2
    finally:
        cobra.configure_mode(config.DEFAULT_MODE)
