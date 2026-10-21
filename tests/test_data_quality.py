"""Data-quality contract tests: OHLCV validation, setup schema, level math, R:R."""

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config

# ---------------------------------------------------------------------------
# config.validate_and_clean_ohlcv
# ---------------------------------------------------------------------------


def _ohlcv(n=5):
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=n, freq="W"),
            "Open": np.linspace(10, 14, n),
            "High": np.linspace(11, 15, n),
            "Low": np.linspace(9, 13, n),
            "Close": np.linspace(10.5, 14.5, n),
            "Volume": np.linspace(1000, 1400, n),
        }
    )


def test_validate_ohlcv_happy_path():
    cleaned = config.validate_and_clean_ohlcv(_ohlcv())
    assert list(config.REQUIRED_OHLCV) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in config.REQUIRED_OHLCV:
        assert col in cleaned.columns


def test_validate_ohlcv_missing_volume_raises():
    df = _ohlcv().drop(columns=["Volume"])
    with pytest.raises(ValueError):
        config.validate_and_clean_ohlcv(df, require_volume=True)


def test_validate_ohlcv_missing_volume_allowed_when_optional():
    df = _ohlcv().drop(columns=["Volume"])
    cleaned = config.validate_and_clean_ohlcv(df, require_volume=False)
    assert "Close" in cleaned.columns


def test_validate_ohlcv_drops_nan_rows_and_coerces():
    df = _ohlcv()
    # Mirror a CSV with a stray non-numeric cell: the column loads as object.
    df["Close"] = df["Close"].astype(object)
    df.loc[2, "Close"] = "not_a_number"
    cleaned = config.validate_and_clean_ohlcv(df)
    assert len(cleaned) == len(df) - 1
    assert pd.api.types.is_numeric_dtype(cleaned["Close"])


def test_validate_ohlcv_promotes_date_index():
    df = _ohlcv().set_index("Date")
    cleaned = config.validate_and_clean_ohlcv(df)
    assert "Date" in cleaned.columns


# ---------------------------------------------------------------------------
# setup schema
# ---------------------------------------------------------------------------


def test_blank_setup_row_matches_schema():
    row = config.blank_setup_row()
    assert set(row.keys()) == set(config.SETUP_ROW_COLUMNS)
    assert all(v is None for v in row.values())


def test_setup_schema_has_v3_attribution_and_no_ml_columns():
    assert "RVOL" in config.SETUP_ROW_COLUMNS
    assert "Market Gate" in config.SETUP_ROW_COLUMNS
    assert not [c for c in config.SETUP_ROW_COLUMNS if c.startswith(("ML_", "Pct_From_"))]


# ---------------------------------------------------------------------------
# benchmark: no-lookahead regime + relative strength
# ---------------------------------------------------------------------------


def _bench_frame(closes, start="2023-01-01"):
    from finance_vibe.market import _ema as ema

    df = pd.DataFrame(
        {
            "Date": pd.date_range(start, periods=len(closes), freq="D"),
            "Close": closes,
        }
    )
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA100"] = ema(df["Close"], 100)
    df["EMA50_rising"] = df["EMA50"] > df["EMA50"].shift(1)
    return df


def test_relative_strength_positive_and_negative():
    from finance_vibe.market import relative_strength

    n = 120
    bench = _bench_frame(list(np.linspace(100, 120, n)))
    dates = bench["Date"]
    # Strong stock: outpaces benchmark.
    strong = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 200, n)})
    ok, rel = relative_strength(strong, bench, as_of=dates.iloc[-1], lookback=63, ratio_ma_bars=20)
    assert ok is True and rel > 0
    # Weak stock: lags benchmark.
    weak = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 101, n)})
    ok2, _rel2 = relative_strength(weak, bench, as_of=dates.iloc[-1], lookback=63, ratio_ma_bars=20)
    assert ok2 is False


def test_relative_strength_no_lookahead_alignment():
    from finance_vibe.market import relative_strength

    n = 120
    bench = _bench_frame(list(np.linspace(100, 120, n)))
    dates = bench["Date"]
    stock = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 200, n)})
    # Insufficient history before the lookback window -> cannot assess (no peeking ahead).
    ok, rel = relative_strength(stock, bench, as_of=dates.iloc[30], lookback=63, ratio_ma_bars=20)
    assert ok is False and rel is None


# ---------------------------------------------------------------------------
# mode routing
# ---------------------------------------------------------------------------


def test_mode_routing():
    assert config.resolve_pipeline_mode("daily") == ("daily", "daily")
    assert config.resolve_pipeline_mode("weekly") == ("weekly", "weekly")
    # The removed high_beta mode no longer routes to daily data.
    assert config.resolve_pipeline_mode("high_beta") == (config.DEFAULT_MODE, config.DEFAULT_MODE)
