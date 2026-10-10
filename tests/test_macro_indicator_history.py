"""Phase 4.5: one-pass indicator history must equal per-prefix recomputation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_vibe import breakout_scanner as bs
from finance_vibe import coiled_cobra as cc
from finance_vibe import indicators


def _bars(n: int = 220, flat_at: tuple[int, ...] = (150,), seed: int = 3) -> pd.DataFrame:
    # Low-priced, range-dominated bars (like the real tickers where this matters):
    # High-Low is the true-range maximum and < 1, so pandas_ta's +epsilon
    # survives rounding and changes ATR.
    rng = np.random.default_rng(seed)
    close = 0.8 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    spread = (0.05 + np.abs(rng.normal(0, 0.02, n))) * close
    high, low = close + spread, close - spread
    for i in flat_at:
        high[i] = low[i] = close[i]
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-06", periods=n, freq="W-MON"),
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(1_000_000, 9_000_000, n).astype("float64"),
        }
    )


def _assert_rows_equal(got: pd.Series, want: pd.Series) -> None:
    for col in want.index:
        a, b = want[col], got[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, f"{col}: {a!r} != {b!r}"  # exact, not approximate


@pytest.mark.parametrize("flat_at", [(150,), (), (60, 61)])
def test_history_rows_equal_prefix_recomputation(flat_at):
    df = _bars(flat_at=flat_at)
    hist = cc.macro_indicator_history(df)
    for t in range(50, len(df)):  # past the longest SMA window (indicators.sma edge row)
        _assert_rows_equal(hist.iloc[t], cc.add_macro_indicators(df.iloc[: t + 1]).iloc[-1])


def test_naive_full_history_leaks_a_later_flat_bar_into_atr():
    df = _bars(flat_at=(200,))
    naive = cc.add_macro_indicators(df)["ATR"]
    hist = cc.macro_indicator_history(df)["ATR"]
    prefix = cc.add_macro_indicators(df.iloc[:150])["ATR"].iloc[-1]
    assert hist.iloc[149] == prefix
    assert naive.iloc[149] != prefix  # the epsilon from bar 200 leaked backwards
    assert (hist.iloc[200:] == naive.iloc[200:]).all()


def test_atr_flat_epsilon_switch():
    df = _bars(flat_at=(100,))
    h, lo, c = df["High"], df["Low"], df["Close"]
    auto = indicators.atr(h, lo, c, 14)
    assert auto is not None
    assert auto.equals(indicators.atr(h, lo, c, 14, flat_epsilon=True))
    off = indicators.atr(h, lo, c, 14, flat_epsilon=False)
    assert off is not None and not auto.equals(off)


def test_display_mask_matches_row_predicate():
    df = pd.DataFrame(
        {
            "Status": [
                bs.STATUS_PRE,
                bs.STATUS_WATCH,
                bs.STATUS_WATCH,
                bs.STATUS_DEV,
                bs.STATUS_DEV,
                "OTHER",
                None,
                bs.STATUS_CONFIRMED,
            ],
            "Breakout Readiness": [10, 55, 54.9, 50, np.nan, 99, 99, None],
        }
    )
    want = [bs._is_display_candidate(r) for r in df.to_dict("records")]
    assert bs._display_mask(df).tolist() == want
