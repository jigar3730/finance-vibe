"""finance_vibe.indicators vs frozen pandas-ta 0.4.71b0 output.

The fixture comes from tests/data/make_indicator_reference.py. The in-house
indicators replaced pandas-ta in Phase 4.4 after matching it bit-for-bit on
all 549 raw files, except SMA of Volume (<= 5e-16 relative, BLAS summation
order). Golden output was identical at atol=0. A failure here means Score may
move: treat it as a rubric change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_vibe import indicators as ind

REF = pd.read_csv(
    Path(__file__).parent / "data" / "indicator_reference.csv", float_precision="round_trip"
)
C, H, L, V = REF["Close"], REF["High"], REF["Low"], REF["Volume"]


def _same(got: pd.Series | None, ref_col: str, rtol: float = 1e-12) -> None:
    assert got is not None
    want = REF[ref_col].to_numpy(dtype=float)
    got_v = got.to_numpy(dtype=float)
    assert np.array_equal(np.isnan(got_v), np.isnan(want)), f"{ref_col}: NaN layout differs"
    np.testing.assert_allclose(got_v, want, rtol=rtol, atol=0, equal_nan=True, err_msg=ref_col)


@pytest.mark.parametrize("n", [10, 20, 50, 100, 200])
def test_ema(n):
    _same(ind.ema(C, length=n), f"ema_{n}")


@pytest.mark.parametrize("n", [20, 50, 200])
def test_sma(n):
    _same(ind.sma(C, length=n), f"sma_{n}")


def test_sma_of_volume():
    _same(ind.sma(V, length=20), "sma_vol_20")


def test_rsi_atr_obv():
    _same(ind.rsi(C, length=14), "rsi_14")
    _same(ind.atr(H, L, C, length=14), "atr_14")  # fixture has flat bars: epsilon quirk
    _same(ind.obv(C, V), "obv")


@pytest.mark.parametrize(
    ("prefix", "frame"),
    [
        ("macd", lambda: ind.macd(C)),
        ("macd_15_30", lambda: ind.macd(C, fast=15, slow=30, signal=9)),
        ("bb", lambda: ind.bbands(C, length=20, num_std=2.0)),
        ("kc", lambda: ind.kc(H, L, C, length=20, scalar=1.5)),
    ],
)
def test_multi_column(prefix, frame):
    out = frame()
    expected = [c.split(":", 1)[1] for c in REF.columns if c.startswith(f"{prefix}:")]
    assert out is not None and list(out.columns) == expected  # callers select by name
    for col in expected:
        _same(out[col], f"{prefix}:{col}")


def test_short_series_return_none_like_pandas_ta():
    short = C.iloc[:10]
    assert ind.ema(short, length=20) is None
    assert ind.sma(short, length=20) is None
    assert ind.rsi(short, length=14) is None
    assert ind.atr(H.iloc[:10], L.iloc[:10], short, length=14) is None
    assert ind.macd(short) is None
    assert ind.bbands(short, length=20) is None
    assert ind.kc(H.iloc[:10], L.iloc[:10], short, length=20) is None
    assert ind.ema(C, length=20) is not None


def test_inputs_are_not_mutated():
    before = REF[["High", "Low", "Close", "Volume"]].copy()
    ind.ema(C, length=10)
    ind.atr(H, L, C, length=14)
    ind.obv(C, V)
    pd.testing.assert_frame_equal(REF[["High", "Low", "Close", "Volume"]], before)
