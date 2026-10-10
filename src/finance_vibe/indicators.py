"""In-house versions of the 8 pandas_ta indicators the pipeline uses.

Each function reproduces ``pandas-ta==0.4.71b0`` (no TA-Lib) for the
arguments we call it with, quirks included, so it is a drop-in replacement:

* EMA is seeded with the SMA of the first ``length`` values ("presma").
* RSI and ATR smooth with Wilder's RMA (``ewm(alpha=1/length, adjust=False)``);
  ATR's RMA is SMA-seeded too.
* If any High == Low, the whole high-low range gets ``+ float epsilon``
  (pandas_ta's ``non_zero_range``); the same applies to Bollinger widths.
* OBV's first value is NaN (pandas_ta multiplies volume by a NaN initial sign).
* A series shorter than the indicator's window returns ``None``, not a Series.

``tests/test_indicators_parity.py`` checks these against pandas_ta itself.
Numeric drift here changes ``Score``: treat any change as a rubric change.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

_EPS = sys.float_info.epsilon


def _too_short(*series: pd.Series, length: int) -> bool:
    return any(s is None or s.size < length for s in series)


def _non_zero_range(x: pd.Series, y: pd.Series) -> pd.Series:
    diff = x - y
    if diff.eq(0).any():
        diff += _EPS
    return diff


def sma(close: pd.Series, length: int) -> pd.Series | None:
    """Simple moving average (pandas_ta computes it as a convolution)."""
    if _too_short(close, length=length):
        return None
    values = np.convolve(np.ones(length) / length, close.to_numpy(dtype="float64"))
    out = np.append(np.full(length - 1, np.nan), values[length - 1 : 1 - length])
    return pd.Series(out, index=close.index, name=f"SMA_{length}")


def ema(close: pd.Series, length: int) -> pd.Series | None:
    """EMA seeded with the SMA of the first ``length`` values."""
    if _too_short(close, length=length):
        return None
    seeded = close.copy()
    seed = seeded.iloc[0:length].mean()
    seeded.iloc[: length - 1] = np.nan
    seeded.iloc[length - 1] = seed
    out = seeded.ewm(span=length, adjust=False).mean()
    out.name = f"EMA_{length}"
    return out


def rma(close: pd.Series, length: int) -> pd.Series | None:
    """Wilder's moving average."""
    if _too_short(close, length=length):
        return None
    return close.ewm(alpha=1.0 / length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series | None:
    if _too_short(close, length=length + 1):
        return None
    negative = close.diff(1)
    positive = negative.copy()
    positive[positive < 0] = 0
    negative[negative > 0] = 0
    pos_avg = rma(positive, length)
    neg_avg = rma(negative, length)
    assert pos_avg is not None and neg_avg is not None  # length already checked
    out = 100 * pos_avg / (pos_avg + neg_avg.abs())
    out.name = f"RSI_{length}"
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series | None:
    if _too_short(high, low, close, length=1):
        return None
    prev_close = close.shift(1)
    ranges = pd.concat([_non_zero_range(high, low), high - prev_close, prev_close - low], axis=1)
    out = ranges.abs().max(axis=1)
    if out.isna().all():
        return None
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series | None:
    """Average true range: SMA-seeded RMA of the true range."""
    if _too_short(high, low, close, length=length + 1):
        return None
    tr = true_range(high, low, close)
    if tr is None:
        return None
    seed = tr.iloc[0:length].mean()
    tr.iloc[: length - 1] = np.nan
    tr.iloc[length - 1] = seed
    out = rma(tr, length)
    if out is None or out.isna().all():
        return None
    out.name = f"ATRr_{length}"
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame | None:
    """MACD line, histogram and signal (columns ``MACD_*``, ``MACDh_*``, ``MACDs_*``)."""
    if slow < fast:
        fast, slow = slow, fast
    if _too_short(close, length=slow + signal - 1):
        return None
    fast_ma, slow_ma = ema(close, fast), ema(close, slow)
    assert fast_ma is not None and slow_ma is not None
    line = fast_ma - slow_ma
    signal_ma = ema(line.loc[line.first_valid_index() :], signal)
    assert signal_ma is not None
    hist = line - signal_ma
    props = f"_{fast}_{slow}_{signal}"
    return pd.DataFrame(
        {f"MACD{props}": line, f"MACDh{props}": hist, f"MACDs{props}": signal_ma},
        index=close.index,
    )


def bbands(close: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.DataFrame | None:
    """Bollinger bands on an SMA with sample (ddof=1) standard deviation.

    Columns ``BBL_*``, ``BBM_*``, ``BBU_*``, ``BBB_*`` (bandwidth %), ``BBP_*`` (%B).
    """
    if _too_short(close, length=length):
        return None
    std = close.rolling(length, min_periods=length).var(1).apply(np.sqrt)
    mid = sma(close, length)
    assert mid is not None
    lower = mid - num_std * std
    upper = mid + num_std * std
    width = _non_zero_range(upper, lower)
    props = f"_{length}_{float(num_std)}_{float(num_std)}"
    return pd.DataFrame(
        {
            f"BBL{props}": lower,
            f"BBM{props}": mid,
            f"BBU{props}": upper,
            f"BBB{props}": 100 * width / mid,
            f"BBP{props}": _non_zero_range(close, lower) / width,
        },
        index=close.index,
    )


def kc(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, scalar: float = 2.0
) -> pd.DataFrame | None:
    """Keltner channels: EMA basis +/- scalar * EMA of the true range."""
    if _too_short(high, low, close, length=length + 1):
        return None
    tr = true_range(high, low, close)
    basis = ema(close, length)
    band = ema(tr, length) if tr is not None else None
    if basis is None or band is None:
        return None
    props = f"e_{length}_{float(scalar)}"
    return pd.DataFrame(
        {
            f"KCL{props}": basis - scalar * band,
            f"KCB{props}": basis,
            f"KCU{props}": basis + scalar * band,
        },
        index=close.index,
    )


def obv(close: pd.Series, volume: pd.Series) -> pd.Series | None:
    """On-balance volume (first value NaN, as in pandas_ta)."""
    if _too_short(close, volume, length=1):
        return None
    sign = close.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = np.nan
    out = (sign * volume).cumsum()
    out.name = "OBV"
    return out
