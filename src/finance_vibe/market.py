"""Benchmark context for Coiled Cobra: benchmark frames, relative strength, market gate."""

from __future__ import annotations

import logging
import os

import pandas as pd

from finance_vibe import config, raw_data
from finance_vibe.config import AsOf

logger = logging.getLogger(__name__)


def _ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (adjust=False) for benchmark regime columns."""
    return s.ewm(span=span, adjust=False).mean()


def _select_benchmark_path(benchmark: str, data_mode: str) -> str | None:
    """Find the longest-history raw CSV for *benchmark* in the mode's raw dir."""
    cfg = config.get_mode_config(data_mode)
    raw_dir: str = cfg["raw_dir"]
    if not os.path.isdir(raw_dir):
        return None
    bench = benchmark.upper()

    def _rank(name: str) -> int:
        parts = name.replace(".csv", "").split("_")
        if len(parts) < 2:
            return 0
        tok = parts[1].lower()
        if tok.endswith("y") and tok[:-1].isdigit():
            return int(tok[:-1]) * 365
        if tok.endswith("mo") and tok[:-2].isdigit():
            return int(tok[:-2]) * 30
        if tok.endswith("d") and tok[:-1].isdigit():
            return int(tok[:-1])
        return 0

    candidates = [
        f
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv") and f.split("_")[0].upper() == bench
    ]
    if not candidates:
        return None
    best = max(candidates, key=_rank)
    return os.path.join(raw_dir, best)


def load_benchmark_frame(benchmark: str, data_mode: str) -> pd.DataFrame | None:
    """Load and enrich a benchmark OHLC frame for regime/RS checks.

    Returns a Date-sorted frame with causal EMA21/EMA50/EMA100, SMA50, and an
    ``EMA50_rising`` flag, or None when the benchmark CSV is unavailable.
    """
    path = _select_benchmark_path(benchmark, data_mode)
    if not path:
        return None
    try:
        df = raw_data.load_raw(path)
        if df.empty:
            raise ValueError("empty csv")
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            f"Benchmark {benchmark} unreadable at {path} ({exc}); treating as unavailable."
        )
        return None
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    df["EMA21"] = _ema(close, 21)
    df["EMA50"] = _ema(close, 50)
    df["EMA100"] = _ema(close, 100)
    df["SMA50"] = close.rolling(50, min_periods=50).mean()
    df["EMA50_rising"] = df["EMA50"] > df["EMA50"].shift(1)
    return df


def check_coiled_cobra_market_gate(
    *,
    close: float | None = None,
    ema50: float | None = None,
    rs_63d: float | None = None,
) -> bool:
    """Ticker-level trend gate for Coiled Cobra scorecard admission.

    Fails only on total trend failure:

    * Close more than 10% below EMA50 (broken trend, not a coil under the 50)
    * 63-day relative strength ≤ −15% (BA/DG-style lag)

    Overextension, Fib distance, low RVOL, slight RS noise, and SPY/QQQ chop do
    **not** fail the gate — those are scorecard deductions, not binary drops.

    Fail-open (True) when no ticker trend inputs are provided.
    """
    if close is not None and ema50 is not None and pd.notna(close) and pd.notna(ema50):
        ema50_f = float(ema50)
        if ema50_f > 0 and float(close) < 0.90 * ema50_f:
            return False
    if rs_63d is not None and pd.notna(rs_63d) and float(rs_63d) <= -0.15:
        return False
    return True


def relative_strength(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    as_of: AsOf | None = None,
    lookback: int = 63,
    ratio_ma_bars: int = 20,
) -> tuple[bool, float | None]:
    """Assess relative strength of *stock_df* vs *benchmark_df* (no lookahead).

    Passing requires the stock/benchmark price ratio above its moving average
    AND a positive lookback-period relative return. Returns (ok, rel_return).
    """
    if benchmark_df is None or benchmark_df.empty:
        return False, None

    s = stock_df[["Date", "Close"]].copy()
    s["Date"] = pd.to_datetime(s["Date"])
    b = benchmark_df[["Date", "Close"]].rename(columns={"Close": "Bench"}).copy()
    b["Date"] = pd.to_datetime(b["Date"])

    if as_of is not None:
        as_of_ts = pd.to_datetime(as_of)
        s = s[s["Date"] <= as_of_ts]
        b = b[b["Date"] <= as_of_ts]

    merged = s.merge(b, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    if len(merged) < max(lookback + 1, ratio_ma_bars):
        return False, None

    ratio = merged["Close"].astype(float) / merged["Bench"].astype(float)
    ratio_ma = ratio.rolling(ratio_ma_bars).mean()
    rs_now = float(ratio.iloc[-1])
    ma_now = ratio_ma.iloc[-1]

    stock_ret = merged["Close"].iloc[-1] / merged["Close"].iloc[-1 - lookback] - 1.0
    bench_ret = merged["Bench"].iloc[-1] / merged["Bench"].iloc[-1 - lookback] - 1.0
    rel_ret = float(stock_ret - bench_ret)

    ok = (not pd.isna(ma_now)) and rs_now > float(ma_now) and rel_ret > 0
    return ok, round(rel_ret, 4)
