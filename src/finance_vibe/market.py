"""Shared OHLCV loading and QQQ relative-strength helpers.

Used by the live Coiled Cobra scanner and by the offline lab. Macro Vibe
scoring stays in ``finance_vibe.lab.analysis_engine``.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd

from finance_vibe import config


def iter_raw_csv_paths(raw_dir: str) -> Iterable[str]:
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"RAW_DIR does not exist: {raw_dir}")
    for name in sorted(os.listdir(raw_dir)):
        if name.lower().endswith(".csv"):
            yield os.path.join(raw_dir, name)


def ticker_from_filename(path: str) -> str:
    base = os.path.basename(path)
    return base.split("_")[0].upper()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def load_ohlc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("empty csv")

    df.columns = [c.strip().capitalize() for c in df.columns]
    date_col = next((c for c in df.columns if "Date" in c), None)
    close_col = next((c for c in df.columns if "Close" in c), None)

    if not date_col or not close_col:
        raise ValueError(f"Missing Date or Close in {path}")

    df = df.rename(columns={date_col: "Date", close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    if "High" in df.columns:
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
    if "Low" in df.columns:
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")

    return df.dropna(subset=["Date", "Close"])


def _period_rank(name: str) -> int:
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


def select_benchmark_path(benchmark: str, data_mode: str) -> Optional[str]:
    """Find the longest-history raw CSV for *benchmark* in the mode's raw dir."""
    cfg = config.get_mode_config(data_mode)
    raw_dir = cfg["raw_dir"]
    if not os.path.isdir(raw_dir):
        return None
    bench = benchmark.upper()
    candidates = [
        f
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv") and f.split("_")[0].upper() == bench
    ]
    if not candidates:
        return None
    best = max(candidates, key=_period_rank)
    return os.path.join(raw_dir, best)


def load_benchmark_frame(benchmark: str, data_mode: str) -> Optional[pd.DataFrame]:
    """Load a benchmark OHLC frame with causal EMA50/EMA100 and EMA50_rising."""
    path = select_benchmark_path(benchmark, data_mode)
    if not path:
        return None
    try:
        df = load_ohlc_csv(path)
    except Exception:
        return None
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    df["EMA50"] = ema(close, 50)
    df["EMA100"] = ema(close, 100)
    df["EMA50_rising"] = df["EMA50"] > df["EMA50"].shift(1)
    return df


def market_regime_ok(benchmark_df: pd.DataFrame, as_of) -> bool:
    """True when the benchmark is in an uptrend as of *as_of* (causal lookup)."""
    if benchmark_df is None or benchmark_df.empty:
        return False
    as_of_ts = pd.to_datetime(as_of) if as_of is not None else None
    sub = benchmark_df if as_of_ts is None else benchmark_df[benchmark_df["Date"] <= as_of_ts]
    if sub.empty:
        return False
    last = sub.iloc[-1]
    if pd.isna(last["EMA50"]) or pd.isna(last["EMA100"]):
        return False
    return bool(
        last["Close"] > last["EMA50"]
        and last["Close"] > last["EMA100"]
        and bool(last["EMA50_rising"])
    )


def relative_strength(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    as_of=None,
    lookback: int = 63,
    ratio_ma_bars: int = 20,
) -> tuple[bool, Optional[float]]:
    """Stock vs benchmark RS: ratio above its MA and positive lookback relative return."""
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
