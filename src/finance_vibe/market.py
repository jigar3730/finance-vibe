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
    ticker, _, _ = parse_raw_filename(path)
    return ticker


def parse_raw_filename(path: str) -> tuple[str, str | None, str | None]:
    """Split ``AAPL_10y_1d.csv`` into ``(ticker, period, interval)``."""
    base = os.path.basename(path)
    if base.lower().endswith(".csv"):
        base = base[:-4]
    parts = base.split("_")
    ticker = parts[0].upper() if parts else ""
    period = parts[1].lower() if len(parts) > 1 else None
    interval = parts[2].lower() if len(parts) > 2 else None
    return ticker, period, interval


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
    _, tok, _ = parse_raw_filename(name)
    if not tok:
        return 0
    if tok.endswith("y") and tok[:-1].isdigit():
        return int(tok[:-1]) * 365
    if tok.endswith("mo") and tok[:-2].isdigit():
        return int(tok[:-2]) * 30
    if tok.endswith("d") and tok[:-1].isdigit():
        return int(tok[:-1])
    return 0


def select_raw_paths(
    raw_dir: str,
    ticker_filter: Optional[set[str]] = None,
    *,
    cfg: dict | None = None,
    mode: str | None = None,
) -> list[str]:
    """One raw CSV per ticker, preferring ``TIMEFRAME_PROFILES`` period/interval.

    Daily config is ``10y`` × ``1d``. If both ``AAPL_5y_1d.csv`` and
    ``AAPL_10y_1d.csv`` exist, only the configured 10y file is scanned.
    """
    if not os.path.isdir(raw_dir):
        return []
    cfg = cfg if cfg is not None else config.get_mode_config(mode)
    want_period = str(cfg.get("period", "")).strip().lower()
    want_interval = str(cfg.get("interval", "")).strip().lower()
    want_tickers = {t.strip().upper() for t in ticker_filter} if ticker_filter else None

    best: dict[str, tuple[int, str]] = {}
    for name in os.listdir(raw_dir):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(raw_dir, name)
        ticker, period, interval = parse_raw_filename(name)
        if not ticker:
            continue
        if want_tickers is not None and ticker not in want_tickers:
            continue
        rank = _period_rank(name)
        if want_period and period == want_period and want_interval and interval == want_interval:
            rank += 1_000_000
        elif want_interval and interval == want_interval:
            rank += 10_000
        prev = best.get(ticker)
        if prev is None or rank > prev[0]:
            best[ticker] = (rank, path)
    return [best[s][1] for s in sorted(best)]


def resolve_raw_path(ticker: str, cfg: dict | None = None, mode: str | None = None) -> Optional[str]:
    """Configured ``{TICKER}_{period}_{interval}.csv``, else longest matching fallback."""
    cfg = cfg if cfg is not None else config.get_mode_config(mode)
    exact = config.get_raw_path(ticker, cfg)
    if os.path.isfile(exact):
        return exact
    paths = select_raw_paths(cfg["raw_dir"], {ticker.upper()}, cfg=cfg)
    return paths[0] if paths else None


def select_benchmark_path(benchmark: str, data_mode: str) -> Optional[str]:
    """Find the configured-period (else longest) raw CSV for *benchmark*."""
    cfg = config.get_mode_config(data_mode)
    return resolve_raw_path(benchmark, cfg)


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
