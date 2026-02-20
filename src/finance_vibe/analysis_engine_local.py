# scan_raw_fast.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import config


# -----------------------------
# Tunables
# -----------------------------
MIN_WEEKS = 60  # enough for SMA50 + signal windows
PRINT_TOP_N = 50  # printing huge markdown tables is slow


# -----------------------------
# Result model
# -----------------------------
@dataclass(frozen=True)
class ScanRow:
    ticker: str
    price: float
    sma20: float
    sma50: float
    cci: float
    cci_s: float
    macd_h: float
    macd_s: float
    rsi: float
    rsi_s: float
    score: int
    sentiment: str
    action: str

    def to_dict(self) -> dict:
        return {
            "Ticker": self.ticker,
            "Price": self.price,
            "SMA20": self.sma20,
            "SMA50": self.sma50,
            "CCI": self.cci,
            "CCI_S": self.cci_s,
            "MACD_H": self.macd_h,
            "MACD_S": self.macd_s,
            "RSI": self.rsi,
            "RSI_S": self.rsi_s,
            "Score": self.score,
            "Sentiment": self.sentiment,
            "Action": self.action,
        }


# -----------------------------
# File discovery / ticker parse
# -----------------------------
def iter_raw_csv_paths(raw_dir: str) -> Iterable[str]:
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"RAW_DIR does not exist: {raw_dir}")
    for name in sorted(os.listdir(raw_dir)):
        if name.lower().endswith(".csv"):
            yield os.path.join(raw_dir, name)


_TICKER_RE = re.compile(r"^([A-Za-z0-9\.\-]+)_", re.IGNORECASE)


def ticker_from_filename(path: str) -> str:
    """
    Expected: {TICKER}_{PERIOD}_{INTERVAL}.csv (ex: SPY_5y_1wk.csv)
    """
    base = os.path.basename(path)
    m = _TICKER_RE.match(base)
    if m:
        return m.group(1).upper()
    return os.path.splitext(base)[0].upper()


# -----------------------------
# CSV loader (fast-ish)
# -----------------------------
def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    norm = {c: c.strip().lower().replace("_", "").replace(" ", "")
            for c in df.columns}
    want = {x.lower().replace("_", "").replace(" ", "") for x in candidates}
    for original, n in norm.items():
        if n in want:
            return original
    return None


def load_ohlc_csv(path: str) -> pd.DataFrame:
    """
    Loads Date + Close (or Adj Close) and optional High/Low.
    Sorts ascending by date and converts numeric columns.
    """
    # Read only likely columns (works even if extra columns exist)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("empty csv")

    df.columns = [c.strip() for c in df.columns]

    date_col = _find_column(df, ["date", "datetime", "time"])
    close_col = _find_column(
        df, ["close", "adjclose", "adj close", "adj_close"])
    if not date_col:
        raise ValueError("missing Date column")
    if not close_col:
        raise ValueError("missing Close/Adj Close column")

    high_col = _find_column(df, ["high"])
    low_col = _find_column(df, ["low"])

    cols = [date_col, close_col]
    if high_col:
        cols.append(high_col)
    if low_col:
        cols.append(low_col)

    out = df[cols].copy()
    out.rename(columns={date_col: "Date", close_col: "Close"}, inplace=True)
    if high_col:
        out.rename(columns={high_col: "High"}, inplace=True)
    if low_col:
        out.rename(columns={low_col: "Low"}, inplace=True)

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values(
        "Date").reset_index(drop=True)

    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Close"]).reset_index(drop=True)

    if "High" in out.columns:
        out["High"] = pd.to_numeric(out["High"], errors="coerce")
    if "Low" in out.columns:
        out["Low"] = pd.to_numeric(out["Low"], errors="coerce")

    return out


# -----------------------------
# Indicators (fast)
# -----------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def cci_fast(tp: pd.Series, period: int = 20) -> pd.Series:
    """
    Vectorized CCI using sliding windows.
    Much faster than rolling().apply().
    """
    x = tp.to_numpy(dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return pd.Series(out, index=tp.index)

    w = np.lib.stride_tricks.sliding_window_view(
        x, period)  # (n-period+1, period)
    w_mean = w.mean(axis=1)
    w_md = np.mean(np.abs(w - w_mean[:, None]), axis=1)

    denom = 0.015 * w_md
    denom = np.where(np.abs(denom) > 1e-9, denom, 1e-9)

    tp_last = w[:, -1]
    out[period - 1:] = (tp_last - w_mean) / denom
    return pd.Series(out, index=tp.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)

    out["SMA20"] = sma(close, 20)
    out["SMA50"] = sma(close, 50)

    out["MACD_H"] = macd_hist(close)
    out["MACD_S"] = ema(out["MACD_H"], 9)

    out["RSI"] = rsi_wilder(close, 14)
    out["RSI_S"] = sma(out["RSI"], 10)

    if {"High", "Low"}.issubset(out.columns):
        tp = (out["High"].astype(float) +
              out["Low"].astype(float) + close) / 3.0
    else:
        tp = close

    out["CCI"] = cci_fast(tp, 20)
    out["CCI_S"] = sma(out["CCI"], 10)

    return out


# -----------------------------
# Scoring
# -----------------------------
def score_last_row(last: pd.Series) -> int:
    required = ["Close", "SMA20", "SMA50", "MACD_H",
                "MACD_S", "RSI", "RSI_S", "CCI", "CCI_S"]
    if any(pd.isna(last.get(k)) for k in required):
        raise ValueError("insufficient indicator history (NaN in last row)")

    price = float(last["Close"])
    sma20_v = float(last["SMA20"])
    sma50_v = float(last["SMA50"])

    macd_strong = float(last["MACD_H"]) > float(last["MACD_S"])
    rsi_strong = float(last["RSI"]) > float(last["RSI_S"])

    cci_v = float(last["CCI"])
    cci_s = float(last["CCI_S"])

    score = 0

    # Trend
    if price > sma20_v > sma50_v:
        score += 4
    elif price < sma20_v < sma50_v:
        score -= 4

    # Momentum
    if macd_strong and rsi_strong:
        score += 3
    elif macd_strong or rsi_strong:
        score += 1
    else:
        score -= 3

    # Volatility / stretch
    if cci_v > 0 and cci_v > cci_s:
        score += 3
    elif cci_v < 0 and cci_v < cci_s:
        score -= 3

    return int(score)


def sentiment_action(score: int) -> tuple[str, str]:
    if score >= 8:
        return "Bullish", "🔥 GO ALL IN"
    if 4 <= score <= 7:
        return "Positive", "📈 ACCUMULATE"
    if -3 <= score <= 3:
        return "Neutral", "⏳ WAIT / CASH"
    return "Bearish", "🧯 DISTRIBUTE / AVOID"


# -----------------------------
# Single file scan (worker-safe)
# -----------------------------
def scan_one_file(path: str) -> ScanRow:
    ticker = ticker_from_filename(path)
    df = load_ohlc_csv(path)

    if len(df) < MIN_WEEKS:
        raise ValueError(f"not enough rows: {len(df)} (<{MIN_WEEKS})")

    feat = build_features(df)
    last = feat.iloc[-1]

    score = score_last_row(last)
    sentiment, action = sentiment_action(score)

    return ScanRow(
        ticker=ticker,
        price=float(last["Close"]),
        sma20=float(last["SMA20"]),
        sma50=float(last["SMA50"]),
        cci=float(last["CCI"]),
        cci_s=float(last["CCI_S"]),
        macd_h=float(last["MACD_H"]),
        macd_s=float(last["MACD_S"]),
        rsi=float(last["RSI"]),
        rsi_s=float(last["RSI_S"]),
        score=score,
        sentiment=sentiment,
        action=action,
    )


# -----------------------------
# Orchestrator
# -----------------------------
def run_scan(max_workers: Optional[int] = None) -> pd.DataFrame:
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    paths = list(iter_raw_csv_paths(config.RAW_DIR))
    if not paths:
        print(f"No CSV files found in {config.RAW_DIR}")
        return pd.DataFrame()

    rows: list[ScanRow] = []
    failures: list[str] = []

    # Parallel scan: one CSV per process
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_one_file, p): p for p in paths}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                failures.append(f"{os.path.basename(p)} -> {e}")

    out = pd.DataFrame([r.to_dict() for r in rows])
    if out.empty:
        print("No results. (All files failed or insufficient history.)")
        if failures:
            print("\nFailures (first 15):")
            for msg in failures[:15]:
                print(" -", msg)
        return out

    out = out.sort_values(["Score", "Ticker"], ascending=[
                          False, True]).reset_index(drop=True)

    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(config.LOGS_DIR, f"vibe_report_local_{stamp}.csv")
    out.to_csv(out_path, index=False)

    # Print only top N to keep printing fast
    print(out.head(PRINT_TOP_N).to_markdown(index=False, floatfmt=".2f"))
    print(f"\nSaved: {out_path}")

    if failures:
        print(f"\nSkipped {len(failures)} file(s). Failures (first 15):")
        for msg in failures[:15]:
            print(" -", msg)

    return out


if __name__ == "__main__":
    # If you want to force a specific number of workers:
    # run_scan(max_workers=os.cpu_count())
    run_scan()
