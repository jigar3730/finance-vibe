"""Macro Vibe Score engine for Finance Vibe.

Scores each ticker in ``data/raw/{mode}/`` on a -10 to +10 scale using SMA trend,
MACD/RSI momentum, pullback timing, and RSI/CCI risk governors. Output is written
to ``data/logs/{mode}/vibe_report_<date>.csv``.

Full rubric: Scoring_Logic.md in this package.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. PACKAGE IMPORT ---
try:
    from finance_vibe import config
except ImportError:
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config

# -----------------------------
# Tunables
# -----------------------------
MIN_ROWS = 60
PRINT_TOP_N = 500

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


def ticker_from_filename(path: str) -> str:
    base = os.path.basename(path)
    return base.split('_')[0].upper()

# -----------------------------
# CSV loader
# -----------------------------


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

# -----------------------------
# Indicators
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


def cci_fast(df: pd.DataFrame, period: int = 20) -> pd.Series:
    if "High" in df.columns and "Low" in df.columns:
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    else:
        tp = df["Close"]

    x = tp.to_numpy(dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return pd.Series(out, index=tp.index)

    w = np.lib.stride_tricks.sliding_window_view(x, period)
    w_mean = w.mean(axis=1)
    w_md = np.mean(np.abs(w - w_mean[:, None]), axis=1)

    denom = 0.015 * w_md
    denom = np.where(np.abs(denom) > 1e-9, denom, 1e-9)
    tp_last = w[:, -1]
    out[period - 1:] = (tp_last - w_mean) / denom
    return pd.Series(out, index=tp.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, MACD, RSI, and CCI columns required for scoring."""
    out = df.copy()
    close = out["Close"].astype(float)
    out["SMA20"] = sma(close, 20)
    out["SMA50"] = sma(close, 50)
    out["MACD_H"] = macd_hist(close)
    out["MACD_S"] = ema(out["MACD_H"], 9)
    out["RSI"] = rsi_wilder(close, 14)
    out["RSI_S"] = sma(out["RSI"], 10)
    out["CCI"] = cci_fast(out, 20)
    out["CCI_S"] = sma(out["CCI"], 10)
    return out


# -----------------------------
# Benchmark: market regime + relative strength (no lookahead)
# -----------------------------


def _select_benchmark_path(benchmark: str, data_mode: str) -> Optional[str]:
    """Find the longest-history raw CSV for *benchmark* in the mode's raw dir."""
    cfg = config.get_mode_config(data_mode)
    raw_dir = cfg["raw_dir"]
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
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv") and f.split("_")[0].upper() == bench
    ]
    if not candidates:
        return None
    best = max(candidates, key=_rank)
    return os.path.join(raw_dir, best)


def load_benchmark_frame(benchmark: str, data_mode: str) -> Optional[pd.DataFrame]:
    """Load and enrich a benchmark OHLC frame for regime/RS checks.

    Returns a Date-sorted frame with causal EMA50/EMA100 and an ``EMA50_rising``
    flag, or None when the benchmark CSV is unavailable.
    """
    path = _select_benchmark_path(benchmark, data_mode)
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
    """True when the benchmark is in an uptrend as of *as_of* (causal lookup).

    Requires close above EMA50 and EMA100 with a rising EMA50.
    """
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

# -----------------------------
# Scoring
# -----------------------------


def _compute_score(last: pd.Series) -> tuple[int, dict[str, int]]:
    """Apply the Vibe Score rubric to the latest indicator row.

    Returns the clipped integer score and a per-component point breakdown.
    See Scoring_Logic.md for the full specification.
    """
    score = 0
    components: dict[str, int] = {}

    close = last["Close"]
    sma20 = last["SMA20"]
    sma50 = last["SMA50"]
    rsi = last["RSI"]
    rsi_s = last["RSI_S"]
    cci = last["CCI"]
    cci_s = last["CCI_S"]
    macd_h = last["MACD_H"]
    macd_s = last["MACD_S"]

    # Trend: full alignment only (+/-4)
    if close > sma20 > sma50:
        components["Trend"] = 4
    elif close < sma20 < sma50:
        components["Trend"] = -4
    else:
        components["Trend"] = 0
    score += components["Trend"]

    # Momentum: MACD histogram and RSI vs their smoothers
    if macd_h > macd_s and rsi > rsi_s:
        components["Momentum"] = 2
    elif macd_h < macd_s and rsi < rsi_s:
        components["Momentum"] = -2
    else:
        components["Momentum"] = 0
    score += components["Momentum"]

    # Weakening momentum while price holds above SMA20
    components["MomentumDecay"] = -1 if macd_h < macd_s and close > sma20 else 0
    score += components["MomentumDecay"]

    # Pullback timing relative to SMA20
    dist_sma20 = (close - sma20) / sma20
    if 0.0 <= dist_sma20 <= 0.05:
        components["Timing"] = 2
    elif dist_sma20 > 0.12:
        components["Timing"] = -2
    elif dist_sma20 < -0.05:
        components["Timing"] = -1
    else:
        components["Timing"] = 0
    score += components["Timing"]

    # CCI cyclical band (not used as raw momentum)
    if -100 < cci < 100 and cci > cci_s:
        components["CCI"] = 1
    elif cci > 200:
        components["CCI"] = -2
    elif cci < -200:
        components["CCI"] = 1
    else:
        components["CCI"] = 0
    score += components["CCI"]

    # RSI overextension caps
    if rsi > 80:
        capped = min(score, 5)
        components["RSI_Risk"] = capped - score
        score = capped
    elif rsi > 70:
        components["RSI_Risk"] = -1
        score -= 1
    elif rsi < 30:
        components["RSI_Risk"] = 1
        score += 1
    else:
        components["RSI_Risk"] = 0

    # High scores need MACD > 0 and RSI > 50
    if score >= 7 and not (macd_h > 0 and rsi > 50):
        components["Persistence"] = -2
        score -= 2
    else:
        components["Persistence"] = 0

    final = int(np.clip(score, -10, 10))
    return final, components


def score_last_row(last: pd.Series) -> int:
    """Return the Vibe Score for the latest bar."""
    score, _ = _compute_score(last)
    return score


def sentiment_action(score: int) -> tuple[str, str]:
    """Map a Vibe Score to display sentiment and sizing-oriented action text."""
    if score >= 9:
        return "Bullish", "🟢 STARTER + ADD ON PULLBACK"
    if 7 <= score <= 8:
        return "Bullish", "🟢 STARTER POSITION"
    if 5 <= score <= 6:
        return "Positive", "📈 WATCH / SCALE IN"
    if 2 <= score <= 4:
        return "Neutral", "⏳ WAIT"
    if -1 <= score <= 1:
        return "Neutral", "💤 NO EDGE"
    if -4 <= score <= -2:
        return "Bearish", "🟠 REDUCE / HEDGE"
    return "Bearish", "🔴 AVOID / SHORT BIAS"

# -----------------------------
# Workers
# -----------------------------


def scan_one_file(path: str) -> ScanRow:
    ticker = ticker_from_filename(path)
    df = load_ohlc_csv(path)
    if len(df) < MIN_ROWS:
        raise ValueError(f"not enough rows: {len(df)}")

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


def calculate_vibe_score(ticker: str, df: pd.DataFrame, return_components: bool = False) -> dict:
    """Score a single OHLC DataFrame (used by tests and ad-hoc analysis).

    Args:
        ticker: Symbol label (included for API compatibility; not used in math).
        df: OHLCV history with Date and Close columns.
        return_components: If True, include per-component point breakdown.

    Returns:
        Dict with ``Score`` and optionally ``Components`` or ``Error``.
    """
    try:
        feat = build_features(df)
        last = feat.iloc[-1]
        score, components = _compute_score(last)

        if return_components:
            return {"Score": score, "Components": components}
        return {"Score": score}

    except Exception as e:
        return {"Score": 0, "Error": str(e)}


def run_scan(mode: str = "weekly", max_workers: Optional[int] = None) -> pd.DataFrame:
    """Scan all raw CSVs for a mode and write the ranked vibe report."""
    mode_cfg = config.get_mode_config(mode)
    raw_dir = mode_cfg["raw_dir"]
    logs_dir = mode_cfg["logs_dir"]

    print(f"--- STEP 3: Macro Vibe Score Scan [{mode.upper()} MODE] ---")

    paths = list(iter_raw_csv_paths(raw_dir))
    if not paths:
        print(f"No CSV files found in {raw_dir}")
        return pd.DataFrame()

    results: list[ScanRow] = []
    errors = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_one_file, p): p for p in paths}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                errors += 1
                continue

    if errors:
        print(f"⚠️ Skipped {errors} ticker(s) due to load/scoring errors.")

    out = pd.DataFrame([r.to_dict() for r in results])
    if out.empty:
        print("No results.")
        return out

    out = out.sort_values(["Score", "Ticker"], ascending=[
                          False, True]).reset_index(drop=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(logs_dir, f"vibe_report_{stamp}.csv")
    out.to_csv(out_path, index=False)

    print(out.head(PRINT_TOP_N).to_markdown(index=False, floatfmt=".2f"))
    print(f"\n✅ Saved: {out_path}")
    return out


if __name__ == "__main__":
    cli_mode = config.DEFAULT_MODE
    if len(sys.argv) > 1 and sys.argv[1].lower() in config.TIMEFRAME_PROFILES:
        cli_mode = sys.argv[1].lower()
    run_scan(mode=cli_mode)
