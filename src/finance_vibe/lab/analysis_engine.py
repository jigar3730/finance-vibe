"""Macro Vibe Score engine (offline lab).

Scores each ticker in ``data/raw/{mode}/`` on a -10 to +10 scale using SMA trend,
MACD/RSI momentum, pullback timing, and RSI/CCI risk governors. Output is written
to ``data/logs/{mode}/vibe_report_<date>.csv``.

Not part of ``run_vibe.py``. Rubric: Scoring_Logic.md in this package.
OHLCV / QQQ RS helpers live in ``finance_vibe.market`` (shared with Coiled Cobra).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from finance_vibe import config
from finance_vibe.market import (
    ema,
    iter_raw_csv_paths,
    load_benchmark_frame,
    load_ohlc_csv,
    market_regime_ok,
    relative_strength,
    ticker_from_filename,
)

# Re-export for callers that imported these from this module.
__all__ = [
    "ScanRow",
    "build_features",
    "calculate_vibe_score",
    "ema",
    "iter_raw_csv_paths",
    "load_benchmark_frame",
    "load_ohlc_csv",
    "market_regime_ok",
    "relative_strength",
    "run_scan",
    "score_last_row",
    "ticker_from_filename",
]

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
# Indicators (vibe-specific; EMA is in market.py)
# -----------------------------


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


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


def run_scan(mode: str | None = None, max_workers: Optional[int] = None) -> pd.DataFrame:
    """Scan all raw CSVs for a mode and write the ranked vibe report."""
    mode = mode or config.DEFAULT_MODE
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
