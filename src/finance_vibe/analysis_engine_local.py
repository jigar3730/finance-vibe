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
        os.path.join(os.path.dirname(__file__), "../../")))
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


def score_last_row(last: pd.Series) -> int:
    score = 0
    close = last["Close"]
    sma20 = last["SMA20"]
    sma50 = last["SMA50"]
    rsi = last["RSI"]
    rsi_s = last["RSI_S"]
    cci = last["CCI"]
    cci_s = last["CCI_S"]
    macd_h = last["MACD_H"]
    macd_s = last["MACD_S"]

    # Trend
    if close > sma20 > sma50:
        score += 4
    elif close < sma20 < sma50:
        score -= 4

    # Momentum
    if macd_h > macd_s and rsi > rsi_s:
        score += 2
    elif macd_h < macd_s and rsi < rsi_s:
        score -= 2

    # Momentum decay
    if macd_h < macd_s and close > sma20:
        score -= 1

    # Timing / Pullback
    dist_sma20 = (close - sma20) / sma20
    if 0.0 <= dist_sma20 <= 0.05:
        score += 2
    elif dist_sma20 > 0.12:
        score -= 2
    elif dist_sma20 < -0.05:
        score -= 1

    # CCI
    if -100 < cci < 100 and cci > cci_s:
        score += 1
    elif cci > 200:
        score -= 2
    elif cci < -200:
        score += 1

    # RSI risk
    if rsi > 80:
        score = min(score, 5)
    elif rsi > 70:
        score -= 1
    if rsi < 30:
        score += 1

    # High score persistence
    if score >= 7:
        if not (macd_h > 0 and rsi > 50):
            score -= 2

    return int(np.clip(score, -10, 10))


def sentiment_action(score: int) -> tuple[str, str]:
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
    try:
        feat = build_features(df)
        last = feat.iloc[-1]
        score = score_last_row(last)

        if return_components:
            close = last["Close"]
            sma20 = last["SMA20"]
            sma50 = last["SMA50"]
            rsi = last["RSI"]
            rsi_s = last["RSI_S"]
            cci = last["CCI"]
            cci_s = last["CCI_S"]
            macd_h = last["MACD_H"]
            macd_s = last["MACD_S"]

            trend = 4 if close > sma20 > sma50 else (
                -4 if close < sma20 < sma50 else 0)
            momentum = 2 if macd_h > macd_s and rsi > rsi_s else (
                -2 if macd_h < macd_s and rsi < rsi_s else 0)
            momentum_decay = -1 if macd_h < macd_s and close > sma20 else 0
            dist_sma20 = (close - sma20) / sma20
            timing = 2 if 0.0 <= dist_sma20 <= 0.05 else (
                -2 if dist_sma20 > 0.12 else (-1 if dist_sma20 < -0.05 else 0))
            rsi_risk = - \
                1 if 70 < rsi <= 80 else (
                    min(score, 5) if rsi > 80 else (1 if rsi < 30 else 0))
            cci_logic = 1 if - \
                100 < cci < 100 and cci > cci_s else (-2 if cci > 200 else (1 if cci < -200 else 0))
            weekly_bonus = 1 if score >= 7 and (macd_h > 0 and rsi > 50) else 0
            persistence_check = 0  # placeholder

            components = {
                "Trend": trend,
                "Momentum": momentum,
                "MomentumDecay": momentum_decay,
                "Timing": timing,
                "CCI_Logic": cci_logic,
                "RSI_Risk": rsi_risk,
                "WeeklyBonus": weekly_bonus,
                "PersistenceCheck": persistence_check
            }

            return {"Score": score, "Components": components}

        return {"Score": score}

    except Exception as e:
        return {"Score": 0, "Error": str(e)}


def run_scan(max_workers: Optional[int] = None) -> pd.DataFrame:
    paths = list(iter_raw_csv_paths(config.RAW_DIR))
    if not paths:
        print(f"No CSV files found in {config.RAW_DIR}")
        return pd.DataFrame()

    results: list[ScanRow] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_one_file, p): p for p in paths}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except:
                continue

    out = pd.DataFrame([r.to_dict() for r in results])
    if out.empty:
        print("No results.")
        return out

    out = out.sort_values(["Score", "Ticker"], ascending=[
                          False, True]).reset_index(drop=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(config.LOGS_DIR, f"vibe_report_local_{stamp}.csv")
    out.to_csv(out_path, index=False)

    print(out.head(PRINT_TOP_N).to_markdown(index=False, floatfmt=".2f"))
    print(f"\n✅ Saved: {out_path}")
    return out


if __name__ == "__main__":
    run_scan()
