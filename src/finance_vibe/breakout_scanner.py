"""Breakout readiness scanner for Finance Vibe.

Two-stage pipeline (features are the source of truth; the score is secondary):

    OHLCV → clean/validate → Daily/Weekly/Monthly
          → FeatureEngine (trend, momentum, volatility, volume, structure)
          → ScoringEngine (100-pt readiness + fakeout penalties + state class)
          → data/logs/{mode}/breakout_setups_<YYYY-MM-DD>.csv

Designed to surface **pre-breakout** candidates (compression + proximity +
accelerating momentum) rather than only names that have already broken out.

Native bars are never invented: daily raw data is resampled to weekly/monthly.
Weekly raw data is resampled to monthly only — it does not pretend to contain
daily information. Intraday (4H/1H) is out of scope until the raw dataset
includes those bars.

Run alongside ``swing_scanner.py``; this module is not part of ``run_vibe.py``.

Usage::

    python src/finance_vibe/breakout_scanner.py weekly
    python src/finance_vibe/breakout_scanner.py daily
    python src/finance_vibe/breakout_scanner.py high_beta
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    from finance_vibe import config
except ImportError:
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config

# =========================
# PROFILE CONFIGURATION
# =========================
if len(sys.argv) > 1 and sys.argv[1].lower() in ["weekly", "daily", "high_beta"]:
    mode = sys.argv[1].lower()
else:
    print("⚠️ Unknown mode parsed to scanner. Defaulting to 'weekly'.")
    mode = "weekly"

_data_mode, _swing_profile = config.resolve_pipeline_mode(mode)
mode = _swing_profile

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", _data_mode)
ACTIVE_TICKERS_PATH = os.path.join(BASE_DIR, "data", "active_tickers.csv")
LOG_DIR = config.get_log_dir(mode)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# =========================
# CALIBRATION
# =========================
SMA_FAST = 20
SMA_MID = 50
SMA_SLOW = 200
RSI_LEN = 14
MACD_FAST = 15
MACD_SLOW = 30
MACD_SIGNAL = 9
BB_LEN = 20
BB_STD = 2.0
KC_LEN = 20
KC_SCALAR = 1.5
ATR_LEN = 14
RVOL_LEN = 20
STRUCTURE_BARS = 20
RANGE_BARS = 20
SLOPE_RSI_BARS = 5
SLOPE_MACD_BARS = 3
VOL_DRYUP_FAST = 5
VOL_DRYUP_RATIO = 0.75
RVOL_DRYUP = 0.85
RVOL_CONFIRM = 1.50
RVOL_IGNITION = 1.00
PRE_BREAKOUT_ATR_MAX = 1.25
AT_RESISTANCE_ATR = 0.25
FAKEOUT_LOOKBACK = 5
PCTL_MIN_PERIODS = 20
MIN_PRIMARY_BARS = 80

# Causal percentile windows by native bar size (never use the full series,
# which would leak future ranks into historical bars).
PCTL_WINDOW = {
    "daily": 252,
    "weekly": 52,
    "monthly": 36,
}

STATUS_PRE = "PRE_BREAKOUT"
STATUS_WATCH = "WATCH"
STATUS_DEV = "DEVELOPING"
STATUS_CONFIRMED = "BREAKOUT_CONFIRMED"
STATUS_FAILED = "FAILED_BREAKOUT"

CANDIDATE_STATUSES = {STATUS_PRE, STATUS_CONFIRMED, STATUS_FAILED}
DISPLAY_SCORE_FLOOR = {
    STATUS_WATCH: 55,
    STATUS_DEV: 50,
}

# Display order for the console table (states first — score is secondary).
DISPLAY_COLUMNS = [
    "Symbol",
    "Status",
    "AsOf Date",
    "Close",
    "Trend",
    "Volatility",
    "Volume State",
    "Momentum",
    "Structure",
    "Breakout Distance",
    "MTF",
    "Breakout Readiness",
    "Distance Resistance ATR",
    "RVOL20",
    "Compression",
]

# Research CSV: identity + raw features + pillar/factor scores + states.
OUTPUT_COLUMNS = [
    "Symbol",
    "Mode",
    "Source",
    "AsOf Date",
    "Close",
    "SMA20",
    "SMA50",
    "SMA200",
    "RSI",
    "ATR",
    "Resistance",
    "Support",
    "BB Width Pctl",
    "KC Width Pctl",
    "ATR Pctl",
    "Range20 Pctl",
    "RVOL20",
    "RSI Slope",
    "MACD Hist Slope",
    "Distance Resistance ATR",
    "Extension ATR",
    "Daily Trend Bull",
    "Weekly Trend Bull",
    "Monthly Trend Bull",
    "MTF Alignment",
    "Compression",
    "Breakout Triggered",
    "Breakout Confirmation",
    "Failed Breakout",
    "Trend",
    "Volatility",
    "Volume State",
    "Momentum",
    "Structure",
    "Breakout Distance",
    "MTF",
    "Status",
    "Breakout Readiness",
    "Trend Score",
    "Compression Score",
    "Momentum Score",
    "Volume Score",
    "Structure Score",
    "Penalty",
    "Daily Trend Score",
    "Weekly Trend Score",
    "Monthly Trend Score",
    "BB Width Score",
    "ATR Pctl Score",
    "Range20 Score",
    "Squeeze Score",
    "RSI Score",
    "RSI Slope Score",
    "MACD Hist Slope Score",
    "Volume Dryup Score",
    "RVOL Score",
    "Proximity Score",
    "Extension Score",
]

_OHLCV_ALIASES = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Close",
    "adj_close": "Close",
    "adjclose": "Close",
    "volume": "Volume",
    "vol": "Volume",
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "time": "Date",
}

_RESAMPLE_AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}


# =========================
# SMALL HELPERS
# =========================

def _safe_float(value, digits: int | None = None) -> float | None:
    """Return a finite float (optionally rounded) or None."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    if digits is not None:
        return round(out, digits)
    return out


def _safe_bool(value) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return bool(value)


def _last(series: pd.Series | None, digits: int | None = None) -> float | None:
    if series is None or len(series) == 0:
        return None
    return _safe_float(series.iloc[-1], digits)


def _ta_col(frame: pd.DataFrame | None, *prefixes: str) -> pd.Series | None:
    """Pick the first pandas_ta column whose name starts with any prefix."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    for prefix in prefixes:
        for col in frame.columns:
            if str(col).startswith(prefix):
                return frame[col]
    return None


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Causal percentile rank of the *current* value inside ``window`` bars.

    Uses pandas' rolling rank (no future bars). Result is 0–100.
    """
    min_periods = min(PCTL_MIN_PERIODS, window)
    ranked = series.rolling(window=window, min_periods=min_periods).rank(pct=True)
    return ranked * 100.0


def _series_or_nan(value, index: pd.Index) -> pd.Series:
    """Coerce a pandas_ta result to a float Series aligned to ``index``."""
    if value is None:
        return pd.Series(np.nan, index=index, dtype=float)
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    series = pd.Series(value, dtype=float)
    if len(series) == len(index):
        series.index = index
        return series
    return series.reindex(index)


def _sma(series: pd.Series, length: int) -> pd.Series:
    computed = ta.sma(series, length=length)
    if computed is None:
        return series.rolling(length, min_periods=length).mean()
    return _series_or_nan(computed, series.index)


def _native_timeframe(data_mode: str) -> str:
    return "daily" if data_mode == "daily" else "weekly"


def _asof_str(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    ts = pd.Timestamp(value)
    return str(ts.date())


# =========================
# DATA CLEANING
# =========================

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Rename, validate, parse dates, sort, and drop unusable rows.

    Safe against common broker/CSV header variants. Volume is required
    (RVOL / OBV / dry-up). Duplicate timestamps keep the last print.
    """
    if df is None or df.empty:
        raise ValueError("empty ohlcv frame")

    out = df.copy()
    if not isinstance(out.columns, pd.Index):
        out.columns = pd.Index(out.columns)

    renamed = {}
    for col in out.columns:
        key = str(col).strip().lower()
        if key in _OHLCV_ALIASES:
            renamed[col] = _OHLCV_ALIASES[key]
    if renamed:
        out = out.rename(columns=renamed)

    out = config.validate_and_clean_ohlcv(out, require_volume=True)

    out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce")
    out["Date"] = out["Date"].dt.tz_localize(None)
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    out = out.reset_index(drop=True)

    if "Volume" in out.columns:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)
        out.loc[out["Volume"] < 0, "Volume"] = 0.0

    if out.empty:
        raise ValueError("no usable rows after ohlcv normalization")
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample native bars to a higher timeframe without lookahead.

    ``label='right'`` / ``closed='right'`` so a period only includes bars
    that have already printed. A trailing forming bar is kept (it uses only
    data through the last native timestamp) and its label is clamped to that
    last native date so the stamp never sits in the future.
    """
    if "Date" not in df.columns:
        raise ValueError("resample_ohlcv requires a Date column; normalize OHLCV first")
    cols = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
    work = df.loc[:, cols].dropna(subset=["Date"]).copy()
    work = work.set_index("Date").sort_index()
    if work.empty:
        return work.reset_index()

    last_native = work.index.max()
    out = work.resample(rule, label="right", closed="right").agg(_RESAMPLE_AGG)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if out.empty:
        return out.reset_index()

    # Forming period: resample may label it at period-end (e.g. Friday)
    # while the last native bar is earlier in the same period.
    if out.index[-1] > last_native:
        out = out.rename(index={out.index[-1]: last_native})
    return out.reset_index()


# =========================
# FEATURE ENGINE
# =========================

def add_indicators(df: pd.DataFrame, *, pctl_window: int, include_sma200: bool = True) -> pd.DataFrame:
    """Vectorized indicator stack. Every rolling window is causal (past + now).

    Structure levels use a 1-bar shift so the current print cannot be its
    own 20-bar resistance/support (no same-bar lookahead).
    """
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float) if "Volume" in out.columns else pd.Series(np.nan, index=out.index)

    out["SMA20"] = _sma(close, SMA_FAST)
    out["SMA50"] = _sma(close, SMA_MID)
    if include_sma200:
        out["SMA200"] = _sma(close, SMA_SLOW)
    else:
        out["SMA200"] = np.nan

    rsi = ta.rsi(close, length=RSI_LEN)
    out["RSI"] = _series_or_nan(rsi, out.index)

    macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    macd_line = _ta_col(macd, "MACD_15_30_9")
    macd_signal = _ta_col(macd, "MACDs_15_30_9", "MACDs_")
    macd_hist = _ta_col(macd, "MACDh_15_30_9", "MACDh_")
    out["MACD"] = _series_or_nan(macd_line, out.index)
    out["MACD_Signal"] = _series_or_nan(macd_signal, out.index)
    out["MACD_Hist"] = _series_or_nan(macd_hist, out.index)

    bb = ta.bbands(close, length=BB_LEN, std=BB_STD)
    bb_lower = _ta_col(bb, "BBL_")
    bb_mid = _ta_col(bb, "BBM_")
    bb_upper = _ta_col(bb, "BBU_")
    if bb_lower is None or bb_mid is None or bb_upper is None:
        out["BB_Lower"] = np.nan
        out["BB_Mid"] = np.nan
        out["BB_Upper"] = np.nan
        out["BB_Width"] = np.nan
    else:
        out["BB_Lower"] = bb_lower
        out["BB_Mid"] = bb_mid
        out["BB_Upper"] = bb_upper
        mid = bb_mid.replace(0, np.nan)
        out["BB_Width"] = (bb_upper - bb_lower) / mid

    kc = ta.kc(high, low, close, length=KC_LEN, scalar=KC_SCALAR)
    kc_lower = _ta_col(kc, "KCLe_", "KCL_")
    kc_mid = _ta_col(kc, "KCBe_", "KCB_")
    kc_upper = _ta_col(kc, "KCUe_", "KCU_")
    if kc_lower is None or kc_mid is None or kc_upper is None:
        out["KC_Lower"] = np.nan
        out["KC_Mid"] = np.nan
        out["KC_Upper"] = np.nan
        out["KC_Width"] = np.nan
    else:
        out["KC_Lower"] = kc_lower
        out["KC_Mid"] = kc_mid
        out["KC_Upper"] = kc_upper
        kmid = kc_mid.replace(0, np.nan)
        out["KC_Width"] = (kc_upper - kc_lower) / kmid

    out["ATR"] = _series_or_nan(ta.atr(high, low, close, length=ATR_LEN), out.index)

    vol_sma = _sma(volume, RVOL_LEN)
    out["VOL_SMA20"] = vol_sma
    out["RVOL20"] = volume / vol_sma.replace(0, np.nan)

    vol_fast = volume.rolling(VOL_DRYUP_FAST, min_periods=VOL_DRYUP_FAST).mean()
    out["VOL_SMA5"] = vol_fast
    out["Volume Dryup"] = (vol_fast / vol_sma.replace(0, np.nan)) < VOL_DRYUP_RATIO

    out["OBV"] = _series_or_nan(ta.obv(close, volume), out.index)
    out["OBV_SMA20"] = _sma(out["OBV"], RVOL_LEN)

    out["Range20"] = (
        high.rolling(RANGE_BARS, min_periods=RANGE_BARS).max()
        - low.rolling(RANGE_BARS, min_periods=RANGE_BARS).min()
    )

    # Prior-bar structure — current high cannot define its own breakout level.
    out["Resistance"] = high.rolling(STRUCTURE_BARS, min_periods=STRUCTURE_BARS).max().shift(1)
    out["Support"] = low.rolling(STRUCTURE_BARS, min_periods=STRUCTURE_BARS).min().shift(1)

    # Confirmed 3-bar swing pivots (pivot at i-1, confirmed on bar i).
    out["Swing High"] = (
        (high.shift(1) > high.shift(2)) & (high.shift(1) > high)
    )
    out["Swing Low"] = (
        (low.shift(1) < low.shift(2)) & (low.shift(1) < low)
    )

    atr = out["ATR"].replace(0, np.nan)
    out["Distance Resistance ATR"] = (out["Resistance"] - close) / atr
    out["Extension ATR"] = (close - out["SMA20"]) / atr
    out["Range20 / ATR"] = out["Range20"] / atr

    out["RSI Slope"] = out["RSI"].diff(SLOPE_RSI_BARS)
    out["MACD Hist Slope"] = out["MACD_Hist"].diff(SLOPE_MACD_BARS)
    out["MACD Hist Accel"] = out["MACD Hist Slope"] - out["MACD Hist Slope"].shift(SLOPE_MACD_BARS)

    out["BB Width Pctl"] = _rolling_percentile(out["BB_Width"], pctl_window)
    out["KC Width Pctl"] = _rolling_percentile(out["KC_Width"], pctl_window)
    out["ATR Pctl"] = _rolling_percentile(out["ATR"], pctl_window)
    out["Range20 Pctl"] = _rolling_percentile(out["Range20"], pctl_window)

    squeeze = (
        out["BB_Upper"].notna()
        & out["KC_Upper"].notna()
        & (out["BB_Upper"] < out["KC_Upper"])
        & (out["BB_Lower"] > out["KC_Lower"])
    )
    out["Squeeze"] = squeeze

    out["Compression"] = (
        (out["BB Width Pctl"] <= 30)
        & (out["ATR Pctl"] <= 40)
        & (out["Range20 Pctl"] <= 40)
    )

    above_res = (close > out["Resistance"]).fillna(False)
    out["Breakout Triggered"] = above_res
    wick_reject = ((high > out["Resistance"]) & (close < out["Resistance"])).fillna(False)
    lost_level = (
        above_res.shift(1).rolling(FAKEOUT_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
        & ~above_res
    )
    out["Wick Reject"] = wick_reject
    out["Failed Breakout"] = lost_level
    out["Breakout Confirmation"] = (
        above_res
        & (out["RVOL20"] >= RVOL_CONFIRM)
        & (close > out["SMA20"])
    ).fillna(False)
    return out


def _trend_bull(df: pd.DataFrame, *, require_sma50: bool = True) -> bool | None:
    """SMA stack on the last bar: Close > SMA20 > SMA50 (and > SMA200 if present).

    Monthly frames often lack SMA50 early in a 5y history. In that case
    ``require_sma50=False`` falls back to Close > rising SMA20.
    """
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    sma20 = _safe_float(last.get("SMA20"))
    sma50 = _safe_float(last.get("SMA50"))
    sma200 = _safe_float(last.get("SMA200"))
    if close is None or sma20 is None:
        return None
    bull = close > sma20
    if sma50 is not None:
        bull = bull and sma20 > sma50 and close > sma50
    elif require_sma50:
        return None
    if sma200 is not None:
        bull = bull and close > sma200
    if len(df) >= 2:
        prev_sma20 = _safe_float(df.iloc[-2].get("SMA20"))
        if prev_sma20 is not None:
            bull = bull and sma20 >= prev_sma20
    return bool(bull)


def _trend_bear(df: pd.DataFrame) -> bool | None:
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    sma20 = _safe_float(last.get("SMA20"))
    sma50 = _safe_float(last.get("SMA50"))
    if close is None or sma20 is None or sma50 is None:
        return None
    return bool(close < sma20 < sma50)


@dataclass
class BreakoutFeatures:
    """Last-bar snapshot produced by :class:`FeatureEngine`. No scores."""

    symbol: str
    mode: str
    asof: str | None
    close: float | None
    sma20: float | None
    sma50: float | None
    sma200: float | None
    rsi: float | None
    atr: float | None
    resistance: float | None
    support: float | None
    bb_width_pctl: float | None
    kc_width_pctl: float | None
    atr_pctl: float | None
    range20_pctl: float | None
    rvol20: float | None
    rsi_slope: float | None
    macd_hist_slope: float | None
    macd_hist_accel: float | None
    distance_resistance_atr: float | None
    extension_atr: float | None
    range20_atr: float | None
    daily_trend_bull: bool | None
    weekly_trend_bull: bool | None
    monthly_trend_bull: bool | None
    daily_trend_bear: bool | None
    weekly_trend_bear: bool | None
    monthly_trend_bear: bool | None
    squeeze: bool | None
    compression: bool | None
    volume_dryup: bool | None
    obv_rising: bool | None
    breakout_triggered: bool | None
    breakout_confirmation: bool | None
    failed_breakout: bool | None
    wick_reject: bool | None
    swing_high: bool | None
    swing_low: bool | None
    macd_hist: float | None
    has_daily: bool = False

    def feature_row(self) -> dict:
        """Underlying features for the research CSV (no scores)."""
        return {
            "Symbol": self.symbol,
            "Mode": self.mode,
            "Source": "breakout",
            "AsOf Date": self.asof,
            "Close": self.close,
            "SMA20": self.sma20,
            "SMA50": self.sma50,
            "SMA200": self.sma200,
            "RSI": self.rsi,
            "ATR": self.atr,
            "Resistance": self.resistance,
            "Support": self.support,
            "BB Width Pctl": self.bb_width_pctl,
            "KC Width Pctl": self.kc_width_pctl,
            "ATR Pctl": self.atr_pctl,
            "Range20 Pctl": self.range20_pctl,
            "RVOL20": self.rvol20,
            "RSI Slope": self.rsi_slope,
            "MACD Hist Slope": self.macd_hist_slope,
            "Distance Resistance ATR": self.distance_resistance_atr,
            "Extension ATR": self.extension_atr,
            "Daily Trend Bull": self.daily_trend_bull,
            "Weekly Trend Bull": self.weekly_trend_bull,
            "Monthly Trend Bull": self.monthly_trend_bull,
            "Compression": self.compression,
            "Breakout Triggered": self.breakout_triggered,
            "Breakout Confirmation": self.breakout_confirmation,
            "Failed Breakout": self.failed_breakout,
        }


class FeatureEngine:
    """OHLCV → timeframes → indicators → last-bar feature snapshot.

    Scoring / classification must not live here. Callers that want a
    historical bar should pass ``df.iloc[: i + 1]`` so only data through
    that bar is visible.
    """

    def __init__(self, native_tf: str):
        if native_tf not in ("daily", "weekly"):
            raise ValueError(f"unsupported native timeframe: {native_tf}")
        self.native_tf = native_tf

    def create_timeframes(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Build Monthly / Weekly / Daily frames from native OHLCV only."""
        frames: dict[str, pd.DataFrame] = {}
        if self.native_tf == "daily":
            frames["daily"] = df.copy()
            frames["weekly"] = resample_ohlcv(df, "W-FRI")
            frames["monthly"] = resample_ohlcv(df, "ME")
        else:
            frames["weekly"] = df.copy()
            frames["monthly"] = resample_ohlcv(df, "ME")
        return frames

    def enrich_timeframes(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        enriched: dict[str, pd.DataFrame] = {}
        for tf, frame in frames.items():
            if frame is None or frame.empty:
                continue
            include_slow = tf != "monthly"
            enriched[tf] = add_indicators(
                frame,
                pctl_window=PCTL_WINDOW[tf],
                include_sma200=include_slow,
            )
        return enriched

    def extract(
        self,
        df: pd.DataFrame,
        symbol: str,
        scan_mode: str,
    ) -> BreakoutFeatures:
        """Clean → resample → indicate → last-bar features."""
        clean = normalize_ohlcv(df)
        if len(clean) < MIN_PRIMARY_BARS:
            raise ValueError(
                f"insufficient bars: {len(clean)} < {MIN_PRIMARY_BARS}"
            )

        frames = self.enrich_timeframes(self.create_timeframes(clean))
        if self.native_tf not in frames:
            raise ValueError("native timeframe missing after enrichment")

        primary = frames[self.native_tf]
        last = primary.iloc[-1]
        weekly = frames.get("weekly")
        monthly = frames.get("monthly")
        daily = frames.get("daily")

        obv = _safe_float(last.get("OBV"))
        obv_sma = _safe_float(last.get("OBV_SMA20"))
        obv_rising = None if obv is None or obv_sma is None else obv > obv_sma

        return BreakoutFeatures(
            symbol=symbol.upper(),
            mode=scan_mode,
            asof=_asof_str(last.get("Date")),
            close=_safe_float(last.get("Close"), 4),
            sma20=_safe_float(last.get("SMA20"), 4),
            sma50=_safe_float(last.get("SMA50"), 4),
            sma200=_safe_float(last.get("SMA200"), 4),
            rsi=_safe_float(last.get("RSI"), 4),
            atr=_safe_float(last.get("ATR"), 4),
            resistance=_safe_float(last.get("Resistance"), 4),
            support=_safe_float(last.get("Support"), 4),
            bb_width_pctl=_safe_float(last.get("BB Width Pctl"), 2),
            kc_width_pctl=_safe_float(last.get("KC Width Pctl"), 2),
            atr_pctl=_safe_float(last.get("ATR Pctl"), 2),
            range20_pctl=_safe_float(last.get("Range20 Pctl"), 2),
            rvol20=_safe_float(last.get("RVOL20"), 4),
            rsi_slope=_safe_float(last.get("RSI Slope"), 4),
            macd_hist_slope=_safe_float(last.get("MACD Hist Slope"), 6),
            macd_hist_accel=_safe_float(last.get("MACD Hist Accel"), 6),
            distance_resistance_atr=_safe_float(last.get("Distance Resistance ATR"), 4),
            extension_atr=_safe_float(last.get("Extension ATR"), 4),
            range20_atr=_safe_float(last.get("Range20 / ATR"), 4),
            daily_trend_bull=_trend_bull(daily) if daily is not None else None,
            weekly_trend_bull=_trend_bull(weekly) if weekly is not None else None,
            monthly_trend_bull=_trend_bull(monthly, require_sma50=False) if monthly is not None else None,
            daily_trend_bear=_trend_bear(daily) if daily is not None else None,
            weekly_trend_bear=_trend_bear(weekly),
            monthly_trend_bear=_trend_bear(monthly),
            squeeze=_safe_bool(last.get("Squeeze")),
            compression=_safe_bool(last.get("Compression")),
            volume_dryup=_safe_bool(last.get("Volume Dryup")),
            obv_rising=obv_rising,
            breakout_triggered=_safe_bool(last.get("Breakout Triggered")),
            breakout_confirmation=_safe_bool(last.get("Breakout Confirmation")),
            failed_breakout=_safe_bool(last.get("Failed Breakout")),
            wick_reject=_safe_bool(last.get("Wick Reject")),
            swing_high=_safe_bool(last.get("Swing High")),
            swing_low=_safe_bool(last.get("Swing Low")),
            macd_hist=_safe_float(last.get("MACD_Hist"), 6),
            has_daily=daily is not None,
        )


# =========================
# SCORING / CLASSIFICATION ENGINE
# =========================

def _pctl_points(pctl: float | None, full: int, mid: int, bands: tuple[float, float]) -> int:
    """Lower percentile = more compressed. ``bands`` is (full_cut, partial_cut)."""
    if pctl is None:
        return 0
    if pctl <= bands[0]:
        return full
    if pctl <= bands[1]:
        return mid
    if pctl <= 50:
        return max(1, mid // 2)
    return 0


def _mtf_label(feat: BreakoutFeatures) -> str:
    flags = []
    bears = []
    if feat.has_daily:
        flags.append(feat.daily_trend_bull)
        bears.append(feat.daily_trend_bear)
    flags.append(feat.weekly_trend_bull)
    flags.append(feat.monthly_trend_bull)
    bears.append(feat.weekly_trend_bear)
    bears.append(feat.monthly_trend_bear)

    known = [f for f in flags if f is not None]
    if len(known) < 2:
        return "INSUFFICIENT"
    if all(known):
        return "ALIGNED"
    if any(bears) and any(known):
        # Primary (or any) bull against a higher-TF bear.
        if any(known) and any(b is True for b in bears):
            return "DIVERGENT"
    if any(known):
        return "PARTIAL"
    return "DIVERGENT"


def _primary_trend_label(feat: BreakoutFeatures) -> str:
    primary_bull = feat.daily_trend_bull if feat.has_daily else feat.weekly_trend_bull
    primary_bear = feat.daily_trend_bear if feat.has_daily else feat.weekly_trend_bear
    if primary_bull:
        return "BULLISH"
    if primary_bear:
        return "BEARISH"
    return "NEUTRAL"


def _volatility_label(feat: BreakoutFeatures) -> str:
    if feat.compression:
        return "COMPRESSING"
    expanding = (
        (feat.bb_width_pctl is not None and feat.bb_width_pctl >= 70)
        or (feat.atr_pctl is not None and feat.atr_pctl >= 70)
    )
    if expanding:
        return "EXPANDING"
    return "NORMAL"


def _volume_label(feat: BreakoutFeatures) -> str:
    if feat.volume_dryup or (feat.rvol20 is not None and feat.rvol20 < RVOL_DRYUP):
        return "DRYING_UP"
    if feat.rvol20 is not None and feat.rvol20 >= 1.20:
        return "EXPANDING"
    return "NORMAL"


def _momentum_label(feat: BreakoutFeatures) -> str:
    rsi_up = feat.rsi_slope is not None and feat.rsi_slope > 0
    hist_up = feat.macd_hist_slope is not None and feat.macd_hist_slope > 0
    accel = feat.macd_hist_accel is not None and feat.macd_hist_accel >= 0
    if rsi_up and hist_up and accel:
        return "ACCELERATING"
    if (feat.rsi_slope is not None and feat.rsi_slope < 0) and (
        feat.macd_hist_slope is not None and feat.macd_hist_slope < 0
    ):
        return "FADING"
    if rsi_up or hist_up:
        return "ACCELERATING" if (rsi_up and hist_up) else "NEUTRAL"
    return "NEUTRAL"


def _structure_label(feat: BreakoutFeatures) -> str:
    dist = feat.distance_resistance_atr
    close = feat.close
    support = feat.support
    if close is not None and support is not None and close < support:
        return "BELOW_SUPPORT"
    if dist is None:
        return "UNKNOWN"
    if dist < 0:
        return "ABOVE_RESISTANCE"
    if dist <= AT_RESISTANCE_ATR:
        return "AT_RESISTANCE"
    return "UNDER_RESISTANCE"


def _distance_label(feat: BreakoutFeatures) -> str:
    if feat.distance_resistance_atr is None:
        return ""
    return f"{feat.distance_resistance_atr:.1f} ATR"


def classify_states(feat: BreakoutFeatures) -> dict[str, str]:
    """State labels — the primary output of the scanner."""
    return {
        "Trend": _primary_trend_label(feat),
        "Volatility": _volatility_label(feat),
        "Volume State": _volume_label(feat),
        "Momentum": _momentum_label(feat),
        "Structure": _structure_label(feat),
        "Breakout Distance": _distance_label(feat),
        "MTF": _mtf_label(feat),
    }


def classify_status(feat: BreakoutFeatures, states: dict[str, str]) -> str:
    """Setup class from states/features. Score is not an input on purpose."""
    if feat.failed_breakout:
        return STATUS_FAILED

    if feat.breakout_confirmation:
        return STATUS_CONFIRMED

    dist = feat.distance_resistance_atr
    near = dist is not None and 0.0 <= dist <= PRE_BREAKOUT_ATR_MAX
    under = states["Structure"] in {"UNDER_RESISTANCE", "AT_RESISTANCE"}
    mtf_ok = states["MTF"] in {"ALIGNED", "PARTIAL"}
    trend_ok = states["Trend"] == "BULLISH"
    compressing = states["Volatility"] == "COMPRESSING"
    accelerating = states["Momentum"] == "ACCELERATING"
    volume_ok = states["Volume State"] in {"DRYING_UP", "NORMAL"} or (
        states["Volume State"] == "EXPANDING"
        and feat.rvol20 is not None
        and feat.rvol20 < RVOL_CONFIRM
    )
    not_through = not bool(feat.breakout_triggered)

    if (
        not_through
        and trend_ok
        and compressing
        and accelerating
        and under
        and near
        and mtf_ok
        and volume_ok
    ):
        return STATUS_PRE

    if feat.breakout_triggered and not feat.breakout_confirmation:
        return STATUS_WATCH

    early_coil = compressing or (
        feat.bb_width_pctl is not None and feat.bb_width_pctl <= 40
    )
    still_far = dist is not None and dist > PRE_BREAKOUT_ATR_MAX
    if trend_ok and early_coil and (accelerating or still_far) and not_through:
        if still_far:
            return STATUS_DEV
        return STATUS_WATCH

    if near and trend_ok and not_through:
        return STATUS_WATCH

    return STATUS_WATCH


def score_readiness(feat: BreakoutFeatures, states: dict[str, str]) -> dict:
    """100-point Breakout Readiness plus per-factor scores for later tuning.

    Pillars (before penalties): Trend 25, Compression 25, Momentum 20,
    Volume 15, Structure 15. When daily bars are unavailable the trend
    budget is reallocated to weekly (15) + monthly (10).
    """
    if feat.has_daily:
        daily_cap, weekly_cap, monthly_cap = 10, 8, 7
    else:
        daily_cap, weekly_cap, monthly_cap = 0, 15, 10

    daily_pts = 0
    if feat.has_daily:
        if feat.daily_trend_bull:
            daily_pts = daily_cap
        elif feat.close is not None and feat.sma50 is not None and feat.close > feat.sma50:
            daily_pts = 6 if (feat.sma200 is None or feat.close > feat.sma200) else 3

    if feat.weekly_trend_bull:
        weekly_pts = weekly_cap
    elif feat.weekly_trend_bull is False and feat.weekly_trend_bear is False:
        weekly_pts = weekly_cap // 2
    else:
        weekly_pts = 0

    if feat.monthly_trend_bull:
        monthly_pts = monthly_cap
    elif feat.monthly_trend_bull is False and feat.monthly_trend_bear is False:
        monthly_pts = monthly_cap // 2
    else:
        monthly_pts = 0

    trend_score = daily_pts + weekly_pts + monthly_pts

    bb_pts = _pctl_points(feat.bb_width_pctl, 8, 6, (15, 25))
    atr_pts = _pctl_points(feat.atr_pctl, 6, 4, (20, 35))
    range_pts = _pctl_points(feat.range20_pctl, 5, 3, (20, 35))
    squeeze_pts = 6 if feat.squeeze else (3 if feat.kc_width_pctl is not None and feat.kc_width_pctl <= 30 else 0)
    compression_score = bb_pts + atr_pts + range_pts + squeeze_pts

    rsi = feat.rsi
    if rsi is None:
        rsi_pts = 0
    elif 50 <= rsi <= 65:
        rsi_pts = 6
    elif 45 <= rsi < 50 or 65 < rsi <= 70:
        rsi_pts = 4
    elif 40 <= rsi < 45:
        rsi_pts = 2
    elif rsi > 75:
        rsi_pts = 0
    else:
        rsi_pts = 1

    if feat.rsi_slope is None:
        rsi_slope_pts = 0
    elif feat.rsi_slope >= 2:
        rsi_slope_pts = 7
    elif feat.rsi_slope > 0:
        rsi_slope_pts = 5
    else:
        rsi_slope_pts = 0

    hist = feat.macd_hist
    hist_slope = feat.macd_hist_slope
    if hist_slope is None:
        macd_pts = 0
    elif hist_slope > 0 and hist is not None and hist > 0:
        macd_pts = 7
    elif hist_slope > 0:
        macd_pts = 5
    else:
        macd_pts = 0
    momentum_score = rsi_pts + rsi_slope_pts + macd_pts

    dry_pts = 8 if feat.volume_dryup else (
        4 if feat.rvol20 is not None and feat.rvol20 < 1.0 else 0
    )
    # Volume quality is regime-aware: dry-up is the pre-breakout ideal;
    # expansion is the confirmation ideal.
    if feat.breakout_triggered:
        if feat.rvol20 is not None and feat.rvol20 >= 2.0:
            rvol_pts = 7
        elif feat.rvol20 is not None and feat.rvol20 >= RVOL_CONFIRM:
            rvol_pts = 6
        elif feat.rvol20 is not None and feat.rvol20 >= RVOL_IGNITION:
            rvol_pts = 3
        else:
            rvol_pts = 0
        if feat.obv_rising:
            rvol_pts = min(7, rvol_pts + 1)
        dry_pts = min(dry_pts, 3)
    else:
        if feat.obv_rising:
            rvol_pts = 7 if (feat.rvol20 or 0) < 1.2 else 5
        elif feat.rvol20 is not None and RVOL_IGNITION <= feat.rvol20 < RVOL_CONFIRM:
            rvol_pts = 4
        else:
            rvol_pts = 2 if feat.volume_dryup else 0
    volume_score = dry_pts + rvol_pts

    dist = feat.distance_resistance_atr
    if dist is None:
        prox_pts = 0
    elif 0.0 <= dist <= 0.75:
        prox_pts = 10
    elif 0.75 < dist <= 1.50:
        prox_pts = 7
    elif 1.50 < dist <= 2.50:
        prox_pts = 4
    elif dist < 0 and dist >= -0.35:
        prox_pts = 5
    elif dist < 0:
        prox_pts = 2
    else:
        prox_pts = 1

    ext = feat.extension_atr
    if ext is None:
        ext_pts = 0
    elif 0.0 <= ext <= 1.50:
        ext_pts = 5
    elif 1.50 < ext <= 2.00:
        ext_pts = 3
    elif ext < 0:
        ext_pts = 2
    else:
        ext_pts = 0
    structure_score = prox_pts + ext_pts

    raw = trend_score + compression_score + momentum_score + volume_score + structure_score

    penalty = 0
    if feat.failed_breakout:
        penalty += 20 if feat.breakout_triggered is False else 10
    elif feat.wick_reject:
        penalty += 8
    if states["MTF"] == "DIVERGENT":
        penalty += 10
    if ext is not None and ext > 2.50:
        penalty += 10
    if rsi is not None and rsi > 75:
        penalty += 5
    if states["Trend"] == "BEARISH":
        penalty += 10

    readiness = int(max(0, min(100, raw - penalty)))

    return {
        "Breakout Readiness": readiness,
        "Trend Score": trend_score,
        "Compression Score": compression_score,
        "Momentum Score": momentum_score,
        "Volume Score": volume_score,
        "Structure Score": structure_score,
        "Penalty": penalty,
        "Daily Trend Score": daily_pts,
        "Weekly Trend Score": weekly_pts,
        "Monthly Trend Score": monthly_pts,
        "BB Width Score": bb_pts,
        "ATR Pctl Score": atr_pts,
        "Range20 Score": range_pts,
        "Squeeze Score": squeeze_pts,
        "RSI Score": rsi_pts,
        "RSI Slope Score": rsi_slope_pts,
        "MACD Hist Slope Score": macd_pts,
        "Volume Dryup Score": dry_pts,
        "RVOL Score": rvol_pts,
        "Proximity Score": prox_pts,
        "Extension Score": ext_pts,
    }


class ScoringEngine:
    """Classify market states and attach a 100-point readiness score.

    Classification is state-driven. The numeric score is recorded for
    later weight tuning and is not the source of truth.
    """

    def evaluate(self, feat: BreakoutFeatures) -> dict:
        states = classify_states(feat)
        status = classify_status(feat, states)
        scores = score_readiness(feat, states)
        row = feat.feature_row()
        row["MTF Alignment"] = states["MTF"] == "ALIGNED"
        row.update(states)
        row["Status"] = status
        row.update(scores)
        return row


# =========================
# PRESENTATION
# =========================

def _format_table(df: pd.DataFrame) -> str:
    """Markdown table when tabulate is installed; plain text otherwise."""
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError, ValueError, OSError, AttributeError):
        return df.to_string(index=False)


def _is_display_candidate(row: dict) -> bool:
    status = row.get("Status")
    if status in CANDIDATE_STATUSES:
        return True
    floor = DISPLAY_SCORE_FLOOR.get(status)
    if floor is None:
        return False
    score = row.get("Breakout Readiness")
    return score is not None and score >= floor


def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rank = {
        STATUS_PRE: 0,
        STATUS_CONFIRMED: 1,
        STATUS_FAILED: 2,
        STATUS_WATCH: 3,
        STATUS_DEV: 4,
    }
    out = df.copy()
    out["_rank"] = out["Status"].map(rank).fillna(9)
    out = out.sort_values(
        ["_rank", "Breakout Readiness", "Symbol"],
        ascending=[True, False, True],
    )
    return out.drop(columns="_rank")


# =========================
# PUBLIC API
# =========================

def evaluate_ticker(
    df: pd.DataFrame,
    symbol: str,
    scan_mode: str | None = None,
    native_tf: str | None = None,
) -> dict:
    """Feature extraction then scoring for a single OHLCV frame (last bar)."""
    scan_mode = scan_mode or mode
    native_tf = native_tf or _native_timeframe(_data_mode)
    features = FeatureEngine(native_tf).extract(df, symbol, scan_mode)
    return ScoringEngine().evaluate(features)


def scan_files(
    raw_files: Iterable[str],
    active_tickers: set[str],
    raw_dir: str,
    scan_mode: str,
    native_tf: str,
) -> tuple[list[dict], dict[str, int]]:
    """Walk raw CSVs and return (rows, rejection_counts)."""
    engine = FeatureEngine(native_tf)
    scorer = ScoringEngine()
    results: list[dict] = []
    rejection_counts: dict[str, int] = {}

    for file in raw_files:
        symbol = file.split("_")[0].upper()
        if symbol not in active_tickers:
            rejection_counts["inactive_ticker"] = rejection_counts.get("inactive_ticker", 0) + 1
            continue

        path = os.path.join(raw_dir, file)
        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            rejection_counts["read_error"] = rejection_counts.get("read_error", 0) + 1
            continue

        try:
            features = engine.extract(raw, symbol, scan_mode)
            results.append(scorer.evaluate(features))
        except ValueError as exc:
            reason = str(exc)
            if "insufficient" in reason:
                key = "insufficient_data"
            elif "empty" in reason or "usable" in reason or "Missing required" in reason:
                key = "missing_columns"
            else:
                key = "invalid_ohlcv"
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
        except Exception as exc:
            logger.error("Error scoring %s: %s", symbol, exc)
            rejection_counts["execution_error"] = rejection_counts.get("execution_error", 0) + 1

    return results, rejection_counts


def run_scanner() -> pd.DataFrame:
    """Scan the active universe and archive feature + state rows."""
    native_tf = _native_timeframe(_data_mode)
    logger.info(
        "--- Breakout Readiness Scan [%s MODE | native=%s] ---",
        mode.upper(),
        native_tf,
    )

    if not os.path.exists(ACTIVE_TICKERS_PATH):
        logger.error("Missing active tickers inventory file at %s", ACTIVE_TICKERS_PATH)
        sys.exit(1)

    active_tickers = set(pd.read_csv(ACTIVE_TICKERS_PATH)["Ticker"].str.upper())
    logger.info("Loaded %s active tickers", len(active_tickers))

    if not os.path.exists(RAW_DATA_DIR):
        logger.warning("Target raw directory empty or non-existent: %s", RAW_DATA_DIR)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    logger.info("Found %s raw data files in target silo", len(raw_files))

    results, rejection_counts = scan_files(
        raw_files, active_tickers, RAW_DATA_DIR, mode, native_tf,
    )

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(LOG_DIR, f"breakout_setups_{today}.csv")
    df_out = pd.DataFrame(results)
    if df_out.empty:
        df_out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        df_out = df_out.reindex(columns=OUTPUT_COLUMNS)
        df_out = _sort_candidates(df_out)

    df_out.to_csv(out_path, index=False)
    logger.info("Archive created: %s (%s row(s))", out_path, len(df_out))

    display = df_out[df_out.apply(lambda r: _is_display_candidate(r.to_dict()), axis=1)] if not df_out.empty else df_out
    if display.empty:
        logger.warning("No breakout setup candidates to display for this window.")
    else:
        view_cols = [c for c in DISPLAY_COLUMNS if c in display.columns]
        print("\n" + _format_table(display[view_cols]) + "\n")
        logger.info("Displayed %s candidate(s) of %s scanned row(s).", len(display), len(df_out))

    logger.info("Scanner rejection summary:")
    for key, value in rejection_counts.items():
        logger.info("  %s: %s", key, value)

    return df_out


if __name__ == "__main__":
    run_scanner()
