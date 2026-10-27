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

Usage::

    python -m finance_vibe.scanners.breakout weekly
    python -m finance_vibe.scanners.breakout daily
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance_vibe import config, indicators
from finance_vibe.data import raw
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)

# =========================
# MODE AND PATHS (set by set_mode(); the module imports as weekly)
# =========================
ACTIVE_TICKERS_PATH = config.TICKER_LIST_PATH
mode: str
_data_mode: str
RAW_DATA_DIR: str
LOG_DIR: str


def set_mode(new_mode: str) -> str:
    """Select the scan mode and its raw/log paths; unknown modes fall back to weekly."""
    global mode, _data_mode, RAW_DATA_DIR, LOG_DIR
    _data_mode, mode = config.resolve_pipeline_mode(new_mode)
    RAW_DATA_DIR = config.TIMEFRAME_PROFILES[_data_mode]["raw_dir"]
    LOG_DIR = config.TIMEFRAME_PROFILES[mode]["logs_dir"]
    return mode


set_mode("weekly")

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
RVOL_CONFIRM = config.BREAKOUT_RVOL_CONFIRM
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
STATUS_TRIGGERED = "BREAKOUT_TRIGGERED"
STATUS_FAILED = "FAILED_BREAKOUT"

CANDIDATE_STATUSES = {STATUS_PRE, STATUS_CONFIRMED, STATUS_TRIGGERED, STATUS_FAILED}
# WATCH (the "nothing happening" catch-all) is never displayed; the CSV keeps it.
DISPLAY_SCORE_FLOOR = {
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
    computed = indicators.sma(series, length=length)
    if computed is None:
        return series.rolling(length, min_periods=length).mean()
    return _series_or_nan(computed, series.index)


def _native_timeframe(data_mode: str) -> str:
    return "daily" if data_mode == "daily" else "weekly"


def _scalar(value):
    """A feature-frame cell as BreakoutFeatures holds it: None, bool, float or str."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


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


def add_indicators(
    df: pd.DataFrame, *, pctl_window: int, include_sma200: bool = True
) -> pd.DataFrame:
    """Vectorized indicator stack. Every rolling window is causal (past + now).

    Structure levels use a 1-bar shift so the current print cannot be its
    own 20-bar resistance/support (no same-bar lookahead).
    """
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = (
        out["Volume"].astype(float)
        if "Volume" in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    out["SMA20"] = _sma(close, SMA_FAST)
    out["SMA50"] = _sma(close, SMA_MID)
    if include_sma200:
        out["SMA200"] = _sma(close, SMA_SLOW)
    else:
        out["SMA200"] = np.nan

    rsi = indicators.rsi(close, length=RSI_LEN)
    out["RSI"] = _series_or_nan(rsi, out.index)

    macd = indicators.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    macd_line = _ta_col(macd, "MACD_15_30_9")
    macd_signal = _ta_col(macd, "MACDs_15_30_9", "MACDs_")
    macd_hist = _ta_col(macd, "MACDh_15_30_9", "MACDh_")
    out["MACD"] = _series_or_nan(macd_line, out.index)
    out["MACD_Signal"] = _series_or_nan(macd_signal, out.index)
    out["MACD_Hist"] = _series_or_nan(macd_hist, out.index)

    # Sample std (ddof=1), as pandas-ta computed it. Deliberately kept: widths are
    # only compared with their own percentile history, so ddof just rescales.
    bb = indicators.bbands(close, length=BB_LEN, num_std=BB_STD)
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

    kc = indicators.kc(high, low, close, length=KC_LEN, scalar=KC_SCALAR)
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

    out["ATR"] = _series_or_nan(indicators.atr(high, low, close, length=ATR_LEN), out.index)

    vol_sma = _sma(volume, RVOL_LEN)
    out["VOL_SMA20"] = vol_sma
    out["RVOL20"] = volume / vol_sma.replace(0, np.nan)

    vol_fast = volume.rolling(VOL_DRYUP_FAST, min_periods=VOL_DRYUP_FAST).mean()
    out["VOL_SMA5"] = vol_fast
    out["Volume Dryup"] = (vol_fast / vol_sma.replace(0, np.nan)) < VOL_DRYUP_RATIO

    out["OBV"] = _series_or_nan(indicators.obv(close, volume), out.index)
    out["OBV_SMA20"] = _sma(out["OBV"], RVOL_LEN)

    out["Range20"] = (
        high.rolling(RANGE_BARS, min_periods=RANGE_BARS).max()
        - low.rolling(RANGE_BARS, min_periods=RANGE_BARS).min()
    )

    # Prior-bar structure — current high cannot define its own breakout level.
    out["Resistance"] = high.rolling(STRUCTURE_BARS, min_periods=STRUCTURE_BARS).max().shift(1)
    out["Support"] = low.rolling(STRUCTURE_BARS, min_periods=STRUCTURE_BARS).min().shift(1)

    # Confirmed 3-bar swing pivots (pivot at i-1, confirmed on bar i).

    atr = out["ATR"].replace(0, np.nan)
    out["Distance Resistance ATR"] = (out["Resistance"] - close) / atr
    out["Extension ATR"] = (close - out["SMA20"]) / atr

    out["RSI Slope"] = out["RSI"].diff(SLOPE_RSI_BARS)
    out["MACD Hist Slope"] = out["MACD_Hist"].diff(SLOPE_MACD_BARS)

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
        (out["BB Width Pctl"] <= 30) & (out["ATR Pctl"] <= 40) & (out["Range20 Pctl"] <= 40)
    )

    above_res = (close > out["Resistance"]).fillna(False)
    out["Breakout Triggered"] = above_res
    wick_reject = ((high > out["Resistance"]) & (close < out["Resistance"])).fillna(False)
    # The level a breakout cleared is the resistance on the first bar of the
    # run. Resistance itself rolls up to include the breakout bar's high the
    # next day, so "not above resistance" alone is not a failure: the close
    # has to fall back below the level that was actually broken.
    run_start = above_res & ~above_res.shift(1, fill_value=False)
    out["Breakout Level"] = out["Resistance"].where(run_start).ffill()
    recently_above = (
        above_res.shift(1).rolling(FAKEOUT_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    lost_level = (recently_above & ~above_res & (close < out["Breakout Level"])).fillna(False)
    out["Wick Reject"] = wick_reject
    out["Failed Breakout"] = lost_level
    out["Breakout Confirmation"] = (
        above_res & (out["RVOL20"] >= RVOL_CONFIRM) & (close > out["SMA20"])
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


def _trend_bear(df: pd.DataFrame | None) -> bool | None:
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    sma20 = _safe_float(last.get("SMA20"))
    sma50 = _safe_float(last.get("SMA50"))
    if close is None or sma20 is None or sma50 is None:
        return None
    return bool(close < sma20 < sma50)


# Research CSV feature columns → BreakoutFeatures fields (after the identity columns).
FEATURE_COLUMNS = {
    "Close": "close",
    "SMA20": "sma20",
    "SMA50": "sma50",
    "SMA200": "sma200",
    "RSI": "rsi",
    "ATR": "atr",
    "Resistance": "resistance",
    "Support": "support",
    "BB Width Pctl": "bb_width_pctl",
    "KC Width Pctl": "kc_width_pctl",
    "ATR Pctl": "atr_pctl",
    "Range20 Pctl": "range20_pctl",
    "RVOL20": "rvol20",
    "RSI Slope": "rsi_slope",
    "MACD Hist Slope": "macd_hist_slope",
    "Distance Resistance ATR": "distance_resistance_atr",
    "Extension ATR": "extension_atr",
    "Daily Trend Bull": "daily_trend_bull",
    "Weekly Trend Bull": "weekly_trend_bull",
    "Monthly Trend Bull": "monthly_trend_bull",
    "Compression": "compression",
    "Breakout Triggered": "breakout_triggered",
    "Breakout Confirmation": "breakout_confirmation",
    "Failed Breakout": "failed_breakout",
}


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
    distance_resistance_atr: float | None
    extension_atr: float | None
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
    macd_hist: float | None
    has_daily: bool = False

    def feature_row(self) -> dict:
        """Underlying features for the research CSV (no scores)."""
        row = {
            "Symbol": self.symbol,
            "Mode": self.mode,
            "Source": "breakout",
            "AsOf Date": self.asof,
        }
        row.update({col: getattr(self, field) for col, field in FEATURE_COLUMNS.items()})
        return row


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
        """Clean → indicate → the last row of the per-bar feature frame."""
        clean = normalize_ohlcv(df)
        if len(clean) < MIN_PRIMARY_BARS:
            raise ValueError(f"insufficient bars: {len(clean)} < {MIN_PRIMARY_BARS}")

        last = _feature_frame(clean, self.native_tf).iloc[-1]
        values = {
            f.name: _scalar(last[f.name])
            for f in dataclasses.fields(BreakoutFeatures)
            if f.name not in ("symbol", "mode")
        }
        return BreakoutFeatures(symbol=symbol.upper(), mode=scan_mode, **values)


# =========================
# PER-BAR FEATURE FRAME (one row per native bar, for scan_history)
# =========================

# BreakoutFeatures float fields: (indicator column, digits extract() rounds to).
# The rules compare these *rounded* values with their thresholds.
_ROUNDED_FEATURES: dict[str, tuple[str, int]] = {
    "close": ("Close", 4),
    "sma20": ("SMA20", 4),
    "sma50": ("SMA50", 4),
    "sma200": ("SMA200", 4),
    "rsi": ("RSI", 4),
    "atr": ("ATR", 4),
    "resistance": ("Resistance", 4),
    "support": ("Support", 4),
    "bb_width_pctl": ("BB Width Pctl", 2),
    "kc_width_pctl": ("KC Width Pctl", 2),
    "atr_pctl": ("ATR Pctl", 2),
    "range20_pctl": ("Range20 Pctl", 2),
    "rvol20": ("RVOL20", 4),
    "rsi_slope": ("RSI Slope", 4),
    "macd_hist_slope": ("MACD Hist Slope", 6),
    "distance_resistance_atr": ("Distance Resistance ATR", 4),
    "extension_atr": ("Extension ATR", 4),
    "macd_hist": ("MACD_Hist", 6),
}

_BOOL_FEATURES: dict[str, str] = {
    "squeeze": "Squeeze",
    "compression": "Compression",
    "volume_dryup": "Volume Dryup",
    "breakout_triggered": "Breakout Triggered",
    "breakout_confirmation": "Breakout Confirmation",
    "failed_breakout": "Failed Breakout",
    "wick_reject": "Wick Reject",
}


def _round_like_builtin(values, digits: int) -> np.ndarray:
    """Vectorized ``round(x, digits)`` that matches the builtin bit for bit; ±inf → NaN.

    ``np.round`` (scale, rint, unscale) agrees with the builtin except near a
    .5 tie of the scaled value, where scaling error can flip the direction;
    those few elements fall back to the builtin.
    """
    x = np.array(values, dtype="float64")
    x[~np.isfinite(x)] = np.nan
    out = np.round(x, digits)
    scaled = x * 10.0**digits
    risky = np.isfinite(x) & (
        (np.abs(scaled - np.floor(scaled) - 0.5) < 1e-6) | (np.abs(scaled) >= 2.0**52)
    )
    for i in np.flatnonzero(risky):
        out[i] = round(float(x[i]), digits)
    return out


def _nullable(values: np.ndarray, known: np.ndarray, index: pd.Index) -> pd.Series:
    """bool values → pandas ``boolean`` Series, NA where not ``known`` (the scalar ``None``)."""
    arr = pd.arrays.BooleanArray(values.astype(bool), ~known.astype(bool))
    return pd.Series(arr, index=index)


def _trend_flags(
    close: np.ndarray,
    sma20: np.ndarray,
    sma50: np.ndarray,
    sma200: np.ndarray,
    prev_sma20: np.ndarray,
    *,
    require_sma50: bool,
    index: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    """Per-bar bull / bear flags: Close > SMA20 > SMA50 (> SMA200 when known), rising SMA20.

    ``prev_sma20`` is NaN where there is no previous bar. Without SMA50,
    ``require_sma50=False`` (monthly) falls back to Close > rising SMA20.
    NA means unknown (not enough history).
    """
    with np.errstate(invalid="ignore"):
        has50 = ~np.isnan(sma50)
        bull = close > sma20
        bull &= ~has50 | ((sma20 > sma50) & (close > sma50))
        bull &= np.isnan(sma200) | (close > sma200)
        bull &= np.isnan(prev_sma20) | (sma20 >= prev_sma20)
        bear = (close < sma20) & (sma20 < sma50)
    known = ~np.isnan(close) & ~np.isnan(sma20)
    bull_known = known & has50 if require_sma50 else known
    return _nullable(bull, bull_known, index), _nullable(bear, known & has50, index)


def _htf_periods(clean: pd.DataFrame, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """Higher-timeframe closes and, per native bar, the index of its HTF period.

    Periods come from :func:`resample_ohlcv` itself (bins are labelled at the
    period end), so week/month bounds are defined in one place.
    """
    htf = resample_ohlcv(clean, rule)
    closes = htf["Close"].to_numpy(dtype="float64")
    pid = np.searchsorted(htf["Date"].to_numpy(), clean["Date"].to_numpy(), side="left")
    return closes, pid


def _forming_smas(
    closes: np.ndarray, pid: np.ndarray, native_close: np.ndarray, lengths: Iterable[int]
) -> dict[int, np.ndarray]:
    """SMA(n) of the HTF bar as it was forming at each native bar, bit-exact.

    At native bar t in period k the HTF frame cut at t is ``closes[:k]`` plus
    a forming bar whose close is t's close. Its SMA must equal
    ``indicators.sma`` on that cut frame exactly (ties in the trend flags are
    real: equal closes give equal SMAs). ``indicators.sma`` is a convolution
    whose full-overlap outputs depend only on the window, so every window is
    laid end to end, one pad value apart, and convolved once. A cut frame of
    exactly n bars hits numpy's reversed single-window sum; those rows repeat
    that exact call.
    """
    out: dict[int, np.ndarray] = {}
    for n in lengths:
        sma = np.full(len(native_close), np.nan)
        rows = np.flatnonzero(pid >= n - 1)
        if rows.size:
            k = pid[rows]
            kernel = np.ones(n) / n
            windows = np.zeros((rows.size, n + 1))
            windows[:, 1:n] = closes[k[:, None] + np.arange(1 - n, 0)]
            windows[:, n] = native_close[rows]
            full = np.convolve(kernel, windows.ravel())
            sma[rows] = full[np.arange(rows.size) * (n + 1) + n]
            for j in np.flatnonzero(k == n - 1):
                sma[rows[j]] = np.convolve(kernel, windows[j, 1:])[n - 1]
        out[n] = sma
    return out


def _forming_htf_smas(
    clean: pd.DataFrame, rule: str, lengths: Iterable[int]
) -> dict[int, np.ndarray]:
    closes, pid = _htf_periods(clean, rule)
    return _forming_smas(closes, pid, clean["Close"].to_numpy(dtype="float64"), lengths)


def _forming_htf_trend(
    clean: pd.DataFrame, rule: str, *, require_sma50: bool, include_sma200: bool
) -> tuple[pd.Series, pd.Series]:
    """HTF bull / bear flags per native bar, from the HTF bar forming at that bar."""
    closes, pid = _htf_periods(clean, rule)
    close = clean["Close"].to_numpy(dtype="float64")
    lengths = (SMA_FAST, SMA_MID, SMA_SLOW) if include_sma200 else (SMA_FAST, SMA_MID)
    smas = _forming_smas(closes, pid, close, lengths)
    completed_sma20 = _sma(pd.Series(closes), SMA_FAST).to_numpy()
    prev_sma20 = np.where(pid >= 1, completed_sma20[np.maximum(pid - 1, 0)], np.nan)
    sma200 = smas[SMA_SLOW] if include_sma200 else np.full(len(close), np.nan)
    return _trend_flags(
        close,
        smas[SMA_FAST],
        smas[SMA_MID],
        sma200,
        prev_sma20,
        require_sma50=require_sma50,
        index=clean.index,
    )


def _native_trend(primary: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Bull / bear flags on the native frame (each bar's previous bar is complete)."""
    sma20 = primary["SMA20"].to_numpy(dtype="float64")
    return _trend_flags(
        primary["Close"].to_numpy(dtype="float64"),
        sma20,
        primary["SMA50"].to_numpy(dtype="float64"),
        primary["SMA200"].to_numpy(dtype="float64"),
        np.r_[np.nan, sma20[:-1]],
        require_sma50=True,
        index=primary.index,
    )


def _feature_frame(clean: pd.DataFrame, native_tf: str) -> pd.DataFrame:
    """Every native bar's :class:`BreakoutFeatures`, one row per bar (no lookahead).

    Row t holds what ``FeatureEngine.extract`` returns on data cut at bar t:
    native indicators are causal, and weekly/monthly trend flags use the
    higher-timeframe bar as it was forming at t. Columns are the dataclass
    field names (minus symbol/mode) plus ``date``.
    """
    primary = add_indicators(clean, pctl_window=PCTL_WINDOW[native_tf], include_sma200=True)
    index = primary.index
    frame = pd.DataFrame(
        {"date": primary["Date"], "asof": primary["Date"].dt.strftime("%Y-%m-%d")}, index=index
    )
    for field, (col, digits) in _ROUNDED_FEATURES.items():
        frame[field] = _round_like_builtin(primary[col], digits)

    na = _nullable(np.zeros(len(index), dtype=bool), np.zeros(len(index), dtype=bool), index)
    native_bull, native_bear = _native_trend(primary)
    if native_tf == "daily":
        frame["daily_trend_bull"], frame["daily_trend_bear"] = native_bull, native_bear
        weekly = _forming_htf_trend(clean, "W-FRI", require_sma50=True, include_sma200=True)
    else:
        frame["daily_trend_bull"], frame["daily_trend_bear"] = na, na.copy()
        weekly = native_bull, native_bear
    frame["weekly_trend_bull"], frame["weekly_trend_bear"] = weekly
    monthly = _forming_htf_trend(clean, "ME", require_sma50=False, include_sma200=False)
    frame["monthly_trend_bull"], frame["monthly_trend_bear"] = monthly

    for field, col in _BOOL_FEATURES.items():
        frame[field] = primary[col].astype("boolean")
    obv = primary["OBV"].to_numpy(dtype="float64")
    obv_sma = primary["OBV_SMA20"].to_numpy(dtype="float64")
    known = np.isfinite(obv) & np.isfinite(obv_sma)
    with np.errstate(invalid="ignore"):
        frame["obv_rising"] = _nullable(obv > obv_sma, known, index)
    frame["has_daily"] = native_tf == "daily"
    return frame


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


# MTF labels with no bullish timeframe. They carry the same score penalty
# DIVERGENT did before they were split out of it (see score_readiness).
MTF_NO_UPTREND = {"BEARISH", "NEUTRAL"}


def _mtf_label(feat: BreakoutFeatures) -> str:
    """Trend agreement across daily (if native) / weekly / monthly (see classify_states_frame)."""
    return classify_states(feat)["MTF"]


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
    expanding = (feat.bb_width_pctl is not None and feat.bb_width_pctl >= 70) or (
        feat.atr_pctl is not None and feat.atr_pctl >= 70
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
    """ACCELERATING: RSI and MACD-histogram slopes both up; FADING: both down."""
    return classify_states(feat)["Momentum"]


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


# =========================
# VECTORIZED RULES (one row per bar; same rules as the scalar functions above)
# =========================

TIMEFRAMES = ("daily", "weekly", "monthly")
STATE_COLUMNS = [
    "Trend",
    "Volatility",
    "Volume State",
    "Momentum",
    "Structure",
    "Breakout Distance",
    "MTF",
]


def _num(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Float column with None → NaN (1-row frames built from a dataclass hold None)."""
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype="float64")


def _is_true(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Truthiness of a bool-or-None column, as ``if feat.x:`` reads it."""
    return frame[name].astype("boolean").fillna(False).to_numpy(dtype=bool)


def _is_false(frame: pd.DataFrame, name: str) -> np.ndarray:
    """``feat.x is False`` (None is not False)."""
    return frame[name].astype("boolean").eq(False).fillna(False).to_numpy(dtype=bool)


def _is_known(frame: pd.DataFrame, name: str) -> np.ndarray:
    return frame[name].astype("boolean").notna().to_numpy(dtype=bool)


def _pctl_points_v(pctl: np.ndarray, full: int, mid: int, bands: tuple[float, float]) -> np.ndarray:
    """Vectorized :func:`_pctl_points` (NaN scores 0)."""
    return np.select(
        [pctl <= bands[0], pctl <= bands[1], pctl <= 50], [full, mid, max(1, mid // 2)], 0
    )


def classify_states_frame(feats: pd.DataFrame) -> pd.DataFrame:
    """Per-row :func:`classify_states` over a frame of feature rows."""
    has_daily = feats["has_daily"].astype(bool).to_numpy()
    d_bull, w_bull, m_bull = (_is_true(feats, f"{tf}_trend_bull") for tf in TIMEFRAMES)
    d_bear, w_bear, m_bear = (_is_true(feats, f"{tf}_trend_bear") for tf in TIMEFRAMES)
    d_known, w_known, m_known = (_is_known(feats, f"{tf}_trend_bull") for tf in TIMEFRAMES)
    close, support = _num(feats, "close"), _num(feats, "support")
    dist = _num(feats, "distance_resistance_atr")
    bb, atr_p, rvol = _num(feats, "bb_width_pctl"), _num(feats, "atr_pctl"), _num(feats, "rvol20")
    rsi_slope, hist_slope = _num(feats, "rsi_slope"), _num(feats, "macd_hist_slope")

    with np.errstate(invalid="ignore"):
        trend = np.select(
            [np.where(has_daily, d_bull, w_bull), np.where(has_daily, d_bear, w_bear)],
            ["BULLISH", "BEARISH"],
            "NEUTRAL",
        )
        volatility = np.select(
            [_is_true(feats, "compression"), (bb >= 70) | (atr_p >= 70)],
            ["COMPRESSING", "EXPANDING"],
            "NORMAL",
        )
        volume = np.select(
            [_is_true(feats, "volume_dryup") | (rvol < RVOL_DRYUP), rvol >= 1.20],
            ["DRYING_UP", "EXPANDING"],
            "NORMAL",
        )
        momentum = np.select(
            [(rsi_slope > 0) & (hist_slope > 0), (rsi_slope < 0) & (hist_slope < 0)],
            ["ACCELERATING", "FADING"],
            "NEUTRAL",
        )
        structure = np.select(
            [close < support, np.isnan(dist), dist < 0, dist <= AT_RESISTANCE_ATR],
            ["BELOW_SUPPORT", "UNKNOWN", "ABOVE_RESISTANCE", "AT_RESISTANCE"],
            "UNDER_RESISTANCE",
        )

    # MTF: the daily flag counts only on daily data.
    d_on = has_daily & d_known
    n_known = d_on.astype(int) + w_known + m_known
    all_bull = (~d_on | d_bull) & (~w_known | w_bull) & (~m_known | m_bull)
    any_bull = (has_daily & d_bull) | w_bull | m_bull
    any_bear = (has_daily & d_bear) | w_bear | m_bear
    mtf = np.select(
        [n_known < 2, all_bull, any_bull & any_bear, any_bull, any_bear],
        ["INSUFFICIENT", "ALIGNED", "DIVERGENT", "PARTIAL", "BEARISH"],
        "NEUTRAL",
    )

    return pd.DataFrame(
        {
            "Trend": trend,
            "Volatility": volatility,
            "Volume State": volume,
            "Momentum": momentum,
            "Structure": structure,
            "Breakout Distance": [f"{d:.1f} ATR" if d == d else "" for d in dist],
            "MTF": mtf,
        },
        index=feats.index,
        dtype=object,
    )


def classify_status_frame(feats: pd.DataFrame, states: pd.DataFrame) -> pd.Series:
    """Per-row :func:`classify_status`; the first matching rule wins, in the same order."""
    failed = _is_true(feats, "failed_breakout")
    confirmed = _is_true(feats, "breakout_confirmation")
    triggered = _is_true(feats, "breakout_triggered")
    dist, rvol, bb = (
        _num(feats, c) for c in ("distance_resistance_atr", "rvol20", "bb_width_pctl")
    )
    volume_state = states["Volume State"].to_numpy()

    with np.errstate(invalid="ignore"):
        near = (dist >= 0.0) & (dist <= PRE_BREAKOUT_ATR_MAX)
        still_far = dist > PRE_BREAKOUT_ATR_MAX
        early_bb = bb <= 40
        quiet_expansion = (volume_state == "EXPANDING") & (rvol < RVOL_CONFIRM)
    under = states["Structure"].isin(["UNDER_RESISTANCE", "AT_RESISTANCE"]).to_numpy()
    mtf_ok = states["MTF"].isin(["ALIGNED", "PARTIAL"]).to_numpy()
    trend_ok = (states["Trend"] == "BULLISH").to_numpy()
    compressing = (states["Volatility"] == "COMPRESSING").to_numpy()
    accelerating = (states["Momentum"] == "ACCELERATING").to_numpy()
    volume_ok = np.isin(volume_state, ["DRYING_UP", "NORMAL"]) | quiet_expansion
    not_through = ~triggered

    pre = not_through & trend_ok & compressing & accelerating & under & near & mtf_ok & volume_ok
    early_coil = compressing | early_bb
    developing = trend_ok & early_coil & (accelerating | still_far) & not_through & still_far
    status = np.select(
        [failed, confirmed, pre, triggered & ~confirmed, developing],
        [STATUS_FAILED, STATUS_CONFIRMED, STATUS_PRE, STATUS_TRIGGERED, STATUS_DEV],
        STATUS_WATCH,
    )
    return pd.Series(status, index=feats.index, dtype=object, name="Status")


def score_readiness_frame(feats: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Per-row :func:`score_readiness`: integer pillar and factor scores."""
    has_daily = feats["has_daily"].astype(bool).to_numpy()
    close, sma50, sma200 = _num(feats, "close"), _num(feats, "sma50"), _num(feats, "sma200")
    rsi, rsi_slope = _num(feats, "rsi"), _num(feats, "rsi_slope")
    hist, hist_slope = _num(feats, "macd_hist"), _num(feats, "macd_hist_slope")
    rvol, ext = _num(feats, "rvol20"), _num(feats, "extension_atr")
    dist = _num(feats, "distance_resistance_atr")
    triggered = _is_true(feats, "breakout_triggered")
    obv_rising = _is_true(feats, "obv_rising")
    dryup = _is_true(feats, "volume_dryup")

    with np.errstate(invalid="ignore"):
        daily_cap = np.where(has_daily, 10, 0)
        weekly_cap = np.where(has_daily, 8, 15)
        monthly_cap = np.where(has_daily, 7, 10)
        above50 = np.where(np.isnan(sma200) | (close > sma200), 6, 3)
        daily_pts = np.select(
            [has_daily & _is_true(feats, "daily_trend_bull"), has_daily & (close > sma50)],
            [daily_cap, above50],
            0,
        )
        weekly_pts, monthly_pts = (
            np.select(
                [
                    _is_true(feats, f"{tf}_trend_bull"),
                    _is_false(feats, f"{tf}_trend_bull") & _is_false(feats, f"{tf}_trend_bear"),
                ],
                [cap, cap // 2],
                0,
            )
            for tf, cap in (("weekly", weekly_cap), ("monthly", monthly_cap))
        )
        trend_score = daily_pts + weekly_pts + monthly_pts

        bb_pts = _pctl_points_v(_num(feats, "bb_width_pctl"), 8, 6, (15, 25))
        atr_pts = _pctl_points_v(_num(feats, "atr_pctl"), 6, 4, (20, 35))
        range_pts = _pctl_points_v(_num(feats, "range20_pctl"), 5, 3, (20, 35))
        squeeze_pts = np.select(
            [_is_true(feats, "squeeze"), _num(feats, "kc_width_pctl") <= 30], [6, 3], 0
        )
        compression_score = bb_pts + atr_pts + range_pts + squeeze_pts

        rsi_pts = np.select(
            [
                np.isnan(rsi),
                (rsi >= 50) & (rsi <= 65),
                ((rsi >= 45) & (rsi < 50)) | ((rsi > 65) & (rsi <= 70)),
                (rsi >= 40) & (rsi < 45),
                rsi > 75,
            ],
            [0, 6, 4, 2, 0],
            1,
        )
        rsi_slope_pts = np.select([rsi_slope >= 2, rsi_slope > 0], [7, 5], 0)
        macd_pts = np.select([(hist_slope > 0) & (hist > 0), hist_slope > 0], [7, 5], 0)
        momentum_score = rsi_pts + rsi_slope_pts + macd_pts

        # Regime-aware volume: dry-up before a breakout, expansion once triggered.
        dry_base = np.select([dryup, rvol < 1.0], [8, 4], 0)
        rvol_trig = np.select(
            [rvol >= 2.0, rvol >= RVOL_CONFIRM, rvol >= RVOL_IGNITION], [7, 6, 3], 0
        )
        rvol_trig = np.where(obv_rising, np.minimum(7, rvol_trig + 1), rvol_trig)
        rvol_quiet = np.select(
            [
                obv_rising & (np.nan_to_num(rvol, nan=0.0) < 1.2),
                obv_rising,
                (rvol >= RVOL_IGNITION) & (rvol < RVOL_CONFIRM),
                dryup,
            ],
            [7, 5, 4, 2],
            0,
        )
        rvol_pts = np.where(triggered, rvol_trig, rvol_quiet)
        dry_pts = np.where(triggered, np.minimum(dry_base, 3), dry_base)
        volume_score = dry_pts + rvol_pts

        prox_pts = np.select(
            [
                np.isnan(dist),
                (dist >= 0.0) & (dist <= 0.75),
                (dist > 0.75) & (dist <= 1.50),
                (dist > 1.50) & (dist <= 2.50),
                (dist < 0) & (dist >= -0.35),
                dist < 0,
            ],
            [0, 10, 7, 4, 5, 2],
            1,
        )
        ext_pts = np.select(
            [
                np.isnan(ext),
                (ext >= 0.0) & (ext <= 1.50),
                (ext > 1.50) & (ext <= 2.00),
                ext < 0,
            ],
            [0, 5, 3, 2],
            0,
        )
        structure_score = prox_pts + ext_pts

        penalty = (
            np.select(
                [_is_true(feats, "failed_breakout"), _is_true(feats, "wick_reject")], [20, 8], 0
            )
            + np.where(states["MTF"].isin(["DIVERGENT", *MTF_NO_UPTREND]).to_numpy(), 10, 0)
            + np.where(ext > 2.50, 10, 0)
            + np.where(rsi > 75, 5, 0)
            + np.where((states["Trend"] == "BEARISH").to_numpy(), 10, 0)
        )

    raw_score = trend_score + compression_score + momentum_score + volume_score + structure_score
    columns = {
        "Breakout Readiness": np.clip(raw_score - penalty, 0, 100),
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
    return pd.DataFrame(
        {k: np.asarray(v, dtype="int64") for k, v in columns.items()}, index=feats.index
    )


def _one_row(feat: BreakoutFeatures) -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(feat)])


def classify_states(feat: BreakoutFeatures) -> dict[str, str]:
    """State labels — the primary output of the scanner (one row of classify_states_frame)."""
    row = classify_states_frame(_one_row(feat)).iloc[0]
    return {col: str(row[col]) for col in STATE_COLUMNS}


def classify_status(feat: BreakoutFeatures, states: dict[str, str]) -> str:
    """Setup class from states/features. Score is not an input on purpose."""
    return str(classify_status_frame(_one_row(feat), pd.DataFrame([states])).iloc[0])


def score_readiness(feat: BreakoutFeatures, states: dict[str, str]) -> dict:
    """100-point Breakout Readiness plus per-factor scores for later tuning.

    Pillars (before penalties): Trend 25, Compression 25, Momentum 20,
    Volume 15, Structure 15. When daily bars are unavailable the trend
    budget is reallocated to weekly (15) + monthly (10).
    """
    row = score_readiness_frame(_one_row(feat), pd.DataFrame([states])).iloc[0]
    return {col: int(value) for col, value in row.items()}


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


def _display_mask(df: pd.DataFrame) -> pd.Series:
    """Display rule: candidate statuses always, DEVELOPING at or above its floor, never WATCH."""
    status = df["Status"]
    floor = status.map(DISPLAY_SCORE_FLOOR)
    score = pd.to_numeric(df["Breakout Readiness"], errors="coerce")
    return status.isin(CANDIDATE_STATUSES) | (score >= floor)


def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rank = {
        STATUS_PRE: 0,
        STATUS_CONFIRMED: 1,
        STATUS_TRIGGERED: 2,
        STATUS_FAILED: 3,
        STATUS_WATCH: 4,
        STATUS_DEV: 5,
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


def _rejection_key(reason: str) -> str:
    """Rejection bucket for a ValueError raised while loading or scoring a ticker."""
    if "insufficient" in reason:
        return "insufficient_data"
    if "empty" in reason or "usable" in reason or "Missing required" in reason:
        return "missing_columns"
    return "invalid_ohlcv"


def scan_history(
    bars: pd.DataFrame, symbol: str, scan_mode: str, native_tf: str | None = None
) -> pd.DataFrame:
    """Every bar's states, Status and Readiness, as ``run_scanner`` would have written them.

    One row per native bar that has at least ``MIN_PRIMARY_BARS`` bars of
    history: the row dated t equals the ``run_scanner --as-of t`` row for this
    ticker. Columns are ``Date`` (the bar timestamp) then ``OUTPUT_COLUMNS``.
    On daily data the weekly/monthly trend flags at t use the bar forming at t.
    Pass bars from :func:`raw.load_raw`. Raises ``ValueError`` like
    ``FeatureEngine.extract`` (e.g. ``insufficient bars``).
    """
    native = native_tf or _native_timeframe(scan_mode)
    clean = normalize_ohlcv(bars)
    if len(clean) < MIN_PRIMARY_BARS:
        raise ValueError(f"insufficient bars: {len(clean)} < {MIN_PRIMARY_BARS}")

    feats = _feature_frame(clean, native)
    states = classify_states_frame(feats)
    scores = score_readiness_frame(feats, states)
    out = pd.DataFrame(
        {"Date": feats["date"], "Symbol": symbol.upper(), "Mode": scan_mode, "Source": "breakout"},
        index=feats.index,
    )
    out["AsOf Date"] = feats["asof"]
    for col, field in FEATURE_COLUMNS.items():
        out[col] = feats[field]
    out["MTF Alignment"] = states["MTF"] == "ALIGNED"
    out[STATE_COLUMNS] = states
    out["Status"] = classify_status_frame(feats, states)
    out[list(scores.columns)] = scores
    out = out.iloc[MIN_PRIMARY_BARS - 1 :].reset_index(drop=True)
    return out[["Date", *OUTPUT_COLUMNS]]


def scan_files(
    raw_files: Iterable[str],
    active_tickers: set[str],
    raw_dir: str,
    scan_mode: str,
    native_tf: str,
    as_of: str | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Walk raw CSVs and return (rows, rejection_counts).

    ``as_of`` (YYYY-MM-DD) drops every bar not yet complete on that date.
    """
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
            bars = raw.load_raw(path, weekly=(native_tf == "weekly"), as_of=as_of)
            features = engine.extract(bars, symbol, scan_mode)
            results.append(scorer.evaluate(features))
        except OSError as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            rejection_counts["read_error"] = rejection_counts.get("read_error", 0) + 1
        except ValueError as exc:
            key = _rejection_key(str(exc))
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
        except Exception:  # per-ticker: one bad file must not stop the scan
            logger.exception("Error scoring %s", symbol)
            rejection_counts["execution_error"] = rejection_counts.get("execution_error", 0) + 1

    return results, rejection_counts


def run_scanner(as_of: str | None = None) -> pd.DataFrame:
    """Scan the active universe and archive feature + state rows.

    ``as_of`` (YYYY-MM-DD) replays a past date: only bars complete on that date
    are used and the archive is stamped with it.
    """
    native_tf = _native_timeframe(_data_mode)
    os.makedirs(LOG_DIR, exist_ok=True)
    if as_of:
        logger.info("AS-OF replay: %s (bars completed on/before this date only)", as_of)
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

    raw_files = [os.path.basename(f.path) for f in raw.raw_files(RAW_DATA_DIR)]
    logger.info("Found %s raw data files in target silo", len(raw_files))

    results, rejection_counts = scan_files(
        raw_files,
        active_tickers,
        RAW_DATA_DIR,
        mode,
        native_tf,
        as_of=as_of,
    )

    today = config.run_stamp(as_of)
    out_path = os.path.join(LOG_DIR, f"breakout_setups_{today}.csv")
    df_out = pd.DataFrame(results)
    if df_out.empty:
        df_out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        df_out = df_out.reindex(columns=OUTPUT_COLUMNS)
        df_out = _sort_candidates(df_out)

    df_out.to_csv(out_path, index=False)
    logger.info("Archive created: %s (%s row(s))", out_path, len(df_out))

    display = df_out[_display_mask(df_out)] if not df_out.empty else df_out
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


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m finance_vibe.scanners.breakout [weekly|daily] [--as-of YYYY-MM-DD]``."""
    args = sys.argv[1:] if argv is None else argv
    cli_mode = args[0].lower() if args and not args[0].startswith("-") else "weekly"
    if cli_mode not in ("weekly", "daily"):
        logger.warning("Unknown mode %r parsed to scanner. Defaulting to 'weekly'.", cli_mode)
    try:
        as_of = config.parse_as_of(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    set_mode(cli_mode)
    run_scanner(as_of=as_of)
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
