"""Coiled Cobra live scanner: v2.2 cleaned coil → expansion scorecard.

Spec: ``Coiled Cobra Rubric .MD``. Scores the *latest bar* of each ticker.
A pass is a compressed leader vs QQQ that is still structurally healthy —
not a deep pullback, not a Fib bounce.

Trade planner uses expansion levels (Close / Coil_Low / 2R-3R).
Historical archive / Rel_Forward vs QQQ → coiled_cobra_backtest.py
ML ranking of new coils → coiled_cobra_ml_training.py + ml_ranker.py
"""
import argparse
import os
import sys
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, time
from zoneinfo import ZoneInfo
from typing import Optional

import pandas as pd
import pandas_ta as ta

# --- PACKAGE IMPORT ---
from finance_vibe import config
from finance_vibe.market import load_benchmark_frame, relative_strength, select_raw_paths

# =========================
# PROFILE CONFIGURATION
# =========================
# Defaults are daily (project primary). configure_mode() updates these globals.
mode = "daily"
LOOKBACK = 252
COIL_BARS = 30
STRUCTURE_STOP_BARS = 30
RS_LOOKBACK = 63
RS_RATIO_MA = 20
BENCHMARK = "QQQ"


def configure_mode(scan_mode: str) -> str:
    """Set weekly/daily calibration and data paths. Safe to call from workers."""
    global mode, LOOKBACK, COIL_BARS, STRUCTURE_STOP_BARS
    global RS_LOOKBACK, RS_RATIO_MA, HIGH_LOOKBACK_BARS, TIGHTNESS_LOOKBACK
    global RAW_DATA_DIR, LOG_DIR
    mode = scan_mode if scan_mode in ("weekly", "daily") else config.DEFAULT_MODE
    LOOKBACK = 252 if mode == "daily" else 52
    COIL_BARS = 30 if mode == "daily" else 8
    # Fallback floor when Coil_Low is missing: match the coil window.
    STRUCTURE_STOP_BARS = COIL_BARS
    RS_LOOKBACK = 63 if mode == "daily" else 13
    RS_RATIO_MA = 20 if mode == "daily" else 5
    # Column names stay Dist_High_{63,126,252}_*; weekly uses 13/26/52 bars.
    HIGH_LOOKBACK_BARS = (63, 126, 252) if mode == "daily" else (13, 26, 52)
    TIGHTNESS_LOOKBACK = 252 if mode == "daily" else 52
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", mode)
    LOG_DIR = os.path.join(BASE_DIR, "data", "logs", mode)
    os.makedirs(LOG_DIR, exist_ok=True)
    return mode


def local_swing_low(df: pd.DataFrame, bars: int | None = None) -> float:
    """Minimum Low over the last ``bars`` sessions (local consolidation floor)."""
    if bars is None:
        bars = STRUCTURE_STOP_BARS
    window = df.iloc[-bars:] if len(df) >= bars else df
    return float(window["Low"].min())


def drop_in_progress_session(
    df: pd.DataFrame,
    *,
    market_tz: str = "America/New_York",
    finalized_after: time = time(16, 15),
) -> pd.DataFrame:
    """Drop today's daily bar only while the regular session may still be incomplete.

    The old implementation dropped *every* bar dated today, even when the scanner
    ran after the close. That made an after-hours daily scan operate on yesterday's
    data. We now keep today's bar once the market has had a small finalization buffer.

    Notes
    -----
    - Weekly ingest is expected to handle its own incomplete-week protection.
    - ``16:15 ET`` is deliberately a little later than the 16:00 regular close so
      upstream data providers have time to finalize the daily candle.
    """
    if mode != "daily" or df.empty or "Date" not in df.columns:
        return df

    last = pd.to_datetime(df["Date"].iloc[-1], errors="coerce")
    if pd.isna(last):
        return df

    now_et = datetime.now(ZoneInfo(market_tz))
    if last.date() != now_et.date():
        return df

    if now_et.time() < finalized_after:
        return df.iloc[:-1].copy()

    return df

# Soft pass floors (v2.2 core six sum to 100; Fib / MACD cross not in the sum)
MIN_PASS_SCORE = 70
GRADE_A_SCORE = 85
MAX_SCORE = 100.0
# Hard gates (ready-to-run leaders, not lagging coils)
MIN_COMPRESSION = 5
MIN_STRUCTURE = 8
# Structure is also a hard boolean gate: price must be above a rising EMA50.
STRUCTURE_SLOPE_BARS = 3
MIN_RS_POINTS = 12  # requires full RS pass (ratio > MA and positive rel-return)
RSI_HEALTHY_LO = 45.0
RSI_HEALTHY_HI = 70.0
HIGH_COLUMN_KEYS = (63, 126, 252)
HIGH_LOOKBACK_BARS = (63, 126, 252)
TIGHTNESS_LOOKBACK = 252
SCORED_PILLAR_KEYS = (
    "volume_contraction",
    "macd_compression",
    "structure",
    "relative_strength",
    "coil_width",
    "proximity_highs",
)
WEEKLY_CONFIRM_BOOST = 1.25

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", mode)
ACTIVE_TICKERS_PATH = os.path.join(BASE_DIR, "data", "active_tickers.csv")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs", mode)

os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =========================
# INDICATORS
# =========================


def add_macro_indicators(df: pd.DataFrame, lookback=None) -> pd.DataFrame:
    """EMA stack, MACD, RSI, ATR, and rolling Fib levels for coil scoring."""
    if lookback is None:
        lookback = LOOKBACK
    out = df.copy()
    out["EMA20"] = ta.ema(out["Close"], length=20)
    out["EMA50"] = ta.ema(out["Close"], length=50)
    out["EMA100"] = ta.ema(out["Close"], length=100)

    macd = ta.macd(out["Close"])
    out["MACD"] = macd["MACD_12_26_9"]
    out["MACD_Signal"] = macd["MACDs_12_26_9"]

    out["RSI"] = ta.rsi(out["Close"], length=14)

    rolling_max = out["High"].rolling(window=lookback, min_periods=lookback).max()
    rolling_min = out["Low"].rolling(window=lookback, min_periods=lookback).min()
    out["Fib_786"] = rolling_max - ((rolling_max - rolling_min) * 0.786)
    out["Fib_618"] = rolling_max - ((rolling_max - rolling_min) * 0.618)

    out["ATR"] = ta.atr(out["High"], out["Low"], out["Close"], length=14)

    # Coil scoring needs EMA50; Fib may be NaN early but is optional now.
    out.dropna(subset=["EMA20", "EMA50", "MACD", "ATR"], inplace=True)
    return out


# =========================
# STRUCTURAL LAYER FILTER (AMT)
# =========================


def _interp_score(x: float, breakpoints: list[tuple[float, float]]) -> float:
    """Linear interpolate between (x, points) knots. Clamp outside. Round to 2 decimals."""
    if not breakpoints:
        return 0.0
    xs = [p[0] for p in breakpoints]
    ys = [p[1] for p in breakpoints]
    if x <= xs[0]:
        return round(float(ys[0]), 2)
    if x >= xs[-1]:
        return round(float(ys[-1]), 2)
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, y0 = xs[i - 1], ys[i - 1]
            x1, y1 = xs[i], ys[i]
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return round(float(y0 + t * (y1 - y0)), 2)
    return round(float(ys[-1]), 2)


def volume_contraction_metrics(
    df: pd.DataFrame,
    coil_bars: int | None = None,
    baseline_bars: int = 20,
) -> tuple[float, Optional[float]]:
    """Score volume dry-up during the coil using only observable OHLCV data.

    Returns
    -------
    score : float
        0-20 score. Lower coil volume relative to the preceding baseline is better.
    ratio : Optional[float]
        Median coil volume / median baseline volume. Values below 1.0 indicate
        contraction; e.g. 0.55 means coil volume is 45% below baseline.

    This replaces the old close-price weighted histogram that was described as a
    volume profile. Daily/weekly OHLCV cannot reveal true intrabar volume-at-price.
    """
    if coil_bars is None:
        coil_bars = COIL_BARS

    required = coil_bars + baseline_bars
    if len(df) < required or "Volume" not in df.columns:
        return 0.0, None

    volume = pd.to_numeric(df["Volume"], errors="coerce")
    baseline = volume.iloc[-required:-coil_bars].median()
    coil_volume = volume.iloc[-coil_bars:].median()

    if (
        pd.isna(baseline)
        or pd.isna(coil_volume)
        or float(baseline) <= 0
        or float(coil_volume) < 0
    ):
        return 0.0, None

    ratio = float(coil_volume) / float(baseline)

    # Strong dry-up gets full credit. Expansion during the coil gets no credit.
    score = _interp_score(
        ratio,
        [
            (0.00, 20.0),
            (0.50, 20.0),
            (0.65, 17.0),
            (0.80, 13.0),
            (1.00, 7.0),
            (1.20, 0.0),
        ],
    )
    return score, round(ratio, 4)


def evaluate_volume_profile_shelf(
    df: pd.DataFrame, current_price: float | None = None, lookback=None
) -> float:
    """Backward-compatible name for the new volume-contraction score.

    ``current_price`` and ``lookback`` are retained so existing tests/callers do
    not break. The implementation no longer pretends daily OHLCV contains a true
    volume-at-price profile.
    """
    score, _ = volume_contraction_metrics(df)
    return score


def pillar_row_fields(parts: dict[str, float]) -> dict:
    """Map internal Parts keys to setup-row / backtest CSV columns."""
    return {
        # Compatibility alias: Volume_Shelf is the 0-20 volume contraction score.
        "Volume_Shelf": parts.get("volume_contraction"),
        "MACD_Compression": parts.get("macd_compression"),
        "Structure": parts.get("structure"),
        "RS_Score": parts.get("relative_strength"),
        "Coil_Width": parts.get("coil_width"),
        "Proximity_Highs": parts.get("proximity_highs"),
        "MACD_Cross": parts.get("macd_cross"),
        "Fib_Bonus": parts.get("fib_bonus"),
    }


def coil_geometry_fields(df: pd.DataFrame, atr: float) -> dict:
    """Unscored coil measurements for trade planning and research.

    ``Pivot_Price`` is the highest high of the *prior* coil bars, excluding the
    latest bar. This lets ``Distance_To_Pivot_Pct`` become negative when today's
    close has already broken through the prior coil pivot.
    """
    if df.empty:
        return {
            "MACD_Spread_ATR": None,
            "Coil_Width_ATR": None,
            "Coil_Width_Pctile": None,
            "Coil_High": None,
            "Coil_Low": None,
            "Pivot_Price": None,
            "Distance_To_Pivot_Pct": None,
        }

    latest = df.iloc[-1]
    coil_n = min(COIL_BARS, len(df))
    coil_slice = df.iloc[-coil_n:]
    coil_high = float(coil_slice["High"].max())
    coil_low = float(coil_slice["Low"].min())

    prior_coil = df.iloc[-(coil_n + 1):-1] if len(df) > 1 else df.iloc[0:0]
    if prior_coil.empty:
        pivot_price = None
        distance_to_pivot = None
    else:
        pivot_price = float(prior_coil["High"].max())
        close_now = float(latest["Close"])
        distance_to_pivot = (pivot_price - close_now) / pivot_price if pivot_price > 0 else None

    macd = latest["MACD"] if "MACD" in df.columns else None
    macd_signal = latest["MACD_Signal"] if "MACD_Signal" in df.columns else None
    if (
        atr > 0
        and macd is not None
        and macd_signal is not None
        and pd.notna(macd)
        and pd.notna(macd_signal)
    ):
        macd_spread_atr = round(abs(float(macd) - float(macd_signal)) / atr, 4)
    else:
        macd_spread_atr = None

    coil_width_atr = round((coil_high - coil_low) / atr, 4) if atr > 0 else None
    return {
        "MACD_Spread_ATR": macd_spread_atr,
        "Coil_Width_ATR": coil_width_atr,
        "Coil_High": round(coil_high, 2),
        "Coil_Low": round(coil_low, 2),
        "Pivot_Price": round(pivot_price, 2) if pivot_price is not None else None,
        "Distance_To_Pivot_Pct": (
            round(distance_to_pivot, 4) if distance_to_pivot is not None else None
        ),
        "Coil_Width_Pctile": coil_width_percentile(df),
    }


def proximity_to_highs_fields(
    df: pd.DataFrame, close: float, atr: float
) -> tuple[dict, list[Optional[float]]]:
    """Distance to rolling highs. Column names are always 63/126/252 (weekly = 13/26/52 bars)."""
    fields: dict = {}
    dists_atr: list[Optional[float]] = []
    for col_key, n_bars in zip(HIGH_COLUMN_KEYS, HIGH_LOOKBACK_BARS):
        pct_key = f"Dist_High_{col_key}_Pct"
        atr_key = f"Dist_High_{col_key}_ATR"
        if len(df) < n_bars or atr <= 0 or close <= 0:
            fields[pct_key] = None
            fields[atr_key] = None
            dists_atr.append(None)
            continue
        high = float(df.iloc[-n_bars:]["High"].max())
        if high <= 0:
            fields[pct_key] = None
            fields[atr_key] = None
            dists_atr.append(None)
            continue
        dist_pct = (high - close) / high
        dist_atr = (high - close) / atr
        fields[pct_key] = round(dist_pct, 4)
        fields[atr_key] = round(dist_atr, 4)
        dists_atr.append(dist_atr)
    return fields, dists_atr


def proximity_highs_score(dists_atr: list[Optional[float]]) -> float:
    """Soft 0-12 pillar: closer to the near high scores more; mid/year highs are a bonus."""
    d63 = dists_atr[0] if len(dists_atr) > 0 else None
    d126 = dists_atr[1] if len(dists_atr) > 1 else None
    d252 = dists_atr[2] if len(dists_atr) > 2 else None
    score = 0.0
    if d63 is not None:
        score += _interp_score(d63, [(0.0, 8.0), (0.5, 6.0), (1.5, 3.0), (4.0, 0.0)])
    if d126 is not None:
        score += _interp_score(d126, [(0.0, 2.0), (2.0, 1.0), (6.0, 0.0)])
    if d252 is not None:
        score += _interp_score(d252, [(0.0, 2.0), (3.0, 1.0), (8.0, 0.0)])
    return round(min(12.0, score), 2)


def rsi_zone_fields(rsi) -> dict:
    """Raw RSI context: healthy 45-70 flag plus a 0-5 tent score (not in Score)."""
    if rsi is None or pd.isna(rsi):
        return {"RSI_Healthy": 0, "RSI_Zone_Score": 0.0}
    r = float(rsi)
    healthy = 1 if RSI_HEALTHY_LO <= r <= RSI_HEALTHY_HI else 0
    zone = _interp_score(
        r,
        [(35.0, 0.0), (45.0, 3.0), (52.0, 5.0), (62.0, 5.0), (70.0, 3.0), (78.0, 0.0)],
    )
    return {"RSI_Healthy": healthy, "RSI_Zone_Score": zone}


def volume_accumulation_fields(
    df: pd.DataFrame, coil_bars: int | None = None
) -> dict:
    """Causal coil-window OBV slope, up-volume share, and declining-volume trend."""
    empty = {
        "OBV_Coil_Slope": None,
        "Up_Volume_Ratio": None,
        "Volume_Trend_Ratio": None,
    }
    if coil_bars is None:
        coil_bars = COIL_BARS
    if len(df) < coil_bars + 1 or "Volume" not in df.columns:
        return empty

    coil = df.iloc[-coil_bars:]
    prev_close = float(df.iloc[-(coil_bars + 1)]["Close"])
    closes = pd.to_numeric(coil["Close"], errors="coerce")
    vols = pd.to_numeric(coil["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if closes.isna().all():
        return empty

    pc = prev_close
    obv = 0.0
    obv_start = 0.0
    up_vol = 0.0
    for i, (c, v) in enumerate(zip(closes.tolist(), vols.tolist())):
        if pd.isna(c):
            c = pc
        if c > pc:
            obv += float(v)
            up_vol += float(v)
        elif c < pc:
            obv -= float(v)
        if i == 0:
            obv_start = obv
        pc = float(c)

    total_vol = float(vols.sum())
    med_vol = float(vols.median()) if len(vols) else 0.0
    slope = None
    if med_vol > 0 and coil_bars > 1:
        slope = round((obv - obv_start) / ((coil_bars - 1) * med_vol), 4)
    up_ratio = round(up_vol / total_vol, 4) if total_vol > 0 else None

    half = coil_bars // 2
    trend = None
    if half >= 1:
        early = float(vols.iloc[:half].median())
        late = float(vols.iloc[half:].median())
        if early > 0:
            trend = round(late / early, 4)

    return {
        "OBV_Coil_Slope": slope,
        "Up_Volume_Ratio": up_ratio,
        "Volume_Trend_Ratio": trend,
    }


def coil_width_percentile(
    df: pd.DataFrame, coil_bars: int | None = None
) -> Optional[float]:
    """Percentile rank of current N-bar range vs prior tightness lookback (0=tight, 1=wide).

    Historical ranges end *before* the signal bar so the current coil is not in the
    comparison set (no look-ahead).
    """
    if coil_bars is None:
        coil_bars = COIL_BARS
    n = len(df)
    if n < coil_bars + 2:
        return None
    highs = pd.to_numeric(df["High"], errors="coerce")
    lows = pd.to_numeric(df["Low"], errors="coerce")
    current = float(highs.iloc[-coil_bars:].max() - lows.iloc[-coil_bars:].min())
    last_hist_end = n - 1
    first_end = coil_bars - 1
    hist_start = max(first_end, last_hist_end - TIGHTNESS_LOOKBACK)
    hist: list[float] = []
    for end in range(hist_start, last_hist_end):
        start = end - coil_bars + 1
        if start < 0:
            continue
        rng = float(highs.iloc[start : end + 1].max() - lows.iloc[start : end + 1].min())
        if pd.notna(rng):
            hist.append(rng)
    if not hist:
        return None
    below = sum(1 for h in hist if h <= current)
    return round(below / len(hist), 4)


# =========================
# FIB SCORE (optional bonus, demoted)
# =========================
def fibonacci_score(
    current_price: float,
    fib_levels: dict,
    atr: float,
    max_atr_distance: float = 0.5,
) -> float:
    """Optional Fib proximity bonus (0-5). Demoted from the old 30-pt gate."""
    if not isinstance(fib_levels, dict) or atr <= 0:
        return 0.0

    best_score = 0.0
    for level_price, max_possible_score in fib_levels.items():
        atr_distance = abs(current_price - level_price) / atr
        if atr_distance >= max_atr_distance:
            continue
        level_score = max_possible_score * (1 - (atr_distance / max_atr_distance) ** 2)
        if level_score > best_score:
            best_score = level_score
    return round(best_score, 2)


def macd_compression_score(macd: float, macd_signal: float, atr: float) -> float:
    """ATR-normalized MACD-signal compression (0-20). No MACD < 0 requirement.

    A tight spread relative to ATR is the coil — works for bases in uptrends
    as well as washed-out reversals.
    """
    if atr <= 0:
        return 0.0
    spread = abs(macd - macd_signal) / atr
    return _interp_score(
        spread,
        [(0.00, 20.0), (0.05, 20.0), (0.10, 15.0), (0.18, 10.0), (0.30, 5.0), (0.40, 0.0)],
    )


def coil_width_score(df: pd.DataFrame, atr: float, coil_bars: int | None = None) -> float:
    """Tight N-bar range vs ATR (0-13). Coiled energy before expansion."""
    if coil_bars is None:
        coil_bars = COIL_BARS
    if atr <= 0 or len(df) < coil_bars:
        return 0.0
    window = df.iloc[-coil_bars:]
    rng = float(window["High"].max() - window["Low"].min())
    width_atr = rng / atr
    return _interp_score(
        width_atr,
        [(0.0, 13.0), (4.0, 13.0), (6.0, 9.0), (8.0, 4.0), (10.0, 0.0)],
    )


def structure_score(df: pd.DataFrame) -> float:
    """Healthy trend structure for a leader coil (0-20).

    Score components:
      - price above EMA50: up to 8
      - EMA50 rising over STRUCTURE_SLOPE_BARS: up to 6
      - EMA50 above EMA100: up to 6

    The numerical score describes *quality*. ``healthy_structure_gate`` separately
    enforces the minimum structural condition so price alone cannot pass the gate.
    """
    if len(df) < STRUCTURE_SLOPE_BARS + 1:
        return 0.0

    latest = df.iloc[-1]
    close = float(latest["Close"])
    ema50 = float(latest["EMA50"])
    atr = float(latest["ATR"]) if pd.notna(latest.get("ATR")) else 0.0
    ema100 = latest.get("EMA100")
    score = 0.0

    if close > ema50:
        if atr > 0:
            dist_atr = (close - ema50) / atr
            score += _interp_score(dist_atr, [(0.0, 4.0), (0.5, 8.0)])
        else:
            score += 8.0

    slope_ref = df.iloc[-(STRUCTURE_SLOPE_BARS + 1)].get("EMA50")
    if pd.notna(slope_ref):
        ema50_ago = float(slope_ref)
        if atr > 0:
            rise_atr = (ema50 - ema50_ago) / atr
            if rise_atr > 0:
                score += round(min(6.0, 6.0 * rise_atr / 0.05), 2)
        elif ema50 > ema50_ago:
            score += 6.0

    if ema100 is not None and pd.notna(ema100) and ema50 > float(ema100):
        if atr > 0:
            gap_atr = (ema50 - float(ema100)) / atr
            score += _interp_score(gap_atr, [(0.0, 3.0), (0.5, 6.0)])
        else:
            score += 6.0

    return round(min(20.0, score), 2)


def healthy_structure_gate(df: pd.DataFrame) -> bool:
    """Minimum long-side structure: price above a rising EMA50.

    EMA50 > EMA100 remains a quality bonus rather than a hard requirement so the
    scanner can still catch emerging leaders early in a trend transition.
    """
    if len(df) < STRUCTURE_SLOPE_BARS + 1:
        return False

    latest = df.iloc[-1]
    slope_ref = df.iloc[-(STRUCTURE_SLOPE_BARS + 1)]
    required = (latest.get("Close"), latest.get("EMA50"), slope_ref.get("EMA50"))
    if any(value is None or pd.isna(value) for value in required):
        return False

    close = float(latest["Close"])
    ema50 = float(latest["EMA50"])
    ema50_ago = float(slope_ref["EMA50"])
    return close > ema50 and ema50 > ema50_ago


def rs_score(
    stock_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    as_of=None,
) -> tuple[float, Optional[float]]:
    """Relative strength vs QQQ (0-15). Leaders before monster runs usually lead."""
    if benchmark_df is None:
        return 0.0, None
    ok, rel = relative_strength(
        stock_df,
        benchmark_df,
        as_of=as_of,
        lookback=RS_LOOKBACK,
        ratio_ma_bars=RS_RATIO_MA,
    )
    if ok and rel is not None:
        pts = 12.0 + 3.0 * min(max(rel, 0.0), 0.10) / 0.10
        return round(min(15.0, pts), 2), rel
    if rel is not None and rel > 0:
        return round(max(1.0, min(5.0, 5.0 * rel / 0.10)), 2), rel
    return 0.0, rel


def macd_crossed_this_bar(
    prev_macd: float,
    prev_signal: float,
    macd: float,
    macd_signal: float,
) -> bool:
    """True when MACD crosses above Signal on this bar (exported flag, not scored)."""
    return prev_macd <= prev_signal and macd > macd_signal


def macd_cross_score(
    prev_macd: float,
    prev_signal: float,
    macd: float,
    macd_signal: float,
    atr: float,
) -> float:
    """v2.2: MACD cross is a binary export flag. Always 0 in the summed Score."""
    del prev_macd, prev_signal, macd, macd_signal, atr
    return 0.0


# =========================
# SYSTEMATIC GRADING MATRIX (coil → expansion)
# =========================


def evaluate_coiled_cobra(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """100-point coil scorecard: catch compressed leaders before they expand.

    Pillars (v2.2, sum clipped at 100):
      Volume contraction 20 · MACD compression 20 · Structure 20 ·
      Relative strength 15 · Coil width 13 · Proximity to highs 12
    MACD cross and Fib are exported only — not added to Score.
    Hard gates: compression, structure, and full RS vs QQQ (no lagging coils).
    """
    if len(df) < max(COIL_BARS + 2, 25):
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    current_price = float(latest["Close"])
    atr = float(latest["ATR"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_Signal"])
    prev_macd = float(prev["MACD"])
    prev_macd_signal = float(prev["MACD_Signal"])
    as_of = latest["Date"] if "Date" in df.columns else None
    rsi_val = latest["RSI"] if "RSI" in df.columns else None

    parts: dict[str, float] = {}
    checks_passed = 0

    # 1. Volume contraction / dry-up (0-20)
    volume_pts = evaluate_volume_profile_shelf(df, current_price)
    _, volume_ratio = volume_contraction_metrics(df)
    parts["volume_contraction"] = volume_pts
    if volume_pts >= 10:
        checks_passed += 1

    # 2. MACD compression (0-20) — no MACD < 0 requirement
    comp = macd_compression_score(macd, macd_signal, atr)
    parts["macd_compression"] = comp
    if comp >= 10:
        checks_passed += 1

    # 3. Structure / rising stack (0-20)
    struct = structure_score(df)
    parts["structure"] = struct
    if struct >= 12:
        checks_passed += 1

    # 4. Relative strength vs QQQ (0-15)
    rs_pts, rs_rel = rs_score(df, benchmark_df, as_of=as_of)
    parts["relative_strength"] = rs_pts
    if rs_pts >= 12:
        checks_passed += 1

    # 5. Coil width (0-13)
    coil = coil_width_score(df, atr)
    parts["coil_width"] = coil
    if coil >= 8:
        checks_passed += 1

    # 6. Proximity to highs (0-12) — not a hard gate
    prox_fields, dists_atr = proximity_to_highs_fields(df, current_price, atr)
    prox = proximity_highs_score(dists_atr)
    parts["proximity_highs"] = prox
    if prox >= 6:
        checks_passed += 1

    crossed = macd_crossed_this_bar(prev_macd, prev_macd_signal, macd, macd_signal)
    parts["macd_cross"] = 0.0

    fib_618 = latest.get("Fib_618")
    fib_786 = latest.get("Fib_786")
    fib_score = 0.0
    if pd.notna(fib_618) and pd.notna(fib_786) and atr > 0:
        fib_score = fibonacci_score(
            current_price,
            {float(fib_618): 4.5, float(fib_786): 5.0},
            atr,
            max_atr_distance=0.75,
        )
    parts["fib_bonus"] = fib_score

    score = min(
        MAX_SCORE,
        float(sum(parts[k] for k in SCORED_PILLAR_KEYS)),
    )

    if (
        comp < MIN_COMPRESSION
        or struct < MIN_STRUCTURE
        or not healthy_structure_gate(df)
        or rs_pts < MIN_RS_POINTS
    ):
        return None

    if score >= GRADE_A_SCORE:
        grade = "A - Coil Ready"
    elif score >= MIN_PASS_SCORE:
        grade = "B - Valid Coil"
    else:
        return None

    return {
        "Score": round(score, 2),
        "Grade": grade,
        "Checks Met": f"{checks_passed}/6",
        "Fib Score": fib_score,
        "Parts": parts,
        "RS 63d": rs_rel,
        "Volume Contraction Ratio": volume_ratio,
        "MACD_Crossed": int(bool(crossed)),
        **prox_fields,
        **rsi_zone_fields(rsi_val),
        **volume_accumulation_fields(df),
    }


# =========================
# SCANNER CORE
# =========================

_WORKER_BENCHMARK_DF = None
_WORKER_ACTIVE = None
_WORKER_MIN_HISTORY = 0


def _print_frame(df: pd.DataFrame) -> None:
    """Pretty-print a frame; tolerate environments without tabulate."""
    try:
        print("\n" + df.to_markdown(index=False) + "\n")
    except (ImportError, AttributeError):
        print("\n" + df.to_string(index=False) + "\n")


def build_setup_row(
    symbol: str, df: pd.DataFrame, setup: dict, scan_mode: str
) -> dict:
    """Fill a SETUP_ROW_COLUMNS dict for the latest bar (live scan and backtest)."""
    latest = df.iloc[-1]
    asof = None
    raw_date = latest["Date"] if "Date" in df.columns else None
    if raw_date is not None and pd.notna(raw_date):
        asof = str(raw_date)[:10]

    close_v = float(latest["Close"])
    ema20_v = float(latest["EMA20"])
    ema50_v = float(latest["EMA50"])
    atr_v = float(latest["ATR"])
    fib618_v = float(latest["Fib_618"]) if pd.notna(latest.get("Fib_618")) else None
    fib786_v = float(latest["Fib_786"]) if pd.notna(latest.get("Fib_786")) else None
    rsi_v = float(latest["RSI"]) if pd.notna(latest.get("RSI")) else None

    geometry = coil_geometry_fields(df, atr_v)

    row = config.blank_setup_row()
    row.update({
        "Symbol": symbol.upper(),
        "Setup Type": "SETUP_LONG",
        "Source": "coiled_cobra",
        "Mode": scan_mode,
        "AsOf Date": asof,
        "Date": raw_date,
        "Close": round(close_v, 2),
        "EMA20": round(ema20_v, 2),
        "EMA50": round(ema50_v, 2),
        "ATR": round(atr_v, 2),
        "RSI": round(rsi_v, 2) if rsi_v is not None else None,
        "Swing Low": round(local_swing_low(df), 2),
        "Notes": setup["Grade"],
        "Score": setup["Score"],
        "Grade": setup["Grade"],
        "Checks Met": setup["Checks Met"],
        **pillar_row_fields(setup.get("Parts") or {}),
        **geometry,
        "Volume_Contraction_Ratio": setup.get("Volume Contraction Ratio"),
        "Fib 61.8%": round(fib618_v, 2) if fib618_v is not None else None,
        "Fib 78.6%": round(fib786_v, 2) if fib786_v is not None else None,
        "Fib Score": setup["Fib Score"],
        "MACD": round(float(latest["MACD"]), 2),
        "MACD Signal": round(float(latest["MACD_Signal"]), 2),
        "MACD_Crossed": setup.get("MACD_Crossed"),
        "RS 63d": setup.get("RS 63d"),
        "Pct_From_EMA20": round((close_v - ema20_v) / ema20_v, 4) if ema20_v else None,
        "Pct_From_EMA50": round((close_v - ema50_v) / ema50_v, 4) if ema50_v else None,
        "Pct_From_Fib618": round((close_v - fib618_v) / fib618_v, 4) if fib618_v else None,
        "Pct_From_Fib786": round((close_v - fib786_v) / fib786_v, 4) if fib786_v else None,
        "ATR_Pct": round(atr_v / close_v, 4) if close_v else None,
        "Proximity_Highs": (setup.get("Parts") or {}).get("proximity_highs"),
        "Dist_High_63_Pct": setup.get("Dist_High_63_Pct"),
        "Dist_High_63_ATR": setup.get("Dist_High_63_ATR"),
        "Dist_High_126_Pct": setup.get("Dist_High_126_Pct"),
        "Dist_High_126_ATR": setup.get("Dist_High_126_ATR"),
        "Dist_High_252_Pct": setup.get("Dist_High_252_Pct"),
        "Dist_High_252_ATR": setup.get("Dist_High_252_ATR"),
        "RSI_Healthy": setup.get("RSI_Healthy"),
        "RSI_Zone_Score": setup.get("RSI_Zone_Score"),
        "OBV_Coil_Slope": setup.get("OBV_Coil_Slope"),
        "Up_Volume_Ratio": setup.get("Up_Volume_Ratio"),
        "Volume_Trend_Ratio": setup.get("Volume_Trend_Ratio"),
    })
    return row


def build_live_setup_row(
    symbol: str, df: pd.DataFrame, setup: dict, scan_mode: str
) -> dict:
    """Alias for ``build_setup_row`` (live scanner)."""
    return build_setup_row(symbol, df, setup, scan_mode)


def _latest_csv_in_dir(directory: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    matches = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(".csv")
    ]
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def attach_weekly_confirmation(df_out: pd.DataFrame, scan_mode: str) -> pd.DataFrame:
    """Join latest weekly setups onto a daily scan; soft-boost ML preds. Never a gate."""
    out = df_out.copy()
    if "Weekly_Coil_Pass" not in out.columns:
        out["Weekly_Coil_Pass"] = 0
    if "Weekly_Score" not in out.columns:
        out["Weekly_Score"] = None
    if scan_mode != "daily" or out.empty or "Symbol" not in out.columns:
        return out

    weekly_dir = os.path.join(BASE_DIR, "data", "logs", "weekly")
    path = _latest_csv_in_dir(weekly_dir, "coiled_cobra_setups_")
    if not path:
        return out

    try:
        weekly = pd.read_csv(path)
    except Exception:
        return out
    if weekly.empty or "Symbol" not in weekly.columns:
        return out

    weekly["Symbol"] = weekly["Symbol"].astype(str).str.upper()
    score_col = "Score" if "Score" in weekly.columns else None
    keep = weekly.drop_duplicates("Symbol", keep="last")
    score_map = (
        pd.to_numeric(keep[score_col], errors="coerce")
        if score_col
        else pd.Series(dtype=float)
    )
    if score_col:
        score_map.index = keep["Symbol"]
        weekly_scores = dict(score_map)
    else:
        weekly_scores = {}
    weekly_syms = set(keep["Symbol"])

    out["Symbol"] = out["Symbol"].astype(str).str.upper()
    out["Weekly_Coil_Pass"] = out["Symbol"].map(lambda s: 1 if s in weekly_syms else 0)
    out["Weekly_Score"] = out["Symbol"].map(lambda s: weekly_scores.get(s))

    if "ML_Pred_Return" in out.columns:
        pred = pd.to_numeric(out["ML_Pred_Return"], errors="coerce")
        boost = out["Weekly_Coil_Pass"].eq(1)
        out.loc[boost & pred.notna(), "ML_Pred_Return"] = (
            pred[boost & pred.notna()] * WEEKLY_CONFIRM_BOOST
        ).round(4)
        if pred.notna().any():
            ranks = pd.to_numeric(out["ML_Pred_Return"], errors="coerce").rank(
                method="dense", ascending=False
            )
            out["ML_Rank"] = ranks.astype("Int64")
            out = out.sort_values(
                ["ML_Pred_Return", "Score"] if "Score" in out.columns else ["ML_Pred_Return"],
                ascending=False,
                na_position="last",
                kind="mergesort",
            ).reset_index(drop=True)
    return out


def _init_live_worker(scan_mode: str, active_tickers: list[str]) -> None:
    """Configure weekly/daily globals and cache QQQ once per worker process."""
    global _WORKER_BENCHMARK_DF, _WORKER_ACTIVE, _WORKER_MIN_HISTORY
    configure_mode(scan_mode)
    _WORKER_ACTIVE = set(active_tickers)
    _WORKER_MIN_HISTORY = max(LOOKBACK // 2, COIL_BARS + 40)
    _WORKER_BENCHMARK_DF = load_benchmark_frame(BENCHMARK, scan_mode)


def _scan_ticker_worker(path: str) -> tuple[Optional[dict], Optional[str]]:
    """Score one raw CSV. Returns (setup_row, rejection_reason)."""
    symbol = os.path.basename(path).split(".")[0].split("_")[0].upper()
    if _WORKER_ACTIVE is not None and symbol not in _WORKER_ACTIVE:
        return None, "inactive_ticker"

    try:
        df = pd.read_csv(path)
        df = config.validate_and_clean_ohlcv(df, require_volume=True)
        df = drop_in_progress_session(df)
    except ValueError:
        return None, "missing_columns"
    except Exception:
        return None, "execution_error"

    if len(df) < _WORKER_MIN_HISTORY:
        return None, "insufficient_history"

    try:
        df = add_macro_indicators(df)
        setup = evaluate_coiled_cobra(df, _WORKER_BENCHMARK_DF)
        if not setup:
            return None, "IGNORE"
        return build_live_setup_row(symbol, df, setup, mode), None
    except Exception:
        return None, "execution_error"


def run_scanner():
    logger.info(f"--- Scanning Coil Setups [{mode.upper()} MODE] ---")

    if not os.path.exists(ACTIVE_TICKERS_PATH):
        logger.error(f"Missing active tickers inventory file at {ACTIVE_TICKERS_PATH}")
        sys.exit(1)

    active_tickers = set(pd.read_csv(ACTIVE_TICKERS_PATH)["Ticker"].str.upper())
    logger.info(f"Loaded {len(active_tickers)} active tickers.")

    cfg = config.get_mode_config(mode)
    if not os.path.exists(cfg["raw_dir"]):
        logger.warning(f"Target raw directory empty or non-existent: {cfg['raw_dir']}")
        return

    raw_files = select_raw_paths(cfg["raw_dir"], cfg=cfg)
    logger.info(
        f"Found {len(raw_files)} {cfg['period']} {cfg['interval']} file(s) "
        f"(one per ticker) in {cfg['raw_dir']}"
    )

    if load_benchmark_frame(BENCHMARK, mode) is None:
        logger.warning(
            f"Benchmark {BENCHMARK} unavailable in {mode} raw data — RS pillar will score 0."
        )

    results = []
    rejection_counts: dict[str, int] = {}
    max_workers = os.cpu_count() or 1
    logger.info(f"Workers: {max_workers}")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_live_worker,
        initargs=(mode, sorted(active_tickers)),
    ) as ex:
        futures = {ex.submit(_scan_ticker_worker, path): path for path in raw_files}
        for fut in as_completed(futures):
            try:
                row, reason = fut.result()
            except Exception as e:
                logger.error(f"Error scoring {futures[fut]}: {e}")
                rejection_counts["execution_error"] = (
                    rejection_counts.get("execution_error", 0) + 1
                )
                continue
            if row:
                results.append(row)
            elif reason:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    if results:
        cobra_extra_columns = [
            "Volume_Contraction_Ratio",
            "Pivot_Price",
            "Distance_To_Pivot_Pct",
            "Proximity_Highs",
            "Dist_High_63_Pct",
            "Dist_High_63_ATR",
            "Dist_High_126_Pct",
            "Dist_High_126_ATR",
            "Dist_High_252_Pct",
            "Dist_High_252_ATR",
            "RSI_Healthy",
            "RSI_Zone_Score",
            "OBV_Coil_Slope",
            "Up_Volume_Ratio",
            "Volume_Trend_Ratio",
            "Coil_Width_Pctile",
            "MACD_Crossed",
            "Weekly_Coil_Pass",
            "Weekly_Score",
            "ML_Prob_Win_10d",
            "ML_Prob_Win_21d",
            "ML_Prob_Win_42d",
        ]
        output_columns = list(config.SETUP_ROW_COLUMNS) + [
            c for c in cobra_extra_columns if c not in config.SETUP_ROW_COLUMNS
        ]
        df_out = pd.DataFrame(results).reindex(columns=output_columns)

        try:
            from finance_vibe.ml_ranker import attach_ml_ranks, ML_PRED_COL
            df_out = attach_ml_ranks(df_out, mode)
            if df_out[ML_PRED_COL].notna().any():
                logger.info("ML ranks attached to scan results.")
            else:
                logger.info("No ML model available; ranking by Score.")
                df_out = df_out.sort_values(by="Score", ascending=False)
        except Exception as e:
            logger.warning(f"ML ranking skipped ({e}); ranking by Score.")
            df_out = df_out.sort_values(by="Score", ascending=False)

        df_out = attach_weekly_confirmation(df_out, mode)

        df_out = df_out.reindex(columns=output_columns)
        _print_frame(df_out)

        today = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(LOG_DIR, f"coiled_cobra_setups_{today}.csv")
        df_out.to_csv(out_path, index=False)
        logger.info(f"Archive logged successfully to: {out_path}")
    else:
        logger.warning(
            "No high-confluence Coiled Cobra coil setups detected across watchlists."
        )

    logger.info("Coil Scanner execution complete. Rejection Summary:")
    for k, v in rejection_counts.items():
        logger.info(f"  {k}: {v}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coiled Cobra live scanner (v2.2 coil → expansion)"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default=config.DEFAULT_MODE,
        choices=["weekly", "daily"],
        help="daily (default, primary) or weekly calibration",
    )
    args = parser.parse_args(argv)
    configure_mode(args.mode)
    run_scanner()
    return 0


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    raise SystemExit(main())
