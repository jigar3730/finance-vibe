"""Frozen copy of the breakout scanner's scalar rules, as of commit 534434d.

This is a test oracle, not a second implementation. ``scanners/breakout.py``
moved these rules to vectorized frame functions (one row per bar, for
``scan_history``) and kept the per-row API as thin wrappers over them. The
differential tests in ``test_breakout_history.py`` check that the port is
exact, including threshold edges that the golden fixture never reaches.

Do not co-edit this file when a rule changes on purpose. Delete it (and the
tests that import it) in that versioned change instead. Constants are frozen
as literals so a changed module constant is caught too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RVOL_DRYUP = 0.85
RVOL_CONFIRM = 1.50
RVOL_IGNITION = 1.00
PRE_BREAKOUT_ATR_MAX = 1.25
AT_RESISTANCE_ATR = 0.25

STATUS_PRE = "PRE_BREAKOUT"
STATUS_WATCH = "WATCH"
STATUS_DEV = "DEVELOPING"
STATUS_CONFIRMED = "BREAKOUT_CONFIRMED"
STATUS_TRIGGERED = "BREAKOUT_TRIGGERED"
STATUS_FAILED = "FAILED_BREAKOUT"

MTF_NO_UPTREND = {"BEARISH", "NEUTRAL"}


def _safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def trend_bull(df: pd.DataFrame | None, *, require_sma50: bool = True) -> bool | None:
    """Bull flag on the last bar of an enriched timeframe frame."""
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


def trend_bear(df: pd.DataFrame | None) -> bool | None:
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    sma20 = _safe_float(last.get("SMA20"))
    sma50 = _safe_float(last.get("SMA50"))
    if close is None or sma20 is None or sma50 is None:
        return None
    return bool(close < sma20 < sma50)


def _pctl_points(pctl, full, mid, bands):
    if pctl is None:
        return 0
    if pctl <= bands[0]:
        return full
    if pctl <= bands[1]:
        return mid
    if pctl <= 50:
        return max(1, mid // 2)
    return 0


def _mtf_label(feat) -> str:
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
    any_bear = any(b is True for b in bears)
    if any(known):
        return "DIVERGENT" if any_bear else "PARTIAL"
    return "BEARISH" if any_bear else "NEUTRAL"


def _primary_trend_label(feat) -> str:
    primary_bull = feat.daily_trend_bull if feat.has_daily else feat.weekly_trend_bull
    primary_bear = feat.daily_trend_bear if feat.has_daily else feat.weekly_trend_bear
    if primary_bull:
        return "BULLISH"
    if primary_bear:
        return "BEARISH"
    return "NEUTRAL"


def _volatility_label(feat) -> str:
    if feat.compression:
        return "COMPRESSING"
    expanding = (feat.bb_width_pctl is not None and feat.bb_width_pctl >= 70) or (
        feat.atr_pctl is not None and feat.atr_pctl >= 70
    )
    if expanding:
        return "EXPANDING"
    return "NORMAL"


def _volume_label(feat) -> str:
    if feat.volume_dryup or (feat.rvol20 is not None and feat.rvol20 < RVOL_DRYUP):
        return "DRYING_UP"
    if feat.rvol20 is not None and feat.rvol20 >= 1.20:
        return "EXPANDING"
    return "NORMAL"


def _momentum_label(feat) -> str:
    rsi, hist = feat.rsi_slope, feat.macd_hist_slope
    if rsi is not None and hist is not None:
        if rsi > 0 and hist > 0:
            return "ACCELERATING"
        if rsi < 0 and hist < 0:
            return "FADING"
    return "NEUTRAL"


def _structure_label(feat) -> str:
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


def _distance_label(feat) -> str:
    if feat.distance_resistance_atr is None:
        return ""
    return f"{feat.distance_resistance_atr:.1f} ATR"


def classify_states(feat) -> dict[str, str]:
    return {
        "Trend": _primary_trend_label(feat),
        "Volatility": _volatility_label(feat),
        "Volume State": _volume_label(feat),
        "Momentum": _momentum_label(feat),
        "Structure": _structure_label(feat),
        "Breakout Distance": _distance_label(feat),
        "MTF": _mtf_label(feat),
    }


def classify_status(feat, states: dict[str, str]) -> str:
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
        return STATUS_TRIGGERED

    early_coil = compressing or (feat.bb_width_pctl is not None and feat.bb_width_pctl <= 40)
    still_far = dist is not None and dist > PRE_BREAKOUT_ATR_MAX
    if trend_ok and early_coil and (accelerating or still_far) and not_through:
        if still_far:
            return STATUS_DEV
        return STATUS_WATCH

    if near and trend_ok and not_through:
        return STATUS_WATCH

    return STATUS_WATCH


def score_readiness(feat, states: dict[str, str]) -> dict:
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
    squeeze_pts = (
        6
        if feat.squeeze
        else (3 if feat.kc_width_pctl is not None and feat.kc_width_pctl <= 30 else 0)
    )
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

    dry_pts = (
        8 if feat.volume_dryup else (4 if feat.rvol20 is not None and feat.rvol20 < 1.0 else 0)
    )
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
        penalty += 20
    elif feat.wick_reject:
        penalty += 8
    if states["MTF"] == "DIVERGENT" or states["MTF"] in MTF_NO_UPTREND:
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
