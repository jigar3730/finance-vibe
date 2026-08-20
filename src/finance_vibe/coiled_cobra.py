"""Coiled Cobra live scanner: v2.1 granular coil → expansion scorecard.

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
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

# --- PACKAGE IMPORT ---
try:
    from finance_vibe import config
    from finance_vibe.analysis_engine import load_benchmark_frame, relative_strength
except ImportError:
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe.analysis_engine import load_benchmark_frame, relative_strength

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
    global RS_LOOKBACK, RS_RATIO_MA, RAW_DATA_DIR, LOG_DIR
    mode = scan_mode if scan_mode in ("weekly", "daily") else config.DEFAULT_MODE
    LOOKBACK = 252 if mode == "daily" else 52
    COIL_BARS = 30 if mode == "daily" else 8
    # Fallback floor when Coil_Low is missing: match the coil window.
    STRUCTURE_STOP_BARS = COIL_BARS
    RS_LOOKBACK = 63 if mode == "daily" else 13
    RS_RATIO_MA = 20 if mode == "daily" else 5
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


def drop_in_progress_session(df: pd.DataFrame) -> pd.DataFrame:
    """Drop today's last daily bar so a live scan does not score a partial session.

    Weekly ingest already drops the in-progress week. Daily has no equivalent
    unless the last Date is today.
    """
    if mode != "daily" or df.empty or "Date" not in df.columns:
        return df
    last = pd.to_datetime(df["Date"].iloc[-1], errors="coerce")
    if pd.isna(last):
        return df
    if last.date() == datetime.now().date():
        return df.iloc[:-1].copy()
    return df

# Soft pass floors (core six sum to 100; Fib bonus clipped in)
MIN_PASS_SCORE = 70
GRADE_A_SCORE = 85
MAX_SCORE = 100.0
# Hard gates (ready-to-run leaders, not lagging coils)
MIN_COMPRESSION = 5
MIN_STRUCTURE = 8
MIN_RS_POINTS = 12  # requires full RS pass (ratio > MA and positive rel-return)

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


def _poc_value_score(distance: float) -> float:
    """Tent map: 5 at POC, peak 8 at 1–3 bins, decay to 0 by 10 bins."""
    d = abs(float(distance))
    if d <= 1:
        return _interp_score(d, [(0.0, 5.0), (1.0, 8.0)])
    if d <= 3:
        return 8.0
    if d <= 6:
        return _interp_score(d, [(3.0, 8.0), (6.0, 4.0)])
    if d <= 10:
        return _interp_score(d, [(6.0, 4.0), (10.0, 0.0)])
    return 0.0


def pillar_row_fields(parts: dict[str, float]) -> dict:
    """Map internal Parts keys to setup-row / backtest CSV columns."""
    return {
        "Volume_Shelf": parts.get("volume_shelf"),
        "MACD_Compression": parts.get("macd_compression"),
        "Structure": parts.get("structure"),
        "RS_Score": parts.get("relative_strength"),
        "Coil_Width": parts.get("coil_width"),
        "MACD_Cross": parts.get("macd_cross"),
        "Fib_Bonus": parts.get("fib_bonus"),
    }


def coil_geometry_fields(df: pd.DataFrame, atr: float) -> dict:
    """Unscored coil measurements for trade-plan stops and research retunes."""
    if df.empty:
        return {
            "MACD_Spread_ATR": None,
            "Coil_Width_ATR": None,
            "Coil_High": None,
            "Coil_Low": None,
        }
    latest = df.iloc[-1]
    coil_n = min(COIL_BARS, len(df))
    coil_slice = df.iloc[-coil_n:]
    coil_high = float(coil_slice["High"].max())
    coil_low = float(coil_slice["Low"].min())
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
    }


def evaluate_volume_profile_shelf(
    df: pd.DataFrame, current_price: float, lookback=None
) -> float:
    """
    Auction-market volume shelf score (0-20).

    Rewards price sitting in a high-volume node with supportive topology —
    the accumulation zone before a coil expands.
    """
    if lookback is None:
        lookback = LOOKBACK
    recent_data = df.iloc[-lookback:] if len(df) >= lookback else df

    v_min = float(recent_data["Low"].min())
    v_max = float(recent_data["High"].max())

    if v_min == v_max:
        return 0.0

    bins = np.linspace(v_min, v_max, 31)
    close_array = recent_data["Close"].to_numpy().flatten()
    volume_array = recent_data["Volume"].to_numpy().flatten()

    binned_volume, bin_edges = np.histogram(
        close_array, bins=bins, weights=volume_array
    )

    price_bin = np.digitize([current_price], bin_edges)[0] - 1
    price_bin = max(0, min(price_bin, len(binned_volume) - 1))

    # Topology / liquidity gradient (max 8) — keep fractional HVN strength
    left_idx = max(0, price_bin - 1)
    right_idx = min(len(binned_volume) - 1, price_bin + 1)
    avg_neighbor_vol = (binned_volume[left_idx] + binned_volume[right_idx]) / 2
    current_vol = binned_volume[price_bin]
    if avg_neighbor_vol > 0:
        topology_score = min(8.0, (current_vol / avg_neighbor_vol) * 2.5)
    else:
        topology_score = 0.0

    # Auction value vs POC (max 8) — prefer near-value, not glued to POC
    poc_bin = int(np.argmax(binned_volume))
    value_score = _poc_value_score(abs(price_bin - poc_bin))

    # Close holding above bin center (max 4); decay toward 1 if below
    right_edge = bin_edges[min(price_bin + 1, len(bin_edges) - 1)]
    bin_center = (bin_edges[price_bin] + right_edge) / 2
    close_now = float(recent_data["Close"].iloc[-1])
    if close_now >= bin_center:
        behavior_score = 4.0
    else:
        half = max(bin_center - bin_edges[price_bin], 1e-12)
        t = min(1.0, max(0.0, (bin_center - close_now) / half))
        behavior_score = 4.0 - 3.0 * t

    return round(float(topology_score + value_score + behavior_score), 2)


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
    """Tight N-bar range vs ATR (0-15). Coiled energy before expansion."""
    if coil_bars is None:
        coil_bars = COIL_BARS
    if atr <= 0 or len(df) < coil_bars:
        return 0.0
    window = df.iloc[-coil_bars:]
    rng = float(window["High"].max() - window["Low"].min())
    width_atr = rng / atr
    # Weekly 8-bar coil ≈ 2 months; daily 30-bar ≈ 6 weeks
    return _interp_score(
        width_atr,
        [(0.0, 15.0), (4.0, 15.0), (6.0, 10.0), (8.0, 5.0), (10.0, 0.0)],
    )


def structure_score(df: pd.DataFrame) -> float:
    """Healthy trend structure for a leader coil (0-20).

    Prefers price holding a rising EMA50, ideally with EMA50 > EMA100.
    Replaces the old deep-markdown requirement that filtered out coiled leaders.
    """
    if len(df) < 3:
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

    if atr > 0 and len(df) >= 4 and pd.notna(df.iloc[-4].get("EMA50")):
        ema50_ago = float(df.iloc[-4]["EMA50"])
        rise_atr = (ema50 - ema50_ago) / atr
        if rise_atr > 0:
            score += round(min(6.0, 6.0 * rise_atr / 0.05), 2)
    else:
        prev_ema50 = float(df.iloc[-2]["EMA50"])
        if ema50 > prev_ema50:
            score += 6.0

    if ema100 is not None and pd.notna(ema100) and ema50 > float(ema100):
        if atr > 0:
            gap_atr = (ema50 - float(ema100)) / atr
            score += _interp_score(gap_atr, [(0.0, 3.0), (0.5, 6.0)])
        else:
            score += 6.0

    return round(min(20.0, score), 2)


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


def macd_cross_score(
    prev_macd: float,
    prev_signal: float,
    macd: float,
    macd_signal: float,
    atr: float,
) -> float:
    """Bullish MACD cross this bar (0-10), scaled by remaining tightness."""
    crossed = prev_macd <= prev_signal and macd > macd_signal
    if not crossed:
        return 0.0
    if atr <= 0:
        return 10.0
    spread = abs(macd - macd_signal) / atr
    factor = 1.0 - (spread - 0.05) / 0.25
    factor = min(1.0, max(0.60, factor))
    return round(10.0 * factor, 2)


# =========================
# SYSTEMATIC GRADING MATRIX (coil → expansion)
# =========================


def evaluate_coiled_cobra(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """100-point coil scorecard: catch compressed leaders before they expand.

    Pillars (core 100, Fib bonus clipped in):
      Volume shelf 20 · MACD compression 20 · Structure 20 ·
      Relative strength 15 · Coil width 15 · MACD cross 10 · Fib bonus 5
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

    parts: dict[str, float] = {}
    checks_passed = 0

    # 1. Volume shelf (0-20)
    vp = evaluate_volume_profile_shelf(df, current_price)
    parts["volume_shelf"] = vp
    if vp >= 10:
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

    # 5. Coil width (0-15)
    coil = coil_width_score(df, atr)
    parts["coil_width"] = coil
    if coil >= 10:
        checks_passed += 1

    # 6. Bullish MACD cross trigger (0-10) — quality-scaled if crossed this bar
    cross_pts = macd_cross_score(
        prev_macd, prev_macd_signal, macd, macd_signal, atr
    )
    parts["macd_cross"] = cross_pts
    if cross_pts > 0:
        checks_passed += 1

    # 7. Optional Fib bonus (0-5) — context only, not a gate
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

    score = min(MAX_SCORE, float(sum(parts.values())))

    # Hard gates for "ready to run":
    #   compression — coiled energy
    #   structure   — healthy / rising stack (not a broken decline)
    #   RS          — leading QQQ (kills BA/DG-style negative-RS coils)
    if (
        comp < MIN_COMPRESSION
        or struct < MIN_STRUCTURE
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


def build_live_setup_row(
    symbol: str, df: pd.DataFrame, setup: dict, scan_mode: str
) -> dict:
    """Fill a SETUP_ROW_COLUMNS dict for the latest bar, including coil geometry."""
    latest = df.iloc[-1]
    asof = None
    if "Date" in df.columns:
        raw_date = latest["Date"]
        asof = str(raw_date)[:10] if pd.notna(raw_date) else None

    close_v = float(latest["Close"])
    ema20_v = float(latest["EMA20"])
    ema50_v = float(latest["EMA50"])
    atr_v = float(latest["ATR"])
    fib618_v = float(latest["Fib_618"]) if pd.notna(latest.get("Fib_618")) else None
    fib786_v = float(latest["Fib_786"]) if pd.notna(latest.get("Fib_786")) else None

    row = config.blank_setup_row()
    row.update({
        "Symbol": symbol,
        "Setup Type": "SETUP_LONG",
        "Source": "coiled_cobra",
        "Mode": scan_mode,
        "AsOf Date": asof,
        "Close": round(close_v, 2),
        "EMA20": round(ema20_v, 2),
        "EMA50": round(ema50_v, 2),
        "ATR": round(atr_v, 2),
        "RSI": round(float(latest["RSI"]), 2) if pd.notna(latest["RSI"]) else None,
        "Swing Low": round(local_swing_low(df), 2),
        "Notes": setup["Grade"],
        "Score": setup["Score"],
        "Grade": setup["Grade"],
        "Checks Met": setup["Checks Met"],
        **pillar_row_fields(setup.get("Parts") or {}),
        **coil_geometry_fields(df, atr_v),
        "Fib 61.8%": round(fib618_v, 2) if fib618_v is not None else None,
        "Fib 78.6%": round(fib786_v, 2) if fib786_v is not None else None,
        "Fib Score": setup["Fib Score"],
        "MACD": round(float(latest["MACD"]), 2),
        "MACD Signal": round(float(latest["MACD_Signal"]), 2),
        "RS 63d": setup.get("RS 63d"),
        "Pct_From_EMA20": round((close_v - ema20_v) / ema20_v, 4) if ema20_v else None,
        "Pct_From_EMA50": round((close_v - ema50_v) / ema50_v, 4) if ema50_v else None,
        "Pct_From_Fib618": round((close_v - fib618_v) / fib618_v, 4) if fib618_v else None,
        "Pct_From_Fib786": round((close_v - fib786_v) / fib786_v, 4) if fib786_v else None,
        "ATR_Pct": round(atr_v / close_v, 4) if close_v else None,
    })
    return row


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

    if not os.path.exists(RAW_DATA_DIR):
        logger.warning(f"Target raw directory empty or non-existent: {RAW_DATA_DIR}")
        return

    raw_files = sorted(
        os.path.join(RAW_DATA_DIR, f)
        for f in os.listdir(RAW_DATA_DIR)
        if f.endswith(".csv")
    )
    logger.info(f"Found {len(raw_files)} historical files to analyze.")

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
        df_out = pd.DataFrame(results).reindex(columns=config.SETUP_ROW_COLUMNS)

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

        df_out = df_out.reindex(columns=config.SETUP_ROW_COLUMNS)
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
        description="Coiled Cobra live scanner (v2.1 coil → expansion)"
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
