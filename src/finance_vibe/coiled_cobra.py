import os
import sys
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

# --- PACKAGE IMPORT ---
try:
    from finance_vibe import config
    from finance_vibe.analysis_engine import (
        check_coiled_cobra_market_gate,
        load_benchmark_frame,
        relative_strength,
    )
except ImportError:
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe.analysis_engine import (
        check_coiled_cobra_market_gate,
        load_benchmark_frame,
        relative_strength,
    )

# =========================
# PROFILE CONFIGURATION
# =========================
if len(sys.argv) > 1 and sys.argv[1].lower() in ["weekly", "daily", "high_beta"]:
    mode = sys.argv[1].lower()
else:
    print("⚠️ Unknown mode parsed to scanner. Defaulting to 'weekly'.")
    mode = "weekly"

# Data timeframe may differ from the signal profile (high_beta -> daily OHLCV,
# its own log silo). Mirrors swing_scanner.py's former pattern.
_data_mode, _signal_mode = config.resolve_pipeline_mode(mode)
mode = _signal_mode  # scanner/planner Mode column = signal profile

# Timeframe-specific technical calibration (mutated by ``apply_timeframe``).
# high_beta reads the same daily bars as ``daily`` and shares its calibration;
# only the log silo differs (see LOG_DIR below).
_is_daily_bars = mode in ("daily", "high_beta")
LOOKBACK = 252 if _is_daily_bars else 52
# Coil window: how many bars define "the base" (weekly ≈ 2 months, daily ≈ 6 weeks)
COIL_BARS = 30 if _is_daily_bars else 8
# Local structural floor for dual-constraint stops (not the macro Fib lookback)
STRUCTURE_STOP_BARS = 10
# RS lookback in bars (weekly ≈ 1 quarter, daily ≈ 63 sessions)
RS_LOOKBACK = 63 if _is_daily_bars else 13
RS_RATIO_MA = 20 if _is_daily_bars else 5
BENCHMARK = "QQQ"
SPY_BENCHMARK = "SPY"

# Open-sky / extension allowances (daily-calibrated fractions; used on both TFs)
OPEN_SKY_PCT = 0.95          # Close >= 95% of 52-week / ATH → full space score
RS_FULL_SCORE = 0.15         # 63d/13w RS at/above this earns a full pillar score
RS_LEADER_EXT = 0.10         # RS above this = "leader" for extension-haircut purposes
RVOL_BONUS = 1.2             # RVOL threshold for the quiet-coil credit check


def local_swing_low(df: pd.DataFrame, bars: int = STRUCTURE_STOP_BARS) -> float:
    """Minimum Low over the last ``bars`` sessions (local consolidation floor)."""
    window = df.iloc[-bars:] if len(df) >= bars else df
    return float(window["Low"].min())


# =====================================================================
# RUBRIC v4.0 CALIBRATION
# =====================================================================
# Weekly-native periods per docs/handbook/coiled_cobra_rubric.md, scaled 5x
# for daily/high_beta bars (5 trading days ~= 1 week). This 5x rule is not
# arbitrary: 10w*5=50d, 30w*5=150d, 40w*5=200d land exactly on Minervini's
# classic daily 50/150/200-SMA trend template -- the rubric explicitly names
# Gate A as "the weekly analog" of that daily template, so the scaling is
# self-consistent rather than an independent daily calibration.
_WK_TO_BAR = 5 if _is_daily_bars else 1

# Trend-template EMAs (Gate A + Structure pillar). Prefixed TT_ to keep them
# distinct from the pre-existing bar-count EMA10/20/50/100 columns, which
# stay native-bar-length (used for Pct_From_EMA20/50 ML features, output
# columns, etc.) and are NOT week-scaled.
TT_EMA_S1 = 10 * _WK_TO_BAR     # structure stack: fast
TT_EMA_S2 = 20 * _WK_TO_BAR     # structure stack: mid
TT_EMA_FAST = 30 * _WK_TO_BAR   # Gate A fast (== Minervini 150-SMA on daily)
TT_EMA_SLOW = 40 * _WK_TO_BAR   # Gate A slow / structure's third anchor (== Minervini 200-SMA on daily)
TREND_RISING_LOOKBACK = 8 * _WK_TO_BAR   # bars EMA_SLOW must be rising over

# BBWidth percentile rolling window: rubric specifies 104-156 weeks (2-3y);
# 130w midpoint, same 5x daily scaling.
BBWIDTH_WINDOW = 130 * _WK_TO_BAR

# History floors. MIN_BARS_TO_EVALUATE: rubric's max(COIL_BARS+2, 60w).
# MIN_BARS_FULL_SCORE: rubric's 160w floor for ATH/trend-template/BBWidth
# context -- tickers between the two are flagged "Insufficient History"
# rather than silently scored on partial data.
MIN_BARS_TO_EVALUATE = max(COIL_BARS + 2, 60 * _WK_TO_BAR)
MIN_BARS_FULL_SCORE = 160 * _WK_TO_BAR

# Stage 1 hard-gate thresholds (Gate C reuses the same per-pillar thresholds
# that drive Gate D's Checks-Met counter -- see each pillar's docstring).
GATE_C_VOL_CONTRACTION_MIN = 12
GATE_C_STRUCTURE_MIN = 8
MIN_CHECKS_MET = 4     # of 6 scored pillars (Gate D)
# Lowered from 5 (original v4.0) to 4 after a 264-ticker/10y walk-forward
# backtest showed the >=5/6 cutoff had no measurable expectancy or win-rate
# edge over >=4/6 -- it just discarded ~65% of otherwise equal-or-better
# B/Watchlist-tier signal volume (see gate_d_ablation review, 2026-09-18).
# A/Actionable, the best-performing cohort, was unaffected by the threshold
# either way since Actionable already implies clearing most pillars.
N_SCORED_PILLARS = 6

# MACD is a small binary directional filter (not a scored pillar in v4.0).
MACD_DIRECTIONAL_PENALTY = 8

# Soft pass floors (scorecard still sums to 100)
MIN_PASS_SCORE = 70
GRADE_A_SCORE = 85


def _calibrate(is_daily_bars: bool) -> None:
    """Recompute every mode-derived constant from a single ``is_daily_bars``
    flag. Shared by module load and ``apply_timeframe`` so the two never
    drift out of sync.
    """
    global LOOKBACK, COIL_BARS, RS_LOOKBACK, RS_RATIO_MA, _WK_TO_BAR
    global TT_EMA_S1, TT_EMA_S2, TT_EMA_FAST, TT_EMA_SLOW, TREND_RISING_LOOKBACK
    global BBWIDTH_WINDOW, MIN_BARS_TO_EVALUATE, MIN_BARS_FULL_SCORE

    LOOKBACK = 252 if is_daily_bars else 52
    COIL_BARS = 30 if is_daily_bars else 8
    RS_LOOKBACK = 63 if is_daily_bars else 13
    RS_RATIO_MA = 20 if is_daily_bars else 5

    _WK_TO_BAR = 5 if is_daily_bars else 1
    TT_EMA_S1 = 10 * _WK_TO_BAR
    TT_EMA_S2 = 20 * _WK_TO_BAR
    TT_EMA_FAST = 30 * _WK_TO_BAR
    TT_EMA_SLOW = 40 * _WK_TO_BAR
    TREND_RISING_LOOKBACK = 8 * _WK_TO_BAR
    BBWIDTH_WINDOW = 130 * _WK_TO_BAR
    MIN_BARS_TO_EVALUATE = max(COIL_BARS + 2, 60 * _WK_TO_BAR)
    MIN_BARS_FULL_SCORE = 160 * _WK_TO_BAR


def apply_timeframe(tf: str) -> str:
    """Set every mode-derived calibration constant for ``weekly``, ``daily``,
    or ``high_beta``.

    Import-time defaults follow ``sys.argv`` (scanner CLI). Historical
    benchmarks and library callers should set the timeframe explicitly so
    daily 63-bar RS and 252-bar overhead windows are used. ``high_beta``
    shares ``daily``'s bar-frequency calibration (only the live pipeline's
    log silo differs; this function does not touch paths).
    """
    global mode
    tf_l = str(tf).lower()
    mode = tf_l if tf_l in ("daily", "high_beta") else "weekly"
    _calibrate(mode in ("daily", "high_beta"))
    return mode

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Raw OHLCV always comes from the data timeframe (_data_mode); high_beta reads
# the same daily silo as the ETF pipeline but gets its own LOG_DIR below so
# outputs don't collide.
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", _data_mode)
ACTIVE_TICKERS_PATH = os.path.join(BASE_DIR, "data", "active_tickers.csv")
LOG_DIR = config.get_log_dir(mode)

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
    """EMA stack, MACD, RSI, ATR, RVOL, BBWidth, and rolling Fib levels for coil scoring."""
    lookback = LOOKBACK if lookback is None else lookback
    out = df.copy()
    out["EMA10"] = ta.ema(out["Close"], length=10)
    out["EMA20"] = ta.ema(out["Close"], length=20)
    out["EMA50"] = ta.ema(out["Close"], length=50)
    out["EMA100"] = ta.ema(out["Close"], length=100)
    out["SMA50"] = ta.sma(out["Close"], length=50)

    # Rubric v4.0 trend-template EMAs (Gate A + Structure pillar) -- distinct
    # from the bar-count EMA10/20/50/100 above, which stay untouched since
    # they drive Pct_From_EMA20/50 ML features and other output columns.
    out["TT_EMA_S1"] = ta.ema(out["Close"], length=TT_EMA_S1)
    out["TT_EMA_S2"] = ta.ema(out["Close"], length=TT_EMA_S2)
    out["TT_EMA_FAST"] = ta.ema(out["Close"], length=TT_EMA_FAST)
    out["TT_EMA_SLOW"] = ta.ema(out["Close"], length=TT_EMA_SLOW)

    macd = ta.macd(out["Close"])
    out["MACD"] = macd["MACD_12_26_9"]
    out["MACD_Signal"] = macd["MACDs_12_26_9"]
    out["MACD_Hist"] = macd["MACDh_12_26_9"]

    out["RSI"] = ta.rsi(out["Close"], length=14)

    rolling_max = out["High"].rolling(window=lookback, min_periods=lookback).max()
    rolling_min = out["Low"].rolling(window=lookback, min_periods=lookback).min()
    out["Fib_786"] = rolling_max - ((rolling_max - rolling_min) * 0.786)
    out["Fib_618"] = rolling_max - ((rolling_max - rolling_min) * 0.618)

    out["ATR"] = ta.atr(out["High"], out["Low"], out["Close"], length=14)

    out["VOL_SMA20"] = ta.sma(out["Volume"], length=20)
    out["RVOL"] = out["Volume"] / out["VOL_SMA20"]

    # Bollinger Band Width (20-period, 2 std) and its percentile rank within
    # its own trailing BBWIDTH_WINDOW history -- true volatility contraction,
    # not the MACD-spread proxy v3.1 used. min_periods kept generous (a
    # quarter of the window) so shorter histories still get a reading.
    sma20 = ta.sma(out["Close"], length=20)
    std20 = out["Close"].rolling(20, min_periods=20).std()
    bbwidth = (4 * std20) / sma20.replace(0, np.nan)
    out["BBWidth"] = bbwidth
    bb_min_periods = max(40, BBWIDTH_WINDOW // 4)
    out["BBWidth_Pctile"] = (
        bbwidth.rolling(BBWIDTH_WINDOW, min_periods=bb_min_periods).rank(pct=True) * 100
    )

    # Keep early rows so RS / lookback windows still see full history.
    return out


# =========================
# STRUCTURAL LAYER FILTER (AMT)
# =========================


def _volume_shelf_once(df: pd.DataFrame, current_price: float, lookback: int) -> int:
    """Score one lookback window against its POC / HVN (0-15)."""
    recent_data = df.iloc[-lookback:] if len(df) >= lookback else df
    v_min = float(recent_data["Low"].min())
    v_max = float(recent_data["High"].max())
    if v_min == v_max:
        return 0

    bins = np.linspace(v_min, v_max, 31)
    close_array = recent_data["Close"].to_numpy().flatten()
    volume_array = recent_data["Volume"].to_numpy().flatten()
    binned_volume, bin_edges = np.histogram(
        close_array, bins=bins, weights=volume_array
    )

    price_bin = np.digitize([current_price], bin_edges)[0] - 1
    price_bin = max(0, min(price_bin, len(binned_volume) - 1))

    left_idx = max(0, price_bin - 1)
    right_idx = min(len(binned_volume) - 1, price_bin + 1)
    avg_neighbor_vol = (binned_volume[left_idx] + binned_volume[right_idx]) / 2
    current_vol = binned_volume[price_bin]
    topology_score = min(6.0, (current_vol / avg_neighbor_vol) * 2) if avg_neighbor_vol > 0 else 0.0

    poc_bin = int(np.argmax(binned_volume))
    distance_from_poc = abs(price_bin - poc_bin)
    if distance_from_poc <= 3:
        value_score = 6
    elif distance_from_poc <= 6:
        value_score = 3
    else:
        value_score = 0

    bin_center = (bin_edges[price_bin] + bin_edges[min(price_bin + 1, len(bin_edges) - 1)]) / 2
    behavior_score = 3 if float(recent_data["Close"].iloc[-1]) > bin_center else 1

    return int(round(topology_score + value_score + behavior_score))


def evaluate_volume_profile_shelf(
    df: pd.DataFrame, current_price: float, lookback=None
) -> int:
    """Auction-market volume shelf score (0-15).

    Rewards price sitting near the Point of Control / high-volume node.
    Takes the better of a short coil window and the full available history
    (previously capped at the 52-week LOOKBACK; now that 10y history is on
    hand, the long window uses everything).
    """
    short = 20
    long_lookback = len(df) if lookback is None else lookback
    return max(
        _volume_shelf_once(df, current_price, short),
        _volume_shelf_once(df, current_price, long_lookback),
    )


def macd_directional_penalty(macd: float) -> int:
    """Binary MACD directional filter (not a scored pillar in v4.0).

    v3.1 scored MACD-histogram "squeeze" as a 15-pt volatility proxy, which
    conflated momentum convergence with actual range contraction (now
    measured properly by ``vol_contraction_score``'s BBWidth percentile).
    MACD is demoted to a small penalty: net-negative momentum costs
    ``MACD_DIRECTIONAL_PENALTY`` points off the Stage-2 total.
    """
    return 0 if macd > 0 else MACD_DIRECTIONAL_PENALTY


def vol_contraction_score(df: pd.DataFrame) -> tuple[int, Optional[float]]:
    """BBWidth-percentile volatility contraction (0-25).

    Replaces v3.1's MACD-spread "squeeze" proxy and ATR-ratio coil_width with
    a true range-contraction measure: current Bollinger Band Width's
    percentile rank within its own trailing ``BBWIDTH_WINDOW`` history.
    Halves the points unless that percentile has been declining over the
    trailing ``COIL_BARS`` -- contraction must be a trend, not a snapshot.
    """
    if df.empty or "BBWidth_Pctile" not in df.columns:
        return 0, None
    pct = df["BBWidth_Pctile"].iloc[-1]
    if pd.isna(pct):
        return 0, None
    pct_f = float(pct)

    if pct_f <= 10:
        base = 25
    elif pct_f <= 20:
        base = 20
    elif pct_f <= 35:
        base = 12
    elif pct_f <= 50:
        base = 6
    else:
        base = 0

    if base > 0 and len(df) > COIL_BARS:
        prior = df["BBWidth_Pctile"].iloc[-(COIL_BARS + 1)]
        if pd.notna(prior) and pct_f > float(prior):
            base = base // 2

    return base, pct_f


def _trend_extension_penalty(pct_from_slow: float, rs_rel: Optional[float]) -> int:
    """Soft haircut for extension above TT_EMA_SLOW (tightened vs v3.1's
    EMA50-based bands). Floors the whole structure pillar at 0 past 40%
    extension -- the rubric treats that as stage-2 markup, not a fresh coil.
    """
    if pct_from_slow <= 0.20:
        return 0
    leader = rs_rel is not None and rs_rel > RS_LEADER_EXT
    if pct_from_slow <= 0.30:
        t = (pct_from_slow - 0.20) / 0.10
        return int(round(4 * t))
    if pct_from_slow <= 0.40:
        t = (pct_from_slow - 0.30) / 0.10
        max_pen = 8 if leader else 12
        return int(round(4 + (max_pen - 4) * t))
    return 999  # forces structure_score's max(0, ...) floor


def structure_score(df: pd.DataFrame, rs_rel: Optional[float] = None) -> int:
    """MA alignment and long-term-EMA extension proximity (0-20).

    Requires a strict ascending EMA hierarchy across all four trend-template
    EMAs -- TT_EMA_S1 >= 0.98x TT_EMA_S2 >= 0.98x TT_EMA_FAST >= 0.98x
    TT_EMA_SLOW, else 0. TT_EMA_FAST (30w) is included so a mid-stack hole
    (e.g. TT_EMA_S2 dipping below TT_EMA_FAST) can't slip through -- the
    original v4.0 check only compared S1/S2/SLOW and left FAST unconstrained,
    letting disordered stacks pass Gate A + Gate C despite not being a clean
    ascending fan. This hard requirement is also Gate C's structural half
    (see ``GATE_C_STRUCTURE_MIN``).
    """
    if len(df) < 2:
        return 0
    latest = df.iloc[-1]
    close = float(latest["Close"])
    s1 = latest.get("TT_EMA_S1")
    s2 = latest.get("TT_EMA_S2")
    fast = latest.get("TT_EMA_FAST")
    slow = latest.get("TT_EMA_SLOW")
    if any(v is None or pd.isna(v) for v in (s1, s2, fast, slow)):
        return 0
    s1, s2, fast, slow = float(s1), float(s2), float(fast), float(slow)

    if not (s1 >= 0.98 * s2 and s2 >= 0.98 * fast and fast >= 0.98 * slow):
        return 0

    score = 10
    if close >= 0.98 * s2:
        score += 5
    if s2 > slow:
        score += 5

    pct_from_slow = (close - slow) / slow if slow else 0.0
    score -= _trend_extension_penalty(pct_from_slow, rs_rel)
    return max(0, score)


def rvol_trigger_score(df: pd.DataFrame) -> tuple[int, Optional[float]]:
    """Breakout relative-volume bonus (0-10). Additive — never a drop.

    RVOL ≥ 1.2 is the bonus trigger. Sub-1.0 RVOL on a tight coil is valid
    compression and simply scores 0 on this pillar.
    """
    if df.empty or "RVOL" not in df.columns:
        return 0, None
    rvol = df.iloc[-1].get("RVOL")
    if rvol is None or pd.isna(rvol):
        return 0, None
    rvol_f = float(rvol)
    if rvol_f >= 2.0:
        pts = 10
    elif rvol_f >= 1.5:
        pts = 8
    elif rvol_f >= RVOL_BONUS:
        pts = 6
    elif rvol_f >= 1.0:
        pts = 4
    else:
        pts = 0
    return pts, rvol_f


def overhead_clearance_score(
    df: pd.DataFrame,
    price: float,
    atr: float,
    lookback: int = 50,
) -> float:
    """Room to run (0-10). Open sky near the lookback high or ATH scores a
    full 10. Fib levels are informational-only (see the v4.0 rubric doc) --
    this pillar is pure price-structure clearance.
    """
    if df.empty:
        return 0.0
    window = df.iloc[-lookback:] if len(df) >= lookback else df
    if window.empty:
        return 0.0
    high_52 = float(window["High"].max())
    ath = float(df["High"].max())
    # Open sky: within 5% of the lookback or all-time high — full points.
    if high_52 > 0 and price >= OPEN_SKY_PCT * high_52:
        return 10.0
    if ath > 0 and price >= OPEN_SKY_PCT * ath:
        return 10.0
    # Local open sky: within 5% of the recent range high (RS-lookback shelf).
    local_n = RS_LOOKBACK if RS_LOOKBACK else 63
    local = df.iloc[-local_n:] if len(df) >= local_n else df
    local_high = float(local["High"].max()) if not local.empty else 0.0
    if local_high > 0 and price >= OPEN_SKY_PCT * local_high:
        return 10.0
    if atr <= 0:
        return 0.0
    if high_52 <= price or local_high <= price:
        return 10.0
    room_atr = (min(high_52, local_high) - price) / atr
    if room_atr >= 3.0:
        return 8.0
    if room_atr >= 2.0:
        return 5.0
    if room_atr >= 1.0:
        return 2.0
    return 0.0


def _rs_line_new_high(
    stock_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    as_of=None,
    lookback: int = 13,
) -> bool:
    """True when the stock/benchmark ratio (RS line) is at its own trailing
    ``lookback`` high on the as-of bar -- RS-line cresting into a base, a
    leading institutional-accumulation tell not checked at all in v3.1.
    """
    if benchmark_df is None or benchmark_df.empty:
        return False
    s = stock_df[["Date", "Close"]].copy()
    s["Date"] = pd.to_datetime(s["Date"])
    b = benchmark_df[["Date", "Close"]].rename(columns={"Close": "Bench"}).copy()
    b["Date"] = pd.to_datetime(b["Date"])
    if as_of is not None:
        cutoff = pd.to_datetime(as_of)
        s = s[s["Date"] <= cutoff]
        b = b[b["Date"] <= cutoff]
    merged = s.merge(b, on="Date", how="inner").sort_values("Date")
    if len(merged) < lookback:
        return False
    ratio = merged["Close"].astype(float) / merged["Bench"].astype(float)
    window = ratio.tail(lookback)
    return bool(window.iloc[-1] >= window.max() - 1e-12)


def relative_strength_score(
    stock_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    as_of=None,
) -> tuple[int, Optional[float]]:
    """Relative strength vs QQQ (0-20), smoothed bands + RS-line-new-high bonus.

    Replaces v3.1's flat "-15% to 0%" plateau with a linear ramp so a stock
    lagging by 1% isn't scored identically to one lagging by 14%, and adds a
    +2 bonus when the RS line is cresting to a new lookback high.
    """
    if benchmark_df is None:
        return 0, None
    ok, rel = relative_strength(
        stock_df, benchmark_df, as_of=as_of,
        lookback=RS_LOOKBACK, ratio_ma_bars=RS_RATIO_MA,
    )
    if rel is None:
        return 0, None

    if rel >= RS_FULL_SCORE:
        pts = 20.0
    elif ok and rel > 0.10:
        pts = 18.0
    elif ok and rel > 0:
        pts = 14.0
    elif rel > 0:
        # ratio <= its MA (choppy) but RS still positive -- linear 6 -> 12;
        # clamped flat at 12 beyond +10% (rubric leaves that combination
        # undefined, this is the interpolation choice made here).
        pts = 6.0 + min(rel, 0.10) / 0.10 * 6.0
    elif rel > -0.15:
        pts = (rel + 0.15) / 0.15 * 6.0
    else:
        pts = 0.0

    if pts > 0 and _rs_line_new_high(stock_df, benchmark_df, as_of=as_of, lookback=RS_LOOKBACK):
        pts = min(20.0, pts + 2.0)

    return int(round(pts)), rel


# =========================
# SYSTEMATIC GRADING MATRIX (coil → expansion)
# =========================


def evaluate_coiled_cobra(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    *,
    spy_df: Optional[pd.DataFrame] = None,
    qqq_df: Optional[pd.DataFrame] = None,
    apply_market_gate: bool = True,
    include_rejects: bool = False,
) -> Optional[dict]:
    """100-point coil scorecard v4.0: catch compressed leaders before they expand.

    Pillars (v4.0): Volatility contraction 25 (BBWidth percentile) ·
    Structure 20 · Relative strength 20 · Volume shelf 15 ·
    Overhead clearance 10 · RVOL trigger 10. MACD is a small (-8) directional
    penalty, not a scored pillar.

    Four Stage-1 hard gates (all must pass, else rejected):
      A. Long-term trend template (TT_EMA_FAST/SLOW stack, slow rising)
      B. Ticker market gate (Close vs EMA50, 63d/13w RS) -- unchanged from v3.1
      C. Coil integrity (vol_contraction AND structure each independently
         clear their own Checks-Met threshold)
      D. Breadth (Checks Met >= MIN_CHECKS_MET of N_SCORED_PILLARS)

    Below ``MIN_BARS_FULL_SCORE`` bars of history, returns None unconditionally
    (not scored, regardless of ``include_rejects``) -- see the rubric's
    "Insufficient History" status.

    Full spec: docs/handbook/coiled_cobra_rubric.md (v4.0).
    """
    if len(df) < MIN_BARS_FULL_SCORE:
        return None

    latest = df.iloc[-1]
    required_cols = ("Close", "EMA50", "MACD", "ATR", "TT_EMA_S1", "TT_EMA_S2", "TT_EMA_FAST", "TT_EMA_SLOW")
    if any(pd.isna(latest.get(col)) for col in required_cols):
        return None

    current_price = float(latest["Close"])
    atr = float(latest["ATR"])
    macd = float(latest["MACD"])
    as_of = latest["Date"] if "Date" in df.columns else None
    ema50_f = float(latest["EMA50"])

    gate_fails: list[str] = []

    # --- Gate A: long-term trend template ---------------------------------
    tt_fast, tt_slow = float(latest["TT_EMA_FAST"]), float(latest["TT_EMA_SLOW"])
    prior_slow = (
        df["TT_EMA_SLOW"].iloc[-(TREND_RISING_LOOKBACK + 1)]
        if len(df) > TREND_RISING_LOOKBACK else None
    )
    trend_ok = (
        prior_slow is not None and pd.notna(prior_slow)
        and current_price > tt_fast > tt_slow
        and tt_slow > float(prior_slow)
    )
    if not trend_ok:
        gate_fails.append("A")

    # --- Stage 2: six scored pillars ---------------------------------------
    parts: dict[str, float] = {}
    checks_passed = 0

    vol_c, bbwidth_pct = vol_contraction_score(df)
    parts["vol_contraction"] = vol_c
    if vol_c >= GATE_C_VOL_CONTRACTION_MIN:
        checks_passed += 1

    rs_pts, rs_rel = relative_strength_score(df, benchmark_df, as_of=as_of)
    parts["relative_strength"] = rs_pts
    if rs_pts >= 14:
        checks_passed += 1

    struct = structure_score(df, rs_rel=rs_rel)
    parts["structure"] = struct
    if struct >= GATE_C_STRUCTURE_MIN:
        checks_passed += 1

    vp = evaluate_volume_profile_shelf(df, current_price)
    parts["volume_shelf"] = vp
    if vp >= 8:
        checks_passed += 1

    overhead = overhead_clearance_score(df, current_price, atr, lookback=LOOKBACK)
    parts["overhead_clearance"] = overhead
    if overhead >= 5:
        checks_passed += 1

    # RVOL bonus — never zeroed by the gate. Quiet volume on a confirmed
    # tight coil (vol_contraction >= 20) is valid compression, credited 4.
    rvol_pts, rvol = rvol_trigger_score(df)
    if rvol_pts == 0 and vol_c >= 20 and rvol is not None and rvol < RVOL_BONUS:
        rvol_pts = 4
    parts["rvol_trigger"] = rvol_pts
    if rvol_pts >= 6:
        checks_passed += 1

    macd_penalty = macd_directional_penalty(macd)
    score = max(0.0, sum(parts.values()) - macd_penalty)

    # --- Gate B: ticker market gate (unchanged from v3.1) ------------------
    gate_ok = True
    if apply_market_gate:
        gate_ok = check_coiled_cobra_market_gate(
            spy_df=spy_df,
            qqq_df=qqq_df if qqq_df is not None else benchmark_df,
            as_of=as_of,
            close=current_price,
            ema50=ema50_f,
            rs_63d=rs_rel,
        )
    if not gate_ok:
        gate_fails.append("B")

    # --- Gate C: coil integrity (structure/vol_contraction each independently) ---
    if vol_c < GATE_C_VOL_CONTRACTION_MIN or struct < GATE_C_STRUCTURE_MIN:
        gate_fails.append("C")

    # --- Gate D: breadth -----------------------------------------------------
    if checks_passed < MIN_CHECKS_MET:
        gate_fails.append("D")

    all_gates_ok = not gate_fails
    passes_threshold = score >= MIN_PASS_SCORE

    # --- Stage 3: Actionable vs Watchlist tiering ---------------------------
    coil_window = df.iloc[-(COIL_BARS + 1):-1] if len(df) > COIL_BARS else df.iloc[:-1]
    coil_high = float(coil_window["High"].max()) if not coil_window.empty else None
    breaking_out = coil_high is not None and current_price > coil_high
    tier = None
    if all_gates_ok and passes_threshold:
        tier = "Actionable" if (rvol_pts >= 6 and breaking_out) else "Watchlist"

    if not all_gates_ok:
        grade = f"Rejected - Gate Fail ({'/'.join(gate_fails)})"
    elif not passes_threshold:
        grade = "Rejected - Below Threshold"
    elif score >= GRADE_A_SCORE:
        grade = "A - Coil Ready" if tier == "Actionable" else "A - Watch"
    else:
        grade = "B - Valid Coil" if tier == "Actionable" else "B - Watch"

    result = {
        "Score": round(score, 2),
        "Grade": grade,
        "Tier": tier,
        "Checks Met": f"{checks_passed}/{N_SCORED_PILLARS}",
        "Fib Score": 0.0,
        "Parts": parts,
        "RS 63d": rs_rel,
        "RVOL": None if rvol is None else round(rvol, 4),
        "Market Gate": gate_ok,
        "BBWidth Pctile": None if bbwidth_pct is None else round(bbwidth_pct, 2),
    }

    if include_rejects:
        return result
    if not all_gates_ok or not passes_threshold:
        return None
    return result


def evaluate_as_of(
    df: pd.DataFrame,
    as_of,
    benchmark_df: Optional[pd.DataFrame] = None,
    *,
    spy_df: Optional[pd.DataFrame] = None,
    include_rejects: bool = True,
) -> Optional[dict]:
    """Score the last bar on or before *as_of* (causal). Used by benchmarks."""
    if df.empty or "Date" not in df.columns:
        return None
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    cutoff = pd.to_datetime(as_of)
    work = work[work["Date"] <= cutoff]
    if len(work) < MIN_BARS_FULL_SCORE:
        return None
    try:
        work = add_macro_indicators(work)
    except Exception:
        return None
    bench = None
    if benchmark_df is not None:
        bench = benchmark_df.copy()
        bench["Date"] = pd.to_datetime(bench["Date"])
        bench = bench[bench["Date"] <= cutoff]
    spy_cut = None
    if spy_df is not None:
        spy_cut = spy_df.copy()
        spy_cut["Date"] = pd.to_datetime(spy_cut["Date"])
        spy_cut = spy_cut[spy_cut["Date"] <= cutoff]
    return evaluate_coiled_cobra(
        work,
        bench,
        spy_df=spy_cut,
        qqq_df=bench,
        include_rejects=include_rejects,
    )


# =========================
# SCANNER CORE
# =========================


def run_scanner(as_of: str | None = None):
    """Scan the active universe; ``as_of`` (YYYY-MM-DD) replays a past date.

    With ``as_of`` every bar not complete on that date is dropped (ticker and
    benchmark frames alike), the archive is stamped with ``as_of``, and ML
    ranking is skipped (a model trained later would leak the future).
    """
    logger.info(f"--- STEP 5: Scanning Coil Setups [{mode.upper()} MODE] ---")
    weekly_bars = not _is_daily_bars
    if as_of:
        logger.info(f"AS-OF replay: {as_of} (bars completed on/before this date only)")

    if not os.path.exists(ACTIVE_TICKERS_PATH):
        logger.error(f"Missing active tickers inventory file at {ACTIVE_TICKERS_PATH}")
        sys.exit(1)

    active_tickers = set(pd.read_csv(ACTIVE_TICKERS_PATH)["Ticker"].str.upper())
    logger.info(f"Loaded {len(active_tickers)} active tickers into Matrix Framework.")

    if not os.path.exists(RAW_DATA_DIR):
        logger.warning(f"Target raw directory empty or non-existent: {RAW_DATA_DIR}")
        return

    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    logger.info(f"Found {len(raw_files)} historical files to analyze in target silo.")

    # Benchmarks live in the data timeframe's raw silo (_data_mode), not the
    # signal profile -- high_beta reads the same daily QQQ/SPY as `daily`.
    qqq_df = load_benchmark_frame(BENCHMARK, _data_mode)
    spy_df = load_benchmark_frame(SPY_BENCHMARK, _data_mode)
    if as_of:
        qqq_df = config.cut_to_as_of(qqq_df, as_of, weekly=weekly_bars) if qqq_df is not None else None
        spy_df = config.cut_to_as_of(spy_df, as_of, weekly=weekly_bars) if spy_df is not None else None
    if qqq_df is None:
        logger.warning(
            f"Benchmark {BENCHMARK} unavailable in {_data_mode} raw data — RS pillar will score 0."
        )
    if spy_df is None:
        logger.warning(
            f"Benchmark {SPY_BENCHMARK} unavailable in {_data_mode} raw data — market gate uses QQQ only."
        )

    results = []
    rejection_counts = {}

    # Cheap file-level pre-filter (evaluate_coiled_cobra's own MIN_BARS_FULL_SCORE
    # check is the real "not scored below this floor" gate; this just skips
    # files with almost no data before paying the indicator-computation cost).
    min_required_history = MIN_BARS_TO_EVALUATE

    for file in raw_files:
        symbol = file.split(".")[0].split("_")[0].upper()

        if symbol not in active_tickers:
            rejection_counts["inactive_ticker"] = (
                rejection_counts.get("inactive_ticker", 0) + 1
            )
            continue

        path = os.path.join(RAW_DATA_DIR, file)
        df = pd.read_csv(path)

        try:
            df = config.validate_and_clean_ohlcv(df, require_volume=True)
        except ValueError:
            rejection_counts["missing_columns"] = (
                rejection_counts.get("missing_columns", 0) + 1
            )
            continue

        if as_of:
            df = config.cut_to_as_of(df, as_of, weekly=weekly_bars)

        if len(df) < min_required_history:
            rejection_counts["insufficient_history"] = (
                rejection_counts.get("insufficient_history", 0) + 1
            )
            continue

        try:
            df = add_macro_indicators(df)
            setup = evaluate_coiled_cobra(
                df, qqq_df, spy_df=spy_df, qqq_df=qqq_df
            )

            if not setup:
                rejection_counts["IGNORE"] = rejection_counts.get("IGNORE", 0) + 1
                continue

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
                "Mode": mode,
                "AsOf Date": asof,
                "Close": round(close_v, 2),
                "EMA20": round(ema20_v, 2),
                "EMA50": round(ema50_v, 2),
                "ATR": round(atr_v, 2),
                "RSI": round(float(latest["RSI"]), 2) if pd.notna(latest["RSI"]) else None,
                # Local 10-session floor for dual-constraint stops (not year Fib).
                "Swing Low": round(local_swing_low(df), 2),
                "Notes": setup["Grade"],
                "Score": setup["Score"],
                "Grade": setup["Grade"],
                "Tier": setup.get("Tier"),
                "Checks Met": setup["Checks Met"],
                "Fib 61.8%": round(fib618_v, 2) if fib618_v is not None else None,
                "Fib 78.6%": round(fib786_v, 2) if fib786_v is not None else None,
                "Fib Score": setup["Fib Score"],
                "MACD": round(float(latest["MACD"]), 2),
                "MACD Signal": round(float(latest["MACD_Signal"]), 2),
                "RS 63d": setup.get("RS 63d"),
                # Pre-signal ML features (parity with backtest / training).
                "Pct_From_EMA20": round((close_v - ema20_v) / ema20_v, 4) if ema20_v else None,
                "Pct_From_EMA50": round((close_v - ema50_v) / ema50_v, 4) if ema50_v else None,
                "Pct_From_Fib618": round((close_v - fib618_v) / fib618_v, 4) if fib618_v else None,
                "Pct_From_Fib786": round((close_v - fib786_v) / fib786_v, 4) if fib786_v else None,
                "ATR_Pct": round(atr_v / close_v, 4) if close_v else None,
                "RVOL": setup.get("RVOL"),
                "Market Gate": setup.get("Market Gate"),
                "Regime OK": setup.get("Market Gate"),
            })
            results.append(row)

        except Exception as e:
            logger.error(f"Error scoring {symbol}: {str(e)}")
            rejection_counts["execution_error"] = (
                rejection_counts.get("execution_error", 0) + 1
            )

    today = config.run_stamp(as_of)
    out_path = os.path.join(LOG_DIR, f"coiled_cobra_setups_{today}.csv")
    df_out = pd.DataFrame(results).reindex(columns=config.SETUP_ROW_COLUMNS)

    if results and as_of:
        logger.info("ML ranking skipped for an as-of replay; ranking by Score.")
        df_out = df_out.sort_values(by="Score", ascending=False)
        print("\n" + df_out.to_markdown(index=False) + "\n")
    elif results:
        # Attach offline-model ranks (soft signal). Falls back to Score sort
        # when no model artifact is available or features are unusable.
        try:
            from finance_vibe.ml_ranker import attach_ml_ranks, ML_PRED_COL
            df_out = attach_ml_ranks(df_out, mode)
            if df_out[ML_PRED_COL].notna().any():
                logger.info("ML ranks attached to scan results.")
            else:
                logger.info("ML ranking inactive (disabled or no valid model); ranking by Score.")
                df_out = df_out.sort_values(by="Score", ascending=False)
        except Exception as e:
            logger.warning(f"ML ranking skipped ({e}); ranking by Score.")
            df_out = df_out.sort_values(by="Score", ascending=False)

        df_out = df_out.reindex(columns=config.SETUP_ROW_COLUMNS)
        print("\n" + df_out.to_markdown(index=False) + "\n")
    else:
        logger.warning(
            "No high-confluence Coiled Cobra coil setups detected across watchlists."
        )

    df_out.to_csv(out_path, index=False)
    logger.info(f"Archive logged successfully to: {out_path} ({len(df_out)} setup(s))")

    logger.info("Coil Scanner execution complete. Rejection Summary:")
    for k, v in rejection_counts.items():
        logger.info(f"  {k}: {v}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run_scanner(as_of=config.parse_as_of())
