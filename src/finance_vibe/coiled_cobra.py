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
EMA50_SOFT_EXT = 0.25        # no structure penalty at or below this extension
EMA50_MOMENTUM_EXT = 0.50    # RS leaders may extend this far with a scaled haircut
RS_FULL_SCORE = 0.15         # 63d RS at/above this earns a full 15 regardless of chop
RS_LEADER_EXT = 0.10         # RS above this unlocks the 0.50 EMA50 allowance
COIL_FULL_ATR = 1.5          # range / ATR for a full coil score
COIL_PARTIAL_ATR = 2.2       # last width that still earns partial coil points
RVOL_BONUS = 1.2             # additive breakout-volume bonus (not a drop)


def local_swing_low(df: pd.DataFrame, bars: int = STRUCTURE_STOP_BARS) -> float:
    """Minimum Low over the last ``bars`` sessions (local consolidation floor)."""
    window = df.iloc[-bars:] if len(df) >= bars else df
    return float(window["Low"].min())

# Soft pass floors (scorecard still sums to 100)
MIN_PASS_SCORE = 70
GRADE_A_SCORE = 85
# Soft check floors (counted in Checks Met; not binary drops)
MIN_COMPRESSION = 5
MIN_STRUCTURE = 8
MIN_RS_POINTS = 12
# Retained for CSV/docs compatibility. v3.1 no longer subtracts this — Market
# Gate only filters Close < EMA50 or negative 63d RS.
MACRO_PENALTY = 0


def apply_timeframe(tf: str) -> str:
    """Set coil / RS / shelf lookbacks for ``weekly``, ``daily``, or ``high_beta``.

    Import-time defaults follow ``sys.argv`` (scanner CLI). Historical
    benchmarks and library callers should set the timeframe explicitly so
    daily 63-bar RS and 252-bar overhead windows are used. ``high_beta``
    shares ``daily``'s bar-frequency calibration (only the live pipeline's
    log silo differs; this function does not touch paths).
    """
    global mode, LOOKBACK, COIL_BARS, RS_LOOKBACK, RS_RATIO_MA
    tf_l = str(tf).lower()
    mode = tf_l if tf_l in ("daily", "high_beta") else "weekly"
    is_daily_bars = mode in ("daily", "high_beta")
    LOOKBACK = 252 if is_daily_bars else 52
    COIL_BARS = 30 if is_daily_bars else 8
    RS_LOOKBACK = 63 if is_daily_bars else 13
    RS_RATIO_MA = 20 if is_daily_bars else 5
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
    """EMA stack, MACD, RSI, ATR, RVOL, and rolling Fib levels for coil scoring."""
    lookback = LOOKBACK if lookback is None else lookback
    out = df.copy()
    out["EMA10"] = ta.ema(out["Close"], length=10)
    out["EMA20"] = ta.ema(out["Close"], length=20)
    out["EMA50"] = ta.ema(out["Close"], length=50)
    out["EMA100"] = ta.ema(out["Close"], length=100)
    out["SMA50"] = ta.sma(out["Close"], length=50)

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

        # Keep early rows so RS / 52-week windows still see full history.
    return out


# =========================
# STRUCTURAL LAYER FILTER (AMT)
# =========================


def _volume_shelf_once(df: pd.DataFrame, current_price: float, lookback: int) -> int:
    """Score one lookback window against its POC / HVN."""
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
    if avg_neighbor_vol > 0:
        topology_score = min(8, max(3, int((current_vol / avg_neighbor_vol) * 2.5)))
    else:
        topology_score = 3

    poc_bin = int(np.argmax(binned_volume))
    distance_from_poc = abs(price_bin - poc_bin)
    if distance_from_poc <= 3:
        value_score = 8
    elif distance_from_poc <= 6:
        value_score = 6
    elif price_bin >= poc_bin:
        value_score = 5  # launched from / holding above the node
    elif distance_from_poc <= 10:
        value_score = 4  # accumulating under value
    else:
        value_score = 2

    bin_center = (bin_edges[price_bin] + bin_edges[min(price_bin + 1, len(bin_edges) - 1)]) / 2
    behavior_score = 4 if float(recent_data["Close"].iloc[-1]) > bin_center or price_bin >= poc_bin else 2
    return int(topology_score + value_score + behavior_score)


def evaluate_volume_profile_shelf(
    df: pd.DataFrame, current_price: float, lookback=None
) -> int:
    """
    Auction-market volume shelf score (0-20).

    Rewards price sitting near the Point of Control / high-volume node —
    proximity to the recent coil shelf, not clearance of a historical peak.
    Takes the better of a short coil window and the RS lookback.
    """
    if lookback is None:
        lookback = RS_LOOKBACK
    short = 20 if lookback >= 20 else max(5, lookback)
    return max(
        _volume_shelf_once(df, current_price, short),
        _volume_shelf_once(df, current_price, lookback),
    )


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


def macd_compression_score(
    macd: float,
    macd_signal: float,
    atr: float,
    macd_hist: Optional[float] = None,
) -> int:
    """MACD squeeze state (0-15). Highest when histogram is tight and MACD > 0.

    Name kept so existing unit tests can monkeypatch this symbol.
    """
    if atr <= 0:
        return 0
    hist = float(macd - macd_signal) if macd_hist is None else float(macd_hist)
    spread = abs(hist) / atr
    if spread <= 0.05:
        base = 15
    elif spread <= 0.10:
        base = 11
    elif spread <= 0.18:
        base = 7
    elif spread <= 0.30:
        base = 4
    else:
        return 0
    if macd <= 0:
        base = max(0, base - 5)
    return base


def _coil_points_for_width(width_atr: float) -> int:
    """Map a single range/ATR reading onto the 20-point coil ladder."""
    if width_atr <= COIL_FULL_ATR:
        return 20
    if width_atr <= COIL_PARTIAL_ATR:
        # 20 at 1.5 ATR → 10 at 2.2 ATR
        t = (width_atr - COIL_FULL_ATR) / (COIL_PARTIAL_ATR - COIL_FULL_ATR)
        return max(1, int(round(20 - 10 * t)))
    # High-beta residual: a pause after a vertical leg can still be a coil.
    if width_atr <= 4.0:
        t = (width_atr - COIL_PARTIAL_ATR) / (4.0 - COIL_PARTIAL_ATR)
        return max(1, int(round(10 - 6 * t)))
    return 0


def coil_width_score(df: pd.DataFrame, atr: float, coil_bars: int = None) -> int:
    """Tight N-bar range vs ATR (0-20). Coiled energy before expansion.

    Full score when range / ATR ≤ 1.5. Partial credit through 2.2, with a
    small residual band to 4.0 ATR for high-beta pauses. The signal bar is
    excluded and the tightest of several short windows is kept so a breakout
    day does not inflate the coil. Quiet volume (RVOL < 1.0) is valid
    compression and is not used here as a drop.
    """
    coil_bars = COIL_BARS if coil_bars is None else coil_bars
    if atr <= 0 or df.empty:
        return 0
    if coil_bars >= 20:
        windows = (3, 5, 8, 13)
    else:
        windows = tuple(sorted({3, 5, coil_bars}))
    best = 0
    for n in windows:
        if len(df) < n:
            continue
        if len(df) >= n + 1:
            window = df.iloc[-(n + 1):-1]
        else:
            window = df.iloc[-n:]
        rng = float(window["High"].max() - window["Low"].min())
        best = max(best, _coil_points_for_width(rng / atr))
        if best == 20:
            return 20
    return best


def _ema50_extension_penalty(pct_ema50: float, rs_rel: Optional[float]) -> int:
    """Soft haircut for EMA50 extension. Never a binary reject."""
    if pct_ema50 <= EMA50_SOFT_EXT:
        return 0
    leader = rs_rel is not None and rs_rel > RS_LEADER_EXT
    if pct_ema50 <= EMA50_MOMENTUM_EXT:
        t = (pct_ema50 - EMA50_SOFT_EXT) / (EMA50_MOMENTUM_EXT - EMA50_SOFT_EXT)
        max_pen = 5 if leader else 8
        return int(round(max_pen * t))
    return 7 if leader else 10


def structure_score(df: pd.DataFrame, rs_rel: Optional[float] = None) -> int:
    """MA alignment and EMA50-extension proximity for a leader coil (0-15).

    Requires EMA20 > EMA50. Overextension is a scaled deduction — RS leaders
    (RS 63d > 0.10) may sit up to 50% above EMA50 without a zero or gate fail.
    """
    if len(df) < 2:
        return 0
    latest = df.iloc[-1]
    close = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema10 = latest.get("EMA10")
    ema50 = latest.get("EMA50")
    sma50 = latest.get("SMA50")

    ma50 = None
    if ema50 is not None and pd.notna(ema50):
        ma50 = float(ema50)
    elif sma50 is not None and pd.notna(sma50):
        ma50 = float(sma50)

    # Enforce EMA20 > EMA50 with 2% slack so a flat coil under the 50 still
    # scores (PLTR-style pre-gap bases). SMA50 is only a fallback.
    if ma50 is None or ema20 < ma50 * 0.98:
        return 0

    score = 0
    if ema10 is not None and pd.notna(ema10) and float(ema10) >= ema20 * 0.98:
        score += 5
    score += 5  # EMA20 > EMA50
    if close >= ema20 * 0.98:
        score += 5

    pct_ema50 = (close - ma50) / ma50 if ma50 else 0.0
    score -= _ema50_extension_penalty(pct_ema50, rs_rel)
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
    """Room to run (0-5). Open sky near 52-week / ATH scores a full 5.

    If Close ≥ 95% of the lookback high or the series ATH, Fib extension
    distance is ignored and the full space weight is awarded.
    """
    if df.empty:
        return 0.0
    window = df.iloc[-lookback:] if len(df) >= lookback else df
    if window.empty:
        return 0.0
    high_52 = float(window["High"].max())
    ath = float(df["High"].max())
    # Open sky: within 5% of the 52-week or all-time high — ignore Fib distance.
    if high_52 > 0 and price >= OPEN_SKY_PCT * high_52:
        return 5.0
    if ath > 0 and price >= OPEN_SKY_PCT * ath:
        return 5.0
    # Local open sky: within 5% of the recent range high (63-bar shelf window).
    local_n = RS_LOOKBACK if RS_LOOKBACK else 63
    local = df.iloc[-local_n:] if len(df) >= local_n else df
    local_high = float(local["High"].max()) if not local.empty else 0.0
    if local_high > 0 and price >= OPEN_SKY_PCT * local_high:
        return 5.0
    if atr <= 0:
        return 0.0
    # Room to the significant high, not the nearest one-bar wick.
    if high_52 <= price or local_high <= price:
        return 5.0
    room_atr = (min(high_52, local_high) - price) / atr
    if room_atr >= 3.0:
        return 5.0
    if room_atr >= 2.0:
        return 3.0
    if room_atr >= 1.0:
        return 1.0
    return 0.0


def rs_score(
    stock_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    as_of=None,
) -> tuple[int, Optional[float]]:
    """Relative strength vs QQQ (0-15). 63-day (daily) / 13-bar (weekly) RS.

    Full 15 points when 63d relative return ≥ +15%, even if QQQ is choppy
    and the ratio fails its moving-average test.
    """
    if benchmark_df is None:
        return 0, None
    ok, rel = relative_strength(
        stock_df,
        benchmark_df,
        as_of=as_of,
        lookback=RS_LOOKBACK,
        ratio_ma_bars=RS_RATIO_MA,
    )
    if rel is not None and rel >= RS_FULL_SCORE:
        return 15, rel
    if ok and rel is not None and rel > RS_LEADER_EXT:
        return 15, rel
    if ok:
        return 12, rel
    if rel is not None and rel > 0:
        return 5, rel
    if rel is not None and rel > -0.15:
        return 5, rel  # mild lag / noise — not a BA/DG washout
    return 0, rel


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
    """100-point coil scorecard: catch compressed leaders before they expand.

    Pillars (v3.1):
      Volume shelf 20 · Coil width 20 · MACD squeeze 15 · RS 15 ·
      MA alignment 15 · RVOL trigger 10 · Overhead clearance 5
    Market Gate filters only total trend failure (Close < EMA50 or RS 63d < 0).
    Extension, Fib distance, low RVOL, and SPY/QQQ chop are scorecard items.
    """
    if len(df) < max(COIL_BARS + 2, 25):
        return None

    latest = df.iloc[-1]
    if any(pd.isna(latest.get(col)) for col in ("Close", "EMA20", "EMA50", "MACD", "ATR")):
        return None

    current_price = float(latest["Close"])
    atr = float(latest["ATR"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_Signal"])
    macd_hist = latest.get("MACD_Hist")
    hist_v = float(macd_hist) if macd_hist is not None and pd.notna(macd_hist) else None
    as_of = latest["Date"] if "Date" in df.columns else None
    ema50_f = float(latest["EMA50"])

    parts: dict[str, float] = {}
    checks_passed = 0

    # 1. Volume shelf (0-20) — proximity to POC / HVN
    vp = evaluate_volume_profile_shelf(df, current_price)
    parts["volume_shelf"] = vp
    if vp >= 10:
        checks_passed += 1

    # 2. Volatility coil (0-20) — low RVOL inside the coil is valid compression
    coil = coil_width_score(df, atr)
    parts["coil_width"] = coil
    if coil >= 10:
        checks_passed += 1

    # 3. MACD squeeze state (0-15) — tightest hist in the last 8 bars so a
    # breakout bar that just released the squeeze still gets credit.
    comp = macd_compression_score(macd, macd_signal, atr, macd_hist=hist_v)
    look = df.iloc[-8:] if len(df) >= 8 else df
    for _, row in look.iterrows():
        row_atr = row.get("ATR")
        row_macd = row.get("MACD")
        row_sig = row.get("MACD_Signal")
        row_hist = row.get("MACD_Hist")
        if pd.isna(row_atr) or pd.isna(row_macd) or pd.isna(row_sig) or float(row_atr) <= 0:
            continue
        h = None if pd.isna(row_hist) else float(row_hist)
        comp = max(comp, macd_compression_score(float(row_macd), float(row_sig), float(row_atr), macd_hist=h))
    if comp == 0 and macd > 0:
        comp = 7  # momentum-leader floor; wide hist is not a drop
    elif comp == 0:
        comp = 4  # washed-out squeeze still has a histogram
    parts["macd_compression"] = comp
    if comp >= 7:
        checks_passed += 1

    # 4. Relative strength vs QQQ (0-15) — full score at RS 63d ≥ 0.15
    rs_pts, rs_rel = rs_score(df, benchmark_df, as_of=as_of)
    parts["relative_strength"] = rs_pts
    if rs_pts >= 12:
        checks_passed += 1

    # 5. MA alignment & EMA50 proximity (0-15) — soft extension haircut
    struct = structure_score(df, rs_rel=rs_rel)
    parts["structure"] = struct
    if struct >= 10:
        checks_passed += 1

    # 6. Breakout RVOL bonus (0-10) — never zeroed by the gate.
    # Quiet volume on a tight coil is valid compression, not a zero.
    rvol_pts, rvol = rvol_trigger_score(df)
    if rvol_pts == 0 and coil >= 15 and rvol is not None and rvol < RVOL_BONUS:
        rvol_pts = 4
    parts["rvol_trigger"] = rvol_pts
    if rvol_pts >= 6:
        checks_passed += 1

    # 7. Overhead clearance (0-5). Open sky awards the full 5.
    overhead = overhead_clearance_score(df, current_price, atr, lookback=LOOKBACK)
    parts["overhead_clearance"] = overhead
    if overhead >= 3:
        checks_passed += 1

    score = sum(parts.values())

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

    if score >= GRADE_A_SCORE and gate_ok:
        grade = "A - Coil Ready"
    elif score >= MIN_PASS_SCORE and gate_ok:
        grade = "B - Valid Coil"
    elif not gate_ok:
        grade = "Rejected - Trend Fail"
    else:
        grade = "Rejected - Below Threshold"

    result = {
        "Score": round(score, 2),
        "Grade": grade,
        "Checks Met": f"{checks_passed}/7",
        "Fib Score": 0.0,
        "Parts": parts,
        "RS 63d": rs_rel,
        "RVOL": None if rvol is None else round(rvol, 4),
        "Market Gate": gate_ok,
    }

    if include_rejects:
        return result
    if not gate_ok or score < MIN_PASS_SCORE:
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
    if len(work) < max(COIL_BARS + 2, 25):
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


def run_scanner():
    logger.info(f"--- STEP 5: Scanning Coil Setups [{mode.upper()} MODE] ---")

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

    min_required_history = max(LOOKBACK // 2, COIL_BARS + 40)

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

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(LOG_DIR, f"coiled_cobra_setups_{today}.csv")
    df_out = pd.DataFrame(results).reindex(columns=config.SETUP_ROW_COLUMNS)

    if results:
        # Attach offline-model ranks (soft signal). Falls back to Score sort
        # when no model artifact is available or features are unusable.
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
    run_scanner()
