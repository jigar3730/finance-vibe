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
if len(sys.argv) > 1 and sys.argv[1].lower() in ["weekly", "daily"]:
    mode = sys.argv[1].lower()
else:
    print("⚠️ Unknown mode parsed to scanner. Defaulting to 'weekly'.")
    mode = "weekly"

# Timeframe-specific technical calibration
LOOKBACK = 252 if mode == "daily" else 52
# Coil window: how many bars define "the base" (weekly ≈ 2 months, daily ≈ 6 weeks)
COIL_BARS = 30 if mode == "daily" else 8
# Local structural floor for dual-constraint stops (not the macro Fib lookback)
STRUCTURE_STOP_BARS = 10
# RS lookback in bars (weekly ≈ 1 quarter, daily ≈ 63 sessions)
RS_LOOKBACK = 63 if mode == "daily" else 13
RS_RATIO_MA = 20 if mode == "daily" else 5
BENCHMARK = "QQQ"
SPY_BENCHMARK = "SPY"


def local_swing_low(df: pd.DataFrame, bars: int = STRUCTURE_STOP_BARS) -> float:
    """Minimum Low over the last ``bars`` sessions (local consolidation floor)."""
    window = df.iloc[-bars:] if len(df) >= bars else df
    return float(window["Low"].min())

# Soft pass floors (scorecard still sums to 100)
MIN_PASS_SCORE = 70
GRADE_A_SCORE = 85
# Hard gates (ready-to-run leaders, not lagging coils)
MIN_COMPRESSION = 5
MIN_STRUCTURE = 8
MIN_RS_POINTS = 12  # requires full RS pass (ratio > MA and positive rel-return)
MACRO_PENALTY = 15  # subtracted when SPY/QQQ fail the 21-EMA / 50-SMA gate

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


def add_macro_indicators(df: pd.DataFrame, lookback=LOOKBACK) -> pd.DataFrame:
    """EMA stack, MACD, RSI, ATR, RVOL, and rolling Fib levels for coil scoring."""
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

    # Coil scoring needs EMA50; Fib may be NaN early but is optional now.
    out.dropna(subset=["EMA20", "EMA50", "MACD", "ATR"], inplace=True)
    return out


# =========================
# STRUCTURAL LAYER FILTER (AMT)
# =========================


def evaluate_volume_profile_shelf(
    df: pd.DataFrame, current_price: float, lookback=LOOKBACK
) -> int:
    """
    Auction-market volume shelf score (0-20).

    Rewards price sitting in a high-volume node with supportive topology —
    the accumulation zone before a coil expands.
    """
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

    # Topology / liquidity gradient (max 8)
    left_idx = max(0, price_bin - 1)
    right_idx = min(len(binned_volume) - 1, price_bin + 1)
    avg_neighbor_vol = (binned_volume[left_idx] + binned_volume[right_idx]) / 2
    current_vol = binned_volume[price_bin]
    if avg_neighbor_vol > 0:
        topology_score = min(8, int((current_vol / avg_neighbor_vol) * 2.5))
    else:
        topology_score = 0

    # Auction value vs POC (max 8) — prefer near-value, not exactly at POC only
    poc_bin = int(np.argmax(binned_volume))
    distance_from_poc = abs(price_bin - poc_bin)
    if price_bin == poc_bin:
        value_score = 4
    elif distance_from_poc <= 3:
        value_score = 8
    elif distance_from_poc <= 6:
        value_score = 4
    else:
        value_score = 0

    # Close holding above bin center (max 4)
    bin_center = (bin_edges[price_bin] + bin_edges[min(price_bin + 1, len(bin_edges) - 1)]) / 2
    behavior_score = 4 if float(recent_data["Close"].iloc[-1]) > bin_center else 1

    return int(topology_score + value_score + behavior_score)


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


def coil_width_score(df: pd.DataFrame, atr: float, coil_bars: int = COIL_BARS) -> int:
    """Tight N-bar range vs ATR (0-20). Coiled energy before expansion."""
    if atr <= 0 or len(df) < coil_bars:
        return 0
    window = df.iloc[-coil_bars:]
    rng = float(window["High"].max() - window["Low"].min())
    width_atr = rng / atr
    if width_atr <= 1.5:
        return 20
    if width_atr <= 2.5:
        return 15
    if width_atr <= 4.0:
        return 10
    if width_atr <= 6.0:
        return 5
    return 0


def structure_score(df: pd.DataFrame) -> int:
    """MA alignment and ATR proximity for a leader coil (0-15).

    Full stack is 10 EMA > 20 EMA > 50 SMA with close holding EMA20.
    Overextension (|Close − EMA20| > 1.5 × ATR14) caps the pillar at 8.
    """
    if len(df) < 2:
        return 0
    latest = df.iloc[-1]
    close = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    atr = float(latest["ATR"]) if pd.notna(latest.get("ATR")) else 0.0
    ema10 = latest.get("EMA10")
    sma50 = latest.get("SMA50")
    score = 0

    if ema10 is not None and pd.notna(ema10) and float(ema10) > ema20:
        score += 5
    if sma50 is not None and pd.notna(sma50) and ema20 > float(sma50):
        score += 5
    if close > ema20:
        score += 5
    if atr > 0 and abs(close - ema20) > 1.5 * atr:
        score = min(score, 8)
    return score


def rvol_trigger_score(df: pd.DataFrame) -> tuple[int, Optional[float]]:
    """Breakout relative-volume trigger (0-10). Full points at RVOL ≥ 2.0×."""
    if df.empty or "RVOL" not in df.columns:
        return 0, None
    rvol = df.iloc[-1].get("RVOL")
    if rvol is None or pd.isna(rvol):
        return 0, None
    rvol_f = float(rvol)
    if rvol_f >= 2.0:
        pts = 10
    elif rvol_f >= 1.5:
        pts = 6
    elif rvol_f >= 1.2:
        pts = 3
    else:
        pts = 0
    return pts, rvol_f


def overhead_clearance_score(
    df: pd.DataFrame,
    price: float,
    atr: float,
    lookback: int = 50,
) -> float:
    """Distance to nearest supply high in ATR units (0-5). Clear air scores 5."""
    if atr <= 0 or df.empty:
        return 0.0
    window = df.iloc[-lookback:] if len(df) >= lookback else df
    above = window.loc[window["High"] > price, "High"]
    if above.empty:
        return 5.0
    dist_atr = (float(above.min()) - price) / atr
    if dist_atr >= 3.0:
        return 5.0
    if dist_atr >= 2.0:
        return 3.0
    if dist_atr >= 1.0:
        return 1.0
    return 0.0


def rs_score(
    stock_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    as_of=None,
) -> tuple[int, Optional[float]]:
    """Relative strength vs QQQ (0-15). Leaders before monster runs usually lead."""
    if benchmark_df is None:
        return 0, None
    ok, rel = relative_strength(
        stock_df,
        benchmark_df,
        as_of=as_of,
        lookback=RS_LOOKBACK,
        ratio_ma_bars=RS_RATIO_MA,
    )
    if ok and rel is not None and rel > 0.10:
        return 15, rel
    if ok:
        return 12, rel
    if rel is not None and rel > 0:
        return 5, rel
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
) -> Optional[dict]:
    """100-point coil scorecard: catch compressed leaders before they expand.

    Pillars (v3):
      Volume shelf 20 · Coil width 20 · MACD squeeze 15 · RS 15 ·
      MA alignment 15 · RVOL trigger 10 · Overhead clearance 5
    Hard gates: squeeze, structure, and full RS vs QQQ (no lagging coils).
    Failed market gate zeros RVOL, subtracts MACRO_PENALTY, and caps Grade A.
    """
    if len(df) < max(COIL_BARS + 2, 25):
        return None

    latest = df.iloc[-1]

    current_price = float(latest["Close"])
    atr = float(latest["ATR"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_Signal"])
    macd_hist = latest.get("MACD_Hist")
    hist_v = float(macd_hist) if macd_hist is not None and pd.notna(macd_hist) else None
    as_of = latest["Date"] if "Date" in df.columns else None

    gate_ok = True
    if apply_market_gate:
        gate_ok = check_coiled_cobra_market_gate(
            spy_df=spy_df,
            qqq_df=qqq_df if qqq_df is not None else benchmark_df,
            as_of=as_of,
        )

    parts: dict[str, float] = {}
    checks_passed = 0

    # 1. Volume shelf (0-20)
    vp = evaluate_volume_profile_shelf(df, current_price)
    parts["volume_shelf"] = vp
    if vp >= 10:
        checks_passed += 1

    # 2. Volatility coil (0-20)
    coil = coil_width_score(df, atr)
    parts["coil_width"] = coil
    if coil >= 10:
        checks_passed += 1

    # 3. MACD squeeze state (0-15)
    comp = macd_compression_score(macd, macd_signal, atr, macd_hist=hist_v)
    parts["macd_compression"] = comp
    if comp >= 7:
        checks_passed += 1

    # 4. Relative strength vs QQQ (0-15)
    rs_pts, rs_rel = rs_score(df, benchmark_df, as_of=as_of)
    parts["relative_strength"] = rs_pts
    if rs_pts >= 12:
        checks_passed += 1

    # 5. MA alignment & ATR proximity (0-15)
    struct = structure_score(df)
    parts["structure"] = struct
    if struct >= 10:
        checks_passed += 1

    # 6. Breakout RVOL trigger (0-10) — zeroed when the macro gate fails
    rvol_pts, rvol = rvol_trigger_score(df)
    if not gate_ok:
        rvol_pts = 0
    parts["rvol_trigger"] = rvol_pts
    if rvol_pts >= 6:
        checks_passed += 1

    # 7. Overhead clearance (0-5). Fib is kept off the sum (CSV still emits 0.0).
    overhead = overhead_clearance_score(df, current_price, atr, lookback=LOOKBACK)
    parts["overhead_clearance"] = overhead
    if overhead >= 3:
        checks_passed += 1

    score = sum(parts.values())
    if not gate_ok:
        score -= MACRO_PENALTY

    # Hard gates for "ready to run":
    #   compression — coiled energy
    #   structure   — healthy / aligned stack (not a broken decline)
    #   RS          — leading QQQ (kills BA/DG-style negative-RS coils)
    if (
        comp < MIN_COMPRESSION
        or struct < MIN_STRUCTURE
        or rs_pts < MIN_RS_POINTS
    ):
        return None

    if score >= GRADE_A_SCORE and gate_ok:
        grade = "A - Coil Ready"
    elif score >= MIN_PASS_SCORE:
        grade = "B - Valid Coil"
    else:
        return None

    return {
        "Score": round(score, 2),
        "Grade": grade,
        "Checks Met": f"{checks_passed}/7",
        "Fib Score": 0.0,
        "Parts": parts,
        "RS 63d": rs_rel,
        "RVOL": None if rvol is None else round(rvol, 4),
        "Market Gate": gate_ok,
    }


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

    qqq_df = load_benchmark_frame(BENCHMARK, mode)
    spy_df = load_benchmark_frame(SPY_BENCHMARK, mode)
    if qqq_df is None:
        logger.warning(
            f"Benchmark {BENCHMARK} unavailable in {mode} raw data — RS pillar will score 0."
        )
    if spy_df is None:
        logger.warning(
            f"Benchmark {SPY_BENCHMARK} unavailable in {mode} raw data — market gate uses QQQ only."
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

    if results:
        df_out = pd.DataFrame(results).reindex(columns=config.SETUP_ROW_COLUMNS)

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


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run_scanner()
