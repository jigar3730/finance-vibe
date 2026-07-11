import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta

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
MARKDOWN_THRESHOLD = 0.93 if mode == "daily" else 0.85

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
    """Calculates systematic CMT indicators for macro reversal zones using pandas_ta."""
    # 1. Trend Tracking
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["EMA50"] = ta.ema(df["Close"], length=50)

    # 2. Momentum Compression (Standard 12, 26, 9 MACD)
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]

    # 3. Macro Fibonacci Cycle Tracking
    rolling_max = df["High"].rolling(window=lookback, min_periods=lookback).max()
    rolling_min = df["Low"].rolling(window=lookback, min_periods=lookback).min()

    df["Fib_786"] = rolling_max - ((rolling_max - rolling_min) * 0.786)
    df["Fib_618"] = rolling_max - ((rolling_max - rolling_min) * 0.618)

    # 4. Volatility (ATR 14)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Clean missing structural rows up to our lookback window
    df.dropna(subset=["EMA20", "Fib_786"], inplace=True)
    return df


# =========================
# STRUCTURAL LAYER FILTER (AMT UPGRADE)
# =========================


def evaluate_volume_profile_shelf(
    df: pd.DataFrame, current_price: float, lookback=LOOKBACK
) -> int:
    """
    Evaluates market structural positioning using Auction Market Theory abstracts.
    Replaces the binary check with a granular score from 0 to 25.
    """
    recent_data = df.iloc[-lookback:]

    v_min = float(recent_data["Low"].min())
    v_max = float(recent_data["High"].max())

    if v_min == v_max:
        return 0

    # Expanded to 30 bins to capture true structural granularity/cliffs
    bins = np.linspace(v_min, v_max, 31)
    close_array = recent_data["Close"].to_numpy().flatten()
    volume_array = recent_data["Volume"].to_numpy().flatten()

    binned_volume, bin_edges = np.histogram(
        close_array, bins=bins, weights=volume_array
    )
    
    # Locate where the current price sits in the market profile matrix
    price_bin = np.digitize([current_price], bin_edges)[0] - 1
    price_bin = max(0, min(price_bin, len(binned_volume) - 1))

    # --- 1. TOPOLOGY / LIQUIDITY GRADIENT (Max 10 Points) ---
    left_idx = max(0, price_bin - 1)
    right_idx = min(len(binned_volume) - 1, price_bin + 1)
    
    avg_neighbor_vol = (binned_volume[left_idx] + binned_volume[right_idx]) / 2
    current_vol = binned_volume[price_bin]
    
    if avg_neighbor_vol > 0:
        gradient_ratio = current_vol / avg_neighbor_vol
        topology_score = min(10, int(gradient_ratio * 3))
    else:
        topology_score = 0

    # --- 2. AUCTION VALUE DYNAMICS (Max 10 Points) ---
    poc_bin = np.argmax(binned_volume)
    
    if price_bin == poc_bin:
        value_score = 2 
    else:
        distance_from_poc = abs(price_bin - poc_bin)
        if 1 <= distance_from_poc <= 3:
            value_score = 10
        elif distance_from_poc <= 6:
            value_score = 6
        else:
            value_score = 0

    # --- 3. VOLUME-TIME DIVERGENCE / BEHAVIOR (Max 5 Points) ---
    last_close = float(recent_data["Close"].iloc[-1])
    bin_center_price = (bin_edges[price_bin] + bin_edges[price_bin + 1]) / 2
    
    if last_close > bin_center_price:
        behavior_score = 5
    else:
        behavior_score = 1

    return topology_score + value_score + behavior_score

# =========================
# FIB SCORE - VOLATILITY REGULATED QUAD DECAY 
# =========================
def fibonacci_score(
    current_price: float,
    fib_levels: dict,
    atr: float,
    max_atr_distance: float = 0.5,
) -> float:
    """
    Fibonacci Confluence Score (0-30 max) based on CMT feedback.

    - Normalizes distance using volatility (ATR) instead of raw percentages.
    - Evaluates proximity to the single closest valid Fibonacci level.
    - Applies a confidence multiplier based on the specific Fibonacci level.
    """
    if not isinstance(fib_levels, dict) or atr <= 0:
        return 0.0

    best_score = 0.0

    for level_price, max_possible_score in fib_levels.items():
        dollar_distance = abs(current_price - level_price)
        atr_distance = dollar_distance / atr

        if atr_distance >= max_atr_distance:
            continue

        level_score = max_possible_score * (1 - (atr_distance / max_atr_distance) ** 2)

        if level_score > best_score:
            best_score = level_score

    return round(best_score, 2)

# =========================
# SYSTEMATIC GRADING MATRIX
# =========================


def evaluate_coiled_cobra(df: pd.DataFrame) -> dict:
    """Applies a 100-point CMT scoring scorecard to identify high-probability macro reversals."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    current_price = float(latest["Close"])
    fib_786 = float(latest["Fib_786"])
    fib_618 = float(latest["Fib_618"])
    atr = float(latest["ATR"])  
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_Signal"])
    prev_macd = float(prev["MACD"])
    prev_macd_signal = float(prev["MACD_Signal"])
    ema20 = float(latest["EMA20"])

    score = 0
    checks_passed = 0

    # 1. Fibonacci Confluence (30 Pts Max)
    current_ticker_fibs = {
        fib_618: 28.5,  
        fib_786: 30.0   
    }

    fib_score = fibonacci_score(
        current_price=current_price,
        fib_levels=current_ticker_fibs,
        atr=atr,
        max_atr_distance=0.5  
    )

    score += fib_score
    if fib_score > 0:
        checks_passed += 1
        
    # 2. Volume Profile Shelf Presence via AMT (25 Pts Max)
    vp_score = evaluate_volume_profile_shelf(df, current_price)
    score += vp_score
    if vp_score >= 12:  # Quantitative minimum threshold to qualify as a passed check
        checks_passed += 1

    # 3. Deep Markdown Extension (15 Pts)
    if current_price < ema20 * MARKDOWN_THRESHOLD:
        score += 15
        checks_passed += 1

    # 4. Oscillator Compression / Deep Oversold (20 Pts)
    if macd < 0 and abs(macd - macd_signal) < (current_price * 0.02):
        score += 20
        checks_passed += 1

    # 5. Bullish Momentum Crossover Trigger (10 Pts)
    if prev_macd <= prev_macd_signal and macd > macd_signal:
        score += 10
        checks_passed += 1

    # Final Classification Breakdown
    if score >= 85:
        grade = "A - Institutional Setup"
    elif score >= 70:
        grade = "B - Valid Reversal"
    else:
        return None  

    return {"Score": round(score, 2), "Grade": grade, "Checks Met": f"{checks_passed}/5", "Fib Score": fib_score}

# =========================
# SCANNER CORE
# =========================


def run_scanner():
    logger.info(f"--- STEP 5: Scanning Macro Reversals [{mode.upper()} MODE] ---")

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

    results = []
    rejection_counts = {}

    min_required_history = LOOKBACK + 15

    for file in raw_files:
        symbol = file.split(".")[0].split("_")[0].upper()

        if symbol not in active_tickers:
            rejection_counts["inactive_ticker"] = (
                rejection_counts.get("inactive_ticker", 0) + 1
            )
            continue

        path = os.path.join(RAW_DATA_DIR, file)
        df = pd.read_csv(path)

        if len(df) < min_required_history:
            rejection_counts["insufficient_history"] = (
                rejection_counts.get("insufficient_history", 0) + 1
            )
            continue

        try:
            df.columns = [c.capitalize() for c in df.columns]

            df = add_macro_indicators(df)
            setup = evaluate_coiled_cobra(df)

            if not setup:
                rejection_counts["IGNORE"] = rejection_counts.get("IGNORE", 0) + 1
                continue

            latest = df.iloc[-1]

            results.append(
                {
                    "Symbol": symbol,
                    "Close": round(latest["Close"], 2),
                    "EMA20": round(latest["EMA20"], 2),
                    "EMA50": round(latest["EMA50"], 2),
                    "Score": setup["Score"],
                    "Grade": setup["Grade"],
                    "Checks Met": setup["Checks Met"],
                    "Fib 61.8%": round(latest["Fib_618"], 2),
                    "Fib 78.6%": round(latest["Fib_786"], 2),
                    "Fib Score": setup["Fib Score"],
                    "ATR": round(latest["ATR"], 2),
                    "Setup Type": "SETUP_LONG",
                }
            )

        except Exception as e:
            logger.error(f"Error scoring {symbol}: {str(e)}")
            rejection_counts["execution_error"] = (
                rejection_counts.get("execution_error", 0) + 1
            )

    if results:
        df_out = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        print("\n" + df_out.to_markdown(index=False) + "\n")

        today = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(LOG_DIR, f"coiled_cobra_setups_{today}.csv")
        df_out.to_csv(out_path, index=False)
        logger.info(f"Archive logged successfully to: {out_path}")
    else:
        logger.warning(
            "No high-confluence Coiled Cobra setups detected across watchlists."
        )

    logger.info("Macro Scanner execution complete. Rejection Summary:")
    for k, v in rejection_counts.items():
        logger.info(f"  {k}: {v}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run_scanner()