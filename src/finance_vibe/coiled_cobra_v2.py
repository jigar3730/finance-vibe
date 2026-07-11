import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta

# =====================================================================
# CONFIGURATION & SCHEMAS (Orthogonal Factor Weights)
# =====================================================================

@dataclass(frozen=True)
class ScannerConfig:
    lookback: int
    decay_sensitivity: float  # Controls Gaussian decay tightness around targets

CONFIG_MODES = {
    "daily": ScannerConfig(lookback=252, decay_sensitivity=1.2),
    "weekly": ScannerConfig(lookback=52, decay_sensitivity=1.0),
}

@dataclass(frozen=True)
class FactorWeights:
    structure: float = 0.40   # Volatility-adjusted distance to Fib levels
    momentum: float = 0.40    # Z-scored MACD Histogram exhaustion
    volatility: float = 0.20  # Bollinger Band width regime compression

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}

# =====================================================================
# LOGGING SETUP
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================================
# DATA VALIDATION & PIPELINE PREPARATION
# =====================================================================

def normalize_and_validate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names, enforces chronological sorting, and validates data."""
    df = df.copy()
    
    # Standardize column casing dynamically without being destructive
    rename_map = {c: c.strip().capitalize() for c in df.columns}
    if "Adj close" in rename_map.values() or "Adj Close" in df.columns:
        df.rename(columns={c: "Adj Close" for c in df.columns if c.strip().lower() == "adj close"}, inplace=True)
    
    df.rename(columns={k: v for k, v in rename_map.items() if v in REQUIRED_COLUMNS}, inplace=True)
    
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required structural columns: {missing_cols}")
        
    # Enforce chronological ordering for valid time-series calculations
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    return df

# =====================================================================
# TECHNICAL FACTOR INGESTION
# =====================================================================

def add_macro_indicators(df: pd.DataFrame, config: ScannerConfig) -> pd.DataFrame:
    """Calculates systematic indicators for macro reversal setups cleanly."""
    df = df.copy()
    
    # 1. Volatility Baseline (The ruler we use to measure all distances)
    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    
    # 2. Trend & Variance Baselines
    df["EMA20"] = ta.ema(df["Close"], length=20)
    
    # Bollinger Band Width for Volatility Regime Mapping
    bb = ta.bbands(df["Close"], length=50, std=2.0)
    if bb is not None and not bb.empty:
        try:
            # Dynamically look up columns by prefix to prevent string-formatting KeyErrors
            bbl_col = [c for c in bb.columns if c.startswith("BBL")][0]
            bbm_col = [c for c in bb.columns if c.startswith("BBM")][0]
            bbu_col = [c for c in bb.columns if c.startswith("BBU")][0]
            
            # Standardized Bandwidth calculation: (Upper - Lower) / Middle
            df["BB_Width"] = (bb[bbu_col] - bb[bbl_col]) / bb[bbm_col]
        except IndexError:
            df["BB_Width"] = np.nan
    else:
        df["BB_Width"] = np.nan
        
    # 3. Momentum (Unpacking MACD safely)
    macd_df = ta.macd(df["Close"])
    if macd_df is None or macd_df.empty or "MACDh_12_26_9" not in macd_df:
        raise ValueError("MACD calculation failed or returned malformed structure.")
        
    df["MACD_Hist"] = macd_df["MACDh_12_26_9"]

    # 4. Macro Fibonacci Cycle Tracking
    rolling_max = df["High"].rolling(window=config.lookback, min_periods=config.lookback).max()
    rolling_min = df["Low"].rolling(window=config.lookback, min_periods=config.lookback).min()

    df["Fib_786"] = rolling_max - ((rolling_max - rolling_min) * 0.786)
    df["Fib_618"] = rolling_max - ((rolling_max - rolling_min) * 0.618)

    return df.dropna(subset=["EMA20", "ATR_14", "Fib_786", "MACD_Hist", "BB_Width"])

# =====================================================================
# STRUCTURAL LAYER FILTER (Volume Profile)
# =====================================================================

def evaluate_volume_profile_shelf(df: pd.DataFrame, current_price: float, config: ScannerConfig) -> bool:
    """Aggregates horizontal volume structures into 10 pricing segments with edge-case clamping."""
    recent_data = df.iloc[-config.lookback:]
    v_min = float(recent_data["Low"].min())
    v_max = float(recent_data["High"].max())

    if v_min == v_max:
        return False

    bins = np.linspace(v_min, v_max, 10)
    close_array = recent_data["Close"].to_numpy().flatten()
    volume_array = recent_data["Volume"].to_numpy().flatten()

    binned_volume, bin_edges = np.histogram(close_array, bins=bins, weights=volume_array)
    
    # Digitization clamping to handle upper-boundary floating-point edge cases safely
    price_bin = np.digitize([current_price], bin_edges)[0] - 1
    price_bin = int(np.clip(price_bin, 0, len(binned_volume) - 1))

    # Identify if price sits inside Top 3 High-Volume Nodes (HVN)
    highest_vol_bins = np.argsort(binned_volume)[-3:]
    return price_bin in highest_vol_bins

# =====================================================================
# QUANTITATIVE GRADING MATRIX
# =====================================================================

def evaluate_coiled_cobra(df: pd.DataFrame, config: ScannerConfig, weights=FactorWeights()) -> Optional[Dict[str, Any]]:
    """
    An institutional-grade factor scoring matrix.
    Replaces arbitrary binary addition with continuous, volatility-normalized decay models.
    """
    if len(df) < 2:
        return None

    latest = df.iloc[-1]
    current_price = float(latest["Close"])
    atr = float(latest["ATR_14"])
    ema20 = float(latest["EMA20"])
    
    # Critical zero/nan defense to prevent runtime mathematical exceptions
    if pd.isna(atr) or atr <= 0 or pd.isna(ema20) or ema20 <= 0 or pd.isna(current_price):
        return None

    # 1. Continuous Structural Factor (Distance measured in ATR units to avoid asset-class bias)
    fib_786 = float(latest["Fib_786"])
    fib_618 = float(latest["Fib_618"])
    
    dist_786 = abs(current_price - fib_786) / atr
    dist_618 = abs(current_price - fib_618) / atr
    min_fib_distance = min(dist_786, dist_618)
    
    # Gaussian decay: Perfect hit = 1.0, decays smoothly toward 0 as distance grows
    structure_factor = np.exp(-(min_fib_distance / config.decay_sensitivity) ** 2)
    
    # Boost structure points if backed by an institutional Volume Profile shelf
    if evaluate_volume_profile_shelf(df, current_price, config):
        structure_factor = min(1.0, structure_factor * 1.25)

    # 2. Continuous Momentum Factor (Z-Score tracking of MACD Histogram Exhaustion)
    macd_hist_series = df["MACD_Hist"].to_numpy()
    if len(macd_hist_series) >= 30:
        recent_hist = macd_hist_series[-60:]
        hist_mean = np.mean(recent_hist)
        hist_std = np.std(recent_hist)
        
        # Calculate standard deviations away from historical variance mean
        momentum_z = (latest["MACD_Hist"] - hist_mean) / (hist_std if hist_std > 0 else 1)
        # Pass through an offset sigmoid to isolate extreme negative extensions (sellers exhausted)
        momentum_factor = 1 / (1 + np.exp(momentum_z + 1.5))
    else:
        momentum_factor = 0.0

    # 3. Continuous Volatility Regime Factor (Is volatility compressing or expanding?)
    # Compares current width to its historical rolling median
    historical_median_width = df["BB_Width"].rolling(100).median().iloc[-1]
    current_width = float(latest["BB_Width"])
    
    if current_width > 0 and historical_median_width > 0:
        # Value > 1.0 implies compression/coiling relative to its historical normal
        volatility_factor = min(1.0, historical_median_width / current_width)
    else:
        volatility_factor = 0.5

    # 4. Multi-Factor Non-Linear Compositing
    composite_score = (
        (structure_factor * weights.structure) +
        (momentum_factor * weights.momentum) +
        (volatility_factor * weights.volatility)
    ) * 100.0  # Scale up to a traditional 100-point tracking base

    # Cutoff hurdles for screening classification
    if composite_score >= 75.0:
        grade = "A - Institutional Factor Core"
    elif composite_score >= 60.0:
        grade = "B - High-Confluence Setup"
    else:
        return None  # Drop asset out of the scan run entirely

    return {
        "Score": round(composite_score, 1),
        "Grade": grade,
        "Structure_Score": round(structure_factor * 100, 1),
        "Momentum_Score": round(momentum_factor * 100, 1),
        "Volatility_Score": round(volatility_factor * 100, 1)
    }

# =====================================================================
# EXECUTION ENGINE CORE
# =====================================================================

def load_active_tickers(path: str) -> Set[str]:
    """Loads and scrubs clean structural active watchlists."""
    if not os.path.exists(path):
        logger.error(f"Missing active tickers inventory file at {path}")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
        if "Ticker" not in df.columns:
            logger.error("Malformed ticker inventory file. Expected 'Ticker' column header.")
            sys.exit(1)
        return set(df["Ticker"].dropna().astype(str).str.strip().str.upper())
    except Exception as e:
        logger.error(f"Failed to parse active tickers file cleanly: {str(e)}")
        sys.exit(1)

def run_scanner(mode: str):
    logger.info(f"--- STEP 5: Running Quant Factor Scan [{mode.upper()} MODE] ---")
    
    config = CONFIG_MODES[mode]

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", mode)
    ACTIVE_TICKERS_PATH = os.path.join(BASE_DIR, "data", "active_tickers.csv")
    LOG_DIR = os.path.join(BASE_DIR, "data", "logs", mode)

    os.makedirs(LOG_DIR, exist_ok=True)
    active_tickers = load_active_tickers(ACTIVE_TICKERS_PATH)
    logger.info(f"Loaded {len(active_tickers)} active reference positions into Matrix Ecosystem.")

    if not os.path.exists(RAW_DATA_DIR):
        logger.warning(f"Target raw matrix data pathway empty: {RAW_DATA_DIR}")
        return

    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    logger.info(f"Found {len(raw_files)} historic records inside target storage directory.")

    results = []
    rejection_counts = {}
    min_required_history = config.lookback + 30

    for file in raw_files:
        symbol = file.split(".")[0].split("_")[0].upper()

        if symbol not in active_tickers:
            rejection_counts["inactive_ticker"] = rejection_counts.get("inactive_ticker", 0) + 1
            continue

        path = os.path.join(RAW_DATA_DIR, file)
        
        try:
            df = pd.read_csv(path)
            df = normalize_and_validate_df(df)

            if len(df) < min_required_history:
                rejection_counts["insufficient_history"] = rejection_counts.get("insufficient_history", 0) + 1
                continue

            df = add_macro_indicators(df, config)
            setup = evaluate_coiled_cobra(df, config)

            if not setup:
                rejection_counts["IGNORE"] = rejection_counts.get("IGNORE", 0) + 1
                continue

            latest = df.iloc[-1]
            results.append({
                "Symbol": symbol,
                "Price": round(latest["Close"], 2),
                "Score": setup["Score"],
                "Grade": setup["Grade"],
                "Struct_Fctr": setup["Structure_Score"],
                "Mmtm_Fctr": setup["Momentum_Score"],
                "Vol_Fctr": setup["Volatility_Score"],
                "ATR": round(latest["ATR_14"], 2)
            })

        except (KeyError, ValueError) as e:
            logger.warning(f"Technical validation failure for {symbol}: {str(e)}")
            rejection_counts["validation_failure"] = rejection_counts.get("validation_failure", 0) + 1
        except Exception:
            logger.exception(f"Unexpected architectural error tracking: {symbol}")
            rejection_counts["execution_error"] = rejection_counts.get("execution_error", 0) + 1

    if results:
        df_out = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        print("\n" + df_out.to_markdown(index=False) + "\n")

        today = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(LOG_DIR, f"quant_cobra_setups_{today}.csv")
        df_out.to_csv(out_path, index=False)
        logger.info(f"Systematic run matrix log stored at: {out_path}")
    else:
        logger.warning("Zero high-confluence Quant Cobra setups matched criteria across assets.")

    logger.info("Factor Matrix Processing Complete. Rejection Summary Tracking:")
    for k, v in rejection_counts.items():
        logger.info(f"  {k}: {v}")

# =====================================================================
# SYSTEM ENTRY
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coiled Cobra Quantitative Factor Scanner Engine")
    parser.add_argument(
        "mode", 
        choices=["daily", "weekly"], 
        nargs="?", 
        default="weekly",
        help="Timeframe tuning parameter for data lookbacks and folders (default: weekly)"
    )
    args = parser.parse_args()
    run_scanner(args.mode)