"""Trade plan generator: stock levels and options metadata from swing setups.

Reads ``swing_setups_<date>.csv`` and writes ``trade_plan_<date>.csv`` with
entry, stop, ATR targets, and LEAPS (weekly) or short-dated options (daily) fields.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# =========================
# PROFILE CONFIGURATION
# =========================
if len(sys.argv) > 1 and sys.argv[1].lower() in ["weekly", "daily"]:
    mode = sys.argv[1].lower()
else:
    print("⚠️ Unknown mode parsed to trade planner. Defaulting to 'weekly'.")
    mode = "weekly"

# --------- CONFIG ----------
DELTA_LONG = (0.65, 0.80)
DELTA_SHORT = (-0.80, -0.65)

# Dynamic path resolution according to isolation architecture
BASE_DIR = Path(__file__).resolve().parents[2]
SCANNER_DIR = BASE_DIR / "data" / "logs" / mode
SCANNER_PREFIX = "swing_setups_"
COILED_PREFIX = "coiled_cobra_setups_"
OUTPUT_PREFIX = "trade_plan_"

# --------- HELPER FUNCTIONS ----------


def calculate_stock_levels(row):
    """
    Derive entry, stop, targets, option side, and delta band from one setup row.
    Supports:
      - Source: 'swing' or 'coiled_cobra'
      - Setup Type: 'SETUP_LONG' (currently), later 'SETUP_SHORT'
    """
    atr = row["ATR"]
    close = row["Close"]
    ema20 = row["EMA20"]
    ema50 = row["EMA50"]
    fib786 = row.get("Fib 78.6%", None)
    source = row.get("Source", "swing")
    setup_type = row["Setup Type"]

    # Default option params (long-only for now)
    options_type = "CALL"
    delta_range = DELTA_LONG

    if setup_type == "SETUP_LONG":
        if source == "coiled_cobra" and fib786 is not None:
            # --- Coiled Cobra macro-reversal long ---
            # Entry: near Fib 78.6 or slight pullback from close
            entry = max(fib786, close - 0.25 * atr)

            # Stop: below Fib 78.6 (structural invalidation), with ATR buffer
            stop = fib786 - 0.5 * atr

            # Targets: ATR-based
            target1 = entry + 1.0 * atr
            target2 = entry + 2.0 * atr

        else:
            # --- Swing long (existing logic or refined) ---
            # Entry: pullback towards EMA20, but not above close by too much
            entry = max(ema20, close - 0.25 * atr)

            # Stop: below EMA50 with ATR buffer
            stop = ema50 - 0.5 * atr

            # Targets: ATR-based
            target1 = entry + 1.0 * atr
            target2 = entry + 2.0 * atr

        # SAFETY: ensure stop < entry for longs
        stop = min(stop, entry - 0.25 * atr)

    elif setup_type == "SETUP_SHORT":
        # Future short logic (placeholder)
        entry = min(ema20, close + 0.25 * atr)
        stop = ema50 + 0.5 * atr
        target1 = entry - 1.0 * atr
        target2 = entry - 2.0 * atr
        options_type = "PUT"
        delta_range = DELTA_SHORT

        # SAFETY: ensure stop > entry for shorts
        stop = max(stop, entry + 0.25 * atr)

    else:
        raise ValueError(f"Unknown Setup Type: {setup_type}")

    return entry, stop, target1, target2, options_type, delta_range


def calculate_options_expiry():
    """
    Dynamically adjusts structural contracts option timeline based on profile timeframe mode.
    Weekly pulls long-term LEAPS setups; daily pulls agile swing cycles.
    """
    today = datetime.today()
    if mode == "daily":
        # Standard swing option cycle boundaries (1 to 3 months forward lookahead)
        expiry_min = today + pd.DateOffset(months=1)
        expiry_max = today + pd.DateOffset(months=3)
    else:
        # Legacy LEAPS macro cycles (12 to 24 months forward lookahead)
        expiry_min = today + pd.DateOffset(months=12)
        expiry_max = today + pd.DateOffset(months=24)

    return expiry_min.strftime("%b-%Y"), expiry_max.strftime("%b-%Y")


# --------- MAIN FUNCTION ----------


def generate_trade_plan(scanner_csv_path=None):
    """Build and export a trade plan CSV from the latest or provided scanner output."""
    print(
        f"--- STEP 5: Drafting Trade Execution Architectures [{mode.upper()} MODE] ---"
    )

    # Auto-detect latest scanner CSVs inside isolated subdirectory if none provided
    if scanner_csv_path is None:
        if not SCANNER_DIR.exists():
            print(f"⚠️ Target scanner directory empty or non-existent: {SCANNER_DIR}")
            return None

        # Find latest swing scanner file
        swing_files = list(SCANNER_DIR.glob(f"{SCANNER_PREFIX}*.csv"))
        swing_files.sort(key=lambda f: f.stem.split("_")[-1], reverse=True)
        swing_csv_path = swing_files[0] if swing_files else None

        # Find latest Coiled Cobra scanner file
        cobra_files = list(SCANNER_DIR.glob(f"{COILED_PREFIX}*.csv"))
        cobra_files.sort(key=lambda f: f.stem.split("_")[-1], reverse=True)
        cobra_csv_path = cobra_files[0] if cobra_files else None

        if swing_csv_path is None and cobra_csv_path is None:
            print(
                f"⚠️ No active setup archives discovered in {SCANNER_DIR}. Exiting plan generation."
            )
            return None

        print(f"Using swing scanner file: {swing_csv_path}")
        print(f"Using Coiled Cobra scanner file: {cobra_csv_path}")

    # Load swing setups
    df_swing = None
    if swing_csv_path:
        df_swing = pd.read_csv(swing_csv_path)
        df_swing["Source"] = "Swing"
        print(f"Loaded {len(df_swing)} swing setups.")

    # Load Coiled Cobra setups
    df_cobra = None
    if cobra_csv_path:
        df_cobra = pd.read_csv(cobra_csv_path)
        df_cobra["Source"] = "Cobra"
        print(f"Loaded {len(df_cobra)} Coiled Cobra setups.")

    # Combine into one DataFrame
    dfs = [df for df in [df_swing, df_cobra] if df is not None and not df.empty]
    if not dfs:
        print("⚠️ All setup archives are empty. Skipping calculations.")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined total setups: {len(df)}")

    plan_rows = []
    expiry_label_min = "Options Expiry Min" if mode == "daily" else "LEAPS Expiry Min"
    expiry_label_max = "Options Expiry Max" if mode == "daily" else "LEAPS Expiry Max"
    contract_label = "Options Type" if mode == "daily" else "LEAPS Type"

    for _, row in df.iterrows():
        entry, stop, t1, t2, opt_type, delta_range = calculate_stock_levels(row)
        expiry_min, expiry_max = calculate_options_expiry()

        plan_rows.append(
            {
                "Symbol": row["Symbol"],
                "Setup Type": row["Setup Type"],
                "Stock Entry": round(entry, 2),
                "Stock Stop": round(stop, 2),
                "Target 1": round(t1, 2),
                "Target 2": round(t2, 2),
                "Score": row.get("Score", None),
                "Grade": row.get("Grade", None),
                "Checks Met": row.get("Checks Met", None),
                "Source": row["Source"],  # optional but useful
            }
        )

    plan_df = pd.DataFrame(plan_rows)

    # Auto-generate output filename within isolated directory block context
    # Use the latest of the two detected files (swing or cobra)
    latest_file = swing_csv_path or cobra_csv_path
    if cobra_csv_path and swing_csv_path:
        swing_date = swing_csv_path.stem.split("_")[-1]
        cobra_date = cobra_csv_path.stem.split("_")[-1]
        latest_file = swing_csv_path if swing_date >= cobra_date else cobra_csv_path

    scanner_file = Path(latest_file)
    date_str = scanner_file.stem.split("_")[-1]
    output_csv_path = SCANNER_DIR / f"{OUTPUT_PREFIX}{date_str}.csv"

    os.makedirs(SCANNER_DIR, exist_ok=True)
    plan_df.to_csv(output_csv_path, index=False)
    print(f"✅ Trade plan exported successfully to: {output_csv_path}")
    return plan_df


# --------- USAGE ----------
if __name__ == "__main__":
    generate_trade_plan()
