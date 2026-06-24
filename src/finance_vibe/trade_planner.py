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
OUTPUT_PREFIX = "trade_plan_"

# --------- HELPER FUNCTIONS ----------

def calculate_stock_levels(row):
    atr = row['ATR']
    close = row['Close']
    ema20 = row['EMA20']
    ema50 = row['EMA50']

    if row['Setup Type'] == 'SETUP_LONG':
        entry = max(ema20, close - 0.25 * atr)
        stop = ema50 - 0.5 * atr
        target1 = entry + 1 * atr
        target2 = entry + 2 * atr
        options_type = 'CALL'
        delta_range = DELTA_LONG
    else:  # SETUP_SHORT
        entry = min(ema20, close + 0.25 * atr)
        stop = ema50 + 0.5 * atr
        target1 = entry - 1 * atr
        target2 = entry - 2 * atr
        options_type = 'PUT'
        delta_range = DELTA_SHORT

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
        
    return expiry_min.strftime('%b-%Y'), expiry_max.strftime('%b-%Y')

# --------- MAIN FUNCTION ----------

def generate_trade_plan(scanner_csv_path=None):
    print(f"--- STEP 4: Drafting Trade Execution Architectures [{mode.upper()} MODE] ---")
    
    # Auto-detect latest scanner CSV inside isolated subdirectory if none provided
    if scanner_csv_path is None:
        if not SCANNER_DIR.exists():
            print(f"⚠️ Target scanner directory empty or non-existent: {SCANNER_DIR}")
            return None
            
        files = list(SCANNER_DIR.glob(f"{SCANNER_PREFIX}*.csv"))
        if not files:
            print(f"⚠️ No active setup archives discovered in {SCANNER_DIR}. Exiting plan generation.")
            return None
            
        # pick the latest by date in filename string indexing
        files.sort(key=lambda f: f.stem.split("_")[-1], reverse=True)
        scanner_csv_path = files[0]
        print(f"Using latest scanner file: {scanner_csv_path}")

    df = pd.read_csv(scanner_csv_path)
    if df.empty:
        print("⚠️ Found setup archive is completely empty. Skipping calculations.")
        return None

    plan_rows = []
    expiry_label_min = 'Options Expiry Min' if mode == 'daily' else 'LEAPS Expiry Min'
    expiry_label_max = 'Options Expiry Max' if mode == 'daily' else 'LEAPS Expiry Max'
    contract_label = 'Options Type' if mode == 'daily' else 'LEAPS Type'

    for _, row in df.iterrows():
        entry, stop, t1, t2, opt_type, delta_range = calculate_stock_levels(row)
        expiry_min, expiry_max = calculate_options_expiry()

        plan_rows.append({
            'Symbol': row['Symbol'],
            'Setup Type': row['Setup Type'],
            'Stock Entry': round(entry, 2),
            'Stock Stop': round(stop, 2),
            'Target 1': round(t1, 2),
            'Target 2': round(t2, 2),
            contract_label: opt_type,
            expiry_label_min: expiry_min,
            expiry_label_max: expiry_max,
            'Suggested Delta': f"{delta_range[0]} – {delta_range[1]}",
            'Risk Notes': 'Stop based on EMA50; adjust if invalidated'
        })

    plan_df = pd.DataFrame(plan_rows)

    # Auto-generate output filename within isolated directory block context
    scanner_file = Path(scanner_csv_path)
    date_str = scanner_file.stem.split("_")[-1]
    output_csv_path = SCANNER_DIR / f"{OUTPUT_PREFIX}{date_str}.csv"

    os.makedirs(SCANNER_DIR, exist_ok=True)
    plan_df.to_csv(output_csv_path, index=False)
    print(f"✅ Trade plan exported successfully to: {output_csv_path}")
    return plan_df


# --------- USAGE ----------
if __name__ == "__main__":
    generate_trade_plan()