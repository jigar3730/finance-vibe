import pandas as pd
from datetime import datetime
from pathlib import Path

# --------- CONFIG ----------
LEAPS_MIN_MONTHS = 12
LEAPS_MAX_MONTHS = 24
DELTA_LONG = (0.65, 0.80)
DELTA_SHORT = (-0.80, -0.65)
SCANNER_DIR = Path("./data/logs")  # directory containing swing scanner CSVs
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
        leaps_type = 'CALL'
        delta_range = DELTA_LONG
    else:  # SETUP_SHORT
        entry = min(ema20, close + 0.25 * atr)
        stop = ema50 + 0.5 * atr
        target1 = entry - 1 * atr
        target2 = entry - 2 * atr
        leaps_type = 'PUT'
        delta_range = DELTA_SHORT

    return entry, stop, target1, target2, leaps_type, delta_range


def calculate_leaps_expiry():
    today = datetime.today()
    expiry_min = today + pd.DateOffset(months=LEAPS_MIN_MONTHS)
    expiry_max = today + pd.DateOffset(months=LEAPS_MAX_MONTHS)
    return expiry_min.strftime('%b-%Y'), expiry_max.strftime('%b-%Y')

# --------- MAIN FUNCTION ----------


def generate_trade_plan(scanner_csv_path=None):
    # Auto-detect latest scanner CSV if none provided
    if scanner_csv_path is None:
        files = list(SCANNER_DIR.glob(f"{SCANNER_PREFIX}*.csv"))
        if not files:
            raise FileNotFoundError(
                "No swing setup CSVs found in logs directory.")
        # pick the latest by date in filename
        files.sort(key=lambda f: f.stem.split("_")[-1], reverse=True)
        scanner_csv_path = files[0]
        print(f"Using latest scanner file: {scanner_csv_path}")

    df = pd.read_csv(scanner_csv_path)

    plan_rows = []
    for _, row in df.iterrows():
        entry, stop, t1, t2, leaps_type, delta_range = calculate_stock_levels(
            row)
        expiry_min, expiry_max = calculate_leaps_expiry()

        plan_rows.append({
            'Symbol': row['Symbol'],
            'Setup Type': row['Setup Type'],
            'Stock Entry': round(entry, 2),
            'Stock Stop': round(stop, 2),
            'Target 1': round(t1, 2),
            'Target 2': round(t2, 2),
            'LEAPS Type': leaps_type,
            'LEAPS Expiry Min': expiry_min,
            'LEAPS Expiry Max': expiry_max,
            'Suggested Delta': f"{delta_range[0]} – {delta_range[1]}",
            'Risk Notes': 'Stop based on EMA50; adjust if invalidated'
        })

    plan_df = pd.DataFrame(plan_rows)

    # Auto-generate output filename
    scanner_file = Path(scanner_csv_path)
    date_str = scanner_file.stem.split("_")[-1]
    output_csv_path = SCANNER_DIR / f"{OUTPUT_PREFIX}{date_str}.csv"

    plan_df.to_csv(output_csv_path, index=False)
    print(f"Trade plan exported: {output_csv_path}")
    return plan_df


# --------- USAGE ----------
if __name__ == "__main__":
    generate_trade_plan()
