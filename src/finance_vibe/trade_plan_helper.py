# src/finance_vibe/trade_plan_helper.py
import pandas as pd
from datetime import datetime
import os
import sys
import traceback

# ---- Config & Absolute Path Handling ----

today_str = datetime.now().strftime("%Y-%m-%d")
filename = f"trade_plan_{today_str}.csv"

# Check standard local path first, then Docker path fallback
possible_dirs = ["./data/logs", "/app/data/logs", "data/logs"]
scanner_csv = None

for p_dir in possible_dirs:
    check_path = os.path.join(p_dir, filename)
    if os.path.exists(check_path):
        TRADE_PLAN_DIR = p_dir
        scanner_csv = check_path
        break

if not scanner_csv:
    # If not found anywhere, default to the local path to show the error message
    TRADE_PLAN_DIR = "./data/logs"
    scanner_csv = os.path.join(TRADE_PLAN_DIR, filename)
    print(f"❌ File not found anywhere: {filename}")
    print(sys.exc_info())
    exit(1)

print(f"🎯 Target trade plan file located: {scanner_csv}")

# ---- Read CSV safely ----

try:
    df = pd.read_csv(scanner_csv)
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# Strip any whitespace from column headers
df.columns = df.columns.str.strip()
print("✅ Loaded CSV columns:", df.columns.tolist())

# ---- Convert numeric columns safely ----

numeric_cols = ["Stock Entry", "Stock Stop", "Target 1", "Target 2"]
numeric_cols_existing = [c for c in numeric_cols if c in df.columns]

for col in numeric_cols_existing:
    # Coerce errors to NaN, and stripping out common symbols like '$' or commas if present
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.replace(r"[$,]", "", regex=True)
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- Parse Suggested Delta ----

if "Suggested Delta" in df.columns:
    # Using raw string parsing while capturing both en-dash and hyphen types
    delta_split = df["Suggested Delta"].astype(str).str.split(r"–|-", expand=True)
    if delta_split.shape[1] >= 2:
        df["Delta Min"] = pd.to_numeric(delta_split[0].str.strip(), errors="coerce")
        df["Delta Max"] = pd.to_numeric(delta_split[1].str.strip(), errors="coerce")
    else:
        # Fallback if no delimiter range is matched (e.g. single static delta value)
        df["Delta Min"] = pd.to_numeric(df["Suggested Delta"], errors="coerce")
        df["Delta Max"] = df["Delta Min"]
else:
    print("⚠️ 'Suggested Delta' column missing from data framework.")

# ---- Risk-to-Reward (R:R) Calculations Engine ----

print("🧮 Calculating Risk-to-Reward distributions...")

try:
    # Defensive math implementation to avoid ZeroDivisionError or crashes on malformed data
    df["Risk Per Share"] = df["Stock Entry"] - df["Stock Stop"]
    
    # Calculate Target rewards safely using .div() to prevent literal 0 division panics
    df["Reward T1"] = df["Target 1"] - df["Stock Entry"]
    df["Reward T2"] = df["Target 2"] - df["Stock Entry"]
    
    # Fill 0 risk with NaN to bypass division errors gracefully
    safe_risk = df["Risk Per Share"].replace(0, pd.NA)
    
    df["R:R T1"] = round(df["Reward T1"].astype(float) / safe_risk.astype(float), 2)
    df["R:R T2"] = round(df["Reward T2"].astype(float) / safe_risk.astype(float), 2)
    
    # Clean up intermediate reward columns to keep output tidy if desired
    df.drop(columns=["Reward T1", "Reward T2"], errors="ignore", inplace=True)

except Exception as math_err:
    print("❌ Fatal exception caught inside metrics distribution generation engine:")
    traceback.print_exc()
    exit(1)

# ---- Inspect first few rows ----

print("\n📄 Cleaned Trade Plan Preview:")
print(df.head(10).to_markdown(index=False))

# ---- Save cleaned CSV ----

clean_csv = os.path.join(TRADE_PLAN_DIR, f"trade_plan_clean_{today_str}.csv")
try:
    df.to_csv(clean_csv, index=False)
    print(f"\n✅ Cleaned trade plan saved: {clean_csv}")
except Exception as save_err:
    print(f"❌ Error saving cleaned file: {save_err}")
# ---- Save cleaned CSV ----

