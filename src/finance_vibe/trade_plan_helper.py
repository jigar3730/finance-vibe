# src/finance_vibe/trade_plan_helper.py

import pandas as pd
from datetime import datetime
import os

# ---- Config ----
TRADE_PLAN_DIR = "./data/logs"
today_str = datetime.now().strftime("%Y-%m-%d")
scanner_csv = os.path.join(TRADE_PLAN_DIR, f"trade_plan_{today_str}.csv")

# ---- Read CSV safely ----
try:
    df = pd.read_csv(scanner_csv)
except FileNotFoundError:
    print(f"❌ File not found: {scanner_csv}")
    exit(1)

# Strip any whitespace from column headers
df.columns = df.columns.str.strip()

print("✅ Loaded CSV columns:", df.columns.tolist())

# ---- Convert numeric columns safely ----
numeric_cols = ["Stock Entry", "Stock Stop", "Target 1", "Target 2"]
numeric_cols_existing = [c for c in numeric_cols if c in df.columns]

for col in numeric_cols_existing:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- Parse Suggested Delta ----
if "Suggested Delta" in df.columns:
    delta_split = df["Suggested Delta"].astype(
        str).str.split("–|-", expand=True)
    if delta_split.shape[1] == 2:
        df["Delta Min"] = pd.to_numeric(delta_split[0], errors="coerce")
        df["Delta Max"] = pd.to_numeric(delta_split[1], errors="coerce")
    else:
        print("⚠️ Could not parse Suggested Delta properly. Check format.")

# ---- Optional: Inspect first few rows ----
print("\n📄 Trade Plan Preview:")
print(df.head(10).to_markdown(index=False))

# ---- Save cleaned CSV if needed ----
clean_csv = os.path.join(TRADE_PLAN_DIR, f"trade_plan_clean_{today_str}.csv")
df.to_csv(clean_csv, index=False)
print(f"\n✅ Cleaned trade plan saved: {clean_csv}")
