import os
import sys
import gc
import subprocess
import pandas as pd
import numpy as np
from scipy import stats

# 1. 40-Ticker Balanced Test Universe
UNIVERSE = [
    "HOOD", "PLTR", "NVDA", "AAPL", "AMD", "COIN", "SMCI", "AMZN", "TSLA", "SPY",
    "QQQ", "DUOL", "SOFI", "MELI", "ANET", "IWM", "UPST", "MSFT", "GOOG", "GOOGL",
    "META", "NKLA", "PTON", "TDOC", "RIVN", "CVNA", "DOCU", "PYPL", "XLE", "XLF",
    "JNJ", "LLY", "JPM", "PG", "COST", "CAT", "LMT", "UNH", "XOM", "GLD"
]

BACKTEST_SCRIPT = "src/finance_vibe/coiled_cobra_backtest.py"
OUTPUT_DIR = "/app/data/logs/daily"


def run_backtest_pipeline(tickers):
    """Runs the CLI backtest and generates the fresh log file on disk."""
    ticker_str = ",".join(tickers)
    cmd = ["python", BACKTEST_SCRIPT, "--backtest", "--tickers", ticker_str]
    print(f"[*] Running backtest across {len(tickers)} tickers...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Backtest execution failed:\n{result.stderr}")
        sys.exit(1)
    print("[+] Backtest run complete. File saved to disk.")


def get_latest_backtest_csv(log_dir):
    """Finds the newly generated CSV file on disk."""
    files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("coiled_cobra_backtest_trades_")]
    if not files:
        raise FileNotFoundError(f"No backtest CSV found in {log_dir}")
    return max(files, key=os.path.getctime)


def audit_generated_file_quality(file_path):
    """Reads ONLY the CSV file from disk and performs data quality & ML readiness checks."""
    print(f"\n[*] Reading generated file from disk: {file_path}")
    
    # Load raw CSV freshly from disk
    df_raw = pd.read_csv(file_path)
    total_records = len(df_raw)
    
    print("\n========================================================")
    print("      DATA QUALITY & ML READINESS ASSESSMENT           ")
    print("========================================================\n")

    # 1. Categorize 114 Columns based on Exact File Header
    target_prefixes = ("Forward_", "Rel_Forward_", "Max_", "MAE_", "Win_", "Hit_", "Held_")
    target_cols = [c for c in df_raw.columns if c.startswith(target_prefixes)]
    
    metadata_cols = ["Symbol", "Setup Type", "Source", "Mode", "AsOf Date", "Date", "Signal Date", "Notes", "Grade", "Score"]
    
    # Raw price levels (Non-stationary for Deep Learning)
    unstationary_raw_cols = ["Close", "EMA20", "EMA50", "ATR", "Swing Low", "Swing High", "Coil_High", "Coil_Low", "Pivot_Price", "Fib 61.8%", "Fib 78.6%"]

    # Filter strictly for clean Deep Learning candidate features
    feature_cols = [
        c for c in df_raw.select_dtypes(include=[np.number]).columns
        if c not in target_cols and c not in metadata_cols and c not in unstationary_raw_cols
    ]

    print(f"1. Dataset Shape: {total_records} rows x {len(df_raw.columns)} total columns across {df_raw['Symbol'].nunique()} tickers.")
    print(f"   - Isolated Target Metrics: {len(target_cols)} columns")
    print(f"   - Deep Learning Candidate Features: {len(feature_cols)} columns")

    # 2. Null & Missing Data Audit (Drop completely empty columns to save memory)
    null_counts = df_raw[feature_cols].isnull().sum()
    all_null_cols = null_counts[null_counts == total_records].index.tolist()
    partial_null_cols = null_counts[(null_counts > 0) & (null_counts < total_records)]

    if all_null_cols:
        print(f"\n2. Null Value Audit:\n   [!] Dropping 100% empty feature columns: {all_null_cols}")
        feature_cols = [c for c in feature_cols if c not in all_null_cols]
    if not partial_null_cols.empty:
        print(f"   [!] Minor missing values detected:\n{partial_null_cols}")
    else:
        print("2. Null Value Check: CLEAN (Active features have 0 missing values).")

    # 3. Target Distribution & Class Balance (21-Day Forward Return)
    target_main = "Forward_Return_21d"
    if target_main in df_raw.columns:
        pos_ratio = (df_raw[target_main] > 0).mean()
        print(f"\n3. Target Balance ({target_main} > 0):")
        print(f"   - Win Rate: {pos_ratio:.2%}")
        print(f"   - Mean Return: {df_raw[target_main].mean():.4f} | Median: {df_raw[target_main].median():.4f}")

    # 4. Strict Lookahead Leakage Audit (Features vs Target)
    corrs = df_raw[feature_cols].apply(lambda col: df_raw[target_main].corr(col))
    leakage_suspects = corrs[corrs.abs() > 0.80]
    print("\n4. Lookahead Leakage Audit (|r| > 0.80 with Target):")
    if not leakage_suspects.empty:
        print(f"   [!] Potential Leakage Detected:\n{leakage_suspects}")
    else:
        print("   [+] CLEAN - 0 features leak forward target performance.")

    # 5. Temporal Feature Stationarity (KS-Test Early vs Late Signals)
    df_sorted = df_raw.sort_values(by="AsOf Date")
    mid = len(df_sorted) // 2
    p1, p2 = df_sorted.iloc[:mid], df_sorted.iloc[mid:]

    drift_features = []
    for col in feature_cols:
        s1, s2 = p1[col].dropna(), p2[col].dropna()
        if len(s1) > 20 and len(s2) > 20:
            stat, p_val = stats.ks_2samp(s1, s2)
            if p_val < 0.01:
                drift_features.append((col, round(stat, 3), round(p_val, 4)))

    print("\n5. Normalized Feature Stationarity (KS-Test):")
    if drift_features:
        print(f"   [!] Features exhibiting temporal distribution shift (p < 0.01):")
        for f, stat, p in drift_features[:5]:
            print(f"       - {f}: KS-Stat={stat}, p-value={p}")
    else:
        print("   [+] CLEAN - Feature distributions remain stable across market regimes.")

    # 6. Redundant Multicollinearity Check (|r| > 0.95)
    corr_matrix = df_raw[feature_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant = [(col, upper_tri[col][upper_tri[col] > 0.95].index.tolist()) for col in upper_tri.columns if any(upper_tri[col] > 0.95)]

    print("\n6. Feature Redundancy Check (|r| > 0.95):")
    if redundant:
        print(f"   [!] Identified {len(redundant)} redundant feature groups:")
        for col, pairs in redundant[:5]:
            print(f"       - {col} correlated with: {pairs}")
    else:
        print("   [+] CLEAN - Features show low pairwise redundancy.")

    # Memory Cleanup: Flush dataframes and force Garbage Collection
    del df_raw, df_sorted, p1, p2, corr_matrix, upper_tri
    gc.collect()
    print("\n[+] Validation complete. Temporary DataFrames flushed from RAM.")
    print("========================================================\n")


if __name__ == "__main__":
    run_backtest_pipeline(UNIVERSE)
    latest_file = get_latest_backtest_csv(OUTPUT_DIR)
    audit_generated_file_quality(latest_file)