import yfinance as yf
import pandas as pd
import os
import sys

# --- 1. PACKAGE IMPORT (Upgraded for Multi-Environment Paths) ---
try:
    # 1st Priority: Docker module structure (python -m src.finance_vibe.data_ingestor)
    from src.finance_vibe import config
except ImportError:
    try:
        # 2nd Priority: Flat direct package fallback
        from finance_vibe import config
    except ImportError:
        # 3rd Priority: Manual repository root insertion for local direct execution
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from src.finance_vibe import config


def ingest_market_data(mode="weekly"):
    # --- 2. EXTRACT SETTINGS DYNAMICALLY FROM PROFILE ---
    mode_cfg = config.get_mode_config(mode)
    
    csv_path = config.TICKER_LIST_PATH
    raw_dir = mode_cfg['raw_dir']
    PERIOD = mode_cfg['period']
    INTERVAL = mode_cfg['interval']

    if not os.path.exists(csv_path):
        print(f"❌ Could not find ticker list at {csv_path}. Please run ticker_provider.py first.")
        return

    # Ensure targeted sub-silo raw data directory exists
    os.makedirs(raw_dir, exist_ok=True)

    # Read tickers and drop any duplicates/NaNs
    tickers = pd.read_csv(csv_path)['Ticker'].dropna().unique().tolist()

    print(f"\n--- STEP 2: Ingesting [{mode.upper()}] {PERIOD} {INTERVAL} data ---")
    print(f"Target Directory: {raw_dir}")

    for ticker in tickers:
        print(f"Processing {ticker:6}...", end=" ", flush=True)
        try:
            # --- 3. DOWNLOAD ---
            # auto_adjust=True handles splits/dividends for cleaner backtesting
            df = yf.download(ticker, period=PERIOD,
                             interval=INTERVAL, progress=False, auto_adjust=True)

            if df.empty:
                print("⚠️ No data found.")
                continue

            # Flatten MultiIndex columns (common in newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # --- 4. DATA CLEANING ---
            # Remove the last row if it's an incomplete weekly candle (only for 1wk)
            if INTERVAL == "1wk":
                last_date = df.index[-1]
                if last_date.weekday() != 4:  # If last row isn't a Friday
                    df = df.iloc[:-1]

            # Ensure columns are standardized (Open, High, Low, Close, Volume)
            df.columns = [c.capitalize() for c in df.columns]

            # --- 5. SAVE ---
            # Construct the path directly using the mode's target folder and parameters
            filename = f"{ticker}_{PERIOD}_{INTERVAL}.csv"
            save_path = os.path.join(raw_dir, filename)
            
            df.to_csv(save_path)
            print(f"✅ {os.path.basename(save_path)}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Check for CLI argument, otherwise default to weekly execution
    selected_mode = "weekly"
    if len(sys.argv) > 1:
        arg_mode = sys.argv[1].lower()
        if arg_mode in ["weekly", "daily"]:
            selected_mode = arg_mode
        else:
            print(f"⚠️ Unknown mode '{arg_mode}'. Defaulting to 'weekly'.")

    ingest_market_data(mode=selected_mode)