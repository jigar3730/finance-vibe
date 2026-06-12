import yfinance as yf
import pandas as pd
import os
import sys

# --- 1. PACKAGE IMPORT ---
try:
    from finance_vibe import config
except ImportError:
    # This allows the script to find config if run directly during testing
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")))
    from finance_vibe import config


def ingest_market_data():
    # --- 2. EXTRACT SETTINGS FROM CONFIG ---
    csv_path = config.TICKER_LIST_PATH
    raw_dir = config.RAW_DIR
    PERIOD = config.PERIOD
    INTERVAL = config.INTERVAL

    if not os.path.exists(csv_path):
        print(
            f"❌ Could not find ticker list at {csv_path}. Please run ticker_provider.py first.")
        return

    # Read tickers and drop any duplicates/NaNs
    tickers = pd.read_csv(csv_path)['Ticker'].dropna().unique().tolist()

    print(f"--- STEP 2: Ingesting {PERIOD} {INTERVAL} data ---")
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
            save_path = config.get_raw_path(ticker)
            df.to_csv(save_path)
            print(f"✅ {os.path.basename(save_path)}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    ingest_market_data()
