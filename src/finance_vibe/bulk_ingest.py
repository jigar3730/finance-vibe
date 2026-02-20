import yfinance as yf
import pandas as pd
import os
import time

def get_most_active_tickers(count=20):
    """
    Dynamically fetches the current top 20 most active tickers.
    """
    print(f"--- DISCOVERY: Finding top {count} most active tickers ---")
    try:
        # Use yfinance screener to get active stocks
        active_df = yf.Search("", max_results=count).active
        tickers = active_df['symbol'].tolist()
        
        # Clean list to ensure they are standard tickers
        tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
        print(f"Discovered: {tickers}")
        return tickers
    except Exception as e:
        print(f"Discovery failed: {e}. Using fallback list.")
        return ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "AMD", "MSFT", "AMZN", "META", "GOOGL"]

def bulk_ingest():
    # 1. Setup Directories
    save_path = "data/raw"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 2. Get Dynamic Ticker List
    tickers = get_most_active_tickers(20)

    # 3. Download Data
    print(f"\n--- INGESTION: Downloading data for {len(tickers)} tickers ---")
    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...")
            # We pull 2 years to ensure we have enough for 200-MA calculations
            df = yf.download(ticker, period="5y", interval="1d", progress=False)
            
            if not df.empty:
                # Standardize format: ensure index is Date and save
                file_name = f"{save_path}/{ticker}.csv"
                df.to_csv(file_name)
            else:
                print(f"Warning: No data found for {ticker}")
                
            # Respectful delay for API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

    print("\n--- DONE: Bulk Ingestion Complete ---")

if __name__ == "__main__":
    bulk_ingest()