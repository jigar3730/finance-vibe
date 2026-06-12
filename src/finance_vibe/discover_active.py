import yfinance as yf
import pandas as pd
import os
import time

def get_most_active_tickers(count=20):
    """
    Tries to fetch active tickers, but falls back to a 
    reliable 'Top 20' list if the API experimental feature fails.
    """
    print(f"--- DISCOVERY: Attempting to find top {count} tickers ---")
    
    # Professional Data Science Fallback List (S&P 500 / NASDAQ Leaders)
    fallback_list = [
        "SPY", "QQQ", "NVDA", "AAPL", "TSLA", "AMD", "MSFT", "AMZN", "META", "GOOGL",
        "PLTR", "MSTR", "AMD", "AVGO", "SMCI", "COIN", "MARA", "IBIT", "NFLX", "BRK-B"
    ]

    try:
        # Attempting the most stable way to get trending tickers in 2026
        # If this fails, it jumps to the 'except' block immediately
        active_df = yf.Search("", max_results=count).active
        tickers = active_df['symbol'].tolist()
        tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
        
        if not tickers:
            return fallback_list
            
        return tickers[:count]

    except Exception:
        print("⚠️ Discovery experimental feature unavailable. Using high-volume fallback list.")
        return fallback_list[:count]

def bulk_ingest():
    save_path = "data/raw"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Use our new robust discovery logic
    tickers = get_most_active_tickers(20)

    print(f"\n--- INGESTION: Downloading data for {len(tickers)} tickers ---")
    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...")
            # We use '2y' to ensure the 200-day Moving Average has enough data points
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            
            if not df.empty:
                # Save to the standardized project path
                file_name = f"{save_path}/{ticker}.csv"
                df.to_csv(file_name)
            
            time.sleep(0.5) # Avoid rate-limiting
            
        except Exception as e:
            print(f"Failed {ticker}: {e}")

    print("\n✅ Bulk Ingestion Complete.")

if __name__ == "__main__":
    bulk_ingest()