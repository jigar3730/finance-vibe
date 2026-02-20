import yfinance as yf
import pandas as pd
import os

def ingest_weekly_data():
    csv_path = 'data/active_tickers.csv'
    raw_dir = 'data/raw'
    os.makedirs(raw_dir, exist_ok=True)

    # Define our parameters as variables so they are easy to change/label
    PERIOD = "5y"
    INTERVAL = "1wk"

    if not os.path.exists(csv_path):
        print(f"❌ Could not find {csv_path}.")
        return

    tickers = pd.read_csv(csv_path)['Ticker'].tolist()
    print(f"--- STEP 2: Ingesting {PERIOD} {INTERVAL} data ---")

    for ticker in tickers:
        print(f"Processing {ticker}...", end=" ", flush=True)
        try:
            df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
            
            if df.empty:
                print("⚠️ Empty.")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Friday Filter
            if df.index[-1].weekday() != 4:
                df = df.iloc[:-1]

            # SMART FILENAME: e.g., NVDA_5y_1wk.csv
            file_name = f"{ticker}_{PERIOD}_{INTERVAL}.csv"
            df.to_csv(os.path.join(raw_dir, file_name))
            print(f"✅ Saved as {file_name}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    ingest_weekly_data()