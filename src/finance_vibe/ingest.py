import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_stock_data(ticker_symbol):
    # Create the data directory if it doesn't exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    # Fetch data
    print(f"Fetching data for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="1mo")
    
    # Save to our 'raw' folder
    file_path = f"data/raw/{ticker_symbol}_history.csv"
    df.to_csv(file_path)
    print(f"Success! Data saved to {file_path}")

if __name__ == "__main__":
    fetch_stock_data("TSLA")