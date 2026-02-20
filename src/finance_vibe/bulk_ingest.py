import yfinance as yf
import argparse
from pathlib import Path

def fetch_bulk_data(tickers):
    # Ensure data directory exists
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    for symbol in tickers:
        symbol = symbol.upper()  # Ensure ticker is uppercase
        print(f"📥 Fetching {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo")
            
            if df.empty:
                print(f"⚠️  No data found for {symbol}. Check the symbol name.")
                continue
                
            file_path = f"data/raw/{symbol}.csv"
            df.to_csv(file_path)
            print(f"✅ Saved {symbol} to {file_path}")
            
        except Exception as e:
            print(f"❌ Failed to download {symbol}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock data for multiple symbols")
    # 'nargs="+"' allows you to pass 1 or more space-separated arguments
    parser.add_argument("symbols", nargs="+", help="Stock tickers (e.g., AAPL TSLA NVDA)")
    
    args = parser.parse_args()
    fetch_bulk_data(args.symbols)