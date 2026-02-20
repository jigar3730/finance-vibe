import yfinance as yf
import argparse
from pathlib import Path

def fetch_bulk_data(tickers, period="2y"): # Default changed to 2y
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    for symbol in tickers:
        symbol = symbol.upper()
        print(f"📥 Fetching {symbol} for period: {period}...")
        
        try:
            ticker = yf.Ticker(symbol)
            # Use the period variable here
            df = ticker.history(period=period)
            
            if df.empty:
                print(f"⚠️  No data found for {symbol}.")
                continue
                
            file_path = f"data/raw/{symbol}.csv"
            df.to_csv(file_path)
            print(f"✅ Saved {symbol} ({len(df)} rows) to {file_path}")
            
        except Exception as e:
            print(f"❌ Failed to download {symbol}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="+")
    # Added a flag to choose the timeframe (defaulting to 2y)
    parser.add_argument("--period", default="2y", help="Time period (1mo, 1y, 2y, 5y, max)")
    
    args = parser.parse_args()
    fetch_bulk_data(args.symbols, args.period)