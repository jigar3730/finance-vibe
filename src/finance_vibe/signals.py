import pandas as pd
import glob
from pathlib import Path

def generate_signals():
    files = glob.glob("data/raw/*.csv")
    print(f"{'TICKER':<10} | {'STATUS':<10} | {'PRICE vs 200MA'}")
    print("-" * 40)

    for file in files:
        ticker = Path(file).stem
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        
        # Cleaning & MA Calculation
        df = df.ffill()
        ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]

        # Signal Logic
        if current_price > ma200 and ma50 > ma200:
            status = "🚀 BULLISH"
        elif current_price < ma200 and ma50 < ma200:
            status = "💀 BEARISH"
        else:
            status = "⚖️ NEUTRAL"

        diff_pct = ((current_price - ma200) / ma200) * 100
        print(f"{ticker:<10} | {status:<10} | {diff_pct:>8.2f}%")

if __name__ == "__main__":
    generate_signals()