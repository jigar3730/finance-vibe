import pandas as pd
import glob
from pathlib import Path

def generate_signals():
    files = glob.glob("data/raw/*.csv")
    # Added 'DIST %' to the header
    print(f"{'TICKER':<8} | {'TREND':<8} | {'RSI':<4} | {'DIST %':<8} | {'ACTION'}")
    print("-" * 55)

    for file in files:
        ticker = Path(file).stem
        df = pd.read_csv(file, index_col='Date', parse_dates=True).ffill()
        
        # Calculations
        ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

        # Mean Reversion Calculation: How far are we from the 200-MA?
        dist_from_mean = ((current_price - ma200) / ma200) * 100

        # Action Logic
        trend = "BULL" if current_price > ma200 else "BEAR"
        
        if dist_from_mean > 25:
            action = "⚠️ EXTENDED"
        elif dist_from_mean < -20 and rsi < 30:
            action = "💎 MEAN REV."
        else:
            action = "STABLE"

        print(f"{ticker:<8} | {trend:<8} | {int(rsi):<4} | {dist_from_mean:>7.1f}% | {action}")

if __name__ == "__main__":
    generate_signals()