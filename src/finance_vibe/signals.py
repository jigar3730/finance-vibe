import pandas as pd
import glob
from pathlib import Path

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_signals():
    files = glob.glob("data/raw/*.csv")
    print(f"{'TICKER':<8} | {'TREND':<10} | {'MOMENTUM':<12} | {'ACTION'}")
    print("-" * 55)

    for file in files:
        ticker = Path(file).stem
        df = pd.read_csv(file, index_col='Date', parse_dates=True).ffill()
        
        # Calculations
        ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]
        rsi = calculate_rsi(df['Close']).iloc[-1]

        # Trend Logic
        trend = "BULLISH" if current_price > ma200 else "BEARISH"
        
        # Momentum & Action Logic
        if rsi > 70:
            momentum = "OVERBOUGHT"
            action = "⚠️ WAIT" if trend == "BULLISH" else "💀 SELL"
        elif rsi < 30:
            momentum = "OVERSOLD"
            action = "💎 BUY" if trend == "BULLISH" else "⚠️ WATCH"
        else:
            momentum = "NEUTRAL"
            action = "HOLD"

        print(f"{ticker:<8} | {trend:<10} | {momentum:<12} | {action}")

if __name__ == "__main__":
    generate_signals()