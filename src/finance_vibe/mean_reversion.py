import pandas as pd
import glob
from pathlib import Path

def analyze_mean_reversion():
    files = glob.glob("data/raw/*.csv")
    print(f"{'TICKER':<8} | {'DIST 200MA':<10} | {'BB STATE':<12} | {'ACTION'}")
    print("-" * 55)

    for file in files:
        ticker = Path(file).stem
        df = pd.read_csv(file, index_col='Date', parse_dates=True).ffill()
        
        # 1. Long-Term Mean (200-MA)
        ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]
        dist_200 = ((current_price - ma200) / ma200) * 100

        # 2. Short-Term Volatility (Bollinger Bands 20-day)
        ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        std20 = df['Close'].rolling(window=20).std().iloc[-1]
        upper_bb = ma20 + (std20 * 2)
        lower_bb = ma20 - (std20 * 2)

        # 3. Determine States
        if current_price >= upper_bb:
            bb_state = "🔥 OVER UPPER"
        elif current_price <= lower_bb:
            bb_state = "🧊 BELOW LOW"
        else:
            bb_state = "⚖️ INSIDE"

        # 4. Combined Action Logic
        if dist_200 < -15 and current_price <= lower_bb:
            action = "🚀 REV. BUY" # Extremely stretched downward
        elif dist_200 > 20 and current_price >= upper_bb:
            action = "⚠️ REV. SELL" # Extremely stretched upward
        else:
            action = "WAIT"

        print(f"{ticker:<8} | {dist_200:>9.1f}% | {bb_state:<12} | {action}")

if __name__ == "__main__":
    analyze_mean_reversion()