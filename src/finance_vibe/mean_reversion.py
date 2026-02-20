import pandas as pd
import glob
from pathlib import Path

def calculate_indicators(df):
    # 1. Long-Term Mean (200-day)
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Dist_200'] = ((df['Close'] - df['MA200']) / df['MA200']) * 100
    df['Dist_Std'] = df['Dist_200'].rolling(window=50).std()

    # 2. RSI (14-period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # 3. Bollinger Bands (20-day)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_BB'] = df['MA20'] - (df['STD20'] * 2)
    
    return df

def analyze_mean_reversion():
    files = glob.glob("data/raw/*.csv")
    header = f"{'TICKER':<7} | {'TREND':<5} | {'RSI':<3} | {'DIST %':<7} | {'BB ACTION':<11} | {'MR ACTION':<11} | {'COMBINED'}"
    print(header)
    print("-" * len(header))

    for file in files:
        ticker = Path(file).stem
        df = pd.read_csv(file, index_col='Date', parse_dates=True).ffill()
        if len(df) < 200: continue

        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        # Data Extraction
        price, rsi, dist = last['Close'], last['RSI'], last['Dist_200']
        trend = "BULL" if price > last['MA200'] else "BEAR"
        
        # 1. BB ACTION Logic
        if price >= last['Upper_BB']: bb_action = "OVER"
        elif price <= last['Lower_BB']: bb_action = "UNDER"
        else: bb_action = "INSIDE"

        # 2. MR ACTION Logic (Dynamic Thresholds)
        thresh_low = -(last['Dist_Std'] * 2)
        thresh_high = (last['Dist_Std'] * 2)
        
        if dist < thresh_low: mr_action = "STRETCH LOW"
        elif dist > thresh_high: mr_action = "STRETCH HI"
        else: mr_action = "NEUTRAL"

        # 3. TRIPLE-CHECK COMBINED LOGIC
        # Strong Buy: Under BB + Stretched Low + RSI < 35
        if bb_action == "UNDER" and mr_action == "STRETCH LOW" and rsi < 35:
            combined = "🚀 STR. BUY"
        # Strong Sell: Over BB + Stretched High + RSI > 65
        elif bb_action == "OVER" and mr_action == "STRETCH HI" and rsi > 65:
            combined = "💀 STR. SELL"
        # Watch: If at least TWO are triggered
        elif (bb_action != "INSIDE" and mr_action != "NEUTRAL") or (rsi < 30 or rsi > 70):
            combined = "🔍 WATCH"
        else:
            combined = "WAIT"

        print(f"{ticker:<7} | {trend:<5} | {int(rsi):<3} | {dist:>6.1f}% | {bb_action:<11} | {mr_action:<11} | {combined}")

if __name__ == "__main__":
    analyze_mean_reversion()