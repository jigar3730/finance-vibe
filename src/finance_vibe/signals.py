import pandas as pd
import glob
from pathlib import Path

def calculate_indicators(df):
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # CCI (20)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: (x - x.mean()).abs().mean())
    df['CCI'] = (tp - ma) / (0.015 * mad)
    
    return df

def generate_signals():
    files = glob.glob("data/raw/*.csv")
    # Added RSI to the header
    print(f"{'TICKER':<8} | {'TREND':<8} | {'MACD':<6} | {'RSI':<4} | {'CCI':<6} | {'ACTION'}")
    print("-" * 65)

    for file in files:
        ticker = Path(file).stem
        df = pd.read_csv(file, index_col='Date', parse_dates=True).ffill()
        df = calculate_indicators(df)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. Trend
        ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
        is_bullish = last['Close'] > ma200

        # 2. Logic Filters
        macd_up = last['MACD'] > last['Signal_Line']
        macd_cross_up = prev['MACD'] < prev['Signal_Line'] and macd_up
        rsi_value = last['RSI']
        cci_value = last['CCI']

        # 3. Final Multi-Factor Action
        # We only call it a STRONG BUY if Trend, MACD, and RSI (not overbought) align
        if is_bullish and macd_cross_up and rsi_value < 65:
            action = "🔥 STR. BUY"
        elif is_bullish and cci_value < -100:
            action = "💎 DIP BUY"
        elif rsi_value > 75:
            action = "⚠️ OVEREXT."
        elif not is_bullish and not macd_up:
            action = "💀 BEARISH"
        else:
            action = "HOLD"

        print(f"{ticker:<8} | {'BULL' if is_bullish else 'BEAR':<8} | {'UP' if macd_up else 'DWN':<6} | {int(rsi_value):<4} | {int(cci_value):>6} | {action}")

if __name__ == "__main__":
    generate_signals()