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
    mad = tp.rolling(window=20).apply(lambda x: pd.Series(x).mad() if hasattr(pd.Series(x), 'mad') else (x - x.mean()).abs().mean())
    df['CCI'] = (tp - ma) / (0.015 * mad)
    
    return df

def generate_signals():
    files = glob.glob("data/raw/*.csv")
    print(f"{'TICKER':<8} | {'TREND':<8} | {'MACD':<8} | {'CCI':<8} | {'ACTION'}")
    print("-" * 60)

    for file in files:
        ticker = Path(file).stem
        df = pd.read_csv(file, index_col='Date', parse_dates=True).ffill()
        df = calculate_indicators(df)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. Trend (Price vs 200MA)
        ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
        is_bullish = last['Close'] > ma200

        # 2. MACD Signal (Crossover)
        macd_cross_up = prev['MACD'] < prev['Signal_Line'] and last['MACD'] > last['Signal_Line']
        
        # 3. CCI Signal
        cci_oversold = last['CCI'] < -100
        
        # Final Logic
        if is_bullish and macd_cross_up:
            action = "🔥 STRONG BUY"
        elif is_bullish and cci_oversold:
            action = "💎 DIP BUY"
        elif not is_bullish and last['MACD'] < last['Signal_Line']:
            action = "⚠️ AVOID"
        else:
            action = "HOLD"

        print(f"{ticker:<8} | {'BULL' if is_bullish else 'BEAR':<8} | {'UP' if last['MACD'] > last['Signal_Line'] else 'DOWN':<8} | {int(last['CCI']):>8} | {action}")

if __name__ == "__main__":
    generate_signals()