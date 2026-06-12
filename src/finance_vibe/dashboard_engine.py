import yfinance as yf
import pandas as pd
import pandas_ta as ta

def calculate_dashboard_metrics(ticker):
    # 1. Download 5 Years of Weekly Data
    df = yf.download(ticker, period="5y", interval="1wk", progress=False)
    
    if df.empty or len(df) < 50:
        return None

    # Clean the multi-index columns if necessary (common in new yf versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Primary Indicators
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    
    macd = ta.macd(df['Close'], fast=15, slow=30, signal=9)
    df['MACD'] = macd.iloc[:, 0]

    # 3. Secondary Smoothing (EMA 20 of the resulting indicators)
    df['EMA_20_RSI'] = ta.ema(df['RSI'], length=20)
    df['EMA_20_CCI'] = ta.ema(df['CCI'], length=20)
    df['EMA_20_MACD'] = ta.ema(df['MACD'], length=20)

    # ==========================================
    # NEW: Swing Highs, Swing Lows & Entry/Exits
    # ==========================================
    # Donchian Channels (4-week rolling window captures recent cyclical pivots)
    donchian = ta.donchian(df['High'], df['Low'], lower_length=4, upper_length=4)
    df['Swing_High_4wk'] = donchian.iloc[:, 2] # Upper band (Highest high of last 4 weeks)
    df['Swing_Low_4wk'] = donchian.iloc[:, 0]  # Lower band (Lowest low of last 4 weeks)

    # Technical Alignment Signal Conditions
    df['Trend_Bullish'] = (df['Close'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_50'])
    df['RSI_Bullish'] = df['RSI'] > df['EMA_20_RSI']
    df['CCI_Bullish'] = df['CCI'] > df['EMA_20_CCI']
    df['MACD_Bullish'] = df['MACD'] > df['EMA_20_MACD']

    # Entry Point: All momentum indicators flip positive, and price breaks above last week's High
    df['Entry_Signal'] = (
        df['Trend_Bullish'] & 
        df['RSI_Bullish'] & 
        df['CCI_Bullish'] & 
        df['MACD_Bullish'] & 
        (df['Close'] > df['High'].shift(1))
    )

    # Exit Point: Price closes below the 4-week Swing Low OR the trend structure completely breaks down
    df['Exit_Signal'] = (df['Close'] < df['Swing_Low_4wk']) | (~df['Trend_Bullish'])

    # 4. Algorithmic Vibe Score Allocation (-10 to +10)
    df['Vibe_Score'] = (
        df['Trend_Bullish'].map({True: 4, False: -4}) +
        df['RSI_Bullish'].map({True: 2, False: -2}) +
        df['CCI_Bullish'].map({True: 2, False: -2}) +
        df['MACD_Bullish'].map({True: 2, False: -2})
    )

    # Filter for the last completed week
    return df.tail(2)

if __name__ == "__main__":
    test_data = calculate_dashboard_metrics("NVDA")
    if test_data is not None:
        print("Latest Weekly Structure & Entry/Exit Triggers for NVDA:")
        print(test_data[[
            'Close', 'Swing_High_4wk', 'Swing_Low_4wk', 
            'Vibe_Score', 'Entry_Signal', 'Exit_Signal'
        ]].tail(1))