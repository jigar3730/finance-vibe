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
    # CCI (20), RSI (14), SMA (20), SMA (50)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    
    # MACD (15, 30, 9) - Returns a DataFrame, we want the main MACD line
    macd = ta.macd(df['Close'], fast=15, slow=30, signal=9)
    df['MACD'] = macd.iloc[:, 0] # MACD_15_30_9 column

    # 3. Secondary Smoothing (EMA 20 of the resulting indicators)
    # This is your "Signal Line" for the oscillators
    df['EMA_20_RSI'] = ta.ema(df['RSI'], length=20)
    df['EMA_20_CCI'] = ta.ema(df['CCI'], length=20)
    df['EMA_20_MACD'] = ta.ema(df['MACD'], length=20)

    # 4. Filter for the last completed week
    # We take the second to last row if today is mid-week to avoid partial data
    return df.tail(2)

if __name__ == "__main__":
    # Quick test for a single ticker
    test_data = calculate_dashboard_metrics("NVDA")
    if test_data is not None:
        print("Latest Weekly Metrics for NVDA:")
        print(test_data[['Close', 'RSI', 'EMA_20_RSI', 'SMA_20']].tail(1))