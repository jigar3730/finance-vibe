import pandas as pd
import pandas_ta as ta
import os
from datetime import datetime
from config import TICKER_LIST_PATH, get_raw_path, LOGS_DIR

def calculate_composite_vibe(df):
    # 1. Trend
    df['SMA20'] = ta.sma(df['Close'], length=20)
    df['SMA50'] = ta.sma(df['Close'], length=50)

    # 2. Momentum
    macd = ta.macd(df['Close'], fast=15, slow=30, signal=9)
    df['MACD_Hist'] = macd.iloc[:, 1] 
    df['MACD_Hist_Signal'] = ta.ema(df['MACD_Hist'], length=20)
    
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['RSI_Signal'] = ta.ema(df['RSI'], length=20)

    # 3. Volatility
    # More robust CCI version
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad_tp = tp.rolling(window=20).apply(lambda x: (x - x.mean()).abs().mean())
    df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp)
    df['CCI_Signal'] = ta.ema(df['CCI'], length=20)

    latest = df.iloc[-1]
    score = 0.0

    # Scoring Logic
    if latest['Close'] > latest['SMA20'] > latest['SMA50']:
        score += 4.0
    elif latest['Close'] > latest['SMA20']:
        score += 2.0
    elif latest['Close'] < latest['SMA20'] < latest['SMA50']:
        score -= 4.0

    macd_bull = latest['MACD_Hist'] > latest['MACD_Hist_Signal']
    rsi_bull = latest['RSI'] > latest['RSI_Signal']
    
    if macd_bull and rsi_bull: score += 3.0
    elif macd_bull or rsi_bull: score += 1.0
    else: score -= 3.0

    if latest['CCI'] > latest['CCI_Signal'] and latest['CCI'] > 0:
        score += 3.0
    elif latest['CCI'] < latest['CCI_Signal'] and latest['CCI'] < 0:
        score -= 3.0

    return score, latest # Return both for the report

def run_scanner():
    tickers = pd.read_csv(TICKER_LIST_PATH)['Ticker'].tolist()
    results = []

    for ticker in tickers:
        file_path = get_raw_path(ticker)
        if not os.path.exists(file_path): continue
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if len(df) < 50: continue

        # Capture both returned values
        score, latest = calculate_composite_vibe(df)

        if score >= 8.0:
            sentiment, action = "Strong Bullish", "🔥 GO ALL IN"
        elif score >= 4.0:
            sentiment, action = "Bullish", "✅ ACCUMULATE"
        elif score >= -3.9:
            sentiment, action = "Neutral", "⏳ WAIT / CASH"
        elif score >= -7.9:
            sentiment, action = "Bearish", "⚠️ DISTRIBUTE"
        else:
            sentiment, action = "Strong Bearish", "🚫 AVOID"

        results.append({
            "Ticker": ticker,
            "Price": round(latest['Close'], 2),
            "SMA20": round(latest['SMA20'], 2),
            "SMA50": round(latest['SMA50'], 2),
            "CCI": round(latest['CCI'], 2),
            "CCI_S": round(latest['CCI_Signal'], 2),
            "MACD_H": round(latest['MACD_Hist'], 2),
            "MACD_S": round(latest['MACD_Hist_Signal'], 2),
            "RSI": round(latest['RSI'], 2),
            "RSI_S": round(latest['RSI_Signal'], 2),
            "Score": score,
            "Sentiment": sentiment,
            "Action": action
        })

    summary_df = pd.DataFrame(results).sort_values(by='Score', ascending=False)
    
    # Archive Logic
    today_str = datetime.now().strftime("%Y-%m-%d")
    archive_path = os.path.join(LOGS_DIR, f"vibe_report_{today_str}.csv")
    summary_df.to_csv(archive_path, index=False)
    
    print(summary_df.to_markdown(index=False))
    print(f"\n📁 Archive created: {archive_path}")

if __name__ == "__main__":
    run_scanner()