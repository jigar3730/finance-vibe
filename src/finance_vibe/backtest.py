import pandas as pd
from pathlib import Path

def run_backtest(ticker_symbol):
    file_path = f"data/raw/{ticker_symbol}.csv"
    if not Path(file_path).exists():
        print(f"No data for {ticker_symbol}")
        return

    df = pd.read_csv(file_path, index_col='Date', parse_dates=True).ffill()
    
    # 1. Re-calculate indicators
    ma200 = df['Close'].rolling(window=200).mean()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))

    # 2. Backtest Variables
    position = 0  # 0 = cash, 1 = holding stock
    balance = 10000.0
    shares = 0

    for i in range(200, len(df)):  # Start after 200-MA is ready
        price = df['Close'].iloc[i]
        
        # BUY SIGNAL: Bullish Trend + MACD Cross Up + RSI not overbought
        if position == 0:
            if price > ma200.iloc[i] and macd.iloc[i] > signal_line.iloc[i] and rsi.iloc[i] < 65:
                shares = balance / price
                balance = 0
                position = 1
                # print(f"BUY {ticker_symbol} at ${price:.2f}")

        # SELL SIGNAL: MACD Cross Down OR RSI Overbought
        elif position == 1:
            if macd.iloc[i] < signal_line.iloc[i] or rsi.iloc[i] > 75:
                balance = shares * price
                shares = 0
                position = 0
                # print(f"SELL {ticker_symbol} at ${price:.2f}")

    final_val = balance if position == 0 else shares * df['Close'].iloc[-1]
    profit_pct = ((final_val - 10000) / 10000) * 100
    print(f"💰 {ticker_symbol} Result: {profit_pct:.2f}% | Final: ${final_val:.2f}")

if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "NVDA", "SPY","PLTR","HOOD"]
    print("--- 2-Year Backtest Results (Start: $10,000) ---")
    for t in tickers:
        run_backtest(t)