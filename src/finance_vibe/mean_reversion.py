import pandas as pd
import glob
from pathlib import Path

def calculate_indicators(df):
    # 200-MA and Distance
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Dist_200'] = ((df['Close'] - df['MA200']) / df['MA200']) * 100
    df['Dist_Std'] = df['Dist_200'].rolling(window=50).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_BB'] = df['MA20'] - (df['STD20'] * 2)
    return df

def analyze_mean_reversion():
    files = glob.glob("data/raw/*.csv")
    # Updated Header to include everything
    header = f"{'TICKER':<7} | {'TRND':<4} | {'DIST%':<7} | {'RSI':<3} | {'BB':<6} | {'MR':<7} | {'SCR':<3} | {'ACTION'}"
    print(header)
    print("-" * len(header))

    results = []
    for file in files:
        ticker = Path(file).stem
        df = pd.read_csv(file, index_col='Date', parse_dates=True).ffill()
        if len(df) < 200: continue
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        score = 0
        dist, rsi, price = last['Dist_200'], last['RSI'], last['Close']
        trend = "BULL" if price > last['MA200'] else "BEAR"
        
        # 1. Scoring & BB State
        if price >= last['Upper_BB']: 
            bb_s, bb_pts = "OVER", 3
        elif price <= last['Lower_BB']: 
            bb_s, bb_pts = "UNDR", 3
        else: 
            bb_s, bb_pts = "IN", 0
        score += bb_pts

        # 2. Scoring & MR State
        t_low, t_high = -(last['Dist_Std'] * 2), (last['Dist_Std'] * 2)
        if dist < t_low: 
            mr_s, mr_pts = "LOW", 4
        elif dist > t_high: 
            mr_s, mr_pts = "HIGH", 4
        else: 
            mr_s, mr_pts = "NEUT", 0
        score += mr_pts

        # 3. RSI Scoring
        if rsi < 30 or rsi > 70: score += 3
        elif rsi < 40 or rsi > 60: score += 1

        # Action Label
        if score >= 7: action = "🔥 STRONG"
        elif score >= 4: action = "⚡ ACTION"
        elif score >= 2: action = "🔍 WATCH"
        else: action = "WAIT"

        results.append({
            'ticker': ticker, 'trend': trend, 'dist': dist, 
            'rsi': rsi, 'bb': bb_s, 'mr': mr_s, 
            'score': score, 'action': action
        })

    # Sort results by score
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"{r['ticker']:<7} | {r['trend']:<4} | {r['dist']:>6.1f}% | {int(r['rsi']):<3} | {r['bb']:<6} | {r['mr']:<7} | {r['score']:<3} | {r['action']}")

if __name__ == "__main__":
    analyze_mean_reversion()