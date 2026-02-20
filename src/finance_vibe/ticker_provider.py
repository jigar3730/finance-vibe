from yahooquery import Screener
import pandas as pd
import os
from config import STATIC_TICKERS, TICKER_LIST_PATH

def refresh_active_tickers():
    print("--- STEP 1: Discovering Tickers (Static + Active) ---")
    s = Screener()
    
    ids = ['most_actives', 'day_gainers']
    discovered_tickers = []
    
    try:
        data = s.get_screeners(ids, count=50)
        for screen_id in ids:
            if screen_id in data and 'quotes' in data[screen_id]:
                discovered_tickers.extend([q['symbol'] for q in data[screen_id]['quotes']])
        
        # Merge Static + Discovered
        # Using a list to maintain order: Static ETFs first, then top active stocks
        combined = STATIC_TICKERS + discovered_tickers
        
        # Deduplicate while preserving order
        seen = set()
        final_list = [x for x in combined if not (x in seen or seen.add(x))][:30] # Top 30 total
        
        os.makedirs('data', exist_ok=True)
        pd.Series(final_list, name='Ticker').to_csv(TICKER_LIST_PATH, index=False)
        
        print(f"✅ Success! Saved {len(final_list)} tickers to {TICKER_LIST_PATH}")
        print(f"Indices included: {STATIC_TICKERS}")
            
    except Exception as e:
        print(f"❌ Error during ticker discovery: {e}")

if __name__ == "__main__":
    refresh_active_tickers()