import os
import pandas as pd
from yahooquery import Screener

# --- 1. PACKAGE IMPORT ---
try:
    from finance_vibe import config
except ImportError:
    import config  # Fallback for local testing

MANIFEST_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'ticker_manifest.csv')


def refresh_active_tickers():
    print("--- STEP 1: Discovering Tickers (Manifest + Active) ---")

    # 1. Load the Static Manifest
    static_tickers = []
    if os.path.exists(MANIFEST_PATH):
        try:
            manifest_df = pd.read_csv(MANIFEST_PATH)
            col_name = 'Symbol' if 'Symbol' in manifest_df.columns else manifest_df.columns[0]
            static_tickers = manifest_df[col_name].dropna().unique().tolist()
            print(f"📦 Loaded {len(static_tickers)} static tickers from manifest.")
        except Exception as e:
            print(f"⚠️ Could not read manifest: {e}")
    else:
        print(f"⚠️ Manifest not found at {MANIFEST_PATH}")

    # 2. Add Static Tickers from config.py
    static_tickers = list(set(static_tickers + config.STATIC_TICKERS))
    
    # Standardize our priority baseline
    static_tickers = [t.upper().strip() for t in static_tickers if t and "^" not in t and "." not in t]

    # 3. Discover Active Tickers via Screener
    s = Screener()
    ids = ['most_actives', 'day_gainers']
    discovered_tickers = []

    try:
        data = s.get_screeners(ids, count=150)
        for screen_id in ids:
            if screen_id in data and 'quotes' in data[screen_id]:
                discovered_tickers.extend([q['symbol'] for q in data[screen_id]['quotes']])

        # Clean discovered tickers
        discovered_tickers = [t.upper().strip() for t in discovered_tickers if t and "^" not in t and "." not in t]

        # 4. CRITICAL FIX: Add discovered tickers AFTER manifest up to the 250 cap
        final_set = set(static_tickers)
        for ticker in discovered_tickers:
            if len(final_set) >= 250:
                break
            final_set.add(ticker)

        final_list = list(final_set)

        # 5. Save using config path
        pd.Series(final_list, name='Ticker').to_csv(
            config.TICKER_LIST_PATH, index=False)

        print(f"✅ Success! Saved {len(final_list)} total tickers to {config.TICKER_LIST_PATH}")

    except Exception as e:
        print(f"❌ Error during ticker discovery: {e}")


if __name__ == "__main__":
    refresh_active_tickers()