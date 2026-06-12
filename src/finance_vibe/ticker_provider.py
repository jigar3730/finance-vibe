import os
import pandas as pd
from yahooquery import Screener

# --- 1. PACKAGE IMPORT ---
# We use the absolute import to match industry standards
try:
    from finance_vibe import config
except ImportError:
    import config  # Fallback for local testing

# Path to your manifest file (lives in the same folder as this script)
MANIFEST_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'ticker_manifest.csv')


def refresh_active_tickers():
    print("--- STEP 1: Discovering Tickers (Manifest + Active) ---")

    # 1. Load the Static Manifest
    static_tickers = []
    if os.path.exists(MANIFEST_PATH):
        try:
            manifest_df = pd.read_csv(MANIFEST_PATH)
            # Standardizing column name check
            col_name = 'Symbol' if 'Symbol' in manifest_df.columns else manifest_df.columns[
                0]
            static_tickers = manifest_df[col_name].dropna().unique().tolist()
            print(
                f"📦 Loaded {len(static_tickers)} static tickers from manifest.")
        except Exception as e:
            print(f"⚠️ Could not read manifest: {e}")
    else:
        print(f"⚠️ Manifest not found at {MANIFEST_PATH}")

    # 2. Add Static Tickers from config.py
    # This ensures SPY, QQQ, etc., are always present even if manifest is missing
    static_tickers = list(set(static_tickers + config.STATIC_TICKERS))

    # 3. Discover Active Tickers via Screener
    s = Screener()
    ids = ['most_actives', 'day_gainers']
    discovered_tickers = []

    try:
        data = s.get_screeners(ids, count=50)
        for screen_id in ids:
            if screen_id in data and 'quotes' in data[screen_id]:
                discovered_tickers.extend([q['symbol']
                                          for q in data[screen_id]['quotes']])

        # 4. Merge, Deduplicate, and Clean
        combined = [t.upper().strip()
                    for t in (static_tickers + discovered_tickers)]

        # Remove empty strings or weird symbols
        seen = set()
        final_list = []
        for x in combined:
            if x and x not in seen and "^" not in x and "." not in x:
                final_list.append(x)
                seen.add(x)

        # Limit to a manageable number for the free tier
        final_list = final_list[:150]

        # 5. Save using config path
        # We don't need os.makedirs here anymore because config.py handles it
        pd.Series(final_list, name='Ticker').to_csv(
            config.TICKER_LIST_PATH, index=False)

        print(
            f"✅ Success! Saved {len(final_list)} total tickers to {config.TICKER_LIST_PATH}")

    except Exception as e:
        print(f"❌ Error during ticker discovery: {e}")


if __name__ == "__main__":
    refresh_active_tickers()
