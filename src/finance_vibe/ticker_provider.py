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


def _normalize_symbols(symbols) -> list[str]:
    """Uppercase, strip, drop blank / index / dotted symbols."""
    return [
        t.upper().strip()
        for t in symbols
        if t and "^" not in str(t) and "." not in str(t)
    ]


def refresh_active_tickers():
    print("--- STEP 1: Discovering Tickers (Manifest + Active) ---")
    cap = int(getattr(config, "ACTIVE_TICKER_CAP", 1000))
    screener_ids = list(getattr(config, "SCREENER_IDS", ["most_actives", "day_gainers"]))
    screener_count = int(getattr(config, "SCREENER_COUNT", 250))

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

    # 2. Add Static Tickers from config.py (priority baseline)
    static_tickers = _normalize_symbols(set(static_tickers + config.STATIC_TICKERS))
    print(f"📌 Priority baseline (manifest + STATIC_TICKERS): {len(static_tickers)}")

    # 3. Discover Active Tickers via Screener
    discovered_tickers: list[str] = []
    try:
        s = Screener()
        data = s.get_screeners(screener_ids, count=screener_count)
        for screen_id in screener_ids:
            quotes = data.get(screen_id, {}).get("quotes") if isinstance(data, dict) else None
            if not quotes:
                print(f"⚠️ Screener '{screen_id}' returned no quotes")
                continue
            batch = _normalize_symbols(q["symbol"] for q in quotes if "symbol" in q)
            discovered_tickers.extend(batch)
            print(f"🔎 Screener '{screen_id}': {len(batch)} symbols")

        # Preserve discovery order while deduping
        seen = set()
        unique_discovered = []
        for t in discovered_tickers:
            if t not in seen:
                seen.add(t)
                unique_discovered.append(t)

        # 4. Manifest/static first, then fill with screener names up to ACTIVE_TICKER_CAP
        final_list = list(dict.fromkeys(static_tickers))  # stable unique
        final_set = set(final_list)
        for ticker in unique_discovered:
            if len(final_list) >= cap:
                break
            if ticker not in final_set:
                final_list.append(ticker)
                final_set.add(ticker)

        # If static alone already exceeds cap, trim (manifest order preserved via dict.fromkeys)
        if len(final_list) > cap:
            final_list = final_list[:cap]

        # 5. Save using config path
        pd.Series(final_list, name='Ticker').to_csv(
            config.TICKER_LIST_PATH, index=False)

        print(
            f"✅ Success! Saved {len(final_list)} total tickers "
            f"(cap={cap}) to {config.TICKER_LIST_PATH}"
        )

    except Exception as e:
        print(f"❌ Error during ticker discovery: {e}")
        # Still persist the priority baseline so the pipeline is not blocked.
        if static_tickers:
            fallback = static_tickers[:cap]
            pd.Series(fallback, name='Ticker').to_csv(
                config.TICKER_LIST_PATH, index=False)
            print(
                f"⚠️ Fell back to {len(fallback)} static tickers "
                f"at {config.TICKER_LIST_PATH}"
            )


if __name__ == "__main__":
    refresh_active_tickers()
