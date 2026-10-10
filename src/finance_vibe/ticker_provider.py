import logging
import os

import pandas as pd
from yahooquery import Screener

# --- 1. PACKAGE IMPORT ---
from finance_vibe import config
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_manifest.csv")


def _normalize_symbols(symbols) -> list[str]:
    """Uppercase, strip, drop blank / index / dotted symbols."""
    return [t.upper().strip() for t in symbols if t and "^" not in str(t) and "." not in str(t)]


def refresh_active_tickers():
    logger.info("--- STEP 1: Discovering Tickers (Manifest + Active) ---")
    cap = int(getattr(config, "ACTIVE_TICKER_CAP", 1000))
    screener_ids = list(getattr(config, "SCREENER_IDS", ["most_actives", "day_gainers"]))
    screener_count = int(getattr(config, "SCREENER_COUNT", 250))

    # 1. Load the Static Manifest
    static_tickers = []
    if os.path.exists(MANIFEST_PATH):
        try:
            manifest_df = pd.read_csv(MANIFEST_PATH)
            col_name = "Symbol" if "Symbol" in manifest_df.columns else manifest_df.columns[0]
            static_tickers = manifest_df[col_name].dropna().unique().tolist()
            logger.info(f"📦 Loaded {len(static_tickers)} static tickers from manifest.")
        except Exception as e:
            logger.warning(f"Could not read manifest: {e}")
    else:
        logger.warning(f"Manifest not found at {MANIFEST_PATH}")

    # 2. Add Static Tickers from config.py (priority baseline)
    static_tickers = _normalize_symbols(set(static_tickers + config.STATIC_TICKERS))
    logger.info(f"📌 Priority baseline (manifest + STATIC_TICKERS): {len(static_tickers)}")

    # 3. Discover Active Tickers via Screener
    discovered_tickers: list[str] = []
    try:
        s = Screener()
        data = s.get_screeners(screener_ids, count=screener_count)
        for screen_id in screener_ids:
            quotes = data.get(screen_id, {}).get("quotes") if isinstance(data, dict) else None
            if not quotes:
                logger.warning(f"Screener '{screen_id}' returned no quotes")
                continue
            batch = _normalize_symbols(q["symbol"] for q in quotes if "symbol" in q)
            discovered_tickers.extend(batch)
            logger.info(f"🔎 Screener '{screen_id}': {len(batch)} symbols")

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
        pd.Series(final_list, name="Ticker").to_csv(config.TICKER_LIST_PATH, index=False)

        logger.info(
            f"✅ Success! Saved {len(final_list)} total tickers "
            f"(cap={cap}) to {config.TICKER_LIST_PATH}"
        )

    except Exception as e:
        logger.error(f"Error during ticker discovery: {e}")
        # Still persist the priority baseline so the pipeline is not blocked.
        if static_tickers:
            fallback = static_tickers[:cap]
            pd.Series(fallback, name="Ticker").to_csv(config.TICKER_LIST_PATH, index=False)
            logger.warning(
                f"Fell back to {len(fallback)} static tickers at {config.TICKER_LIST_PATH}"
            )


if __name__ == "__main__":
    setup_logging()
    refresh_active_tickers()
