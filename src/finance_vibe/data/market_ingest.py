"""Market-context download: VIX, credit and sector funds for the home page's market facts.

These tickers are not part of the scanned universe. Files go to
``data/raw/market/`` (``config.MARKET_CONTEXT_PROFILE``), never
``data/raw/{daily,weekly}/``, so neither scanner sees them. They get the same
Yahoo download, OHLCV contract, in-progress-bar rule and bar-quality report
(``data/logs/market/data_quality_<date>.csv``) as the daily ingest.

The folder is not wiped: a ticker that fails to download keeps its old file,
and the quality report then flags it ``stale``.

    python -m finance_vibe.data.market_ingest
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd

from finance_vibe import config
from finance_vibe.data import daily_ingest, ingest, quality
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)


def ingest_market_context(now: datetime | None = None) -> tuple[int, int]:
    """Download every ``config.MARKET_CONTEXT_TICKERS`` symbol; returns (saved, rejected)."""
    cfg = config.MARKET_CONTEXT_PROFILE
    logs_dir = cfg["logs_dir"]
    os.makedirs(cfg["raw_dir"], exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    symbols = config.MARKET_CONTEXT_TICKERS
    logger.info(
        f"--- Market context: {len(symbols)} tickers, {cfg['period']} {cfg['interval']} ---"
    )
    try:
        batch = ingest._download_batch(list(symbols.values()), cfg["period"], cfg["interval"])
    except Exception as e:  # yfinance/network: anything can fail; keep the old files
        logger.error(f"Market-context download failed after retries: {e}")
        for name in symbols:
            ingest._log_ingest_error(logs_dir, name, f"batch_exception:{e}")
        return 0, len(symbols)

    saved = 0
    for name, yahoo in symbols.items():
        try:
            df = ingest._ticker_frame(batch, yahoo)
            if df is None or df.empty:
                logger.warning(f"{name}: no data found.")
                ingest._log_ingest_error(logs_dir, name, "empty_download")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            clean = config.validate_and_clean_ohlcv(df, require_volume=True)
            # Same rule as daily_ingest: today's bar only once its session has closed.
            if not clean.empty and not daily_ingest.daily_bar_is_complete(
                clean["Date"].iloc[-1], now=now
            ):
                clean = clean.iloc[:-1]
            if len(clean) < config.MIN_SAVE_ROWS:
                logger.warning(f"{name}: only {len(clean)} valid rows. Skipped.")
                ingest._log_ingest_error(
                    logs_dir, name, f"insufficient_rows:{len(clean)}<{config.MIN_SAVE_ROWS}"
                )
                continue
            clean.to_csv(config.get_raw_path(name, cfg), index=False)
            saved += 1
        except ValueError as e:
            logger.warning(f"{name}: validation: {e}")
            ingest._log_ingest_error(logs_dir, name, f"validation:{e}")
        except Exception as e:  # per-ticker: log and keep going
            logger.exception(f"{name}: error: {e}")
            ingest._log_ingest_error(logs_dir, name, f"exception:{e}")
    rejected = len(symbols) - saved
    logger.info(f"📊 Market context: {saved} saved, {rejected} rejected.")
    return saved, rejected


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m finance_vibe.data.market_ingest`` (takes no arguments).

    Always exits 0: failures are logged per ticker and show up as stale or
    missing in the quality report and the daily health check.
    """
    ingest_market_context()
    cfg = config.MARKET_CONTEXT_PROFILE
    quality.write_dir_report(
        cfg["raw_dir"],
        cfg["logs_dir"],
        label="market",
        expected_last=daily_ingest.last_complete_session(),
        no_volume=config.MARKET_NO_VOLUME,
    )
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
