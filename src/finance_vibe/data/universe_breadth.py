"""Breadth across the scanned tickers, cached once per daily run for the market home.

Counts how many of the daily raw files closed above their 50- and 200-day averages and
at a 252-session high or low, for the newest session and the one before. Both use today's
files, so the two can be compared. Stocks only: the universe funds (SPY, QQQ, ...) are
left out. The members are the S&P 500 / S&P 400 / watchlist stocks that pass the
universe rules and change once a quarter, so this is not index breadth.

Runs last in the daily pipeline. It reads raw files and writes only
``config.UNIVERSE_BREADTH_PATH``, so no scan output changes.

    python -m finance_vibe.data.universe_breadth
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

import pandas as pd

from finance_vibe import config, indicators
from finance_vibe.data import market_snapshot, raw
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)

HIGH_LOW_BARS = 252
AVERAGES = (50, 200)


def _above(close: pd.Series, bars: int) -> bool | None:
    if len(close) < bars:
        return None
    ma = indicators.sma(close.reset_index(drop=True), bars)
    return None if ma is None or pd.isna(ma.iloc[-1]) else bool(close.iloc[-1] > ma.iloc[-1])


def _counts(closes: list[pd.Series]) -> dict[str, int]:
    """Counts for the newest bar of each series; a series too short for a fact is left out."""
    out: dict[str, int] = {}
    for bars in AVERAGES:
        states = [s for s in (_above(c, bars) for c in closes) if s is not None]
        out[f"above_{bars}"] = sum(states)
        out[f"counted_{bars}"] = len(states)
    year = [c.tail(HIGH_LOW_BARS) for c in closes if len(c) >= HIGH_LOW_BARS]
    out["new_highs"] = sum(bool(c.iloc[-1] >= c.max()) for c in year)
    out["new_lows"] = sum(bool(c.iloc[-1] <= c.min()) for c in year)
    return out


def summarize(frames: list[pd.DataFrame]) -> dict[str, Any] | None:
    """Counts at the newest date in ``frames`` and the session before it.

    A ticker whose newest bar is older than that date is left out, so a delisted or
    failed name never counts as today's. None when ``frames`` holds no bars.
    """
    frames = [f for f in frames if not f.empty]
    if not frames:
        return None
    newest = max(pd.Timestamp(f["Date"].iloc[-1]) for f in frames)
    current = [f for f in frames if pd.Timestamp(f["Date"].iloc[-1]) == newest]
    earlier = [f["Date"].iloc[-2] for f in current if len(f) > 1]
    closes = [f["Close"] for f in current]
    return {
        "date": newest.date().isoformat(),
        "previous_date": pd.Timestamp(max(earlier)).date().isoformat() if earlier else None,
        "tickers": len(current),
        "current": _counts(closes),
        "previous": _counts([c.iloc[:-1] for c in closes if len(c) > 1]),
    }


def build() -> dict[str, Any] | None:
    """Summarize every daily raw stock file (``UNIVERSE_FUNDS`` are left out); an
    unreadable file is logged and skipped."""
    pattern = os.path.join(config.TIMEFRAME_PROFILES["daily"]["raw_dir"], "*.csv")
    frames = []
    for path in sorted(glob.glob(pattern)):
        if raw.ticker_from_filename(os.path.basename(path)) in config.UNIVERSE_FUNDS:
            continue
        try:
            frames.append(raw.load_raw(path)[["Date", "Close"]])
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Skipping %s: %s", os.path.basename(path), exc)
    return summarize(frames)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Always exits 0: without new counts the page flags the old ones."""
    summary = build()
    if summary is None:
        logger.warning("Universe breadth: no daily raw files, keeping the previous file.")
        return 0
    market_snapshot._write_atomic(summary, config.UNIVERSE_BREADTH_PATH)
    logger.info(f"Universe breadth: {summary['tickers']} tickers at {summary['date']}.")
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
