"""Live quotes for the market home page, taken every few minutes during the session.

One Yahoo batch of 5-minute bars for ``config.MARKET_SNAPSHOT_TICKERS``, reduced to the
newest price per symbol and written atomically to ``config.MARKET_SNAPSHOT_PATH``. Nothing
here touches ``data/raw/``, so scanners, ``--as-of`` replays and golden outputs never see
a quote. A failed fetch keeps the previous file, and a symbol missing from a partial fetch
keeps its previous quote with its own fetch time; the page shows how old either one is.

    python -m finance_vibe.data.market_snapshot          # no-op unless a slot is due
    python -m finance_vibe.data.market_snapshot --now    # fetch regardless of the clock
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd

from finance_vibe import config
from finance_vibe.data import daily_ingest, ingest
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)

MARKET_TZ = ingest.MARKET_TZ
PRE_MARKET_START = time(8, 0)
REGULAR_OPEN = time(9, 30)
# VIX and other indexes have no bars before the open and run about 15 minutes behind.
QUOTES_EXPECTED_FROM = time(9, 50)
LAST_SLOT = time(16, 15)
PRE_MARKET_EVERY_MINUTES = 15
CRON_EVERY_MINUTES = 5


def now_et() -> datetime:
    return datetime.now(MARKET_TZ)


def in_window(now: datetime) -> bool:
    """True on an NYSE session day between the first and last snapshot slots."""
    now = now.astimezone(MARKET_TZ)
    last_minute = (datetime.combine(now.date(), LAST_SLOT) + timedelta(minutes=1)).time()
    return daily_ingest.is_session_day(now.date()) and PRE_MARKET_START <= now.time() < last_minute


def is_due(now: datetime) -> bool:
    """True when a cron firing every 5 minutes should fetch: all of them in regular
    hours, the first of each 15 minutes before the open."""
    now = now.astimezone(MARKET_TZ)
    if not in_window(now):
        return False
    if now.time() >= REGULAR_OPEN:
        return True
    return now.minute % PRE_MARKET_EVERY_MINUTES < CRON_EVERY_MINUTES


def latest_quote(frame: pd.DataFrame, day: date) -> dict[str, Any] | None:
    """Newest usable price from ``frame``'s bars on ``day``, ignoring after-hours bars.

    Returns None when the symbol has no bar that day (an index before the open, say)
    or its newest price is not a positive number.
    """
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    index = pd.DatetimeIndex(close.index)
    index = index.tz_localize(MARKET_TZ) if index.tz is None else index.tz_convert(MARKET_TZ)
    keep = (index.date == day) & (index.time < daily_ingest.session_close(day))
    if not keep.any():
        return None
    last = float(close[keep].iloc[-1])
    if last <= 0:
        return None
    return {"last": last, "time": index[keep][-1].isoformat()}


def _write_atomic(payload: dict[str, Any], path: str) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.chmod(tmp, 0o644)  # mkstemp is 0600; the file should read like the raw CSVs
        os.replace(tmp, path)
    except OSError:
        os.unlink(tmp)
        raise


def take_snapshot(now: datetime | None = None) -> bool:
    """Fetch quotes and replace the snapshot file; returns False if the file was kept."""
    now = (now or now_et()).astimezone(MARKET_TZ)
    symbols = config.MARKET_SNAPSHOT_TICKERS
    try:
        batch = ingest._download_batch(
            list(symbols.values()),
            config.MARKET_SNAPSHOT_PERIOD,
            config.MARKET_SNAPSHOT_INTERVAL,
            prepost=True,
        )
    except Exception as e:  # yfinance/network: anything can fail; keep the old file
        logger.error(f"Market snapshot download failed, keeping the previous file: {e}")
        return False

    quotes: dict[str, dict[str, Any]] = {}
    for name, yahoo in symbols.items():
        try:
            frame = ingest._ticker_frame(batch, yahoo)
            quote = None if frame is None else latest_quote(frame, now.date())
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"{name}: unusable snapshot data: {e}")
            continue
        if quote is not None:
            quotes[name] = quote | {"fetched_at": now.isoformat()}
    if not quotes:
        logger.warning("Market snapshot: no quotes returned, keeping the previous file.")
        return False
    got = len(quotes)
    quotes = _carried_quotes(symbols, quotes, now.date()) | quotes
    _write_atomic({"fetched_at": now.isoformat(), "quotes": quotes}, config.MARKET_SNAPSHOT_PATH)
    logger.info(f"Market snapshot: {got} of {len(symbols)} quotes, {len(quotes) - got} carried.")
    return True


def _carried_quotes(
    symbols: dict[str, str], fresh: dict[str, dict[str, Any]], day: date
) -> dict[str, dict[str, Any]]:
    """Same-day quotes from the previous file for symbols this fetch did not return."""
    previous = load_snapshot()
    if previous is None:
        return {}
    return {
        name: {
            "last": quote["last"],
            "time": quote["time"].isoformat(),
            "fetched_at": quote["fetched_at"].isoformat(),
        }
        for name, quote in previous["quotes"].items()
        if name in symbols and name not in fresh and quote["date"] == day
    }


def load_snapshot() -> dict[str, Any] | None:
    """The snapshot as ``{"fetched_at", "quotes": {symbol: {"last", "date", "time",
    "fetched_at"}}}``, or None when the file is missing or unreadable. A malformed quote
    is left out."""
    try:
        with open(config.MARKET_SNAPSHOT_PATH) as handle:
            payload = json.load(handle)
        fetched = datetime.fromisoformat(payload["fetched_at"]).astimezone(MARKET_TZ)
        items = payload["quotes"].items()
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("Market snapshot unreadable: %s", exc)
        return None
    quotes = {}
    for symbol, quote in items:
        try:
            last = float(quote["last"])
            when = datetime.fromisoformat(quote["time"]).astimezone(MARKET_TZ)
            got_at = datetime.fromisoformat(quote["fetched_at"]).astimezone(MARKET_TZ)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Market snapshot quote for %s unreadable: %s", symbol, exc)
            continue
        if last > 0:
            quotes[symbol] = {"last": last, "date": when.date(), "time": when, "fetched_at": got_at}
    return {"fetched_at": fetched, "quotes": quotes}


def _old_minutes(fetched_at: datetime, now: datetime) -> int | None:
    """Minutes since ``fetched_at`` once it is overdue, else None.

    On the day of the fetch, lateness counts only up to that day's last slot, so the final
    quotes of a session are not called old while the page waits for the evening's closes.
    A fetch from an earlier day is measured to ``now``.
    """
    now = now.astimezone(MARKET_TZ)
    fetched_at = fetched_at.astimezone(MARKET_TZ)
    due = now
    if fetched_at.date() == now.date():
        due = min(now, datetime.combine(now.date(), LAST_SLOT, tzinfo=MARKET_TZ))
    overdue = due - fetched_at > timedelta(minutes=config.MARKET_SNAPSHOT_STALE_MINUTES)
    return int((now - fetched_at).total_seconds() // 60) if overdue else None


def status(fetched_at: datetime, now: datetime) -> dict[str, Any]:
    """Badge label for a snapshot, plus ``old_minutes`` once it is overdue, else None."""
    now = now.astimezone(MARKET_TZ)
    fetched_at = fetched_at.astimezone(MARKET_TZ)
    fmt = "%H:%M ET" if fetched_at.date() == now.date() else "%a %H:%M ET"
    return {"label": fetched_at.strftime(fmt), "old_minutes": _old_minutes(fetched_at, now)}


def quote_note(
    close_day: date | None,
    quote: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    now: datetime,
) -> str | None:
    """Why a symbol's number is not current while the snapshot is: "no quote", "quote N
    min old", or None.

    None when the symbol's close already covers the snapshot's day, when the whole snapshot
    is overdue (the page says so once), and for a missing quote before indexes can have one.
    """
    if snapshot is None or close_day is None:
        return None
    fetched = snapshot["fetched_at"]
    if close_day >= fetched.date() or _old_minutes(fetched, now) is not None:
        return None
    if quote is None:
        early = fetched.date() == now.date() and now.time() < QUOTES_EXPECTED_FROM
        return None if early else "no quote"
    minutes = _old_minutes(quote["fetched_at"], now)
    return None if minutes is None else f"quote {minutes} min old"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Always exits 0: a failure is logged and the page shows the age."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now", action="store_true", help="fetch regardless of the clock")
    args = parser.parse_args(argv)
    if args.now or is_due(now_et()):
        take_snapshot()
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
