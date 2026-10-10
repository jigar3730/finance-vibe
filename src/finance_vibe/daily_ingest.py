"""Daily OHLCV ingest: the shared Yahoo download plus a daily-only bar guard.

``run_vibe.py --mode daily`` runs this instead of ``data_ingestor.py``. The
download itself is ``data_ingestor.ingest_market_data("daily")`` (unchanged);
afterwards every raw daily CSV is checked and today's bar is dropped while its
session is still trading. During market hours (and until ``DAY_FINAL_HOUR_ET``)
yfinance returns a bar for today built from partial trading, and every later
stage would otherwise score it as a finished candle.

Also holds the NYSE session calendar the daily health check uses
(``last_complete_session``).

    python src/finance_vibe/daily_ingest.py
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import ClassVar

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay

from finance_vibe import config, data_ingestor

MODE = "daily"
MARKET_TZ = data_ingestor.MARKET_TZ
# A day's bar is final once the 16:00 ET close has passed plus a buffer for
# Yahoo to publish the close (half-day sessions close earlier, so are covered).
DAY_FINAL_HOUR_ET = 17


def _market_now(now: datetime | None) -> datetime:
    now = now or datetime.now(MARKET_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TZ)
    return now.astimezone(MARKET_TZ)


def _bar_day(bar_date) -> date:
    day = pd.Timestamp(bar_date)
    if day.tzinfo is not None:
        day = day.tz_convert(MARKET_TZ).tz_localize(None)
    return day.date()


def daily_bar_is_complete(bar_date, now: datetime | None = None) -> bool:
    """True once the session dated ``bar_date`` has closed, in market time.

    Earlier days are always complete; today's bar only from
    ``DAY_FINAL_HOUR_ET`` onward.
    """
    now = _market_now(now)
    day, today = _bar_day(bar_date), now.date()
    return day < today or (day == today and now.hour >= DAY_FINAL_HOUR_ET)


def drop_incomplete_daily_bars(raw_dir: str, logs_dir: str, now: datetime | None = None) -> int:
    """Drop an in-progress final bar from every raw daily CSV in ``raw_dir``.

    A file left with fewer than ``config.MIN_SAVE_ROWS`` rows is deleted and
    logged to ``ingest_errors_<date>.csv``, matching the ingestor's rule.
    Returns the number of files trimmed.
    """
    trimmed = 0
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(raw_dir, name)
        df = pd.read_csv(path)
        if (
            df.empty
            or "Date" not in df.columns
            or daily_bar_is_complete(df["Date"].iloc[-1], now=now)
        ):
            continue
        df = df.iloc[:-1]
        trimmed += 1
        if len(df) < config.MIN_SAVE_ROWS:
            os.remove(path)
            ticker = name.split("_", 1)[0]
            data_ingestor._log_ingest_error(
                logs_dir, ticker, f"insufficient_rows:{len(df)}<{config.MIN_SAVE_ROWS}"
            )
            continue
        df.to_csv(path, index=False)
    return trimmed


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Full-day NYSE closures (early closes are still sessions)."""

    rules: ClassVar[list[Holiday]] = [
        # NYSE does not move a Saturday New Year's Day to the Friday before.
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_SESSION = CustomBusinessDay(calendar=NYSEHolidayCalendar())


def last_complete_session(now: datetime | None = None) -> date:
    """Date of the newest NYSE session whose daily bar is complete at ``now``."""
    now = _market_now(now)
    today = pd.Timestamp(now.date())
    if _SESSION.is_on_offset(today) and now.hour >= DAY_FINAL_HOUR_ET:
        return today.date()
    return (today - _SESSION).date()


def ingest_daily(now: datetime | None = None) -> None:
    data_ingestor.ingest_market_data(mode=MODE)
    cfg = config.get_mode_config(MODE)
    trimmed = drop_incomplete_daily_bars(cfg["raw_dir"], cfg["logs_dir"], now=now)
    if trimmed:
        print(
            f"⏳ Dropped today's in-progress bar from {trimmed} files "
            f"(daily bars are final after {DAY_FINAL_HOUR_ET}:00 ET)."
        )


if __name__ == "__main__":
    ingest_daily()
