"""Weekly candle completeness: Monday-dated bars are final after Friday's close (ET)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from finance_vibe import data_ingestor as di

ET = di.MARKET_TZ
MONDAY = pd.Timestamp("2026-09-14")  # week of Mon 09-14 .. Fri 09-18


@pytest.mark.parametrize(
    "now, complete",
    [
        (datetime(2026, 9, 16, 12, 0, tzinfo=ET), False),  # Wednesday: week in progress
        (datetime(2026, 9, 18, 15, 59, tzinfo=ET), False),  # Friday before the close
        (datetime(2026, 9, 18, 16, 59, tzinfo=ET), False),  # inside the publish buffer
        (datetime(2026, 9, 18, 17, 0, tzinfo=ET), True),
        (datetime(2026, 9, 18, 18, 0, tzinfo=ET), True),  # the scheduled Friday 6 PM run
        (datetime(2026, 9, 19, 11, 16, tzinfo=ET), True),  # the Saturday run that lost this bar
        (datetime(2026, 9, 21, 9, 0, tzinfo=ET), True),  # following Monday
    ],
)
def test_monday_dated_week_completes_after_friday_close(now, complete):
    assert di.weekly_bar_is_complete(MONDAY, now=now) is complete


def test_utc_clock_is_converted_to_market_time():
    # 21:30 UTC on Friday is 17:30 EDT -> complete; 20:30 UTC is 16:30 EDT -> not yet.
    assert di.weekly_bar_is_complete(
        MONDAY, now=pd.Timestamp("2026-09-18 21:30", tz="UTC").to_pydatetime()
    )
    assert not di.weekly_bar_is_complete(
        MONDAY, now=pd.Timestamp("2026-09-18 20:30", tz="UTC").to_pydatetime()
    )


def test_tz_aware_bar_index_and_midweek_dates():
    bar = pd.Timestamp("2026-09-14 00:00", tz="America/New_York")
    assert di.weekly_bar_is_complete(bar, now=datetime(2026, 9, 18, 18, 0, tzinfo=ET))
    # any date inside the week maps to the same Friday
    assert not di.weekly_bar_is_complete(
        pd.Timestamp("2026-09-16"), now=datetime(2026, 9, 18, 9, 0, tzinfo=ET)
    )


def test_previous_week_is_always_complete():
    assert di.weekly_bar_is_complete(
        pd.Timestamp("2026-09-07"), now=datetime(2026, 9, 14, 9, 0, tzinfo=ET)
    )
