"""Scheduled releases on the market home (config.MARKET_EVENTS, insights.next_events)."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from finance_vibe import config
from finance_vibe.data import daily_ingest
from finance_vibe.web import insights

ET = daily_ingest.MARKET_TZ


def _et(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=ET)


def test_every_date_is_a_trading_day_in_order():
    for name, (_, days) in config.MARKET_EVENTS.items():
        assert list(days) == sorted(set(days)), name
        for day in days:
            assert daily_ingest.is_session_day(date.fromisoformat(day)), (name, day)


@pytest.mark.parametrize(("name", "weekday"), [("Jobs report", 4), ("FOMC", 2)])
def test_dates_fall_on_the_usual_weekday(name, weekday):
    # Catches a mistyped date: jobs reports here are Fridays, FOMC decisions Wednesdays.
    assert {date.fromisoformat(d).weekday() for d in config.MARKET_EVENTS[name][1]} == {weekday}


def test_next_events_soonest_first_in_calendar_days():
    got = insights.next_events(_et(2026, 11, 5, 13, 0))
    assert [(e["name"], e["when"]) for e in got["items"]] == [
        ("Jobs report", "tomorrow"),
        ("CPI", "in 5 days"),
        ("FOMC", "in 34 days"),
    ]
    assert got["items"][0]["detail"] == "Fri 2026-11-06 08:30 ET"
    assert got["items"][2]["detail"] == "Wed 2026-12-09 14:00 ET"
    assert got["missing"] == []


def test_a_release_leaves_the_list_at_its_time():
    before = insights.next_events(_et(2026, 11, 6, 8, 29))["items"][0]
    assert (before["name"], before["when"]) == ("Jobs report", "today")
    after = insights.next_events(_et(2026, 11, 6, 8, 30))["items"]
    jobs = next(e for e in after if e["name"] == "Jobs report")
    assert jobs["detail"].startswith("Fri 2026-12-04") and after[0]["name"] == "CPI"


def test_an_fomc_decision_stays_until_two_pm():
    def fomc(now):
        return next(e for e in insights.next_events(now)["items"] if e["name"] == "FOMC")

    assert fomc(_et(2026, 12, 9, 13, 59))["when"] == "today"
    assert fomc(_et(2026, 12, 9, 14, 0))["detail"].startswith("Wed 2027-01-27")


def test_kinds_that_ran_out_are_named():
    got = insights.next_events(_et(2027, 12, 9, 12, 0))
    assert got["missing"] == ["Jobs report", "FOMC"]
    assert [e["name"] for e in got["items"]] == ["CPI"]
    assert insights.next_events(_et(2028, 1, 1, 0, 0))["items"] == []
