"""Live market quotes (data.market_snapshot): slots, parsing, failure handling, overlay."""

from __future__ import annotations

import json
import os
from datetime import date, datetime

import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.data import ingest, market_snapshot
from finance_vibe.web import health, insights

ET = market_snapshot.MARKET_TZ
THU = date(2026, 11, 5)


def _et(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET)


def _bars(day: date, times: list[str], close: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([f"{day} {t}" for t in times]).tz_localize(ET)
    return pd.DataFrame({"Close": close}, index=index)


def _batch(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, axis=1, sort=True)


def _closes(values: list[float], last: str = "2026-11-04") -> pd.DataFrame:
    dates = pd.bdate_range(end=last, periods=len(values))
    return pd.DataFrame({"Date": dates, "Close": [float(v) for v in values]})


@pytest.mark.parametrize(
    ("when", "due"),
    [
        (_et(THU, 7, 55), False),
        (_et(THU, 8, 0), True),
        (_et(THU, 8, 5), False),  # pre-market: every 15 minutes only
        (_et(THU, 8, 15), True),
        (_et(THU, 9, 30), True),
        (_et(THU, 9, 35), True),  # regular hours: every 5 minutes
        (_et(THU, 16, 15), True),
        (_et(THU, 16, 20), False),
        (_et(date(2026, 11, 7), 10, 0), False),  # Saturday
        (_et(date(2026, 11, 26), 10, 0), False),  # Thanksgiving
    ],
)
def test_is_due(when, due):
    assert market_snapshot.is_due(when) is due


def test_latest_quote_takes_newest_regular_or_pre_market_bar_today():
    frame = _bars(THU, ["08:00", "10:20", "10:25"], [750.0, 748.0, 746.28])
    got = market_snapshot.latest_quote(frame, THU)
    assert got == {"last": 746.28, "time": "2026-11-05T10:25:00-05:00"}


def test_latest_quote_ignores_after_hours_bars():
    frame = _bars(THU, ["15:55", "16:00", "16:10"], [746.0, 747.0, 760.0])
    assert market_snapshot.latest_quote(frame, THU)["last"] == 746.0


@pytest.mark.parametrize(
    "frame",
    [
        _bars(date(2026, 11, 4), ["15:55"], [700.0]),  # nothing today: VIX before the open
        _bars(THU, ["10:00"], [float("nan")]),
        _bars(THU, ["10:00"], [0.0]),
    ],
)
def test_latest_quote_rejects_unusable_prices(frame):
    assert market_snapshot.latest_quote(frame, THU) is None


def test_take_snapshot_writes_file_and_never_touches_raw(tmp_path, monkeypatch):
    frames = {"SPY": _bars(THU, ["10:25"], [746.28]), "^VIX": _bars(THU, ["10:10"], [21.68])}
    monkeypatch.setattr(ingest, "_download_batch", lambda *a, **k: _batch(frames))
    assert market_snapshot.take_snapshot(_et(THU, 10, 26)) is True
    loaded = market_snapshot.load_snapshot()
    assert loaded["fetched_at"] == _et(THU, 10, 26)
    assert loaded["quotes"]["SPY"] == {"last": 746.28, "date": THU}
    assert loaded["quotes"]["VIX"] == {"last": 21.68, "date": THU}  # saved under its file name
    assert "QQQ" not in loaded["quotes"]
    assert not config.MARKET_SNAPSHOT_PATH.startswith(config.MARKET_CONTEXT_PROFILE["raw_dir"])
    assert os.listdir(os.path.dirname(config.MARKET_SNAPSHOT_PATH)) == ["snapshot.json"]


def _write_previous() -> None:
    os.makedirs(os.path.dirname(config.MARKET_SNAPSHOT_PATH))
    payload = {
        "fetched_at": "2026-11-05T10:00:00-05:00",
        "quotes": {"SPY": {"last": 750.0, "time": "2026-11-05T10:00:00-05:00"}},
    }
    with open(config.MARKET_SNAPSHOT_PATH, "w") as handle:
        json.dump(payload, handle)


def test_failed_download_keeps_previous_file(monkeypatch):
    _write_previous()

    def boom(*a, **k):
        raise OSError("throttled")

    monkeypatch.setattr(ingest, "_download_batch", boom)
    assert market_snapshot.take_snapshot(_et(THU, 10, 26)) is False
    assert market_snapshot.load_snapshot()["quotes"]["SPY"]["last"] == 750.0


def test_empty_download_keeps_previous_file(monkeypatch):
    _write_previous()
    stale = {"SPY": _bars(date(2026, 11, 4), ["15:55"], [700.0])}
    monkeypatch.setattr(ingest, "_download_batch", lambda *a, **k: _batch(stale))
    assert market_snapshot.take_snapshot(_et(THU, 10, 26)) is False
    assert market_snapshot.load_snapshot()["quotes"]["SPY"]["last"] == 750.0


def test_main_skips_the_download_outside_its_slots(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not download")

    monkeypatch.setattr(ingest, "_download_batch", boom)
    monkeypatch.setattr(market_snapshot, "now_et", lambda: _et(THU, 20, 0))
    assert market_snapshot.main([]) == 0


@pytest.mark.parametrize("content", ["not json", "{}", '{"fetched_at": "x", "quotes": {}}', "[]"])
def test_load_snapshot_survives_a_bad_file(content):
    os.makedirs(os.path.dirname(config.MARKET_SNAPSHOT_PATH))
    with open(config.MARKET_SNAPSHOT_PATH, "w") as handle:
        handle.write(content)
    assert market_snapshot.load_snapshot() is None


def test_load_snapshot_missing_file_and_bad_quote():
    assert market_snapshot.load_snapshot() is None
    os.makedirs(os.path.dirname(config.MARKET_SNAPSHOT_PATH))
    payload = {
        "fetched_at": "2026-11-05T10:00:00-05:00",
        "quotes": {
            "SPY": {"last": "x", "time": "2026-11-05T10:00:00-05:00"},
            "QQQ": {"last": 0, "time": "2026-11-05T10:00:00-05:00"},
            "IEF": {"last": 91.0, "time": "2026-11-05T10:00:00-05:00"},
        },
    }
    with open(config.MARKET_SNAPSHOT_PATH, "w") as handle:
        json.dump(payload, handle)
    assert list(market_snapshot.load_snapshot()["quotes"]) == ["IEF"]


@pytest.mark.parametrize(
    ("now", "label", "old"),
    [
        (_et(THU, 10, 30), "10:25 ET", None),
        (_et(THU, 10, 45), "10:25 ET", None),  # exactly 20 minutes is not overdue
        (_et(THU, 10, 46), "10:25 ET", 21),
        (_et(date(2026, 11, 6), 8, 0), "Thu 10:25 ET", 1295),
    ],
)
def test_status_label_and_old_minutes(now, label, old):
    got = market_snapshot.status(_et(THU, 10, 25), now)
    assert got == {"label": label, "old_minutes": old}


def test_status_last_slot_is_not_old_in_the_evening():
    assert market_snapshot.status(_et(THU, 16, 15), _et(THU, 18, 0))["old_minutes"] is None


def test_quote_applies_only_until_that_days_close_is_saved():
    quote = {"last": 12.0, "date": THU}
    assert insights.quote_is_new(_closes([10.0, 11.0]), quote) is True
    assert insights.quote_is_new(_closes([10.0, 11.0], last="2026-11-05"), quote) is False
    assert insights.quote_is_new(None, quote) is False
    assert insights.quote_is_new(_closes([10.0]), None) is False


def test_with_quote_appends_a_row_for_returns():
    df = insights.with_quote(_closes([10.0, 11.0]), {"last": 12.0, "date": THU})
    assert df["Close"].tolist() == [10.0, 11.0, 12.0]
    assert pd.Timestamp(df["Date"].iloc[-1]).date() == THU
    assert (
        len(insights.with_quote(_closes([10.0, 11.0], "2026-11-05"), {"last": 9, "date": THU})) == 2
    )


def test_volatility_uses_quote_against_last_close_and_closed_average():
    vix = _closes([15.0] * 19 + [16.61])
    closed = health.volatility(vix)
    got = health.volatility(vix, {"last": 21.68, "date": THU})
    assert got["live"] is True and closed["live"] is False
    assert got["level"] == 21.68
    assert got["change"] == pytest.approx(21.68 - 16.61)
    assert got["ma"] == closed["ma"]  # the average ignores the live price
    assert got["as_of"] == "2026-11-04"  # as_of stays the newest close


def test_credit_needs_both_quotes():
    closes = [10.0 + i * 0.1 for i in range(60)]
    hyg, ief = _closes(closes), _closes([20.0] * 60)
    one = {"HYG": {"last": 13.0, "date": THU}}
    assert health.credit(hyg, ief, one)["live"] is False
    assert health.credit(hyg, ief, one)["ratio"] == pytest.approx(15.9 / 20)
    both = one | {"IEF": {"last": 20.0, "date": THU}}
    got = health.credit(hyg, ief, both)
    assert got["live"] is True and got["ratio"] == pytest.approx(0.65)
    # 21 sessions before today is the 21st close from the end, not the 22nd.
    assert got["month_change"] == pytest.approx(0.65 / (closes[-21] / 20) - 1)
    assert got["ma"] == health.credit(hyg, ief)["ma"]


def test_breadth_counts_a_fund_on_its_live_price():
    frames = {"XLK": _closes([100.0] * 60)}
    got = health.breadth(frames, {"XLK": {"last": 90.0, "date": THU}})
    tile = next(t for t in got["tiles"] if t["symbol"] == "XLK")
    assert tile["live"] is True and tile["above"] is False
    assert tile["close"] == 90.0 and tile["pct"] == pytest.approx(-0.10)
    assert got["as_of"] == "2026-11-04"
