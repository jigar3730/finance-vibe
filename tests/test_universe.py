"""data.universe: quarterly schedule, join / stay hysteresis, funds, failures, outputs."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.data import daily_ingest, ingest
from finance_vibe.data import universe as uv

NOW = datetime(2026, 11, 10, 19, 0, tzinfo=ingest.MARKET_TZ)  # Tue after the close
LAST = date(2026, 11, 10)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [(2026, 9, date(2026, 9, 18)), (2026, 12, date(2026, 12, 18)), (2027, 3, date(2027, 3, 19))],
)
def test_third_friday(year, month, expected):
    assert uv.third_friday(year, month) == expected


def test_latest_and_next_rebalance():
    assert uv.latest_rebalance(date(2026, 11, 10)) == date(2026, 9, 18)
    assert uv.latest_rebalance(date(2026, 12, 18)) == date(2026, 9, 18)  # effective next day
    assert uv.latest_rebalance(date(2026, 12, 19)) == date(2026, 12, 18)
    assert uv.latest_rebalance(date(2027, 1, 5)) == date(2026, 12, 18)
    assert uv.next_rebalance(date(2026, 11, 10)) == date(2026, 12, 18)
    assert uv.next_rebalance(date(2026, 12, 19)) == date(2027, 3, 19)


def test_rebuild_reason():
    assert uv.rebuild_reason({}, date(2026, 11, 10)) == "initial"
    built = {"built": "2026-11-10"}
    assert uv.rebuild_reason(built, date(2026, 11, 14)) is None
    assert uv.rebuild_reason(built, date(2026, 12, 18)) is None
    assert uv.rebuild_reason(built, date(2026, 12, 19)) == "quarterly"  # Sat weekly run
    assert uv.rebuild_reason(built, date(2026, 11, 14), force=True) == "forced"
    assert uv.rebuild_reason({"built": "2026-12-19"}, date(2027, 2, 1)) is None


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _m(close=50.0, dv=30e6, bars=500, last=LAST, quality=()):
    return {
        "Close": close,
        "Dollar Volume": dv,
        "Bars": bars,
        "Last Bar": last,
        "quality": list(quality),
    }


@pytest.mark.parametrize(
    ("m", "keep", "reason"),
    [
        (_m(), True, None),
        (_m(dv=19.9e6), False, "dollar_volume_below_20M"),
        (_m(close=4.99), False, "close_below_5"),
        (_m(bars=251), False, "short_history"),
        (_m(last=date(2026, 11, 6)), False, "stale"),
        (_m(quality=["spike_reversal"]), False, "quality:spike_reversal"),
        ({}, False, "missing_download"),
    ],
)
def test_join_rules(m, keep, reason):
    ok, reasons = uv.judge(m, member=False, expected_last=LAST)
    assert ok is keep
    assert reasons == [] if reason is None else reason in reasons


@pytest.mark.parametrize(
    ("m", "keep", "reason"),
    [
        (_m(dv=15e6, close=4.0), True, None),  # inside the buffer: stays
        (_m(dv=14.9e6), False, "dollar_volume_below_15M"),
        (_m(close=3.99), False, "close_below_4"),
        (_m(bars=100, quality=["flat_closes"]), True, "quality:flat_closes"),  # a note only
        (_m(last=date(2026, 11, 6)), True, "stale"),
        ({}, True, "no_data_kept"),
    ],
)
def test_stay_rules(m, keep, reason):
    ok, reasons = uv.judge(m, member=True, expected_last=LAST)
    assert ok is keep
    assert reasons == [] if reason is None else reason in reasons


def _members(rows):
    return pd.DataFrame(
        [
            {
                "Symbol": s,
                "Role": role,
                "Sources": src,
                "Added": "2026-09-21",
                "Close": 1.0,
                "Dollar Volume": 1.0,
                "Bars": 1,
                "Notes": "",
            }
            for s, role, src in rows
        ],
        columns=uv.MEMBER_COLUMNS,
    )


def test_build_first_time_adds_funds_and_passing_stocks():
    candidates = {"AAA": ["S&P 500"], "BBB": ["S&P 400", "watchlist"], "CCC": ["watchlist"]}
    measures = {"AAA": _m(), "BBB": _m(dv=25e6), "CCC": _m(close=3.0)}
    members, changes, rejected = uv.build(
        candidates, measures, uv.read_members(Path("/nonexistent")), today=LAST, expected_last=LAST
    )
    assert list(members.columns) == uv.MEMBER_COLUMNS
    assert list(members.loc[members["Role"] == "fund", "Symbol"]) == list(config.UNIVERSE_FUNDS)
    stocks = members[members["Role"] == "stock"].set_index("Symbol")
    assert list(stocks.index) == ["AAA", "BBB"]
    assert stocks.loc["BBB", "Sources"] == "S&P 400;watchlist"
    assert (members["Added"] == "2026-11-10").all()
    assert list(rejected["Symbol"]) == ["CCC"] and rejected["Reasons"].iloc[0] == "close_below_5"
    stock_changes = changes[~changes["Symbol"].isin(config.UNIVERSE_FUNDS)]
    assert set(stock_changes["Reason"]) == {"initial build"}
    assert len(changes) == len(config.UNIVERSE_FUNDS) + 2


def test_build_hysteresis_and_removals():
    previous = _members(
        [("SPY", "fund", "fund"), ("KEEP", "stock", "S&P 500"), ("DROP", "stock", "S&P 400"),
         ("GONE", "stock", "S&P 400"), ("FAIL", "stock", "S&P 500")]
    )  # fmt: skip
    candidates = {s: ["S&P 500"] for s in ("KEEP", "DROP", "FAIL", "EDGE", "NEW")}
    measures = {
        "KEEP": _m(dv=16e6, close=4.5),  # a member inside the buffer
        "DROP": _m(dv=10e6),
        "EDGE": _m(dv=16e6),  # same numbers as KEEP, but not a member: no join
        "NEW": _m(),
    }  # FAIL has no download
    members, changes, rejected = uv.build(
        candidates, measures, previous, today=date(2026, 12, 19), expected_last=LAST
    )
    stocks = members[members["Role"] == "stock"].set_index("Symbol")
    assert sorted(stocks.index) == ["FAIL", "KEEP", "NEW"]
    assert stocks.loc["KEEP", "Added"] == "2026-09-21"  # date added is kept
    assert stocks.loc["NEW", "Added"] == "2026-12-19"
    assert stocks.loc["FAIL", "Notes"] == "no_data_kept"
    assert members.set_index("Symbol").loc["SPY", "Added"] == "2026-09-21"
    by = changes.set_index("Symbol")
    assert by.loc["NEW", "Reason"] == "passed the join rules"
    assert (
        by.loc["DROP", "Change"] == "removed"
        and "dollar_volume_below_15M" in by.loc["DROP", "Reason"]
    )
    assert by.loc["GONE", "Reason"] == "left_index_and_watchlist"
    assert "KEEP" not in by.index and "EDGE" not in by.index
    assert set(rejected["Symbol"]) == {"DROP", "EDGE"}
    funds_added = changes[changes["Symbol"].isin(config.UNIVERSE_FUNDS)]
    assert "SPY" not in set(funds_added["Symbol"])  # already a member


def test_ticker_list_funds_first():
    members = _members([("ZZZ", "stock", "x"), ("QQQ", "fund", "fund"), ("AAA", "stock", "x"),
                        ("SPY", "fund", "fund")])  # fmt: skip
    assert list(uv.ticker_list(members)) == ["SPY", "QQQ", "AAA", "ZZZ"]


def test_measure_uses_the_dollar_volume_window():
    dates = pd.bdate_range(end="2026-11-10", periods=300)
    close = np.full(300, 10.0)
    volume = np.r_[np.full(237, 1.0), np.full(63, 3e6)]  # only the last 63 bars count
    bars = pd.DataFrame({"Date": dates, "Open": close, "High": close * 1.01,
                         "Low": close * 0.99, "Close": close, "Volume": volume})  # fmt: skip
    bars["Close"] += np.arange(300) % 2 * 0.01  # never flat
    m = uv.measure(bars, pd.DatetimeIndex(dates))
    assert m["Bars"] == 300 and m["Last Bar"] == LAST
    assert m["Dollar Volume"] == pytest.approx(3e7, rel=1e-3)
    assert m["quality"] == []
    assert uv.measure(None, pd.DatetimeIndex(dates)) == {}


def test_candidates_merge_sources_and_skip_funds(tmp_path, monkeypatch):
    (tmp_path / "a.csv").write_text("Ticker\nAAA\nSPY\nBBB\n")
    (tmp_path / "b.csv").write_text("Ticker\nCCC\n")
    (tmp_path / "w.csv").write_text("Symbol\n bbb\nDDD\nQQQ\n")
    monkeypatch.setattr(config, "SP500_TICKERS_PATH", str(tmp_path / "a.csv"))
    monkeypatch.setattr(config, "SP400_TICKERS_PATH", str(tmp_path / "b.csv"))
    monkeypatch.setattr(config, "WATCHLIST_PATH", str(tmp_path / "w.csv"))
    assert uv.load_candidates() == {
        "AAA": ["S&P 500"],
        "BBB": ["S&P 500", "watchlist"],
        "CCC": ["S&P 400"],
        "DDD": ["watchlist"],
    }


# ---------------------------------------------------------------------------
# refresh(): IO, schedule and failures with a fake downloader
# ---------------------------------------------------------------------------


def _bars_csv(path: Path, *, close: float, volume: float, n: int = 300, end: str = "2026-11-10"):
    dates = pd.bdate_range(end=end, periods=n)
    c = close + np.arange(n) % 2 * 0.01
    pd.DataFrame({"Date": dates, "Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                  "Volume": volume}).to_csv(path, index=False)  # fmt: skip


@pytest.fixture
def lists(tmp_path, monkeypatch):
    (tmp_path / "sp500.csv").write_text("Ticker\nAAA\nBBB\n")
    (tmp_path / "sp400.csv").write_text("Ticker\nCCC\n")
    (tmp_path / "watch.csv").write_text("Symbol\nDDD\n")
    (tmp_path / "meta.json").write_text(json.dumps({"sp500": {"taken": "2026-11-10"}}))
    monkeypatch.setattr(config, "SP500_TICKERS_PATH", str(tmp_path / "sp500.csv"))
    monkeypatch.setattr(config, "SP400_TICKERS_PATH", str(tmp_path / "sp400.csv"))
    monkeypatch.setattr(config, "WATCHLIST_PATH", str(tmp_path / "watch.csv"))
    monkeypatch.setattr(config, "INDEX_LISTS_META_PATH", str(tmp_path / "meta.json"))
    return tmp_path


def _fake_downloader(prices, calls=None):
    """Writes bars for symbols in ``prices`` ({symbol: (close, volume)}); others fail."""

    def download(symbols, bars_dir, now):
        if calls is not None:
            calls.append(list(symbols))
        failed = {}
        for s in symbols:
            if s in prices:
                close, volume = prices[s]
                end = str(daily_ingest.last_complete_session(now))
                _bars_csv(bars_dir / f"{s}_2y_1d.csv", close=close, volume=volume, end=end)
            else:
                failed[s] = "empty_download"
        return failed

    return download


PRICES = {s: (100.0, 1e6) for s in config.UNIVERSE_FUNDS} | {
    "AAA": (50.0, 1e6),  # 50M a day
    "BBB": (50.0, 1e5),  # 5M a day
    "CCC": (3.0, 1e8),  # under 5.00
    "DDD": (30.0, 1e6),
}


def test_refresh_first_build_writes_everything(lists, tmp_path):
    out, tickers = tmp_path / "universe", tmp_path / "active_tickers.csv"
    calls = []
    meta = uv.refresh(out=out, ticker_list_path=tickers, now=NOW,
                      downloader=_fake_downloader(PRICES, calls))  # fmt: skip
    assert meta["status"] == "ok" and meta["built"] == "2026-11-10" and meta["reason"] == "initial"
    assert meta["stocks"] == 2 and meta["funds"] == len(config.UNIVERSE_FUNDS)
    assert meta["next_rebalance"] == "2026-12-18" and meta["index_lists_stale"] == []
    assert "SPY" in calls[0] and "AAA" in calls[0]
    assert list(pd.read_csv(tickers)["Ticker"]) == [*config.UNIVERSE_FUNDS, "AAA", "DDD"]
    rejected = pd.read_csv(out / uv.REJECTED).set_index("Symbol")["Reasons"].to_dict()
    assert rejected == {"BBB": "dollar_volume_below_20M", "CCC": "close_below_5"}
    assert (out / uv.CHANGES).exists() and (out / uv.BARS / "AAA_2y_1d.csv").exists()

    # Next run, same quarter: no download, the list is rewritten from the members.
    tickers.unlink()
    calls.clear()
    uv.refresh(out=out, ticker_list_path=tickers,
               now=datetime(2026, 11, 14, 9, 0, tzinfo=ingest.MARKET_TZ),
               downloader=_fake_downloader(PRICES, calls))  # fmt: skip
    assert calls == [] and len(pd.read_csv(tickers)) == len(config.UNIVERSE_FUNDS) + 2


def test_refresh_quarterly_rebuild_appends_changes(lists, tmp_path):
    out, tickers = tmp_path / "universe", tmp_path / "active_tickers.csv"
    uv.refresh(out=out, ticker_list_path=tickers, now=NOW, downloader=_fake_downloader(PRICES))
    later = PRICES | {"AAA": (50.0, 3.5e5), "BBB": (50.0, 5e5)}  # AAA 17.5M stays; BBB 25M joins
    uv.refresh(out=out, ticker_list_path=tickers,
               now=datetime(2026, 12, 19, 9, 0, tzinfo=ingest.MARKET_TZ),
               downloader=_fake_downloader(later))  # fmt: skip
    changes = pd.read_csv(out / uv.CHANGES)
    assert list(changes.loc[changes["Date"] == "2026-12-19", "Symbol"]) == ["BBB"]
    assert set(pd.read_csv(tickers)["Ticker"]) >= {"AAA", "BBB", "DDD"}
    meta = uv.read_meta(out)
    assert meta["reason"] == "quarterly" and meta["index_lists_stale"] == ["sp500"]


def test_refresh_too_many_failures_keeps_previous(lists, tmp_path):
    out, tickers = tmp_path / "universe", tmp_path / "active_tickers.csv"
    uv.refresh(out=out, ticker_list_path=tickers, now=NOW, downloader=_fake_downloader(PRICES))
    before = (out / uv.MEMBERS).read_text()
    meta = uv.refresh(out=out, ticker_list_path=tickers, now=NOW, force=True,
                      downloader=_fake_downloader({}))  # fmt: skip
    assert meta["status"] == "kept_previous" and meta["built"] == "2026-11-10"
    assert (out / uv.MEMBERS).read_text() == before
    assert len(pd.read_csv(tickers)) == len(config.UNIVERSE_FUNDS) + 2


def test_refresh_failed_first_build_leaves_ticker_list_alone(lists, tmp_path):
    out, tickers = tmp_path / "universe", tmp_path / "active_tickers.csv"
    tickers.write_text("Ticker\nOLD\n")
    meta = uv.refresh(out=out, ticker_list_path=tickers, now=NOW,
                      downloader=_fake_downloader({}))  # fmt: skip
    assert meta["status"] == "kept_previous" and "built" not in meta
    assert tickers.read_text() == "Ticker\nOLD\n"
    assert not (out / uv.MEMBERS).exists()


def test_main_out_dir(lists, tmp_path, monkeypatch):
    monkeypatch.setattr(uv, "download_bars", _fake_downloader(PRICES))
    out = tmp_path / "trial"
    assert uv.main(["--out", str(out), "--rebuild"]) == 0
    assert (out / "active_tickers.csv").exists() and (out / uv.MEMBERS).exists()


def _yf_batch(rows_by_symbol):
    """A yfinance-style grouped frame: {symbol: n bars} up to 2026-11-10."""
    frames = {}
    for sym, n in rows_by_symbol.items():
        idx = pd.bdate_range(end="2026-11-10", periods=n, name="Date")
        c = 10.0 + np.arange(n) % 2 * 0.01
        frames[sym] = pd.DataFrame(
            {"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}, index=idx
        )
    return pd.concat(frames, axis=1)


def test_download_bars_retries_empty_and_short_results(tmp_path, monkeypatch):
    calls = []
    first = {"AAA": 300, "SHORT": 2}  # EMPTY missing, SHORT truncated

    def fake_batch(chunk, period, interval):
        calls.append(sorted(chunk))
        if len(calls) == 1:
            return _yf_batch({s: n for s, n in first.items() if s in chunk})
        return _yf_batch({"SHORT": 300, "NEW": 30})  # retry: full SHORT, NEW stays short

    monkeypatch.setattr(uv.ingest, "_download_batch", fake_batch)
    failed = uv.download_bars(["AAA", "EMPTY", "NEW", "SHORT"], tmp_path, NOW)
    assert calls[1] == ["EMPTY", "NEW", "SHORT"]
    assert failed == {"EMPTY": "empty_download"}
    assert len(pd.read_csv(tmp_path / "SHORT_2y_1d.csv")) == 300
    assert len(pd.read_csv(tmp_path / "NEW_2y_1d.csv")) == 30  # a real new listing is kept
