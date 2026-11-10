"""Backfill snapshot: one immutable download, rule-based universe, full-history checks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.data import ingest, quality
from finance_vibe.research import snapshot as sn

NOW = datetime(2026, 11, 9, 9, 0, tzinfo=ingest.MARKET_TZ)  # Monday before the close
DAILY_END = "2026-11-06"
WEEKLY_END = "2026-11-02"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _bars(dates: pd.DatetimeIndex, close: float = 100.0, volume: float = 1e6) -> pd.DataFrame:
    # Small alternating drift so closes never run flat.
    c = close * (1 + 0.001 * (np.arange(len(dates)) % 2))
    return pd.DataFrame(
        {"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": volume},
        index=dates.rename("Date"),
    )


def _daily(start: str = "2016-11-09") -> pd.DatetimeIndex:
    return pd.bdate_range(start, DAILY_END)


def _weekly(start: str = "2016-11-07") -> pd.DatetimeIndex:
    return pd.date_range(start, WEEKLY_END, freq="W-MON")


def _write(folder: Path, symbol: str, mode: str, df: pd.DataFrame) -> str:
    name = f"{mode}/{symbol}_10y_{config.BACKFILL_INTERVALS[mode]}.csv"
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(path, index=False)
    return name


def _snapshot(tmp_path: Path, frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> Path:
    folder = tmp_path / "2026-11-09"
    names = []
    for symbol, (day, week) in frames.items():
        names += [_write(folder, symbol, "daily", day), _write(folder, symbol, "weekly", week)]
    manifest = {
        "stamp": "2026-11-09",
        "expected_last": {"daily": DAILY_END, "weekly": WEEKLY_END},
        "requested": sorted(frames),
        "failures": {"daily": {}, "weekly": {}},
        "files": {n: sn._sha256(folder / n) for n in names},
    }
    (folder / sn.MANIFEST).write_text(json.dumps(manifest))
    return folder


def _good() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _bars(_daily()), _bars(_weekly())


def _reasons(tmp_path: Path, day: pd.DataFrame, week: pd.DataFrame | None = None) -> str:
    frames = {"SPY": _good(), "BAD": (day, _bars(_weekly()) if week is None else week)}
    table = sn.evaluate(_snapshot(tmp_path, frames)).set_index("Symbol")
    return str(table.loc["BAD", "Reasons"])


def test_clean_ticker_joins_universe_and_benchmark_does_not(tmp_path):
    table = sn.evaluate(_snapshot(tmp_path, {"SPY": _good(), "GOOD": _good()})).set_index("Symbol")
    assert bool(table.loc["GOOD", "In Universe"])
    assert not bool(table.loc["SPY", "In Universe"])
    assert table.loc["SPY", "Role"] == "benchmark"
    assert (tmp_path / "2026-11-09" / sn.UNIVERSE).exists()
    assert "Universe: 1 of 1" in (tmp_path / "2026-11-09" / sn.REPORT).read_text()


def test_recent_listing_is_short_history(tmp_path):
    assert "short_history" in _reasons(tmp_path, _bars(_daily("2020-01-02")))


def test_price_and_dollar_volume_floors(tmp_path):
    assert "low_price" in _reasons(tmp_path, _bars(_daily(), close=4.0))
    assert "low_dollar_volume" in _reasons(tmp_path / "v", _bars(_daily(), volume=1000.0))


def test_fund_is_excluded(tmp_path):
    frames = {"SPY": _good(), "SCHD": _good()}
    table = sn.evaluate(_snapshot(tmp_path, frames)).set_index("Symbol")
    assert "fund" in table.loc["SCHD", "Reasons"]


def test_stale_ticker_is_excluded(tmp_path):
    assert "daily:stale" in _reasons(tmp_path, _bars(_daily()).iloc[:-3])


def test_old_incoherent_bar_is_found_outside_the_recent_window(tmp_path):
    day = _bars(_daily())
    day.iloc[100, day.columns.get_loc("High")] = 1.0
    assert "daily:ohlc_incoherent" in _reasons(tmp_path, day)


def test_missing_sessions_over_the_limit(tmp_path):
    day = _bars(_daily())
    keep = np.ones(len(day), dtype=bool)
    keep[200 : 201 + config.BACKFILL_MAX_MISSING_BARS] = False
    assert "daily:missing_bars" in _reasons(tmp_path, day[keep])


def test_real_big_move_is_kept_but_spike_and_split_are_not():
    dates = pd.bdate_range("2020-01-01", periods=30)
    base = _bars(dates).reset_index()

    kept = base.copy()  # +60% that holds, not near a whole ratio
    kept.loc[10:, OHLCV[:4]] *= 1.6
    stats = quality.bar_stats(kept, pd.DatetimeIndex(dates))
    assert stats["big_moves"] == 1
    assert quality.quality_failures(stats) == []

    spike = base.copy()  # one bad print, undone next bar
    spike.loc[10, OHLCV[:4]] *= 1.6
    assert "spike_reversal" in quality.quality_failures(
        quality.bar_stats(spike, pd.DatetimeIndex(dates))
    )

    split = base.copy()  # unadjusted 4:1 split
    split.loc[10:, OHLCV[:4]] /= 4
    assert "split_like_move" in quality.quality_failures(
        quality.bar_stats(split, pd.DatetimeIndex(dates))
    )


def test_big_move_on_heavy_volume_is_real_news():
    dates = pd.bdate_range("2020-01-01", periods=30)
    frame = _bars(dates).reset_index()
    frame.loc[10, OHLCV[:4]] *= 2  # doubles for one bar, then undone
    frame.loc[[10, 11], "Volume"] *= 20
    stats = quality.bar_stats(frame, pd.DatetimeIndex(dates))
    assert stats["big_moves"] == 2
    assert quality.quality_failures(stats) == []


def test_read_tickers_merges_lists(tmp_path):
    (tmp_path / "a.csv").write_text("Ticker\nAAPL\nmsft\n")
    (tmp_path / "b.csv").write_text("Ticker\nMSFT\nBRK-B\n")
    paths = [str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]
    assert sn.read_tickers(paths) == ["AAPL", "BRK-B", "MSFT"]


def test_sp500_list_is_loadable():
    tickers = sn.read_tickers([config.BACKFILL_SP500_PATH])
    assert len(tickers) > 490
    assert "BRK-B" in tickers


def test_changed_file_is_rejected(tmp_path):
    folder = _snapshot(tmp_path, {"SPY": _good(), "GOOD": _good()})
    path = folder / "daily" / "GOOD_10y_1d.csv"
    path.write_text(path.read_text() + "\n2026-11-07,1,1,1,1,1\n")
    with pytest.raises(ValueError, match="changed"):
        sn.evaluate(folder)


def _fake_batch(calls):
    def fake(tickers, period, interval):
        calls.append((tuple(tickers), period, interval))
        if interval == "1d":
            dates = pd.bdate_range("2016-11-09", "2026-11-09")  # includes today's partial bar
        else:
            dates = pd.date_range("2016-11-07", "2026-11-09", freq="W-MON")  # week in progress
        return pd.concat({t: _bars(dates) for t in tickers}, axis=1)

    return fake


def test_download_drops_unfinished_bars_and_records_hashes(tmp_path, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(ingest, "_download_batch", _fake_batch(calls))
    folder = sn.download(["good"], tmp_path, now=NOW, commit="abc")

    assert folder.name == "2026-11-09"
    assert not (tmp_path / "2026-11-09.partial").exists()
    assert {c[1] for c in calls} == {"10y"}
    assert {c[2] for c in calls} == {"1d", "1wk"}
    manifest = sn.verify(folder)
    assert manifest["requested"] == ["GOOD", "QQQ", "SPY"]
    assert manifest["expected_last"] == {"daily": DAILY_END, "weekly": "2026-11-02"}
    day = pd.read_csv(folder / "daily" / "GOOD_10y_1d.csv")
    week = pd.read_csv(folder / "weekly" / "GOOD_10y_1wk.csv")
    assert str(day["Date"].iloc[-1])[:10] == DAILY_END
    assert str(week["Date"].iloc[-1])[:10] == "2026-11-02"


def test_download_refuses_to_overwrite(tmp_path, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(ingest, "_download_batch", _fake_batch(calls))
    (tmp_path / "2026-11-09").mkdir()
    with pytest.raises(FileExistsError):
        sn.download(["GOOD"], tmp_path, now=NOW)
    assert calls == []


def test_download_fails_loudly_without_a_benchmark(tmp_path, monkeypatch):
    def no_spy(tickers, period, interval):
        return _fake_batch([])([t for t in tickers if t != "SPY"], period, interval)

    monkeypatch.setattr(ingest, "_download_batch", no_spy)
    with pytest.raises(ValueError, match="benchmark"):
        sn.download(["GOOD"], tmp_path, now=NOW)
    assert not (tmp_path / "2026-11-09").exists()
