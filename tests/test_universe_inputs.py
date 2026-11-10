"""Scan-universe inputs: index lists, the stock watchlist, config, the refresh script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from finance_vibe import config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refresh_index_lists.py"


@pytest.fixture(scope="module")
def refresh():
    spec = importlib.util.spec_from_file_location("refresh_index_lists", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("path", "lo", "hi"),
    [(config.SP500_TICKERS_PATH, 495, 510), (config.SP400_TICKERS_PATH, 395, 410)],
)
def test_index_lists_are_clean(path, lo, hi):
    tickers = pd.read_csv(path)["Ticker"]
    assert lo <= len(tickers) <= hi
    assert tickers.is_unique and tickers.is_monotonic_increasing
    assert not tickers.str.contains(r"[.\s^]").any()  # Yahoo spelling (BRK-B), no blanks


def test_index_lists_do_not_overlap_and_are_dated():
    sp500 = set(pd.read_csv(config.SP500_TICKERS_PATH)["Ticker"])
    sp400 = set(pd.read_csv(config.SP400_TICKERS_PATH)["Ticker"])
    assert not sp500 & sp400
    meta = json.loads(Path(config.INDEX_LISTS_META_PATH).read_text())
    for name, tickers in (("sp500", sp500), ("sp400", sp400)):
        assert meta[name]["count"] == len(tickers)
        pd.Timestamp(meta[name]["taken"])  # an ISO date


def test_watchlist_is_stocks_only_with_a_header():
    watch = pd.read_csv(config.WATCHLIST_PATH)
    assert list(watch.columns) == ["Symbol"]  # without a header the first row was lost
    symbols = watch["Symbol"]
    assert symbols.is_unique and (symbols == symbols.str.strip().str.upper()).all()
    assert not set(symbols) & set(config.UNIVERSE_FUNDS)
    assert "SPY" not in set(symbols)


def test_universe_thresholds_have_a_buffer():
    assert config.UNIVERSE_STAY_DOLLAR_VOLUME < config.UNIVERSE_JOIN_DOLLAR_VOLUME
    assert config.UNIVERSE_STAY_CLOSE < config.UNIVERSE_JOIN_CLOSE
    assert config.UNIVERSE_REBALANCE_MONTHS == (3, 6, 9, 12)
    assert {"SPY", "QQQ", "SCHG", "SCHD"} <= set(config.UNIVERSE_FUNDS)
    assert config.BACKFILL_SP500_PATH == config.SP500_TICKERS_PATH


PAGE = """
<table class="infobox"><tr><th>Symbol</th></tr><tr><td>NOPE</td></tr></table>
<table class="wikitable sortable" id="constituents">
<tr><th>Symbol</th><th>Security</th></tr>
<tr><td><a href="#">BRK.B</a></td><td>Berkshire <table><tr><td>X</td></tr></table></td></tr>
<tr><td> aapl </td><td>Apple</td></tr>
<tr><td>AAPL</td><td>Apple again</td></tr>
</table>
<table class="wikitable"><tr><th>Date</th><th>Symbol</th></tr><tr><td>1</td><td>OLD</td></tr></table>
"""


def test_refresh_parses_the_first_symbol_wikitable(refresh):
    assert refresh.symbols_from_html(PAGE) == ["AAPL", "BRK-B"]
    with pytest.raises(ValueError, match="Symbol"):
        refresh.symbols_from_html('<table class="wikitable"><tr><th>Name</th></tr></table>')


def test_refresh_refuses_a_short_list(refresh, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(refresh, "DATA_DIR", tmp_path)
    monkeypatch.setattr(refresh, "META_PATH", tmp_path / "index_lists.json")
    monkeypatch.setattr(refresh, "fetch", lambda url: PAGE)
    assert refresh.main([]) == 1
    assert "expected" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())  # nothing written


def test_refresh_writes_lists_and_dates(refresh, tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "DATA_DIR", tmp_path)
    monkeypatch.setattr(refresh, "META_PATH", tmp_path / "index_lists.json")
    monkeypatch.setattr(refresh, "fetch", lambda url: PAGE)
    monkeypatch.setattr(refresh, "LISTS", {"tiny": ("https://example.invalid", "tiny.csv", (1, 5))})
    assert refresh.main(["--dry-run"]) == 0 and not (tmp_path / "tiny.csv").exists()
    assert refresh.main([]) == 0
    assert (tmp_path / "tiny.csv").read_text() == "Ticker\nAAPL\nBRK-B\n"
    meta = json.loads((tmp_path / "index_lists.json").read_text())
    assert meta["tiny"]["count"] == 2 and meta["tiny"]["file"] == "tiny.csv"
