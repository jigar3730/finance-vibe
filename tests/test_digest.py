"""Top 3 digest, step 1: filter, 12-1 momentum rank, drops, ledger (no network)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config, digest
from finance_vibe.research import ranker_features as rf
from finance_vibe.scanners import breakout as bs

AS_OF = date(2026, 11, 13)  # a Friday


def _row(symbol, status="BREAKOUT_CONFIRMED", trend="BULLISH", mtf="ALIGNED", dist=0.5):
    return {
        "Symbol": symbol,
        "Status": status,
        "Close": 100.0,
        "Resistance": 101.0,
        "Support": 95.0,
        "Distance Resistance ATR": dist,
        "ATR": 2.0,
        "RVOL20": 1.1,
        "Trend": trend,
        "MTF": mtf,
        "AsOf Date": AS_OF.isoformat(),
    }


def _closes(momentum: float, n: int = 300) -> pd.Series:
    """Closes whose MOM_12_1 on the last bar is exactly ``momentum``."""
    c = np.full(n, 100.0)
    c[-1 - 21] = 100.0 * (1 + momentum)
    return pd.Series(c)


def test_statuses_match_the_scanner():
    assert set(config.DIGEST_BREAKOUT_STATUSES) == {bs.STATUS_CONFIRMED, bs.STATUS_TRIGGERED}
    assert digest.PRE_BREAKOUT == bs.STATUS_PRE


def test_momentum_matches_the_research_definition():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=400)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    bars = pd.DataFrame(
        {"Date": days, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6}
    )
    bench = {k: bars for k in rf.BENCHMARKS}
    research = rf.ticker_features(bars, bench)["MOM_12_1"].iloc[-1]
    assert digest.momentum_12_1(bars["Close"]) == pytest.approx(research, rel=1e-12)
    assert np.isnan(digest.momentum_12_1(bars["Close"].iloc[:252]))  # needs C[t-252]


def test_candidate_filter():
    setups = pd.DataFrame(
        [
            _row("CONF"),
            _row("TRIG", status="BREAKOUT_TRIGGERED"),
            _row("NEAR", status="PRE_BREAKOUT", dist=1.5),
            _row("FAR", status="PRE_BREAKOUT", dist=1.6),
            _row("WATCH", status="WATCH"),
            _row("BEAR", trend="BEARISH"),
            _row("PART", mtf="PARTIAL"),
            _row("SPY"),  # a fund
        ]
    )
    assert list(digest.candidates(setups)["Symbol"]) == ["CONF", "TRIG", "NEAR"]


def test_sessions_and_the_earnings_window():
    assert digest.sessions_after(date(2026, 11, 20), date(2026, 11, 27)) == 4  # Thanksgiving
    assert digest.sessions_after(AS_OF, AS_OF) == 0
    assert digest.earnings_soon(AS_OF, AS_OF)  # reported on t: dropped
    assert digest.earnings_soon(AS_OF, date(2026, 11, 20))  # 5 sessions later
    assert not digest.earnings_soon(AS_OF, date(2026, 11, 23))  # 6 sessions later
    assert not digest.earnings_soon(AS_OF, date(2026, 11, 4))  # already past
    assert not digest.earnings_soon(AS_OF, None)  # unknown


def test_build_ranks_drops_and_selects():
    names = ["A", "B", "C", "D", "E", "F"]
    setups = pd.DataFrame([_row(s) for s in [*names, "NEW"]])
    mom = {"A": 0.10, "B": 0.50, "C": 0.40, "D": 0.30, "E": 0.20, "F": 0.05}
    closes = {s: _closes(m) for s, m in mom.items()} | {"NEW": _closes(0.9, n=200)}
    soon = {"C": date(2026, 11, 17)}  # earnings in 2 sessions
    asked = []

    def lookup(symbol):
        asked.append(symbol)
        return soon.get(symbol)

    table = digest.build(setups, closes, {"D"}, AS_OF, lookup)
    assert list(table.columns) == digest.DIGEST_COLUMNS
    by = table.set_index("Symbol")
    assert list(table["Symbol"]) == ["B", "C", "D", "E", "A", "F", "NEW"]  # momentum order
    assert by.loc["NEW", "Dropped"] == digest.DROP_NO_MOMENTUM  # under 253 bars
    assert by.loc["D", "Dropped"] == digest.DROP_QUALITY
    assert by.loc["C", "Dropped"] == digest.DROP_EARNINGS
    assert by.loc["C", "Earnings Date"] == "2026-11-17"
    assert by.loc["B", "Earnings Date"] == "unknown"
    assert list(table.loc[table["Selected"], "Symbol"]) == ["B", "E", "A"]
    assert list(table.loc[table["Selected"], "Rank"]) == [1, 2, 3]
    assert "D" not in asked and "NEW" not in asked  # dropped before the lookup
    assert by.loc["B", "Momentum Rank"] == 1 and pd.isna(by.loc["NEW", "Momentum Rank"])


def test_earnings_lookups_stop_once_enough_survive(monkeypatch):
    monkeypatch.setattr(config, "DIGEST_REVIEW_N", 2)
    setups = pd.DataFrame([_row(f"S{i}") for i in range(6)])
    closes = {f"S{i}": _closes(0.5 - i / 10) for i in range(6)}
    asked = []
    table = digest.build(setups, closes, set(), AS_OF, lambda s: asked.append(s))
    assert asked == ["S0", "S1"]
    assert list(table["Review"]) == [True, True, False, False, False, False]


def test_no_earnings_lookup_and_no_candidates():
    setups = pd.DataFrame([_row("A"), _row("B", status="WATCH")])
    table = digest.build(setups, {"A": _closes(0.1)}, set(), AS_OF, None)
    assert table.loc[0, "Selected"] and table.loc[0, "Earnings Date"] == ""
    empty = digest.build(pd.DataFrame([_row("W", status="WATCH")]), {}, set(), AS_OF, None)
    assert empty.empty and list(empty.columns) == digest.DIGEST_COLUMNS


def test_ledger_replaces_the_same_date(tmp_path):
    path = str(tmp_path / "digest" / "ledger.csv")
    setups = pd.DataFrame([_row("A"), _row("B")])
    first = digest.build(setups, {"A": _closes(0.1), "B": _closes(0.2)}, set(), AS_OF, None)
    digest.update_ledger(first, path)
    digest.update_ledger(first, path)  # a rerun of the same day
    assert len(pd.read_csv(path)) == 2
    later = first.assign(Date="2026-11-16")
    ledger = digest.update_ledger(later, path)
    assert list(ledger["Date"]) == ["2026-11-13", "2026-11-13", "2026-11-16", "2026-11-16"]


def test_run_end_to_end_from_run_files(tmp_path, monkeypatch):
    logs, raw_dir = tmp_path / "logs", tmp_path / "raw"
    logs.mkdir()
    raw_dir.mkdir()
    monkeypatch.setattr(
        config,
        "get_mode_config",
        lambda mode=None: {"logs_dir": str(logs), "raw_dir": str(raw_dir)},
    )
    monkeypatch.setattr(config, "DIGEST_DIR", str(tmp_path / "digest"))
    monkeypatch.setattr(config, "DIGEST_LEDGER", str(tmp_path / "digest" / "ledger.csv"))
    pd.DataFrame([_row("A"), _row("B"), _row("C")]).to_csv(
        logs / "breakout_setups_2026-11-13.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "Symbol": "C",
                "Check": "big_move",
                "Severity": "warn",
                "Date": "2026-11-12",
                "Detail": "x",
            }
        ]
    ).to_csv(logs / "data_quality_2026-11-13.csv", index=False)
    days = pd.bdate_range(end="2026-11-16", periods=320)  # one bar past the as-of date
    for sym, m in {"A": 0.1, "B": 0.3, "C": 0.5}.items():
        c = _closes(m, n=319).tolist() + [1.0]  # the bar after t must be ignored
        pd.DataFrame(
            {"Date": days, "Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}
        ).to_csv(raw_dir / f"{sym}_5y_1d.csv", index=False)

    table = digest.run(earnings=None)
    assert list(table.loc[table["Selected"], "Symbol"]) == ["B", "A"]
    assert table.set_index("Symbol").loc["C", "Dropped"] == digest.DROP_QUALITY
    assert table.set_index("Symbol").loc["B", digest.MOMENTUM] == pytest.approx(0.3)
    assert (tmp_path / "digest" / "digest_2026-11-13.csv").exists()
    assert len(pd.read_csv(tmp_path / "digest" / "ledger.csv")) == 3
