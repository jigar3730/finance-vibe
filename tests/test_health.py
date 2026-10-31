"""Market health facts (web.health): volatility, credit, breadth from the market silo."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.web import health

LAST = date(2026, 10, 30)


def _bars(close, last: date = LAST, volume: float = 1e6) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range(end=pd.Timestamp(last), periods=len(close)).strftime("%Y-%m-%d"),
            "Open": close,
            "High": close + 0.1,
            "Low": close - 0.1,
            "Close": close,
            "Volume": volume,
        }
    )


@pytest.fixture
def silo(tmp_path, monkeypatch):
    profile = dict(config.MARKET_CONTEXT_PROFILE)
    profile |= {"raw_dir": str(tmp_path / "raw"), "logs_dir": str(tmp_path / "logs")}
    (tmp_path / "raw").mkdir()
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(config, "MARKET_CONTEXT_PROFILE", profile)

    def write(sym: str, df: pd.DataFrame) -> None:
        df.to_csv(config.get_raw_path(sym, profile), index=False)

    return write


UP, DOWN = np.linspace(10, 20, 80), np.linspace(20, 10, 80)


def test_volatility_level_change_and_20_day_average():
    vix = _bars([15.0] * 30 + [16.0, 18.5])
    v = health.volatility(vix[["Date", "Close"]].assign(Date=pd.to_datetime(vix["Date"])))
    assert v["level"] == 18.5 and v["change"] == pytest.approx(2.5)
    assert v["ma"] == pytest.approx((15.0 * 18 + 16.0 + 18.5) / 20)
    assert v["above_ma"] is True and v["as_of"] == "2026-10-30"


def test_credit_ratio_vs_50_day_average_and_month_change(silo):
    silo("HYG", _bars(np.linspace(80, 76, 80)))  # high yield falling
    silo("IEF", _bars([95.0] * 80))
    c = health.credit(health.closes("HYG"), health.closes("IEF"))
    assert c["ratio"] == pytest.approx(76 / 95)
    assert c["above_ma"] is False
    hyg = np.linspace(80, 76, 80)
    assert c["month_change"] == pytest.approx(hyg[-1] / hyg[-22] - 1)


def test_breadth_counts_sectors_with_plain_names(silo):
    for sym in config.MARKET_SECTOR_FUNDS:
        silo(sym, _bars(UP if sym in {"XLK", "XLE", "XLF"} else DOWN))
    frames = {s: health.closes(s) for s in config.MARKET_SECTOR_FUNDS}
    b = health.breadth(frames)
    assert (len(b["above"]), b["counted"], b["total"]) == (3, 11, 11)
    assert {s["symbol"] for s in b["above"]} == {"XLE", "XLF", "XLK"}
    assert {"symbol": "XLK", "name": "Technology"} in b["above"]
    assert {"symbol": "XLU", "name": "Utilities"} in b["below"]
    ranked = [t["symbol"] for t in b["ranked"]]
    assert set(ranked[:3]) == {"XLE", "XLF", "XLK"} and len(ranked) == 11
    pcts = [t["pct"] for t in b["ranked"]]
    assert pcts == sorted(pcts, reverse=True)
    xlk = next(t for t in b["tiles"] if t["symbol"] == "XLK")
    assert max(len(v) for v in config.MARKET_SECTOR_SHORT.values()) <= 9  # one row of 11
    assert xlk["short"] == "Tech" and xlk["pct"] == pytest.approx(xlk["close"] / xlk["ma"] - 1)


def test_missing_stale_and_quality_errors_are_flagged(silo, tmp_path):
    silo("VIX", _bars(UP, volume=0.0))  # no volume: not a flag
    silo("HYG", _bars(UP, last=date(2026, 10, 28)))  # two sessions old
    for sym in config.MARKET_SECTOR_FUNDS:
        if sym != "XLC":
            silo(sym, _bars(UP))
    pd.DataFrame(
        [{"Symbol": "XLE", "Check": "ohlc_incoherent", "Severity": "error", "Date": "2026-10-30"}]
    ).to_csv(tmp_path / "logs" / "data_quality_2026-10-30.csv", index=False)
    h = health.market_health(LAST)
    assert h["volatility_flags"] == []
    assert h["credit"] is None and h["credit_flags"] == ["HYG stale", "IEF missing"]
    assert h["breadth_flags"] == ["XLC missing", "XLE ohlc incoherent"]
    b = h["breadth"]
    assert b["counted"] == 10
    assert b["unknown"] == [{"symbol": "XLC", "name": "Communication services"}]


def test_empty_silo_is_no_data(silo):
    h = health.market_health(LAST)
    assert h["volatility"] is None and h["credit"] is None and h["breadth"]["counted"] == 0
