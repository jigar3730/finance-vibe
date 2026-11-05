"""Market-context download (data.market_ingest): own silo, same contract and quality checks."""

from __future__ import annotations

import importlib.util
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.data import daily_ingest, ingest, market_ingest, quality

ET = ingest.MARKET_TZ
AFTER_CLOSE = datetime(2026, 10, 30, 18, 30, tzinfo=ET)
MIDDAY = datetime(2026, 10, 30, 12, 0, tzinfo=ET)


def _frame(n: int = 80, last: str = "2026-10-30", volume: float = 1e6) -> pd.DataFrame:
    close = np.linspace(20.0, 25.0, n)
    idx = pd.bdate_range(end=last, periods=n, name="Date")
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": volume},
        index=idx,
    )


def _batch(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """A yfinance group_by="ticker" batch keyed by Yahoo symbol."""
    return pd.concat(frames, axis=1, sort=True)


@pytest.fixture
def silo(tmp_path, monkeypatch):
    profile = dict(config.MARKET_CONTEXT_PROFILE)
    profile |= {"raw_dir": str(tmp_path / "raw" / "market"), "logs_dir": str(tmp_path / "logs")}
    monkeypatch.setattr(config, "MARKET_CONTEXT_PROFILE", profile)
    monkeypatch.setattr(config, "MARKET_CONTEXT_TICKERS", {"VIX": "^VIX", "HYG": "HYG"})
    return profile


def test_market_context_has_funds_vix_credit_and_eleven_sectors():
    tickers = config.MARKET_CONTEXT_TICKERS
    assert tickers["VIX"] == "^VIX" and {"HYG", "IEF"} <= set(tickers)
    assert len(config.MARKET_SECTOR_FUNDS) == 11 and set(config.MARKET_SECTOR_FUNDS) <= set(tickers)
    assert {"SPY", "QQQ", "IWM", "RSP", "SCHG", "SCHD"} <= set(tickers)
    # Only the broad funds may also be scanned, and scanners read a different folder.
    assert set(tickers) & set(config.STATIC_TICKERS) <= set(config.MARKET_INDEX_FUNDS)
    raw = config.MARKET_CONTEXT_PROFILE["raw_dir"]
    assert raw not in {p["raw_dir"] for p in config.TIMEFRAME_PROFILES.values()}


def test_saves_files_by_file_symbol_in_own_silo(silo, monkeypatch):
    batch = _batch({"^VIX": _frame(volume=0.0), "HYG": _frame()})
    monkeypatch.setattr(ingest, "_download_batch", lambda syms, *a, **k: batch)
    assert market_ingest.ingest_market_context(now=AFTER_CLOSE) == (2, 0)
    vix = pd.read_csv(config.get_raw_path("VIX", silo))
    assert config.get_raw_path("VIX", silo).endswith("market/VIX_2y_1d.csv")
    assert list(vix.columns) == config.REQUIRED_OHLCV and len(vix) == 80


def test_drops_todays_bar_while_the_session_is_open(silo, monkeypatch):
    batch = _batch({"^VIX": _frame(), "HYG": _frame()})
    monkeypatch.setattr(ingest, "_download_batch", lambda syms, *a, **k: batch)
    market_ingest.ingest_market_context(now=MIDDAY)
    hyg = pd.read_csv(config.get_raw_path("HYG", silo))
    assert len(hyg) == 79 and hyg["Date"].iloc[-1].startswith("2026-10-29")


def test_failed_ticker_is_logged_and_keeps_its_old_file(silo, monkeypatch):
    old = config.get_raw_path("HYG", silo)
    pd.DataFrame({"Date": ["2026-10-01"]}).to_csv(_mkdirs(old), index=False)
    batch = _batch({"^VIX": _frame(), "HYG": _frame(n=10)})  # too short to save
    monkeypatch.setattr(ingest, "_download_batch", lambda syms, *a, **k: batch)
    assert market_ingest.ingest_market_context(now=AFTER_CLOSE) == (1, 1)
    assert len(pd.read_csv(old)) == 1  # untouched
    errors = pd.read_csv(next(p for p in _logs(silo) if p.name.startswith("ingest_errors_")))
    assert errors["Ticker"].tolist() == ["HYG"] and "insufficient_rows" in errors["Error"][0]


def test_download_failure_logs_every_ticker(silo, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("yahoo down")

    monkeypatch.setattr(ingest, "_download_batch", boom)
    assert market_ingest.ingest_market_context(now=AFTER_CLOSE) == (0, 2)
    errors = pd.read_csv(next(p for p in _logs(silo) if p.name.startswith("ingest_errors_")))
    assert sorted(errors["Ticker"]) == ["HYG", "VIX"]


def test_main_writes_quality_report_without_vix_volume_flag(silo, monkeypatch):
    batch = _batch({"^VIX": _frame(volume=0.0), "HYG": _frame(volume=0.0, last="2026-10-28")})
    monkeypatch.setattr(ingest, "_download_batch", lambda syms, *a, **k: batch)
    monkeypatch.setattr(daily_ingest, "last_complete_session", lambda now=None: date(2026, 10, 30))
    assert market_ingest.main([]) == 0
    report = quality.latest_report(silo["logs_dir"])
    checks = set(zip(report["Symbol"], report["Check"], strict=True))
    assert ("HYG", "zero_volume") in checks and ("HYG", "stale") in checks
    assert not {c for c in checks if c[0] == "VIX"}  # an index has no volume


def _mkdirs(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _logs(silo):
    return list(Path(silo["logs_dir"]).iterdir())


def _health_check_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_daily_outputs.py"
    spec = importlib.util.spec_from_file_location("check_daily_outputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("scan", "market", "code"),
    [([], [], 0), (["x"], [], 1), (["x"], ["y"], 1), ([], ["y"], 2)],
)
def test_health_check_exit_code_tells_the_retry_what_failed(scan, market, code):
    assert _health_check_module().exit_code(scan, market) == code
