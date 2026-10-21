"""_fetch_live_prices: pooled per-symbol lookups, N/A fallback, short-lived cache."""

from __future__ import annotations

import pytest

from finance_vibe.web import app as app_module


@pytest.fixture
def fake_quotes(monkeypatch):
    """Record each per-symbol lookup; ZZZ always fails."""
    calls: list[str] = []

    def fake_one(sym):
        calls.append(sym)
        return "N/A" if sym == "ZZZ" else 10.0 + len(calls)

    monkeypatch.setattr(app_module, "_fetch_one_live_price", fake_one)
    monkeypatch.setattr(app_module, "_live_price_cache", {})
    return calls


def test_empty_symbols_makes_no_lookups(fake_quotes):
    assert app_module._fetch_live_prices([]) == {}
    assert fake_quotes == []


def test_every_symbol_is_quoted_and_failures_map_to_na(fake_quotes):
    prices = app_module._fetch_live_prices(["AAA", "BBB", "ZZZ"])
    assert set(prices) == {"AAA", "BBB", "ZZZ"}
    assert prices["ZZZ"] == "N/A"
    assert isinstance(prices["AAA"], float)
    assert sorted(fake_quotes) == ["AAA", "BBB", "ZZZ"]


def test_cache_reuses_quotes_but_retries_failures(fake_quotes):
    first = app_module._fetch_live_prices(["AAA", "ZZZ"])
    second = app_module._fetch_live_prices(["AAA", "ZZZ"])
    assert second["AAA"] == first["AAA"]
    assert sorted(fake_quotes) == ["AAA", "ZZZ", "ZZZ"]


def test_cache_expires_after_ttl(fake_quotes, monkeypatch):
    app_module._fetch_live_prices(["AAA"])
    monkeypatch.setattr(app_module, "LIVE_PRICE_TTL_SECONDS", 0)
    app_module._fetch_live_prices(["AAA"])
    assert fake_quotes == ["AAA", "AAA"]
