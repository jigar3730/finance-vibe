"""Shared fixtures."""

from __future__ import annotations

import pytest

from finance_vibe import config


@pytest.fixture(autouse=True)
def no_real_market_snapshot(tmp_path, monkeypatch):
    """Keep every test away from the real data/live quotes, which change by the minute."""
    monkeypatch.setattr(config, "MARKET_SNAPSHOT_PATH", str(tmp_path / "live" / "snapshot.json"))
