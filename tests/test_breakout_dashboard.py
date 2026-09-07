"""Breakout dashboard routes: run listing, KPI counts, and status filtering."""
from __future__ import annotations

import pandas as pd
import pytest

from finance_vibe import app as app_module
from finance_vibe.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _write_scan(folder, date_str):
    """Write a minimal breakout_setups CSV with a known status/readiness mix."""
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {"Symbol": "AAA", "Status": "PRE_BREAKOUT", "Breakout Readiness": 83},
            {"Symbol": "BBB", "Status": "WATCH", "Breakout Readiness": 72},
            {"Symbol": "CCC", "Status": "WATCH", "Breakout Readiness": 60},
            {"Symbol": "DDD", "Status": "FAILED_BREAKOUT", "Breakout Readiness": 25},
        ]
    )
    path = folder / f"breakout_setups_{date_str}.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def breakout_logs(tmp_path, monkeypatch):
    """Point breakout silos at tmp dirs and seed one daily scan."""
    modes = {
        "weekly": str(tmp_path / "weekly"),
        "daily": str(tmp_path / "daily"),
        "high_beta": str(tmp_path / "high_beta"),
    }
    monkeypatch.setattr(app_module, "BREAKOUT_MODES", modes)
    _write_scan(tmp_path / "daily", "2026-09-04")
    return modes


def test_breakout_routes_registered():
    rules = {rule.rule: rule.endpoint for rule in app.url_map.iter_rules()}
    assert rules.get("/breakout") == "breakout_index"
    assert rules.get("/breakout/<mode>/<date>") == "breakout_view"


def test_breakout_index_lists_runs(client, breakout_logs):
    response = client.get("/breakout")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "2026-09-04" in body
    assert "/breakout/daily/2026-09-04" in body


def test_breakout_view_kpis(client, breakout_logs):
    response = client.get("/breakout/daily/2026-09-04")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    # Setups = 4, actionable (>=70) = 2, failed = 1, median = 66.0.
    assert "Setups" in body
    assert ">4<" in body
    assert "66.0" in body  # median of [83, 72, 60, 25]
    # All four symbols present when unfiltered.
    for sym in ("AAA", "BBB", "CCC", "DDD"):
        assert sym in body


def test_breakout_view_status_filter(client, breakout_logs):
    response = client.get("/breakout/daily/2026-09-04?status=WATCH")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    # Table shows only WATCH rows; KPIs still reflect the full file.
    assert "BBB" in body
    assert "CCC" in body
    assert "AAA" not in body.split("table-container")[-1]
    assert "DDD" not in body.split("table-container")[-1]


def test_breakout_view_unknown_mode_404(client, breakout_logs):
    assert client.get("/breakout/bogus/2026-09-04").status_code == 404


def test_breakout_view_missing_file_404(client, breakout_logs):
    assert client.get("/breakout/daily/1999-01-01").status_code == 404
