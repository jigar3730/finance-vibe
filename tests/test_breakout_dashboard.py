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
            {"Symbol": "AAA", "Status": "PRE_BREAKOUT", "Close": 100.0, "Breakout Readiness": 83},
            {"Symbol": "BBB", "Status": "WATCH", "Close": 50.0, "Breakout Readiness": 72},
            {"Symbol": "CCC", "Status": "WATCH", "Close": 20.0, "Breakout Readiness": 60},
            {"Symbol": "DDD", "Status": "FAILED_BREAKOUT", "Close": 10.0, "Breakout Readiness": 25},
        ]
    )
    path = folder / f"breakout_setups_{date_str}.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def fake_live_prices(monkeypatch):
    """Avoid network: AAA live below close, BBB above close, CCC equal, DDD unavailable."""
    prices = {"AAA": 95.0, "BBB": 55.5, "CCC": 20.0, "DDD": "N/A"}
    monkeypatch.setattr(app_module, "_fetch_live_prices", lambda syms: {s: prices[s] for s in syms})


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


def test_breakout_view_live_price_next_to_close(client, breakout_logs):
    body = client.get("/breakout/daily/2026-09-04").get_data(as_text=True)
    header = body.split("<thead>")[1].split("</thead>")[0]
    ths = [h.split("<")[0].strip() for h in header.split("<th>")[1:]]
    assert ths.index("Live Price") == ths.index("Close") + 1
    # close > live -> red; close < live -> green; equal / N/A -> neutral.
    assert 'live-price-cell live-below-close">$95.00' in body
    assert 'live-price-cell live-above-close">$55.50' in body
    assert 'live-price-cell live-flat">$20.00' in body
    assert 'live-price-cell live-flat">N/A' in body


def test_trade_plan_live_price_colour_and_surge_row(client, tmp_path, monkeypatch):
    modes = {"weekly": str(tmp_path / "weekly"), "daily": str(tmp_path / "daily")}
    monkeypatch.setattr(app_module, "MODES", modes)
    (tmp_path / "weekly").mkdir()
    # AAA live 95 < 100 (red), BBB live 55.5 vs 50 = +11% (green + surge),
    # CCC live == close (neutral), DDD live N/A.
    pd.DataFrame(
        [{"Symbol": s, "Score": 1, "Close": c} for s, c in
         [("AAA", 100.0), ("BBB", 50.0), ("CCC", 20.0), ("DDD", 10.0)]]
    ).to_csv(tmp_path / "weekly" / "trade_plan_2026-09-04.csv", index=False)

    body = client.get("/view/weekly/2026-09-04?file=trade_plan_2026-09-04.csv").get_data(as_text=True)
    header = body.split("<thead>")[1].split("</thead>")[0]
    ths = [h.split("<")[0].strip() for h in header.split("<th>")[1:]]
    assert ths.index("Live Price") == ths.index("Close") + 1
    assert 'live-price-cell live-below-close">$95.00' in body
    assert 'live-price-cell live-above-close live-surge">$55.50' in body
    assert 'live-price-cell live-flat">$20.00' in body
    assert body.count("live-surge") == 1


def test_live_price_cell_surge_threshold_is_strictly_over_five_percent():
    assert "live-surge" not in app_module._live_price_cell(105.0, 100.0, 5.0)
    assert "live-surge" in app_module._live_price_cell(105.01, 100.0, 5.0)
    assert "live-surge" not in app_module._live_price_cell(120.0, 100.0)
