"""Dashboard routes: run listing, Coiled Cobra setups, breakout KPIs and status filtering."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from finance_vibe.web import app as app_module
from finance_vibe.web.app import app


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
    """Point the log silos at tmp dirs and seed one daily breakout scan."""
    modes = {
        "weekly": str(tmp_path / "weekly"),
        "daily": str(tmp_path / "daily"),
    }
    monkeypatch.setattr(app_module, "LOG_DIRS", modes)
    _write_scan(tmp_path / "daily", "2026-09-04")
    return modes


def test_routes_registered():
    rules = {rule.rule: rule.endpoint for rule in app.url_map.iter_rules()}
    assert rules.get("/") == "index"
    assert rules.get("/runs") == "runs"
    assert rules.get("/changes") == "changes"
    assert rules.get("/cobra/<mode>/<date>") == "cobra_view"
    assert rules.get("/breakout/<mode>/<date>") == "breakout_view"


def test_index_links_each_scanner_that_ran(client, breakout_logs, tmp_path):
    pd.DataFrame([{"Symbol": "AAA", "Score": 90, "Close": 1.0}]).to_csv(
        tmp_path / "daily" / "coiled_cobra_setups_2026-09-05.csv", index=False
    )
    body = client.get("/runs").get_data(as_text=True)
    assert "/breakout/daily/2026-09-04" in body and "/cobra/daily/2026-09-04" not in body
    assert "/cobra/daily/2026-09-05" in body and "/breakout/daily/2026-09-05" not in body
    assert body.index("2026-09-05") < body.index("2026-09-04")  # newest first


def test_breakout_view_kpis(client, breakout_logs):
    response = client.get("/breakout/daily/2026-09-04")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    # Setups = 4, actionable (>=70) = 2, failed = 1, median = 66.0.
    assert "Setups" in body
    assert ">4<" in body
    assert "66.0" in body  # median of [83, 72, 60, 25]
    # Default view: PRE_BREAKOUT and the failed section; WATCH rows are hidden.
    for sym in ("AAA", "DDD"):
        assert sym in body
    for sym in ("BBB", "CCC"):
        assert sym not in body
    # "All" lists every row.
    show_all = client.get("/breakout/daily/2026-09-04?show=all").get_data(as_text=True)
    for sym in ("AAA", "BBB", "CCC", "DDD"):
        assert sym in show_all


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
    body = client.get("/breakout/daily/2026-09-04?show=all").get_data(as_text=True)
    header = body.split("<thead>")[1].split("</thead>")[0]
    ths = [h.split("<")[0].strip() for h in header.split("<th>")[1:]]
    assert ths.index("Live Price") == ths.index("Close") + 1
    # close > live -> red; close < live -> green; equal / N/A -> neutral.
    assert 'live-price-cell live-below-close">$95.00' in body
    assert 'live-price-cell live-above-close">$55.50' in body
    assert 'live-price-cell live-flat">$20.00' in body
    assert 'live-price-cell live-flat">N/A' in body


def test_cobra_view_orders_by_score_with_live_price_and_surge_row(client, tmp_path, monkeypatch):
    modes = {"weekly": str(tmp_path / "weekly"), "daily": str(tmp_path / "daily")}
    monkeypatch.setattr(app_module, "LOG_DIRS", modes)
    (tmp_path / "weekly").mkdir()
    # AAA live 95 < 100 (red), BBB live 55.5 vs 50 = +11% (green + surge),
    # CCC live == close (neutral), DDD live N/A.
    pd.DataFrame(
        [
            {"Symbol": s, "Score": score, "Close": c, "Pct_From_EMA20": 0.1}
            for s, score, c in [
                ("CCC", 70, 20.0),
                ("AAA", 95, 100.0),
                ("DDD", 60, 10.0),
                ("BBB", 80, 50.0),
            ]
        ]
    ).to_csv(tmp_path / "weekly" / "coiled_cobra_setups_2026-09-04.csv", index=False)

    body = client.get("/cobra/weekly/2026-09-04").get_data(as_text=True)
    header = body.split("<thead>")[1].split("</thead>")[0]
    ths = [h.split("<")[0].strip() for h in header.split("<th>")[1:]]
    assert ths.index("Live Price") == ths.index("Close") + 1
    assert "Pct_From_EMA20" not in ths  # only the display columns
    table = body.split("<tbody>")[1]
    assert [table.index(s) for s in ("AAA", "BBB", "CCC", "DDD")] == sorted(
        table.index(s) for s in ("AAA", "BBB", "CCC", "DDD")
    )  # Score descending
    assert 'live-price-cell live-below-close">$95.00' in body
    assert 'live-price-cell live-above-close live-surge">$55.50' in body
    assert 'live-price-cell live-flat">$20.00' in body
    assert body.count("live-surge") == 1


def test_live_price_cell_surge_threshold_is_strictly_over_five_percent():
    assert "live-surge" not in app_module._live_price_cell(105.0, 100.0, 5.0)
    assert "live-surge" in app_module._live_price_cell(105.01, 100.0, 5.0)
    assert "live-surge" not in app_module._live_price_cell(120.0, 100.0)


def _write_rows(folder, date_str, rows):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / f"breakout_setups_{date_str}.csv", index=False)


@pytest.fixture
def display_logs(tmp_path, monkeypatch):
    """A scan exercising the display rule, the failed split, and the old-CSV relabel."""
    monkeypatch.setattr(
        app_module, "LOG_DIRS", {"weekly": str(tmp_path / "w"), "daily": str(tmp_path / "d")}
    )
    monkeypatch.setattr(app_module, "_fetch_live_prices", lambda syms: {s: "N/A" for s in syms})
    row = {"Close": 10.0, "Breakout Triggered": False}
    _write_rows(
        tmp_path / "d",
        "2026-10-20",
        [
            {**row, "Symbol": "LOWW", "Status": "WATCH", "Breakout Readiness": 54},
            {**row, "Symbol": "HIGW", "Status": "WATCH", "Breakout Readiness": 55},
            {**row, "Symbol": "LOWD", "Status": "DEVELOPING", "Breakout Readiness": 49},
            {**row, "Symbol": "CONF", "Status": "BREAKOUT_CONFIRMED", "Breakout Readiness": 40},
            {**row, "Symbol": "FAIL", "Status": "FAILED_BREAKOUT", "Breakout Readiness": 5},
            # Pre-2026-10-20 CSV: triggered-but-unconfirmed filed under WATCH.
            {
                **row,
                "Symbol": "TRIG",
                "Status": "WATCH",
                "Breakout Readiness": 20,
                "Breakout Triggered": True,
            },
        ],
    )


def _main_table(body):
    return body.split('class="table-container"')[1].split("</table>")[0]


def test_default_view_applies_display_rule_and_splits_failed(client, display_logs):
    body = client.get("/breakout/daily/2026-10-20").get_data(as_text=True)
    main = _main_table(body)
    for sym in ("CONF", "TRIG"):
        assert sym in main
    for sym in ("LOWW", "HIGW", "LOWD", "FAIL"):  # WATCH never, DEVELOPING below its floor
        assert sym not in main
    assert "failed-section" in body and "FAIL" in body.split("failed-section")[1]
    # Status order: confirmed, then triggered.
    assert main.index("CONF") < main.index("TRIG")


def test_old_watch_rows_with_trigger_flag_are_relabelled(client, display_logs):
    body = client.get("/breakout/daily/2026-10-20?status=BREAKOUT_TRIGGERED").get_data(as_text=True)
    main = _main_table(body)
    assert "TRIG" in main and "HIGW" not in main


def test_show_all_lists_every_row_in_one_table(client, display_logs):
    body = client.get("/breakout/daily/2026-10-20?show=all").get_data(as_text=True)
    main = _main_table(body)
    for sym in ("LOWW", "HIGW", "LOWD", "CONF", "FAIL", "TRIG"):
        assert sym in main
    assert "failed-section" not in body


def test_status_chip_ignores_readiness_floor(client, display_logs):
    main = _main_table(client.get("/breakout/daily/2026-10-20?status=WATCH").get_data(as_text=True))
    assert "LOWW" in main and "HIGW" in main and "TRIG" not in main


def test_dashboard_display_rule_mirrors_scanner():
    from finance_vibe.scanners import breakout as bs

    assert bs.CANDIDATE_STATUSES == app_module.BREAKOUT_ALWAYS_SHOWN
    assert bs.DISPLAY_SCORE_FLOOR == app_module.BREAKOUT_SCORE_FLOOR
    assert bs.RVOL_CONFIRM == app_module.playbook.RVOL_CONFIRM
    rank = bs._sort_candidates(
        pd.DataFrame(
            {
                "Status": app_module.BREAKOUT_STATUS_ORDER[::-1],
                "Breakout Readiness": 0,
                "Symbol": "X",
            }
        )
    )
    assert list(rank["Status"]) == app_module.BREAKOUT_STATUS_ORDER


def test_top_setups_page_shows_cards_per_mode(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "LOG_DIRS", {"weekly": str(tmp_path / "w"), "daily": str(tmp_path / "d")}
    )
    monkeypatch.setattr(app_module, "_fetch_live_prices", lambda syms: {s: "N/A" for s in syms})
    (tmp_path / "d").mkdir()
    levels = {"Close": 48.68, "Resistance": 50.31, "Support": 44.62, "ATR": 1.7, "MTF": "ALIGNED"}
    pd.DataFrame(
        [
            {"Symbol": "VRNS", "Status": "WATCH", "Breakout Readiness": 72, **levels},
            {"Symbol": "MMM", "Status": "BREAKOUT_CONFIRMED", "Breakout Readiness": 62, **levels},
            {"Symbol": "DHR", "Status": "FAILED_BREAKOUT", "Breakout Readiness": 28, **levels},
        ]
    ).to_csv(tmp_path / "d" / "breakout_setups_2026-10-20.csv", index=False)
    pd.DataFrame(
        [{"Symbol": "VRNS", "Score": 85, "Grade": "A - Watch", "Swing Low": 45.65}]
    ).to_csv(tmp_path / "d" / "coiled_cobra_setups_2026-10-20.csv", index=False)
    pd.DataFrame(
        [
            {
                "Symbol": "VRNS",
                "Check": "ohlc_incoherent",
                "Severity": "error",
                "Date": "2026-10-20",
            },
            {"Symbol": "MMM", "Check": "big_move", "Severity": "warn", "Date": "2026-10-20"},
        ]
    ).to_csv(tmp_path / "d" / "data_quality_2026-10-20.csv", index=False)

    body = client.get("/").get_data(as_text=True)
    assert "run 2026-10-20" in body
    vrns, mmm = (body.split(f"quote.ashx?t={s}")[1].split("</header>")[0] for s in ("VRNS", "MMM"))
    assert "data: ohlc incoherent" in vrns and "dq-flag" not in mmm  # errors only
    assert body.index("VRNS") < body.index("MMM")  # Cobra 85 ranks above 62 + 20
    assert "DHR" not in body  # failed breakouts are excluded
    assert "50.31" in body and "45.23" in body  # trigger and stop
    assert "status-waiting" in body and "wide stop" in body
    assert "No weekly setups" in body
    # Each card carries a level chart wired to its plan.
    chart = body.split('class="level-chart"')[1].split("></div>")[0]
    assert 'data-mode="daily"' in chart and 'data-symbol="VRNS"' in chart
    levels = json.loads(chart.split("data-levels='")[1].split("'")[0])
    assert {line["label"]: line["price"] for line in levels} == {
        "Trigger": 50.31,
        "Stop": 45.23,
        "T1": 60.48,
        "T2": 65.56,
        "Support": 44.62,
    }
    assert "headwind-banner" not in body
    assert "lightweight-charts-4.2.3.standalone.production.js" in body


@pytest.fixture
def raw_bars(tmp_path, monkeypatch):
    """A daily raw silo with one ticker of 150 bars (out of date order on disk)."""
    raw_dir = tmp_path / "raw" / "daily"
    raw_dir.mkdir(parents=True)
    dates = pd.bdate_range("2026-01-01", periods=150)
    frame = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": range(150),
            "High": [x + 2 for x in range(150)],
            "Low": [x - 1 for x in range(150)],
            "Close": [x + 1 for x in range(150)],
            "Volume": 1000,
        }
    )
    frame.iloc[::-1].to_csv(raw_dir / "VRNS_5y_1d.csv", index=False)
    profiles = {m: dict(p) for m, p in app_module.config.TIMEFRAME_PROFILES.items()}
    profiles["daily"]["raw_dir"] = str(raw_dir)
    monkeypatch.setattr(app_module.config, "TIMEFRAME_PROFILES", profiles)
    return dates


def test_api_bars_returns_recent_bars_in_date_order(client, raw_bars):
    response = client.get("/api/bars/daily/VRNS")
    assert response.status_code == 200
    bars = response.get_json()
    assert len(bars) == app_module.CHART_BARS["daily"] == 60
    assert [b["time"] for b in bars] == sorted(b["time"] for b in bars)
    assert bars[-1] == {
        "time": raw_bars[-1].strftime("%Y-%m-%d"),
        "open": 149.0,
        "high": 151.0,
        "low": 148.0,
        "close": 150.0,
    }
    assert response.headers["Cache-Control"] == "max-age=60"


@pytest.mark.parametrize(
    "url",
    [
        "/api/bars/daily/NOPE",  # no raw file
        "/api/bars/hourly/VRNS",  # unknown mode
        "/api/bars/daily/vrns",  # symbols are upper-case
        "/api/bars/daily/..%2Fx",  # path tricks never reach the filesystem
        "/api/bars/daily/TOOLONGSYMBOL",
    ],
)
def test_api_bars_rejects_unknown_or_unsafe_requests(client, raw_bars, url):
    assert client.get(url).status_code == 404


def test_api_bars_missing_raw_dir_is_404(client, raw_bars, tmp_path, monkeypatch):
    app_module.config.TIMEFRAME_PROFILES["daily"]["raw_dir"] = str(tmp_path / "gone")
    assert client.get("/api/bars/daily/VRNS").status_code == 404


def _market_logs(tmp_path, monkeypatch, weekly_trend):
    dirs = {"weekly": tmp_path / "w", "daily": tmp_path / "d"}
    for d in dirs.values():
        d.mkdir()
    monkeypatch.setattr(app_module, "LOG_DIRS", {m: str(d) for m, d in dirs.items()})
    monkeypatch.setattr(app_module, "_fetch_live_prices", lambda syms: {s: "N/A" for s in syms})
    profiles = {m: dict(p) for m, p in app_module.config.TIMEFRAME_PROFILES.items()}
    for m in profiles:
        profiles[m]["raw_dir"] = str(tmp_path / "raw" / m)  # no raw files: row dates only
    monkeypatch.setattr(app_module.config, "TIMEFRAME_PROFILES", profiles)
    monkeypatch.setattr(
        app_module.insights,
        "expected_bars",
        lambda now=None: {"daily": date(2026, 10, 21), "weekly": date(2026, 10, 19)},
    )
    base = {
        "AsOf Date": "2026-10-20",
        "Status": "WATCH",
        "Breakout Readiness": 10,
        "Resistance": 781.62,
        "Support": 758.79,
    }
    base |= {"ATR": 7.08, "SMA20": 770.62, "SMA50": 767.26, "Momentum": "FADING"}
    daily = [
        {"Symbol": s, "Close": 760.0, "Trend": "NEUTRAL", "MTF": "PARTIAL", **base}
        for s in ("SPY", "QQQ")
    ]
    weekly = [{"Symbol": "SPY", "Close": 760.0, "Trend": weekly_trend, "MTF": "PARTIAL", **base}]
    pd.DataFrame(daily).to_csv(dirs["daily"] / "breakout_setups_2026-10-20.csv", index=False)
    pd.DataFrame(weekly).to_csv(dirs["weekly"] / "breakout_setups_2026-10-20.csv", index=False)


def test_market_page_shows_bias_and_both_plans(client, tmp_path, monkeypatch):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BEARISH")
    body = client.get("/market").get_data(as_text=True)
    spy = body.split("quote.ashx?t=SPY")[1].split("</article>")[0]
    assert "bias-short" in spy and "SHORT" in spy
    assert "Long: break above resistance" in spy and "Short: break below support" in spy
    long_block = spy.split("Long: break above resistance")[0].rsplit('<div class="plan-block', 1)[1]
    assert long_block.startswith(" plan-off")  # off-bias side is dimmed
    assert "758.79" in spy and "781.62" in spy
    levels = json.loads(spy.split("data-levels='")[1].split("'")[0])
    assert levels[0]["label"] == "Short Trigger" and levels[-1]["label"] == "Long trigger"
    assert "No daily scan row for SCHG" in body and "No daily scan row for SCHD" in body


def test_market_cards_show_freshness(client, tmp_path, monkeypatch):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    body = client.get("/market").get_data(as_text=True)
    spy = body.split("quote.ashx?t=SPY")[1].split("</article>")[0]
    fresh = spy.split('class="setup-note freshness"')[1].split("</p>")[0]
    # Daily row is a session behind; the weekly row is current.
    assert "Daily as of 2026-10-20" in fresh and "stale" in fresh.split("Weekly")[0]
    assert "Weekly as of 2026-10-20" in fresh and "warn" not in fresh.split("Weekly")[1]
    qqq = body.split("quote.ashx?t=QQQ")[1].split("</article>")[0]
    assert "missing" in qqq.split("Weekly")[1].split("</p>")[0]  # no weekly QQQ row


def test_top_setups_banner_when_spy_is_short(client, tmp_path, monkeypatch):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BEARISH")
    body = client.get("/").get_data(as_text=True)
    assert "headwind-banner" in body and "<strong>SPY</strong> bias SHORT" in body


def test_no_banner_when_spy_is_not_short(client, tmp_path, monkeypatch):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    assert "headwind-banner" not in client.get("/").get_data(as_text=True)


def test_market_page_without_runs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "LOG_DIRS", {"weekly": str(tmp_path / "x"), "daily": str(tmp_path / "y")}
    )
    monkeypatch.setattr(app_module, "_fetch_live_prices", lambda syms: {})
    body = client.get("/market").get_data(as_text=True)
    assert body.count("No daily scan row") == 4


def test_nan_quote_from_yfinance_is_not_a_price(monkeypatch):
    class FakeTicker:
        def __init__(self, sym):
            self.fast_info = {"last_price": float("nan")}

    monkeypatch.setattr(app_module.yf, "Ticker", FakeTicker)
    assert app_module._fetch_one_live_price("XYZ") == "N/A"


@pytest.mark.parametrize(
    ("value", "shown"), [(float("nan"), "—"), (None, "—"), ("", "—"), ("ALIGNED", "ALIGNED")]
)
def test_text_filter_hides_blank_cells(value, shown):
    assert app_module._text_filter(value) == shown


def test_changes_page_lists_transitions_and_cobra_changes(client, tmp_path, monkeypatch):
    dirs = {"weekly": tmp_path / "w", "daily": tmp_path / "d"}
    for d in dirs.values():
        d.mkdir()
    monkeypatch.setattr(app_module, "LOG_DIRS", {m: str(d) for m, d in dirs.items()})
    for day, statuses in (
        ("2026-10-29", {"AAA": "WATCH", "BBB": "BREAKOUT_TRIGGERED", "SPY": "WATCH"}),
        ("2026-10-30", {"AAA": "PRE_BREAKOUT", "BBB": "FAILED_BREAKOUT", "SPY": "WATCH"}),
    ):
        pd.DataFrame(
            [{"Symbol": s, "Status": st, "Breakout Readiness": 50} for s, st in statuses.items()]
        ).to_csv(dirs["daily"] / f"breakout_setups_{day}.csv", index=False)
    pd.DataFrame([{"Symbol": "CCC", "Score": 80}]).to_csv(
        dirs["daily"] / "coiled_cobra_setups_2026-10-30.csv", index=False
    )
    body = client.get("/changes").get_data(as_text=True)
    daily = body.split("⚡ Daily")[1]
    assert "2026-10-29 → 2026-10-30" in daily
    assert "Moved toward a breakout (1)" in daily and "quote.ashx?t=AAA" in daily
    assert "Failed breakouts (1)" in daily and "quote.ashx?t=BBB" in daily
    assert "New (1)" in daily and "CCC 80" in daily
    assert "Need two weekly runs to compare" in body
    assert "Market bias" in body and "SCHD" in body


def test_changes_page_without_runs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "LOG_DIRS", {"weekly": str(tmp_path / "x"), "daily": str(tmp_path / "y")}
    )
    body = client.get("/changes").get_data(as_text=True)
    assert body.count("Need two") == 2 and "Market bias" not in body
