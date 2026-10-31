"""Dashboard routes: run listing, Coiled Cobra setups, breakout KPIs and status filtering."""

from __future__ import annotations

import html
import json
import os
import re
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


@pytest.fixture(autouse=True)
def market_context_silo(tmp_path, monkeypatch):
    """Market-health files live in a tmp silo, empty unless a test writes to it."""
    profile = dict(app_module.config.MARKET_CONTEXT_PROFILE)
    profile |= {"raw_dir": str(tmp_path / "mraw"), "logs_dir": str(tmp_path / "mlogs")}
    monkeypatch.setattr(app_module.config, "MARKET_CONTEXT_PROFILE", profile)
    return profile


def test_routes_registered():
    rules = {rule.rule: rule.endpoint for rule in app.url_map.iter_rules()}
    assert rules.get("/") == "market"
    assert rules.get("/setups") == "setups"
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


def _home_logs(client, tmp_path, monkeypatch):
    """Weekly runs 2026-10-23 and 10-30 plus a daily run; returns the page body."""
    dirs = {"weekly": tmp_path / "w", "daily": tmp_path / "d"}
    for d in dirs.values():
        d.mkdir()
    monkeypatch.setattr(app_module, "LOG_DIRS", {m: str(d) for m, d in dirs.items()})
    monkeypatch.setattr(app_module, "_fetch_live_prices", lambda syms: {s: "N/A" for s in syms})
    lv = {"Close": 48.68, "Resistance": 50.31, "Support": 44.62, "ATR": 1.7, "MTF": "ALIGNED"}
    lv |= {"Extension ATR": 2.4, "RVOL20": 1.8, "SMA50": 46.0}

    def scan(mode, day, rows):
        pd.DataFrame([{"Symbol": s, "Status": st, **lv} for s, st in rows]).to_csv(
            dirs[mode] / f"breakout_setups_{day}.csv", index=False
        )

    scan("weekly", "2026-10-23", [("GMAB", "BREAKOUT_CONFIRMED"), ("GPN", "WATCH")])
    scan(
        "weekly",
        "2026-10-30",
        [
            ("GPN", "BREAKOUT_CONFIRMED"),
            ("GMAB", "BREAKOUT_CONFIRMED"),
            ("WPRE", "PRE_BREAKOUT"),
            ("WTRG", "BREAKOUT_TRIGGERED"),
            ("WTRG2", "BREAKOUT_TRIGGERED"),
        ],
    )
    scan(
        "daily",
        "2026-10-30",
        [
            ("DCONF", "BREAKOUT_CONFIRMED"),
            ("DPRE", "PRE_BREAKOUT"),
            ("VRNS", "WATCH"),
            ("DHR", "FAILED_BREAKOUT"),
        ],
    )
    pd.DataFrame([{"Symbol": "VRNS", "Score": 85, "Grade": "A - Watch", "Close": 48.68}]).to_csv(
        dirs["daily"] / "coiled_cobra_setups_2026-10-30.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "Symbol": "GPN",
                "Check": "ohlc_incoherent",
                "Severity": "error",
                "Date": "2026-10-30",
            },
            {"Symbol": "GMAB", "Check": "big_move", "Severity": "warn", "Date": "2026-10-30"},
        ]
    ).to_csv(dirs["weekly"] / "data_quality_2026-10-30.csv", index=False)
    return client.get("/setups").get_data(as_text=True)


def _cards(body):
    return body.split('tested-section">')[1].split("</section>")[0]


def test_home_cards_are_weekly_confirmed_only(client, tmp_path, monkeypatch):
    body = _home_logs(client, tmp_path, monkeypatch)
    cards = _cards(body)
    assert "run 2026-10-30" in cards
    assert cards.count('class="setup-card"') == 2
    assert cards.index("t=GMAB") < cards.index("t=GPN")  # alphabetical, not ranked
    for sym in ("DCONF", "DPRE", "VRNS", "WPRE", "WTRG", "DHR"):
        assert f"t={sym}" not in cards
    assert cards.count("Unproven: passed old-data test only") == 2
    assert cards.count("Tested as: buy next open, hold 4 weeks, no stop.") == 2


def test_home_cards_have_no_trade_levels_or_rank(client, tmp_path, monkeypatch):
    body = _home_logs(client, tmp_path, monkeypatch)
    cards = _cards(body)
    for gone in ("Trigger", "Stop", "Target", "2R", "3R", "wide stop"):
        assert gone not in cards
    assert "Rank" not in body
    assert "50.31" in cards and "44.62" in cards  # resistance and support as context
    chart = cards.split('class="level-chart"')[1].split("></div>")[0]
    assert 'data-mode="weekly"' in chart and 'data-symbol="GMAB"' in chart
    levels = json.loads(chart.split("data-levels='")[1].split("'")[0])
    assert {line["label"]: line["price"] for line in levels} == {
        "Resistance": 50.31,
        "Support": 44.62,
    }


def test_home_card_flags_continuation_and_data_errors(client, tmp_path, monkeypatch):
    cards = _cards(_home_logs(client, tmp_path, monkeypatch))
    gmab, gpn = cards.split("t=GPN")
    assert "Also CONFIRMED last week" in gmab and "Also CONFIRMED last week" not in gpn
    assert "data: ohlc incoherent" in gpn.split("</header>")[0]
    assert "dq-flag" not in gmab  # warnings are not errors


def test_home_cobra_is_collapsed_and_untested(client, tmp_path, monkeypatch):
    body = _home_logs(client, tmp_path, monkeypatch)
    section = body.split('class="failed-section untested-section"')[1].split("</details>")[0]
    assert '<details class="failed-section untested-section">' in body  # closed by default
    assert "Untested: Coiled Cobra setups" in section and "t=VRNS" in section
    assert "No weekly Cobra setups." in section and ">85<" in section


def test_home_failed_signals_are_counts_only(client, tmp_path, monkeypatch):
    body = _home_logs(client, tmp_path, monkeypatch)
    failed = body.split('class="top-section failed-counts"')[1].split("</section>")[0]
    daily = failed.split("<strong>Daily:</strong>")[1].split("</p>")[0]
    weekly = failed.split("<strong>Weekly:</strong>")[1].split("</p>")[0]
    assert "PRE BREAKOUT 1" in daily and "BREAKOUT TRIGGERED 0" in daily
    assert "BREAKOUT CONFIRMED 1" in daily
    assert "PRE BREAKOUT 1" in weekly and "BREAKOUT TRIGGERED 2" in weekly
    assert "CONFIRMED" not in weekly  # weekly CONFIRMED passed: cards, not a count
    assert "status=BREAKOUT_TRIGGERED" in weekly and "/breakout/weekly/2026-10-30" in weekly
    for sym in ("DCONF", "DPRE", "WPRE", "WTRG"):
        assert f"t={sym}" not in body


def test_home_without_weekly_confirmed(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "LOG_DIRS", {"weekly": str(tmp_path / "w"), "daily": str(tmp_path / "d")}
    )
    body = client.get("/setups").get_data(as_text=True)
    assert "No weekly CONFIRMED setups in the latest run." in body
    assert "headwind" not in body


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


@pytest.mark.parametrize("url", ["/setups", "/changes", "/runs"])
def test_market_strip_on_every_page(client, tmp_path, monkeypatch, url):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    body = client.get(url).get_data(as_text=True)
    strip = body.split('class="market-strip"')[1].split("</div>")[0]
    for sym in ("SPY", "QQQ", "SCHG", "SCHD"):
        assert f"<strong>{sym}</strong>" in strip
    # Fixture: close 760.0 is below SMA50 767.26 but above Support 758.79; no SMA200.
    assert "below 50d" in strip and "200d —" in strip and "Growth vs dividend:" in strip
    assert "closed below support" not in strip


def _write_market_context(profile, rising: set[str]):
    os.makedirs(profile["raw_dir"], exist_ok=True)
    dates = pd.bdate_range(end="2026-10-30", periods=80).strftime("%Y-%m-%d")
    for sym in app_module.config.MARKET_CONTEXT_TICKERS:
        close = [10 + i * 0.1 for i in range(80)]
        if sym not in rising:
            close = close[::-1]
        frame = pd.DataFrame(
            {"Date": dates, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1}
        )
        frame.to_csv(app_module.config.get_raw_path(sym, profile), index=False)


BANNED = (
    "elevated",
    "risk-off",
    "risk-on",
    "bullish",
    "bearish",
    "strong",
    "weak",
    "fading",
    "accelerating",
    "aligned",
    "failed",
    "triggered",
    "confirmed",
)


def _home(client, tmp_path, monkeypatch, market_context_silo, **kw):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _write_market_context(market_context_silo, rising=kw.get("rising", {"VIX", "XLK", "XLE"}))
    return client.get("/").get_data(as_text=True)


def _visible(html):
    return re.sub(r"<[^>]*>", " ", html.replace("<strong>", "").replace("</strong>", ""))


def test_home_has_one_as_of_badge_and_no_notes(client, tmp_path, monkeypatch, market_context_silo):
    body = _home(client, tmp_path, monkeypatch, market_context_silo)
    head = body.split('class="page-head"')[1].split("</div>")[0]
    assert "as of Fri 2026-10-30" in head
    assert body.count("asof-badge") == 1 and "As of" not in body
    for gone in ("not a signal", "Not a signal", "not a call", "advice", "Unvalidated", "Facts,"):
        assert gone not in body
    assert 'class="market-strip"' not in body
    assert "level-chart" not in body and "lightweight-charts" not in body  # no charts


def test_home_text_has_no_verdict_or_status_words(
    client, tmp_path, monkeypatch, market_context_silo
):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BEARISH")
    daily = tmp_path / "d" / "breakout_setups_2026-10-20.csv"
    df = pd.read_csv(daily)
    df["Status"], df["Momentum"], df["MTF"] = "FAILED_BREAKOUT", "ACCELERATING", "ALIGNED"
    df.to_csv(daily, index=False)
    _write_market_context(market_context_silo, rising={"VIX"})
    text = _visible(client.get("/").get_data(as_text=True)).lower()
    assert [w for w in BANNED if w in text] == []


def test_home_one_table_with_returns_and_rs(client, tmp_path, monkeypatch, market_context_silo):
    body = _home(client, tmp_path, monkeypatch, market_context_silo)
    table = body.split('class="table overview-table"')[1].split("</table>")[0]
    headers = re.findall(r"<th>([^<]*)", table)
    assert [h.strip() for h in headers] == [
        "Ticker",
        "Daily",
        "1W",
        "1M",
        "3M",
        "6M",
        "RS vs SPY",
        "",
    ]
    assert body.count("<table") == 1  # Today and the relative view are merged
    for sym in ("SPY", "QQQ", "SCHG", "SCHD"):
        assert f'<td class="ticker-cell">{sym}</td>' in table
    assert "63-day return minus SPY" in table  # the ? tooltip
    assert "Growth vs dividend" in body and "Mixed" in body


def test_home_table_formats_and_warnings(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    daily = tmp_path / "d" / "breakout_setups_2026-10-20.csv"
    df = pd.read_csv(daily)
    df["RVOL20"], df["Close"] = 2.16, 750.0
    df.to_csv(daily, index=False)
    raw = tmp_path / "raw" / "daily"
    raw.mkdir(parents=True)
    dates = pd.bdate_range(end="2026-10-30", periods=140).strftime("%Y-%m-%d")
    for sym in ("SPY", "QQQ"):
        close = [100 + i * (0.5 if sym == "QQQ" else 0.25) for i in range(140)]
        frame = pd.DataFrame(
            {"Date": dates, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1}
        )
        frame.to_csv(raw / f"{sym}_5y_1d.csv", index=False)
    body = client.get("/").get_data(as_text=True)
    table = body.split('class="table overview-table"')[1].split("</table>")[0]
    qqq = table.split('ticker-cell">QQQ')[1].split("</tr>")[0]
    assert '<td class="pos">+0.30%</td>' in qqq  # 2 decimals with a sign
    assert "⚠ closed below support on heavy volume" in qqq
    schg = table.split('ticker-cell">SCHG')[1].split("</tr>")[0]
    assert "⚠ no data" in schg


def test_home_health_tiles_and_sector_tiles(client, tmp_path, monkeypatch, market_context_silo):
    body = _home(client, tmp_path, monkeypatch, market_context_silo)
    vix = body.split('class="metric-tile tile-volatility"')[1].split('class="metric-tile')[0]
    assert '<div class="tile-value">17.90</div>' in vix
    assert "+0.10 since previous close" in vix and "above its 20-day average" in vix
    credit = body.split('class="metric-tile tile-credit"')[1].split('class="metric-tile')[0]
    assert '<div class="tile-value">1.0000</div>' in credit and "over 1 month" in credit
    assert "HYG = high-yield company bonds" in credit
    breadth = body.split('class="metric-tile tile-breadth"')[1].split("sector-tiles")[0]
    assert '<div class="tile-value">2 of 11</div>' in breadth
    tiles = re.findall(r'class="sector-tile (\w+)"\s+title="([^"]+)">(\w+)<', body)
    assert len(tiles) == 11
    by_sym = {sym: (state, title) for state, title, sym in tiles}
    assert by_sym["XLK"][0] == "up" and by_sym["XLK"][1].startswith("Technology: close ")
    assert by_sym["XLU"] == ("down", by_sym["XLU"][1]) and by_sym["XLU"][1].startswith("Utilities:")
    health = body.split('class="health"')[1].split('class="sector-tiles"')[0]
    assert 'class="pos"' not in health and 'class="neg"' not in health  # tiles stay neutral


def test_home_health_without_data(client, tmp_path, monkeypatch):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    body = client.get("/").get_data(as_text=True)
    sec = body.split('class="health"')[1].split("</section>")[0]
    assert sec.count("No data") == 3 and "⚠ VIX missing" in sec and "⚠ XLK missing" in sec
    assert sec.count('class="sector-tile nodata"') == 11


def test_home_cards_have_position_bar_and_smas(client, tmp_path, monkeypatch, market_context_silo):
    body = _home(client, tmp_path, monkeypatch, market_context_silo)
    spy = body.split("quote.ashx?t=SPY")[1].split("</article>")[0]
    # Fixture: Support 758.79, Resistance 781.62, Close 760.0 -> 5.3% along the bar.
    assert 'style="left: 5.3%"' in spy
    assert "Support</span> 758.79" in spy and "Resistance</span> 781.62" in spy
    assert "Close</span> 760.00" in spy
    assert (
        "SMA 20</span> 770.62" in spy
        and "SMA 50</span> 767.26" in spy
        and "SMA 200</span> —" in spy
    )
    assert "⚠ stale" in spy.split("</header>")[0]  # daily row is a session behind
    assert "states-table" not in body and "weekly trend" not in body
    assert "No support / resistance" in body.split("quote.ashx?t=SCHG")[1].split("</article>")[0]


def test_home_without_runs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "LOG_DIRS", {"weekly": str(tmp_path / "x"), "daily": str(tmp_path / "y")}
    )
    body = client.get("/").get_data(as_text=True)
    assert body.count("No support / resistance in the latest daily scan.") == 4


def test_old_market_link_redirects_home(client):
    response = client.get("/market")
    assert response.status_code == 301
    location = response.headers["Location"]
    assert location.endswith("/") and "/market" not in location


def test_nav_order_market_changes_setups_runs(client, tmp_path, monkeypatch):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    nav = client.get("/runs").get_data(as_text=True).split('class="top-nav"')[1].split("</nav>")[0]
    labels = [part.split("</a>")[0].split(">")[-1] for part in nav.split('class="nav-link')[1:]]
    assert labels == ["Market", "Changes", "Setups (unproven)", "Runs"]
    assert 'href="/setups"' in nav


def test_market_strip_warns_on_heavy_volume_support_break(client, tmp_path, monkeypatch):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    daily = tmp_path / "d" / "breakout_setups_2026-10-20.csv"
    df = pd.read_csv(daily)
    df["RVOL20"] = 2.16
    df.loc[df["Symbol"] == "SPY", "RVOL20"] = 1.2  # SPY: below support too, light volume
    df["Close"] = 750.0
    df.to_csv(daily, index=False)
    strip = client.get("/runs").get_data(as_text=True).split('class="market-strip"')[1]
    strip = strip.split("</div>")[0]
    qqq = strip.split("<strong>QQQ</strong>")[1].split("<strong>SCHG</strong>")[0]
    assert "closed below support on heavy volume" in qqq
    assert "closed below support" not in strip.split("<strong>QQQ</strong>")[0]


def test_market_strip_has_no_bias_pill(client, tmp_path, monkeypatch):
    """The strip shows facts only: no bias pill, even when SPY's bias is SHORT."""
    _market_logs(tmp_path, monkeypatch, weekly_trend="BEARISH")
    strip = client.get("/runs").get_data(as_text=True).split('class="market-strip"')[1]
    assert "bias-" not in strip.split("</div>")[0]


@pytest.mark.parametrize("trend", ["BEARISH", "BULLISH"])
def test_setups_page_always_shows_spy_qqq_facts(client, tmp_path, monkeypatch, trend):
    _market_logs(tmp_path, monkeypatch, weekly_trend=trend)
    body = client.get("/setups").get_data(as_text=True)
    box = body.split('class="market-facts"')[1].split("</div>")[0]
    assert f"<strong>SPY:</strong> weekly trend {trend} · weekly MTF PARTIAL" in box
    assert "<strong>QQQ:</strong> no weekly scan row · daily close below" in box
    assert "SCHG" not in box and 'href="/"' in box
    assert "headwind" not in body and "bias" not in body.lower()


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


def _changes_logs(tmp_path, monkeypatch):
    dirs = {"weekly": tmp_path / "w", "daily": tmp_path / "d"}
    for d in dirs.values():
        d.mkdir()
    monkeypatch.setattr(app_module, "LOG_DIRS", {m: str(d) for m, d in dirs.items()})
    profiles = {m: dict(p) for m, p in app_module.config.TIMEFRAME_PROFILES.items()}
    for m in profiles:
        profiles[m]["raw_dir"] = str(tmp_path / "raw" / m)  # no raw files
    monkeypatch.setattr(app_module.config, "TIMEFRAME_PROFILES", profiles)
    etf = {"SMA50": 100.0, "SMA200": 90.0, "Support": 95.0, "RVOL20": 1.0}

    def scan(mode, day, rows):
        pd.DataFrame(rows).to_csv(dirs[mode] / f"breakout_setups_{day}.csv", index=False)

    def st(sym, status, **kw):
        return {"Symbol": sym, "Status": status, "Breakout Readiness": 50} | kw

    scan(
        "daily",
        "2026-10-29",
        [
            st("AAA", "WATCH"),
            st("BBB", "BREAKOUT_TRIGGERED"),
            st("LEFTD", "WATCH"),
            st("SPY", "WATCH", Close=105.0, **etf),
            st("QQQ", "WATCH", Close=105.0, **etf),
        ],
    )
    scan(
        "daily",
        "2026-10-30",
        [
            st("AAA", "PRE_BREAKOUT"),
            st("BBB", "FAILED_BREAKOUT"),
            st("NEWD", "WATCH"),
            st("SPY", "WATCH", Close=106.0, **etf),  # no change: not listed
            st("QQQ", "WATCH", Close=94.0, **(etf | {"RVOL20": 2.2})),
        ],
    )
    pd.DataFrame([{"Symbol": "CCC", "Score": 80}]).to_csv(
        dirs["daily"] / "coiled_cobra_setups_2026-10-30.csv", index=False
    )
    scan(
        "weekly",
        "2026-10-23",
        [st("FIRST", "BREAKOUT_TRIGGERED"), st("GMAB", "BREAKOUT_CONFIRMED"), st("LEFTW", "WATCH")],
    )
    scan(
        "weekly",
        "2026-10-30",
        [
            st("FIRST", "BREAKOUT_CONFIRMED"),
            st("GMAB", "BREAKOUT_CONFIRMED"),
            st("GPN", "BREAKOUT_CONFIRMED"),
        ],
    )


def _section(body, cls):
    return body.split(f'class="top-section {cls}"')[1].split("</section>")[0]


def test_changes_page_order_and_dates(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    body = client.get("/changes").get_data(as_text=True)
    top = body.split('class="compare-dates"')[1].split("</p>")[0]
    assert "2026-10-29 → 2026-10-30" in top and "2026-10-23 → 2026-10-30" in top
    order = ["market-changes", "new-confirmed", "universe-changes", "everything-else"]
    assert [body.index(c) for c in order] == sorted(body.index(c) for c in order)
    assert "Market bias" not in body  # the old bias table is gone


def test_changes_market_section_lists_only_new_facts(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    market = _section(client.get("/changes").get_data(as_text=True), "market-changes")
    qqq = market.split("<strong>QQQ:</strong>")[1].split("</li>")[0]
    assert "closed below its 50-day average (was above)" in qqq
    assert "newly closed below support on heavy volume" in qqq
    assert "SPY:" not in market and "Growth vs dividend unchanged" in market


def test_changes_new_weekly_confirmed_split(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    sec = _section(client.get("/changes").get_data(as_text=True), "new-confirmed")
    new, unscanned = sec.split("Confirmed now, but not scanned last week (")
    assert "New this week (1)" in new and "t=FIRST" in new
    assert "Unproven: passed old-data test only" in new
    assert unscanned.startswith("1)") and "t=GPN" in unscanned
    assert "so we can't tell if this is the first week" in html.unescape(unscanned)
    assert "GMAB" not in sec  # CONFIRMED both weeks: a continuation


def test_changes_stock_list_entered_and_left(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    sec = _section(client.get("/changes").get_data(as_text=True), "universe-changes")
    weekly, daily = sec.split("⚡ Daily")
    assert "Entered (1)</th><td>GPN" in weekly and "Left (1)</th><td>LEFTW" in weekly
    assert "Entered (1)</th><td>NEWD" in daily and "Left (1)</th><td>LEFTD" in daily


def test_changes_everything_else_is_collapsed_counts(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    body = client.get("/changes").get_data(as_text=True)
    assert '<details class="failed-section everything-else">' in body  # closed
    rest = body.split("everything-else")[1]
    daily = rest.split("⚡ Daily")[1]
    assert "Moved toward a breakout 1 · Failed breakouts 1" in daily
    assert "Cobra setups new 1 / dropped 0" in daily
    assert "t=AAA" not in body and "t=BBB" not in body  # counts only
    assert 'href="/changes?show=all"' in rest


def test_changes_full_list(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    body = client.get("/changes?show=all").get_data(as_text=True)
    assert '<details class="failed-section everything-else" open>' in body
    daily = body.split("everything-else")[1].split("⚡ Daily")[1]
    assert "Moved toward a breakout (1)" in daily and "quote.ashx?t=AAA" in daily
    assert "Failed breakouts (1)" in daily and "quote.ashx?t=BBB" in daily
    assert "New (1)" in daily and "CCC 80" in daily
    assert "Change</th>" not in body  # no Readiness change column


def test_changes_page_without_runs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "LOG_DIRS", {"weekly": str(tmp_path / "x"), "daily": str(tmp_path / "y")}
    )
    body = client.get("/changes").get_data(as_text=True)
    assert "Need two daily runs to compare." in body and "Need two weekly runs to compare." in body
    assert "Comparing <strong>daily</strong> — → —" in body
