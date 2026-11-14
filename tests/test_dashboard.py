"""Dashboard routes: run listing, Coiled Cobra setups, breakout KPIs and status filtering."""

from __future__ import annotations

import io
import json
import os
import re
from datetime import date, datetime

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
    assert rules.get("/runs/ticker") == "ticker_search"
    assert rules.get("/runs/ticker/<symbol>") == "ticker_runs"


def test_index_links_each_scanner_that_ran(client, breakout_logs, tmp_path):
    pd.DataFrame([{"Symbol": "AAA", "Score": 90, "Close": 1.0}]).to_csv(
        tmp_path / "daily" / "coiled_cobra_setups_2026-09-05.csv", index=False
    )
    body = client.get("/runs").get_data(as_text=True)
    assert "/breakout/daily/2026-09-04" in body and "/cobra/daily/2026-09-04" not in body
    assert "/cobra/daily/2026-09-05" in body and "/breakout/daily/2026-09-05" not in body
    assert body.index("2026-09-05") < body.index("2026-09-04")  # newest first
    assert "Sat 2026-09-05" in body and "Fri 2026-09-04" in body  # weekday on each date
    assert body.count("asof-badge") == 1 and "as of Sat 2026-09-05" in body
    assert "<p>Each run date" not in body  # the intro is a ? tooltip now


def test_runs_search_box_suggests_latest_symbols(client, breakout_logs):
    body = client.get("/runs").get_data(as_text=True)
    assert 'action="/runs/ticker"' in body and 'list="symbols"' in body
    for sym in ("AAA", "BBB", "CCC", "DDD"):
        assert f'<option value="{sym}">' in body


@pytest.mark.parametrize(
    ("query", "target"),
    [
        ("?symbol=+aaa+", "/runs/ticker/AAA"),
        ("?symbol=", "/runs"),
        ("?symbol=a/b", "/runs"),
        ("", "/runs"),
    ],
)
def test_ticker_search_redirects(client, breakout_logs, query, target):
    response = client.get(f"/runs/ticker{query}")
    assert response.status_code == 302 and response.headers["Location"].endswith(target)


def test_ticker_runs_one_row_per_run(client, breakout_logs, tmp_path):
    _write_scan(tmp_path / "daily", "2026-09-05")
    _write_scan(tmp_path / "weekly", "2026-09-04")
    pd.DataFrame([{"Symbol": "AAA", "Score": 88, "Grade": "A - Watch", "Close": 100.0}]).to_csv(
        tmp_path / "daily" / "coiled_cobra_setups_2026-09-05.csv", index=False
    )
    response = client.get("/runs/ticker/AAA")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert body.count("asof-badge") == 1 and "as of Sat 2026-09-05" in body
    assert "2 of 2" in body and "1 of 1" in body
    assert "Sat 2026-09-05" in body and "Fri 2026-09-04" in body
    daily = body.split("<h2>Daily</h2>")[1]
    assert daily.index("Sat 2026-09-05") < daily.index("Fri 2026-09-04")  # newest first
    assert "/cobra/daily/2026-09-05" in body and "/cobra/daily/2026-09-04" not in body
    assert "/breakout/weekly/2026-09-04?show=all" in body
    assert "PRE_BREAKOUT" in body and "88" in body and "100.00" in body
    tile = body.split('class="live-tile"')[1].split("</h1>")[0] if "live-tile" in body else ""
    assert "95.00" in tile and '<span class="neg">-5.00%</span>' in tile  # vs daily close 100
    assert "daily run close 100.00 on Sat 2026-09-05" in tile
    rows = body.split("<tbody>")[1]
    assert "95.00" not in rows  # the table stays on run closes
    nav = body.split('class="top-nav"')[1].split("</nav>")[0]
    assert 'nav-link active" href="/runs"' in nav


def test_ticker_runs_sparkline_closes_against_resistance(
    client, breakout_logs, tmp_path, monkeypatch
):
    profiles = {m: dict(p) for m, p in app_module.config.TIMEFRAME_PROFILES.items()}
    for mode in profiles:
        profiles[mode]["raw_dir"] = str(tmp_path / "raw" / mode)
    monkeypatch.setattr(app_module.config, "TIMEFRAME_PROFILES", profiles)
    raw_dir = tmp_path / "raw" / "daily"
    raw_dir.mkdir(parents=True)
    days = pd.bdate_range(end="2026-09-04", periods=12)
    closes = [90.0 + i for i in range(12)]  # 90 .. 101, last close 101
    pd.DataFrame(
        {"Date": days, "Open": closes, "High": closes, "Low": closes, "Close": closes,
         "Volume": 1e6}
    ).to_csv(raw_dir / "AAA_5y_1d.csv", index=False)  # fmt: skip
    scan = tmp_path / "daily" / "breakout_setups_2026-09-04.csv"
    df = pd.read_csv(scan).assign(Resistance=[100.0, 60.0, 30.0, 12.0])
    df.to_csv(scan, index=False)
    _write_scan(tmp_path / "weekly", "2026-09-04")  # no weekly raw file: a dash
    body = client.get("/runs/ticker/AAA").get_data(as_text=True)
    daily = body.split("<h2>Daily</h2>")[1]
    weekly = body.split("<h2>Weekly</h2>")[1].split("<h2>")[0]
    assert daily.count('<svg class="spark"') == 1 and "spark-last up" in daily
    assert 'class="spark-level"' in daily and "<polyline" in daily
    assert "Last 10 closes to Fri 2026-09-04, last 101.00, +1.00% vs Resistance 100.00" in daily
    assert "<svg" not in weekly and 'class="spark-cell">—' in weekly
    csv = client.get("/runs/ticker/AAA/history.csv").get_data(as_text=True)
    assert "Resistance" not in csv.splitlines()[0]  # page only, the download is unchanged


def _badge(body):
    """The badge's visible text, without its title tooltip."""
    return body.split('class="tf-badge"')[1].split('">', 1)[1].split("</span>\n")[0]


def test_ticker_runs_timeframe_badge_same_status(client, breakout_logs, tmp_path):
    _write_scan(tmp_path / "weekly", "2026-09-04")
    body = client.get("/runs/ticker/AAA").get_data(as_text=True)
    badge = _badge(body)
    assert "= PRE_BREAKOUT" in badge and "≠" not in badge
    assert "Readiness W 83 · D 83 (+0)" in badge and "Cobra neither" in badge
    assert "⚠" not in body


def test_ticker_runs_timeframe_badge_differs_and_stale(client, breakout_logs, tmp_path):
    weekly = tmp_path / "weekly"
    weekly.mkdir()
    pd.DataFrame(
        [{"Symbol": "AAA", "Status": "FAILED_BREAKOUT", "Close": 99.0, "Breakout Readiness": 40}]
    ).to_csv(weekly / "breakout_setups_2026-08-28.csv", index=False)
    pd.DataFrame([{"Symbol": "ZZZ", "Status": "WATCH", "Close": 1.0}]).to_csv(
        weekly / "breakout_setups_2026-09-04.csv", index=False
    )  # newest weekly run has no AAA row
    pd.DataFrame([{"Symbol": "AAA", "Score": 88, "Grade": "A", "Close": 100.0}]).to_csv(
        tmp_path / "daily" / "coiled_cobra_setups_2026-09-04.csv", index=False
    )
    body = client.get("/runs/ticker/AAA").get_data(as_text=True)
    badge = _badge(body)
    assert "≠ W: FAILED_BREAKOUT · D: PRE_BREAKOUT" in badge
    assert "Readiness W 40 · D 83 (-43)" in badge and "Cobra daily only (D 88)" in badge
    assert "⚠ weekly row from Fri 2026-08-28" in body and "⚠ daily" not in body


def test_ticker_runs_live_tile_no_quote_and_weekly_fallback(
    client, breakout_logs, tmp_path, monkeypatch
):
    body = client.get("/runs/ticker/DDD").get_data(as_text=True)  # DDD quote is N/A
    assert "⚠ no quote" in body and "live-tile" not in body
    weekly = tmp_path / "weekly"
    weekly.mkdir()
    pd.DataFrame([{"Symbol": "WWW", "Status": "WATCH", "Close": 50.0}]).to_csv(
        weekly / "breakout_setups_2026-09-04.csv", index=False
    )
    monkeypatch.setattr(app_module, "_fetch_live_prices", lambda syms: {"WWW": 55.0})
    body = client.get("/runs/ticker/WWW").get_data(as_text=True)
    assert '<span class="pos">+10.00%</span>' in body and "weekly run close 50.00" in body


def test_ticker_runs_unknown_lowercase_and_unsafe(client, breakout_logs):
    body = client.get("/runs/ticker/ZZZ").get_data(as_text=True)
    assert "No runs include ZZZ." in body and "asof-badge" not in body and "live-tile" not in body
    lower = client.get("/runs/ticker/aaa")
    assert lower.status_code == 302 and lower.headers["Location"].endswith("/runs/ticker/AAA")
    assert client.get("/runs/ticker/A%20B").status_code == 404


def test_breakout_view_kpis(client, breakout_logs):
    response = client.get("/breakout/daily/2026-09-04")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    # Setups = 4, actionable (>=70) = 2, failed = 1, median = 66.0.
    assert "Setups" in body
    assert ">4<" in body
    assert "66.0" in body  # median of [83, 72, 60, 25]
    head = body.split('class="page-head"')[1].split("</div>")[0]
    assert "daily run 2026-09-04" in head and "none is validated as an entry signal" in head
    assert "Descriptive only" not in body and "scan-note" not in body
    assert "kpi-value pos" not in body and "kpi-value neg" not in body  # counts, not up / down
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
    assert ths.index("vs Run Close") == ths.index("Close") + 2
    assert "this is the move since that run" in header  # ? tooltip on the header
    # close > live -> red; close < live -> green; equal -> neutral; missing -> ⚠ no quote.
    assert 'live-price-cell live-below-close">95.00' in body
    assert 'live-price-cell live-above-close">55.50' in body
    assert 'live-price-cell live-flat">20.00' in body
    assert '<span class="warn plain">⚠ no quote</span>' in body
    assert "$" not in body.split("<tbody>")[1]


def test_cobra_view_orders_by_score_with_live_price_and_no_row_highlight(
    client, tmp_path, monkeypatch
):
    modes = {"weekly": str(tmp_path / "weekly"), "daily": str(tmp_path / "daily")}
    monkeypatch.setattr(app_module, "LOG_DIRS", modes)
    (tmp_path / "weekly").mkdir()
    # AAA live 95 < 100 (red), BBB live 55.5 vs 50 = +11% (green, no row highlight),
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
    assert ths.index("vs Run Close") == ths.index("Close") + 2
    assert 'live-price-cell live-below-close">95.00' in body
    assert 'live-price-cell live-above-close">55.50' in body
    assert 'live-price-cell live-flat">20.00' in body
    assert "live-surge" not in body
    assert ">+11.00%<" in body and ">-5.00%<" in body  # BBB, AAA
    head = body.split('class="page-head"')[1].split("</div>")[0]
    assert "weekly run 2026-09-04" in head and "Weekly: never backtested" in head


def test_run_grids_drop_asof_column_and_flag_only_lagging_rows(client, tmp_path, monkeypatch):
    modes = {"weekly": str(tmp_path / "weekly"), "daily": str(tmp_path / "daily")}
    monkeypatch.setattr(app_module, "LOG_DIRS", modes)
    (tmp_path / "weekly").mkdir()
    # Weekly bars are dated by week start (Monday); the run file by its Friday.
    rows = [
        {"Symbol": "AAA", "AsOf Date": "2026-11-02", "Score": 90, "Close": 100.0},
        {"Symbol": "BBB", "AsOf Date": "2026-10-26", "Score": 80, "Close": 50.0},
    ]
    pd.DataFrame(rows).to_csv(
        tmp_path / "weekly" / "coiled_cobra_setups_2026-11-06.csv", index=False
    )
    statuses = [
        {"Status": "PRE_BREAKOUT", "Breakout Readiness": 80},
        {"Status": "FAILED_BREAKOUT", "Breakout Readiness": 40},
    ]
    pd.DataFrame([r | st for r, st in zip(rows, statuses, strict=True)]).to_csv(
        tmp_path / "weekly" / "breakout_setups_2026-11-06.csv", index=False
    )
    for url in ("/cobra/weekly/2026-11-06", "/breakout/weekly/2026-11-06"):
        body = client.get(url).get_data(as_text=True)
        assert "AsOf Date" not in body
        assert body.count("⚠ last bar") == 1  # BBB only; the run-date mismatch is not flagged
        assert "⚠ last bar 2026-10-26" in body
        bbb = body.split(">BBB</a>")[1].split("</td>")[0]
        assert "⚠ last bar 2026-10-26" in bbb and "Newest bar in this run: 2026-11-02" in bbb


def test_vs_close_cell_is_signed_percent_from_run_close():
    cell = app_module._vs_close_cell
    assert cell(12.0, 10.0) == '<span class="live-price-cell live-above-close">+20.00%</span>'
    assert cell(95.0, 100.0) == '<span class="live-price-cell live-below-close">-5.00%</span>'
    assert cell(20.0, 20.0) == '<span class="live-price-cell live-flat">+0.00%</span>'
    for live, close in [("N/A", 10.0), (None, 10.0), (float("nan"), 10.0), (12.0, float("nan"))]:
        assert cell(live, close) == ""
    assert cell(12.0, 0.0) == ""


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
    assert cards.count('class="setup-card"') == 2
    assert cards.index("t=GMAB") < cards.index("t=GPN")  # alphabetical, not ranked
    for sym in ("DCONF", "DPRE", "VRNS", "WPRE", "WTRG", "DHR"):
        assert f"t={sym}" not in cards
    # Trust notes are one ? tooltip on the heading, not a pill or note per card.
    for gone in ("Unproven: passed old-data test only", "status-pill", "advice"):
        assert gone not in body
    assert "Tested as:" not in cards.split("</h2>")[-1]
    heading = body.split('tested-section">')[1].split("</h2>")[0]
    assert (
        'class="tip"' in heading and "Tested as: buy next open, hold 4 weeks, no stop." in heading
    )
    assert "Tested on the old scan list (2026-10-27 snapshot" in heading  # not this universe
    head = body.split('class="page-head"')[1].split("</div>")[0]
    assert body.count("asof-badge") == 1 and "Fri 2026-10-30" in head


def test_home_cards_have_no_trade_levels_or_rank(client, tmp_path, monkeypatch):
    body = _home_logs(client, tmp_path, monkeypatch)
    cards = _cards(body)
    for gone in ("Trigger", "Stop", "Target", "2R", "3R", "wide stop"):
        assert gone not in cards
    assert "Rank" not in body
    gmab = cards.split("t=GPN")[0]
    assert "Support</span> 44.62" in gmab and "Resistance</span> 50.31" in gmab  # position bar
    assert "pos-dot" in gmab and "level-chart" not in body and "lightweight-charts" not in body
    assert "Live" not in cards  # run closes only


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
    assert "Coiled Cobra setups" in section
    assert "Daily: failed the in-sample test" in section and "Weekly: never backtested" in section
    assert "t=VRNS" in section and "Untested:" not in section
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
    assert 'below <span class="label">50d</span>' in strip
    assert '<span class="label">200d</span> —' in strip
    assert '<span class="label">Growth vs dividend</span> Mixed' in strip
    assert "muted" not in strip and "leading" not in strip  # white facts, grey labels
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


def _snapshot(symbols, fetched="2026-11-02T10:25:00-05:00", day="2026-11-02", last=50.0):
    quotes = {
        s: {"last": last, "time": f"{day}T10:25:00-05:00", "fetched_at": fetched} for s in symbols
    }
    path = app_module.config.MARKET_SNAPSHOT_PATH
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump({"fetched_at": fetched, "quotes": quotes}, handle)


def _now(monkeypatch, text):
    when = datetime.fromisoformat(text)
    monkeypatch.setattr(app_module.market_snapshot, "now_et", lambda: when)


def test_home_live_quotes_overlay_table_and_tiles(
    client, tmp_path, monkeypatch, market_context_silo
):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _write_market_context(market_context_silo, rising={"XLK", "SPY", "QQQ", "IWM", "RSP"})
    _snapshot(("SPY", "QQQ", "IWM", "VIX", "XLK"))
    _now(monkeypatch, "2026-11-02T10:30:00-05:00")
    body = client.get("/").get_data(as_text=True)
    head = body.split('class="page-head"')[1].split("</div>")[0]
    assert "as of 10:25 ET · last close Fri 2026-10-30" in head
    assert "min old" not in head
    assert body.count("asof-badge") == 1
    assert '<meta http-equiv="refresh" content="300">' in body
    table = body.split('class="table overview-table"')[1].split("</table>")[0]
    assert "+179.33%" in table  # the quote 50.00 vs the last close 17.90
    assert '<td class="num">50.00</td>' in table  # Last
    rsp = table.split('ticker-cell">RSP')[1]
    assert "⚠ no quote" in rsp and "⚠ no quote" not in table.split('ticker-cell">RSP')[0]
    assert "50.00" in body.split("tile-volatility")[1]  # VIX moved to the quote
    assert "last 50.00" in body  # sector hover names the live price
    levels = body.split('class="levels"')[1].split("</section>")[0]
    assert '<td class="num">50.00</td>' in levels  # Last, against levels from closes


def test_home_table_survives_the_daily_raw_wipe(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")  # no daily raw files at all
    _write_market_context(market_context_silo, rising={"SPY", "QQQ", "IWM", "RSP"})
    table = client.get("/").get_data(as_text=True).split('class="table overview-table"')[1]
    table = table.split("</table>")[0]
    assert "no data" not in table
    assert "+0.56%" in table  # 17.90 vs 17.80, from the market folder


def test_home_warns_when_quotes_are_old(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _write_market_context(market_context_silo, rising={"XLK"})
    _snapshot(("VIX",))
    _now(monkeypatch, "2026-11-02T11:00:00-05:00")
    head = client.get("/").get_data(as_text=True).split('class="page-head"')[1].split("</div>")[0]
    assert "⚠ quotes 35 min old" in head


def test_home_rolls_over_once_the_close_is_saved(
    client, tmp_path, monkeypatch, market_context_silo
):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _write_market_context(market_context_silo, rising={"XLK"})
    # Quotes from the same day as the newest close: closes win, and it is 20:00, no refresh.
    _snapshot(("VIX", "XLK"), fetched="2026-10-30T16:15:00-04:00", day="2026-10-30")
    _now(monkeypatch, "2026-10-30T20:00:00-04:00")
    body = client.get("/").get_data(as_text=True)
    head = body.split('class="page-head"')[1].split("</div>")[0]
    assert "as of Fri 2026-10-30" in head and " ET" not in head
    assert 'http-equiv="refresh"' not in body
    assert "⚠ no quote" not in body


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
    headers = re.findall(r"<th[^>]*>([^<]*)", table)
    assert [h.strip() for h in headers] == [
        "Ticker",
        "Last",
        "Day",
        "Move",
        "vs 20d",
        "vs 50d",
        "vs 200d",
        "Off high",
        "1W",
        "1M",
        "RS vs SPY",
        "",
    ]
    assert body.count("<table") == 2  # the merged market table, then the levels table
    for sym in ("SPY", "QQQ", "IWM", "RSP"):
        assert f'<td class="ticker-cell">{sym}</td>' in table
    assert "SCHG<" not in table and "SCHD<" not in table  # they feed the label under it
    assert 'class="fact-line"' not in body  # nothing to flag
    assert "63-day return minus SPY" in table  # the ? tooltip
    assert "Growth vs dividend" in body and "Mixed" in body


def test_home_table_formats_and_warnings(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    daily = tmp_path / "d" / "breakout_setups_2026-10-20.csv"
    df = pd.read_csv(daily)
    df["RVOL20"], df["Close"], df["AsOf Date"] = 2.16, 750.0, "2026-10-30"
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
    assert '<td class="num pos">+0.30%</td>' in qqq  # 2 decimals, sign, right-aligned
    assert '<span class="warn plain"' in qqq and "⚠ below support · heavy vol" in qqq
    facts = body.split('class="fact-line"')[1].split("</p>")[0]
    assert " ".join(facts.split()) == ">SPY, QQQ closed below support on heavy volume"
    iwm = table.split('ticker-cell">IWM')[1].split("</tr>")[0]
    assert "⚠ no data" in iwm


def test_home_health_tiles_and_sector_tiles(client, tmp_path, monkeypatch, market_context_silo):
    body = _home(client, tmp_path, monkeypatch, market_context_silo)
    vix = body.split('class="metric-tile tile-volatility"')[1].split('class="metric-tile')[0]
    assert '<div class="tile-value">17.90</div>' in vix
    assert "+0.10 since previous close" in vix and "above its 20-day average" in vix
    credit = body.split('class="metric-tile tile-credit"')[1].split('class="metric-tile')[0]
    assert '<div class="tile-value">1.0000</div>' in credit and "over 1 month" in credit
    assert "HYG = high-yield company bonds" in credit
    breadth = body.split('class="metric-tile tile-breadth"')[1].split("</section>")[0]
    assert '<div class="tile-value">2 of 11 <span class="tile-unit">' in breadth
    tiles = re.findall(
        r'class="sector-tile (\w+)" title="([^"]+)"><b>(\w+)</b><small>([^<]+)</small>'
        r"<small>([^<]+)</small><small>50d ([^<]+)</small>",
        breadth,  # inside the Breadth tile
    )
    assert len(tiles) == 11
    by_sym = {sym: (state, title, short, day) for state, title, sym, short, day, _ in tiles}
    assert by_sym["XLK"][:3] == ("up", by_sym["XLK"][1], "Tech")
    assert by_sym["XLK"][1].startswith("Technology: close ")
    assert by_sym["XLU"][0] == "down" and by_sym["XLU"][2] == "Utilities"
    assert by_sym["XLE"][2] == "Energy" and by_sym["XLE"][3].startswith("+")
    assert all(t[5].endswith("%") for t in tiles)
    # Sorted by today's % change, highest first: the two rising funds lead.
    assert {t[2] for t in tiles[:2]} == {"XLE", "XLK"}
    days = [float(t[4].rstrip("%")) for t in tiles]
    assert days == sorted(days, reverse=True)
    vix_credit = body.split('class="health"')[1].split('class="metric-tile tile-breadth"')[0]
    assert 'class="pos"' not in vix_credit and 'class="neg"' not in vix_credit  # neutral
    assert "since previous close" in body.split('tile-rates"')[1].split('class="metric-tile')[0]
    assert "since previous close" in body.split('tile-dollar"')[1].split('class="metric-tile')[0]
    curve = body.split('tile-vol-curve"')[1].split('class="metric-tile')[0]
    assert "at previous close" in curve


def test_home_health_without_data(client, tmp_path, monkeypatch):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    body = client.get("/").get_data(as_text=True)
    sec = body.split('class="health"')[1].split("</section>")[0]
    assert sec.count("No data") == 7 and "⚠ VIX missing" in sec and "⚠ XLK missing" in sec
    assert "⚠ VIX3M missing" in sec and "⚠ TNX missing" in sec and "⚠ DXY missing" in sec
    assert sec.count('class="sector-tile nodata"') == 11


def test_position_bar_macro_outside_the_range():
    macro = app.jinja_env.get_template("_macros.html").module.position_bar
    below = macro(
        app_module.playbook.position_bar({"Close": 750.0, "Support": 758.79, "Resistance": 781.62})
    )
    assert 'class="pos-dot pos-dot-out" style="left: -14px"' in below
    assert "below support by 1.16%" in below and '<div class="pos-status neg">' in below
    above = macro(
        app_module.playbook.position_bar({"Close": 800.0, "Support": 758.79, "Resistance": 781.62})
    )
    assert 'style="left: calc(100% + 14px)"' in above
    assert '<div class="pos-status pos">' in above and "above resistance by 2.35%" in above


def test_home_without_runs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "LOG_DIRS", {"weekly": str(tmp_path / "x"), "daily": str(tmp_path / "y")}
    )
    response = client.get("/")
    assert response.status_code == 200
    assert 'class="levels"' in response.get_data(as_text=True)


def _bars_file(profile, symbol, n=260, last="2026-10-30"):
    """Rising bars with High = Close + 1 and Low = Close - 1."""
    os.makedirs(profile["raw_dir"], exist_ok=True)
    close = [100.0 + i for i in range(n)]
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range(end=last, periods=n).strftime("%Y-%m-%d"),
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": 1,
        }
    )
    frame.to_csv(app_module.config.get_raw_path(symbol, profile), index=False)


def test_home_levels_table_and_no_cards(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    for sym in ("SPY", "QQQ", "IWM", "RSP"):
        _bars_file(market_context_silo, sym)
    body = client.get("/").get_data(as_text=True)
    assert "etf-card" not in body and "pos-bar" not in body and "<article" not in body
    levels = body.split('class="levels"')[1].split("</section>")[0]
    spy = levels.split("t=SPY")[1].split("</tr>")[0]
    cells = re.findall(r'<td class="num ?(\w*)">([^<]*)</td>', spy)
    # Newest close 359: its bar is 360 / 358. The 20 bars before it close 339..358.
    assert cells == [
        ("", "359.00"),  # Last
        ("neg", "360.00"),  # prior high: Last is below it
        ("pos", "358.00"),  # prior low
        ("", "359.00"),  # resistance, equal to Last
        ("pos", "338.00"),  # support
        ("pos", "349.50"),  # SMA 20
        ("pos", "334.50"),  # SMA 50
        ("pos", "259.50"),  # SMA 200
    ]
    assert "RSP" in levels  # covered even though the scanner does not scan it


def test_home_table_shows_bar_quality_errors(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _bars_file(market_context_silo, "SPY")
    os.makedirs(market_context_silo["logs_dir"], exist_ok=True)
    report = os.path.join(market_context_silo["logs_dir"], "data_quality_2026-10-30.csv")
    pd.DataFrame(
        [
            {
                "Symbol": "SPY",
                "Check": "ohlc_incoherent",
                "Severity": "error",
                "Date": "2026-10-30",
                "Detail": "x",
            }
        ]
    ).to_csv(report, index=False)
    table = client.get("/").get_data(as_text=True).split('class="table overview-table"')[1]
    assert "⚠ ohlc incoherent" in table.split('ticker-cell">SPY')[1].split("</tr>")[0]


def test_home_hides_a_support_break_from_an_older_scan(
    client, tmp_path, monkeypatch, market_context_silo
):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    daily = tmp_path / "d" / "breakout_setups_2026-10-20.csv"
    df = pd.read_csv(daily)
    df["RVOL20"], df["Close"], df["AsOf Date"] = 2.16, 750.0, "2026-10-29"
    df.to_csv(daily, index=False)
    _bars_file(market_context_silo, "SPY")  # closes through 2026-10-30: newer than the scan
    body = client.get("/").get_data(as_text=True)
    assert "below support" not in body and 'class="fact-line"' not in body


def test_home_flags_tiles_without_a_current_quote(
    client, tmp_path, monkeypatch, market_context_silo
):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _write_market_context(market_context_silo, rising={"XLK"})
    _snapshot(("SPY", "HYG", "TNX"))
    path = app_module.config.MARKET_SNAPSHOT_PATH
    with open(path) as handle:
        payload = json.load(handle)
    payload["quotes"]["TNX"]["fetched_at"] = "2026-11-02T09:55:00-05:00"
    with open(path, "w") as handle:
        json.dump(payload, handle)
    _now(monkeypatch, "2026-11-02T10:30:00-05:00")
    body = client.get("/").get_data(as_text=True)
    assert "⚠ VIX no quote" in body.split('tile-volatility"')[1].split('class="metric-tile')[0]
    assert "⚠ TNX quote 35 min old" in body.split('tile-rates"')[1].split('class="metric-tile')[0]
    credit = body.split('tile-credit"')[1].split('class="metric-tile')[0]
    assert "⚠ IEF no quote" in credit and "HYG no quote" not in credit
    assert "min old" not in body.split('class="page-head"')[1].split("</div>")[0]


def test_home_reads_each_funds_file_once(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _write_market_context(market_context_silo, rising={"SPY", "QQQ", "IWM", "RSP"})
    calls = []
    real = app_module.insights.market_bars

    def counted(symbol):
        calls.append(symbol)
        return real(symbol)

    monkeypatch.setattr(app_module.insights, "market_bars", counted)
    assert client.get("/").status_code == 200
    assert sorted(calls) == sorted({*app_module.config.MARKET_HOME_FUNDS, "SCHG", "SCHD"})


def test_home_since_last_close_line(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    for sym in ("SPY", "QQQ"):
        _bars_file(market_context_silo, sym)
    _snapshot(("SPY",), last=300.0)  # below its 50-day 334.50, above its 200-day 259.50
    _now(monkeypatch, "2026-11-02T10:30:00-05:00")
    line = client.get("/").get_data(as_text=True).split('class="since-line"')[1].split("</p>")[0]
    assert "Since last close" in line and "SPY crossed below the 50-day average" in line
    assert "QQQ" not in line and "200-day" not in line


def test_home_has_no_since_line_without_a_crossing(
    client, tmp_path, monkeypatch, market_context_silo
):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _bars_file(market_context_silo, "SPY")
    assert "since-line" not in client.get("/").get_data(as_text=True)


def _breadth_file(date_text="2026-10-30"):
    counts = {
        "above_50": 124,
        "counted_50": 277,
        "above_200": 134,
        "counted_200": 270,
        "new_highs": 19,
        "new_lows": 13,
    }
    payload = {
        "date": date_text,
        "previous_date": "2026-10-29",
        "tickers": 277,
        "current": counts,
        "previous": counts | {"above_50": 106, "new_highs": 1, "new_lows": 8},
    }
    path = app_module.config.UNIVERSE_BREADTH_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle)


def test_home_scanned_tickers_tile(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _breadth_file()
    _now(monkeypatch, "2026-10-30T20:00:00-04:00")
    tile = client.get("/").get_data(as_text=True).split('tile-scanned"')[1].split("</section>")[0]
    assert "at the 2026-10-30 close" in tile and "not S&amp;P breadth" in tile
    assert "124 of 277" in tile and "106 of 277 at previous close" in tile
    assert "134 of 270" in tile and "19 / 13" in tile and "1 / 8 at previous close" in tile
    assert "stale" not in tile


@pytest.mark.parametrize(
    ("now", "stale"),
    [
        ("2026-11-02T10:00:00-05:00", False),  # Monday session: Friday is the last close
        ("2026-11-02T18:45:00-05:00", False),  # Monday's scan may still be running
        ("2026-11-02T19:30:00-05:00", True),  # it should have finished
    ],
)
def test_home_scanned_tickers_stale_after_the_scan_grace(
    client, tmp_path, monkeypatch, market_context_silo, now, stale
):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _breadth_file()
    _now(monkeypatch, now)
    tile = client.get("/").get_data(as_text=True).split('tile-scanned"')[1].split("</section>")[0]
    assert ("⚠ scanned tickers stale" in tile) is stale


def test_home_scanned_tickers_without_a_file(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    tile = client.get("/").get_data(as_text=True).split('tile-scanned"')[1].split("</section>")[0]
    assert "No data" in tile


def test_home_next_events_line(client, tmp_path, monkeypatch, market_context_silo):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    _now(monkeypatch, "2026-11-05T13:00:00-05:00")
    body = client.get("/").get_data(as_text=True)
    line = body.split('<span class="label">Next</span>')[1].split("</p>")[0]
    assert '<span title="Fri 2026-11-06 08:30 ET">Jobs report tomorrow</span>' in line
    assert "CPI in 5 days" in line and "FOMC in 34 days" in line
    assert "dates need updating" not in line
    _now(monkeypatch, "2028-01-03T13:00:00-05:00")
    assert "⚠ CPI dates need updating" in client.get("/").get_data(as_text=True)


def test_old_market_link_redirects_home(client):
    response = client.get("/market")
    assert response.status_code == 301
    location = response.headers["Location"]
    assert location.endswith("/") and "/market" not in location


def test_nav_order_market_changes_setups_runs(client, tmp_path, monkeypatch):
    _market_logs(tmp_path, monkeypatch, weekly_trend="BULLISH")
    nav = client.get("/runs").get_data(as_text=True).split('class="top-nav"')[1].split("</nav>")[0]
    labels = [part.split("</a>")[0].split(">")[-1] for part in nav.split('class="nav-link')[1:]]
    assert labels == [
        "Market",
        "Changes",
        "Setups (unproven)",
        "Scan stats",
        "Runs",
        "Universe",
        "Docs",
    ]
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
    monkeypatch.setattr(app_module, "_fetch_live_prices", lambda syms: {s: "N/A" for s in syms})
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


def test_changes_is_market_facts_only(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    body = client.get("/changes").get_data(as_text=True)
    head = body.split('class="page-head"')[1].split("</div>")[0]
    assert "Fri 2026-10-30 vs Thu 2026-10-29" in head and body.count("asof-badge") == 1
    for gone in (
        "new-confirmed",
        "universe-changes",
        "everything-else",
        "CONFIRMED",
        "buy or sell",
    ):
        assert gone not in body
    market = _section(body, "market-changes")
    qqq = market.split("<strong>QQQ:</strong>")[1].split("</li>")[0]
    assert "closed below its 50-day average (was above)" in qqq
    assert "newly closed below support on heavy volume" in qqq
    assert "SPY:" not in market and "Growth vs dividend unchanged:" in market


def test_old_changes_full_list_link_goes_to_setups(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    response = client.get("/changes?show=all")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/setups?show=all#since-last-run")


def _since(body):
    return body.split('id="since-last-run"')[1].split('class="top-section failed-counts"')[0]


def test_setups_since_last_run_new_weekly_confirmed(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    sec = _section(_since(client.get("/setups").get_data(as_text=True)), "new-confirmed")
    new, unscanned = sec.split("Confirmed now, but not scanned last week (")
    assert "New this week (1)" in new and "t=FIRST" in new
    assert unscanned.startswith("1)") and "t=GPN" in unscanned
    assert "GMAB" not in sec  # CONFIRMED both weeks: a continuation


def test_setups_since_last_run_stock_list(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    sec = _section(_since(client.get("/setups").get_data(as_text=True)), "universe-changes")
    weekly, daily = sec.split("⚡ Daily")
    assert "Entered (1)</th><td>GPN" in weekly and "Left (1)</th><td>LEFTW" in weekly
    assert "Entered (1)</th><td>NEWD" in daily and "Left (1)</th><td>LEFTD" in daily


def test_setups_since_last_run_counts_then_full_list(client, tmp_path, monkeypatch):
    _changes_logs(tmp_path, monkeypatch)
    body = _since(client.get("/setups").get_data(as_text=True))
    assert '<details class="failed-section everything-else">' in body  # closed
    daily = body.split("everything-else")[1].split("⚡ Daily")[1]
    assert "Moved toward a breakout 1 · Failed breakouts 1" in daily
    assert "Cobra setups new 1 / dropped 0" in daily
    assert "t=AAA" not in body and "t=BBB" not in body  # counts only
    assert 'href="/setups?show=all#since-last-run"' in body
    full = _since(client.get("/setups?show=all").get_data(as_text=True))
    assert '<details class="failed-section everything-else" open>' in full
    daily = full.split("everything-else")[1].split("⚡ Daily")[1]
    assert "Moved toward a breakout (1)" in daily and "quote.ashx?t=AAA" in daily
    assert "Failed breakouts (1)" in daily and "quote.ashx?t=BBB" in daily
    assert "New (1)" in daily and "CCC 80" in daily


def test_changes_page_without_runs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "LOG_DIRS", {"weekly": str(tmp_path / "x"), "daily": str(tmp_path / "y")}
    )
    body = client.get("/changes").get_data(as_text=True)
    assert "Need two daily runs to compare." in body and "asof-badge" not in body
    setups = client.get("/setups").get_data(as_text=True)
    assert "Need two weekly runs to compare." in setups


def test_download_scan_serves_the_file_as_written(client, breakout_logs, tmp_path):
    src = tmp_path / "daily" / "breakout_setups_2026-09-04.csv"
    resp = client.get("/download/daily/breakout/2026-09-04.csv")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    assert "attachment" in resp.headers["Content-Disposition"]
    assert "breakout_setups_2026-09-04.csv" in resp.headers["Content-Disposition"]
    assert resp.get_data() == src.read_bytes()  # every row and column, no live prices
    resp.close()


def test_download_scan_rejects_unknown_scan_mode_or_date(client, breakout_logs):
    assert client.get("/download/daily/bogus/2026-09-04.csv").status_code == 404
    assert client.get("/download/bogus/breakout/2026-09-04.csv").status_code == 404
    assert client.get("/download/daily/breakout/1999-01-01.csv").status_code == 404
    assert client.get("/download/daily/breakout/..%2F..%2Fx.csv").status_code in (403, 404)


def test_run_pages_link_their_download(client, breakout_logs, tmp_path):
    pd.DataFrame([{"Symbol": "AAA", "Score": 90, "Close": 1.0}]).to_csv(
        tmp_path / "daily" / "coiled_cobra_setups_2026-09-04.csv", index=False
    )
    body = client.get("/breakout/daily/2026-09-04").get_data(as_text=True)
    assert 'href="/download/daily/breakout/2026-09-04.csv" download' in body
    body = client.get("/cobra/daily/2026-09-04").get_data(as_text=True)
    assert 'href="/download/daily/cobra/2026-09-04.csv" download' in body


def test_ticker_runs_csv_has_one_row_per_mode_and_run(client, breakout_logs, tmp_path):
    _write_scan(tmp_path / "daily", "2026-09-05")
    _write_scan(tmp_path / "weekly", "2026-09-04")
    pd.DataFrame([{"Symbol": "AAA", "Score": 88, "Grade": "A - Watch", "Close": 100.0}]).to_csv(
        tmp_path / "daily" / "coiled_cobra_setups_2026-09-05.csv", index=False
    )
    resp = client.get("/runs/ticker/aaa/history.csv")
    assert resp.status_code == 200 and resp.mimetype == "text/csv"
    assert "filename=AAA_runs.csv" in resp.headers["Content-Disposition"]
    out = pd.read_csv(io.StringIO(resp.get_data(as_text=True)))
    assert list(out.columns) == [
        "Mode", "Date", "Close", "Status", "Breakout Readiness", "Trend",
        "Breakout Distance", "Cobra Score", "Cobra Grade",
    ]  # fmt: skip
    assert list(zip(out["Mode"], out["Date"], strict=True)) == [
        ("weekly", "2026-09-04"),
        ("daily", "2026-09-05"),
        ("daily", "2026-09-04"),
    ]
    assert out["Cobra Score"].tolist()[1] == 88 and out["Close"].ne(95.0).all()  # no live price
    page = client.get("/runs/ticker/AAA").get_data(as_text=True)
    assert 'href="/runs/ticker/AAA/history.csv" download' in page


def test_ticker_runs_csv_unknown_and_unsafe(client, breakout_logs):
    empty = pd.read_csv(
        io.StringIO(client.get("/runs/ticker/ZZZ/history.csv").get_data(as_text=True))
    )
    assert empty.empty and "Mode" in empty.columns
    assert "history.csv" not in client.get("/runs/ticker/ZZZ").get_data(as_text=True)
    assert client.get("/runs/ticker/A%20B/history.csv").status_code == 404


def _scan_stats_mode(newest="2026-11-06"):
    return {
        "newest_run": newest,
        "current": [
            {
                "symbol": "ROKU",
                "score": 83.0,
                "grade": "B - Watch",
                "status": "WATCH",
                "breakout_readiness": 75.0,
            }
        ],
        "previous_run": "2026-11-05",
        "entered": ["ROKU"],
        "left": [],
        "year": "2026",
        "year_median": 16.0,
        "universe": 520,
        "blockers": [
            {
                "Blocker": "Gate C: coil integrity",
                "Rejected bars": 693303,
                "Only blocker": 88166,
                "Pct of rejected": 82.8,
            }
        ],
        "gate_c": [{"Gate C miss": "Both", "Bars": 187757}],
        "durations": [
            {
                "Status": "Cobra passes",
                "Runs": 16634,
                "Open runs": 91,
                "Median bars": 2.0,
                "90th pct bars": 7.0,
                "Longest bars": 44,
                "One-bar runs": 7222,
            },
            {
                "Status": "NO_RUNS",
                "Runs": 0,
                "Open runs": 2,
                "Median bars": None,
                "90th pct bars": None,
                "Longest bars": None,
                "One-bar runs": 0,
            },
        ],
        "pass_rates": [
            {"Status": "All bars", "All bars": 891800, "Cobra passes": 53990, "Pass rate": 6.1}
        ],
    }


def _near_miss_fields():
    return {
        "near_misses": [
            {
                "Missing": "Score 65-69 (needs 70)",
                "Bars": 16476,
                "Pass soon %": 52.5,
                "Pass later %": 58.9,
            }
        ],
        "near_miss_bars": [5, 10],
        "near_miss_date": "2026-11-06",
        "near_miss_count": 95,
        "near_miss_tickers": [
            {
                "Symbol": "SYF",
                "Score": 80.0,
                "Checks met": "5/6",
                "Missing": "Gate A only: long-term trend",
            }
        ],
    }


@pytest.fixture
def scan_stats(tmp_path, monkeypatch, breakout_logs):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "built_at": "2026-11-09T14:48:27",
                "modes": {
                    "daily": {**_scan_stats_mode("2026-09-04"), **_near_miss_fields()},
                    "weekly": _scan_stats_mode(),  # a summary saved before near misses existed
                },
            }
        )
    )
    monkeypatch.setattr(app_module.config, "ANALYSIS_SUMMARY_PATH", str(path))
    return path


def test_scan_stats_page(client, scan_stats):
    html = client.get("/scan-stats").get_data(as_text=True)
    assert "as of database rebuild 2026-11-09 14:48" in html
    assert "693,303" in html
    assert "82.8%" in html
    assert "7,222 of 16,634" in html
    assert "not in the database" not in html
    assert "ROKU" in html


def test_scan_stats_warns_when_a_scan_is_not_in_the_summary(client, scan_stats, tmp_path):
    _write_scan(tmp_path / "daily", "2026-09-05")
    html = client.get("/scan-stats").get_data(as_text=True)
    assert "⚠ scan 2026-09-05 not in the database" in html


def test_scan_stats_unknown_mode_falls_back_to_daily(client, scan_stats):
    assert client.get("/scan-stats?mode=monthly").status_code == 200


@pytest.mark.parametrize("content", [None, "{not json"])
def test_scan_stats_without_a_readable_summary(client, scan_stats, content):
    if content is None:
        scan_stats.unlink()
    else:
        scan_stats.write_text(content)
    html = client.get("/scan-stats").get_data(as_text=True)
    assert "No summary yet" in html


def test_scan_stats_near_misses(client, scan_stats):
    html = client.get("/scan-stats").get_data(as_text=True)
    assert "Cobra near misses" in html
    assert "16,476" in html and "52.5%" in html and "Pass within 5 bars" in html
    assert "(1 of 95)" in html and "⚠ backfill bar 2026-11-06" in html
    assert "SYF" in html and "Gate A only: long-term trend" in html


def test_scan_stats_without_near_misses_in_an_older_summary(client, scan_stats):
    html = client.get("/scan-stats?mode=weekly").get_data(as_text=True)
    assert "Cobra near misses" not in html


def _seed_universe(monkeypatch, tmp_path, *, status="ok"):
    out = tmp_path / "universe"
    out.mkdir()
    pd.DataFrame(
        [
            {"Symbol": "SPY", "Role": "fund", "Sources": "fund", "Added": "2026-11-10",
             "Close": 680.0, "Dollar Volume": 3.5e10, "Bars": 500, "Notes": ""},
            {"Symbol": "AAA", "Role": "stock", "Sources": "S&P 500;watchlist",
             "Added": "2026-11-10", "Close": 50.0, "Dollar Volume": 2.25e7, "Bars": 500,
             "Notes": ""},
        ]
    ).to_csv(out / "members.csv", index=False)  # fmt: skip
    pd.DataFrame(
        [{"Date": "2026-11-10", "Symbol": "AAA", "Change": "added", "Reason": "initial build"},
         {"Date": "2026-12-19", "Symbol": "ZZZ", "Change": "removed",
          "Reason": "dollar_volume_below_15M"}]
    ).to_csv(out / "changes.csv", index=False)  # fmt: skip
    pd.DataFrame(
        [{"Symbol": "AMC", "Sources": "watchlist", "Reasons": "close_below_5", "Close": 3.4,
          "Dollar Volume": 5.89e7, "Bars": 500}]
    ).to_csv(out / "rejected.csv", index=False)  # fmt: skip
    meta = {"built": "2026-12-19", "status": status, "attempted": "2026-12-20T09:00:00-05:00",
            "index_lists_stale": []}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(app_module.config, "UNIVERSE_DIR", str(out))


def test_universe_page(client, breakout_logs, tmp_path, monkeypatch):
    body = client.get("/universe").get_data(as_text=True)
    assert "No universe build yet." in body
    _seed_universe(monkeypatch, tmp_path)
    body = client.get("/universe").get_data(as_text=True)
    assert body.count("asof-badge") == 1 and "built Sat 2026-12-19" in body
    assert "⚠" not in body
    changes = body.split("<h2>Changes")[1].split("<h2>")[0]
    assert "ZZZ" in changes and "dollar_volume_below_15M" in changes
    assert "AAA" not in changes  # first-build additions are only counted
    assert "AMC" in body and "close_below_5" in body and "58.9M" in body
    members = body.split("<h2>Members</h2>")[1]
    assert members.index("SPY") < members.index("AAA") and "22.5M" in members
    assert 'href="/runs/ticker/AAA"' in members
    nav = body.split('class="top-nav"')[1].split("</nav>")[0]
    assert 'nav-link active" href="/universe"' in nav


def test_universe_page_flags_a_failed_rebuild(client, breakout_logs, tmp_path, monkeypatch):
    _seed_universe(monkeypatch, tmp_path, status="kept_previous")
    body = client.get("/universe").get_data(as_text=True)
    assert "⚠ last rebuild failed 2026-12-20; members from 2026-12-19" in body


def test_runs_page_notes_universe_size(client, breakout_logs, tmp_path, monkeypatch):
    assert '/universe">Universe</a>:' not in client.get("/runs").get_data(as_text=True)
    _seed_universe(monkeypatch, tmp_path)
    body = client.get("/runs").get_data(as_text=True)
    assert '<a href="/universe">Universe</a>: 1 stocks + 1 funds' in body
