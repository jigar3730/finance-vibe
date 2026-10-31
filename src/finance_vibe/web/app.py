"""Flask dashboard for the two scanners' dated CSVs (Coiled Cobra, breakout), with live quotes."""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import yfinance as yf
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for
from werkzeug.wrappers import Response as RedirectResponse

from finance_vibe import config
from finance_vibe.data import raw
from finance_vibe.log import setup_logging
from finance_vibe.web import health, insights, playbook
from finance_vibe.web.playbook import (
    BREAKOUT_PREFIX,
    COBRA_PREFIX,
    read_scan,
    relabel_triggered_watch,
)

logger = logging.getLogger(__name__)

WEB_DIR = os.path.dirname(os.path.abspath(__file__))

# One log silo per mode; both scanners write dated CSVs into it.
LOG_DIRS = {mode: profile["logs_dir"] for mode, profile in config.TIMEFRAME_PROFILES.items()}

COBRA_DISPLAY_COLUMNS = [
    "Symbol",
    "AsOf Date",
    "Close",
    "Score",
    "Grade",
    "Tier",
    "Checks Met",
    "RVOL",
    "RS 63d",
    "Market Gate",
    "Fib 78.6%",
    "Swing Low",
    "ATR",
    "RSI",
]

# Table columns mirror scanners.breakout.DISPLAY_COLUMNS (states first; the
# 100-pt score is secondary). Defined locally so the UI doesn't import the
# scanner and its indicator stack.
BREAKOUT_DISPLAY_COLUMNS = [
    "Symbol",
    "Status",
    "AsOf Date",
    "Close",
    "Trend",
    "Volatility",
    "Volume State",
    "Momentum",
    "Structure",
    "Breakout Distance",
    "MTF",
    "Breakout Readiness",
    "Distance Resistance ATR",
    "RVOL20",
    "Compression",
]

# Mirrors scanners.breakout's display rule and status order (tests/test_dashboard.py
# checks they agree): event statuses always show; DEVELOPING needs a readiness floor; WATCH
# is only listed via its status chip or "All".
BREAKOUT_TRIGGERED = "BREAKOUT_TRIGGERED"
BREAKOUT_FAILED = "FAILED_BREAKOUT"
BREAKOUT_ALWAYS_SHOWN = {"PRE_BREAKOUT", "BREAKOUT_CONFIRMED", BREAKOUT_TRIGGERED, BREAKOUT_FAILED}
BREAKOUT_SCORE_FLOOR = {"DEVELOPING": 50}
BREAKOUT_STATUS_ORDER = [
    "PRE_BREAKOUT",
    "BREAKOUT_CONFIRMED",
    BREAKOUT_TRIGGERED,
    BREAKOUT_FAILED,
    "WATCH",
    "DEVELOPING",
]

# Live quotes: parallel fast_info lookups, cached briefly per symbol.
LIVE_PRICE_WORKERS = 16
LIVE_PRICE_TTL_SECONDS = 60
# Live price this far above the close (percent) highlights a Coiled Cobra row.
LIVE_SURGE_PCT = 5.0
_live_price_cache: dict[str, tuple[float, float]] = {}

app = Flask(
    __name__,
    template_folder=os.path.join(WEB_DIR, "templates"),
    static_folder=os.path.join(WEB_DIR, "static"),
)


@app.template_filter("text")
def _text_filter(value: object) -> str:
    """Display a CSV cell as text; blanks and NaN become an em dash."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return "—"
    return str(value)


@app.template_filter("num")
def _num_filter(value: object, digits: int = 2) -> str:
    """Format a number for display; blanks and NaN become an em dash."""
    v = pd.to_numeric(value, errors="coerce")  # type: ignore[call-overload]
    return "—" if pd.isna(v) else f"{float(v):,.{digits}f}"


@app.template_filter("signed")
def _signed_filter(value: object, digits: int = 0) -> str:
    """A change with its sign (+3, -2, 0); blanks and NaN become an em dash."""
    v = pd.to_numeric(value, errors="coerce")  # type: ignore[call-overload]
    return "—" if pd.isna(v) else f"{float(v):+,.{digits}f}"


# ---------------------------------------------------------------------------
# Run discovery and file resolution
# ---------------------------------------------------------------------------


def _list_runs() -> dict[str, list[dict[str, object]]]:
    """Run dates per mode, newest first, flagging which scanners wrote a CSV that day."""
    runs: dict[str, list[dict[str, object]]] = {}
    for mode, folder in LOG_DIRS.items():
        dates = playbook.run_dates(folder)
        runs[mode] = [
            {"date": d, "cobra": COBRA_PREFIX in p, "breakout": BREAKOUT_PREFIX in p}
            for d, p in sorted(dates.items(), reverse=True)
        ]
    return runs


def _resolve_csv(mode: str, prefix: str, date: str) -> str:
    """Path to ``<prefix><date>.csv`` in the mode's log silo, or abort 404/403."""
    if mode not in LOG_DIRS:
        abort(404, "Invalid mode.")
    folder = os.path.abspath(LOG_DIRS[mode])
    path = os.path.abspath(os.path.join(folder, f"{prefix}{date}.csv"))
    if not path.startswith(folder + os.sep):
        abort(403, "Access restricted outside the mode's log directory.")
    if not os.path.exists(path):
        abort(404, f"No scan found for {mode} on {date}.")
    return path


# ---------------------------------------------------------------------------
# Live quotes and table rendering (shared by both scanner pages)
# ---------------------------------------------------------------------------


def _fetch_one_live_price(sym: str) -> float | str:
    """Last price for one symbol via yfinance ``fast_info``; ``N/A`` on any failure.

    yfinance can return NaN for a thin, halted or pre-open symbol; that is ``N/A``
    too, so it is never cached or treated as a price.
    """
    try:
        price = playbook.live_price_or_none(yf.Ticker(sym).fast_info["last_price"])
    except Exception:
        return "N/A"
    return "N/A" if price is None else round(price, 2)


def _fetch_live_prices(symbols: list[str]) -> dict[str, float | str]:
    """Fetch last prices via yfinance ``fast_info``; missing symbols map to ``N/A``.

    Each ``fast_info`` lookup is its own HTTP round-trip, so they run in a thread
    pool (a ~275-symbol breakout scan: ~33s sequential, ~4s pooled). Successful
    quotes are cached for ``LIVE_PRICE_TTL_SECONDS`` so reloads and status-filter
    clicks don't re-fetch.
    """
    if not symbols:
        return {}
    now = time.monotonic()
    prices: dict[str, float | str] = {}
    missing = []
    for sym in symbols:
        cached = _live_price_cache.get(sym)
        if cached is not None and now - cached[0] < LIVE_PRICE_TTL_SECONDS:
            prices[sym] = cached[1]
        else:
            missing.append(sym)
    if missing:
        with ThreadPoolExecutor(max_workers=min(LIVE_PRICE_WORKERS, len(missing))) as pool:
            fetched = dict(zip(missing, pool.map(_fetch_one_live_price, missing), strict=True))
        failed = sum(1 for p in fetched.values() if p == "N/A")
        if failed:
            logger.warning("Live price unavailable for %d/%d symbols", failed, len(missing))
        for sym, price in fetched.items():
            if isinstance(price, float):
                _live_price_cache[sym] = (now, price)
        prices.update(fetched)
    return prices


def _live_price_cell(live: float | str | None, close: float, surge_pct: float | None = None) -> str:
    """HTML cell for a live price: red if the close is above it, green if below, else neutral.

    With ``surge_pct`` set, a live price more than that percent above the close also gets a
    ``live-surge`` class, which the stylesheet uses to highlight the whole table row.
    """
    if not isinstance(live, (int, float)) or pd.isna(live):
        return '<span class="live-price-cell live-flat">N/A</span>'
    css = "live-flat"
    if pd.notna(close):
        if close > live:
            css = "live-below-close"
        elif close < live:
            css = "live-above-close"
            if surge_pct is not None and live > close * (1 + surge_pct / 100):
                css += " live-surge"
    return f'<span class="live-price-cell {css}">${live:.2f}</span>'


def _symbols(*frames: pd.DataFrame) -> list[str]:
    """Unique upper-case symbols across frames, in order of appearance."""
    seen = (str(s).strip().upper() for f in frames for s in f["Symbol"].dropna())
    return list(dict.fromkeys(seen))


def _table_html(
    df: pd.DataFrame,
    live_price_map: dict[str, float | str],
    *,
    surge_pct: float | None = None,
) -> str:
    """Render rows with a Live Price column right after Close and Finviz symbol links."""
    df = df.copy()
    if "Symbol" in df.columns and "Close" in df.columns:
        closes = pd.to_numeric(df["Close"], errors="coerce")
        df["Live Price"] = [
            _live_price_cell(live_price_map.get(str(sym).strip().upper()), close, surge_pct)
            if pd.notna(sym)
            else ""
            for sym, close in zip(df["Symbol"], closes, strict=True)
        ]
        cols = list(df.columns)
        cols.insert(cols.index("Close") + 1, cols.pop(cols.index("Live Price")))
        df = df[cols]
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].apply(
            lambda x: (
                f'<a href="https://finviz.com/quote.ashx?t={str(x).strip().upper()}" '
                f'target="_blank" rel="noopener noreferrer" class="ticker-link">{x}</a>'
                if pd.notna(x)
                else ""
            )
        )
    return df.to_html(classes="table", index=False, border=0, escape=False)


# ---------------------------------------------------------------------------
# Breakout helpers
# ---------------------------------------------------------------------------


def _breakout_kpis(df: pd.DataFrame) -> dict[str, object]:
    """Compute run-level KPI counts from the full (unfiltered) scan frame."""
    total = int(len(df))
    readiness = pd.to_numeric(df.get("Breakout Readiness"), errors="coerce")  # type: ignore[call-overload]
    median_readiness = readiness.median()
    actionable = int((readiness >= 70).sum())
    status = df.get("Status")
    failed = int((status == "FAILED_BREAKOUT").sum()) if status is not None else 0
    return {
        "setups": total,
        "median_readiness": None
        if pd.isna(median_readiness)
        else round(float(median_readiness), 1),
        "actionable": actionable,
        "failed": failed,
    }


def _breakout_status_mix(df: pd.DataFrame) -> list[dict]:
    """Ordered per-status counts for the status chips (scanner status order)."""
    status = df.get("Status")
    if status is None:
        return []
    counts = status.value_counts()
    ordered = [s for s in BREAKOUT_STATUS_ORDER if s in counts.index]
    ordered += [s for s in counts.index if s not in BREAKOUT_STATUS_ORDER]
    return [{"status": s, "count": int(counts[s])} for s in ordered]


def _breakout_display_mask(df: pd.DataFrame) -> pd.Series:
    """Scanner display rule: event statuses always, DEVELOPING above a floor, never WATCH."""
    if "Status" not in df.columns:
        return pd.Series(True, index=df.index)
    status = df["Status"]
    floor = status.map(BREAKOUT_SCORE_FLOOR)
    readiness = pd.to_numeric(df.get("Breakout Readiness"), errors="coerce")  # type: ignore[call-overload]
    return status.isin(BREAKOUT_ALWAYS_SHOWN) | (readiness >= floor)


def _sort_breakout_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Status order first, then readiness (descending), then symbol."""
    out = df.copy()
    sort_cols, ascending = [], []
    if "Status" in out.columns:
        rank = {s: i for i, s in enumerate(BREAKOUT_STATUS_ORDER)}
        out["_rank"] = out["Status"].map(rank).fillna(len(rank))
        sort_cols.append("_rank")
        ascending.append(True)
    if "Breakout Readiness" in out.columns:
        sort_cols.append("Breakout Readiness")
        ascending.append(False)
    if "Symbol" in out.columns:
        sort_cols.append("Symbol")
        ascending.append(True)
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending)
    return out.drop(columns="_rank", errors="ignore")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


# /api/bars: recent bars for the Setups and Market level charts.
CHART_BARS = {"daily": 60, "weekly": 52}
_SYMBOL = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def _latest_runs() -> dict[str, tuple[str | None, pd.DataFrame, pd.DataFrame]]:
    """Latest (date, cobra, breakout) per mode, read once per request."""
    return {mode: playbook.load_latest(folder) for mode, folder in LOG_DIRS.items()}


def _market(
    latest: dict[str, tuple[str | None, pd.DataFrame, pd.DataFrame]],
    live: dict[str, float | str] | None = None,
    symbols: tuple[str, ...] = playbook.MARKET_SYMBOLS,
) -> list[dict[str, Any]]:
    """Market tab rows from the latest daily and weekly breakout frames."""
    return playbook.market_rows(latest["daily"][2], latest["weekly"][2], live, symbols)


def _market_strip() -> dict[str, Any] | None:
    """Facts for SPY / QQQ / SCHG / SCHD from the latest daily run, or None if unavailable."""
    try:
        return insights.market_strip(playbook.load_latest(LOG_DIRS["daily"])[2])
    except (OSError, ValueError, KeyError):
        logger.exception("Market strip unavailable")
        return None


@app.context_processor
def market_strip_context() -> dict[str, Any]:
    """The market strip in ``base.html`` on every page but the market home, which shows
    the same facts as its "Today" table."""
    return {"market_strip": None if request.endpoint == "market" else _market_strip()}


@app.route("/setups")
def setups() -> str:
    """Setups (unproven): weekly CONFIRMED cards, untested Cobra setups, failed counts."""
    latest = _latest_runs()
    week_date, _, week_bo = latest["weekly"]
    previous = playbook.load_previous_breakout(LOG_DIRS["weekly"], week_date)
    confirmed = playbook.weekly_confirmed(week_bo, previous)
    live = _fetch_live_prices(list(confirmed["Symbol"]) if not confirmed.empty else [])
    dq_errors = insights.quality_errors(LOG_DIRS["weekly"])
    cards = []
    for row in confirmed.to_dict("records"):
        price = playbook.live_price_or_none(live.get(row["Symbol"]))
        cards.append(
            {
                "symbol": row["Symbol"],
                "row": row,
                "price": price if price is not None else playbook.num(row.get("Close")),
                "price_is_live": price is not None,
                "levels": playbook.range_levels(row),  # type: ignore[arg-type]
                "dq": dq_errors.get(row["Symbol"], []),
            }
        )
    cobra = [
        {"mode": mode, "date": date, "rows": playbook.untested_cobra(cb, bo).to_dict("records")}
        for mode, (date, cb, bo) in latest.items()
    ]
    failed = [
        {"mode": mode, "date": date, "counts": playbook.failed_counts(bo, mode)}
        for mode, (date, _, bo) in latest.items()
    ]
    # Plain facts for SPY / QQQ above the setups; no live prices needed.
    market_rows = _market(latest, symbols=playbook.SETUPS_MARKET_SYMBOLS)
    return render_template(
        "setups.html",
        changes={mode: insights.run_changes(folder) for mode, folder in LOG_DIRS.items()},
        show_all=request.args.get("show") == "all",
        week_date=week_date,
        cards=cards,
        cobra=cobra,
        failed=failed,
        market_rows=market_rows,
    )


def _page_as_of(*dates: str | None) -> str | None:
    """The page's one "as of" badge: the newest of the given ISO dates, e.g. "Fri 2026-10-30"."""
    known = [d for d in dates if d]
    if not known:
        return None
    newest = pd.Timestamp(max(known))
    return f"{newest:%a} {newest.date().isoformat()}"


@app.route("/")
def market() -> str:
    """Home: one table for SPY / QQQ / SCHG / SCHD, market health tiles, sector tiles,
    then a card per ETF with a support-to-resistance position bar. Daily closes only."""
    rows = _market(_latest_runs())
    expected = insights.expected_bars()
    for m in rows:
        m["flags"] = insights.etf_freshness(
            m["symbol"], {"daily": m["daily"], "weekly": m["weekly"]}, expected
        )["daily"]["flags"]
    daily_bo = playbook.load_latest(LOG_DIRS["daily"])[2]
    table = insights.market_table(daily_bo)
    market_health = health.market_health(expected["daily"])
    newest = max((r["as_of"] for r in table["rows"] if r["as_of"]), default=None)
    for r in table["rows"]:
        r["stale"] = r["as_of"] is None or (newest is not None and r["as_of"] < newest)
    as_of = _page_as_of(
        newest,
        (market_health["volatility"] or {}).get("as_of"),
        (market_health["credit"] or {}).get("as_of"),
        market_health["breadth"]["as_of"],
    )
    return render_template("market.html", rows=rows, table=table, health=market_health, as_of=as_of)


@app.route("/market")
def market_redirect() -> RedirectResponse:
    """The market page moved to ``/``; keep old links working."""
    return redirect(url_for("market"), code=301)


@app.route("/api/bars/<mode>/<symbol>")
def api_bars(mode: str, symbol: str) -> Response:
    """Recent OHLC bars for one ticker from the raw silo, for the level charts."""
    if mode not in CHART_BARS or not _SYMBOL.match(symbol):
        abort(404)
    # The raw-file naming contract gives the path directly (no directory scan).
    path = config.get_raw_path(symbol, config.TIMEFRAME_PROFILES[mode])
    if not os.path.isfile(path):
        abort(404)
    try:
        bars = raw.load_raw(path, weekly=mode == "weekly").tail(CHART_BARS[mode])
    except (OSError, ValueError) as exc:
        logger.warning("Bars unavailable for %s %s: %s", mode, symbol, exc)
        abort(404)
    payload = [
        {
            "time": d.strftime("%Y-%m-%d"),
            "open": round(float(o), 4),
            "high": round(float(h), 4),
            "low": round(float(lo), 4),
            "close": round(float(c), 4),
        }
        for d, o, h, lo, c in zip(
            pd.to_datetime(bars["Date"]),
            bars["Open"],
            bars["High"],
            bars["Low"],
            bars["Close"],
            strict=True,
        )
    ]
    response = jsonify(payload)
    response.headers["Cache-Control"] = "max-age=60"
    return response


@app.route("/changes")
def changes() -> str | RedirectResponse:
    """Market facts that changed since the previous daily run. Stock changes live on
    Setups ("Since last run"); the old ``?show=all`` link goes there."""
    if request.args.get("show") == "all":
        return redirect(url_for("setups", show="all", _anchor="since-last-run"))
    return render_template("changes.html", market=insights.market_changes(LOG_DIRS["daily"]))


@app.route("/runs")
def runs() -> str:
    """List run dates per mode with links to each scanner's page."""
    return render_template("runs.html", runs=_list_runs())


@app.route("/cobra/<mode>/<date>")
def cobra_view(mode: str, date: str) -> str | tuple[str, int]:
    """Render one Coiled Cobra setups CSV, Score-ordered, with live quotes."""
    path = _resolve_csv(mode, COBRA_PREFIX, date)
    try:
        df = read_scan(path)
        cols = [c for c in COBRA_DISPLAY_COLUMNS if c in df.columns] or list(df.columns)
        table_df = df[cols]
        if "Score" in table_df.columns:
            table_df = table_df.sort_values("Score", ascending=False, kind="mergesort")
        live = _fetch_live_prices(_symbols(table_df)) if "Symbol" in cols else {}
        return render_template(
            "cobra_view.html",
            mode=mode,
            date=date,
            file_name=os.path.basename(path),
            row_count=int(len(table_df)),
            table_html=_table_html(table_df, live, surge_pct=LIVE_SURGE_PCT),
        )
    except Exception as e:  # route boundary: show the error page, keep the traceback in the log
        logger.exception("Failed to render %s", request.path)
        return f"<h3>❌ Failed to parse data contents:</h3><pre>{e!s}</pre>", 500


@app.route("/breakout/<mode>/<date>")
def breakout_view(mode: str, date: str) -> str | tuple[str, int]:
    """Render one breakout scan CSV: KPI counts, status chips, and the table.

    Read-only. The raw CSV is never rewritten. By default the table follows the
    scanner's display rule, with FAILED_BREAKOUT rows in a separate collapsed
    table; ``?show=all`` lists every row and ``?status=`` every row of one status.
    KPIs and the status mix are always computed on the full file.
    """
    path = _resolve_csv(mode, BREAKOUT_PREFIX, date)
    try:
        df = relabel_triggered_watch(read_scan(path))
        kpis = _breakout_kpis(df)
        status_mix = _breakout_status_mix(df)

        # Status filter or "show all" list every matching row; otherwise apply the
        # scanner's display rule and split failed breakouts into their own table.
        active_status = request.args.get("status")
        show_all = request.args.get("show") == "all" and not active_status
        has_status = "Status" in df.columns
        if active_status and has_status:
            rows = df[df["Status"] == active_status]
        elif show_all:
            rows = df
        else:
            rows = df[_breakout_display_mask(df)]

        if not active_status and not show_all and has_status:
            failed_rows = rows[rows["Status"] == BREAKOUT_FAILED]
            rows = rows[rows["Status"] != BREAKOUT_FAILED]
        else:
            failed_rows = rows.iloc[0:0]

        view_cols = [c for c in BREAKOUT_DISPLAY_COLUMNS if c in df.columns] or list(df.columns)

        def _prepare(part: pd.DataFrame) -> pd.DataFrame:
            out = part[view_cols].copy()
            if "Breakout Readiness" in out.columns:
                out["Breakout Readiness"] = pd.to_numeric(
                    out["Breakout Readiness"], errors="coerce"
                )
            return _sort_breakout_rows(out)

        table_df, failed_df = _prepare(rows), _prepare(failed_rows)
        live = _fetch_live_prices(_symbols(table_df, failed_df)) if "Symbol" in view_cols else {}

        return render_template(
            "breakout_view.html",
            mode=mode,
            date=date,
            file_name=os.path.basename(path),
            kpis=kpis,
            status_mix=status_mix,
            active_status=active_status,
            show_all=show_all,
            row_count=int(len(table_df)),
            table_html=_table_html(table_df, live),
            failed_count=int(len(failed_df)),
            failed_table_html=_table_html(failed_df, live) if len(failed_df) else "",
            score_floor=BREAKOUT_SCORE_FLOOR,
        )
    except Exception as e:  # route boundary: show the error page, keep the traceback in the log
        logger.exception("Failed to render %s", request.path)
        return f"<h3>❌ Failed to parse data contents:</h3><pre>{e!s}</pre>", 500


def main() -> None:
    """Serve the dashboard (``finance-vibe-app`` console script)."""
    setup_logging()
    for path in LOG_DIRS.values():
        os.makedirs(path, exist_ok=True)

    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    logger.info("Starting the Finance Vibe dashboard on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=debug)


if __name__ == "__main__":
    main()
