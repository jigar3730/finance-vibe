"""Flask dashboard for browsing weekly/daily trade-plan CSVs with live quotes."""

from __future__ import annotations

import glob
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import yfinance as yf
from flask import Flask, abort, render_template, request

from finance_vibe.docs_routes import docs_bp
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)

# Ensure absolute paths resolve relative to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGS_BASE_DIR = os.path.join(BASE_DIR, "data", "logs")

# Breakout scanner log silos (breakout_scanner.py writes one CSV per run date
# per swing profile).
BREAKOUT_MODES = {
    "weekly": os.path.join(LOGS_BASE_DIR, "weekly"),
    "daily": os.path.join(LOGS_BASE_DIR, "daily"),
}

# Table columns mirror breakout_scanner.DISPLAY_COLUMNS (states first; the
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

# Mirrors breakout_scanner's display rule and status order (tests/test_breakout_dashboard.py
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
_live_price_cache: dict[str, tuple[float, float]] = {}

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

app.register_blueprint(docs_bp)

# The dual-timeframe folder structure paths
MODES = {
    "weekly": os.path.join(LOGS_BASE_DIR, "weekly"),
    "daily": os.path.join(LOGS_BASE_DIR, "daily"),
}


def _get_available_runs() -> dict[str, list[dict]]:
    """Scan weekly/daily log folders and return dated trade-plan files."""
    runs: dict[str, list[dict]] = {"weekly": [], "daily": []}

    for mode, folder_path in MODES.items():
        if not os.path.exists(folder_path):
            continue

        # Look for trade plan files matching the naming pattern
        file_pattern = os.path.join(folder_path, "trade_plan_*.csv")
        all_files = glob.glob(file_pattern)

        seen_dates = set()
        for file_path in sorted(all_files, reverse=True):
            file_name = os.path.basename(file_path)
            parts = (
                file_name.replace("trade_plan_clean_", "")
                .replace("trade_plan_", "")
                .replace(".csv", "")
            )
            try:
                # Validate string format is a valid date
                datetime.strptime(parts[:10], "%Y-%m-%d")
                date_str = parts[:10]
                if date_str not in seen_dates:
                    seen_dates.add(date_str)
                    runs[mode].append(
                        {"date": date_str, "file_name": file_name, "is_clean": "clean" in file_name}
                    )
            except ValueError:
                continue

    return runs


def _get_breakout_runs() -> dict[str, list[dict]]:
    """Scan each breakout log silo for dated ``breakout_setups_<date>.csv``.

    Returns one entry per mode, newest run date first. The filename date is the
    scan/run date used in the view URL (distinct from the CSV's ``AsOf Date``).
    """
    runs: dict[str, list[dict]] = {mode: [] for mode in BREAKOUT_MODES}

    for mode, folder_path in BREAKOUT_MODES.items():
        if not os.path.exists(folder_path):
            continue

        file_pattern = os.path.join(folder_path, "breakout_setups_*.csv")
        all_files = glob.glob(file_pattern)

        seen_dates = set()
        for file_path in sorted(all_files, reverse=True):
            file_name = os.path.basename(file_path)
            parts = file_name.replace("breakout_setups_", "").replace(".csv", "")
            try:
                datetime.strptime(parts[:10], "%Y-%m-%d")
            except ValueError:
                continue
            date_str = parts[:10]
            if date_str in seen_dates:
                continue
            seen_dates.add(date_str)
            runs[mode].append({"date": date_str, "file_name": file_name})

    return runs


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


def _relabel_triggered_watch(df: pd.DataFrame) -> pd.DataFrame:
    """Scans before 2026-10-20 filed triggered-but-unconfirmed rows under WATCH.

    The scanner's ``Breakout Triggered`` flag identifies them (a triggered row only
    reaches WATCH via that rule), so older CSVs get the same BREAKOUT_TRIGGERED status.
    """
    if "Status" not in df.columns or "Breakout Triggered" not in df.columns:
        return df
    triggered = df["Breakout Triggered"].astype(str).str.strip().str.lower() == "true"
    df.loc[(df["Status"] == "WATCH") & triggered, "Status"] = BREAKOUT_TRIGGERED
    return df


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


def _breakout_table_html(table_df: pd.DataFrame, live_price_map: dict[str, float | str]) -> str:
    """Render breakout rows with a Live Price column after Close and Finviz symbol links."""
    table_df = table_df.copy()
    if "Symbol" in table_df.columns and "Close" in table_df.columns:
        closes = pd.to_numeric(table_df["Close"], errors="coerce")
        table_df["Live Price"] = [
            _live_price_cell(live_price_map.get(str(sym).strip().upper()), close)
            if pd.notna(sym)
            else ""
            for sym, close in zip(table_df["Symbol"], closes, strict=True)
        ]
        cols = list(table_df.columns)
        cols.insert(cols.index("Close") + 1, cols.pop(cols.index("Live Price")))
        table_df = table_df[cols]

    # Finviz quote links on the Symbol column (matches trade-plan view).
    if "Symbol" in table_df.columns:
        table_df["Symbol"] = table_df["Symbol"].apply(
            lambda x: (
                f'<a href="https://finviz.com/quote.ashx?t={str(x).strip().upper()}" '
                f'target="_blank" rel="noopener noreferrer" class="ticker-link">{x}</a>'
                if pd.notna(x)
                else ""
            )
        )
    return table_df.to_html(classes="table", index=False, border=0, escape=False)


def _fetch_one_live_price(sym: str) -> float | str:
    """Last price for one symbol via yfinance ``fast_info``; ``N/A`` on any failure."""
    try:
        return round(yf.Ticker(sym).fast_info["last_price"], 2)
    except Exception:
        return "N/A"


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


# Live price this far above the close (percent) highlights the trade-plan row.
LIVE_SURGE_PCT = 5.0


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


@app.route("/")
def index() -> str:
    """Render the dashboard index of available weekly and daily runs."""
    runs = _get_available_runs()
    return render_template("index.html", runs=runs)


@app.route("/view/<mode>/<date>")
def view_run(mode: str, date: str) -> str | tuple[str, int]:
    """Render one trade-plan CSV with live Yahoo quotes injected as HTML."""
    if mode not in MODES:
        abort(404, "Invalid historical directory mode context.")

    requested_file = request.args.get("file")
    if not requested_file:
        abort(400, "Missing reference log file parameter.")

    # Strictly validate path safety to prevent directory traversal
    target_path = os.path.abspath(os.path.join(MODES[mode], requested_file))
    if not target_path.startswith(MODES[mode]):
        abort(403, "Access restricted outside authorized mode workspace boundaries.")

    if not os.path.exists(target_path):
        abort(404, f"The selected file record does not exist: {requested_file}")

    try:
        df = pd.read_csv(target_path)

        # Clean whitespaces out of column definitions
        df.columns = df.columns.str.strip()

        # Target the ticker symbol column dynamically
        symbol_col = None
        for col in df.columns:
            if col.lower() in ["symbol", "ticker"]:
                symbol_col = col
                break

        if symbol_col is not None:
            # 1. Gather clean, raw symbol strings to request live quotes
            raw_symbols = [str(x).strip().upper() for x in df[symbol_col].dropna().unique()]
            live_price_map = _fetch_live_prices(raw_symbols)

            # 2. Add 'Live Price' values aligned with symbols (green above close, red below;
            #    whole row highlighted when live is more than LIVE_SURGE_PCT above close)
            closes = pd.to_numeric(df["Close"], errors="coerce") if "Close" in df.columns else None
            df["Live Price"] = [
                _live_price_cell(
                    live_price_map.get(str(sym).strip().upper()),
                    closes.iloc[i] if closes is not None else float("nan"),
                    LIVE_SURGE_PCT,
                )
                if pd.notna(sym)
                else ""
                for i, sym in enumerate(df[symbol_col])
            ]

            # 3. Restructure layout: Inject 'Close' then 'Live Price' right after the 'Symbol' column
            cols = list(df.columns)
            symbol_idx = cols.index(symbol_col)
            cols.insert(symbol_idx + 1, cols.pop(cols.index("Live Price")))
            # Close sits immediately before Live Price so the two are easy to compare.
            if "Close" in cols:
                cols.insert(cols.index("Live Price"), cols.pop(cols.index("Close")))
            df = df[cols]

            # 4. Convert plain strings into operational Finviz anchor links
            df[symbol_col] = df[symbol_col].apply(
                lambda x: (
                    f'<a href="https://finviz.com/quote.ashx?t={str(x).strip().upper()}" target="_blank" rel="noopener noreferrer" class="ticker-link">{x}</a>'
                    if pd.notna(x)
                    else ""
                )
            )

        # escape=False ensures Pandas treats custom injected HTML elements cleanly
        table_html = df.to_html(classes="table", index=False, border=0, escape=False)
        return render_template(
            "view.html", mode=mode, date=date, file_name=requested_file, table_html=table_html
        )
    except Exception as e:  # route boundary: show the error page, keep the traceback in the log
        logger.exception("Failed to render %s", request.path)
        return f"<h3>❌ Failed to parse data contents:</h3><pre>{e!s}</pre>", 500


@app.route("/breakout")
def breakout_index() -> str:
    """List available breakout scan runs (per mode, newest first)."""
    runs = _get_breakout_runs()
    return render_template("breakout_index.html", runs=runs)


@app.route("/breakout/<mode>/<date>")
def breakout_view(mode: str, date: str) -> str | tuple[str, int]:
    """Render one breakout scan CSV: KPI counts, status chips, and the table.

    Read-only. The raw CSV is never rewritten. By default the table follows the
    scanner's display rule, with FAILED_BREAKOUT rows in a separate collapsed
    table; ``?show=all`` lists every row and ``?status=`` every row of one status.
    KPIs and the status mix are always computed on the full file.
    """
    if mode not in BREAKOUT_MODES:
        abort(404, "Invalid breakout mode.")

    folder = BREAKOUT_MODES[mode]
    file_name = f"breakout_setups_{date}.csv"

    # Path-safety: resolved path must stay inside the mode's log silo.
    target_path = os.path.abspath(os.path.join(folder, file_name))
    if not target_path.startswith(os.path.abspath(folder) + os.sep):
        abort(403, "Access restricted outside authorized mode workspace boundaries.")

    if not os.path.exists(target_path):
        abort(404, f"No breakout scan found for {mode} on {date}.")

    try:
        df = pd.read_csv(target_path)
        df.columns = df.columns.str.strip()
        df = _relabel_triggered_watch(df)

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

        split_failed = not active_status and not show_all and has_status
        if split_failed:
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

        # Live price next to Close, coloured by live vs. close (only visible rows are quoted).
        live_price_map: dict[str, float | str] = {}
        if "Symbol" in view_cols and "Close" in view_cols:
            visible = pd.concat([table_df["Symbol"], failed_df["Symbol"]]).dropna()
            live_price_map = _fetch_live_prices(
                list(dict.fromkeys(str(x).strip().upper() for x in visible))
            )

        return render_template(
            "breakout_view.html",
            mode=mode,
            date=date,
            file_name=file_name,
            kpis=kpis,
            status_mix=status_mix,
            active_status=active_status,
            show_all=show_all,
            row_count=int(len(table_df)),
            table_html=_breakout_table_html(table_df, live_price_map),
            failed_count=int(len(failed_df)),
            failed_table_html=_breakout_table_html(failed_df, live_price_map)
            if len(failed_df)
            else "",
            score_floor=BREAKOUT_SCORE_FLOOR,
        )
    except Exception as e:  # route boundary: show the error page, keep the traceback in the log
        logger.exception("Failed to render %s", request.path)
        return f"<h3>❌ Failed to parse data contents:</h3><pre>{e!s}</pre>", 500


def main() -> None:
    """Serve the dashboard (``finance-vibe-app`` console script)."""
    setup_logging()
    # Ensure standard fallback logs dirs exist locally
    for path in list(MODES.values()) + list(BREAKOUT_MODES.values()):
        os.makedirs(path, exist_ok=True)

    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    logger.info("🚀 Launching Upgraded Finance Vibe UI Dashboard Context...")
    app.run(host="0.0.0.0", port=5000, debug=debug)


if __name__ == "__main__":
    main()
