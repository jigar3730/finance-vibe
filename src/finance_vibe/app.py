"""Flask dashboard for browsing weekly/daily trade-plan CSVs with live quotes."""
from __future__ import annotations

import glob
import os
from datetime import datetime

import pandas as pd
import yfinance as yf
from flask import Flask, abort, render_template, request

# Ensure absolute paths resolve relative to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGS_BASE_DIR = os.path.join(BASE_DIR, "data", "logs")

# Breakout scanner log silos (breakout_scanner.py writes one CSV per run date
# per swing profile). high_beta shares daily raw data but its own log dir.
BREAKOUT_MODES = {
    "weekly": os.path.join(LOGS_BASE_DIR, "weekly"),
    "daily": os.path.join(LOGS_BASE_DIR, "daily"),
    "high_beta": os.path.join(LOGS_BASE_DIR, "high_beta"),
}

# Table columns mirror breakout_scanner.DISPLAY_COLUMNS (states first; the
# 100-pt score is secondary). Defined locally to avoid importing the scanner
# module (which reads sys.argv at import time).
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

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

try:
    from finance_vibe.docs_routes import docs_bp
except ImportError:
    from docs_routes import docs_bp

app.register_blueprint(docs_bp)

# The dual-timeframe folder structure paths
MODES = {
    "weekly": os.path.join(LOGS_BASE_DIR, "weekly"),
    "daily": os.path.join(LOGS_BASE_DIR, "daily")
}

def _get_available_runs() -> dict[str, list[dict]]:
    """Scan weekly/daily log folders and return dated trade-plan files."""
    runs = {"weekly": [], "daily": []}
    
    for mode, folder_path in MODES.items():
        if not os.path.exists(folder_path):
            continue
        
        # Look for trade plan files matching the naming pattern
        file_pattern = os.path.join(folder_path, "trade_plan_*.csv")
        all_files = glob.glob(file_pattern)
        
        seen_dates = set()
        for file_path in sorted(all_files, reverse=True):
            file_name = os.path.basename(file_path)
            parts = file_name.replace("trade_plan_clean_", "").replace("trade_plan_", "").replace(".csv", "")
            try:
                # Validate string format is a valid date
                datetime.strptime(parts[:10], "%Y-%m-%d")
                date_str = parts[:10]
                if date_str not in seen_dates:
                    seen_dates.add(date_str)
                    runs[mode].append({
                        "date": date_str,
                        "file_name": file_name,
                        "is_clean": "clean" in file_name
                    })
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
    readiness = pd.to_numeric(df.get("Breakout Readiness"), errors="coerce")
    median_readiness = readiness.median()
    actionable = int((readiness >= 70).sum())
    status = df.get("Status")
    failed = int((status == "FAILED_BREAKOUT").sum()) if status is not None else 0
    return {
        "setups": total,
        "median_readiness": None if pd.isna(median_readiness) else round(float(median_readiness), 1),
        "actionable": actionable,
        "failed": failed,
    }

def _breakout_status_mix(df: pd.DataFrame) -> list[dict]:
    """Ordered per-status counts for the status chips (primary states first)."""
    status = df.get("Status")
    if status is None:
        return []
    counts = status.value_counts()
    preferred = [
        "PRE_BREAKOUT",
        "BREAKOUT_CONFIRMED",
        "WATCH",
        "DEVELOPING",
        "FAILED_BREAKOUT",
    ]
    ordered = [s for s in preferred if s in counts.index]
    ordered += [s for s in counts.index if s not in preferred]
    return [{"status": s, "count": int(counts[s])} for s in ordered]

def _fetch_live_prices(symbols: list[str]) -> dict[str, float | str]:
    """Fetch last prices via yfinance ``fast_info``; missing symbols map to ``N/A``."""
    if not symbols:
        return {}
    try:
        # Create a batch query string (e.g., "DKNG GOOGL HLT")
        tickers_str = " ".join(symbols)
        tickers = yf.Tickers(tickers_str)
        
        prices = {}
        for sym in symbols:
            try:
                # fast_info fetches the live feed price rapidly without scraping overhead
                prices[sym] = round(tickers.tickers[sym].fast_info['last_price'], 2)
            except Exception:
                prices[sym] = "N/A"  # Fallback if ticker data fetch fails
        return prices
    except Exception as e:
        print(f"Error fetching live prices: {e}")
        return {sym: "N/A" for sym in symbols}

def _live_price_cell(live: float | str | None, close: float) -> str:
    """HTML cell for a live price: red if the close is above it, green if below, else neutral."""
    if not isinstance(live, (int, float)) or pd.isna(live):
        return '<span class="live-price-cell live-flat">N/A</span>'
    css = "live-flat"
    if pd.notna(close):
        if close > live:
            css = "live-below-close"
        elif close < live:
            css = "live-above-close"
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
            
            # 2. Add 'Live Price' values aligned with symbols
            df['Live Price'] = df[symbol_col].apply(
                lambda x: f'<span class="live-price-cell">${live_price_map.get(str(x).strip().upper(), "N/A")}</span>'
                if pd.notna(x) else ""
            )
            
            # 3. Restructure layout: Inject 'Live Price' right after the 'Symbol' column
            cols = list(df.columns)
            symbol_idx = cols.index(symbol_col)
            cols.insert(symbol_idx + 1, cols.pop(cols.index('Live Price')))
            df = df[cols]
            
            # 4. Convert plain strings into operational Finviz anchor links
            df[symbol_col] = df[symbol_col].apply(
                lambda x: f'<a href="https://finviz.com/quote.ashx?t={str(x).strip().upper()}" target="_blank" rel="noopener noreferrer" class="ticker-link">{x}</a>'
                if pd.notna(x) else ""
            )
            
        # escape=False ensures Pandas treats custom injected HTML elements cleanly
        table_html = df.to_html(classes="table", index=False, border=0, escape=False)
        return render_template("view.html", mode=mode, date=date, file_name=requested_file, table_html=table_html)
    except Exception as e:
        return f"<h3>❌ Failed to parse data contents:</h3><pre>{str(e)}</pre>", 500

@app.route("/breakout")
def breakout_index() -> str:
    """List available breakout scan runs (per mode, newest first)."""
    runs = _get_breakout_runs()
    return render_template("breakout_index.html", runs=runs)

@app.route("/breakout/<mode>/<date>")
def breakout_view(mode: str, date: str) -> str | tuple[str, int]:
    """Render one breakout scan CSV: KPI counts, status chips, and the table.

    Read-only. The raw CSV is never rewritten. ``?status=`` filters the table
    only; KPIs and the status mix are always computed on the full file.
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

        kpis = _breakout_kpis(df)
        status_mix = _breakout_status_mix(df)

        # Table view: known display columns, sorted by readiness then symbol.
        view_cols = [c for c in BREAKOUT_DISPLAY_COLUMNS if c in df.columns]
        table_df = df[view_cols].copy() if view_cols else df.copy()

        if "Breakout Readiness" in table_df.columns:
            table_df["Breakout Readiness"] = pd.to_numeric(
                table_df["Breakout Readiness"], errors="coerce"
            )

        # Status filter applies to the table only (KPIs stay full-file).
        active_status = request.args.get("status")
        if active_status and "Status" in table_df.columns:
            table_df = table_df[table_df["Status"] == active_status]

        sort_cols = [c for c in ["Breakout Readiness", "Symbol"] if c in table_df.columns]
        if sort_cols:
            table_df = table_df.sort_values(
                sort_cols,
                ascending=[c == "Symbol" for c in sort_cols],
            )

        # Live price next to Close, coloured by live vs. close (only visible rows are quoted).
        if "Symbol" in table_df.columns and "Close" in table_df.columns:
            symbols = [str(x).strip().upper() for x in table_df["Symbol"].dropna().unique()]
            live_price_map = _fetch_live_prices(symbols)
            closes = pd.to_numeric(table_df["Close"], errors="coerce")
            table_df["Live Price"] = [
                _live_price_cell(live_price_map.get(str(sym).strip().upper()), close)
                if pd.notna(sym) else ""
                for sym, close in zip(table_df["Symbol"], closes)
            ]
            cols = list(table_df.columns)
            cols.insert(cols.index("Close") + 1, cols.pop(cols.index("Live Price")))
            table_df = table_df[cols]

        # Finviz quote links on the Symbol column (matches trade-plan view).
        if "Symbol" in table_df.columns:
            table_df["Symbol"] = table_df["Symbol"].apply(
                lambda x: f'<a href="https://finviz.com/quote.ashx?t={str(x).strip().upper()}" '
                f'target="_blank" rel="noopener noreferrer" class="ticker-link">{x}</a>'
                if pd.notna(x) else ""
            )

        table_html = table_df.to_html(classes="table", index=False, border=0, escape=False)
        return render_template(
            "breakout_view.html",
            mode=mode,
            date=date,
            file_name=file_name,
            kpis=kpis,
            status_mix=status_mix,
            active_status=active_status,
            row_count=int(len(table_df)),
            table_html=table_html,
        )
    except Exception as e:
        return f"<h3>❌ Failed to parse data contents:</h3><pre>{str(e)}</pre>", 500

if __name__ == "__main__":
    # Ensure standard fallback logs dirs exist locally
    for path in list(MODES.values()) + list(BREAKOUT_MODES.values()):
        os.makedirs(path, exist_ok=True)
        
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    print("🚀 Launching Upgraded Finance Vibe UI Dashboard Context...")
    app.run(host="0.0.0.0", port=5000, debug=debug)