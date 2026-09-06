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

if __name__ == "__main__":
    # Ensure standard fallback logs dirs exist locally
    for path in MODES.values():
        os.makedirs(path, exist_ok=True)
        
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    print("🚀 Launching Upgraded Finance Vibe UI Dashboard Context...")
    app.run(host="0.0.0.0", port=5000, debug=debug)