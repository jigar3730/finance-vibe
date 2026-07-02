import os
import glob
from datetime import datetime
import pandas as pd
from flask import Flask, render_template_string, request, abort

app = Flask(__name__)

# Ensure absolute paths resolve relative to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGS_BASE_DIR = os.path.join(BASE_DIR, "data", "logs")

# The new dual-timeframe folder structure paths
MODES = {
    "weekly": os.path.join(LOGS_BASE_DIR, "weekly"),
    "daily": os.path.join(LOGS_BASE_DIR, "daily")
}

def get_available_runs():
    """Scans both weekly and daily logs folders to find all execution dates and files."""
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

# Embedded lightweight HTML templates for simple maintenance
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Finance Vibe Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background: #fdfdfd; color: #333; }
        h1 { color: #111; border-bottom: 2px solid #eaeaea; padding-bottom: 10px; }
        h2 { color: #444; margin-top: 30px; text-transform: uppercase; font-size: 1.1em; letter-spacing: 0.5px;}
        .container { display: flex; gap: 40px; }
        .column { flex: 1; background: #fff; padding: 20px; border-radius: 6px; border: 1px solid #e1e4e8; }
        ul { list-style: none; padding: 0; }
        li { padding: 10px; border-bottom: 1px solid #f1f1f1; display: flex; justify-content: space-between; align-items: center; }
        li:last-child { border: none; }
        a { color: #0366d6; text-decoration: none; font-weight: 500; }
        a:hover { text-decoration: underline; }
        .badge { font-size: 0.8em; padding: 3px 8px; border-radius: 12px; background: #e1f5fe; color: #0288d1; font-weight: bold; }
        .badge.clean { background: #e8f5e9; color: #2e7d32; }
    </style>
</head>
<body>
    <h1>🚀 Finance Vibe Execution Hub</h1>
    <p>Select an isolated timeframe run to inspect detected trade plans and scanning outputs.</p>
    
    <div class="container">
        <div class="column">
            <h2>📅 Weekly Framework (10Y Lookback)</h2>
            {% if runs['weekly'] %}
                <ul>
                {% for run in runs['weekly'] %}
                    <li>
                        <a href="/view/weekly/{{ run['date'] }}?file={{ run['file_name'] }}">Trade Plan ({{ run['date'] }})</a>
                        <span class="badge {% if run['is_clean'] %}clean{% endif %}">
                            {{ 'Cleaned' if run['is_clean'] else 'Raw' }}
                        </span>
                    </li>
                {% endfor %}
                </ul>
            {% else %}
                <p style="color:#777;">No weekly log files discovered yet.</p>
            {% endif %}
        </div>
        
        <div class="column">
            <h2>⚡ Daily Framework (2Y Lookback)</h2>
            {% if runs['daily'] %}
                <ul>
                {% for run in runs['daily'] %}
                    <li>
                        <a href="/view/daily/{{ run['date'] }}?file={{ run['file_name'] }}">Trade Plan ({{ run['date'] }})</a>
                        <span class="badge {% if run['is_clean'] %}clean{% endif %}">
                            {{ 'Cleaned' if run['is_clean'] else 'Raw' }}
                        </span>
                    </li>
                {% endfor %}
                </ul>
            {% else %}
                <p style="color:#777;">No daily log files discovered yet.</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

VIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>View Plan - {{ date }} ({{ mode }})</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; background: #fdfdfd; }
        h1 { color: #111; margin-bottom: 5px; }
        .meta { color: #666; margin-bottom: 20px; font-size: 0.95em; }
        .back { display: inline-block; margin-bottom: 20px; color: #0366d6; text-decoration: none; font-weight: 500; }
        .table-container { width: 100%; overflow-x: auto; background: #fff; border: 1px solid #e1e4e8; border-radius: 6px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9em; }
        th { background: #f6f8fa; padding: 12px; border-bottom: 2px solid #e1e4e8; font-weight: 600; text-transform: capitalize; }
        td { padding: 10px 12px; border-bottom: 1px solid #eaecef; }
        tr:hover { background: #f8f9fa; }
        
        /* Interactive ticker links layout modifications */
        .ticker-link {
            color: #1a73e8; 
            text-decoration: none;
            font-weight: bold;
        }
        .ticker-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <a class="back" href="/">← Back to Dashboard</a>
    <h1>📋 Trade Plan Preview</h1>
    <div class="meta">Execution Cadence: <strong>{{ mode|upper }}</strong> | Log Date: <strong>{{ date }}</strong> | Source File: <code>{{ file_name }}</code></div>
    
    <div class="table-container">
        {{ table_html|safe }}
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    runs = get_available_runs()
    return render_template_string(INDEX_TEMPLATE, runs=runs)

@app.route("/view/<mode>/<date>")
def view_run(mode, date):
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
        
        # Clean whitespaces out of column definitions to ensure formatting accuracy
        df.columns = df.columns.str.strip()
        
        # Target the ticker symbol column dynamically regardless of exact capitalization variations
        symbol_col = None
        for col in df.columns:
            if col.lower() in ["symbol", "ticker"]:
                symbol_col = col
                break
                
        if symbol_col is not None:
            # Transform the plain strings into fully custom operational anchor tags
            # Format pattern fits: https://www.google.com/finance/beta/quote/ECL:NYSE
            df[symbol_col] = df[symbol_col].apply(
                lambda x: f'<a href="https://www.google.com/finance/beta/quote/{str(x).strip().upper()}:NYSE" target="_blank" rel="noopener noreferrer" class="ticker-link">{x}</a>'
                if pd.notna(x) else ""
            )
            
        # Crucial configuration shift: escape=False ensures Pandas treats raw HTML string injection as valid code blocks
        table_html = df.to_html(classes="table", index=False, border=0, escape=False)
        return render_template_string(VIEW_TEMPLATE, mode=mode, date=date, file_name=requested_file, table_html=table_html)
    except Exception as e:
        return f"<h3>❌ Failed to parse data contents:</h3><pre>{str(e)}</pre>", 500

if __name__ == "__main__":
    # Ensure standard fallback logs dirs exist locally
    for path in MODES.values():
        os.makedirs(path, exist_ok=True)
        
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    print("🚀 Launching Upgraded Finance Vibe UI Dashboard Context...")
    app.run(host="0.0.0.0", port=5000, debug=debug)