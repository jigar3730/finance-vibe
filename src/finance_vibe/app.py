import glob
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from flask import Flask, abort, render_template_string, request

try:
    import markdown as md
except ImportError:  # pragma: no cover - declared in requirements.txt
    md = None

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGS_BASE_DIR = os.path.join(BASE_DIR, "data", "logs")

MODES = {
    "daily": os.path.join(LOGS_BASE_DIR, "daily"),
    "weekly": os.path.join(LOGS_BASE_DIR, "weekly"),
}

# Allowlisted markdown only — never serve data/, .env, or unlisted files.
DOC_CATALOG = {
    "learn": {
        "file": "Learn.md",
        "title": "Learn (curriculum)",
        "blurb": "Three tracks: TA, the system, and ML.",
    },
    "learn-ta": {
        "file": "LearnTA.md",
        "title": "TA primer (Cobra)",
        "blurb": "OHLCV, EMA, ATR, MACD, RSI, RS vs QQQ.",
    },
    "learn-ml": {
        "file": "LearnML.md",
        "title": "ML primer",
        "blurb": "Leakage, splits, boosting, Spearman, fail-soft rank.",
    },
    "rubric": {
        "file": "Coiled Cobra Rubric .MD",
        "title": "Coiled Cobra rubric",
        "blurb": "Live scorecard, pillars, and hard gates.",
    },
    "trade-plan": {
        "file": "Trade Plan Calculations.md",
        "title": "Trade plan math",
        "blurb": "Close entry, Coil_Low stop, 2R / 3R.",
    },
    "mlops": {
        "file": "MLOps.md",
        "title": "MLOps runbook",
        "blurb": "Docker train, evaluate, deploy.",
    },
    "cobra-ml": {
        "file": "CoiledCobraML.md",
        "title": "ML feature contract",
        "blurb": "Ten features, targets, metrics tables.",
    },
    "ops": {
        "file": "OperationManual.md",
        "title": "Operation manual",
        "blurb": "SOP, troubleshooting, pipeline order.",
    },
    "backtest": {
        "file": "BacktestAndBackfill.md",
        "title": "Backtest and backfill",
        "blurb": "How Rel_Forward labels are produced.",
    },
    "vibe": {
        "file": "src/finance_vibe/Scoring_Logic.md",
        "title": "Macro vibe (offline)",
        "blurb": "SMA / MACD / RSI score — not live Cobra.",
    },
    "swing": {
        "file": "swing_setup_readme.md",
        "title": "Quality swing (offline)",
        "blurb": "Pullback scanner — not in run_vibe.",
    },
}

LEARN_TRACKS = [
    {
        "title": "Track A — Markets and TA",
        "text": "OHLCV, QQQ, EMA/ATR/MACD, relative strength, coil vs pullback.",
        "href": "/docs/learn-ta",
    },
    {
        "title": "Track B — The system",
        "text": "Universe, walk-forward, Score vs gates vs ML_Rank, Docker volumes.",
        "href": "/docs/learn",
    },
    {
        "title": "Track C — AI / ML",
        "text": "Leakage, embargo, boosting, Spearman, fail-soft inference.",
        "href": "/docs/learn-ml",
    },
]


def daily_ml_model_found() -> bool:
    return os.path.isfile(
        os.path.join(LOGS_BASE_DIR, "daily", "coiled_cobra_xgb_model.json")
    )


def resolve_doc_path(slug: str) -> Path:
    """Return an allowlisted markdown path under BASE_DIR, or abort."""
    spec = DOC_CATALOG.get(slug)
    if spec is None:
        abort(404)
    base = Path(BASE_DIR).resolve()
    path = (base / spec["file"]).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        abort(404)
    if not path.is_file():
        abort(404)
    return path


def render_markdown_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if md is None:
        return "<pre>" + text.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
    return md.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )


def get_available_runs():
    """List dated trade plans; prefer trade_plan_clean_* when both exist."""
    runs = {"daily": [], "weekly": []}

    for mode, folder_path in MODES.items():
        if not os.path.exists(folder_path):
            continue

        by_date: dict[str, dict] = {}
        for file_path in glob.glob(os.path.join(folder_path, "trade_plan_*.csv")):
            file_name = os.path.basename(file_path)
            parts = (
                file_name.replace("trade_plan_clean_", "")
                .replace("trade_plan_", "")
                .replace(".csv", "")
            )
            try:
                datetime.strptime(parts[:10], "%Y-%m-%d")
            except ValueError:
                continue
            date_str = parts[:10]
            is_clean = "clean" in file_name
            prev = by_date.get(date_str)
            if prev is None or (is_clean and not prev["is_clean"]):
                by_date[date_str] = {
                    "date": date_str,
                    "file_name": file_name,
                    "is_clean": is_clean,
                }

        runs[mode] = [by_date[k] for k in sorted(by_date.keys(), reverse=True)]

    return runs


def fetch_live_prices(symbols):
    """Fetches real-time prices efficiently using yfinance fast_info."""
    if not symbols:
        return {}
    try:
        tickers_str = " ".join(symbols)
        tickers = yf.Tickers(tickers_str)

        prices = {}
        for sym in symbols:
            try:
                prices[sym] = round(tickers.tickers[sym].fast_info["last_price"], 2)
            except Exception:
                prices[sym] = "N/A"
        return prices
    except Exception as e:
        print(f"Error fetching live prices: {e}")
        return {sym: "N/A" for sym in symbols}


SHARED_CSS = """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #fdfdfd; color: #333; }
        .nav { display: flex; align-items: center; gap: 18px; padding: 12px 40px; border-bottom: 1px solid #eaeaea; background: #fff; }
        .nav a { color: #0366d6; text-decoration: none; font-weight: 500; }
        .nav a:hover { text-decoration: underline; }
        .nav a.active { color: #111; font-weight: 700; text-decoration: none; }
        .nav-brand { font-weight: 700; color: #111; margin-right: 8px; }
        .ml-chip { margin-left: auto; font-size: 0.8em; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
        .ml-chip.found { background: #e8f5e9; color: #2e7d32; }
        .ml-chip.missing { background: #fff3e0; color: #e65100; }
        .page { margin: 32px 40px 48px; }
        h1 { color: #111; border-bottom: 2px solid #eaeaea; padding-bottom: 10px; }
        h2 { color: #444; margin-top: 28px; text-transform: uppercase; font-size: 1.05em; letter-spacing: 0.4px; }
        a { color: #0366d6; text-decoration: none; font-weight: 500; }
        a:hover { text-decoration: underline; }
        .container { display: flex; gap: 24px; flex-wrap: wrap; }
        .column { flex: 1; min-width: 240px; background: #fff; padding: 20px; border-radius: 6px; border: 1px solid #e1e4e8; }
        ul.runs { list-style: none; padding: 0; }
        ul.runs li { padding: 10px; border-bottom: 1px solid #f1f1f1; display: flex; justify-content: space-between; align-items: center; }
        .badge { font-size: 0.8em; padding: 3px 8px; border-radius: 12px; background: #e1f5fe; color: #0288d1; font-weight: bold; }
        .badge.clean { background: #e8f5e9; color: #2e7d32; }
        .empty-help { color: #555; line-height: 1.5; }
        .legend { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px; padding: 12px 14px; margin: 12px 0 20px; font-size: 0.92em; line-height: 1.45; color: #444; }
        .docs-body { max-width: 900px; line-height: 1.55; }
        .docs-body table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.92em; }
        .docs-body th, .docs-body td { border: 1px solid #e1e4e8; padding: 8px 10px; text-align: left; }
        .docs-body th { background: #f6f8fa; }
        .docs-body pre, .docs-body code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.88em; }
        .docs-body pre { background: #f6f8fa; padding: 12px; overflow-x: auto; border-radius: 6px; }
        .docs-body h1 { font-size: 1.6em; }
        .doc-list { list-style: none; padding: 0; }
        .doc-list li { padding: 12px 0; border-bottom: 1px solid #f1f1f1; }
        .track-card { margin-bottom: 8px; }
"""

NAV_HTML = """
    <nav class="nav">
        <span class="nav-brand">Finance Vibe</span>
        <a href="/" class="{{ 'active' if nav == 'plans' else '' }}">Plans</a>
        <a href="/learn" class="{{ 'active' if nav == 'learn' else '' }}">Learn</a>
        <a href="/docs" class="{{ 'active' if nav == 'docs' else '' }}">Docs</a>
        {% if ml_daily %}
        <span class="ml-chip found" title="coiled_cobra_xgb_model.json in data/logs/daily">ML daily: found</span>
        {% else %}
        <span class="ml-chip missing" title="Train into data/logs/daily — see MLOps">ML daily: missing</span>
        {% endif %}
    </nav>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Finance Vibe Dashboard</title>
    <style>""" + SHARED_CSS + """</style>
</head>
<body>
""" + NAV_HTML + """
    <div class="page">
    <h1>Execution hub</h1>
    <p>Daily is the live Coiled Cobra book. Cleaned plans are preferred when both files exist. Use <a href="/learn">Learn</a> for TA and ML internals.</p>

    <div class="container">
        <div class="column">
            <h2>Daily (5Y) — primary</h2>
            {% if runs['daily'] %}
                <ul class="runs">
                {% for run in runs['daily'] %}
                    <li>
                        <a href="/view/daily/{{ run['date'] }}?file={{ run['file_name'] }}">Trade plan ({{ run['date'] }})</a>
                        <span class="badge {% if run['is_clean'] %}clean{% endif %}">
                            {{ 'Cleaned' if run['is_clean'] else 'Raw' }}
                        </span>
                    </li>
                {% endfor %}
                </ul>
            {% else %}
                <p class="empty-help">No daily trade plans yet. In the <code>finance_vibe</code> container run ingest and scan
                (<a href="/docs/ops">operation manual</a>), then refresh. To attach ranks, train a daily model
                (<a href="/docs/mlops">MLOps</a>).</p>
            {% endif %}
        </div>

        <div class="column">
            <h2>Weekly (10Y)</h2>
            {% if runs['weekly'] %}
                <ul class="runs">
                {% for run in runs['weekly'] %}
                    <li>
                        <a href="/view/weekly/{{ run['date'] }}?file={{ run['file_name'] }}">Trade plan ({{ run['date'] }})</a>
                        <span class="badge {% if run['is_clean'] %}clean{% endif %}">
                            {{ 'Cleaned' if run['is_clean'] else 'Raw' }}
                        </span>
                    </li>
                {% endfor %}
                </ul>
            {% else %}
                <p class="empty-help">No weekly plans. Weekly is the slower confirmation horizon — see
                <a href="/docs/ops">ops</a>.</p>
            {% endif %}
        </div>

        <div class="column">
            <h2>Learn</h2>
            {% for t in tracks %}
            <p class="track-card"><a href="{{ t.href }}">{{ t.title }}</a><br>
            <span style="color:#666;font-size:0.9em;">{{ t.text }}</span></p>
            {% endfor %}
            <p><a href="/docs">All allowlisted docs</a> · <a href="/docs/learn">Full curriculum</a></p>
        </div>
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
    <style>""" + SHARED_CSS + """
        .table-container { width: 100%; overflow-x: auto; background: #fff; border: 1px solid #e1e4e8; border-radius: 6px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9em; }
        th { background: #f6f8fa; padding: 12px; border-bottom: 2px solid #e1e4e8; font-weight: 600; text-transform: capitalize; }
        td { padding: 10px 12px; border-bottom: 1px solid #eaecef; }
        tr:hover { background: #f8f9fa; }
        .ticker-link { color: #0056b3; text-decoration: none; font-weight: bold; }
        .live-price-cell { font-weight: 600; color: #2e7d32; background-color: #f4faf4; }
        .meta { color: #666; margin-bottom: 8px; font-size: 0.95em; }
    </style>
</head>
<body>
""" + NAV_HTML + """
    <div class="page">
    <h1>Trade plan preview</h1>
    <div class="meta">Cadence: <strong>{{ mode|upper }}</strong> | Date: <strong>{{ date }}</strong> | File: <code>{{ file_name }}</code></div>
    <div class="legend">
        <strong>How to read this table:</strong>
        Entry is the coil bar <em>Close</em>. Stop protects <em>Coil_Low</em> (ATR and 5% caps).
        Targets are <em>2R</em> and <em>3R</em>. <em>Score</em> is the rubric gate (already passed).
        <em>ML_Rank</em> 1 = highest predicted 2-week return vs QQQ — a sort, not a gate.
        <a href="/docs/rubric">Rubric</a> ·
        <a href="/docs/trade-plan">Level math</a> ·
        <a href="/docs/learn-ml">Learn ML</a>
    </div>
    <div class="table-container">
        {{ table_html|safe }}
    </div>
    </div>
</body>
</html>
"""

LEARN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Learn — Finance Vibe</title>
    <style>""" + SHARED_CSS + """</style>
</head>
<body>
""" + NAV_HTML + """
    <div class="page">
    <h1>Learn</h1>
    <p>Operate on <a href="/">Plans</a>. Study here. Full index with Docker commands:
    <a href="/docs/learn">Learn.md</a>.</p>
    <div class="container">
        {% for t in tracks %}
        <div class="column">
            <h2>{{ t.title }}</h2>
            <p>{{ t.text }}</p>
            <p><a href="{{ t.href }}">Open primer / index</a></p>
        </div>
        {% endfor %}
    </div>
    <p style="margin-top:24px;color:#555;">Specs:
        <a href="/docs/rubric">Rubric</a> ·
        <a href="/docs/mlops">MLOps</a> ·
        <a href="/docs/ops">Operations</a> ·
        Offline lab: <a href="/docs/vibe">vibe</a>, <a href="/docs/swing">swing</a>.
    </p>
    </div>
</body>
</html>
"""

DOCS_INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Docs — Finance Vibe</title>
    <style>""" + SHARED_CSS + """</style>
</head>
<body>
""" + NAV_HTML + """
    <div class="page">
    <h1>Documentation</h1>
    <p>Allowlisted files only (curriculum and ops). Worklogs and review notes are not served.</p>
    <ul class="doc-list">
        {% for slug, spec in catalog %}
        <li>
            <a href="/docs/{{ slug }}">{{ spec.title }}</a>
            <div style="color:#666;font-size:0.9em;">{{ spec.blurb }}</div>
        </li>
        {% endfor %}
    </ul>
    </div>
</body>
</html>
"""

DOC_VIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} — Finance Vibe</title>
    <style>""" + SHARED_CSS + """</style>
</head>
<body>
""" + NAV_HTML + """
    <div class="page">
    <p class="meta"><a href="/docs">Docs</a> / {{ title }}</p>
    <div class="docs-body">
        {{ body|safe }}
    </div>
    </div>
</body>
</html>
"""


def _chrome(nav: str) -> dict:
    return {"nav": nav, "ml_daily": daily_ml_model_found()}


@app.route("/")
def index():
    runs = get_available_runs()
    return render_template_string(
        INDEX_TEMPLATE,
        runs=runs,
        tracks=LEARN_TRACKS,
        **_chrome("plans"),
    )


@app.route("/learn")
def learn_hub():
    return render_template_string(
        LEARN_TEMPLATE,
        tracks=LEARN_TRACKS,
        **_chrome("learn"),
    )


@app.route("/docs")
def docs_index():
    return render_template_string(
        DOCS_INDEX_TEMPLATE,
        catalog=list(DOC_CATALOG.items()),
        **_chrome("docs"),
    )


@app.route("/docs/<slug>")
def docs_view(slug: str):
    # Slashes never appear in allowlisted slugs; traversal tokens 404.
    if slug not in DOC_CATALOG:
        abort(404)
    path = resolve_doc_path(slug)
    spec = DOC_CATALOG[slug]
    body = render_markdown_file(path)
    return render_template_string(
        DOC_VIEW_TEMPLATE,
        title=spec["title"],
        body=body,
        **_chrome("docs"),
    )


@app.route("/view/<mode>/<date>")
def view_run(mode, date):
    if mode not in MODES:
        abort(404, "Invalid historical directory mode context.")

    requested_file = request.args.get("file")
    if not requested_file:
        abort(400, "Missing reference log file parameter.")

    target_path = os.path.abspath(os.path.join(MODES[mode], requested_file))
    if not target_path.startswith(os.path.abspath(MODES[mode])):
        abort(403, "Access restricted outside authorized mode workspace boundaries.")

    if not os.path.exists(target_path):
        abort(404, f"The selected file record does not exist: {requested_file}")

    try:
        df = pd.read_csv(target_path)

        df.columns = df.columns.str.strip()

        symbol_col = None
        for col in df.columns:
            if col.lower() in ["symbol", "ticker"]:
                symbol_col = col
                break

        if symbol_col is not None:
            raw_symbols = [str(x).strip().upper() for x in df[symbol_col].dropna().unique()]
            live_price_map = fetch_live_prices(raw_symbols)

            df["Live Price"] = df[symbol_col].apply(
                lambda x: f'<span class="live-price-cell">${live_price_map.get(str(x).strip().upper(), "N/A")}</span>'
                if pd.notna(x) else ""
            )

            cols = list(df.columns)
            symbol_idx = cols.index(symbol_col)
            cols.insert(symbol_idx + 1, cols.pop(cols.index("Live Price")))
            df = df[cols]

            df[symbol_col] = df[symbol_col].apply(
                lambda x: f'<a href="https://finance.yahoo.com/quote/{str(x).strip().upper()}" target="_blank" rel="noopener noreferrer" class="ticker-link">{x}</a>'
                if pd.notna(x) else ""
            )

        table_html = df.to_html(classes="table", index=False, border=0, escape=False)
        return render_template_string(
            VIEW_TEMPLATE,
            mode=mode,
            date=date,
            file_name=requested_file,
            table_html=table_html,
            **_chrome("plans"),
        )
    except Exception as e:
        return f"<h3>Failed to parse data contents:</h3><pre>{str(e)}</pre>", 500


if __name__ == "__main__":
    for path in MODES.values():
        os.makedirs(path, exist_ok=True)

    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    print("Launching Finance Vibe dashboard (Plans / Learn / Docs)...")
    app.run(host="0.0.0.0", port=5000, debug=debug)
