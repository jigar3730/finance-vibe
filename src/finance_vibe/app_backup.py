from pathlib import Path
import re
from functools import lru_cache
import pandas as pd
from flask import Flask, render_template_string, abort, send_from_directory

APP = Flask(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "data" / "logs"
DATE_PATTERN = re.compile(r".*_(\d{4}-\d{2}-\d{2})\.csv$")

# Pro-grade Widescreen Financial Dashboard Layout
TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Finance Vibe Dashboard</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { 
        font-family: 'Inter', -apple-system, sans-serif; 
        background-color: #f3f4f6; 
        color: #111827;
        padding-top: 1rem; 
        padding-bottom: 2rem; 
      }
      .card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        background-color: #fff;
        margin-bottom: 1rem;
      }
      .card-header-custom {
        background-color: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
        padding: 0.75rem 1rem;
      }
      .mono { 
        font-family: 'JetBrains Mono', monospace; 
        font-size: 0.85rem;
      }
      .table {
        font-size: 0.825rem;
        vertical-align: middle;
        margin-bottom: 0;
      }
      .table th {
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.5px;
        color: #4b5563;
        background-color: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
        padding: 0.5rem 0.75rem;
      }
      .table td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #f3f4f6;
        white-space: nowrap; /* Prevents ugly text-wrapping inside data frames */
      }
      .summary-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2563eb;
      }
      .ticker-link {
        color: #2563eb;
        font-weight: 600;
        text-decoration: none;
        padding: 2px 6px;
        border-radius: 4px;
        background-color: rgba(37, 99, 235, 0.06);
        transition: all 0.1s ease-in-out;
      }
      .ticker-link:hover {
        background-color: #2563eb;
        color: #fff;
      }
      /* Compact styling for horizontal filter bar */
      .history-scroll-container {
        display: flex;
        overflow-x: auto;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        background-color: #fff;
      }
      .history-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.375rem 0.75rem;
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        text-decoration: none;
        color: #374151;
        font-size: 0.825rem;
        transition: all 0.1s ease;
      }
      .history-chip:hover {
        background-color: #e5e7eb;
        color: #111827;
      }
      .history-chip.active {
        background-color: #2563eb;
        border-color: #2563eb;
        color: #fff;
        font-weight: 500;
      }
      .history-chip.active .chip-count {
        background-color: rgba(255,255,255,0.2);
        color: #fff;
      }
      .chip-count {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        background-color: #e5e7eb;
        color: #4b5563;
        padding: 1px 4px;
        border-radius: 4px;
      }
      .tradingview-widget-container {
        height: 550px;
        width: 100%;
      }
    </style>
  </head>
  <body>
    <div class="container-fluid px-4">
      
      <header class="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center mb-3 pb-2 border-bottom gap-2">
        <div>
          <h1 class="h4 fw-bold mb-0 text-dark">Finance Vibe Dashboard</h1>
          <p class="text-muted small mb-0">Operational intelligence system & automated sequence parser.</p>
        </div>
        <div>
          <a href="/" class="btn btn-sm btn-white border bg-white shadow-sm fw-medium text-secondary">🔄 Hard Reset Workspace</a>
        </div>
      </header>

      <div class="row g-3 mb-3">
        <div class="col-md-4">
          <div class="card h-100 mb-0">
            <div class="card-body p-3 d-flex align-items-center justify-content-between">
              <div>
                <h2 class="h6 text-uppercase fw-bold text-muted small mb-1">Target Engine Sequence</h2>
                {% if latest_date %}
                  <div><span class="badge bg-primary-subtle text-primary fw-semibold mono" style="font-size:0.85rem;">{{ latest_date }}</span></div>
                {% else %}
                  <span class="text-muted small">Null Index</span>
                {% endif %}
              </div>
              {% if latest_date %}
                <div class="text-end small">
                  <div class="text-muted">Swing: <strong class="mono text-dark">{{ latest_counts.swing_setups }}</strong></div>
                  <div class="text-muted">Clean: <strong class="mono text-dark">{{ latest_counts.trade_plan_clean }}</strong></div>
                </div>
              {% endif %}
            </div>
          </div>
        </div>

        <div class="col-md-4">
          <div class="card h-100 mb-0">
            <div class="card-body p-3 d-flex align-items-center justify-content-between">
              <div>
                <h2 class="h6 text-uppercase fw-bold text-muted small mb-1">Historical Snapshots</h2>
                <div class="summary-value">{{ summary|length }} <span class="text-muted fs-6 fw-normal">snapshots found</span></div>
              </div>
            </div>
          </div>
        </div>

        <div class="col-md-4">
          <div class="card h-100 mb-0">
            <div class="card-body p-3 small d-flex flex-column justify-content-center">
              <div class="text-truncate text-muted"><span class="fw-semibold">Path:</span> <span class="mono text-dark bg-light px-1 border rounded">{{ logs_dir }}</span></div>
            </div>
          </div>
        </div>
      </div>

      <div class="card mb-3">
        <div class="card-header-custom py-2 px-3">
          <h2 class="h6 fw-bold mb-0 text-secondary small text-uppercase">Log Archive Sequence Selection Index</h2>
        </div>
        <div class="history-scroll-container">
          {{ horizontal_nav|safe }}
        </div>
      </div>

      <div class="card">
        <div class="card-header-custom d-flex justify-content-between align-items-center">
          <h2 class="h6 fw-bold mb-0">Active Snapshot Data Matrix: <span class="mono text-primary font-monospace fs-5">{% if details %}{{ details.date }}{% else %}No Focus Target Selected{% endif %}</span></h2>
          {% if details %}
            <span class="badge bg-light border text-secondary font-monospace small px-2 py-1">Display Bound: First 20 Rows</span>
          {% endif %}
        </div>
        <div class="card-body p-3">
          {% if details %}
            {% for section in details.tables %}
              <div class="mb-4 border rounded shadow-sm bg-white overflow-hidden">
                <div class="d-flex justify-content-between align-items-center bg-light px-3 py-2 border-bottom">
                  <h3 class="h6 mb-0 fw-bold text-dark mono">{{ section.title }}</h3>
                  <a href="/download/{{ section.filename }}" class="btn btn-xs btn-outline-primary py-0 px-2 fw-medium mono" style="font-size:0.75rem; height:22px; line-height:20px;">
                    📥 Download File
                  </a>
                </div>
                <div class="table-responsive w-100">
                  {{ section.html|safe }}
                </div>
              </div>
            {% endfor %}
          {% else %}
            <div class="text-center py-5 text-muted">
              <p class="mb-0">Select a snapshot timestamp variant string inside the control sequence vector above to bind processing threads.</p>
            </div>
          {% endif %}
        </div>
      </div>

      <div class="modal fade" id="chartModal" tabindex="-1" aria-labelledby="chartModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-xl modal-dialog-centered">
          <div class="modal-content border-0 shadow-lg" style="border-radius: 8px;">
            <div class="modal-header bg-light py-2 px-3">
              <h5 class="modal-title fw-bold small text-uppercase text-secondary" id="chartModalLabel">Technical Engine Analytics Frame</h5>
              <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body p-0">
              <div id="tradingview_workspace" class="tradingview-widget-container"></div>
            </div>
          </div>
        </div>
      </div>

      <footer class="text-muted text-center small mt-4 pt-2 border-top" style="font-size:0.75rem;">
        Finance Vibe Engine Infrastructure v2.4 • 2026 Runtime Environment
      </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script>
      document.addEventListener('DOMContentLoaded', function() {
        const chartModal = new bootstrap.Modal(document.getElementById('chartModal'));
        const modalTitle = document.getElementById('chartModalLabel');

        document.body.addEventListener('click', function(e) {
          if (e.target && e.target.classList.contains('ticker-link')) {
            e.preventDefault();
            const symbol = e.target.getAttribute('data-symbol').toUpperCase();
            modalTitle.innerHTML = `Technical Analysis Interface Workspace Vector: <span class="text-primary font-monospace fw-bold">${symbol}</span>`;
            chartModal.show();

            setTimeout(() => {
              new TradingView.widget({
                "width": "100%",
                "height": 550,
                "symbol": symbol,
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "light",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_workspace"
              });
            }, 150);
          }
        });
      });
    </script>
  </body>
</html>"""


def parse_date(path: Path):
    match = DATE_PATTERN.match(path.name)
    return match.group(1) if match else None


def available_dates():
    dates = set()
    patterns = ["swing_setups_*.csv", "trade_plan_clean_*.csv", "vibe_report_*.csv"]
    if not LOGS_DIR.exists():
        return []
    for pattern in patterns:
        for path in LOGS_DIR.glob(pattern):
            date = parse_date(path)
            if date:
                dates.add(date)
    return sorted(dates, reverse=True)


def get_latest_date():
    dates = available_dates()
    return dates[0] if dates else None


def load_csv_preview(path: Path, max_rows: int = 20):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return '<p class="text-muted p-3 mb-0">Target data frame matrix contains zero entries.</p>'
        
        preview = df.head(max_rows).copy()
        
        symbol_col = next((col for col in preview.columns if col.upper() == "SYMBOL"), None)
        if symbol_col:
            preview[symbol_col] = preview[symbol_col].apply(
                lambda s: f'<a href="#" class="ticker-link" data-symbol="{s}">{s}</a>' if pd.notna(s) else s
            )

        return preview.to_html(
            classes="table table-sm table-striped table-hover mb-0 mono border-top-0", 
            index=False, 
            border=0, 
            escape=False, 
            justify="left"
        )
    except Exception as exc:
        return f'<div class="alert alert-danger m-2 p-2 small">Error loading structural dataset file {path.name}: {exc}</div>'


def get_run_details(date):
    files = []
    file_manifest = [
        (f"swing_setups_{date}.csv", "Swing Setups Frame"),
        (f"trade_plan_{date}.csv", "Trade Plan Vector Matrix"),
        (f"trade_plan_clean_{date}.csv", "Optimized Core Trade Strategy"),
        (f"vibe_report_{date}.csv", "Structural Market Vibe Report"),
        (f"vibe_report_local_{date}.csv", "Localized Vibe Engine Log Execution")
    ]
    for name, label in file_manifest:
        path = LOGS_DIR / name
        if path.exists():
            html = load_csv_preview(path)
            files.append({"title": label, "html": html, "filename": path.name})
    return {"date": date, "tables": files}


@lru_cache(maxsize=128)
def get_csv_length(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return "-"
    try:
        return str(len(pd.read_csv(path)))
    except Exception:
        return "Err"


# Renders a sleek, responsive horizontal scroll bar component instead of a gigantic vertical table
def build_horizontal_navigation(dates, active_date=None):
    chips = []
    for date in dates:
        swing_path = LOGS_DIR / f"swing_setups_{date}.csv"
        swing_len = get_csv_length(str(swing_path))
        
        is_active = "active" if date == active_date else ""
        
        chips.append(
            f'<a href="/run/{date}" class="history-chip {is_active} mono">'
            f'<span>{date}</span>'
            f'<span class="chip-count">{swing_len}</span>'
            f'</a>'
        )
    if not chips:
        return '<span class="text-muted small px-3 py-1">No recorded logs found in target engine workspace directory variables.</span>'
    return "\\n".join(chips)


@APP.route("/download/<path:filename>")
def download_file(filename):
    safe_path = (LOGS_DIR / filename).resolve()
    if not safe_path.exists() or LOGS_DIR.resolve() not in safe_path.parents:
        abort(404, description="Target sandbox verification failed for requested asset.")
    return send_from_directory(LOGS_DIR, filename, as_attachment=True)


@APP.route("/")
def home():
    dates = available_dates()
    latest_date = get_latest_date()
    
    latest_counts = {
        "swing_setups": get_csv_length(str(LOGS_DIR / f"swing_setups_{latest_date}.csv")) if latest_date else "0",
        "trade_plan_clean": get_csv_length(str(LOGS_DIR / f"trade_plan_clean_{latest_date}.csv")) if latest_date else "0",
    }
    
    horizontal_nav = build_horizontal_navigation(dates, active_date=latest_date)
    details = get_run_details(latest_date) if latest_date else None
    
    return render_template_string(
        TEMPLATE,
        latest_date=latest_date,
        latest_counts=latest_counts,
        summary=dates,
        horizontal_nav=horizontal_nav,
        details=details,
        logs_dir=str(LOGS_DIR)
    )


@APP.route("/run/<date>")
def run_detail(date):
    dates = available_dates()
    if date not in dates:
        abort(404, description=f"Execution error finding directory reference matching date variable value: {date}")
        
    latest_date = get_latest_date()
    latest_counts = {
        "swing_setups": get_csv_length(str(LOGS_DIR / f"swing_setups_{latest_date}.csv")) if latest_date else "0",
        "trade_plan_clean": get_csv_length(str(LOGS_DIR / f"trade_plan_clean_{latest_date}.csv")) if latest_date else "0",
    }
    
    details = get_run_details(date)
    horizontal_nav = build_horizontal_navigation(dates, active_date=date)
    
    return render_template_string(
        TEMPLATE,
        latest_date=latest_date,
        latest_counts=latest_counts,
        summary=dates,
        horizontal_nav=horizontal_nav,
        details=details,
        logs_dir=str(LOGS_DIR)
    )


if __name__ == "__main__":
    APP.run(debug=True, host="0.0.0.0", port=5000)