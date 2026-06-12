from pathlib import Path
import re
from functools import lru_cache
import pandas as pd
from flask import Flask, render_template_string, abort, send_from_directory

APP = Flask(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "data" / "logs"
DATE_PATTERN = re.compile(r".*_(\d{4}-\d{2}-\d{2})\.csv$")

# Modern, high-scannability UX dashboard template
TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Finance Vibe Dashboard</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { 
        font-family: 'Inter', -apple-system, sans-serif; 
        background-color: #f8f9fa; 
        color: #212529;
        padding-top: 2rem; 
        padding-bottom: 3rem; 
      }
      .card {
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,.02);
      }
      .card-header-custom {
        background-color: #fff;
        border-bottom: 1px solid rgba(0,0,0,.08);
        padding: 1rem 1.25rem;
      }
      .mono { 
        font-family: 'JetBrains Mono', monospace; 
        font-size: 0.870em;
      }
      .table {
        font-size: 0.9rem;
        vertical-align: middle;
      }
      .table th {
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
        color: #6c757d;
        background-color: #f8f9fa;
      }
      .summary-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #0d6efd;
      }
    </style>
  </head>
  <body>
    <div class="container-xl">
      
      <header class="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center mb-4 pb-3 border-bottom gap-3">
        <div>
          <h1 class="h3 fw-bold mb-1">Finance Vibe Dashboard</h1>
          <p class="text-muted mb-0">Latest weekly pipeline metrics and historical output previews.</p>
        </div>
        <div>
          <a href="/" class="btn btn-white btn-sm border bg-white shadow-sm fw-medium">🔄 Refresh Dashboard</a>
        </div>
      </header>

      <div class="row g-3 mb-4">
        <div class="col-md-6 col-lg-4">
          <div class="card h-100">
            <div class="card-body d-flex flex-column justify-content-between">
              <div>
                <h2 class="h6 text-muted text-uppercase fw-semibold mb-3">Latest Pipeline Run</h2>
                {% if latest_date %}
                  <div class="mb-2"><span class="badge bg-primary-subtle text-primary fw-medium mono">{{ latest_date }}</span></div>
                  <div class="row g-2 pt-1 small">
                    <div class="col-7 text-muted">Swing Setups:</div>
                    <div class="col-5 text-end fw-semibold mono">{{ latest_counts.swing_setups }}</div>
                    <div class="col-7 text-muted">Clean Trade Plans:</div>
                    <div class="col-5 text-end fw-semibold mono">{{ latest_counts.trade_plan_clean }}</div>
                  </div>
                {% else %}
                  <p class="text-muted mb-0">No historical run data detected.</p>
                {% endif %}
              </div>
              {% if latest_date %}
                <a href="/run/{{ latest_date }}" class="btn btn-primary btn-sm w-100 mt-3 fw-medium">View Latest Dataset</a>
              {% endif %}
            </div>
          </div>
        </div>

        <div class="col-md-6 col-lg-4">
          <div class="card h-100">
            <div class="card-body">
              <h2 class="h6 text-muted text-uppercase fw-semibold mb-3">Historic Archive</h2>
              <div class="summary-value mb-1">{{ summary|length }}</div>
              <p class="text-muted small mb-0">Total historical snapshots processed and verified in the current pipeline directory.</p>
            </div>
          </div>
        </div>

        <div class="col-md-12 col-lg-4">
          <div class="card h-100">
            <div class="card-body">
              <h2 class="h6 text-muted text-uppercase fw-semibold mb-2">Environment Meta</h2>
              <div class="small pt-1">
                <span class="text-muted d-block mb-1">Active Input Schema:</span>
                <code class="d-block bg-light p-2 rounded border text-break mb-2 mono">..._[YYYY-MM-DD].csv</code>
                <span class="text-muted d-block mb-1">Target Engine Directory:</span>
                <span class="mono text-dark text-break small bg-light p-1 px-2 border rounded d-block">{{ logs_dir }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="row g-4">
        <div class="col-xl-4 col-lg-5">
          <div class="card">
            <div class="card-header-custom">
              <h2 class="h6 fw-bold mb-0">Historical Log Pipeline</h2>
            </div>
            <div class="table-responsive" style="max-height: 600px; overflow-y: auto;">
              {{ summary_table|safe }}
            </div>
          </div>
        </div>

        <div class="col-xl-8 col-lg-7">
          <div class="card">
            <div class="card-header-custom">
              <h2 class="h6 fw-bold mb-0">Data Stream Preview</h2>
            </div>
            <div class="card-body">
              {% if details %}
                <div class="alert alert-info bg-light border text-dark py-2 px-3 mb-4 d-flex justify-content-between align-items-center small">
                  <div>Focus Target: <strong class="mono">{{ details.date }}</strong></div>
                  <span class="text-muted">Showing first 20 records</span>
                </div>
                
                {% for section in details.tables %}
                  <div class="mb-5 border rounded bg-white">
                    <div class="d-flex justify-content-between align-items-center bg-light p-2 px-3 border-bottom">
                      <h3 class="h6 mb-0 fw-semibold text-secondary mono">{{ section.title }}</h3>
                      <a href="/download/{{ section.filename }}" class="btn btn-sm btn-outline-primary py-0 px-2 small font-monospace" style="font-size:0.8rem;">
                        📥 Download CSV
                      </a>
                    </div>
                    <div class="table-responsive p-2" style="max-height: 350px;">
                      {{ section.html|safe }}
                    </div>
                  </div>
                {% endfor %}
              {% else %}
                <div class="text-center py-5 text-muted">
                  <p class="mb-0">Select a snapshot version from the active tracking index table to map operational metrics.</p>
                </div>
              {% endif %}
            </div>
          </div>
        </div>
      </div>

      <footer class="text-muted text-center small mt-5 pt-3 border-top">
        Finance Vibe Internal Management Dashboard Engine • 2026 Production Environment
      </footer>
    </div>
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


# Avoid crashing on reading raw dataframes; optimize presentation classes
def load_csv_preview(path: Path, max_rows: int = 20):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return '<p class="text-muted p-3 mb-0">This CSV file contains no data fields.</p>'
        preview = df.head(max_rows)
        return preview.to_html(
            classes="table table-sm table-striped table-hover mb-0 mono align-middle border-top-0", 
            index=False, 
            border=0, 
            justify="left"
        )
    except Exception as exc:
        return f'<div class="alert alert-danger m-3">Failed to load payload {path.name}: {exc}</div>'


def get_run_details(date):
    files = []
    file_manifest = [
        (f"swing_setups_{date}.csv", "Swing Setups"),
        (f"trade_plan_{date}.csv", "Trade Plan"),
        (f"trade_plan_clean_{date}.csv", "Clean Trade Plan"),
        (f"vibe_report_{date}.csv", "Vibe Report"),
        (f"vibe_report_local_{date}.csv", "Local Vibe Report")
    ]
    for name, label in file_manifest:
        path = LOGS_DIR / name
        if path.exists():
            html = load_csv_preview(path)
            files.append({"title": label, "html": html, "filename": path.name})
    return {"date": date, "tables": files}


# Cache calculations for reading lines to maintain optimized server performance metrics
@lru_cache(maxsize=128)
def get_csv_length(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return "-"
    try:
        return str(len(pd.read_csv(path)))
    except Exception:
        return "Err"


def build_summary_table(dates, active_date=None):
    rows = []
    for date in dates:
        swing_path = LOGS_DIR / f"swing_setups_{date}.csv"
        clean_path = LOGS_DIR / f"trade_plan_clean_{date}.csv"
        
        # Pull length counts using cached helper
        swing_len = get_csv_length(str(swing_path))
        clean_len = get_csv_length(str(clean_path))
        
        is_active = "table-active fw-semibold" if date == active_date else ""
        
        rows.append({
            "Date": f'<span class="mono">{date}</span>',
            "Swing": f'<span class="mono">{swing_len}</span>',
            "Clean Plan": f'<span class="mono">{clean_len}</span>',
            "Action": f"<a href='/run/{date}' class='btn btn-xs btn-primary py-0 px-2 fw-medium' style='font-size:0.75rem;'>View</a>",
            "_css_class": is_active
        })
        
    if not rows:
        return '<p class="text-muted p-3 text-center mb-0">No active runs recorded.</p>'

    # Programmatically convert to beautiful HTML table with state tracking variables
    summary_df = pd.DataFrame(rows)
    css_classes = summary_df["_css_class"].tolist()
    summary_df = summary_df.drop(columns=["_css_class"])
    
    html = summary_df.to_html(classes="table table-hover mb-0 align-middle", index=False, escape=False, border=0, justify="left")
    
    # Inject active row tracking classes natively into bootstrap generation
    if active_date:
        for date, css in zip(dates, css_classes):
            if css:
                html = html.replace(f'<td><span class="mono">{date}</span></td>', f'<td class="{css}"><span class="mono">{date}</span></td>')
                
    return html


@APP.route("/download/<path:filename>")
def download_file(filename):
    safe_path = (LOGS_DIR / filename).resolve()
    if not safe_path.exists() or LOGS_DIR.resolve() not in safe_path.parents:
        abort(404, description="File access violation or target file not found.")
    return send_from_directory(LOGS_DIR, filename, as_attachment=True)


@APP.route("/")
def home():
    dates = available_dates()
    latest_date = get_latest_date()
    
    latest_counts = {
        "swing_setups": get_csv_length(str(LOGS_DIR / f"swing_setups_{latest_date}.csv")) if latest_date else "0",
        "trade_plan_clean": get_csv_length(str(LOGS_DIR / f"trade_plan_clean_{latest_date}.csv")) if latest_date else "0",
    }
    
    summary_table = build_summary_table(dates, active_date=latest_date)
    details = get_run_details(latest_date) if latest_date else None
    
    return render_template_string(
        TEMPLATE,
        latest_date=latest_date,
        latest_counts=latest_counts,
        summary=dates,
        summary_table=summary_table,
        details=details,
        logs_dir=str(LOGS_DIR)
    )


@APP.route("/run/<date>")
def run_detail(date):
    dates = available_dates()
    if date not in dates:
        abort(404, description=f"No processed run found containing date: {date}")
        
    latest_date = get_latest_date()
    latest_counts = {
        "swing_setups": get_csv_length(str(LOGS_DIR / f"swing_setups_{latest_date}.csv")) if latest_date else "0",
        "trade_plan_clean": get_csv_length(str(LOGS_DIR / f"trade_plan_clean_{latest_date}.csv")) if latest_date else "0",
    }
    
    details = get_run_details(date)
    summary_table = build_summary_table(dates, active_date=date)
    
    return render_template_string(
        TEMPLATE,
        latest_date=latest_date,
        latest_counts=latest_counts,
        summary=dates,
        summary_table=summary_table,
        details=details,
        logs_dir=str(LOGS_DIR)
    )


if __name__ == "__main__":
    # Internal dev server parameters
    APP.run(debug=True, host="0.0.0.0", port=5000)