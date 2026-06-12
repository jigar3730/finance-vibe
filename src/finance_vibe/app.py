from pathlib import Path
import re
import pandas as pd
from flask import Flask, render_template_string, abort, send_from_directory

APP = Flask(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "data" / "logs"
DATE_PATTERN = re.compile(r".*_(\d{4}-\d{2}-\d{2})\.csv$")

TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Finance Vibe Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { padding-top: 1.5rem; padding-bottom: 1.5rem; }
      .table-wrap { margin-top: 1rem; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .summary-card { min-height: 130px; }
      .file-link { font-family: ui-monospace, monospace; font-size: 0.95rem; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="d-flex flex-column flex-md-row justify-content-between align-items-start mb-4 gap-3">
        <div>
          <h1 class="h3 mb-1">Finance Vibe Dashboard</h1>
          <p class="text-muted mb-0">Latest weekly pipeline results plus historic output previews.</p>
        </div>
        <div class="text-end">
          <a href="/" class="btn btn-outline-primary btn-sm">Home</a>
        </div>
      </div>

      <div class="row g-3 mb-4">
        <div class="col-md-4">
          <div class="card summary-card">
            <div class="card-body">
              <h2 class="h6">Latest Run</h2>
              {% if latest_date %}
                <p class="mb-1"><strong>Date:</strong> {{ latest_date }}</p>
                <p class="mb-1"><strong>Swing setups:</strong> {{ latest_counts.swing_setups }}</p>
                <p class="mb-1"><strong>Clean trade plan:</strong> {{ latest_counts.trade_plan_clean }}</p>
                <a href="/run/{{ latest_date }}" class="btn btn-primary btn-sm mt-2">View latest results</a>
              {% else %}
                <p class="text-muted mb-0">No historic run data found.</p>
              {% endif %}
            </div>
          </div>
        </div>

        <div class="col-md-4">
          <div class="card summary-card">
            <div class="card-body">
              <h2 class="h6">Historic Runs</h2>
              <p class="mb-1"><strong>Total dates:</strong> {{ summary|length }}</p>
              <p class="mb-0"><em>Select a date below to preview that run.</em></p>
            </div>
          </div>
        </div>

        <div class="col-md-4">
          <div class="card summary-card">
            <div class="card-body">
              <h2 class="h6">Output Files</h2>
              <ul class="list-unstyled mb-0 small">
                <li><code>swing_setups_YYYY-MM-DD.csv</code></li>
                <li><code>trade_plan_YYYY-MM-DD.csv</code></li>
                <li><code>trade_plan_clean_YYYY-MM-DD.csv</code></li>
                <li><code>vibe_report_YYYY-MM-DD.csv</code></li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div class="row gy-4">
        <div class="col-lg-5">
          <div class="card">
            <div class="card-body">
              <h2 class="h5">Historic Run Dates</h2>
              <div class="table-wrap table-responsive">
                {{ summary_table|safe }}
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-7">
          <div class="card">
            <div class="card-body">
              <h2 class="h5">Run Preview</h2>
              {% if details %}
                <p class="text-muted">Previewing <strong>{{ details.date }}</strong>. Click a file name to open the full CSV.</p>
                {% for section in details.tables %}
                  <div class="mb-4">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                      <h3 class="h6 mb-0">{{ section.title }}</h3>
                      <a href="/download/{{ section.filename }}" class="btn btn-sm btn-outline-secondary">Download CSV</a>
                    </div>
                    <div class="table-wrap table-responsive">{{ section.html|safe }}</div>
                  </div>
                {% endfor %}
              {% else %}
                <p class="text-muted mb-0">Select a date from the table to load preview data.</p>
              {% endif %}
            </div>
          </div>
        </div>
      </div>

      <footer class="text-muted small mt-4">
        Data folder: <span class="mono">{{ logs_dir }}</span>
      </footer>
    </div>
  </body>
</html>"""


def parse_date(path: Path):
    match = DATE_PATTERN.match(path.name)
    return match.group(1) if match else None


def available_dates():
    dates = set()
    for pattern in ["swing_setups_*.csv", "trade_plan_clean_*.csv", "vibe_report_*.csv"]:
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
            return "<p class=\"text-muted\">Empty file</p>"
        preview = df.head(max_rows)
        return preview.to_html(classes="table table-sm table-striped", index=False, border=0, justify="left")
    except Exception as exc:
        return f"<div class=\"alert alert-danger\">Failed to load {path.name}: {exc}</div>"


def get_run_details(date):
    files = []
    for name, label in [
        (f"swing_setups_{date}.csv", "Swing Setups"),
        (f"trade_plan_{date}.csv", "Trade Plan"),
        (f"trade_plan_clean_{date}.csv", "Clean Trade Plan"),
        (f"vibe_report_{date}.csv", "Vibe Report"),
        (f"vibe_report_local_{date}.csv", "Local Vibe Report")
    ]:
        path = LOGS_DIR / name
        if path.exists():
            html = load_csv_preview(path)
            files.append({"title": label, "html": html, "filename": path.name})
    return {"date": date, "tables": files}


def build_summary_table(dates):
    rows = []
    for date in dates:
        swing_path = LOGS_DIR / f"swing_setups_{date}.csv"
        clean_path = LOGS_DIR / f"trade_plan_clean_{date}.csv"
        rows.append({
            "Date": date,
            "Swing Setups": len(pd.read_csv(swing_path)) if swing_path.exists() else "-",
            "Clean Trade Plan": len(pd.read_csv(clean_path)) if clean_path.exists() else "-",
            "Details": f"<a href='/run/{date}' class='btn btn-sm btn-outline-secondary'>View</a>"
        })
    summary_df = pd.DataFrame(rows)
    return summary_df.to_html(classes="table table-sm table-hover", index=False, escape=False, border=0, justify="left")


@APP.route("/download/<path:filename>")
def download_file(filename):
    safe_path = LOGS_DIR / filename
    if not safe_path.exists() or safe_path.resolve().parent != LOGS_DIR.resolve():
        abort(404, description="File not found")
    return send_from_directory(LOGS_DIR, filename, as_attachment=True)


@APP.route("/")
def home():
    dates = available_dates()
    latest_date = get_latest_date()
    latest_counts = {
        "swing_setups": len(pd.read_csv(LOGS_DIR / f"swing_setups_{latest_date}.csv")) if latest_date and (LOGS_DIR / f"swing_setups_{latest_date}.csv").exists() else 0,
        "trade_plan_clean": len(pd.read_csv(LOGS_DIR / f"trade_plan_clean_{latest_date}.csv")) if latest_date and (LOGS_DIR / f"trade_plan_clean_{latest_date}.csv").exists() else 0,
    }
    summary_table = build_summary_table(dates)
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
    if date not in available_dates():
        abort(404, description=f"No run found for date: {date}")
    details = get_run_details(date)
    summary_table = build_summary_table(available_dates())
    latest_date = get_latest_date()
    latest_counts = {
        "swing_setups": len(pd.read_csv(LOGS_DIR / f"swing_setups_{latest_date}.csv")) if latest_date and (LOGS_DIR / f"swing_setups_{latest_date}.csv").exists() else 0,
        "trade_plan_clean": len(pd.read_csv(LOGS_DIR / f"trade_plan_clean_{latest_date}.csv")) if latest_date and (LOGS_DIR / f"trade_plan_clean_{latest_date}.csv").exists() else 0,
    }
    return render_template_string(
        TEMPLATE,
        latest_date=latest_date,
        latest_counts=latest_counts,
        summary=available_dates(),
        summary_table=summary_table,
        details=details,
        logs_dir=str(LOGS_DIR)
    )


if __name__ == "__main__":
    APP.run(debug=True, host="0.0.0.0", port=5000)
