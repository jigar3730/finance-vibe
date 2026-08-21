import argparse
import csv
import io
import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()


def get_pipeline_outputs(mode: str, root_dir: Path) -> str:
    """Combines all recent log/report files in data/logs/{mode}/ into a single string."""
    log_dir = root_dir / "data" / "logs" / mode
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    files = sorted(
        list(log_dir.glob("*.txt"))
        + list(log_dir.glob("*.log"))
        + list(log_dir.glob("*.csv")),
        key=os.path.getmtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(f"No log files found in {log_dir}")

    combined_content = ""
    # Process up to top 5 recent output files
    for file in files[:5]:
        combined_content += f"\n--- FILE: {file.name} ---\n"
        try:
            with open(file, "r", encoding="utf-8") as f:
                combined_content += f.read() + "\n"
        except Exception as e:
            print(f"⚠️ Failed to read {file.name}: {e}")

    return combined_content


def generate_fallback_html(combined_data: str, error_reason: str) -> str:
    """Constructs a basic HTML email directly from raw CSV data when Gemini is unavailable."""
    print("⚠️ Generating fallback HTML email directly from raw pipeline data...")

    html_tables = ""
    sections = combined_data.split("--- FILE: ")

    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split("\n")
        filename = lines[0].replace(" ---", "").strip()
        content = "\n".join(lines[1:])

        if filename.endswith(".csv") and content.strip():
            try:
                csv_reader = csv.reader(io.StringIO(content.strip()))
                rows = list(csv_reader)
                if rows:
                    table_html = "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%; margin-bottom: 20px; font-family: Arial, sans-serif; font-size: 13px;'>"
                    # Header
                    table_html += "<tr style='background-color: #003366; color: white;'>"
                    for col in rows[0]:
                        table_html += f"<th>{col}</th>"
                    table_html += "</tr>"
                    # Rows
                    for i, row in enumerate(rows[1:]):
                        bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
                        table_html += f"<tr style='background-color: {bg};'>"
                        for col in row:
                            table_html += f"<td>{col}</td>"
                        table_html += "</tr>"
                    table_html += "</table>"
                    html_tables += f"<h3>📄 Data Source: {filename}</h3>{table_html}"
            except Exception as e:
                print(f"⚠️ Could not parse CSV {filename} for fallback HTML: {e}")

    if not html_tables:
        html_tables = f"<pre style='background: #f4f4f4; padding: 10px;'>{combined_data[:3000]}</pre>"

    return f"""
    <div style="font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; color: #333;">
        <div style="background-color: #fff3cd; color: #856404; padding: 15px; border: 1px solid #ffeeba; border-radius: 5px; margin-bottom: 20px;">
            <strong>⚠️ AI Summary Service Warning:</strong> Gemini models were unavailable during this run. 
            Below is the direct raw pipeline data output.<br>
            <small><strong>Error Detail:</strong> {error_reason}</small>
        </div>
        <h2>📊 Finance-Vibe Pipeline Data (Fallback View)</h2>
        {html_tables}
    </div>
    """


def analyze_with_gemini(combined_data: str) -> str:
    """Sends raw metrics to Gemini API with backoff retry and multi-model fallback chain."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an elite quantitative portfolio manager and Senior Market Analyst reviewing automated pipeline results.
    Below is raw setup data (Vibe Scores, CCI, RSI, Trend) and Trade Plans.

    YOUR TASK:
    Provide an executive analysis and present a COMPLETE table of ALL tickers.
    Output your response directly as clean, responsive HTML (no markdown code blocks, no ```html wrappers).

    CRITICAL RULE:
    DO NOT drop, truncate, or summarize the ticker list. You MUST include EVERY single ticker/setup present in the input data across both Coiled Cobra and Swing scanners in the master table.

    LINKING REQUIREMENT:
    - Every ticker symbol MUST be formatted as a clickable hyperlink to Finviz opening in a new tab:
      <a href="[https://finviz.com/quote.ashx?t=TICKER](https://finviz.com/quote.ashx?t=TICKER)" target="_blank" style="color: #0066cc; text-decoration: none; font-weight: bold;">TICKER</a>
    - Apply this link formatting everywhere ticker symbols appear (Master Table, Headers, and Detailed Breakdown bullet points).

    HTML STYLING REQUIREMENTS:
    - Main container: max-width 900px, font-family Arial/sans-serif.
    - Tables: Clean border-collapse, light padding (6px-8px), alternating row backgrounds (#f9f9f9).
    - Status Badges: Highlight Longs in light green, Shorts in light red, Risk Warnings in yellow.

    REQUIRED SECTIONS:
    1. Executive Summary & Market Vibe: Concise summary of market breadth and dominant trends.
    2. Master Setup Table (ALL TICKERS): MUST contain ALL rows from the CSV inputs with columns: Ticker (Linked to Finviz), Strategy, Direction, Vibe Score/Grade, RSI, Entry, Stop Loss, Target 1 (R:R), ML Rank, and AI Note.
    3. Quantitative Anomaly & Risk Alerts: Key warnings regarding high ATR, overbought RSI, or ML divergences.
    4. Senior Market Analyst Validation & Qualitative Insights Top 5 Composite Ranked Tickers (Combining ML Rank + Grade A Vibe Scores):
           Act as a Senior Market Analyst providing expert qualitative validation for ONLY the top 5 ML-ranked tickers. 
           DO NOT simply repeat or list the raw CSV metrics (CCI, RSI, Vibe, Entry levels) already presented in the table above. 
           Instead, synthesize market context, sector tailwinds/headwinds, earnings/catalyst risk, volume quality, structural market structure, and overall trade viability.
           Keep each ticker breakdown under 60 words.
           
           Format each of the Top 5 tickers as:
           • <a href="[https://finviz.com/quote.ashx?t=](https://finviz.com/quote.ashx?t=)[TICKER]" target="_blank">[TICKER]</a> | Strategy: [Strategy/Direction] | Senior Analyst Rating: [High Confidence / Moderate / Caution]
             - Thesis & Validation: Senior-level perspective on why this quantitative setup holds valid structural market edge (e.g., sector momentum, volume absorption, relative strength).
             - Risk & Catalyst Check: Fundamental/macro/catalyst risks or invalidation dynamics not captured in raw scanner numbers.
             - Senior Execution Verdict: Strategic instruction on price action confirmation needed before committing capital.

    INPUT DATA:
    {combined_data}
    """

    models_to_try = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
    max_retries = 3

    for model_name in models_to_try:
        print(f"🧠 Attempting analysis with model: {model_name}...")
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt
                )
                print(f"✅ Successfully generated report using {model_name}")
                return response.text
            except (errors.ServerError, errors.APIError) as e:
                wait_time = attempt * 5
                print(
                    f"⚠️ [{model_name}] Attempt {attempt}/{max_retries} failed with API error: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            except Exception as e:
                print(f"❌ Unexpected error calling model {model_name}: {e}")
                break  # Skip to next model on non-transient error

    raise RuntimeError("All configured Gemini models failed or were unavailable.")


def send_html_email(subject: str, html_content: str, root_dir: Path):
    """Dispatches HTML email via Gmail SMTP with local fallback logging on failure."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("RECIPIENT_EMAIL", smtp_user)

    if not smtp_user or not smtp_password:
        print("⚠️ SMTP credentials not fully configured. Check SMTP_USER and SMTP_PASSWORD env vars.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient

    clean_html = html_content.replace("```html", "").replace("```", "").strip()

    msg.set_content("Your email client does not support HTML emails.")
    msg.add_alternative(clean_html, subtype="html")

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        print("✅ Master email dispatched successfully!")
    except Exception as e:
        print(f"❌ Failed to send email via SMTP: {e}")
        # Save HTML locally so report is not lost
        fail_dir = root_dir / "data" / "logs" / "failed_emails"
        fail_dir.mkdir(parents=True, exist_ok=True)
        file_path = fail_dir / f"failed_email_{int(time.time())}.html"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_html)
        print(f"💾 Report saved locally to: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Finance-Vibe Enhanced AI Notifier")
    parser.add_argument("mode", help="Execution profile (weekly, daily, high_beta)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent

    print(
        f"🤖 [AI Notifier] Gathering ALL raw signals & trade plans for mode: {args.mode}..."
    )

    try:
        combined_data = get_pipeline_outputs(args.mode, root_dir)
    except FileNotFoundError as e:
        print(f"❌ Pipeline halted: {e}")
        return

    subject = f"📊 Finance-Vibe [{args.mode.upper()}] Master Pipeline Briefing"

    try:
        print("🧠 Generating complete ticker report with Gemini...")
        html_insights = analyze_with_gemini(combined_data)
    except Exception as e:
        print(f"⚠️ Gemini processing completely failed: {e}")
        subject = f"📊 [LIMITED DATA] Finance-Vibe [{args.mode.upper()}] Pipeline Briefing"
        html_insights = generate_fallback_html(combined_data, str(e))

    print("📧 Dispatching report...")
    send_html_email(subject, html_insights, root_dir)


if __name__ == "__main__":
    main()