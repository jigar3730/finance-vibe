import argparse
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from dotenv import load_dotenv
from google import genai

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
        with open(file, "r", encoding="utf-8") as f:
            combined_content += f.read() + "\n"

    return combined_content


def analyze_with_gemini(combined_data: str) -> str:
    """Sends raw metrics + trade plans to Gemini API for full HTML execution output."""
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
    4. Senior Market Analyst Validation & Qualitative Insights (Top 5 ML Ranked Tickers ONLY):
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
    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    return response.text


def send_html_email(subject: str, html_content: str):
    """Dispatches HTML email via Gmail SMTP."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("RECIPIENT_EMAIL", smtp_user)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient

    clean_html = html_content.replace("```html", "").replace("```", "").strip()

    msg.set_content("Your email client does not support HTML emails.")
    msg.add_alternative(clean_html, subtype="html")

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def main():
    parser = argparse.ArgumentParser(description="Finance-Vibe Enhanced AI Notifier")
    parser.add_argument("mode", help="Execution profile (weekly, daily, high_beta)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent

    print(
        f"🤖 [AI Notifier] Gathering ALL raw signals & trade plans for mode: {args.mode}..."
    )
    combined_data = get_pipeline_outputs(args.mode, root_dir)

    print("🧠 Generating complete ticker report with Gemini...")
    html_insights = analyze_with_gemini(combined_data)

    print("📧 Dispatching full master HTML report...")
    subject = f"📊 Finance-Vibe [{args.mode.upper()}] Master Pipeline Briefing"
    send_html_email(subject, html_insights)
    print("✅ Complete master email dispatched successfully!")


if __name__ == "__main__":
    main()
