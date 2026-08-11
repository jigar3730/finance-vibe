import argparse
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()


def get_latest_report(mode: str, root_dir: Path) -> Path:
    """Finds the most recent report file in data/logs/{mode}/."""
    log_dir = root_dir / "data" / "logs" / mode
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    files = (
        list(log_dir.glob("*.txt"))
        + list(log_dir.glob("*.log"))
        + list(log_dir.glob("*.csv"))
    )
    if not files:
        raise FileNotFoundError(f"No log or report files found in {log_dir}")

    return max(files, key=os.path.getmtime)


def analyze_with_gemini(report_content: str) -> str:
    """Sends the report contents to Gemini API for trading insights."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert quantitative trader and market analyst.
    Below is the latest trading execution/plan data from our Finance-Vibe pipeline.
    
    Please analyze the data and generate a clear executive summary containing:
    1. Overall Market Vibe & Key Insights
    2. Top Actionable Trade Setups (including Tickers, Entry, Stop Loss, Targets if available)
    3. Notable Risk Factors / Warnings
    
    REPORT DATA:
    {report_content}
    """

    response = client.models.generate_content(model="gemini-3.5-flash", contents=prompt)
    return response.text


def send_email(subject: str, body: str):
    """Sends the analysis via Gmail SMTP."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("RECIPIENT_EMAIL", smtp_user)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.set_content(body)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def main():
    parser = argparse.ArgumentParser(description="Finance-Vibe AI Notifier")
    parser.add_argument("mode", help="Execution profile (weekly, daily, high_beta)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent

    print(f"🤖 [AI Notifier] Locating latest log for mode: {args.mode}...")
    report_path = get_latest_report(args.mode, root_dir)
    print(f"📄 Processing report: {report_path.name}")

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("🧠 Requesting insights from Gemini...")
    insights = analyze_with_gemini(content)

    print("📧 Sending email notification...")
    subject = f"Finance-Vibe [{args.mode.upper()}] Executive Report"
    send_email(subject, insights)
    print("✅ Email notification dispatched successfully!")


if __name__ == "__main__":
    main()
