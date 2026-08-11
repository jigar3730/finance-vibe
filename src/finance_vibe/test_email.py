import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()


def run_test():
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    raw_recipients = os.getenv("RECIPIENT_EMAIL", smtp_user)

    # 1. Parse comma-separated recipients into a Python list
    recipient_list = [
        email.strip() for email in raw_recipients.split(",") if email.strip()
    ]

    print(f"📧 SMTP User: {smtp_user}")
    print(f"🎯 Target Recipients ({len(recipient_list)}): {recipient_list}")

    # 2. Build test email
    msg = EmailMessage()
    msg["Subject"] = "🧪 Test Email - Finance Vibe Container"
    msg["From"] = smtp_user
    msg["To"] = smtp_user  # Shows sender in To: line for clean BCC dispatch

    html_content = """
    <div style="font-family: sans-serif; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px;">
        <h2 style="color: #2563eb;">Finance Vibe Email Test</h2>
        <p>If you are receiving this message, your container SMTP setup and multi-recipient parsing are working correctly!</p>
    </div>
    """

    msg.set_content(
        "If you are receiving this message, your container SMTP setup is working."
    )
    msg.add_alternative(html_content, subtype="html")

    # 3. Dispatch email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg, to_addrs=recipient_list)
        print("✅ Email sent successfully to all recipients!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


if __name__ == "__main__":
    run_test()
