#!/usr/bin/env python3
"""Send a plain-text email alert for scheduled Finance Vibe runs (stdlib only).

Runs on the host, so it still works when the container is down. SMTP settings
come from a KEY=VALUE env file (default ``~/.config/finance-vibe/notify.env``,
override with ``--env-file`` or ``FINANCE_VIBE_NOTIFY_ENV``). It uses the same
variable names as quant-hub: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
SMTP_USE_TLS, EMAIL_FROM, EMAIL_TO (comma-separated).

    scripts/notify_email.py --subject "..." [--body-file run.log] [--tail 80]

Exit codes: 0 sent, 2 not configured, 1 send failed.
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

DEFAULT_ENV = Path.home() / ".config" / "finance-vibe" / "notify.env"


def load_env(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines (blank lines and # comments ignored, quotes stripped)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def build_message(cfg: dict[str, str], subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.get("EMAIL_FROM") or cfg.get("SMTP_USER", "")
    msg["To"] = ", ".join(a.strip() for a in cfg["EMAIL_TO"].split(",") if a.strip())
    msg.set_content(body)
    return msg


def send(cfg: dict[str, str], msg: EmailMessage) -> None:
    port = int(cfg.get("SMTP_PORT") or 587)
    with smtplib.SMTP(cfg["SMTP_HOST"], port, timeout=30) as server:
        if cfg.get("SMTP_USE_TLS", "true").lower() != "false":
            server.starttls()
        if cfg.get("SMTP_USER") and cfg.get("SMTP_PASSWORD"):
            server.login(cfg["SMTP_USER"], cfg["SMTP_PASSWORD"])
        server.send_message(msg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--body", default="", help="Message text placed before the file excerpt")
    ap.add_argument("--body-file", help="Append the last --tail lines of this file (e.g. the run log)")
    ap.add_argument("--tail", type=int, default=80)
    ap.add_argument("--env-file", default=os.environ.get("FINANCE_VIBE_NOTIFY_ENV", str(DEFAULT_ENV)))
    args = ap.parse_args(argv)

    cfg = load_env(Path(args.env_file))
    if not cfg.get("SMTP_HOST") or not cfg.get("EMAIL_TO"):
        print(f"notify_email: not configured ({args.env_file}); alert not sent", file=sys.stderr)
        return 2

    body = args.body
    if args.body_file:
        path = Path(args.body_file)
        lines = path.read_text(errors="replace").splitlines() if path.is_file() else ["(log file missing)"]
        body += f"\n\nLast {min(args.tail, len(lines))} lines of {path}:\n\n" + "\n".join(lines[-args.tail:])
    try:
        send(cfg, build_message(cfg, args.subject, body))
    except Exception as exc:  # report, never raise into the caller's error path
        print(f"notify_email: send failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
