#!/usr/bin/env python3
"""Send a plain-text email alert for scheduled Finance Vibe runs (stdlib only).

Runs on the host, so it still works when the container is down. SMTP settings
come from a KEY=VALUE env file (default ``~/.config/finance-vibe/notify.env``,
override with ``--env-file`` or ``FINANCE_VIBE_NOTIFY_ENV``). It uses the same
variable names as quant-hub: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
SMTP_USE_TLS, EMAIL_FROM, EMAIL_TO (comma-separated). Optional DASHBOARD_URL
(default ``http://<hostname>:5000``) sets the runbook link.

    scripts/notify_email.py --subject "..." [--body "..."] [--body-file run.log]
    scripts/notify_email.py --subject "..." --body-file run.log --dry-run   # print, send nothing

With ``--body-file`` the email carries a summary of the run log, not the whole log:
stage start / end lines, the health check result, errors, the last traceback and the
first warnings. Scanner console tables are left out. ``--tail N`` sends the last N
raw lines instead.

Exit codes: 0 sent (or printed), 2 not configured, 1 send failed.
"""

from __future__ import annotations

import argparse
import os
import re
import smtplib
import socket
import sys
from email.message import EmailMessage
from pathlib import Path

DEFAULT_ENV = Path.home() / ".config" / "finance-vibe" / "notify.env"
RUNBOOK_PATH = "/docs/runbook#when-something-fails"

MAX_LINES = 60
MAX_WARNINGS = 10
MAX_PROBLEMS = 20
MAX_TRACEBACK = 30

# Runner echoes start with an ISO timestamp (`date -Is`): "2026-11-12T22:31:39-05:00 ...".
_RUNNER = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")
_LEVEL = re.compile(r"\| (ERROR|CRITICAL|WARNING) \|")
_STAGE = ("✅ Finished:", "🏁 Workflow Complete", "Pipeline halted")


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


def summarize_log(lines: list[str]) -> list[str]:
    """The lines of a run log that say what happened, in log order.

    Kept: ``===`` start / end markers, finished stages, runner messages, health check
    results (up to ``MAX_PROBLEMS`` problems), ERROR / CRITICAL lines and the last
    traceback (its last ``MAX_TRACEBACK`` lines), at most ``MAX_LINES``; then, in their own
    section, the first ``MAX_WARNINGS`` WARNING lines and their total. Everything else
    (console tables, info lines) is dropped.
    """
    # The last traceback: from its header to the first unindented line after it (the
    # exception), keeping the header and its last MAX_TRACEBACK lines.
    tb: set[int] = set()
    tb_cut: int | None = None  # header index when lines were left out after it
    starts = [i for i, line in enumerate(lines) if line.startswith("Traceback (most recent")]
    if starts:
        head = end = starts[-1]
        end += 1
        while end < len(lines) and (lines[end].startswith((" ", "\t")) or not lines[end]):
            end += 1
        body = range(head + 1, min(end + 1, len(lines)))
        tb = {head, *body[-MAX_TRACEBACK:]}
        tb_cut = head if len(body) > MAX_TRACEBACK else None

    out: list[str] = []
    warnings: list[str] = []
    problems = 0
    in_check = False
    for i, line in enumerate(lines):
        if in_check and line.startswith("  - "):
            problems += 1
            if problems <= MAX_PROBLEMS:
                out.append(line)
            continue
        if in_check and problems > MAX_PROBLEMS:
            out.append(f"  ... and {problems - MAX_PROBLEMS} more problems")
        in_check = False
        level = _LEVEL.search(line)
        if line.startswith("HEALTH CHECK FAILED"):
            in_check, problems = True, 0
            out.append(line)
        elif i in tb:
            out.append(line)
            if i == tb_cut:
                out.append("  ...")
        elif level and level.group(1) == "WARNING":
            warnings.append(line)
        elif (
            level
            or line.startswith(("===", "health check ok"))
            or _RUNNER.match(line)
            or any(s in line for s in _STAGE)
        ):
            out.append(line)
    if in_check and problems > MAX_PROBLEMS:
        out.append(f"  ... and {problems - MAX_PROBLEMS} more problems")
    if len(out) > MAX_LINES:
        out = [f"... {len(out) - MAX_LINES + 1} earlier lines left out (see the log)"] + out[
            -(MAX_LINES - 1) :
        ]
    if warnings:
        shown = warnings[:MAX_WARNINGS]
        out += ["", f"Warnings ({len(shown)} of {len(warnings)}):", *shown]
    return out


def build_body(body: str, body_file: str | None, tail: int | None, dashboard: str) -> str:
    """The alert text: the caller's lines, the runbook link, then the log summary."""
    parts = [body.rstrip(), f"What to do: {dashboard.rstrip('/')}{RUNBOOK_PATH}"]
    if body_file:
        path = Path(body_file)
        if not path.is_file():
            parts.append(f"(log file missing: {path})")
        else:
            lines = path.read_text(errors="replace").splitlines()
            if tail:
                excerpt = lines[-tail:]
                parts.append(f"Last {len(excerpt)} lines of the log:\n\n" + "\n".join(excerpt))
            else:
                parts.append(
                    f"Summary of the log ({len(lines)} lines; tables and info lines left out):\n\n"
                    + "\n".join(summarize_log(lines))
                )
    return "\n\n".join(p for p in parts if p) + "\n"


def build_message(cfg: dict[str, str], subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.get("EMAIL_FROM") or cfg.get("SMTP_USER", "")
    msg["To"] = ", ".join(a.strip() for a in cfg.get("EMAIL_TO", "").split(",") if a.strip())
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
    ap.add_argument("--body", default="", help="Message text placed before the log summary")
    ap.add_argument("--body-file", help="Run log to summarize into the email")
    ap.add_argument("--tail", type=int, help="Send the last N raw log lines instead of a summary")
    ap.add_argument("--dry-run", action="store_true", help="Print the email, send nothing")
    ap.add_argument(
        "--env-file", default=os.environ.get("FINANCE_VIBE_NOTIFY_ENV", str(DEFAULT_ENV))
    )
    args = ap.parse_args(argv)

    cfg = load_env(Path(args.env_file))
    dashboard = cfg.get("DASHBOARD_URL") or f"http://{socket.gethostname()}:5000"
    body = build_body(args.body, args.body_file, args.tail, dashboard)
    msg = build_message(cfg, args.subject, body)
    if args.dry_run:
        print(f"To: {msg['To'] or '(EMAIL_TO not set)'}\nSubject: {args.subject}\n\n{body}", end="")
        return 0
    if not cfg.get("SMTP_HOST") or not cfg.get("EMAIL_TO"):
        print(f"notify_email: not configured ({args.env_file}); alert not sent", file=sys.stderr)
        return 2
    try:
        send(cfg, msg)
    except Exception as exc:  # report, never raise into the caller's error path
        print(f"notify_email: send failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
