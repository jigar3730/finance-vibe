"""scripts/notify_email.py: the alert body is a short summary of the run log."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "notify_email.py"
spec = importlib.util.spec_from_file_location("notify_email", SCRIPT)
assert spec and spec.loader
notify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notify)

TABLE = [f"| T{i:03d} | WATCH | 2026-11-13 | 10.0 | BULLISH |" for i in range(300)]


def _log(*parts: list[str]) -> list[str]:
    return [line for part in parts for line in part]


def test_summary_keeps_what_happened_and_drops_tables():
    lines = _log(
        ["=== daily pipeline --mode daily start (first attempt) 2026-11-13T18:30:01-05:00 ==="],
        ["2026-11-13 18:30:05,000 | INFO | 🔹 Running: universe..."],
        ["2026-11-13 18:30:06,000 | INFO | ✅ Finished: universe"],
        TABLE,
        ["2026-11-13 18:33:09,101 | ERROR | Error in market_ingest. Pipeline halted."],
        ["=== daily pipeline --mode daily end 2026-11-13T18:33:09-05:00 exit=1 ==="],
        ["2026-11-13T18:33:10-05:00 health check failed; will retry at 22:30 ET"],
    )
    out = notify.summarize_log(lines)
    assert out == [
        lines[0],
        lines[2],
        lines[-3],
        lines[-2],
        lines[-1],
    ]


def test_health_check_block_is_kept_and_capped():
    problems = [f"  - SYM{i} data quality stale" for i in range(25)]
    out = notify.summarize_log(["HEALTH CHECK FAILED:", *problems, "after"])
    assert out[0] == "HEALTH CHECK FAILED:"
    assert out[1 : notify.MAX_PROBLEMS + 1] == problems[: notify.MAX_PROBLEMS]
    assert out[-1] == "  ... and 5 more problems"
    assert "after" not in out


def test_last_traceback_is_kept_with_its_exception():
    tb = [
        "Traceback (most recent call last):",
        '  File "a.py", line 1, in main',
        "    boom()",
        "OSError: Yahoo returned HTTP 429",
    ]
    old = ["Traceback (most recent call last):", '  File "old.py"', "ValueError: old"]
    out = notify.summarize_log([*old, *TABLE[:5], *tb, *TABLE[:5]])
    assert out == tb


def test_long_traceback_keeps_header_and_tail():
    frames = [f'  File "f{i}.py", line {i}' for i in range(50)]
    out = notify.summarize_log(["Traceback (most recent call last):", *frames, "KeyError: x"])
    assert out[:2] == ["Traceback (most recent call last):", "  ..."]
    assert out[-1] == "KeyError: x"
    assert len(out) == 2 + notify.MAX_TRACEBACK


def test_warnings_go_last_capped_with_a_count():
    warns = [f"2026-11-13 18:31:22,959 | WARNING | S{i}: data quality big_move" for i in range(15)]
    end = "=== daily pipeline --mode daily end exit=0 ==="
    out = notify.summarize_log([*warns, end])
    assert out[0] == end
    assert out[1:3] == ["", "Warnings (10 of 15):"]
    assert out[3:] == warns[:10]


def test_summary_is_capped():
    lines = [f"2026-11-13 18:00:00,000 | ERROR | e{i}" for i in range(100)]
    out = notify.summarize_log(lines)
    assert len(out) == notify.MAX_LINES
    assert out[0].startswith("... 41 earlier lines left out")
    assert out[-1] == lines[-1]


def test_body_has_link_and_summary(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join([*TABLE, "HEALTH CHECK FAILED:", "  - SPY stale"]) + "\n")
    body = notify.build_body("first line", str(log), None, "http://box:5000/")
    assert body.startswith(
        "first line\n\nWhat to do: http://box:5000/docs/runbook#when-something-fails"
    )
    assert "Summary of the log (302 lines" in body
    assert "  - SPY stale" in body
    assert "T001" not in body


def test_tail_sends_raw_lines(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join(TABLE) + "\n")
    body = notify.build_body("x", str(log), 3, "http://box:5000")
    assert body.rstrip().endswith("\n".join(TABLE[-3:]))
    assert "Last 3 lines of the log" in body


def test_missing_log_is_said(tmp_path):
    body = notify.build_body("x", str(tmp_path / "nope.log"), None, "http://box:5000")
    assert "(log file missing:" in body


def test_dry_run_prints_without_config(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text("HEALTH CHECK FAILED:\n  - SPY stale\n")
    rc = notify.main(
        ["--subject", "S", "--body", "B", "--body-file", str(log), "--dry-run",
         "--env-file", str(tmp_path / "none.env")]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0
    assert "Subject: S" in out and "  - SPY stale" in out


def test_unconfigured_send_returns_2(tmp_path):
    rc = notify.main(["--subject", "S", "--env-file", str(tmp_path / "none.env")])
    assert rc == 2


def test_dashboard_url_from_env(tmp_path, capsys):
    env = tmp_path / "n.env"
    env.write_text("DASHBOARD_URL=http://fv.local:5000\n")
    notify.main(["--subject", "S", "--dry-run", "--env-file", str(env)])
    assert "http://fv.local:5000/docs/runbook#when-something-fails" in capsys.readouterr().out


def test_message_json_sends_the_finished_message(tmp_path, capsys):
    msg = tmp_path / "mail.json"
    msg.write_text(
        json.dumps(
            {"subject": "Top 3", "text": "A\nSetups: {DASHBOARD}/setups\n", "html": "<p>A</p>"}
        )
    )
    env = tmp_path / "notify.env"
    env.write_text("DASHBOARD_URL=http://box:5000/\n")
    assert notify.main(["--message-json", str(msg), "--env-file", str(env), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Subject: Top 3" in out and "Setups: http://box:5000/setups" in out
    assert "[html part: 8 characters]" in out
    assert "What to do" not in out  # not an alert: no runbook line


def test_message_json_builds_a_text_and_html_email(tmp_path):
    m = notify.build_message({"EMAIL_TO": "a@b.c"}, "s", "plain", "<p>x</p>")
    assert m.is_multipart()
    assert [p.get_content_type() for p in m.iter_parts()] == ["text/plain", "text/html"]


def test_unusable_message_json_is_refused(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("")
    assert notify.main(["--message-json", str(bad), "--dry-run"]) == 1
    bad.write_text('{"subject": "x"}')
    assert notify.main(["--message-json", str(bad), "--dry-run"]) == 1
    assert "needs a subject and a text" in capsys.readouterr().err
