"""Cron runners: the database is rebuilt after a good scan and can never fail one."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
DOCKER_STUB = """#!/usr/bin/env bash
echo "$*" >> "$CALLS"
case "$*" in
    *finance_vibe.data.database*) exit "$DATABASE_EXIT" ;;
    *finance_vibe.pipeline*) exit "$PIPELINE_EXIT" ;;
esac
exit 0
"""


def _run(tmp_path: Path, mode: str, *, pipeline: int = 0, database: int = 0):
    stub = tmp_path / "docker_stub.sh"
    stub.write_text(DOCKER_STUB)
    calls = tmp_path / "calls.txt"
    calls.touch()
    logs = tmp_path / "logs"
    env = {
        **os.environ,
        "HOME": str(tmp_path),  # no alert-mail settings here
        "FINANCE_VIBE_CRON_LOG_DIR": str(logs),
        "CALLS": str(calls),
        "PIPELINE_EXIT": str(pipeline),
        "DATABASE_EXIT": str(database),
        # An exported function wins over any real docker on PATH.
        "BASH_FUNC_docker%%": f'() {{ bash "{stub}" "$@"; }}',
    }
    done = subprocess.run(
        ["bash", str(SCRIPTS / f"run_{mode}_pipeline.sh"), "--now"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    text = "".join(p.read_text() for p in logs.glob(f"{mode}_2*.log"))
    return done.returncode, calls.read_text().splitlines(), text, logs


@pytest.mark.parametrize("mode", ["daily", "weekly"])
def test_rebuilds_after_the_scan_and_before_the_health_check(tmp_path, mode):
    code, calls, text, _ = _run(tmp_path, mode)
    assert code == 0
    assert len(calls) == 3
    assert "finance_vibe.pipeline" in calls[0]
    assert "finance_vibe.data.database rebuild" in calls[1]
    assert "exec -i" in calls[2]
    assert "database rebuild FAILED" not in text


@pytest.mark.parametrize("mode", ["daily", "weekly"])
def test_failed_rebuild_is_logged_and_the_run_still_counts(tmp_path, mode):
    code, calls, text, logs = _run(tmp_path, mode, database=1)
    assert code == 0
    assert "exec -i" in calls[-1]  # the health check still ran
    assert "database rebuild FAILED (exit 1)" in text
    if mode == "daily":
        assert list(logs.glob("daily_ok_*"))


@pytest.mark.parametrize("mode", ["daily", "weekly"])
def test_failed_scan_is_not_loaded(tmp_path, mode):
    code, calls, _, _ = _run(tmp_path, mode, pipeline=1)
    assert code == 1
    assert not any("finance_vibe.data.database" in c for c in calls)
