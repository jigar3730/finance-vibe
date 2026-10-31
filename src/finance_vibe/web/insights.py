"""Read-only insight over the generated data: ETF freshness and bar quality.

Pure pandas and file reads, no Flask, like :mod:`finance_vibe.web.playbook`.
Nothing here feeds back into a scanner or a score.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

import pandas as pd

from finance_vibe import config
from finance_vibe.data import daily_ingest, ingest, quality, raw

logger = logging.getLogger(__name__)

MODES = ("daily", "weekly")


def expected_bars(now: datetime | None = None) -> dict[str, date]:
    """Newest bar each mode should have at ``now``: last closed session / week (Monday)."""
    return {
        "daily": daily_ingest.last_complete_session(now),
        "weekly": ingest.last_complete_week(now),
    }


def _row_date(row: dict[str, Any] | None) -> date | None:
    value = (row or {}).get("AsOf Date")
    stamp = pd.to_datetime(value, errors="coerce") if value is not None else pd.NaT
    return None if pd.isna(stamp) else stamp.date()


def raw_bar_issues(symbol: str, mode: str, expected_last: date) -> list[dict[str, str]]:
    """``quality.bar_issues`` for one ticker's raw file; [] when there is no file."""
    path = config.get_raw_path(symbol, config.TIMEFRAME_PROFILES[mode])
    if not os.path.isfile(path):
        return []
    try:
        bars = raw.load_raw(path)
    except (OSError, ValueError) as exc:
        logger.warning("Raw %s %s unreadable for quality check: %s", mode, symbol, exc)
        return [
            {"Symbol": symbol, "Check": "unreadable", "Severity": quality.ERROR, "Detail": str(exc)}
        ]
    return quality.bar_issues(bars, symbol, expected_last=expected_last)


def etf_freshness(
    symbol: str,
    rows: dict[str, dict[str, Any] | None],
    expected: dict[str, date],
) -> dict[str, dict[str, Any]]:
    """Per mode: the scan row's as-of date, whether it is current, and raw-bar issues.

    ``ok`` is False when the row is missing or older than the expected bar.
    ``flags`` are short labels for the card ("missing", "stale", issue checks).
    """
    out = {}
    for mode in MODES:
        as_of = _row_date(rows.get(mode))
        flags = []
        if rows.get(mode) is None:
            flags.append("missing")
        elif as_of is None or as_of < expected[mode]:
            flags.append("stale")
        issues = raw_bar_issues(symbol, mode, expected[mode])
        # A stale raw file repeats the row's own stale flag.
        flags += [i["Check"] for i in issues if not (i["Check"] == "stale" and "stale" in flags)]
        out[mode] = {
            "as_of": as_of.isoformat() if as_of else None,
            "expected": expected[mode].isoformat(),
            "ok": not flags,
            "flags": flags,
            "issues": issues,
        }
    return out


def quality_errors(logs_dir: str) -> dict[str, list[str]]:
    """Symbol → error checks from the newest ``data_quality_<date>.csv`` in ``logs_dir``."""
    report = quality.latest_report(logs_dir)
    if report.empty:
        return {}
    errors = report[report["Severity"] == quality.ERROR]
    return {str(sym): sorted(set(grp["Check"])) for sym, grp in errors.groupby("Symbol")}
