"""Bar-level quality flags for the raw OHLCV silo.

``config.validate_and_clean_ohlcv`` rejects files that break the column
contract; this module looks at the bars that pass it and flags ones that are
suspect: a stale newest bar, High/Low that don't contain Open/Close, zero
volume, a run of identical closes, or a move large enough to be a split or a
bad print. It flags for review and never changes what is scanned or scored.

After ingest, :func:`write_report` writes one row per issue to
``data/logs/{mode}/data_quality_<date>.csv``. The callers (``ingest.main`` for
weekly, ``daily_ingest.ingest_daily`` for daily) work out the expected newest
bar, so this module doesn't import them.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime

import pandas as pd

from finance_vibe import config
from finance_vibe.data import raw

logger = logging.getLogger(__name__)

REPORT_PREFIX = "data_quality_"
REPORT_COLUMNS = ["Symbol", "Check", "Severity", "Date", "Detail"]
ERROR, WARN = "error", "warn"


def _issue(symbol: str, check: str, severity: str, day: object, detail: str) -> dict[str, str]:
    return {
        "Symbol": symbol,
        "Check": check,
        "Severity": severity,
        "Date": pd.Timestamp(day).strftime("%Y-%m-%d"),  # type: ignore[arg-type]
        "Detail": detail,
    }


def bar_issues(df: pd.DataFrame, symbol: str, *, expected_last: date) -> list[dict[str, str]]:
    """Quality issues in one ticker's bars (a :func:`raw.load_raw` frame).

    ``stale`` compares the newest bar with ``expected_last``. The other checks
    look at the newest ``config.DQ_RECENT_BARS`` bars and report the latest bar
    that fails each one.
    """
    if df.empty:
        return [_issue(symbol, "stale", ERROR, expected_last, "no bars")]
    issues = []
    newest = pd.Timestamp(df["Date"].iloc[-1]).normalize()
    if newest < pd.Timestamp(expected_last):
        issues.append(
            _issue(
                symbol,
                "stale",
                ERROR,
                newest,
                f"newest bar {newest.date()}, expected {expected_last}",
            )
        )

    # One extra bar so the first close-to-close change in the window is real.
    tail = df.tail(config.DQ_RECENT_BARS + 1).reset_index(drop=True)
    recent = tail.iloc[1:] if len(tail) > config.DQ_RECENT_BARS else tail
    o, h, lo, c = (recent[col] for col in ("Open", "High", "Low", "Close"))
    tol = config.DQ_PRICE_TOL * c.abs()
    body_hi, body_lo = pd.concat([o, c], axis=1).max(axis=1), pd.concat([o, c], axis=1).min(axis=1)
    incoherent = (
        (h < body_hi - tol)
        | (lo > body_lo + tol)
        | (h < lo)
        | (recent[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    )
    if incoherent.any():
        i = incoherent[incoherent].index[-1]
        detail = f"O {o[i]:.4g} H {h[i]:.4g} L {lo[i]:.4g} C {c[i]:.4g}"
        if incoherent.sum() > 1:
            detail += f" ({int(incoherent.sum())} bars)"
        issues.append(_issue(symbol, "ohlc_incoherent", ERROR, recent.at[i, "Date"], detail))

    zero = recent["Volume"].fillna(0) <= 0
    if zero.any():
        i = zero[zero].index[-1]
        issues.append(
            _issue(
                symbol,
                "zero_volume",
                WARN,
                recent.at[i, "Date"],
                f"{int(zero.sum())} bars with no volume",
            )
        )

    # Length of the current run of unchanged closes at every bar.
    same = tail["Close"].diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum() + 1
    flat = run.iloc[-len(recent) :] >= config.DQ_FLAT_RUN
    if flat.any():
        i = flat[flat].index[-1]
        issues.append(
            _issue(
                symbol,
                "flat_closes",
                WARN,
                tail.at[i, "Date"],
                f"{int(run[i])} identical closes at {tail.at[i, 'Close']:.4g}",
            )
        )

    move = tail["Close"].pct_change().iloc[-len(recent) :]
    big = move.abs() >= config.DQ_JUMP_PCT
    if big.any():
        i = big[big].index[-1]
        issues.append(
            _issue(symbol, "big_move", WARN, tail.at[i, "Date"], f"close {move[i]:+.0%} in one bar")
        )
    return issues


def scan_raw_dir(mode: str, *, expected_last: date) -> pd.DataFrame:
    """Issues for every raw CSV of ``mode``, one row each (:data:`REPORT_COLUMNS`)."""
    raw_dir = config.get_mode_config(mode)["raw_dir"]
    rows: list[dict[str, str]] = []
    for f in raw.raw_files(raw_dir):
        try:
            df = raw.load_raw(f.path)
        except (OSError, ValueError) as exc:
            rows.append(_issue(f.ticker, "unreadable", ERROR, expected_last, str(exc)))
            continue
        rows.extend(bar_issues(df, f.ticker, expected_last=expected_last))
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def report_path(logs_dir: str, stamp: str) -> str:
    return os.path.join(logs_dir, f"{REPORT_PREFIX}{stamp}.csv")


def write_report(mode: str, *, expected_last: date, now: datetime | None = None) -> str | None:
    """Scan ``mode``'s raw silo and write ``data_quality_<date>.csv`` (header only when clean).

    Returns the path, or None if the scan or the write failed; a report
    failure is logged and never fails ingest.
    """
    cfg = config.get_mode_config(mode)
    stamp = (now or datetime.now()).strftime("%Y-%m-%d")
    path = report_path(cfg["logs_dir"], stamp)
    try:
        report = scan_raw_dir(mode, expected_last=expected_last)
        os.makedirs(cfg["logs_dir"], exist_ok=True)
        report.to_csv(path, index=False)
    except (OSError, ValueError) as exc:
        logger.warning(f"Data-quality report for {mode} not written: {exc}")
        return None
    errors = report[report["Severity"] == ERROR]
    for r in errors.itertuples(index=False):
        logger.warning(f"{r.Symbol}: data quality {r.Check} ({r.Date}): {r.Detail}")
    logger.info(
        f"🩺 Data quality [{mode}]: {len(errors)} errors, {len(report) - len(errors)} warnings "
        f"-> {os.path.basename(path)}"
    )
    return path


def latest_report(logs_dir: str) -> pd.DataFrame:
    """The newest ``data_quality_<date>.csv`` in ``logs_dir``, or an empty frame."""
    if not os.path.isdir(logs_dir):
        return pd.DataFrame(columns=REPORT_COLUMNS)
    names = sorted(
        n for n in os.listdir(logs_dir) if n.startswith(REPORT_PREFIX) and n.endswith(".csv")
    )
    if not names:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    try:
        return pd.read_csv(os.path.join(logs_dir, names[-1]), dtype=str).fillna("")
    except (OSError, ValueError) as exc:
        logger.warning(f"Unreadable data-quality report {names[-1]}: {exc}")
        return pd.DataFrame(columns=REPORT_COLUMNS)
