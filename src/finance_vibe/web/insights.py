"""Read-only insight over the generated data: ETF freshness, bar quality, run
changes, and the Market tab's ETF relative view.

Pure pandas and file reads, no Flask, like :mod:`finance_vibe.web.playbook`.
Nothing here feeds back into a scanner or a score.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

import pandas as pd

from finance_vibe import config, indicators, market
from finance_vibe.data import daily_ingest, ingest, quality, raw
from finance_vibe.web import playbook

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


# ---------------------------------------------------------------------------
# Day-over-day changes between the two newest runs of a mode
# ---------------------------------------------------------------------------

# Breakout progress: anything else (WATCH, DEVELOPING, FAILED_BREAKOUT, ...) is 0.
STATUS_STAGE = {"PRE_BREAKOUT": 1, "BREAKOUT_TRIGGERED": 2, playbook.CONFIRMED_STATUS: 3}


def _breakout_dates(folder: str) -> list[str]:
    dated = playbook.run_dates(folder)
    return sorted(d for d, prefixes in dated.items() if playbook.BREAKOUT_PREFIX in prefixes)


def _scan(folder: str, prefix: str, day: str) -> pd.DataFrame:
    path = os.path.join(folder, f"{prefix}{day}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = playbook.read_scan(path)
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    return playbook.relabel_triggered_watch(df) if prefix == playbook.BREAKOUT_PREFIX else df


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in r.items()} for r in df.to_dict("records")]


def status_changes(now: pd.DataFrame, prev: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Breakout status transitions for tickers in both runs, bucketed.

    ``upgrades``: moved forward along PRE_BREAKOUT → BREAKOUT_TRIGGERED →
    BREAKOUT_CONFIRMED (from any other status into one counts);
    ``failures``: moved into FAILED_BREAKOUT; ``other``: any other change. Tickers
    that entered or left the universe are not transitions and are left out.
    """
    empty: dict[str, list[dict[str, Any]]] = {"upgrades": [], "failures": [], "other": []}
    cols = ["Symbol", "Status", "Breakout Readiness"]
    if any(c not in df.columns for df in (now, prev) for c in cols[:2]):
        return empty
    pick = [c for c in cols if c in now.columns and c in prev.columns]
    both = now[pick].merge(prev[pick], on="Symbol", suffixes=("", " Prev"))
    both[["Status", "Status Prev"]] = both[["Status", "Status Prev"]].fillna("").astype(str)
    moved = both[both["Status"] != both["Status Prev"]]
    if "Breakout Readiness" in moved.columns:
        moved = moved.assign(
            **{"Readiness Change": moved["Breakout Readiness"] - moved["Breakout Readiness Prev"]}
        ).sort_values("Breakout Readiness", ascending=False, kind="stable")
    stage = moved["Status"].map(STATUS_STAGE).fillna(0)
    upgrade = stage > moved["Status Prev"].map(STATUS_STAGE).fillna(0)
    failed = moved["Status"] == playbook.FAILED_STATUS
    return {
        "upgrades": _records(moved[upgrade]),
        "failures": _records(moved[failed]),
        "other": _records(moved[~upgrade & ~failed]),
    }


def cobra_changes(now: pd.DataFrame, prev: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Coiled Cobra setups new since the previous run, dropped, and kept (with Score change)."""
    out: dict[str, list[dict[str, Any]]] = {"new": [], "dropped": [], "kept": []}
    if "Symbol" not in now.columns and "Symbol" not in prev.columns:
        return out
    cols = ["Symbol", "Score", "Grade"]
    now = now.reindex(columns=cols) if not now.empty else pd.DataFrame(columns=cols)
    prev = prev.reindex(columns=cols) if not prev.empty else pd.DataFrame(columns=cols)
    both = now.merge(prev, on="Symbol", how="outer", suffixes=("", " Prev"), indicator=True)
    both["Score Change"] = both["Score"] - both["Score Prev"]
    by_score = both.sort_values(["Score", "Score Prev"], ascending=False, kind="stable")
    out["new"] = _records(by_score[by_score["_merge"] == "left_only"].drop(columns="_merge"))
    out["dropped"] = _records(by_score[by_score["_merge"] == "right_only"].drop(columns="_merge"))
    out["kept"] = _records(by_score[by_score["_merge"] == "both"].drop(columns="_merge"))
    return out


def run_changes(folder: str) -> dict[str, Any]:
    """What changed between the two newest breakout runs in ``folder``.

    ``prev`` is None (and every list empty) until there are two runs.
    """
    dates = _breakout_dates(folder)
    now_date = dates[-1] if dates else None
    prev_date = dates[-2] if len(dates) > 1 else None
    result: dict[str, Any] = {"now": now_date, "prev": prev_date}
    result["cobra_ran"] = now_date is not None and os.path.exists(
        os.path.join(folder, f"{playbook.COBRA_PREFIX}{now_date}.csv")
    )
    if now_date is None or prev_date is None:
        empty_status = status_changes(pd.DataFrame(), pd.DataFrame())
        return result | {
            "status": empty_status,
            "cobra": cobra_changes(pd.DataFrame(), pd.DataFrame()),
        }
    bo, cc = playbook.BREAKOUT_PREFIX, playbook.COBRA_PREFIX
    result["status"] = status_changes(_scan(folder, bo, now_date), _scan(folder, bo, prev_date))
    result["cobra"] = cobra_changes(_scan(folder, cc, now_date), _scan(folder, cc, prev_date))
    return result


def bias_changes(daily_folder: str, weekly_folder: str) -> list[dict[str, Any]]:
    """Market ETF bias at the newest daily run vs the previous one.

    Each daily run is paired with the newest weekly run on or before its date,
    the same pairing the Market tab sees on that day.
    """
    days, weeks = _breakout_dates(daily_folder), _breakout_dates(weekly_folder)
    if len(days) < 2:
        return []
    bo = playbook.BREAKOUT_PREFIX

    def biases(day: str) -> dict[str, str]:
        week = max((w for w in weeks if w <= day), default=None)
        weekly = _scan(weekly_folder, bo, week) if week else pd.DataFrame()
        rows = playbook.market_rows(_scan(daily_folder, bo, day), weekly)
        return {r["symbol"]: r["bias"] for r in rows}

    now, prev = biases(days[-1]), biases(days[-2])
    return [
        {"symbol": s, "bias": now[s], "prev": prev[s], "changed": now[s] != prev[s]}
        for s in playbook.MARKET_SYMBOLS
    ]


# ---------------------------------------------------------------------------
# ETF relative view (Market tab)
# ---------------------------------------------------------------------------

# Display windows in daily bars.
RETURN_WINDOWS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126}
RS_LOOKBACK = 63
GROWTH_LEADING, DIVIDEND_LEADING, MIXED = "Growth leading", "Dividend/value leading", "Mixed"


def daily_closes(symbol: str) -> pd.DataFrame | None:
    """``Date`` / ``Close`` from a ticker's daily raw file, or None if unavailable.

    Raw prices are Yahoo adjusted closes (``auto_adjust=True``), so returns
    include dividends.
    """
    path = config.get_raw_path(symbol, config.TIMEFRAME_PROFILES["daily"])
    if not os.path.isfile(path):
        return None
    try:
        return raw.load_raw(path)[["Date", "Close"]]
    except (OSError, ValueError) as exc:
        logger.warning("Daily raw for %s unreadable: %s", symbol, exc)
        return None


def _ret(close: pd.Series, bars: int) -> float | None:
    if len(close) <= bars:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - bars] - 1.0)


def rotation(growth: pd.DataFrame | None, dividend: pd.DataFrame | None) -> dict[str, Any]:
    """Growth vs dividend leadership from the growth / dividend close ratio.

    "Growth leading" when the ratio is above its ``ROTATION_MA_BARS`` SMA and
    the ``ROTATION_LOOKBACK`` return spread (growth − dividend) is positive;
    "Dividend/value leading" when it is below the SMA and the spread is
    negative; "Mixed" otherwise or with too little history. Unvalidated context.
    """
    out: dict[str, Any] = {"label": MIXED, "ratio": None, "ma": None, "spread": None, "as_of": None}
    if growth is None or dividend is None:
        return out
    both = growth.merge(dividend, on="Date", suffixes=(" G", " D")).sort_values("Date")
    if len(both) <= max(config.ROTATION_MA_BARS, config.ROTATION_LOOKBACK):
        return out
    ratio = (both["Close G"] / both["Close D"]).reset_index(drop=True)
    ma = indicators.sma(ratio, config.ROTATION_MA_BARS)
    lookback = config.ROTATION_LOOKBACK
    spread = _ret(both["Close G"], lookback) - _ret(both["Close D"], lookback)  # type: ignore[operator]
    out |= {
        "ratio": float(ratio.iloc[-1]),
        "ma": None if ma is None or pd.isna(ma.iloc[-1]) else float(ma.iloc[-1]),
        "spread": spread,
        "as_of": pd.Timestamp(both["Date"].iloc[-1]).date().isoformat(),
    }
    if out["ma"] is None:
        return out
    if out["ratio"] > out["ma"] and spread > 0:
        out["label"] = GROWTH_LEADING
    elif out["ratio"] < out["ma"] and spread < 0:
        out["label"] = DIVIDEND_LEADING
    return out


def etf_relative(
    symbols: tuple[str, ...] = playbook.MARKET_SYMBOLS, benchmark: str = "SPY"
) -> dict[str, Any]:
    """Returns over :data:`RETURN_WINDOWS`, 63-day RS vs ``benchmark``, and the rotation label."""
    frames = {s: daily_closes(s) for s in dict.fromkeys((*symbols, benchmark))}
    bench = frames[benchmark]
    rows = []
    for sym in symbols:
        df = frames[sym]
        row: dict[str, Any] = {
            "symbol": sym,
            "as_of": None,
            "returns": {},
            "rs": None,
            "rs_ok": None,
        }
        if df is not None and not df.empty:
            row["as_of"] = pd.Timestamp(df["Date"].iloc[-1]).date().isoformat()
            row["returns"] = {k: _ret(df["Close"], n) for k, n in RETURN_WINDOWS.items()}
            if sym != benchmark and bench is not None:
                ok, rel = market.relative_strength(df, bench, lookback=RS_LOOKBACK)
                row["rs"], row["rs_ok"] = rel, (ok if rel is not None else None)
        rows.append(row)
    for sym in (config.ROTATION_GROWTH, config.ROTATION_DIVIDEND):
        if sym not in frames:
            frames[sym] = daily_closes(sym)
    rot = rotation(frames[config.ROTATION_GROWTH], frames[config.ROTATION_DIVIDEND])
    return {"rows": rows, "windows": list(RETURN_WINDOWS), "benchmark": benchmark, "rotation": rot}
