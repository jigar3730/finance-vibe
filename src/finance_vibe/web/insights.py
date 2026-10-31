"""Read-only insight over the generated data: ETF freshness, bar quality, run
changes, the Market tab's ETF relative view and the market strip on every page.

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
        moved = moved.sort_values("Breakout Readiness", ascending=False, kind="stable")
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


def universe_changes(now: pd.DataFrame, prev: pd.DataFrame) -> dict[str, list[str]]:
    """Tickers that entered (in ``now`` only) or left (in ``prev`` only) the scanned list."""
    syms = [set(df["Symbol"]) if "Symbol" in df.columns else set() for df in (now, prev)]
    return {"entered": sorted(syms[0] - syms[1]), "left": sorted(syms[1] - syms[0])}


def new_confirmed(now: pd.DataFrame, prev: pd.DataFrame) -> dict[str, list[str]]:
    """BREAKOUT_CONFIRMED in ``now`` but not in ``prev``, split by whether ``prev`` scanned it.

    ``new``: ``prev`` had the ticker with another status (the first-bar event).
    ``unscanned``: ``prev`` had no row for it, so its first week is unknown.
    Tickers CONFIRMED in both runs (continuations) are in neither list.
    """
    rows = playbook.weekly_confirmed(now, prev)
    if rows.empty:
        return {"new": [], "unscanned": []}
    fresh = rows[~rows["Confirmed Last Run"]]
    return {
        "new": list(fresh.loc[fresh["In Last Run"], "Symbol"]),
        "unscanned": list(fresh.loc[~fresh["In Last Run"], "Symbol"]),
    }


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
    bo, cc = playbook.BREAKOUT_PREFIX, playbook.COBRA_PREFIX
    if now_date is None or prev_date is None:
        now = prev = cobra_now = cobra_prev = pd.DataFrame()
    else:
        now, prev = _scan(folder, bo, now_date), _scan(folder, bo, prev_date)
        cobra_now, cobra_prev = _scan(folder, cc, now_date), _scan(folder, cc, prev_date)
    return result | {
        "status": status_changes(now, prev),
        "cobra": cobra_changes(cobra_now, cobra_prev),
        "universe": universe_changes(now, prev),
        "confirmed": new_confirmed(now, prev),
    }


def weekly_state_changes(daily_folder: str, weekly_folder: str) -> dict[str, list[str]]:
    """Per market ETF: weekly Trend / MTF changes between the weekly runs paired with
    the two newest daily runs (the newest weekly run on or before each day)."""
    days, weeks = _breakout_dates(daily_folder), _breakout_dates(weekly_folder)
    if len(days) < 2:
        return {}
    bo = playbook.BREAKOUT_PREFIX

    def weekly_at(day: str) -> pd.DataFrame:
        week = max((w for w in weeks if w <= day), default=None)
        return _scan(weekly_folder, bo, week) if week else pd.DataFrame()

    now_df, prev_df = weekly_at(days[-1]), weekly_at(days[-2])
    out: dict[str, list[str]] = {}
    for sym in playbook.MARKET_SYMBOLS:
        now, prev = playbook.row_for(now_df, sym), playbook.row_for(prev_df, sym)
        if not now or not prev:
            continue
        out[sym] = [
            f"weekly {label} {prev.get(col)} → {now.get(col)}"
            for col, label in (("Trend", "trend"), ("MTF", "MTF"))
            if prev.get(col) != now.get(col)
        ]
    return out


def _side(close: Any, level: Any) -> str | None:
    c, lv = playbook.num(close), playbook.num(level)
    return None if c is None or lv is None else ("above" if c > lv else "below")


def market_changes(daily_folder: str, weekly_folder: str) -> dict[str, Any] | None:
    """Market ETF facts that changed between the two newest daily runs; None until two runs.

    Per ETF, plain-English facts: the weekly Trend or MTF changed, the daily close crossed its
    SMA50 or SMA200, or it newly closed below support on heavy volume
    (:func:`heavy_break`). ``rows`` lists only ETFs with a fact. ``rotation`` is the
    growth-vs-dividend label from daily closes up to each run's date.
    """
    days = _breakout_dates(daily_folder)
    if len(days) < 2:
        return None
    prev_day, now_day = days[-2], days[-1]
    bo = playbook.BREAKOUT_PREFIX
    now_df, prev_df = _scan(daily_folder, bo, now_day), _scan(daily_folder, bo, prev_day)
    weekly = weekly_state_changes(daily_folder, weekly_folder)
    rows = []
    for sym in playbook.MARKET_SYMBOLS:
        now = playbook.row_for(now_df, sym) or {}
        prev = playbook.row_for(prev_df, sym) or {}
        facts = list(weekly.get(sym, []))
        for col, label in (("SMA50", "50-day"), ("SMA200", "200-day")):
            was, is_ = (
                _side(prev.get("Close"), prev.get(col)),
                _side(now.get("Close"), now.get(col)),
            )
            if was and is_ and was != is_:
                facts.append(f"closed {is_} its {label} average (was {was})")
        if heavy_break(now) and not heavy_break(prev):
            facts.append("newly closed below support on heavy volume")
        if facts:
            rows.append({"symbol": sym, "facts": facts})

    growth, dividend = daily_closes(config.ROTATION_GROWTH), daily_closes(config.ROTATION_DIVIDEND)

    def label_at(day: str) -> str:
        if growth is None or dividend is None:
            return MIXED
        cut = pd.Timestamp(day)
        return str(
            rotation(growth[growth["Date"] <= cut], dividend[dividend["Date"] <= cut])["label"]
        )

    rot_now, rot_prev = label_at(now_day), label_at(prev_day)
    return {
        "now": now_day,
        "prev": prev_day,
        "rows": rows,
        "rotation": {"now": rot_now, "prev": rot_prev, "changed": rot_now != rot_prev},
    }


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


# ---------------------------------------------------------------------------
# Market strip (top of every page but the market home): facts only
# ---------------------------------------------------------------------------


def heavy_break(row: dict[str, Any]) -> bool:
    """Daily close below ``Support`` with RVOL20 at or above ``config.BREAKOUT_RVOL_CONFIRM``."""
    close, support = playbook.num(row.get("Close")), playbook.num(row.get("Support"))
    rvol = playbook.num(row.get("RVOL20"))
    if close is None or support is None or rvol is None:
        return False
    return close < support and rvol >= config.BREAKOUT_RVOL_CONFIRM


def _above(close: float | None, level: float | None) -> bool | None:
    return None if close is None or level is None else close > level


def market_strip(
    daily_breakout: pd.DataFrame, symbols: tuple[str, ...] = playbook.MARKET_SYMBOLS
) -> dict[str, Any]:
    """Per ETF: last daily % change, close vs SMA50 / SMA200, and a heavy-volume support break.

    The % change compares the last two closes in the daily raw file. The SMA
    and support facts come from the latest daily breakout row (``heavy_break``:
    :func:`heavy_break`). Plus the growth-vs-dividend label.
    """
    wanted = (*symbols, config.ROTATION_GROWTH, config.ROTATION_DIVIDEND)
    frames = {s: daily_closes(s) for s in dict.fromkeys(wanted)}
    rows = []
    for sym in symbols:
        df = frames[sym]
        row = playbook.row_for(daily_breakout, sym) or {}
        last = None if df is None or df.empty else pd.Timestamp(df["Date"].iloc[-1])
        close = playbook.num(row.get("Close"))
        rows.append(
            {
                "symbol": sym,
                "change": None if df is None else _ret(df["Close"], 1),
                "as_of": None if last is None else last.date().isoformat(),
                "above_sma50": _above(close, playbook.num(row.get("SMA50"))),
                "above_sma200": _above(close, playbook.num(row.get("SMA200"))),
                "heavy_break": heavy_break(row),
                "rvol": playbook.num(row.get("RVOL20")),
            }
        )
    rot = rotation(frames[config.ROTATION_GROWTH], frames[config.ROTATION_DIVIDEND])
    return {"rows": rows, "rotation": rot["label"]}
