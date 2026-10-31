"""Home page sections by evidence, rule-based key levels, and the Market tab rows.

Pure pandas, no Flask, so it can be unit-tested. The dashboard's ``/`` page calls
:func:`load_latest`, :func:`weekly_confirmed`, :func:`untested_cobra` and
:func:`failed_counts`, which sort the scans by what the research found
(``RESEARCH.md`` §7, §11): weekly BREAKOUT_CONFIRMED passed in-sample (holdout not
yet run), Coiled Cobra was never tested, and every other breakout event failed.
Nothing here ranks or scores.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

COBRA_PREFIX = "coiled_cobra_setups_"
BREAKOUT_PREFIX = "breakout_setups_"
_DATED_CSV = re.compile(r"^(?P<prefix>.+_)(?P<date>\d{4}-\d{2}-\d{2})\.csv$")

FAILED_STATUS = "FAILED_BREAKOUT"
CONFIRMED_STATUS = "BREAKOUT_CONFIRMED"
# Breakout events that failed the in-sample test, per mode (RESEARCH.md §7, §11).
FAILED_IN_SAMPLE = {
    "daily": ("PRE_BREAKOUT", "BREAKOUT_TRIGGERED", CONFIRMED_STATUS),
    "weekly": ("PRE_BREAKOUT", "BREAKOUT_TRIGGERED"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_scan(path: str) -> pd.DataFrame:
    """Read one scanner CSV with stripped column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def relabel_triggered_watch(df: pd.DataFrame) -> pd.DataFrame:
    """Scans before 2026-10-20 filed triggered-but-unconfirmed rows under WATCH.

    The scanner's ``Breakout Triggered`` flag identifies them (a triggered row only
    reaches WATCH via that rule), so older CSVs get the same BREAKOUT_TRIGGERED status.
    """
    if "Status" not in df.columns or "Breakout Triggered" not in df.columns:
        return df
    triggered = df["Breakout Triggered"].astype(str).str.strip().str.lower() == "true"
    df.loc[(df["Status"] == "WATCH") & triggered, "Status"] = "BREAKOUT_TRIGGERED"
    return df


def run_dates(folder: str) -> dict[str, set[str]]:
    """Dates in ``folder`` mapped to the scanner prefixes that wrote a CSV that day."""
    dates: dict[str, set[str]] = {}
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            m = _DATED_CSV.match(name)
            if m and m["prefix"] in (COBRA_PREFIX, BREAKOUT_PREFIX):
                dates.setdefault(m["date"], set()).add(m["prefix"])
    return dates


def load_latest(folder: str) -> tuple[str | None, pd.DataFrame, pd.DataFrame]:
    """Newest run in ``folder`` with a breakout CSV: (date, cobra, breakout).

    The Cobra frame is empty when that day has no Cobra CSV.
    """
    dated = [d for d, prefixes in run_dates(folder).items() if BREAKOUT_PREFIX in prefixes]
    if not dated:
        return None, pd.DataFrame(), pd.DataFrame()
    date = max(dated)
    breakout = relabel_triggered_watch(
        read_scan(os.path.join(folder, f"{BREAKOUT_PREFIX}{date}.csv"))
    )
    cobra_path = os.path.join(folder, f"{COBRA_PREFIX}{date}.csv")
    cobra = read_scan(cobra_path) if os.path.exists(cobra_path) else pd.DataFrame()
    return date, cobra, breakout


def load_previous_breakout(folder: str, date: str | None) -> pd.DataFrame:
    """The newest breakout CSV in ``folder`` dated before ``date``; empty if none."""
    dated = [
        d
        for d, prefixes in run_dates(folder).items()
        if BREAKOUT_PREFIX in prefixes and d < (date or "")
    ]
    if not dated:
        return pd.DataFrame()
    return relabel_triggered_watch(
        read_scan(os.path.join(folder, f"{BREAKOUT_PREFIX}{max(dated)}.csv"))
    )


# ---------------------------------------------------------------------------
# Home page sections
# ---------------------------------------------------------------------------


def live_price_or_none(value: object) -> float | None:
    """A usable live quote, or None (missing, "N/A", NaN or infinite)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def num(value: Any) -> float | None:
    v = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(v) else float(v)


def _symbols_upper(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    return df.drop_duplicates("Symbol")


def weekly_confirmed(breakout: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Weekly BREAKOUT_CONFIRMED rows, by symbol, with ``Confirmed Last Run`` and ``In Last Run``.

    ``Confirmed Last Run`` is True when ``previous`` (the prior weekly scan) had
    the symbol CONFIRMED too: a continuation, not the first-bar event that was
    tested. ``In Last Run`` is False when ``previous`` has no row for the symbol,
    so whether this is its first CONFIRMED week is unknown.
    """
    if breakout.empty or "Status" not in breakout.columns:
        return pd.DataFrame()
    rows = _symbols_upper(breakout[breakout["Status"] == CONFIRMED_STATUS])
    before: set[str] = set()
    scanned: set[str] = set()
    if not previous.empty and {"Symbol", "Status"} <= set(previous.columns):
        prev = _symbols_upper(previous)
        before = set(prev.loc[prev["Status"] == CONFIRMED_STATUS, "Symbol"])
        scanned = set(prev["Symbol"])
    rows["Confirmed Last Run"] = rows["Symbol"].isin(before)
    rows["In Last Run"] = rows["Symbol"].isin(scanned)
    return rows.sort_values("Symbol").reset_index(drop=True)


def untested_cobra(cobra: pd.DataFrame, breakout: pd.DataFrame) -> pd.DataFrame:
    """Coiled Cobra setups by symbol, with the same ticker's breakout ``Status``."""
    if cobra.empty or "Symbol" not in cobra.columns:
        return pd.DataFrame()
    rows = _symbols_upper(cobra)
    status = pd.Series(dtype=object)
    if not breakout.empty and {"Symbol", "Status"} <= set(breakout.columns):
        status = _symbols_upper(breakout).set_index("Symbol")["Status"]
    rows["Breakout Status"] = rows["Symbol"].map(status)
    return rows.sort_values("Symbol").reset_index(drop=True)


def failed_counts(breakout: pd.DataFrame, mode: str) -> dict[str, int]:
    """Row count per breakout status that failed in-sample for ``mode`` (zeros kept)."""
    statuses = FAILED_IN_SAMPLE[mode]
    if breakout.empty or "Status" not in breakout.columns:
        return dict.fromkeys(statuses, 0)
    counts = breakout["Status"].value_counts()
    return {s: int(counts.get(s, 0)) for s in statuses}


# ---------------------------------------------------------------------------
# Chart levels
# ---------------------------------------------------------------------------

LEVEL_COLORS = {"resistance": "#2962ff", "support": "#787b86"}
SOLID, DOTTED = 0, 1  # Lightweight Charts LineStyle


def _line(price: float | None, label: str, color: str, style: int) -> dict[str, Any] | None:
    if price is None:
        return None
    return {"price": price, "label": label, "color": color, "style": style}


# ---------------------------------------------------------------------------
# Market home: SPY / QQQ / SCHG / SCHD facts from the breakout scanner's states
# ---------------------------------------------------------------------------

MARKET_SYMBOLS = ("SPY", "QQQ", "SCHG", "SCHD")
SETUPS_MARKET_SYMBOLS = ("SPY", "QQQ")  # their facts head the Setups page


def market_facts(daily: dict[str, Any] | None, weekly: dict[str, Any] | None) -> list[str]:
    """Plain facts for one ETF: weekly Trend and MTF, daily close vs its SMA50. No rule."""
    facts = (
        [f"weekly trend {weekly.get('Trend')}", f"weekly MTF {weekly.get('MTF')}"]
        if weekly
        else ["no weekly scan row"]
    )
    if not daily:
        return [*facts, "no daily scan row"]
    close, sma50 = num(daily.get("Close")), num(daily.get("SMA50"))
    if close is None or sma50 is None:
        return [*facts, "daily 50-day average unavailable"]
    side = "above" if close > sma50 else "below"
    return [*facts, f"daily close {side} its 50-day average ({sma50:,.2f})"]


def row_for(df: pd.DataFrame, symbol: str) -> dict[str, Any] | None:
    if df.empty or "Symbol" not in df.columns:
        return None
    hit = df[df["Symbol"].astype(str).str.strip().str.upper() == symbol]
    return None if hit.empty else {str(k): v for k, v in hit.iloc[0].items()}


def range_levels(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Chart lines for a scan row's Resistance and Support, nothing else."""
    lines = [
        _line(num(row.get("Resistance")), "Resistance", LEVEL_COLORS["resistance"], SOLID),
        _line(num(row.get("Support")), "Support", LEVEL_COLORS["support"], DOTTED),
    ]
    return [line for line in lines if line is not None]


def market_rows(
    daily_breakout: pd.DataFrame,
    weekly_breakout: pd.DataFrame,
    live: dict[str, float | str] | None = None,
    symbols: tuple[str, ...] = MARKET_SYMBOLS,
) -> list[dict[str, Any]]:
    """One dict per ETF: its scan rows, plain facts, price and chart lines.

    ``price`` is the live quote when usable, else the daily Close. A symbol
    without a daily row gets ``levels`` = None.
    """
    rows = []
    for symbol in symbols:
        daily, weekly = row_for(daily_breakout, symbol), row_for(weekly_breakout, symbol)
        live_price = live_price_or_none((live or {}).get(symbol))
        close = num(daily.get("Close")) if daily else None
        rows.append(
            {
                "symbol": symbol,
                "daily": daily,
                "weekly": weekly,
                "facts": market_facts(daily, weekly),
                "price": live_price if live_price is not None else close,
                "price_is_live": live_price is not None,
                "levels": range_levels(daily) if daily else None,
            }
        )
    return rows
