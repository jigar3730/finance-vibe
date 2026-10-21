"""Top setups: rank both scanners' rows together and derive rule-based key levels.

Pure pandas, no Flask, so it can be unit-tested. The dashboard's ``/`` page calls
:func:`load_latest`, :func:`top_setups` and :func:`action_plan`.

Every ticker the breakout scanner saw has a row with ``Resistance``, ``Support`` and
``ATR``, so a Coiled Cobra setup gets its levels from the same ticker's breakout row.
The levels are mechanical (the breakout confirmation rule and a structural stop);
neither scanner has a validated edge, so they are a plan to review, not advice.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd

COBRA_PREFIX = "coiled_cobra_setups_"
BREAKOUT_PREFIX = "breakout_setups_"
_DATED_CSV = re.compile(r"^(?P<prefix>.+_)(?P<date>\d{4}-\d{2}-\d{2})\.csv$")

# Breakout statuses that are events (FAILED_BREAKOUT is excluded on purpose).
BREAKOUT_EVENT_BONUS = {
    "BREAKOUT_CONFIRMED": 20,
    "PRE_BREAKOUT": 15,
    "BREAKOUT_TRIGGERED": 10,
}
FAILED_STATUS = "FAILED_BREAKOUT"
CONFIRMED_STATUS = "BREAKOUT_CONFIRMED"
AGREEMENT_BONUS = 10  # a Cobra setup the breakout scan also flags

# Mirrors scanners.breakout.RVOL_CONFIRM (tests/test_dashboard.py checks it); the
# dashboard doesn't import the scanner and its indicator stack.
RVOL_CONFIRM = 1.5
STOP_BUFFER_ATR = 0.25
EXTENDED_ATR = 1.0  # above trigger + this many ATR: don't chase
TARGET_R = (2.0, 3.0)
WIDE_STOP_PCT = 8.0


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


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric view of column ``name``; all-NaN when the column is absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(float("nan"), index=df.index, dtype="float64")


def _num(value: Any) -> float | None:
    v = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(v) else float(v)


def candidates(cobra: pd.DataFrame, breakout: pd.DataFrame) -> pd.DataFrame:
    """Every Cobra setup and breakout event, one row per symbol, ranked best first.

    Columns: the breakout row's fields, Cobra fields prefixed ``Cobra `` (``Cobra
    Score``, ``Cobra Grade`` ...), ``Sources`` (tuple) and ``Rank Score``.
    Symbols whose breakout status is FAILED_BREAKOUT, and rows without usable
    Resistance / Support / ATR, are dropped.
    """
    if breakout.empty or "Symbol" not in breakout.columns:
        return pd.DataFrame()
    bo = breakout.copy()
    bo["Symbol"] = bo["Symbol"].astype(str).str.strip().str.upper()
    bo = bo.drop_duplicates("Symbol").set_index("Symbol")

    cb = pd.DataFrame()
    if not cobra.empty and "Symbol" in cobra.columns:
        cb = cobra.copy()
        cb["Symbol"] = cb["Symbol"].astype(str).str.strip().str.upper()
        cb = cb.drop_duplicates("Symbol").set_index("Symbol").add_prefix("Cobra ")

    status = bo.get("Status", pd.Series(dtype=object))
    events = set(status[status.isin(BREAKOUT_EVENT_BONUS)].index)
    pool = (set(cb.index) | events) & set(bo.index)
    pool -= set(status[status == FAILED_STATUS].index)
    if not pool:
        return pd.DataFrame()

    rows = bo.loc[sorted(pool)].join(cb, how="left")
    has_levels = _num_col(rows, "Resistance").notna() & _num_col(rows, "Support").notna()
    rows = rows[has_levels & (_num_col(rows, "ATR") > 0)]
    if rows.empty:
        return pd.DataFrame()

    cobra_score = _num_col(rows, "Cobra Score")
    readiness = _num_col(rows, "Breakout Readiness").fillna(0)
    bonus = rows["Status"].map(BREAKOUT_EVENT_BONUS)
    is_cobra, is_event = cobra_score.notna(), bonus.notna()
    breakout_rank = readiness + bonus.fillna(0)

    rank = cobra_score.where(is_cobra, breakout_rank)
    both = is_cobra & is_event
    rank[both] = pd.concat([cobra_score, breakout_rank], axis=1).max(axis=1)[both] + AGREEMENT_BONUS

    rows = rows.assign(**{"Rank Score": rank.round(1)})
    rows["Sources"] = [
        tuple(s for s, on in (("cobra", c), ("breakout", e)) if on)
        for c, e in zip(is_cobra, is_event, strict=True)
    ]
    rows = rows.assign(_cobra=cobra_score.fillna(-1), _ready=readiness).reset_index()
    rows = rows.sort_values(
        ["Rank Score", "_cobra", "_ready", "Symbol"], ascending=[False, False, False, True]
    )
    return rows.drop(columns=["_cobra", "_ready"]).reset_index(drop=True)


def top_setups(cobra: pd.DataFrame, breakout: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The ``n`` best-ranked candidates (see :func:`candidates`)."""
    return candidates(cobra, breakout).head(n)


# ---------------------------------------------------------------------------
# Levels and action plan
# ---------------------------------------------------------------------------


def action_plan(row: pd.Series | dict[str, Any], live_price: float | None = None) -> dict[str, Any]:
    """Trigger / stop / targets and a status for one candidate row.

    Trigger is the breakout Resistance (a close above it on RVOL20 >= RVOL_CONFIRM
    and above SMA20 confirms). Stop is the higher of Support and Cobra Swing Low
    (when below the trigger) minus ``STOP_BUFFER_ATR`` x ATR. Targets are
    ``TARGET_R`` multiples of the risk above the trigger. Status uses the live
    price when given, else the scan's Close.
    """
    entry = _num(row.get("Resistance"))
    atr = _num(row.get("ATR"))
    if entry is None or atr is None or atr <= 0:
        raise ValueError("row needs Resistance and a positive ATR")
    support = _num(row.get("Support"))
    floors = [support, _num(row.get("Cobra Swing Low"))]
    floor = max((f for f in floors if f is not None and f < entry), default=None)
    if floor is None:
        raise ValueError("row needs a Support or Swing Low below the trigger")
    stop = floor - STOP_BUFFER_ATR * atr
    risk = entry - stop
    close = _num(row.get("Close"))
    price = live_price if live_price is not None else close

    if price is None:
        state, note = "UNKNOWN", "No price available."
    elif price < stop:
        state, note = "INVALIDATED", "Price is below the stop; the setup is broken."
    elif price < entry:
        gap = (entry - price) / atr
        state, note = "WAITING", f"{gap:.1f} ATR below the trigger."
    elif price <= entry + EXTENDED_ATR * atr:
        state = "TRIGGERED"
        if row.get("Status") == CONFIRMED_STATUS:
            note = "Breakout confirmed by the scan (close above on volume)."
        else:
            note = f"Above the trigger; needs a close above it on RVOL ≥ {RVOL_CONFIRM:g}."
    else:
        state = "EXTENDED"
        note = "More than 1 ATR above the trigger: don't chase, wait for a pullback toward it."

    risk_pct = 100 * risk / entry
    return {
        "trigger": round(entry, 2),
        "stop": round(stop, 2),
        "support": None if support is None else round(support, 2),
        "risk": round(risk, 2),
        "risk_pct": round(risk_pct, 1),
        "targets": [round(entry + r * risk, 2) for r in TARGET_R],
        "price": None if price is None else round(price, 2),
        "price_is_live": live_price is not None,
        "status": state,
        "note": note,
        "wide_stop": risk_pct > WIDE_STOP_PCT,
        "condition": f"Close above {entry:.2f} with RVOL20 ≥ {RVOL_CONFIRM:g} and above SMA20",
    }
