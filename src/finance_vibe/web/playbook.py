"""Top setups: rank both scanners' rows together and derive rule-based key levels.

Pure pandas, no Flask, so it can be unit-tested. The dashboard's ``/`` page calls
:func:`load_latest`, :func:`top_setups` and :func:`action_plan`.

Every ticker the breakout scanner saw has a row with ``Resistance``, ``Support`` and
``ATR``, so a Coiled Cobra setup gets its levels from the same ticker's breakout row.
The levels are mechanical (the breakout confirmation rule and a structural stop);
neither scanner has a validated edge, so they are a plan to review, not advice.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

import pandas as pd

from finance_vibe import config

logger = logging.getLogger(__name__)

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

RVOL_CONFIRM = config.BREAKOUT_RVOL_CONFIRM
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


def live_price_or_none(value: object) -> float | None:
    """A usable live quote, or None (missing, "N/A", NaN or infinite)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def num(value: Any) -> float | None:
    v = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(v) else float(v)


def candidates(cobra: pd.DataFrame, breakout: pd.DataFrame) -> pd.DataFrame:
    """Every Cobra setup and breakout event, one row per symbol, ranked best first.

    Columns: the breakout row's fields, Cobra fields prefixed ``Cobra `` (``Cobra
    Score``, ``Cobra Grade`` ...), ``Sources`` (tuple) and ``Rank Score``.
    Symbols whose breakout status is FAILED_BREAKOUT, and rows without usable
    Resistance / Support / ATR, are dropped; so are Cobra setups the breakout
    scan has no row for (logged at INFO), since their levels come from that row.
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

    if "Status" not in bo.columns:
        bo["Status"] = None
    status = bo["Status"]
    events = set(status[status.isin(BREAKOUT_EVENT_BONUS)].index)
    unmatched = sorted(set(cb.index) - set(bo.index))
    if unmatched:
        logger.info(
            "Top setups: %d Coiled Cobra setup(s) have no breakout row and no levels: %s",
            len(unmatched),
            ", ".join(unmatched),
        )
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


def action_plan(
    row: pd.Series | dict[str, Any], live_price: float | None = None, side: str = "long"
) -> dict[str, Any]:
    """Trigger / stop / targets and a status for one row, long or short.

    Long: trigger is the breakout Resistance (a close above it on RVOL20 >=
    RVOL_CONFIRM and above SMA20 confirms); stop is the higher of Support and Cobra
    Swing Low (when below the trigger) minus ``STOP_BUFFER_ATR`` x ATR.
    Short (the mirror): trigger is Support (a close below it); stop is the lower of
    Resistance and SMA20 (when above the trigger) plus the same buffer.
    Targets are ``TARGET_R`` multiples of the risk beyond the trigger. Status uses
    the live price when given, else the scan's Close.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', not {side!r}")
    long = side == "long"
    atr = num(row.get("ATR"))
    support, resistance = num(row.get("Support")), num(row.get("Resistance"))
    entry = resistance if long else support
    if entry is None or atr is None or atr <= 0:
        raise ValueError("row needs a trigger level (Resistance / Support) and a positive ATR")
    if long:
        floors = [support, num(row.get("Cobra Swing Low"))]
        level = max((f for f in floors if f is not None and f < entry), default=None)
    else:
        ceilings = [resistance, num(row.get("SMA20"))]
        level = min((c for c in ceilings if c is not None and c > entry), default=None)
    if level is None:
        raise ValueError("row needs a stop level on the other side of the trigger")
    d = 1 if long else -1  # +1: profit when price rises
    stop = level - d * STOP_BUFFER_ATR * atr
    risk = d * (entry - stop)
    close = num(row.get("Close"))
    live_price = live_price_or_none(live_price)
    price = live_price if live_price is not None else close
    beyond, toward = ("above", "below") if long else ("below", "above")

    if price is None:
        state, note = "UNKNOWN", "No price available."
    elif d * (price - stop) < 0:
        state, note = "INVALIDATED", f"Price is {toward} the stop; the setup is broken."
    elif d * (price - entry) < 0:
        gap = d * (entry - price) / atr
        state, note = "WAITING", f"{gap:.1f} ATR {toward} the trigger."
    elif d * (price - entry) <= EXTENDED_ATR * atr:
        state = "TRIGGERED"
        if not long:
            note = (
                "Below the trigger. The breakout scan does not confirm breakdowns; "
                "judge the close and volume yourself."
            )
        elif row.get("Status") == CONFIRMED_STATUS:
            note = "Breakout confirmed by the scan (close above on volume)."
        else:
            note = f"Above the trigger; needs a close above it on RVOL ≥ {RVOL_CONFIRM:g}."
    else:
        state = "EXTENDED"
        note = (
            f"More than {EXTENDED_ATR:g} ATR {beyond} the trigger: don't chase, "
            "wait for a pullback toward it."
        )

    risk_pct = 100 * risk / entry
    return {
        "side": side,
        "trigger": round(entry, 2),
        "stop": round(stop, 2),
        "support": None if support is None else round(support, 2),
        "resistance": None if resistance is None else round(resistance, 2),
        "risk": round(risk, 2),
        "risk_pct": round(risk_pct, 1),
        "targets": [round(entry + d * r * risk, 2) for r in TARGET_R],
        "price": None if price is None else round(price, 2),
        "price_is_live": live_price is not None,
        "status": state,
        "note": note,
        "wide_stop": risk_pct > WIDE_STOP_PCT,
        "condition": (
            f"Close above {entry:.2f} with RVOL20 ≥ {RVOL_CONFIRM:g} and above SMA20"
            if long
            else f"Close below {entry:.2f} (mirror of the long rule; not a scanner signal)"
        ),
    }


# ---------------------------------------------------------------------------
# Chart levels (Top Setups plans; Market cards use the same colours)
# ---------------------------------------------------------------------------

LEVEL_COLORS = {
    "trigger": "#2962ff",
    "stop": "#f23645",
    "target": "#26a69a",
    "support": "#787b86",
}
SOLID, DOTTED, DASHED = 0, 1, 2  # Lightweight Charts LineStyle


def _line(price: float | None, label: str, color: str, style: int) -> dict[str, Any] | None:
    if price is None:
        return None
    return {"price": price, "label": label, "color": color, "style": style}


def chart_levels(plan: dict[str, Any], *, with_support: bool = False) -> list[dict[str, Any]]:
    """Price lines for one plan: trigger, stop and both targets (and Support if asked)."""
    prefix = "" if plan["side"] == "long" else "Short "
    lines = [
        _line(plan["trigger"], f"{prefix}Trigger", LEVEL_COLORS["trigger"], SOLID),
        _line(plan["stop"], f"{prefix}Stop", LEVEL_COLORS["stop"], SOLID),
        _line(plan["targets"][0], f"{prefix}T1", LEVEL_COLORS["target"], DASHED),
        _line(plan["targets"][1], f"{prefix}T2", LEVEL_COLORS["target"], DASHED),
    ]
    if with_support:
        lines.append(_line(plan.get("support"), "Support", LEVEL_COLORS["support"], DOTTED))
    return [line for line in lines if line is not None]


# ---------------------------------------------------------------------------
# Market tab: SPY / QQQ / SCHG / SCHD bias from the breakout scanner's states
# ---------------------------------------------------------------------------

MARKET_SYMBOLS = ("SPY", "QQQ", "SCHG", "SCHD")
HEADWIND_SYMBOLS = ("SPY", "QQQ")  # a SHORT bias here puts a banner on Top Setups
_LONG_MTF = {"ALIGNED", "PARTIAL"}


def market_bias(
    daily: dict[str, Any] | None, weekly: dict[str, Any] | None
) -> tuple[str, list[str]]:
    """LONG / SHORT / NEUTRAL for one ETF, with the reasons shown on its card.

    LONG: weekly Trend BULLISH, weekly MTF ALIGNED or PARTIAL, daily Close above
    daily SMA50. SHORT: weekly Trend or MTF BEARISH, daily Close below daily SMA50.
    Anything else (or a missing row) is NEUTRAL.
    """
    if not daily or not weekly:
        return "NEUTRAL", ["missing a daily or weekly scan row"]
    trend, mtf = weekly.get("Trend"), weekly.get("MTF")
    close, sma50 = num(daily.get("Close")), num(daily.get("SMA50"))
    reasons = [f"weekly trend {trend}", f"weekly MTF {mtf}"]
    if close is None or sma50 is None:
        return "NEUTRAL", [*reasons, "daily SMA50 unavailable"]
    above = close > sma50
    reasons.append(f"daily close {'above' if above else 'below'} SMA50 ({sma50:,.2f})")
    if trend == "BULLISH" and mtf in _LONG_MTF and above:
        return "LONG", reasons
    if (trend == "BEARISH" or mtf == "BEARISH") and not above:
        return "SHORT", reasons
    return "NEUTRAL", reasons


def row_for(df: pd.DataFrame, symbol: str) -> dict[str, Any] | None:
    if df.empty or "Symbol" not in df.columns:
        return None
    hit = df[df["Symbol"].astype(str).str.strip().str.upper() == symbol]
    return None if hit.empty else {str(k): v for k, v in hit.iloc[0].items()}


def market_levels(daily: dict[str, Any]) -> list[dict[str, Any]]:
    """Market card chart lines: the daily row's Resistance and Support, nothing else."""
    lines = [
        _line(num(daily.get("Resistance")), "Resistance", LEVEL_COLORS["trigger"], SOLID),
        _line(num(daily.get("Support")), "Support", LEVEL_COLORS["support"], DOTTED),
    ]
    return [line for line in lines if line is not None]


def market_rows(
    daily_breakout: pd.DataFrame,
    weekly_breakout: pd.DataFrame,
    live: dict[str, float | str] | None = None,
    symbols: tuple[str, ...] = MARKET_SYMBOLS,
) -> list[dict[str, Any]]:
    """One dict per ETF: its scan rows, bias and reasons, price and chart lines.

    ``price`` is the live quote when usable, else the daily Close. A symbol
    without a daily row gets ``levels`` = None.
    """
    rows = []
    for symbol in symbols:
        daily, weekly = row_for(daily_breakout, symbol), row_for(weekly_breakout, symbol)
        bias, reasons = market_bias(daily, weekly)
        live_price = live_price_or_none((live or {}).get(symbol))
        close = num(daily.get("Close")) if daily else None
        rows.append(
            {
                "symbol": symbol,
                "daily": daily,
                "weekly": weekly,
                "bias": bias,
                "reasons": reasons,
                "price": live_price if live_price is not None else close,
                "price_is_live": live_price is not None,
                "levels": market_levels(daily) if daily else None,
            }
        )
    return rows
