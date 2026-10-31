"""Market health facts for the home page: volatility, credit, breadth.

Read from the market-context silo (``config.MARKET_CONTEXT_PROFILE``, written by
:mod:`finance_vibe.data.market_ingest`), last daily closes only. Facts, never a
verdict: no score, no risk-on / risk-off label, nothing feeds a scanner.

Each block carries ``flags``: short notes when a file is missing, older than
the last closed session, or has an error row in the newest market
``data_quality_<date>.csv``.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

import pandas as pd

from finance_vibe import config, indicators
from finance_vibe.data import raw
from finance_vibe.web import insights

logger = logging.getLogger(__name__)

VIX = "VIX"
CREDIT = ("HYG", "IEF")  # high-yield bonds / 7-10 year Treasuries
VIX_MA_BARS = 20
RATIO_MA_BARS = 50
SECTOR_MA_BARS = 50
MONTH_BARS = 21


def closes(symbol: str) -> pd.DataFrame | None:
    """``Date`` / ``Close`` from a market-context raw file, or None if unavailable."""
    path = config.get_raw_path(symbol, config.MARKET_CONTEXT_PROFILE)
    if not os.path.isfile(path):
        return None
    try:
        df = raw.load_raw(path)[["Date", "Close"]]
    except (OSError, ValueError) as exc:
        logger.warning("Market-context raw for %s unreadable: %s", symbol, exc)
        return None
    return None if df.empty else df


def _last_ma(close: pd.Series, bars: int) -> float | None:
    ma = indicators.sma(close.reset_index(drop=True), bars)
    return None if ma is None or pd.isna(ma.iloc[-1]) else float(ma.iloc[-1])


def _as_of(df: pd.DataFrame) -> date:
    return pd.Timestamp(df["Date"].iloc[-1]).date()


def _flags(
    symbols: tuple[str, ...],
    frames: dict[str, pd.DataFrame | None],
    expected: date,
    errors: dict[str, list[str]],
) -> list[str]:
    out = []
    for sym in symbols:
        df = frames.get(sym)
        if df is None:
            out.append(f"{sym} missing")
            continue
        checks = list(errors.get(sym, []))
        if _as_of(df) < expected and "stale" not in checks:
            checks.append("stale")
        out += [f"{sym} {c.replace('_', ' ')}" for c in checks]
    return out


def volatility(vix: pd.DataFrame | None) -> dict[str, Any] | None:
    """VIX close, change vs the previous close (points), and its 20-day average."""
    if vix is None:
        return None
    c = vix["Close"]
    ma = _last_ma(c, VIX_MA_BARS)
    level = float(c.iloc[-1])
    return {
        "level": level,
        "change": float(c.iloc[-1] - c.iloc[-2]) if len(c) > 1 else None,
        "ma": ma,
        "above_ma": None if ma is None else level > ma,
        "as_of": _as_of(vix).isoformat(),
    }


def credit(hyg: pd.DataFrame | None, ief: pd.DataFrame | None) -> dict[str, Any] | None:
    """HYG / IEF close ratio, its 50-day average, and its change over about a month."""
    if hyg is None or ief is None:
        return None
    both = hyg.merge(ief, on="Date", suffixes=(" H", " I")).sort_values("Date")
    if both.empty:
        return None
    ratio = (both["Close H"] / both["Close I"]).reset_index(drop=True)
    ma = _last_ma(ratio, RATIO_MA_BARS)
    now = float(ratio.iloc[-1])
    month = (
        float(ratio.iloc[-1] / ratio.iloc[-1 - MONTH_BARS] - 1.0)
        if len(ratio) > MONTH_BARS
        else None
    )
    return {
        "ratio": now,
        "ma": ma,
        "above_ma": None if ma is None else now > ma,
        "month_change": month,
        "as_of": _as_of(both).isoformat(),
    }


def breadth(frames: dict[str, pd.DataFrame | None]) -> dict[str, Any]:
    """Sector funds above / below their 50-day average, with plain sector names."""
    above: list[dict[str, str]] = []
    below: list[dict[str, str]] = []
    unknown: list[dict[str, str]] = []
    dates = []
    for sym, name in config.MARKET_SECTOR_FUNDS.items():
        df = frames.get(sym)
        ma = None if df is None else _last_ma(df["Close"], SECTOR_MA_BARS)
        item = {"symbol": sym, "name": name}
        if df is None or ma is None:
            unknown.append(item)
            continue
        dates.append(_as_of(df))
        (above if float(df["Close"].iloc[-1]) > ma else below).append(item)
    return {
        "above": above,
        "below": below,
        "unknown": unknown,
        "counted": len(above) + len(below),
        "total": len(config.MARKET_SECTOR_FUNDS),
        "as_of": max(dates).isoformat() if dates else None,
    }


def market_health(expected: date) -> dict[str, Any]:
    """Volatility, credit and breadth facts; ``expected`` is the last closed session."""
    symbols = (VIX, *CREDIT, *config.MARKET_SECTOR_FUNDS)
    frames = {s: closes(s) for s in symbols}
    errors = insights.quality_errors(config.MARKET_CONTEXT_PROFILE["logs_dir"])
    sectors = tuple(config.MARKET_SECTOR_FUNDS)
    return {
        "volatility": volatility(frames[VIX]),
        "volatility_flags": _flags((VIX,), frames, expected, errors),
        "credit": credit(frames[CREDIT[0]], frames[CREDIT[1]]),
        "credit_flags": _flags(CREDIT, frames, expected, errors),
        "breadth": breadth(frames),
        "breadth_flags": _flags(sectors, frames, expected, errors),
    }
