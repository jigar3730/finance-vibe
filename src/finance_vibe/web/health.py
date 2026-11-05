"""Market health facts for the home page: volatility, credit, breadth.

Read from the market-context silo (``config.MARKET_CONTEXT_PROFILE``, written by
:mod:`finance_vibe.data.market_ingest`). With live ``quotes``
(:mod:`finance_vibe.data.market_snapshot`) the level moves to the live price, while
every average stays on closed bars. Facts, never a verdict: no score, no risk-on /
risk-off label, nothing feeds a scanner.

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


def volatility(
    vix: pd.DataFrame | None, quote: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """VIX level, change vs the previous close (points), and its 20-day closing average."""
    if vix is None:
        return None
    c = vix["Close"]
    ma = _last_ma(c, VIX_MA_BARS)
    live = insights.quote_is_new(vix, quote)
    level = float(quote["last"]) if quote is not None and live else float(c.iloc[-1])
    previous = c.iloc[-1] if live else (c.iloc[-2] if len(c) > 1 else None)
    change = None if previous is None else float(level - previous)
    return {
        "level": level,
        "change": change,
        "live": live,
        "ma": ma,
        "above_ma": None if ma is None else level > ma,
        "as_of": _as_of(vix).isoformat(),
    }


def credit(
    hyg: pd.DataFrame | None,
    ief: pd.DataFrame | None,
    quotes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """HYG / IEF ratio, its 50-day closing average, and its change over about a month.

    Live only when both funds have a new quote, so the ratio never mixes days.
    """
    quotes = quotes or {}
    if hyg is None or ief is None:
        return None
    both = hyg.merge(ief, on="Date", suffixes=(" H", " I")).sort_values("Date")
    if both.empty:
        return None
    ratio = (both["Close H"] / both["Close I"]).reset_index(drop=True)
    ma = _last_ma(ratio, RATIO_MA_BARS)
    h_quote, i_quote = quotes.get(CREDIT[0]), quotes.get(CREDIT[1])
    live = (
        h_quote is not None
        and i_quote is not None
        and insights.quote_is_new(both, h_quote)
        and insights.quote_is_new(both, i_quote)
    )
    if live:
        now = float(h_quote["last"] / i_quote["last"])  # type: ignore[index]
        base = ratio.iloc[-MONTH_BARS] if len(ratio) >= MONTH_BARS else None
    else:
        now = float(ratio.iloc[-1])
        base = ratio.iloc[-1 - MONTH_BARS] if len(ratio) > MONTH_BARS else None
    month = None if base is None else float(now / base - 1.0)
    return {
        "ratio": now,
        "live": live,
        "ma": ma,
        "above_ma": None if ma is None else now > ma,
        "month_change": month,
        "as_of": _as_of(both).isoformat(),
    }


def breadth(
    frames: dict[str, pd.DataFrame | None],
    quotes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sector funds above / below their 50-day closing average, with plain sector names.

    ``tiles`` lists every fund in ``config.MARKET_SECTOR_FUNDS`` order with its price
    (a new live quote, else the close), ``live``, 50-day average, ``pct`` (price /
    average - 1) and ``above`` (None without data);
    ``ranked`` is the same tiles by ``pct``, highest first, funds without data last.
    ``above`` / ``below`` / ``unknown`` list ``{"symbol", "name"}``.
    """
    quotes = quotes or {}
    tiles: list[dict[str, Any]] = []
    dates = []
    for sym, name in config.MARKET_SECTOR_FUNDS.items():
        df = frames.get(sym)
        ma = None if df is None else _last_ma(df["Close"], SECTOR_MA_BARS)
        close = None if df is None else float(df["Close"].iloc[-1])
        live = insights.quote_is_new(df, quotes.get(sym))
        if live:
            close = float(quotes[sym]["last"])
        if df is not None and ma is not None:
            dates.append(_as_of(df))
        above_ma = None if close is None or ma is None else close > ma
        tiles.append(
            {
                "symbol": sym,
                "name": name,
                "short": config.MARKET_SECTOR_SHORT.get(sym, name),
                "close": close,
                "live": live,
                "ma": ma,
                "pct": None if above_ma is None else close / ma - 1.0,  # type: ignore[operator]
                "above": above_ma,
            }
        )

    def names(state: bool | None) -> list[dict[str, str]]:
        return [{"symbol": t["symbol"], "name": t["name"]} for t in tiles if t["above"] is state]

    above, below = names(True), names(False)
    ranked = sorted(tiles, key=lambda t: (t["pct"] is None, -(t["pct"] or 0.0)))
    return {
        "tiles": tiles,
        "ranked": ranked,
        "above": above,
        "below": below,
        "unknown": names(None),
        "counted": len(above) + len(below),
        "total": len(config.MARKET_SECTOR_FUNDS),
        "as_of": max(dates).isoformat() if dates else None,
    }


def market_health(
    expected: date, quotes: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Volatility, credit and breadth facts; ``expected`` is the last closed session.

    ``quotes`` moves the levels to live prices. Every ``as_of`` stays the newest close.
    """
    quotes = quotes or {}
    symbols = (VIX, *CREDIT, *config.MARKET_SECTOR_FUNDS)
    frames = {s: closes(s) for s in symbols}
    errors = insights.quality_errors(config.MARKET_CONTEXT_PROFILE["logs_dir"])
    sectors = tuple(config.MARKET_SECTOR_FUNDS)
    return {
        "volatility": volatility(frames[VIX], quotes.get(VIX)),
        "volatility_flags": _flags((VIX,), frames, expected, errors),
        "credit": credit(frames[CREDIT[0]], frames[CREDIT[1]], quotes),
        "credit_flags": _flags(CREDIT, frames, expected, errors),
        "breadth": breadth(frames, quotes),
        "breadth_flags": _flags(sectors, frames, expected, errors),
    }
