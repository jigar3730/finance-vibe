"""Market health facts for the home page: volatility, rates, dollar, credit, breadth.

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

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from finance_vibe import config
from finance_vibe.data import daily_ingest, market_snapshot, raw
from finance_vibe.web import insights

logger = logging.getLogger(__name__)

VIX = "VIX"
VIX3M = "VIX3M"
TNX = "TNX"
DXY = "DXY"
CREDIT = ("HYG", "IEF")  # high-yield bonds / 7-10 year Treasuries
VIX_MA_BARS = 20
RATIO_MA_BARS = 50
SECTOR_MA_BARS = 50
MONTH_BARS = 21
YEAR_BARS = 252
SCAN_GRACE = timedelta(hours=2)


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


_last_ma = insights.last_ma


def _level(df: pd.DataFrame, quote: dict[str, Any] | None) -> dict[str, Any]:
    """A series' level (a new quote, else the newest close), the close before it and the
    close about a month before it."""
    close = df["Close"]
    level, live = insights.price_now(df, quote)
    past = close if live else close.iloc[:-1]
    return {
        "level": level,
        "live": live,
        "previous": float(past.iloc[-1]) if len(past) else None,
        "month_base": float(past.iloc[-MONTH_BARS]) if len(past) >= MONTH_BARS else None,
    }


def _as_of(df: pd.DataFrame) -> date:
    return pd.Timestamp(df["Date"].iloc[-1]).date()


def _flags(
    symbols: tuple[str, ...],
    frames: dict[str, pd.DataFrame | None],
    expected: date,
    errors: dict[str, list[str]],
    snapshot: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> list[str]:
    quotes = (snapshot or {}).get("quotes", {})
    out = []
    for sym in symbols:
        df = frames.get(sym)
        if df is None:
            out.append(f"{sym} missing")
            continue
        checks = list(errors.get(sym, []))
        if _as_of(df) < expected and "stale" not in checks:
            checks.append("stale")
        if now is not None:
            note = market_snapshot.quote_note(_as_of(df), quotes.get(sym), snapshot, now)
            checks += [note] if note else []
        out += [f"{sym} {c.replace('_', ' ')}" for c in checks]
    return out


def volatility(
    vix: pd.DataFrame | None, quote: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """VIX level, change vs the previous close (points), its 20-day closing average, and
    ``rank``: the % of the past year's closes it is above (None under a year of history)."""
    if vix is None:
        return None
    c = vix["Close"]
    ma = _last_ma(c, VIX_MA_BARS)
    now = _level(vix, quote)
    level, previous = now["level"], now["previous"]
    year = c.tail(YEAR_BARS)
    return {
        "level": level,
        "change": None if previous is None else level - previous,
        "live": now["live"],
        "ma": ma,
        "above_ma": None if ma is None else level > ma,
        "rank": float((year < level).mean() * 100) if len(year) == YEAR_BARS else None,
        "as_of": _as_of(vix).isoformat(),
    }


def vol_curve(
    vix: pd.DataFrame | None,
    vix3m: pd.DataFrame | None,
    quotes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """VIX / VIX3M and the same ratio at the previous close. Live only with both quotes."""
    if vix is None or vix3m is None:
        return None
    got = _ratio_now(vix, vix3m, (quotes or {}).get(VIX), (quotes or {}).get(VIX3M))
    if got is None:
        return None
    return {k: got[k] for k in ("ratio", "live", "previous", "as_of")}


def _ratio_now(
    top: pd.DataFrame,
    bottom: pd.DataFrame,
    top_quote: dict[str, Any] | None,
    bottom_quote: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Close ratio of two series on shared dates, moved to live only when both quotes are
    new, so a ratio never mixes days."""
    both = top.merge(bottom, on="Date", suffixes=(" T", " B")).sort_values("Date")
    if both.empty:
        return None
    ratio = both[["Date"]].assign(Close=both["Close T"] / both["Close B"]).reset_index(drop=True)
    live = insights.quote_is_new(both, top_quote) and insights.quote_is_new(both, bottom_quote)
    quote = (
        {"last": top_quote["last"] / bottom_quote["last"], "date": top_quote["date"]}
        if (live and top_quote is not None and bottom_quote is not None)
        else None
    )
    now = _level(ratio, quote)
    return {
        "ratio": now["level"],
        "live": now["live"],
        "previous": now["previous"],
        "month_base": now["month_base"],
        "series": ratio["Close"],
        "as_of": _as_of(both).isoformat(),
    }


def rates(tnx: pd.DataFrame | None, quote: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """10-year Treasury yield in %, its change since the previous close and over about a
    month, both in points."""
    if tnx is None:
        return None
    now = _level(tnx, quote)
    level, previous, base = now["level"], now["previous"], now["month_base"]
    return {
        "level": level,
        "live": now["live"],
        "change": None if previous is None else level - previous,
        "month_change": None if base is None else level - base,
        "as_of": _as_of(tnx).isoformat(),
    }


def dollar(dxy: pd.DataFrame | None, quote: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """US dollar index, its % change since the previous close and over about a month."""
    if dxy is None:
        return None
    now = _level(dxy, quote)
    level, previous, base = now["level"], now["previous"], now["month_base"]
    return {
        "level": level,
        "live": now["live"],
        "change": None if previous is None else level / previous - 1.0,
        "month_change": None if base is None else level / base - 1.0,
        "as_of": _as_of(dxy).isoformat(),
    }


def credit(
    hyg: pd.DataFrame | None,
    ief: pd.DataFrame | None,
    quotes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """HYG / IEF ratio, its 50-day closing average, and its % change since the previous
    close and over about a month. Live only when both funds have a new quote."""
    quotes = quotes or {}
    if hyg is None or ief is None:
        return None
    got = _ratio_now(hyg, ief, quotes.get(CREDIT[0]), quotes.get(CREDIT[1]))
    if got is None:
        return None
    now, previous, base = got["ratio"], got["previous"], got["month_base"]
    ma = _last_ma(got["series"], RATIO_MA_BARS)
    return {
        "ratio": now,
        "live": got["live"],
        "ma": ma,
        "above_ma": None if ma is None else now > ma,
        "day_change": None if previous is None else now / previous - 1.0,
        "month_change": None if base is None else now / base - 1.0,
        "as_of": got["as_of"],
    }


def breadth(
    frames: dict[str, pd.DataFrame | None],
    quotes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sector funds above / below their 50-day closing average, with plain sector names.

    ``tiles`` lists every fund in ``config.MARKET_SECTOR_FUNDS`` order with its price
    (a new live quote, else the close), ``live``, ``day`` (% change since the previous
    close), 50-day average, ``pct`` (price / average - 1) and ``above`` (None without data);
    ``ranked`` is the same tiles by ``day``, highest first, funds without data last.
    ``above`` / ``below`` / ``unknown`` list ``{"symbol", "name"}``.
    """
    quotes = quotes or {}
    tiles: list[dict[str, Any]] = []
    dates = []
    for sym, name in config.MARKET_SECTOR_FUNDS.items():
        df = frames.get(sym)
        ma = None if df is None else _last_ma(df["Close"], SECTOR_MA_BARS)
        now = None if df is None else _level(df, quotes.get(sym))
        close = None if now is None else now["level"]
        previous = None if now is None else now["previous"]
        live = False if now is None else now["live"]
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
                "day": None if close is None or previous is None else close / previous - 1.0,
                "crossed": (
                    None
                    if df is None or close is None
                    else insights.crossed_ma(df["Close"], close, live, SECTOR_MA_BARS)
                ),
                "ma": ma,
                "pct": None if above_ma is None else close / ma - 1.0,  # type: ignore[operator]
                "above": above_ma,
            }
        )

    def names(state: bool | None) -> list[dict[str, str]]:
        return [{"symbol": t["symbol"], "name": t["name"]} for t in tiles if t["above"] is state]

    above, below = names(True), names(False)
    ranked = sorted(tiles, key=lambda t: (t["day"] is None, -(t["day"] or 0.0)))
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


def since_last_close(table: dict[str, Any], market_health_facts: dict[str, Any]) -> list[str]:
    """Plain sentences for what crossed since the previous close: a home fund or sector
    fund through its 50- or 200-day average, or VIX / VIX3M through 1.0000."""
    groups: dict[tuple[str, int], list[str]] = {}
    for row in table["rows"]:
        for bars, side in row["crossed"].items():
            if side:
                groups.setdefault((side, bars), []).append(row["symbol"])
    for tile in market_health_facts["breadth"]["ranked"]:
        if tile["crossed"]:
            groups.setdefault((tile["crossed"], SECTOR_MA_BARS), []).append(tile["symbol"])
    lines = [
        f"{', '.join(names)} crossed {side} the {bars}-day average"
        for (side, bars), names in sorted(groups.items(), key=lambda g: (-g[0][1], g[0][0]))
    ]
    curve = market_health_facts["vol_curve"]
    if curve and curve["previous"] is not None:
        was, now = curve["previous"] > 1.0, curve["ratio"] > 1.0
        if was != now:
            lines.append(f"VIX / VIX3M crossed {'above' if now else 'below'} 1.0000")
    return lines


def scanned_breadth(now: datetime) -> dict[str, Any] | None:
    """The cached counts across the scanned tickers, or None without a readable file.

    ``stale`` is measured :data:`SCAN_GRACE` behind ``now``, so the page does not warn
    between a session's bars becoming final and that evening's scan finishing.
    """
    try:
        with open(config.UNIVERSE_BREADTH_PATH) as handle:
            got = json.load(handle)
        day = date.fromisoformat(got["date"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Universe breadth unreadable: %s", exc)
        return None
    return got | {"stale": day < daily_ingest.last_complete_session(now - SCAN_GRACE)}


def market_health(
    expected: date, snapshot: dict[str, Any] | None = None, now: datetime | None = None
) -> dict[str, Any]:
    """Volatility, rates, dollar, credit and breadth facts; ``expected`` is the last
    closed session.

    ``snapshot``'s quotes move the levels to live prices, and flags name any symbol without
    a current quote at ``now``. Every ``as_of`` stays the newest close. ``errors`` is the
    newest market bar-quality report.
    """
    quotes = (snapshot or {}).get("quotes", {})
    symbols = (VIX, VIX3M, TNX, DXY, *CREDIT, *config.MARKET_SECTOR_FUNDS)
    frames = {s: closes(s) for s in symbols}
    errors = insights.quality_errors(config.MARKET_CONTEXT_PROFILE["logs_dir"])
    sectors = tuple(config.MARKET_SECTOR_FUNDS)

    def flags(symbols: tuple[str, ...]) -> list[str]:
        return _flags(symbols, frames, expected, errors, snapshot, now)

    return {
        "errors": errors,
        "volatility": volatility(frames[VIX], quotes.get(VIX)),
        "volatility_flags": flags((VIX,)),
        "vol_curve": vol_curve(frames[VIX], frames[VIX3M], quotes),
        "vol_curve_flags": flags((VIX3M,)),
        "rates": rates(frames[TNX], quotes.get(TNX)),
        "rates_flags": flags((TNX,)),
        "dollar": dollar(frames[DXY], quotes.get(DXY)),
        "dollar_flags": flags((DXY,)),
        "credit": credit(frames[CREDIT[0]], frames[CREDIT[1]], quotes),
        "credit_flags": flags(CREDIT),
        "breadth": breadth(frames, quotes),
        "breadth_flags": flags(sectors),
    }
