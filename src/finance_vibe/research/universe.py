"""Point-in-time research universe (docs/research_review.md fix #1).

For each date t, a ticker is in the universe when, using only bars on or
before t:

1. it is not in ``config.RESEARCH_EXCLUDED_SYMBOLS`` (funds / ETPs);
2. it has at least ``config.RESEARCH_MIN_HISTORY_BARS`` sessions of listing
   history (see :func:`prior_sessions`);
3. its close at t is at least ``config.RESEARCH_MIN_CLOSE``;
4. it ranks in the top ``config.RESEARCH_UNIVERSE_TOP_N`` of the tickers that
   pass 1-3 by trailing ``config.RESEARCH_DV_WINDOW``-bar median dollar volume
   (Close x Volume, bar t included). Ties go to the alphabetically first symbol.

Every input to row t comes from that ticker's bars up to t, and the ranking
only compares rows of the same date, so membership at t never changes when
later data is added.

The candidate pool is still the set of raw files on disk (today's ticker
list), so survivorship bias is reduced, not removed (RESEARCH.md §8.4).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from finance_vibe import config
from finance_vibe.data import raw

logger = logging.getLogger(__name__)

# Weekly raw bars before the daily window count as this many sessions each.
# Overcounts holidays and partial first weeks (at most ~13 sessions a year).
SESSIONS_PER_WEEK = 5

MEMBERSHIP_COLUMNS = [
    "Date",
    "Symbol",
    "in_universe",
    "rank",
    "bars",
    "dv_median20",
    "close",
    "reason",
]
# ``reason`` values: the first rule a row fails, or "member".
REASONS = ["excluded", "history", "price", "dv_window", "rank", "member"]


def prior_sessions(
    daily_first: Mapping[str, pd.Timestamp], weekly_dates: Mapping[str, pd.Series]
) -> dict[str, int]:
    """Sessions of listing history each ticker had before its first daily bar.

    The daily raw files cover a fixed window (5y), so their bar count says
    when the download started, not when the stock listed. Weekly raw bars
    (Monday-dated, longer window) fill in the gap: every weekly bar whose week
    (Friday) ended before the first daily bar counts ``SESSIONS_PER_WEEK``.
    A mid-week IPO's first weekly bar ends on or after its first daily bar, so
    it adds nothing. Tickers without weekly data get 0 (the daily count only).
    """
    out: dict[str, int] = {}
    for symbol, first in daily_first.items():
        dates = weekly_dates.get(symbol)
        if dates is None or dates.empty:
            out[symbol] = 0
            continue
        dates = pd.to_datetime(dates).dt.normalize()
        week_end = dates - pd.to_timedelta(dates.dt.weekday, unit="D") + pd.Timedelta(days=4)
        out[symbol] = SESSIONS_PER_WEEK * int((week_end < pd.Timestamp(first).normalize()).sum())
    return out


def build_membership(
    panel: pd.DataFrame,
    prior: Mapping[str, int] | None = None,
    *,
    top_n: int | None = None,
    window: int | None = None,
    min_bars: int | None = None,
    min_close: float | None = None,
    excluded: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Membership table for a long ``[Date, Symbol, Close, Volume]`` panel.

    One row per (Date, Symbol) in the panel, sorted by Date then Symbol, with
    :data:`MEMBERSHIP_COLUMNS`. ``rank`` (nullable int) is the dollar-volume
    rank among the rows that pass the exclusion, history and price rules on
    that date; ``prior`` is :func:`prior_sessions`. Thresholds default to
    the ``config.RESEARCH_*`` values.
    """
    top_n = config.RESEARCH_UNIVERSE_TOP_N if top_n is None else top_n
    window = config.RESEARCH_DV_WINDOW if window is None else window
    min_bars = config.RESEARCH_MIN_HISTORY_BARS if min_bars is None else min_bars
    min_close = config.RESEARCH_MIN_CLOSE if min_close is None else min_close
    excluded = config.RESEARCH_EXCLUDED_SYMBOLS if excluded is None else excluded

    missing = [c for c in ("Date", "Symbol", "Close", "Volume") if c not in panel.columns]
    if missing:
        raise ValueError(f"build_membership: missing columns {missing}")
    df = panel[["Date", "Symbol", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if df.duplicated(["Symbol", "Date"]).any():
        raise ValueError("build_membership: duplicate (Symbol, Date) rows")
    df = df.sort_values(["Symbol", "Date"], kind="stable").reset_index(drop=True)

    close = df["Close"].astype("float64")
    dollar_volume = close * df["Volume"].astype("float64")
    by_symbol = dollar_volume.groupby(df["Symbol"], sort=False)
    median = by_symbol.rolling(window, min_periods=window).median()
    df["dv_median20"] = median.reset_index(level=0, drop=True).sort_index()
    df["bars"] = df.groupby("Symbol", sort=False).cumcount() + 1
    df["bars"] += df["Symbol"].map(dict(prior or {})).fillna(0).astype("int64")
    df["close"] = close

    with np.errstate(invalid="ignore"):
        reason = np.select(
            [
                df["Symbol"].isin(excluded).to_numpy(),
                (df["bars"] < min_bars).to_numpy(),
                ~(close >= min_close).to_numpy(),
                df["dv_median20"].isna().to_numpy(),
            ],
            ["excluded", "history", "price", "dv_window"],
            "",
        )
    eligible = reason == ""

    ranked = df.loc[eligible, ["Date", "Symbol", "dv_median20"]].sort_values(
        ["Date", "dv_median20", "Symbol"], ascending=[True, False, True], kind="stable"
    )
    df["rank"] = (ranked.groupby("Date", sort=False).cumcount() + 1).reindex(df.index)
    df["rank"] = df["rank"].astype("Int64")
    df["in_universe"] = eligible & (df["rank"] <= top_n).fillna(False).to_numpy(dtype=bool)
    df["reason"] = np.where(eligible, np.where(df["in_universe"], "member", "rank"), reason)

    out = df.sort_values(["Date", "Symbol"], kind="stable").reset_index(drop=True)
    return out[MEMBERSHIP_COLUMNS]


# ---------------------------------------------------------------------------
# Loading from raw files (read-only, through raw.load_raw)
# ---------------------------------------------------------------------------


def load_daily_bars(raw_dir: str) -> dict[str, pd.DataFrame]:
    """Every daily raw file as validated bars, keyed by ticker."""
    bars: dict[str, pd.DataFrame] = {}
    for f in raw.raw_files(raw_dir):
        try:
            bars[f.ticker] = raw.load_raw(f.path)
        except (OSError, ValueError) as exc:
            logger.warning("skip %s: %s", f.path, exc)
    return bars


def panel_from_bars(bars: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [b[["Date", "Close", "Volume"]].assign(Symbol=sym) for sym, b in bars.items()]
    return pd.concat(frames, ignore_index=True)[["Date", "Symbol", "Close", "Volume"]]


def load_weekly_dates(weekly_dir: str, symbols: set[str]) -> dict[str, pd.Series]:
    """Weekly bar dates for ``symbols`` (only Date is used: listing age)."""
    out: dict[str, pd.Series] = {}
    try:
        files = raw.raw_files(weekly_dir)
    except FileNotFoundError:
        logger.warning("no weekly raw directory at %s: daily bar counts only", weekly_dir)
        return out
    for f in files:
        if f.ticker not in symbols:
            continue
        try:
            out[f.ticker] = raw.load_raw(f.path)["Date"]
        except (OSError, ValueError) as exc:
            logger.warning("skip weekly %s: %s", f.path, exc)
    return out
