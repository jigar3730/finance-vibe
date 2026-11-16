"""RESEARCH.md §13 features: point-in-time price/volume features for the cross-sectional ranker.

One row per (Date, Symbol). Every feature on bar t reads bars t and earlier only;
shifts are in the ticker's own bars and benchmark closes are taken on the ticker's
own bar dates. Support and Resistance are the breakout scanner's values on bar t
(its backfill history, itself point-in-time).

A stock is **eligible** on date t when it is a snapshot universe ticker, passes the
§12 point-in-time check at t (:func:`research.prebreakout.point_in_time_members`)
and every feature is defined. :func:`rank_features` turns each feature into its
percentile rank among the stocks eligible that day; the model only sees those ranks.

This module reads no bar after t and computes no forward return: labels live in
:mod:`research.ranker`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from finance_vibe import config, indicators
from finance_vibe.research import prebreakout as pb

FEATURES = [
    "RS_QQQ_63",
    "RS_QQQ_126",
    "RS_SPY_63",
    "RS_SPY_126",
    "MOM_12_1",
    "REV_1M",
    "HIGH_52W",
    "ATR_PCT",
    "RES_ATR",
    "SUP_ATR",
    "OBV_SLOPE_20",
    "VOL_RATIO",
]
ATR_LENGTH = 14
BENCHMARKS = ("QQQ", "SPY")
PANEL_COLUMNS = ["Date", "Symbol", "in_universe", *FEATURES, "eligible"]


def _benchmark_on(dates: pd.Series, bench: pd.DataFrame) -> np.ndarray:
    """The benchmark's Close on each of ``dates`` (NaN where it has no bar)."""
    close = bench.set_index("Date")["Close"]
    return close.reindex(pd.DatetimeIndex(dates)).to_numpy(dtype="float64")


def ticker_features(
    bars: pd.DataFrame,
    benchmarks: Mapping[str, pd.DataFrame],
    levels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The §13 features for every bar of one ticker.

    ``bars`` comes from ``raw.load_raw`` (Date, Open, High, Low, Close, Volume, sorted).
    ``benchmarks`` maps ``"QQQ"`` and ``"SPY"`` to their bars. ``levels`` has Date,
    Resistance, Support (the breakout history); missing dates give NaN level features.
    """
    b = bars.reset_index(drop=True)
    c, h, v = b["Close"], b["High"], b["Volume"].astype("float64")
    out = pd.DataFrame({"Date": b["Date"]})

    for name in BENCHMARKS:
        bench = pd.Series(_benchmark_on(b["Date"], benchmarks[name]), index=b.index)
        for n in (63, 126):
            out[f"RS_{name}_{n}"] = (c / c.shift(n)) / (bench / bench.shift(n)) - 1.0

    out["MOM_12_1"] = c.shift(21) / c.shift(252) - 1.0
    out["REV_1M"] = c / c.shift(21) - 1.0
    out["HIGH_52W"] = c / h.rolling(252, min_periods=252).max() - 1.0

    atr = indicators.atr(b["High"], b["Low"], c, length=ATR_LENGTH)
    atr = atr if atr is not None else pd.Series(np.nan, index=b.index)
    atr = atr.where(atr > 0)
    out["ATR_PCT"] = atr / c

    if levels is not None and len(levels):
        lv = b[["Date"]].merge(
            levels[["Date", "Resistance", "Support"]], on="Date", how="left", validate="one_to_one"
        )
        out["RES_ATR"] = (lv["Resistance"].to_numpy(dtype="float64") - c) / atr
        out["SUP_ATR"] = (c - lv["Support"].to_numpy(dtype="float64")) / atr
    else:
        out["RES_ATR"] = np.nan
        out["SUP_ATR"] = np.nan

    obv = indicators.obv(c, v)
    obv = obv if obv is not None else pd.Series(np.nan, index=b.index)
    vol_sum = v.rolling(20, min_periods=20).sum()
    out["OBV_SLOPE_20"] = (obv - obv.shift(20)) / vol_sum.where(vol_sum > 0)

    prior = v.shift(1).rolling(20, min_periods=20).mean()
    out["VOL_RATIO"] = v / prior.where(prior > 0)

    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


def build_panel(
    bars: Mapping[str, pd.DataFrame],
    symbols: Iterable[str],
    levels: pd.DataFrame,
    dates: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Features, point-in-time membership and eligibility per (Date, Symbol).

    ``bars`` holds the universe tickers plus QQQ and SPY. ``levels`` is the breakout
    history (Date, Symbol, Resistance, Support). ``dates`` keeps only those dates
    (the training grid); None keeps every bar.
    """
    symbols = sorted(symbols)
    missing = [s for s in [*symbols, *BENCHMARKS] if s not in bars]
    if missing:
        raise ValueError(f"build_panel: no bars for {missing[:5]}")
    benchmarks = {name: bars[name] for name in BENCHMARKS}
    by_symbol = {str(s): g for s, g in levels.groupby("Symbol", sort=False)}
    frames = []
    for symbol in symbols:
        f = ticker_features(bars[symbol], benchmarks, by_symbol.get(symbol))
        frames.append(f.assign(Symbol=symbol))
    panel = pd.concat(frames, ignore_index=True)
    members = pb.point_in_time_members(bars, symbols)[["Date", "Symbol", "in_universe"]]
    panel = panel.merge(members, on=["Date", "Symbol"], how="left", validate="one_to_one")
    panel["in_universe"] = panel["in_universe"].astype("boolean").fillna(False).astype(bool)
    if dates is not None:
        keep = pd.DatetimeIndex(list(dates))
        panel = panel[panel["Date"].isin(keep)]
    panel["eligible"] = panel["in_universe"] & panel[FEATURES].notna().all(axis=1)
    panel = panel.sort_values(["Date", "Symbol"], kind="stable").reset_index(drop=True)
    return panel[PANEL_COLUMNS]


def rank_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Eligible rows only, each feature replaced by its percentile rank within the date.

    Ranks are in (0, 1]; ties share their average rank. Other columns are kept.
    """
    elig = panel[panel["eligible"]].copy()
    elig[FEATURES] = elig.groupby("Date", sort=False)[FEATURES].rank(pct=True, method="average")
    return elig.reset_index(drop=True)


def session_grid(
    calendar: Iterable[pd.Timestamp],
    stride: int,
    start: str | None = None,
    end: str | None = None,
) -> pd.DatetimeIndex:
    """Every ``stride``-th session of ``calendar`` (SPY's dates) from ``start`` through ``end``.

    The grid is anchored on the first session on or after ``start``
    (``RESEARCH_RANKER_IN_SAMPLE_START``), so the 5- and 20-session grids line up.
    ``end`` defaults to ``RESEARCH_IN_SAMPLE_END``.
    """
    if stride < 1:
        raise ValueError("session_grid: stride must be >= 1")
    cal = pd.DatetimeIndex(sorted(pd.DatetimeIndex(list(calendar)).unique()))
    lo = pd.Timestamp(start or config.RESEARCH_RANKER_IN_SAMPLE_START)
    hi = pd.Timestamp(end or config.RESEARCH_IN_SAMPLE_END)
    cal = cal[cal >= lo]
    grid = cal[::stride]
    return grid[grid <= hi]
