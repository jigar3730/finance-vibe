"""Pre-breakout addendum (RESEARCH.md §12): events, point-in-time check, labels.

| ID | Event | Definition |
| C1 | Cobra first pass | Cobra ``Passed`` on bar t and not on bar t-1 |
| C2 | Cobra on PRE_BREAKOUT | On t, ``Passed`` and Status == PRE_BREAKOUT; on t-1, not both |

Events come from the backfill history of one frozen snapshot: Coiled Cobra and
breakout rows joined on (Symbol, Date). "Bar t-1" is the ticker's previous joined
row, and a ticker's first joined row never counts.

An event counts only if its ticker passes the point-in-time check at t
(:func:`point_in_time_members`): Close >= ``BACKFILL_MIN_CLOSE`` and the
``BACKFILL_DV_WINDOW``-bar median Close x Volume (bar t included) >=
``BACKFILL_MIN_DOLLAR_VOLUME``, among the snapshot's universe tickers.

Labels are those of :mod:`research.labels` (R10, SPY_R10, REACH10, DQ flags) plus
``LEVEL_FIRST`` (:func:`level_first`). Every signal and draw must be in-sample:
the holdout stays sealed. This module computes no averages, rates or excess
returns; analysis is a separate, later step.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from finance_vibe import config
from finance_vibe.research import labels as lb
from finance_vibe.scanners import breakout as bs

C1, C2 = "C1", "C2"
EVENTS = [C1, C2]
LEVEL_FIRST = "LEVEL_FIRST"
RESISTANCE, SUPPORT, NEITHER = "resistance", "support", "neither"
MEMBER_COLUMNS = ["Date", "Symbol", "in_universe", "close", "dv_median"]


def events(cobra: pd.DataFrame, breakout: pd.DataFrame) -> pd.DataFrame:
    """C1 and C2 rows from Cobra (Date, Symbol, Passed) and breakout (Date, Symbol, Status).

    One row per (Symbol, Date, Event); a bar can be both C1 and C2. Sorted by
    Date, Symbol, Event.
    """
    panel = cobra[["Date", "Symbol", "Passed"]].merge(
        breakout[["Date", "Symbol", "Status"]],
        on=["Date", "Symbol"],
        how="inner",
        validate="one_to_one",
    )
    panel = panel.sort_values(["Symbol", "Date"], kind="stable").reset_index(drop=True)
    passed = panel["Passed"].astype(bool)
    flags = {C1: passed, C2: passed & panel["Status"].eq(bs.STATUS_PRE)}
    frames = []
    for name, flag in flags.items():
        prev = flag.groupby(panel["Symbol"], sort=False).shift(1)
        first = prev.notna() & flag & ~prev.astype("boolean").fillna(True).astype(bool)
        frames.append(panel.loc[first, ["Symbol", "Date"]].assign(Event=name))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Date", "Symbol", "Event"], kind="stable").reset_index(drop=True)


def point_in_time_members(bars: Mapping[str, pd.DataFrame], symbols: Iterable[str]) -> pd.DataFrame:
    """Per (Date, Symbol) of the given universe tickers: the §12 check, bars up to t only."""
    window = config.BACKFILL_DV_WINDOW
    frames = []
    for symbol in sorted(symbols):
        b = bars[symbol]
        dv = (b["Close"] * b["Volume"]).rolling(window, min_periods=window).median()
        ok = (b["Close"] >= config.BACKFILL_MIN_CLOSE) & (dv >= config.BACKFILL_MIN_DOLLAR_VOLUME)
        frames.append(
            pd.DataFrame(
                {
                    "Date": b["Date"],
                    "Symbol": symbol,
                    "in_universe": ok.to_numpy(),
                    "close": b["Close"].to_numpy(),
                    "dv_median": dv.to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[MEMBER_COLUMNS]


def level_first(
    bars: Mapping[str, pd.DataFrame], levels: pd.DataFrame, horizon: int | None = None
) -> pd.DataFrame:
    """Which of the bar-t breakout levels price reaches first within t+1 ... t+horizon.

    ``levels`` has Symbol, Date, Resistance, Support (the breakout scanner's values on
    bar t). ``resistance``: a High >= Resistance comes first. ``support``: a Low <=
    Support comes first, or both on the same bar. ``neither``: no touch in the
    window. NA when a level is missing or the ticker has no bar on that date.
    """
    h = config.RESEARCH_HORIZON if horizon is None else horizon
    never = h + 1
    frames = []
    for symbol, group in levels.groupby("Symbol", sort=False):
        out = pd.Series(pd.NA, index=group.index, dtype="string")
        b = bars.get(str(symbol))
        if b is not None:
            dates = b["Date"].to_numpy()
            when = group["Date"].to_numpy(dtype=dates.dtype)
            pos = np.searchsorted(dates, when)
            found = (pos < len(dates)) & (dates[np.minimum(pos, len(dates) - 1)] == when)
            res = group["Resistance"].to_numpy(dtype="float64")
            sup = group["Support"].to_numpy(dtype="float64")
            high, low = b["High"].to_numpy(), b["Low"].to_numpy()
            first_r = np.full(len(group), never)
            first_s = np.full(len(group), never)
            for k in range(1, h + 1):
                j = pos + k
                inside = found & (j < len(dates))
                jj = np.minimum(j, len(dates) - 1)
                with np.errstate(invalid="ignore"):
                    first_r[(first_r == never) & inside & (high[jj] >= res)] = k
                    first_s[(first_s == never) & inside & (low[jj] <= sup)] = k
            result = np.where(
                (first_s < never) & (first_s <= first_r),
                SUPPORT,
                np.where(first_r < never, RESISTANCE, NEITHER),
            )
            usable = found & ~np.isnan(res) & ~np.isnan(sup)
            out[usable] = result[usable]
        frames.append(pd.DataFrame({"Symbol": symbol, "Date": group["Date"], LEVEL_FIRST: out}))
    if not frames:
        return pd.DataFrame(columns=["Symbol", "Date", LEVEL_FIRST])
    return pd.concat(frames, ignore_index=True)


def build_labels(
    signal_events: pd.DataFrame,
    members: pd.DataFrame,
    bars: Mapping[str, pd.DataFrame],
    breakout: pd.DataFrame,
    *,
    benchmark: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    horizon: int | None = None,
    start: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(event labels, draw labels) for in-sample, point-in-time C1/C2 ``signal_events``.

    ``breakout`` is the breakout history (Date, Symbol, Status, Resistance, Support):
    REACH10 and LEVEL_FIRST are read from it for events and draws alike. Raises
    ``ValueError`` for an event outside the in-sample window or not a member on its date.
    """
    n = config.RESEARCH_RANDOM_DRAWS if n is None else n
    benchmark = benchmark or config.RESEARCH_BENCHMARK
    start = start or config.RESEARCH_PREBREAKOUT_IN_SAMPLE_START
    h = config.RESEARCH_HORIZON if horizon is None else horizon
    lb.check_in_sample(signal_events["Date"], start=start)
    if signal_events.duplicated(["Symbol", "Date", "Event"]).any():
        raise ValueError("build_labels: duplicate (Symbol, Date, Event) events")
    mem = members.loc[members["in_universe"], ["Date", "Symbol"]].assign(_m=True)
    tagged = signal_events.merge(mem, on=["Date", "Symbol"], how="left")
    if tagged["_m"].isna().any():
        raise ValueError("build_labels: every event must pass the point-in-time check on its date")
    if benchmark not in bars:
        raise ValueError(f"build_labels: no bars for benchmark {benchmark}")

    draws = lb.draw_random(signal_events, members, n=n, seed=seed, start=start)
    keys = pd.concat(
        [signal_events[["Symbol", "Date"]], draws[["Symbol", "Date"]]], ignore_index=True
    ).drop_duplicates()
    status = breakout[["Date", "Symbol", "Status"]]
    lab = lb.labels_at(keys, bars, bars[benchmark], status, h, start=start)
    levels = keys.merge(
        breakout[["Symbol", "Date", "Resistance", "Support"]], on=["Symbol", "Date"], how="left"
    )
    lab = lab.merge(level_first(bars, levels, h), on=["Symbol", "Date"], how="left")
    versions = {"Rubric Version": config.RUBRIC_VERSION}
    versions["Breakout Rubric Version"] = config.BREAKOUT_RUBRIC_VERSION

    ev_out = signal_events[["Symbol", "Date", "Event"]].merge(
        lab, on=["Symbol", "Date"], how="left"
    )
    n_draws = draws.groupby(["event_symbol", "Date", "Event"]).size().rename("n_draws")
    ev_out = ev_out.merge(
        n_draws, left_on=["Symbol", "Date", "Event"], right_index=True, how="left"
    ).fillna({"n_draws": 0})
    ev_out["n_draws"] = ev_out["n_draws"].astype("int64")
    ev_out["draws_short"] = ev_out["n_draws"] < n
    dr_out = draws.merge(lab, on=["Symbol", "Date"], how="left")
    ev_out = ev_out.assign(**versions).sort_values(["Date", "Symbol", "Event"], kind="stable")
    dr_out = dr_out.assign(**versions).sort_values(
        ["Date", "event_symbol", "Event", "draw"], kind="stable"
    )
    return ev_out.reset_index(drop=True), dr_out.reset_index(drop=True)
