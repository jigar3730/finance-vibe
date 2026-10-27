"""Forward labels and benchmarks as RESEARCH.md §4-§5 define them (fix #2).

For a signal on bar t (known at t's close):

- entry is the open of t+1, exit the close of t+``horizon`` (the 10th session
  counting the entry session as the 1st): ``R10 = Close[t+10] / Open[t+1] - 1``;
- ``SPY_R10`` is SPY's ``Close[exit date] / Open[entry date] - 1``;
- ``REACH10`` (E1 only) is whether the ticker's Status is BREAKOUT_TRIGGERED or
  BREAKOUT_CONFIRMED on any ``scan_history`` row t+1 ... t+10;
- the random benchmark is ``n`` date-matched universe members per event
  (without replacement, never the event's own ticker), labelled the same way.

Labels are pure shifts of the ticker's own bars, so row t reads only bars
t+1 ... t+horizon. Every signal and draw date must be in-sample
(:func:`check_in_sample`): the holdout stays sealed.

Weekly mode (RESEARCH.md §11, ``weekly=True``) runs on weekly bars with
``horizon = config.RESEARCH_WEEKLY_HORIZON`` (4). The label columns keep their
daily names in memory; files written for weekly runs rename them with
:data:`WEEKLY_NAMES` (``R4W``, ``SPY_R4W``, ``REACH4W``). A weekly bar is dated
Monday but known at the week's close, so the seal is applied to
:func:`known_date` (Monday + 4 days), not to the bar date. Rows with data problems
are flagged (:data:`DQ_COLUMNS`), never dropped. This module computes no
averages, rates or excess returns; analysis is a separate, later step.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from finance_vibe import config
from finance_vibe.research import events as ev
from finance_vibe.scanners import breakout as bs

LABEL_COLUMNS = ["entry_date", "entry_open", "exit_date", "exit_close", "fwd_bars", "R10"]
# no_next_open: no bar after t. short_window: fewer than `horizon` bars after t.
# session_gap: t ... exit spans more SPY sessions than `horizon` (bars missing from
# the ticker's file). split_suspect: a one-bar move (open or close vs previous
# close) beyond SPLIT_JUMP or its inverse on t+1 ... t+horizon; every common split
# ratio (3:2 and up) is one. spy_missing: SPY has no bar on the entry or exit date.
DQ_COLUMNS = ["no_next_open", "short_window", "session_gap", "split_suspect", "spy_missing"]
SPLIT_JUMP = 1.4
WEEKLY_NAMES = {"R10": "R4W", "SPY_R10": "SPY_R4W", "REACH10": "REACH4W"}
_BREAKOUT = [bs.STATUS_TRIGGERED, bs.STATUS_CONFIRMED]


def known_date(dates: Iterable[pd.Timestamp] | pd.Series, *, weekly: bool = False) -> pd.Series:
    """When a bar's signal is known: the bar date (daily) or that week's Friday (weekly).

    Same week-end rule as :func:`config.cut_to_as_of`.
    """
    d = pd.to_datetime(pd.Series(list(dates), dtype="datetime64[ns]"))
    if not weekly:
        return d
    d = d.dt.normalize()
    return d - pd.to_timedelta(d.dt.weekday, unit="D") + pd.Timedelta(days=4)


def check_in_sample(
    dates: Iterable[pd.Timestamp] | pd.Series,
    *,
    weekly: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> None:
    """Raise ``ValueError`` if any signal is known outside the in-sample window.

    Defaults: ``config.RESEARCH_IN_SAMPLE_START`` (``RESEARCH_WEEKLY_IN_SAMPLE_START``
    when ``weekly``) .. ``RESEARCH_IN_SAMPLE_END``, applied to :func:`known_date`.
    A real raise, not ``assert``, so ``python -O`` cannot skip the holdout seal.
    """
    default_start = (
        config.RESEARCH_WEEKLY_IN_SAMPLE_START if weekly else config.RESEARCH_IN_SAMPLE_START
    )
    lo = pd.Timestamp(start or default_start)
    hi = pd.Timestamp(end or config.RESEARCH_IN_SAMPLE_END)
    d = known_date(dates, weekly=weekly)
    if d.isna().any():
        raise ValueError("check_in_sample: missing signal date")
    if len(d) and d.max() > hi:
        raise ValueError(f"holdout is sealed: signal known {d.max().date()} > {hi.date()}")
    if len(d) and d.min() < lo:
        raise ValueError(f"signal known {d.min().date()} is before the in-sample start {lo.date()}")


def _horizon(horizon: int | None, weekly: bool) -> int:
    if horizon is not None:
        return horizon
    return config.RESEARCH_WEEKLY_HORIZON if weekly else config.RESEARCH_HORIZON


def _forward_any(flag: pd.Series, horizon: int) -> pd.Series:
    """True at row i if ``flag`` is True on any row i+1 ... i+horizon."""
    rev = flag.astype("float64").iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]
    return rev.shift(-1).fillna(0.0).astype(bool)


def forward_frame(bars: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """Labels for every bar of one ticker (``Date``/``Open``/``Close`` from ``load_raw``)."""
    h = config.RESEARCH_HORIZON if horizon is None else horizon
    b = bars.reset_index(drop=True)
    out = pd.DataFrame({"Date": b["Date"]})
    out["entry_date"] = b["Date"].shift(-1)
    out["entry_open"] = b["Open"].shift(-1)
    out["exit_date"] = b["Date"].shift(-h)
    out["exit_close"] = b["Close"].shift(-h)
    out["fwd_bars"] = np.minimum(len(b) - 1 - np.arange(len(b)), h)
    out["R10"] = out["exit_close"] / out["entry_open"] - 1.0
    prev = b["Close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        jumps = [np.abs(np.log(b[col] / prev)) > np.log(SPLIT_JUMP) for col in ("Open", "Close")]
    out["no_next_open"] = out["fwd_bars"] == 0
    out["short_window"] = out["fwd_bars"] < h
    out["split_suspect"] = _forward_any(jumps[0] | jumps[1], h)
    return out


def reach_frame(history: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """``REACH10`` for every row of a stacked ``scan_history`` panel (Date, Symbol, Status)."""
    h = config.RESEARCH_HORIZON if horizon is None else horizon
    hist = history.sort_values(["Symbol", "Date"], kind="stable").reset_index(drop=True)
    is_bo = hist["Status"].isin(_BREAKOUT)
    reach = is_bo.groupby(hist["Symbol"], sort=False).transform(lambda s: _forward_any(s, h))
    return pd.DataFrame({"Symbol": hist["Symbol"], "Date": hist["Date"], "REACH10": reach})


def draw_random(
    events: pd.DataFrame,
    members: pd.DataFrame,
    n: int | None = None,
    seed: int | None = None,
    *,
    weekly: bool = False,
) -> pd.DataFrame:
    """``n`` date-matched universe members per event: one row per draw.

    One generator per signal date, seeded with ``[seed, date ordinal]``; that
    date's events take their draws in (Symbol, Event) order. The result does
    not depend on input row order. With fewer than ``n`` candidates every
    candidate is taken (``n_draws`` on the event says how many).
    """
    n = config.RESEARCH_RANDOM_DRAWS if n is None else n
    seed = config.RESEARCH_SEED if seed is None else seed
    check_in_sample(events["Date"], weekly=weekly)
    pool = members.loc[members["in_universe"], ["Date", "Symbol"]]
    by_date = {d: np.sort(g["Symbol"].to_numpy()) for d, g in pool.groupby("Date", sort=False)}
    rows: list[tuple[object, ...]] = []
    for date, day in events.sort_values(["Date", "Symbol", "Event"]).groupby("Date", sort=True):
        rng = np.random.default_rng([seed, pd.Timestamp(date).toordinal()])  # type: ignore[arg-type]
        cands = by_date.get(date, np.array([], dtype=object))
        for sym, event in zip(day["Symbol"], day["Event"], strict=True):
            others = cands[cands != sym]
            pick = rng.choice(len(others), size=min(n, len(others)), replace=False)
            rows.extend((sym, date, event, k, others[i]) for k, i in enumerate(pick))
    return pd.DataFrame(rows, columns=["event_symbol", "Date", "Event", "draw", "Symbol"])


def labels_at(
    keys: pd.DataFrame,
    bars: Mapping[str, pd.DataFrame],
    spy: pd.DataFrame,
    history: pd.DataFrame | None = None,
    horizon: int | None = None,
    *,
    weekly: bool = False,
) -> pd.DataFrame:
    """Labels, SPY return, REACH10 and DQ flags for (Symbol, Date) ``keys``.

    ``keys`` must be in-sample. A key with no bars for its symbol, or no bar on
    its date, gets NaN labels and ``no_next_open``.
    """
    h = _horizon(horizon, weekly)
    check_in_sample(keys["Date"], weekly=weekly)
    k = keys[["Symbol", "Date"]].drop_duplicates()
    frames = [
        forward_frame(bars[s], h).assign(Symbol=s)
        for s in k["Symbol"].unique()
        if s in bars
    ]  # fmt: skip
    if not frames:
        raise ValueError("labels_at: no bars for any key symbol")
    fwd = pd.concat(frames, ignore_index=True)
    out = k.merge(fwd, on=["Symbol", "Date"], how="left", validate="one_to_one")
    for col in ("no_next_open", "short_window"):
        out[col] = out[col].astype("boolean").fillna(True).astype(bool)
    out["split_suspect"] = out["split_suspect"].astype("boolean").fillna(False).astype(bool)

    s = spy.set_index("Date")
    out["SPY_R10"] = (
        out["exit_date"].map(s["Close"]).to_numpy() / out["entry_date"].map(s["Open"]).to_numpy()
        - 1.0
    )
    out["spy_missing"] = out["SPY_R10"].isna() & out["R10"].notna()
    cal = s.index.to_numpy()
    pos_t = np.searchsorted(cal, out["Date"].to_numpy(), side="right")
    pos_exit = np.searchsorted(cal, out["exit_date"].to_numpy(), side="right")
    out["session_gap"] = out["exit_date"].notna().to_numpy() & (pos_exit - pos_t > h)

    if history is not None:
        reach = reach_frame(history, h)
        out = out.merge(reach, on=["Symbol", "Date"], how="left", validate="one_to_one")
    else:
        out["REACH10"] = pd.NA
    out["REACH10"] = out["REACH10"].astype("boolean")  # NA: no scan_history row
    cols = ["Symbol", "Date", *LABEL_COLUMNS, "SPY_R10", "REACH10", *DQ_COLUMNS]
    return out[cols]


def build_labels(
    events: pd.DataFrame,
    members: pd.DataFrame,
    bars: Mapping[str, pd.DataFrame],
    history: pd.DataFrame,
    *,
    benchmark: str | None = None,
    n: int | None = None,
    seed: int | None = None,
    horizon: int | None = None,
    weekly: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(event labels, draw labels) for in-universe, in-sample E1/E2/E3 ``events``.

    Raises ``ValueError`` if an event is dated in the holdout or before the
    in-sample start, or is not a universe member on its date. REACH10 is kept
    for E1 events and their draws only (RESEARCH.md §4).
    """
    n = config.RESEARCH_RANDOM_DRAWS if n is None else n
    benchmark = benchmark or config.RESEARCH_BENCHMARK
    check_in_sample(events["Date"], weekly=weekly)
    if events.duplicated(["Symbol", "Date"]).any():
        raise ValueError("build_labels: duplicate (Symbol, Date) events")
    mem = members.loc[members["in_universe"], ["Date", "Symbol"]].assign(_m=True)
    tagged = events.merge(mem, on=["Date", "Symbol"], how="left")
    if tagged["_m"].isna().any():
        raise ValueError("build_labels: every event must be a universe member on its date")
    if benchmark not in bars:
        raise ValueError(f"build_labels: no bars for benchmark {benchmark}")

    draws = draw_random(events, members, n=n, seed=seed, weekly=weekly)
    keys = pd.concat([events[["Symbol", "Date"]], draws[["Symbol", "Date"]]], ignore_index=True)
    lab = labels_at(keys, bars, bars[benchmark], history, _horizon(horizon, weekly), weekly=weekly)
    version = config.BREAKOUT_RUBRIC_VERSION

    ev_out = events[["Symbol", "Date", "Event"]].merge(lab, on=["Symbol", "Date"], how="left")
    n_draws = draws.groupby(["event_symbol", "Date"]).size().rename("n_draws")
    ev_out = ev_out.merge(n_draws, left_on=["Symbol", "Date"], right_index=True, how="left").fillna(
        {"n_draws": 0}
    )
    ev_out["n_draws"] = ev_out["n_draws"].astype("int64")
    ev_out["draws_short"] = ev_out["n_draws"] < n
    ev_out.loc[ev_out["Event"] != ev.E1, "REACH10"] = pd.NA
    ev_out["Rubric Version"] = version

    dr_out = draws.merge(lab, on=["Symbol", "Date"], how="left")
    dr_out.loc[dr_out["Event"] != ev.E1, "REACH10"] = pd.NA
    dr_out["Rubric Version"] = version
    ev_out = ev_out.sort_values(["Date", "Symbol"], kind="stable").reset_index(drop=True)
    dr_out = dr_out.sort_values(["Date", "event_symbol", "draw"], kind="stable")
    return ev_out, dr_out.reset_index(drop=True)
