"""In-sample analysis for RESEARCH.md §12 (used by ``scripts/prebreakout_insample_report.py``).

The pass criterion is §7 unchanged: per signal, the date-clustered one-sided t-test
of the excess vs SPY and vs the mean of its own random draws, the larger p, Holm
across C1 and C2, PASS if below ``ALPHA``. A bar can be both C1 and C2, so draws
are matched to their event on (Symbol, Date, Event), never on (Symbol, Date) alone.

Reported, never a pass criterion: REACH10 and LEVEL_FIRST shares against the
draws', results by signal year, the scorecard without ``split_suspect`` rows, and
PRE_BREAKOUT alone on the same universe as a reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finance_vibe import config
from finance_vibe.research import events as ev
from finance_vibe.research import labels as lb
from finance_vibe.research import prebreakout as pb
from finance_vibe.research import stats
from finance_vibe.research.insample import ALPHA, DATES, _bool

SIGNALS = {pb.C1: "Cobra first pass", pb.C2: "Cobra on PRE_BREAKOUT"}
REFERENCE_EVENT = "REF"
REFERENCE = "PRE_BREAKOUT alone (reference)"
KEY = ["Symbol", "Date", "Event"]
LEVELS = [pb.RESISTANCE, pb.SUPPORT, pb.NEITHER]


def _start() -> str:
    return config.RESEARCH_PREBREAKOUT_IN_SAMPLE_START


def load(labels_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Events, draws and manifest of a §12 label run, with the holdout seal checked."""
    manifest = json.loads((labels_dir / "manifest.json").read_text(encoding="utf-8"))
    events = pd.read_csv(labels_dir / "events.csv.gz", parse_dates=DATES)
    draws = pd.read_csv(labels_dir / "draws.csv.gz", parse_dates=DATES)
    for frame in (events, draws):
        _clean(frame)
    lb.check_in_sample(events["Date"], start=_start())  # the holdout stays sealed
    lb.check_in_sample(draws["Date"], start=_start())
    return events, draws, manifest


def _clean(frame: pd.DataFrame) -> None:
    frame["REACH10"] = _bool(frame["REACH10"])
    frame["split_suspect"] = _bool(frame["split_suspect"]).fillna(False).astype(bool)
    frame[pb.LEVEL_FIRST] = frame[pb.LEVEL_FIRST].astype("string")


def _level_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """1.0 / 0.0 per level, NaN where LEVEL_FIRST is missing."""
    known = frame[pb.LEVEL_FIRST].notna().to_numpy()
    out = pd.DataFrame(index=frame.index)
    for level in LEVELS:
        hit = (frame[pb.LEVEL_FIRST] == level).fillna(False).astype("float64")
        out[level] = hit.where(known, np.nan)
    return out


def with_benchmarks(events: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Per event: excess vs SPY and vs the mean of its own draws, and its draws' rates."""
    d = pd.concat([draws, _level_shares(draws).add_prefix("lvl_")], axis=1)
    d["reach"] = d["REACH10"].astype("float64")
    rand = d.groupby(["event_symbol", "Date", "Event"]).agg(
        rand_R10=("R10", "mean"),
        rand_reach=("reach", "mean"),
        **{f"rand_{lv}": (f"lvl_{lv}", "mean") for lv in LEVELS},
    )
    rand.index = rand.index.set_names(KEY)
    out = events.merge(rand, left_on=KEY, right_index=True, how="left", validate="one_to_one")
    out = pd.concat([out, _level_shares(out)], axis=1)
    out["ex_spy"] = out["R10"] - out["SPY_R10"]
    out["ex_rand"] = out["R10"] - out["rand_R10"]
    return out


def metrics(e: pd.DataFrame, name: str) -> dict[str, Any]:
    """§6 scorecard row for one signal's events (rows with an R10 only)."""
    valid = e[e["R10"].notna()]
    t_spy = stats.clustered_ttest(valid["ex_spy"], valid["Date"])
    t_rand = stats.clustered_ttest(valid["ex_rand"], valid["Date"])
    p = [t_spy.p, t_rand.p]
    return {
        "Signal": name,
        "Count": len(valid),
        "Dates": valid["Date"].nunique(),
        "% up": 100.0 * float((valid["R10"] > 0).mean()) if len(valid) else np.nan,
        "Mean R10 %": 100.0 * float(valid["R10"].mean()),
        "Mean − SPY %": 100.0 * t_spy.mean,
        "Mean − random %": 100.0 * t_rand.mean,
        "t vs SPY": t_spy.t,
        "t vs random": t_rand.t,
        "p vs SPY": t_spy.p,
        "p vs random": t_rand.p,
        "p (larger)": np.nan if np.isnan(p).any() else max(p),
    }


def scorecard(e: pd.DataFrame, decide: bool = True) -> pd.DataFrame:
    """C1 and C2: the pass criterion (Holm across the two)."""
    rows = [metrics(e[e["Event"] == k], name) for k, name in SIGNALS.items()]
    card = pd.DataFrame(rows).set_index("Signal")
    card["p (Holm)"] = stats.holm(card["p (larger)"].tolist())
    if decide:
        card["Result"] = np.where(card["p (Holm)"] < ALPHA, "PASS", "FAIL")
    return card


def _pct(series: pd.Series) -> float:
    return 100.0 * float(series.astype("float64").mean())


def reported_row(e: pd.DataFrame, name: str) -> dict[str, Any]:
    """REACH10 and LEVEL_FIRST shares for one signal against its draws (not a criterion)."""
    valid = e[e["R10"].notna()]
    row: dict[str, Any] = {
        "Signal": name,
        "Count": len(valid),
        "Reach breakout %": _pct(valid["REACH10"]),
        "Random reach %": _pct(valid["rand_reach"]),
    }
    for level in LEVELS:
        row[f"{level.capitalize()} first %"] = _pct(valid[level])
        row[f"Random {level} first %"] = _pct(valid[f"rand_{level}"])
    return row


def reported(e: pd.DataFrame) -> pd.DataFrame:
    rows = [reported_row(e[e["Event"] == k], name) for k, name in SIGNALS.items()]
    return pd.DataFrame(rows).set_index("Signal")


def by_year(e: pd.DataFrame) -> pd.DataFrame:
    """The scorecard metrics by signal year, unadjusted (descriptive only)."""
    rows = []
    for year, g in e.groupby(e["Date"].dt.year):
        for k, name in SIGNALS.items():
            sub = g[g["Event"] == k]
            if len(sub):
                rows.append({"Year": year, **metrics(sub, name)})
    out = pd.DataFrame(rows).drop(columns=["p (larger)"])
    return out.rename(
        columns={"p vs SPY": "p vs SPY (unadj.)", "p vs random": "p vs random (unadj.)"}
    )


def reference_events(
    breakout: pd.DataFrame, members: pd.DataFrame, start: str | None = None
) -> pd.DataFrame:
    """PRE_BREAKOUT first bars (§3 E1) in the §12 window that pass the point-in-time check."""
    first = ev.first_bar_events(breakout[["Date", "Symbol", "Status"]])
    first = first[first["Event"] == ev.E1]
    lo = pd.Timestamp(start or _start())
    hi = pd.Timestamp(config.RESEARCH_IN_SAMPLE_END)
    first = first[(first["Date"] >= lo) & (first["Date"] <= hi)]
    ok = members.loc[members["in_universe"], ["Date", "Symbol"]]
    kept = first.merge(ok, on=["Date", "Symbol"], how="inner")[["Symbol", "Date"]]
    out = kept.assign(Event=REFERENCE_EVENT)
    return out.sort_values(["Date", "Symbol"], kind="stable").reset_index(drop=True)


def reference(
    breakout: pd.DataFrame, members: pd.DataFrame, bars: dict[str, pd.DataFrame]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """PRE_BREAKOUT alone, labelled with the §12 code: (scorecard row, reported row)."""
    events = reference_events(breakout, members)
    ev_out, dr_out = pb.build_labels(events, members, bars, breakout)
    for frame in (ev_out, dr_out):
        _clean(frame)
    e = with_benchmarks(ev_out, dr_out)
    return metrics(e, REFERENCE), reported_row(e, REFERENCE)
