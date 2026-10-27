"""In-sample scorecard for RESEARCH.md §6-§7 (used by ``scripts/insample_report.py``).

Per signal: count, % up, mean R10, mean - SPY, mean - random (each event minus the
mean of its date-matched draws), date-clustered one-sided p-values vs SPY and vs
random, the larger of the two, Holm across the 3 signals, PASS if < ``ALPHA``.
Inputs are a blind label run; every date is checked in-sample before anything else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from finance_vibe.research import events as ev
from finance_vibe.research import labels as lb
from finance_vibe.research import stats

SIGNALS = {ev.E1: "PRE_BREAKOUT", ev.E2: "TRIGGERED", ev.E3: "CONFIRMED"}
ALPHA = 0.05
DATES = ["Date", "entry_date", "exit_date"]


def _bool(col: pd.Series) -> pd.Series:
    return col.map({True: True, False: False, "True": True, "False": False}).astype("boolean")


def load(labels_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Events, draws and manifest of a label run, with the holdout seal checked.

    Weekly runs (manifest ``mode``) store ``R4W`` / ``SPY_R4W`` / ``REACH4W``; they
    are renamed back to the in-memory names and sealed on the week-end date.
    """
    manifest = json.loads((labels_dir / "manifest.json").read_text(encoding="utf-8"))
    weekly = manifest.get("mode", "daily") == "weekly"
    back = {v: k for k, v in lb.WEEKLY_NAMES.items()} if weekly else {}
    events = pd.read_csv(labels_dir / "events.csv.gz", parse_dates=DATES).rename(columns=back)
    draws = pd.read_csv(labels_dir / "draws.csv.gz", parse_dates=DATES).rename(columns=back)
    for frame in (events, draws):
        frame["REACH10"] = _bool(frame["REACH10"])
        frame["split_suspect"] = _bool(frame["split_suspect"]).fillna(False).astype(bool)
    lb.check_in_sample(events["Date"], weekly=weekly)  # the holdout stays sealed
    lb.check_in_sample(draws["Date"], weekly=weekly)
    return events, draws, manifest


def with_benchmarks(events: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Per event: excess vs SPY, excess vs the mean of its random draws, draws' REACH10 rate."""
    key = ["event_symbol", "Date"]
    rand = draws.groupby(key).agg(
        rand_R10=("R10", "mean"),
        rand_reach=("REACH10", lambda s: s.astype("float64").mean()),
    )
    rand.index = rand.index.set_names(["Symbol", "Date"])
    out = events.merge(rand, left_on=["Symbol", "Date"], right_index=True, how="left")
    out["ex_spy"] = out["R10"] - out["SPY_R10"]
    out["ex_rand"] = out["R10"] - out["rand_R10"]
    return out


def metrics(e: pd.DataFrame, event_id: str) -> dict:
    valid = e[e["R10"].notna()]
    t_spy = stats.clustered_ttest(valid["ex_spy"], valid["Date"])
    t_rand = stats.clustered_ttest(valid["ex_rand"], valid["Date"])
    row = {
        "Signal": SIGNALS[event_id],
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
        "p (larger)": np.nan if np.isnan([t_spy.p, t_rand.p]).any() else max(t_spy.p, t_rand.p),
    }
    if event_id == ev.E1:
        row["Follow-through %"] = 100.0 * float(valid["REACH10"].astype("float64").mean())
        row["Random follow-through %"] = 100.0 * float(valid["rand_reach"].mean())
    return row


def scorecard(e: pd.DataFrame, decide: bool = True) -> pd.DataFrame:
    rows = [metrics(e[e["Event"] == k], k) for k in SIGNALS]
    card = pd.DataFrame(rows).set_index("Signal")
    card["p (Holm)"] = stats.holm(card["p (larger)"].tolist())
    if decide:
        card["Result"] = np.where(card["p (Holm)"] < ALPHA, "PASS", "FAIL")
    return card


def by_year(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in e.groupby(e["Date"].dt.year):
        for k in SIGNALS:
            sub = g[g["Event"] == k]
            if len(sub):
                rows.append({"Year": year, **metrics(sub, k)})
    out = pd.DataFrame(rows).drop(columns=["p (larger)"])
    return out.rename(
        columns={"p vs SPY": "p vs SPY (unadj.)", "p vs random": "p vs random (unadj.)"}
    )
