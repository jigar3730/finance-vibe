"""Standard descriptive queries over the SQLite copy of the scan outputs.

What the scanners said and how often: state durations, blockers, counts,
overlaps and changes between runs. No forward returns; those stay in research.

Backfill questions read the newest snapshot and rubric only, so a future second
snapshot cannot double-count. A run of consecutive bars touching a ticker's first
or last bar is "open": its true length is unknown, so it is counted but left out
of the length figures.

    python -m finance_vibe.data.analysis [--mode daily|weekly]
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from itertools import pairwise
from pathlib import Path

import pandas as pd

from finance_vibe import config
from finance_vibe.log import setup_logging
from finance_vibe.scanners import breakout as bs
from finance_vibe.scanners import coiled_cobra as cc

COBRA, BREAKOUT = "backfill_cobra", "backfill_breakout"
STATUSES = [
    bs.STATUS_WATCH,
    bs.STATUS_DEV,
    bs.STATUS_PRE,
    bs.STATUS_TRIGGERED,
    bs.STATUS_CONFIRMED,
    bs.STATUS_FAILED,
]
GATE_NAMES = {
    "A": "Gate A: long-term trend",
    "B": "Gate B: ticker market gate",
    "C": "Gate C: coil integrity",
    "D": "Gate D: checks met",
}
BELOW_THRESHOLD = "Score below threshold"
PILLAR_MAX = {
    "vol_contraction_points": 25,
    "structure_points": 20,
    "relative_strength_points": 20,
    "volume_shelf_points": 15,
    "overhead_clearance_points": 10,
    "rvol_trigger_points": 10,
}
_GATES = re.compile(r"Gate Fail \(([A-D/]+)\)")


def connect(path: Path | str = config.DATABASE_PATH) -> sqlite3.Connection:
    """Read-only connection; a missing database raises instead of creating an empty one."""
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)


def _latest(conn: sqlite3.Connection, table: str) -> tuple[str, float]:
    """Newest snapshot in ``table`` and the newest rubric run on it."""
    row = conn.execute(
        f'SELECT snapshot, MAX(rubric_version) FROM "{table}" '
        f'WHERE snapshot = (SELECT MAX(snapshot) FROM "{table}")'
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"{table} is empty")
    return row[0], row[1]


def _where(conn: sqlite3.Connection, table: str, mode: str) -> tuple[str, tuple]:
    snapshot, rubric = _latest(conn, table)
    return "snapshot = ? AND rubric_version = ? AND mode = ?", (snapshot, rubric, mode)


def _sql(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def _bars(conn: sqlite3.Connection, table: str, column: str, mode: str) -> pd.DataFrame:
    where, params = _where(conn, table, mode)
    return _sql(
        conn,
        f'SELECT symbol, date, {column} AS state FROM "{table}" WHERE {where} '
        "ORDER BY symbol, date",
        params,
    )


def runs(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per unbroken stretch of the same state on a ticker (bars sorted)."""
    if bars.empty:
        return pd.DataFrame(columns=["symbol", "state", "start", "end", "bars", "open"])
    new_ticker = bars["symbol"] != bars["symbol"].shift()
    new_state = bars["state"] != bars["state"].shift()
    run_id = (new_ticker | new_state).cumsum()
    out = (
        bars.groupby(run_id, sort=False)
        .agg(
            symbol=("symbol", "first"),
            state=("state", "first"),
            start=("date", "first"),
            end=("date", "last"),
            bars=("date", "size"),
        )
        .reset_index(drop=True)
    )
    first = out.groupby("symbol")["start"].transform("min")
    last = out.groupby("symbol")["end"].transform("max")
    out["open"] = (out["start"] == first) | (out["end"] == last)
    return out


def _length_stats(frame: pd.DataFrame) -> dict[str, float | int | None]:
    done = frame.loc[~frame["open"], "bars"]
    return {
        "Runs": len(done),
        "Open runs": int(frame["open"].sum()),
        "Median bars": float(done.median()) if len(done) else None,
        "90th pct bars": float(done.quantile(0.9)) if len(done) else None,
        "Longest bars": int(done.max()) if len(done) else None,
        "One-bar runs": int((done == 1).sum()),
    }


def state_durations(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """How long each Breakout status lasts, in bars."""
    found = runs(_bars(conn, BREAKOUT, "status", mode))
    rows = [
        {"Status": s, **_length_stats(found[found["state"] == s])}
        for s in STATUSES
        if (found["state"] == s).any()
    ]
    return pd.DataFrame(rows)


def qualifying_runs(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """How long an unbroken stretch of Coiled Cobra passes lasts, in bars."""
    found = runs(_bars(conn, COBRA, "passed", mode))
    return pd.DataFrame([_length_stats(found[found["state"] == 1])])


def blockers(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """Which gate rejects Cobra bars: in total, and as the only thing in the way."""
    where, params = _where(conn, COBRA, mode)
    grades = _sql(
        conn,
        f"SELECT grade, COUNT(*) AS n FROM {COBRA} WHERE {where} AND passed = 0 GROUP BY grade",
        params,
    )
    total = int(grades["n"].sum())
    failed = dict.fromkeys(GATE_NAMES, 0)
    only = dict.fromkeys(GATE_NAMES, 0)
    below = 0
    for grade, n in zip(grades["grade"], grades["n"], strict=True):
        match = _GATES.search(grade)
        if match is None:
            below += int(n)
            continue
        gates = match.group(1).split("/")
        for gate in gates:
            failed[gate] += int(n)
        if len(gates) == 1:
            only[gates[0]] += int(n)
    rows = [
        {"Blocker": name, "Rejected bars": failed[g], "Only blocker": only[g]}
        for g, name in GATE_NAMES.items()
    ]
    rows.append({"Blocker": BELOW_THRESHOLD, "Rejected bars": below, "Only blocker": below})
    out = pd.DataFrame(rows)
    out["Pct of rejected"] = (100 * out["Rejected bars"] / total).round(1) if total else 0.0
    return out


def gate_c_detail(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """Of the bars failing Gate C: which of its two requirements was missed."""
    where, params = _where(conn, COBRA, mode)
    low_vol = f"vol_contraction_points < {cc.GATE_C_VOL_CONTRACTION_MIN}"
    low_struct = f"structure_points < {cc.GATE_C_STRUCTURE_MIN}"
    row = conn.execute(
        f"SELECT SUM({low_vol} AND NOT {low_struct}), SUM({low_struct} AND NOT {low_vol}), "
        f"SUM({low_vol} AND {low_struct}) FROM {COBRA} WHERE {where} "
        f"AND ({low_vol} OR {low_struct})",
        params,
    ).fetchone()
    labels = ["Volatility contraction only", "Structure only", "Both"]
    return pd.DataFrame({"Gate C miss": labels, "Bars": [int(v or 0) for v in row]})


def pillar_points(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """Average points per Cobra pillar on passing and on rejected bars."""
    where, params = _where(conn, COBRA, mode)
    means = ", ".join(f"AVG({column}) AS {column}" for column in PILLAR_MAX)
    frame = _sql(
        conn, f"SELECT passed, {means} FROM {COBRA} WHERE {where} GROUP BY passed", params
    ).set_index("passed")
    rows = []
    for column, maximum in PILLAR_MAX.items():
        rows.append(
            {
                "Pillar": column.removesuffix("_points").replace("_", " ").capitalize(),
                "Max": maximum,
                "Avg when passed": round(float(frame[column].get(1, float("nan"))), 1),
                "Avg when rejected": round(float(frame[column].get(0, float("nan"))), 1),
            }
        )
    return pd.DataFrame(rows)


def per_day(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """Per bar date: Cobra passes and the number of tickers in each Breakout status."""
    where, params = _where(conn, COBRA, mode)
    cobra = _sql(
        conn,
        f"SELECT date, SUM(passed) AS cobra_passed, COUNT(*) AS cobra_scored FROM {COBRA} "
        f"WHERE {where} GROUP BY date",
        params,
    ).set_index("date")
    where, params = _where(conn, BREAKOUT, mode)
    counts = _sql(
        conn,
        f"SELECT date, status, COUNT(*) AS n FROM {BREAKOUT} WHERE {where} GROUP BY date, status",
        params,
    )
    status = (
        counts.pivot_table(index="date", columns="status", values="n", aggfunc="sum")
        .fillna(0)
        .astype(int)
    )
    status = status.reindex(columns=[s for s in STATUSES if s in status.columns])
    return cobra.join(status, how="outer").sort_index().reset_index()


def per_year(daily: pd.DataFrame) -> pd.DataFrame:
    """Median and highest count per date, by year, from :func:`per_day`."""
    frame = daily.assign(Year=daily["date"].str[:4])
    grouped = frame.groupby("Year")
    out = pd.DataFrame({"Bars": grouped.size()})
    out["Cobra bars"] = grouped["cobra_passed"].count()
    out["Cobra passes"] = grouped["cobra_passed"].sum(min_count=1)
    out["Cobra median per bar"] = grouped["cobra_passed"].median()
    out["Cobra most in a bar"] = grouped["cobra_passed"].max()
    for status in [bs.STATUS_PRE, bs.STATUS_TRIGGERED, bs.STATUS_CONFIRMED]:
        if status in frame:
            out[f"{status} median"] = grouped[status].median()
    return out.reset_index()


def overlaps(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """Breakout status on the same ticker and bar: Cobra passes against all Cobra bars."""
    cobra_where, cobra_params = _where(conn, COBRA, mode)
    break_where, break_params = _where(conn, BREAKOUT, mode)
    qualified = re.sub(r"\b(snapshot|rubric_version|mode)\b", r"c.\1", cobra_where)
    frame = _sql(
        conn,
        f"SELECT b.status AS Status, COUNT(*) AS all_bars, SUM(c.passed) AS cobra_passes "
        f"FROM {COBRA} c JOIN (SELECT symbol, date, status FROM {BREAKOUT} "
        f"WHERE {break_where}) b ON b.symbol = c.symbol AND b.date = c.date "
        f"WHERE {qualified} GROUP BY b.status",
        (*break_params, *cobra_params),
    )
    frame["Pct of Cobra passes"] = (
        100 * frame["cobra_passes"] / frame["cobra_passes"].sum()
    ).round(1)
    frame["Pct of all bars"] = (100 * frame["all_bars"] / frame["all_bars"].sum()).round(1)
    order = {s: i for i, s in enumerate(STATUSES)}
    frame = frame.sort_values("Status", key=lambda s: s.map(order)).reset_index(drop=True)
    return frame.rename(columns={"all_bars": "All bars", "cobra_passes": "Cobra passes"})


def current_overlaps(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """Tickers in the newest real Cobra scan, with their Breakout status in that run."""
    return _sql(
        conn,
        "SELECT c.run_date, c.symbol, c.score, c.grade, b.status, b.breakout_readiness "
        "FROM live_cobra c LEFT JOIN live_breakout b "
        "ON b.run_date = c.run_date AND b.mode = c.mode AND b.symbol = c.symbol "
        "WHERE c.mode = ? AND c.run_date = (SELECT MAX(run_date) FROM live_cobra WHERE mode = ?) "
        "ORDER BY c.score DESC, c.symbol",
        (mode, mode),
    )


def _members(conn: sqlite3.Connection, table: str, mode: str) -> dict[str, set[str]]:
    frame = _sql(conn, f'SELECT run_date, symbol FROM "{table}" WHERE mode = ?', (mode,))
    return {str(d): set(g["symbol"]) for d, g in frame.groupby("run_date")}


def changes_between_runs(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """Per real run against the one before: Cobra entries and exits, Breakout status changes."""
    cobra = _members(conn, "live_cobra", mode)
    status = _sql(
        conn, "SELECT run_date, symbol, status FROM live_breakout WHERE mode = ?", (mode,)
    ).pivot_table(index="symbol", columns="run_date", values="status", aggfunc="first")
    dates = sorted(set(cobra) | {str(c) for c in status.columns})
    rows = []
    for before, after in pairwise(dates):
        old, new = cobra.get(before, set()), cobra.get(after, set())
        row = {
            "Run": after,
            "Cobra passes": len(new),
            "Entered": len(new - old),
            "Left": len(old - new),
        }
        if before in status and after in status:
            both = status[[before, after]].dropna()
            row["Breakout compared"] = len(both)
            row["Status changed"] = int((both[before] != both[after]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def status_transitions(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """Breakout status moves between consecutive real runs, most frequent first."""
    frame = _sql(
        conn,
        "SELECT symbol, run_date, status FROM live_breakout WHERE mode = ? "
        "ORDER BY symbol, run_date",
        (mode,),
    )
    dates = sorted(frame["run_date"].unique())
    following = dict(pairwise(dates))
    frame["next_run"] = frame["run_date"].map(following)
    moved = frame.merge(
        frame[["symbol", "run_date", "status"]],
        left_on=["symbol", "next_run"],
        right_on=["symbol", "run_date"],
        suffixes=("", "_next"),
    )
    moved = moved[moved["status"] != moved["status_next"]]
    return (
        moved.groupby(["status", "status_next"])
        .size()
        .reset_index(name="Tickers")
        .rename(columns={"status": "From", "status_next": "To"})
        .sort_values(["Tickers", "From", "To"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def ticker_counts(conn: sqlite3.Connection, mode: str) -> pd.DataFrame:
    """Per ticker: Cobra passing bars and stretches, Breakout confirmed stretches."""
    bars = _bars(conn, COBRA, "passed", mode)
    found = runs(bars)
    passes = found[found["state"] == 1]
    out = pd.DataFrame({"Scored bars": bars.groupby("symbol").size()})
    grouped = passes.groupby("symbol")
    out["Cobra bars passed"] = grouped["bars"].sum()
    out["Cobra stretches"] = grouped.size()
    out["First pass"] = grouped["start"].min()
    out["Last pass"] = grouped["end"].max()
    breakout = runs(_bars(conn, BREAKOUT, "status", mode))
    confirmed = breakout[breakout["state"] == bs.STATUS_CONFIRMED]
    out["Breakout confirmed stretches"] = confirmed.groupby("symbol").size()
    counts = ["Cobra bars passed", "Cobra stretches", "Breakout confirmed stretches"]
    out[counts] = out[counts].fillna(0).astype(int)
    out["Pct of bars passed"] = (100 * out["Cobra bars passed"] / out["Scored bars"]).round(1)
    out = out.reset_index().rename(columns={"symbol": "Symbol"})
    return out.sort_values(["Cobra bars passed", "Symbol"], ascending=[False, True]).reset_index(
        drop=True
    )


def report(conn: sqlite3.Connection, mode: str) -> str:
    """Every standard query as one text report."""
    tickers = ticker_counts(conn, mode)
    never = int((tickers["Cobra bars passed"] == 0).sum())
    sections = [
        ("Breakout state durations (bars)", state_durations(conn, mode)),
        ("Cobra qualifying runs (bars)", qualifying_runs(conn, mode)),
        ("Cobra blockers", blockers(conn, mode)),
        ("Gate C misses", gate_c_detail(conn, mode)),
        ("Cobra pillar points", pillar_points(conn, mode)),
        ("Counts by year", per_year(per_day(conn, mode))),
        ("Breakout status on Cobra bars", overlaps(conn, mode)),
        ("Newest real Cobra scan with Breakout status", current_overlaps(conn, mode)),
        ("Changes between real runs", changes_between_runs(conn, mode)),
        ("Breakout status moves between real runs", status_transitions(conn, mode)),
        (
            f"Tickers with most Cobra passes (top 15; {never} of {len(tickers)} never passed)",
            tickers.head(15),
        ),
    ]
    parts = [f"# {mode}"]
    for title, frame in sections:
        parts += [f"\n## {title}\n", frame.to_markdown(index=False)]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    """CLI: print the report for one mode, or for both."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["daily", "weekly"])
    parser.add_argument("--path", default=config.DATABASE_PATH)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    conn = connect(args.path)
    try:
        for mode in [args.mode] if args.mode else ["daily", "weekly"]:
            print(report(conn, mode) + "\n")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
