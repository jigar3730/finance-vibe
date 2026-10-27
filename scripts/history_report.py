"""One-off research report on breakout state history (not a pipeline stage).

Runs ``breakout.scan_history()`` over every daily raw file and summarizes how
the states behave over time: status mix per year, the PRE_BREAKOUT base rate,
next-day transitions, run lengths, first-bar vs repeat breakouts, and
readiness by status. Descriptive only: **no forward prices or returns** are
computed (see docs/research_review.md; outcome labels are a separate,
pre-registered step).

Raw files are read through ``raw.load_raw`` and never written. Outputs:

    <out-dir>/history_report_<date>.md
    <out-dir>/history_<table>_<date>.csv     one CSV per table in the report

Usage (inside the container, against the working tree)::

    python scripts/history_report.py
    python scripts/history_report.py --raw-dir /app/data/raw/daily \\
        --out-dir /app/data/logs/research
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from finance_vibe import config
from finance_vibe.data import raw
from finance_vibe.log import setup_logging
from finance_vibe.scanners import breakout as bs

logger = logging.getLogger(__name__)

STATUSES = [
    bs.STATUS_PRE,
    bs.STATUS_CONFIRMED,
    bs.STATUS_TRIGGERED,
    bs.STATUS_FAILED,
    bs.STATUS_DEV,
    bs.STATUS_WATCH,
]
BREAKOUTS = [bs.STATUS_CONFIRMED, bs.STATUS_TRIGGERED]
KEEP = ["Date", "Symbol", "Status", "Breakout Readiness", "Distance Resistance ATR",
        "Breakout Triggered", "Resistance", "ATR"]  # fmt: skip
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
# Distance Resistance ATR is negative above resistance: these bins size how far
# above the broken level the first breakout bar closed.
DIST_BINS = [-np.inf, -3.0, -2.0, -1.0, -0.5, -0.25, 0.0]
DIST_LABELS = ["< -3", "-3 to -2", "-2 to -1", "-1 to -0.5", "-0.5 to -0.25", "-0.25 to 0"]


# ---------------------------------------------------------------------------
# Build the ticker-day panel
# ---------------------------------------------------------------------------


def build_panel(raw_dir: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """scan_history for every raw file, stacked; plus rejection counts."""
    frames: list[pd.DataFrame] = []
    rejected: dict[str, int] = {}
    files = raw.raw_files(raw_dir)
    for i, f in enumerate(files, 1):
        try:
            bars = raw.load_raw(f.path)
            hist = bs.scan_history(bars, f.ticker, "daily", "daily")
        except (OSError, ValueError) as exc:
            key = bs._rejection_key(str(exc))
            rejected[key] = rejected.get(key, 0) + 1
            logger.info("skip %s: %s", f.ticker, exc)
            continue
        except Exception:  # per ticker: one bad file must not stop the report
            logger.exception("scan_history failed for %s", f.ticker)
            rejected["execution_error"] = rejected.get("execution_error", 0) + 1
            continue
        # Same-bar open, for the gap question (no later bar is touched).
        opens = bars.set_index("Date")["Open"].astype(float)
        hist = hist[KEEP].assign(Open=hist["Date"].map(opens))
        frames.append(hist)
        if i % 50 == 0:
            logger.info("scanned %s/%s files", i, len(files))
    if not frames:
        raise SystemExit(f"no usable raw files in {raw_dir}")
    panel = pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"], kind="stable")
    panel = panel.reset_index(drop=True)
    panel["Year"] = panel["Date"].dt.year
    return panel, rejected


def add_runs(panel: pd.DataFrame) -> pd.DataFrame:
    """Same-status runs per ticker (consecutive bars of that ticker's history)."""
    new_ticker = panel["Symbol"].ne(panel["Symbol"].shift())
    run_start = new_ticker | panel["Status"].ne(panel["Status"].shift())
    out = panel.assign(run_start=run_start, run_id=run_start.cumsum())
    # Fresh breakout: first bar of a run of closes above resistance (the
    # Breakout Triggered flag switching on), whatever the status label.
    trig = out["Breakout Triggered"].astype(bool)
    out["fresh_breakout"] = trig & (new_ticker | ~trig.shift(fill_value=False))
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def status_by_year(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(panel["Year"], panel["Status"]).reindex(columns=STATUSES, fill_value=0)
    counts.loc["All"] = counts.sum()
    counts["Ticker-days"] = counts.sum(axis=1)
    share = counts[STATUSES].div(counts["Ticker-days"], axis=0) * 100
    return counts, share.round(2)


def pre_breakout_rate(panel: pd.DataFrame) -> pd.DataFrame:
    pre = panel[panel["Status"] == bs.STATUS_PRE]
    by_year = panel.groupby("Year").agg(
        ticker_days=("Status", "size"), tickers_scanned=("Symbol", "nunique")
    )
    by_year["PRE_BREAKOUT rows"] = pre.groupby("Year").size()
    by_year["Distinct tickers"] = pre.groupby("Year")["Symbol"].nunique()
    by_year = by_year.fillna(0).astype(int)
    by_year.loc["All"] = [
        len(panel),
        panel["Symbol"].nunique(),
        len(pre),
        pre["Symbol"].nunique(),
    ]
    by_year["Per 10k ticker-days"] = (
        by_year["PRE_BREAKOUT rows"] / by_year["ticker_days"] * 10_000
    ).round(2)
    return by_year.rename(columns={"ticker_days": "Ticker-days", "tickers_scanned": "Tickers"})


def transitions(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nxt = panel.groupby("Symbol", sort=False)["Status"].shift(-1)
    pairs = panel.assign(Next=nxt).dropna(subset=["Next"])
    counts = pd.crosstab(pairs["Status"], pairs["Next"]).reindex(
        index=STATUSES, columns=STATUSES, fill_value=0
    )
    rows = counts.sum(axis=1).replace(0, np.nan)
    share = (counts.div(rows, axis=0) * 100).round(2)
    share["n"] = counts.sum(axis=1)
    return counts, share


def run_lengths(panel: pd.DataFrame) -> pd.DataFrame:
    """Run length stats. Runs that touch the start or end of a ticker's history
    are censored (true length unknown) and excluded."""
    runs = panel.groupby("run_id").agg(
        Status=("Status", "first"), Symbol=("Symbol", "first"), length=("Status", "size")
    )
    first = runs.groupby("Symbol").head(1).index
    last = runs.groupby("Symbol").tail(1).index
    complete = runs.drop(index=first.union(last))
    stats = complete.groupby("Status")["length"].agg(
        runs="size",
        mean="mean",
        median="median",
        p90=lambda s: s.quantile(0.90),
        max="max",
    )
    stats["single-bar share %"] = (
        complete.assign(one=complete["length"].eq(1)).groupby("Status")["one"].mean() * 100
    )
    stats["censored runs (excluded)"] = runs.groupby("Status").size() - stats["runs"]
    return stats.reindex(STATUSES).round(2)


def breakout_first_bars(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for status in BREAKOUTS:
        sub = panel[panel["Status"] == status]
        firsts = sub[sub["run_start"]]
        rows.append(
            {
                "Status": status,
                "Rows": len(sub),
                "First bar of a status run": len(firsts),
                "Repeat bars": len(sub) - len(firsts),
                "First-bar share %": round(len(firsts) / len(sub) * 100, 2) if len(sub) else None,
                "Fresh breakouts (close crosses resistance)": int(sub["fresh_breakout"].sum()),
            }
        )
    summary = pd.DataFrame(rows).set_index("Status")

    firsts = panel[panel["Status"].isin(BREAKOUTS) & panel["run_start"]].copy()
    atr = firsts["ATR"].replace(0, np.nan)
    firsts["Open gap ATR"] = (firsts["Open"] - firsts["Resistance"]) / atr
    dist = firsts.groupby("Status")["Distance Resistance ATR"].describe(percentiles=QUANTILES)
    dist["opened above resistance %"] = (
        firsts.assign(gap=firsts["Open"] > firsts["Resistance"]).groupby("Status")["gap"].mean()
        * 100
    )
    dist["median open gap ATR"] = firsts.groupby("Status")["Open gap ATR"].median()
    bins = pd.cut(firsts["Distance Resistance ATR"], DIST_BINS, labels=DIST_LABELS, right=True)
    buckets = pd.crosstab(firsts["Status"], bins).reindex(columns=DIST_LABELS, fill_value=0)
    return summary, dist.reindex(BREAKOUTS).round(3), buckets.reindex(BREAKOUTS)


def readiness_by_status(panel: pd.DataFrame) -> pd.DataFrame:
    stats = panel.groupby("Status")["Breakout Readiness"].describe(percentiles=QUANTILES)
    return stats.reindex(STATUSES).round(1)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt="g")


def write_report(
    panel: pd.DataFrame, rejected: dict[str, int], out_dir: Path, stamp: str, raw_dir: str
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, share = status_by_year(panel)
    pre = pre_breakout_rate(panel)
    trans_counts, trans_share = transitions(panel)
    runs = run_lengths(panel)
    bo_summary, bo_dist, bo_buckets = breakout_first_bars(panel)
    ready = readiness_by_status(panel)

    tables = {
        "status_counts_by_year": counts,
        "status_share_by_year": share,
        "pre_breakout_rate": pre,
        "transition_counts": trans_counts,
        "transition_share": trans_share,
        "run_lengths": runs,
        "breakout_first_bars": bo_summary,
        "breakout_first_bar_distance": bo_dist,
        "breakout_first_bar_distance_buckets": bo_buckets,
        "readiness_by_status": ready,
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"history_{name}_{stamp}.csv")

    first, last = panel["Date"].min().date(), panel["Date"].max().date()
    rejected_txt = ", ".join(f"{k}: {v}" for k, v in sorted(rejected.items())) or "none"
    md = f"""# Breakout state history report ({stamp})

Descriptive statistics of breakout scanner states over time, from
`scan_history()` on the daily raw data. **No forward prices or returns are computed
here.** Nothing in this report says whether any status is a good entry.

- Code: `{_git_rev()}`. Raw: `{raw_dir}` ({panel["Symbol"].nunique()} tickers scanned,
  skipped: {rejected_txt}).
- Ticker-days: {len(panel):,}, from {first} to {last}. A ticker's rows start at its 80th
  bar (`MIN_PRIMARY_BARS`). Each row uses only bars on or before its date; weekly and
  monthly flags use the bar that was forming at that date.

## Caveats (read first)

- **Survivorship bias.** The universe is *today's* list (the raw files present
  now: a fixed manifest plus Yahoo `most_actives`, picked by recent volume). Names that
  delisted, or were not active enough to be picked today, are missing from every past
  year. Past counts are therefore biased toward stocks that survived and are active now.
  See docs/research_review.md item 1.
- **Not point-in-time prices.** Raw data is re-downloaded with `auto_adjust=True`, so
  past bars reflect today's split and dividend adjustments (review item 2).
- Statuses are per-bar states, not events (review item 6). The breakout section
  separates the first bar of each run from repeat bars.

## 1. Status mix per year

Counts of ticker-days:

{_md(counts)}

Share of ticker-days in that year (%):

{_md(share)}

## 2. PRE_BREAKOUT base rate

{_md(pre)}

## 3. Next-day transitions

Row = status on day t, column = status on the ticker's next bar; % of row (row
total `n`). The last bar of each ticker has no next bar and is excluded.

{_md(trans_share)}

## 4. Run lengths (consecutive bars in the same status)

Runs that touch the first or last bar of a ticker's history are censored and
excluded from the stats (counted in the last column).

{_md(runs)}

## 5. Breakouts: first bar vs repeats

A *status run* is consecutive bars with the same status, so TRIGGERED followed by
CONFIRMED starts a new CONFIRMED run. A *fresh breakout* is the bar where the close
first moves above resistance (`Breakout Triggered` switches on), whatever the label.

{_md(bo_summary)}

### Distance Resistance ATR on the first bar of each run

Negative = closed above resistance; −1 means one ATR above the broken level, which
is roughly what an entry at the close pays over the level. "Opened above resistance"
uses the same bar's open: the bar gapped over the level, so it could not be bought
at the level that day. The open gap is (Open − Resistance) / ATR.

{_md(bo_dist)}

First bars per distance bucket (ATR):

{_md(bo_buckets)}

## 6. Breakout Readiness by status

{_md(ready)}

---
CSV for each table: `history_<table>_{stamp}.csv` in this folder.
"""
    path = out_dir / f"history_report_{stamp}.md"
    path.write_text(md, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=config.TIMEFRAME_PROFILES["daily"]["raw_dir"])
    parser.add_argument("--out-dir", default=str(Path(config.BASE_DIR) / "logs" / "research"))
    parser.add_argument("--stamp", default=date.today().isoformat())
    args = parser.parse_args(argv)

    started = time.perf_counter()
    panel, rejected = build_panel(args.raw_dir)
    panel = add_runs(panel)
    path = write_report(panel, rejected, Path(args.out_dir), args.stamp, args.raw_dir)
    logger.info(
        "wrote %s (%s tickers, %s ticker-days) in %.1fs",
        path,
        panel["Symbol"].nunique(),
        len(panel),
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
