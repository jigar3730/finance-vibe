"""One-off research step: point-in-time universe membership + event survival.

Builds the research universe (``finance_vibe.research.universe``) over every
daily raw file, extracts RESEARCH.md §3 events (E1/E2/E3) from
``scan_history()``, and reports how many events survive the universe filter.
No forward prices or returns are computed. Raw files are read through
``raw.load_raw`` and never written. Outputs under ``--out-dir``:

    universe_membership_<date>.csv.gz   Date, Symbol, in_universe, rank + audit columns
    universe_events_<date>.csv          E1/E2/E3 rows with their membership
    universe_event_survival_<date>.csv
    universe_report_<date>.md

The container has no git, so pass the code version from the host with
``--commit`` or ``GIT_COMMIT`` (falls back to ``git rev-parse``, then
``unknown``). Usage (inside the container, against the working tree)::

    python scripts/universe_report.py --raw-dir /app/data/raw/daily \\
        --weekly-dir /app/data/raw/weekly --out-dir /app/data/research \\
        --commit "$(git rev-parse --short HEAD)"
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from datetime import date
from pathlib import Path

import pandas as pd

from finance_vibe import config
from finance_vibe.log import setup_logging
from finance_vibe.research import events as ev
from finance_vibe.research import universe as uv
from finance_vibe.scanners import breakout as bs

logger = logging.getLogger(__name__)

EVENTS = [ev.E1, ev.E2, ev.E3]
REMOVAL_REASONS = ["excluded", "history", "price", "dv_window", "rank", "no_row"]


def _git_rev() -> str:
    """Short HEAD hash, or ``unknown`` where git is missing (the container)."""
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


def history_panel(bars: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, str]]:
    frames, skipped = [], {}
    for symbol, b in bars.items():
        try:
            hist = bs.scan_history(b, symbol, "daily", "daily")
        except ValueError as exc:
            skipped[symbol] = str(exc)
            continue
        except Exception:  # per ticker: one bad file must not stop the report
            logger.exception("scan_history failed for %s", symbol)
            skipped[symbol] = "execution_error"
            continue
        frames.append(hist[["Date", "Symbol", "Status", "Breakout Readiness"]])
    return pd.concat(frames, ignore_index=True), skipped


def survival_table(events: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(events["Event"], events["reason"]).reindex(
        index=EVENTS, columns=[*REMOVAL_REASONS, "member"], fill_value=0
    )
    counts.loc["All"] = counts.sum()
    out = pd.DataFrame({"Events": counts.sum(axis=1), "In universe": counts["member"]})
    out["Survival %"] = (out["In universe"] / out["Events"] * 100).round(1)
    for reason in REMOVAL_REASONS:
        out[f"Removed: {reason}"] = counts[reason]
    return out


def survival_by_year(events: pd.DataFrame) -> pd.DataFrame:
    year = events["Date"].dt.year.rename("Year")
    total = pd.crosstab(year, events["Event"]).reindex(columns=EVENTS, fill_value=0)
    kept = pd.crosstab(year[events["in_universe"]], events.loc[events["in_universe"], "Event"])
    kept = kept.reindex(index=total.index, columns=EVENTS, fill_value=0)
    out = pd.DataFrame(index=total.index)
    for e in EVENTS:
        out[f"{e} events"] = total[e]
        out[f"{e} kept"] = kept[e]
    return out


def universe_size(members: pd.DataFrame) -> pd.DataFrame:
    year = members["Date"].dt.year
    per_day = members.groupby("Date").agg(
        tickers=("Symbol", "size"),
        eligible=("rank", "count"),
        members=("in_universe", "sum"),
    )
    by_year = per_day.groupby(per_day.index.year).median().round(0).astype(int)
    by_year.index.name = "Year"
    reasons = pd.crosstab(year, members["reason"], normalize="index").reindex(
        columns=uv.REASONS, fill_value=0
    )
    reasons = (reasons * 100).round(1).add_suffix(" %")
    by_year.columns = ["Tickers with a bar (median/day)", "Eligible", "Members"]
    return by_year.join(reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=config.TIMEFRAME_PROFILES["daily"]["raw_dir"])
    parser.add_argument("--weekly-dir", default=config.TIMEFRAME_PROFILES["weekly"]["raw_dir"])
    parser.add_argument("--out-dir", default=config.RESEARCH_DIR)
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument(
        "--commit",
        default=os.environ.get("GIT_COMMIT"),
        help="code version for the report (default: $GIT_COMMIT, else git rev-parse)",
    )
    args = parser.parse_args(argv)
    commit = args.commit or _git_rev()
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = uv.load_daily_bars(args.raw_dir)
    panel = uv.panel_from_bars(bars)
    daily_first = {s: b["Date"].min() for s, b in bars.items()}
    weekly = uv.load_weekly_dates(args.weekly_dir, set(bars))
    prior = uv.prior_sessions(daily_first, weekly)
    members = uv.build_membership(panel, prior)
    members.to_csv(out_dir / f"universe_membership_{args.stamp}.csv.gz", index=False)
    logger.info("membership: %s rows", len(members))

    history, skipped = history_panel(bars)
    events = ev.first_bar_events(history)
    events = events.merge(members, on=["Date", "Symbol"], how="left", validate="one_to_one")
    events["in_universe"] = events["in_universe"].fillna(False).astype(bool)
    events["reason"] = events["reason"].fillna("no_row")
    events["Rubric Version"] = config.BREAKOUT_RUBRIC_VERSION
    events.to_csv(out_dir / f"universe_events_{args.stamp}.csv", index=False)

    survival = survival_table(events)
    survival.to_csv(out_dir / f"universe_event_survival_{args.stamp}.csv")
    by_year = survival_by_year(events)
    size = universe_size(members)

    window_start = min(daily_first.values())
    no_weekly = sorted(s for s in bars if s not in weekly)
    long_listed_fallback = sorted(s for s in no_weekly if daily_first[s] == window_start)
    listed_in_window = sorted(s for s in no_weekly if daily_first[s] != window_start)
    fallback_removed = events[
        events["Symbol"].isin(long_listed_fallback) & events["reason"].eq("history")
    ]
    fallback_counts = fallback_removed["Event"].value_counts().reindex(EVENTS, fill_value=0)

    md = f"""# Research universe: membership and event survival ({args.stamp})

Point-in-time universe (docs/research_review.md fix #1) applied to RESEARCH.md §3
events. **No forward prices or returns are computed here.** Event counts are not
outcomes; nothing in the holdout is evaluated.

- Code `{commit}`, breakout rubric v{config.BREAKOUT_RUBRIC_VERSION}. Daily raw `{args.raw_dir}` ({len(bars)} tickers,
  {window_start.date()} → {panel["Date"].max().date()}), weekly raw `{args.weekly_dir}`.
- Rule at date t (bars ≤ t only): not excluded; ≥ {config.RESEARCH_MIN_HISTORY_BARS} sessions
  of listing history; Close ≥ ${config.RESEARCH_MIN_CLOSE:g}; top {config.RESEARCH_UNIVERSE_TOP_N}
  by trailing {config.RESEARCH_DV_WINDOW}-bar median Close × Volume (ties: symbol A→Z).
- Excluded funds/ETPs (`RESEARCH_EXCLUDED_SYMBOLS`, reviewed by hand):
  {", ".join(sorted(config.RESEARCH_EXCLUDED_SYMBOLS))}.
- Membership is checked on the signal bar's date (the signal is known at that close).

## Caveats

- **Survivorship is reduced, not removed.** Candidates are still today's raw files
  (today's manifest + `most_actives`); delisted names are absent (RESEARCH.md §8.4). The
  rule removes *same-day* selection: membership at t depends only on data up to t.
- Prices are Yahoo auto-adjusted, so past Close (the $5 floor, dollar volume) reflects
  later dividend adjustments (fix #2).
- Listing age = daily bars + 5 × weekly bars whose week ended before the daily window.
  This overcounts holidays and partial first weeks by at most ~13 sessions, which matters
  only for names listed 2020-10 → 2021-10.
- **{len(no_weekly)} tickers have no weekly raw file**, so listing age falls back to the
  daily bar count. That is exact for the {len(listed_in_window)} whose first daily bar falls
  inside the window (listed then, so the daily file is their whole history):
  {", ".join(listed_in_window) or "none"}. It undercounts the other
  {len(long_listed_fallback)}, which start on the daily window's first day (listed earlier,
  true age unknown) and so wait until they have {config.RESEARCH_MIN_HISTORY_BARS} daily bars:
  {", ".join(long_listed_fallback) or "none"}. Events the fallback removes from those
  {len(long_listed_fallback)} (reason `history`): E1 {fallback_counts[ev.E1]},
  E2 {fallback_counts[ev.E2]}, E3 {fallback_counts[ev.E3]}.
- `scan_history` skipped {len(skipped)} ticker(s): {", ".join(f"{k} ({v})" for k, v in skipped.items()) or "none"}.

## Event survival

`reason` is the first rule that fails at the event date: `excluded` (fund/ETP),
`history` (< {config.RESEARCH_MIN_HISTORY_BARS} sessions), `price` (< ${config.RESEARCH_MIN_CLOSE:g}),
`dv_window` (< {config.RESEARCH_DV_WINDOW} daily bars for the median), `rank`
(outside the top {config.RESEARCH_UNIVERSE_TOP_N}).

{_md(survival)}

### Per year (signal date)

{_md(by_year)}

## Universe size

Median per day, and the share of ticker-days by reason (%):

{_md(size)}

---
Files: `universe_membership_{args.stamp}.csv.gz`, `universe_events_{args.stamp}.csv`,
`universe_event_survival_{args.stamp}.csv`.
"""
    path = out_dir / f"universe_report_{args.stamp}.md"
    path.write_text(md, encoding="utf-8")
    logger.info("wrote %s in %.1fs", path, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
