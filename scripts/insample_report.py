"""In-sample analysis, exactly as RESEARCH.md §4-§7 specify. Run once.

Reads a blind label run (``scripts/research_labels.py``: ``events.csv.gz``,
``draws.csv.gz``, ``manifest.json``) and writes, under
``<out-dir>/insample_<stamp>/`` (``insample_weekly_<stamp>/`` for a weekly
label run, RESEARCH.md §11; the mode comes from the label manifest):

    scorecard.csv     one row per signal: count, % up, mean R10, mean - SPY,
                      mean - random, date-clustered p-values, Holm, PASS/FAIL
    by_year.csv       the same metrics by signal year (descriptive only)
    robustness.csv    the scorecard with split_suspect rows excluded (robustness only)
    report.md         all three, plus inputs and caveats

§7: per signal, average the excess within each signal date, then a one-sided
one-sample t-test on those daily means (H1: mean > 0), vs SPY and vs random
separately. The signal's p-value is the larger of the two; Holm across the 3
signals; PASS if the Holm-adjusted p < 0.05. Every signal and draw date must be
in-sample (``labels.check_in_sample``): the holdout is never read. Refuses to
overwrite an existing output dir.

The container has no git: pass ``--commit`` or ``GIT_COMMIT``. Usage::

    python scripts/insample_report.py --labels-dir /app/data/research/labels_2026-10-27 \\
        --out-dir /app/data/research --commit "$(git rev-parse --short HEAD)"
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from finance_vibe import config
from finance_vibe.log import setup_logging
from finance_vibe.research.insample import ALPHA, by_year, load, scorecard, with_benchmarks

logger = logging.getLogger(__name__)


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


def _fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt=".4g")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-dir", default=config.RESEARCH_DIR)
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument(
        "--commit",
        default=os.environ.get("GIT_COMMIT"),
        help="code version for the report (default: $GIT_COMMIT, else git rev-parse)",
    )
    args = parser.parse_args(argv)
    commit = args.commit or _git_rev()
    labels_dir = Path(args.labels_dir)
    events, draws, manifest = load(labels_dir)  # seal checked before anything is written
    weekly = manifest.get("mode", "daily") == "weekly"
    run_dir = Path(args.out_dir) / (
        f"insample_weekly_{args.stamp}" if weekly else f"insample_{args.stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    e = with_benchmarks(events, draws)
    card = scorecard(e)
    years = by_year(e)
    no_split_draws = draws[~draws["split_suspect"]]
    robust = scorecard(
        with_benchmarks(events[~events["split_suspect"]], no_split_draws), decide=False
    )
    if weekly:
        shown = {"Mean R10 %": "Mean R4W %"}
        card, years, robust = (f.rename(columns=shown) for f in (card, years, robust))
        title, horizon = (
            "Weekly in-sample results (RESEARCH.md §11)",
            config.RESEARCH_WEEKLY_HORIZON,
        )
        label = "R4W = Close[t+4] / Open[t+1] − 1 on weekly bars (entry: next week's open)"
        dates = "signal weeks"
        end_rule = f"signals known (week end) after {config.RESEARCH_IN_SAMPLE_END}"
    else:
        title, horizon = "In-sample results", config.RESEARCH_HORIZON
        label = "R10 = Close[t+10] / Open[t+1] − 1"
        dates = "signal dates"
        end_rule = f"signal dates after {config.RESEARCH_IN_SAMPLE_END}"
    card.to_csv(run_dir / "scorecard.csv")
    years.to_csv(run_dir / "by_year.csv", index=False)
    robust.to_csv(run_dir / "robustness.csv")

    inputs = {
        "events.csv.gz": _sha(labels_dir / "events.csv.gz"),
        "draws.csv.gz": _sha(labels_dir / "draws.csv.gz"),
    }
    md = f"""# {title} ({args.stamp})

RESEARCH.md §4–§7, run once. Holdout ({end_rule})
not read: every event and draw date was checked in-sample.

- Report code `{commit}`. Labels `{labels_dir}` (label code `{manifest["commit"]}`,
  breakout rubric v{manifest["breakout_rubric_version"]}, snapshot `{manifest["snapshot"]}`).
- Inputs sha256: events `{inputs["events.csv.gz"][:16]}…`, draws `{inputs["draws.csv.gz"][:16]}…`.
- {dates.capitalize()} {events["Date"].min().date()} → {events["Date"].max().date()}
  ({events["Date"].nunique()} {dates}). {label}, no costs. Excess vs SPY on the same dates; vs the mean of
  {config.RESEARCH_RANDOM_DRAWS} date-matched random universe members.
- Test: per {dates[:-1]} mean excess → one-sided t-test (H1: > 0), vs SPY and vs random;
  p = the larger; Holm across the 3 signals; PASS if p (Holm) < {ALPHA}.
- **Survivorship:** reduced, not removed. Yahoo has no delisted names (RESEARCH.md §8.4);
  a re-test on a survivorship-free source is required before real money.

## Scorecard (the pass criterion)

{_fmt(card)}

Follow-through (PRE_BREAKOUT only): share of events whose Status is TRIGGERED or
CONFIRMED on any of the next {horizon} bars, vs the same rate for its random draws.
Reported, not a pass criterion.

## By signal year — descriptive, not a pass criterion

{_fmt(years.set_index(["Year", "Signal"]))}

## Robustness only: split_suspect rows excluded

{len(events) - int((~events["split_suspect"]).sum())} events and {len(draws) - len(no_split_draws)} draws
excluded. Not a pass criterion.

{_fmt(robust)}
"""
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print(card.to_markdown(floatfmt=".4g"))
    logger.info("wrote %s", run_dir)
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
