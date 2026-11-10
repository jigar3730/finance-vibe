"""In-sample analysis for RESEARCH.md §12, exactly as frozen. Run once.

Reads the blind label run (``scripts/research_prebreakout_labels.py``) and writes,
under ``<out-dir>/prebreakout_insample_<stamp>/``:

    scorecard.csv     C1 and C2: count, % up, mean R10, mean - SPY, mean - random,
                      date-clustered p-values, Holm across the two, PASS/FAIL
    reported.csv      breakout reached within 10 bars and which level came first,
                      each against the random draws (not a pass criterion)
    by_year.csv       the scorecard metrics by signal year (descriptive only)
    robustness.csv    the scorecard with split_suspect rows excluded (not a criterion)
    reference.csv     PRE_BREAKOUT alone on the same universe (reference only)
    report.md         all of the above, plus inputs and caveats

The PRE_BREAKOUT reference is labelled here, with the same §12 label code and
snapshot, because the blind label run did not include it. It is not a criterion.
Every signal and draw date is checked in-sample before anything is written: the
holdout is never read. Refuses to overwrite an existing output dir.

The container has no git: pass ``--commit`` or ``GIT_COMMIT``. Usage::

    python scripts/prebreakout_insample_report.py --commit "$(git rev-parse --short HEAD)"
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd

from finance_vibe import config
from finance_vibe.data import raw
from finance_vibe.log import setup_logging
from finance_vibe.research import backfill as bf
from finance_vibe.research import prebreakout as pb
from finance_vibe.research import prebreakout_insample as pi
from finance_vibe.research import snapshot as sn

logger = logging.getLogger(__name__)


def _fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt=".4g")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _reference_inputs(
    snap: Path, history: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Breakout history, point-in-time members and bars of the frozen snapshot."""
    sn.verify(snap)
    symbols = bf.universe_symbols(snap)
    bench = config.RESEARCH_BENCHMARK
    bars = {s: raw.load_raw(str(snap / "daily" / f"{s}_10y_1d.csv")) for s in [*symbols, bench]}
    path = history / f"rubric-{config.BREAKOUT_RUBRIC_VERSION}" / "breakout_daily_history.csv.gz"
    columns = ["Date", "Symbol", "Status", "Resistance", "Support", "Rubric Version"]
    breakout = pd.read_csv(path, usecols=columns)
    if {float(v) for v in breakout["Rubric Version"].unique()} != {
        float(config.BREAKOUT_RUBRIC_VERSION)
    }:
        raise ValueError(f"{path}: rubric version is not {config.BREAKOUT_RUBRIC_VERSION}")
    breakout["Date"] = pd.to_datetime(breakout["Date"])
    return breakout, pb.point_in_time_members(bars, symbols), bars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    stamp = config.RESEARCH_PREBREAKOUT_SNAPSHOT
    parser.add_argument(
        "--labels-dir", default=os.path.join(config.RESEARCH_DIR, f"prebreakout_labels_{stamp}")
    )
    parser.add_argument("--snapshot", default=os.path.join(config.BACKFILL_SNAPSHOT_DIR, stamp))
    parser.add_argument("--history", default=os.path.join(config.BACKFILL_HISTORY_DIR, stamp))
    parser.add_argument("--out-dir", default=config.RESEARCH_DIR)
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument(
        "--commit", default=os.environ.get("GIT_COMMIT"), help="code version for the report"
    )
    args = parser.parse_args(argv)
    if not args.commit:
        parser.error("--commit (or GIT_COMMIT) is required: the container has no git")
    labels_dir = Path(args.labels_dir)
    events, draws, manifest = pi.load(labels_dir)  # seal checked before anything is written
    run_dir = Path(args.out_dir) / f"prebreakout_insample_{args.stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    e = pi.with_benchmarks(events, draws)
    card = pi.scorecard(e)
    shown = pi.reported(e)
    years = pi.by_year(e)
    no_split = draws[~draws["split_suspect"]]
    robust = pi.scorecard(
        pi.with_benchmarks(events[~events["split_suspect"]], no_split), decide=False
    )
    breakout, members, bars = _reference_inputs(Path(args.snapshot), Path(args.history))
    ref_card, ref_shown = pi.reference(breakout, members, bars)
    ref = pd.DataFrame([{**ref_card, **ref_shown}]).set_index("Signal")

    card.to_csv(run_dir / "scorecard.csv")
    shown.to_csv(run_dir / "reported.csv")
    years.to_csv(run_dir / "by_year.csv", index=False)
    robust.to_csv(run_dir / "robustness.csv")
    ref.to_csv(run_dir / "reference.csv")

    inputs = {name: _sha(labels_dir / name) for name in ("events.csv.gz", "draws.csv.gz")}
    md = f"""# Pre-breakout in-sample results ({args.stamp})

RESEARCH.md §12, run once. Holdout (signal dates after {config.RESEARCH_IN_SAMPLE_END}) not
read: every event and draw date was checked in-sample.

- Report code `{args.commit}`. Labels `{labels_dir}` (label code `{manifest["commit"]}`, Coiled
  Cobra rubric v{manifest["cobra_rubric_version"]}, breakout rubric
  v{manifest["breakout_rubric_version"]}, snapshot `{manifest["snapshot"]}`).
- Inputs sha256: events `{inputs["events.csv.gz"][:16]}…`, draws `{inputs["draws.csv.gz"][:16]}…`.
- Signal dates {events["Date"].min().date()} → {events["Date"].max().date()}
  ({events["Date"].nunique()} signal dates). R10 = Close[t+10] / Open[t+1] − 1, no costs.
  Excess vs SPY on the same dates; vs the mean of {config.RESEARCH_RANDOM_DRAWS} date-matched
  random members of the same universe.
- Test: per signal date mean excess → one-sided t-test (H1: > 0), vs SPY and vs random;
  p = the larger; Holm across C1 and C2; PASS if p (Holm) < {pi.ALPHA}.
- **Universe:** 520 tickers chosen with data up to 2026-11 (look-ahead in the choice, not just
  survivorship), with a point-in-time price and liquidity check at each signal. The random draws
  share that bias and SPY does not, so a result against SPY alone means little.
- **Contamination (§12):** C2 is a subset of PRE_BREAKOUT, whose returns §9 already computed for
  2022-02 → 2026-03; Cobra rubric v4.0 was partly tuned with returns over overlapping years. The
  in-sample result is likely to look better than it is; the holdout carries the weight.

## Scorecard (the pass criterion)

{_fmt(card)}

## Reported, not a pass criterion

Reach breakout: Status TRIGGERED or CONFIRMED on any of the next {config.RESEARCH_HORIZON} bars.
Level first: whether a High at or above bar t's Resistance comes before a Low at or below its
Support in the next {config.RESEARCH_HORIZON} bars (both on one bar counts as Support). C2 starts
nearer resistance than a random draw, so its level shares are not a fair comparison.

{_fmt(shown)}

## PRE_BREAKOUT alone on this universe (reference only, dropped in §9)

Labelled in this run with the §12 label code, because the blind label run did not include it.

{_fmt(ref)}

## By signal year — descriptive, not a pass criterion

{_fmt(years.set_index(["Year", "Signal"]))}

## Robustness only: split_suspect rows excluded

{int(events["split_suspect"].sum())} events and {len(draws) - len(no_split)} draws excluded.
Not a pass criterion.

{_fmt(robust)}
"""
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print(card.to_markdown(floatfmt=".4g"))
    logger.info("wrote %s", run_dir)
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
