"""Blind label run for RESEARCH.md §12: C1 / C2 labels from the frozen backfill snapshot.

1. Checks the snapshot against its manifest (``research.snapshot.verify``) and reads
   the daily backfill history of both scanners for it. Every price comes from that
   snapshot, through ``raw.load_raw``.
2. Finds C1 / C2 events, keeps those known in-sample
   (``RESEARCH_PREBREAKOUT_IN_SAMPLE_START`` .. ``RESEARCH_IN_SAMPLE_END``; the holdout
   stays sealed) that pass the point-in-time check, and labels them and their 20
   date-matched random draws (``research.prebreakout``).
3. Writes ``<out-dir>/prebreakout_labels_<stamp>/``: ``events.csv.gz``,
   ``draws.csv.gz``, ``manifest.json`` (commit, rubric versions, config, file hashes,
   row counts, data-quality counts).

**Blind:** prints row counts and data-quality counts only. No mean, rate, REACH10,
LEVEL_FIRST tally or excess return is computed here.

The container has no git: pass ``--commit`` or ``GIT_COMMIT``. Usage (inside the
container, against the working tree)::

    python scripts/research_prebreakout_labels.py --commit "$(git rev-parse --short HEAD)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd

from finance_vibe import config
from finance_vibe.data import raw
from finance_vibe.log import setup_logging
from finance_vibe.research import backfill as bf
from finance_vibe.research import labels as lb
from finance_vibe.research import prebreakout as pb
from finance_vibe.research import snapshot as sn

logger = logging.getLogger(__name__)

DQ_EVENT_COLUMNS = [*lb.DQ_COLUMNS, "draws_short"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _history(path: Path, columns: list[str], version: str | float) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[*columns, "Rubric Version"])
    if {float(v) for v in frame["Rubric Version"].unique()} != {float(version)}:
        raise ValueError(f"{path}: rubric version is not {version}")
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame[columns]


def dq_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {c: int(frame[c].sum()) for c in columns}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    stamp = config.RESEARCH_PREBREAKOUT_SNAPSHOT
    parser.add_argument("--snapshot", default=os.path.join(config.BACKFILL_SNAPSHOT_DIR, stamp))
    parser.add_argument("--history", default=os.path.join(config.BACKFILL_HISTORY_DIR, stamp))
    parser.add_argument("--out-dir", default=config.RESEARCH_DIR)
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument(
        "--commit", default=os.environ.get("GIT_COMMIT"), help="code version for the manifest"
    )
    args = parser.parse_args(argv)
    if not args.commit:
        parser.error("--commit (or GIT_COMMIT) is required: the container has no git")
    started = time.perf_counter()
    snap, history_dir = Path(args.snapshot), Path(args.history)
    run_dir = Path(args.out_dir) / f"prebreakout_labels_{args.stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    snapshot_manifest = sn.verify(snap)
    symbols = bf.universe_symbols(snap)
    bench = config.RESEARCH_BENCHMARK
    bars = {s: raw.load_raw(str(snap / "daily" / f"{s}_10y_1d.csv")) for s in [*symbols, bench]}
    cobra_path = (
        history_dir / f"rubric-{config.RUBRIC_VERSION}" / "coiled_cobra_daily_history.csv.gz"
    )
    breakout_path = (
        history_dir / f"rubric-{config.BREAKOUT_RUBRIC_VERSION}" / "breakout_daily_history.csv.gz"
    )
    cobra = _history(cobra_path, ["Date", "Symbol", "Passed"], config.RUBRIC_VERSION)
    breakout = _history(
        breakout_path,
        ["Date", "Symbol", "Status", "Resistance", "Support"],
        config.BREAKOUT_RUBRIC_VERSION,
    )
    joined = len(cobra.merge(breakout[["Date", "Symbol"]], on=["Date", "Symbol"]))

    members = pb.point_in_time_members(bars, symbols)
    found = pb.events(cobra, breakout)
    found = found.merge(members, on=["Date", "Symbol"], how="left", validate="many_to_one")
    start = pd.Timestamp(config.RESEARCH_PREBREAKOUT_IN_SAMPLE_START)
    end = pd.Timestamp(config.RESEARCH_IN_SAMPLE_END)
    in_window = (found["Date"] >= start) & (found["Date"] <= end)
    member = found["in_universe"].astype("boolean").fillna(False).astype(bool)
    kept = found.loc[in_window & member, ["Symbol", "Date", "Event"]].reset_index(drop=True)

    ev_labels, dr_labels = pb.build_labels(kept, members, bars, breakout)
    ev_labels.to_csv(run_dir / "events.csv.gz", index=False)
    dr_labels.to_csv(run_dir / "draws.csv.gz", index=False)

    def by_event(frame: pd.DataFrame) -> dict[str, int]:
        return frame["Event"].value_counts().reindex(pb.EVENTS, fill_value=0).to_dict()

    rows = {
        "cobra_rows": len(cobra),
        "breakout_rows": len(breakout),
        "joined_rows": joined,
        "events_in_sample": by_event(found[in_window]),
        "removed_by_point_in_time_check": by_event(found[in_window & ~member]),
        "events_labelled": by_event(ev_labels),
        "draws": by_event(dr_labels),
    }
    dq = {
        "events": dq_counts(ev_labels, DQ_EVENT_COLUMNS)
        | {"level_missing": int(ev_labels[pb.LEVEL_FIRST].isna().sum())},
        "draws": dq_counts(dr_labels, lb.DQ_COLUMNS)
        | {"level_missing": int(dr_labels[pb.LEVEL_FIRST].isna().sum())},
    }
    manifest = {
        "stamp": args.stamp,
        "section": "RESEARCH.md §12",
        "commit": args.commit,
        "cobra_rubric_version": config.RUBRIC_VERSION,
        "breakout_rubric_version": config.BREAKOUT_RUBRIC_VERSION,
        "config": {
            k: (sorted(v) if isinstance(v, frozenset) else v)
            for k, v in vars(config).items()
            if k.startswith(("RESEARCH_", "BACKFILL_MIN_", "BACKFILL_DV_")) and k != "RESEARCH_DIR"
        },
        "split_jump": lb.SPLIT_JUMP,
        "snapshot": str(snap),
        "snapshot_stamp": snapshot_manifest.get("stamp"),
        "universe_tickers": len(symbols),
        "history_files": {str(p): _sha256(p) for p in (cobra_path, breakout_path)},
        "min_event_date": str(ev_labels["Date"].min().date()),
        "max_event_date": str(ev_labels["Date"].max().date()),
        "rows": rows,
        "data_quality": dq,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Blind console output: counts only.
    counts = pd.DataFrame({k: rows[k] for k in rows if isinstance(rows[k], dict)})
    print(counts.to_markdown())
    print()
    print(pd.DataFrame(dq).fillna(0).astype(int).to_markdown())
    logger.info(
        "wrote %s (%s events, %s draws, %s .. %s) in %.1fs",
        run_dir,
        len(ev_labels),
        len(dr_labels),
        manifest["min_event_date"],
        manifest["max_event_date"],
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
