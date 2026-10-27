"""Blind label run: RESEARCH.md §4-§5 labels for in-sample E1/E2/E3 events (fix #2).

1. Copies the daily and weekly raw dirs to ``<out-dir>/snapshots/<stamp>/``
   (cron wipes ``data/raw`` nightly) and sha256-hashes every file. Every later
   step reads only that copy, through ``raw.load_raw``.
2. Builds the point-in-time universe and ``scan_history`` events from the
   snapshot, keeps in-universe events dated on or before
   ``config.RESEARCH_IN_SAMPLE_END`` (the holdout stays sealed), and labels
   them and their date-matched random draws (``research.labels``).
3. Writes ``<out-dir>/labels_<stamp>/``: ``events.csv.gz``, ``draws.csv.gz``,
   ``manifest.json`` (commit, rubric version, research config, file hashes,
   row counts, data-quality counts).

**Blind:** prints row counts and data-quality counts only. No mean, rate, hit
rate or excess return is computed here.

The container has no git: pass ``--commit`` or ``GIT_COMMIT``. Usage (inside
the container, against the working tree)::

    python scripts/research_labels.py --raw-dir /app/data/raw/daily \\
        --weekly-dir /app/data/raw/weekly --out-dir /app/data/research \\
        --commit "$(git rev-parse --short HEAD)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import date
from pathlib import Path

import pandas as pd

from finance_vibe import config
from finance_vibe.log import setup_logging
from finance_vibe.research import events as ev
from finance_vibe.research import labels as lb
from finance_vibe.research import universe as uv

logger = logging.getLogger(__name__)

EVENTS = [ev.E1, ev.E2, ev.E3]
DQ_EVENT_COLUMNS = [*lb.DQ_COLUMNS, "draws_short"]


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


def snapshot(raw_dir: Path, weekly_dir: Path, dest: Path) -> dict[str, str]:
    """Copy both raw dirs under ``dest`` (which must not exist); sha256 of every file."""
    if dest.exists():
        raise FileExistsError(f"snapshot already exists: {dest} (use --snapshot to reuse it)")
    shutil.copytree(raw_dir, dest / "daily")
    shutil.copytree(weekly_dir, dest / "weekly")
    return hash_tree(dest)


def hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.csv"))
    }


def dq_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {c: int(frame[c].sum()) for c in columns}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default=config.TIMEFRAME_PROFILES["daily"]["raw_dir"])
    parser.add_argument("--weekly-dir", default=config.TIMEFRAME_PROFILES["weekly"]["raw_dir"])
    parser.add_argument("--out-dir", default=config.RESEARCH_DIR)
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument(
        "--snapshot", help="reuse this existing snapshot dir (with daily/ and weekly/)"
    )
    parser.add_argument(
        "--commit",
        default=os.environ.get("GIT_COMMIT"),
        help="code version for the manifest (default: $GIT_COMMIT, else git rev-parse)",
    )
    args = parser.parse_args(argv)
    commit = args.commit or _git_rev()
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    run_dir = out_dir / f"labels_{args.stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    if args.snapshot:
        snap = Path(args.snapshot)
        hashes = hash_tree(snap)
    else:
        snap = out_dir / "snapshots" / args.stamp
        hashes = snapshot(Path(args.raw_dir), Path(args.weekly_dir), snap)
    logger.info("snapshot %s: %s files hashed", snap, len(hashes))

    bars = uv.load_daily_bars(str(snap / "daily"))
    daily_first = {s: b["Date"].min() for s, b in bars.items()}
    weekly = uv.load_weekly_dates(str(snap / "weekly"), set(bars))
    members = uv.build_membership(uv.panel_from_bars(bars), uv.prior_sessions(daily_first, weekly))
    history, skipped = ev.history_panel(bars)
    events = ev.first_bar_events(history)
    events = events.merge(members, on=["Date", "Symbol"], how="left", validate="one_to_one")
    all_events = len(events)
    end = pd.Timestamp(config.RESEARCH_IN_SAMPLE_END)
    start = pd.Timestamp(config.RESEARCH_IN_SAMPLE_START)
    in_window = events["Date"].between(start, end)
    in_universe = events["in_universe"].fillna(False).astype(bool)
    events = events.loc[in_window & in_universe, ["Symbol", "Date", "Event"]]

    ev_labels, dr_labels = lb.build_labels(events, members, bars, history)
    ev_labels.to_csv(run_dir / "events.csv.gz", index=False)
    dr_labels.to_csv(run_dir / "draws.csv.gz", index=False)

    rows = {
        "events_all_dates_all_members": all_events,
        "events_in_sample_in_universe": len(ev_labels),
        "events_by_type": ev_labels["Event"].value_counts().reindex(EVENTS, fill_value=0).to_dict(),
        "draws": len(dr_labels),
        "draws_by_type": dr_labels["Event"].value_counts().reindex(EVENTS, fill_value=0).to_dict(),
    }
    dq = {
        "events": dq_counts(ev_labels, DQ_EVENT_COLUMNS),
        "draws": dq_counts(dr_labels, lb.DQ_COLUMNS),
    }
    manifest = {
        "stamp": args.stamp,
        "commit": commit,
        "breakout_rubric_version": config.BREAKOUT_RUBRIC_VERSION,
        "research_config": {
            k: (sorted(v) if isinstance(v, frozenset) else v)
            for k, v in vars(config).items()
            if k.startswith("RESEARCH_") and k != "RESEARCH_DIR"
        },
        "split_jump": lb.SPLIT_JUMP,
        "snapshot": str(snap),
        "max_event_date": str(ev_labels["Date"].max().date()),
        "scan_history_skipped": skipped,
        "rows": rows,
        "data_quality": dq,
        "files": hashes,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Blind console output: counts only.
    counts = pd.DataFrame({"events": rows["events_by_type"], "draws": rows["draws_by_type"]})
    print(counts.to_markdown())
    print()
    dq_table = pd.DataFrame({"events": dq["events"], "draws": pd.Series(dq["draws"])})
    print(dq_table.fillna(0).astype(int).to_markdown())
    logger.info(
        "wrote %s (%s events, %s draws, max signal date %s) in %.1fs",
        run_dir,
        len(ev_labels),
        len(dr_labels),
        manifest["max_event_date"],
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
