"""Blind feature run for RESEARCH.md §13: ranker features from the frozen backfill snapshot.

1. Checks the snapshot against its manifest (``research.snapshot.verify``) and reads the
   breakout daily backfill history (Support / Resistance) for it. Every price comes from
   that snapshot, through ``raw.load_raw``.
2. Builds the §13 feature panel (``research.ranker_features``) on the training grid:
   every ``RESEARCH_RANKER_TRAIN_STRIDE``-th SPY session from
   ``RESEARCH_RANKER_IN_SAMPLE_START`` through ``RESEARCH_IN_SAMPLE_END``. The score grid
   (every ``RESEARCH_RANKER_SCORE_STRIDE``-th session) is a subset of it.
3. Writes ``<out-dir>/ranker_features_<stamp>/``: ``features.csv.gz`` (every universe
   ticker on every grid date, eligible or not) and ``manifest.json`` (commit, config,
   file hashes, row counts, eligibility counts).

**Blind:** no forward bar is read and no label or return is computed. The console shows
counts only: grid dates, eligible stocks per date, and what each rule removes.

The container has no git: pass ``--commit`` or ``GIT_COMMIT``. Usage (inside the
container, against the working tree)::

    python scripts/research_ranker_features.py --commit "$(git rev-parse --short HEAD)"
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
from finance_vibe.research import ranker_features as rf
from finance_vibe.research import snapshot as sn

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _levels(path: Path) -> pd.DataFrame:
    cols = ["Date", "Symbol", "Resistance", "Support"]
    frame = pd.read_csv(path, usecols=[*cols, "Rubric Version"])
    if {float(v) for v in frame["Rubric Version"].unique()} != {
        float(config.BREAKOUT_RUBRIC_VERSION)
    }:
        raise ValueError(f"{path}: rubric version is not {config.BREAKOUT_RUBRIC_VERSION}")
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame[cols]


def fold_of(dates: pd.Series) -> pd.Series:
    """The §13 test fold (1-6) of each date, 0 before the first fold."""
    out = pd.Series(0, index=dates.index, dtype="int64")
    for i, (lo, hi) in enumerate(config.RESEARCH_RANKER_FOLDS, start=1):
        out[(dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))] = i
    return out


def eligibility_counts(panel: pd.DataFrame, score_dates: pd.DatetimeIndex) -> dict:
    """Counts only: how many rows each rule removes and eligible stocks per date."""
    member = panel["in_universe"]
    per_date = panel.groupby("Date")["eligible"].sum()
    score = panel[panel["Date"].isin(score_dates)]
    score_per_date = score.groupby("Date")["eligible"].sum()
    folds = fold_of(pd.Series(score_per_date.index, index=score_per_date.index))

    def stats(s: pd.Series) -> dict[str, float]:
        return {
            "dates": len(s),
            "min": int(s.min()),
            "median": float(s.median()),
            "max": int(s.max()),
        }

    return {
        "rows": len(panel),
        "rows_eligible": int(panel["eligible"].sum()),
        "removed_point_in_time_check": int((~member).sum()),
        "missing_feature_among_members": {
            f: int(panel.loc[member, f].isna().sum()) for f in rf.FEATURES
        },
        "train_grid_eligible_per_date": stats(per_date),
        "score_grid_eligible_per_date": stats(score_per_date),
        "score_dates_by_fold": {
            str(k): int(v) for k, v in folds.value_counts().sort_index().items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    stamp = config.RESEARCH_RANKER_SNAPSHOT
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
    run_dir = Path(args.out_dir) / f"ranker_features_{args.stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    snapshot_manifest = sn.verify(snap)
    symbols = bf.universe_symbols(snap)
    bars = {
        s: raw.load_raw(str(snap / "daily" / f"{s}_10y_1d.csv")) for s in [*symbols, *rf.BENCHMARKS]
    }
    breakout_path = (
        history_dir / f"rubric-{config.BREAKOUT_RUBRIC_VERSION}" / "breakout_daily_history.csv.gz"
    )
    levels = _levels(breakout_path)

    train_dates = rf.session_grid(bars["SPY"]["Date"], config.RESEARCH_RANKER_TRAIN_STRIDE)
    score_dates = rf.session_grid(bars["SPY"]["Date"], config.RESEARCH_RANKER_SCORE_STRIDE)
    panel = rf.build_panel(bars, symbols, levels, dates=train_dates)
    if panel["Date"].max() > pd.Timestamp(config.RESEARCH_IN_SAMPLE_END):
        raise ValueError("holdout is sealed: feature panel runs past RESEARCH_IN_SAMPLE_END")
    out_path = run_dir / "features.csv.gz"
    panel.to_csv(out_path, index=False)

    counts = eligibility_counts(panel, score_dates)
    manifest = {
        "stamp": args.stamp,
        "section": "RESEARCH.md §13",
        "commit": args.commit,
        "breakout_rubric_version": config.BREAKOUT_RUBRIC_VERSION,
        "config": {
            k: v
            for k, v in vars(config).items()
            if k.startswith(("RESEARCH_RANKER_", "BACKFILL_MIN_", "BACKFILL_DV_"))
        }
        | {"RESEARCH_IN_SAMPLE_END": config.RESEARCH_IN_SAMPLE_END},
        "features": rf.FEATURES,
        "snapshot": str(snap),
        "snapshot_stamp": snapshot_manifest.get("stamp"),
        "universe_tickers": len(symbols),
        "history_files": {str(breakout_path): _sha256(breakout_path)},
        "output_files": {out_path.name: _sha256(out_path)},
        "train_dates": len(train_dates),
        "score_dates": len(score_dates),
        "first_date": str(train_dates.min().date()),
        "last_date": str(train_dates.max().date()),
        "counts": counts,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Blind console output: counts only.
    print(
        pd.DataFrame(
            {
                "train grid": counts["train_grid_eligible_per_date"],
                "score grid": counts["score_grid_eligible_per_date"],
            }
        ).to_markdown()
    )
    print()
    print(
        pd.Series(counts["missing_feature_among_members"], name="missing among members")
        .to_frame()
        .to_markdown()
    )
    print()
    print(pd.Series(counts["score_dates_by_fold"], name="score dates").to_frame().to_markdown())
    logger.info(
        "wrote %s (%d rows, %d eligible, %s .. %s, %d removed by the point-in-time check) in %.1fs",
        run_dir,
        counts["rows"],
        counts["rows_eligible"],
        manifest["first_date"],
        manifest["last_date"],
        counts["removed_point_in_time_check"],
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
