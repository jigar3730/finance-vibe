"""Scanner history (Coiled Cobra or Breakout) for every universe ticker of a frozen snapshot.

Reads only ``research.snapshot`` folders (after checking their hashes) and
writes ``<scanner>_<mode>_history.csv.gz`` plus a ``.json`` run record to
``config.BACKFILL_HISTORY_DIR/<snapshot>/rubric-<version>/``, away from
``data/logs/`` so backfilled rows never show up as real past runs. A finished
file is never overwritten. It shows what today's rules say about past bars.

    python -m finance_vibe.research.backfill <snapshot-dir> --scanner coiled_cobra --commit <hash>
    python -m finance_vibe.research.backfill <snapshot-dir> --scanner breakout --commit <hash>
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from finance_vibe import config
from finance_vibe.data import raw
from finance_vibe.log import setup_logging
from finance_vibe.research import snapshot as sn
from finance_vibe.scanners import breakout as bs
from finance_vibe.scanners import coiled_cobra as cc

logger = logging.getLogger(__name__)

Task = tuple[str, str, str, str, str]  # scanner, symbol, bars path, benchmark path, mode

COBRA, BREAKOUT = "coiled_cobra", "breakout"
SCANNER_COLUMNS = {
    COBRA: config.COBRA_HISTORY_COLUMNS,
    BREAKOUT: ["Date", *bs.OUTPUT_COLUMNS, "Rubric Version"],
}
RUBRIC_VERSIONS = {COBRA: config.RUBRIC_VERSION, BREAKOUT: config.BREAKOUT_RUBRIC_VERSION}


def universe_symbols(snapshot: Path) -> list[str]:
    """The ``In Universe`` tickers of a snapshot, sorted."""
    table = pd.read_csv(snapshot / sn.UNIVERSE)
    return sorted(table.loc[table["In Universe"], "Symbol"].astype(str))


def _scan_one(task: Task) -> tuple[str, pd.DataFrame | None, str | None]:
    """One ticker's history, or the error; a bad ticker must not stop the run."""
    scanner, symbol, bars_path, benchmark_path, mode = task
    try:
        bars = raw.load_raw(bars_path)
        if scanner == BREAKOUT:
            return symbol, bs.scan_history(bars, symbol, mode, mode), None
        return symbol, cc.scan_history(bars, raw.load_raw(benchmark_path), symbol, mode), None
    except Exception as exc:  # per-ticker scan loop
        logger.exception("scan_history failed for %s", symbol)
        return symbol, None, f"{type(exc).__name__}: {exc}"


def _results(
    tasks: list[Task], workers: int
) -> Iterator[tuple[str, pd.DataFrame | None, str | None]]:
    """Per-ticker results in task order, so the output never depends on ``workers``."""
    if workers <= 1:
        yield from map(_scan_one, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_scan_one, tasks)


def run_mode(
    snapshot: Path,
    mode: str,
    *,
    out_root: Path,
    scanner: str = COBRA,
    workers: int = config.BACKFILL_WORKERS,
    commit: str = "unknown",
    only: Iterable[str] | None = None,
) -> Path:
    """Write one mode's history file; returns its path.

    Raises ``FileExistsError`` if it already exists and ``ValueError`` if the
    snapshot changed or lacks the benchmark. ``only`` narrows the tickers,
    for trial runs into a separate ``out_root``.
    """
    manifest = sn.verify(snapshot)
    if scanner not in SCANNER_COLUMNS:
        raise ValueError(f"unknown scanner: {scanner!r}")
    out_dir = out_root / manifest["stamp"] / f"rubric-{RUBRIC_VERSIONS[scanner]}"
    final = out_dir / f"{scanner}_{mode}_history.csv.gz"
    if final.exists():
        raise FileExistsError(f"history already exists: {final}")

    paths = {f.ticker: f.path for f in raw.raw_files(str(snapshot / mode))}
    if scanner == COBRA and cc.BENCHMARK not in paths:
        raise ValueError(f"snapshot has no {cc.BENCHMARK} {mode} bars")
    symbols = universe_symbols(snapshot)
    if only is not None:
        wanted = {s.upper() for s in only}
        symbols = [s for s in symbols if s in wanted]
    benchmark = paths.get(cc.BENCHMARK, "")
    tasks = [(scanner, s, paths[s], benchmark, mode) for s in symbols if s in paths]
    failures = {s: "no bars in snapshot" for s in symbols if s not in paths}

    out_dir.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(f"{final.name}.partial")
    rows = passed = 0
    first_date: Any = None
    last_date: Any = None
    with gzip.open(partial, "wt", newline="") as handle:
        handle.write(",".join(SCANNER_COLUMNS[scanner]) + "\n")
        for done, (symbol, history, error) in enumerate(_results(tasks, workers), 1):
            if done % 25 == 0:
                logger.info("%s: %d of %d tickers, %d rows", mode, done, len(tasks), rows)
            if history is None:
                failures[symbol] = str(error)
                continue
            if history.empty:
                continue
            history.to_csv(handle, index=False, header=False)
            rows += len(history)
            passed += int(history["Passed"].sum()) if "Passed" in history else 0
            low, high = history["Date"].min(), history["Date"].max()
            first_date = low if first_date is None else min(first_date, low)
            last_date = high if last_date is None else max(last_date, high)

    record = {
        "snapshot": manifest["stamp"],
        "mode": mode,
        "scanner": scanner,
        "rubric_version": RUBRIC_VERSIONS[scanner],
        "commit": commit,
        "tickers": len(symbols) - len(failures),
        "rows": rows,
        "passed_rows": passed if scanner == COBRA else None,
        "first_date": None if first_date is None else str(first_date)[:10],
        "last_date": None if last_date is None else str(last_date)[:10],
        "failures": failures,
    }
    final.with_suffix("").with_suffix(".json").write_text(json.dumps(record, indent=1))
    partial.rename(final)
    logger.info("%s: %d rows, %d passed, %d failed -> %s", mode, rows, passed, len(failures), final)
    return final


def main(argv: list[str] | None = None) -> int:
    """CLI: ``<snapshot-dir> [--scanner S] [--modes ...] [--workers N] [--only ...] [--out-root DIR]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot")
    parser.add_argument("--scanner", default=COBRA, choices=sorted(SCANNER_COLUMNS))
    parser.add_argument(
        "--modes", nargs="+", default=["daily", "weekly"], choices=["daily", "weekly"]
    )
    parser.add_argument("--workers", type=int, default=config.BACKFILL_WORKERS)
    parser.add_argument("--out-root", default=config.BACKFILL_HISTORY_DIR)
    parser.add_argument("--only", nargs="+", help="trial runs: these tickers only")
    parser.add_argument("--commit", default="unknown")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    for mode in args.modes:
        run_mode(
            Path(args.snapshot),
            mode,
            out_root=Path(args.out_root),
            scanner=args.scanner,
            workers=args.workers,
            commit=args.commit,
            only=args.only,
        )
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
