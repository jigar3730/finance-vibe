"""Finance Vibe pipeline: refresh the ticker universe and raw OHLCV, then run both scanners.

Stages run in order for one mode (weekly or daily), each as
``python -m <module> ...`` (or in-process with ``--in-process``).

``--as-of YYYY-MM-DD`` replays a past date from the raw data already on disk
(implies ``--reuse-raw``): every scanner sees only bars completed on that date
and writes its usual dated files stamped with it, e.g.
``python -m finance_vibe.pipeline --as-of 2025-11-07``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from finance_vibe import config
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    """One pipeline stage: a module with an entry point ``main(argv)``."""

    module: str
    modes: tuple[str, ...] = ("weekly", "daily")
    pass_mode: bool = True  # pass the mode as the first CLI argument
    as_of: bool = False  # pass --as-of on a replay
    ingest: bool = False  # refreshes data/raw; skipped with --reuse-raw / --as-of

    @property
    def name(self) -> str:
        return self.module.rsplit(".", 1)[-1]

    def argv(self, mode: str, as_of: str | None) -> list[str]:
        args = [mode] if self.pass_mode else []
        if as_of and self.as_of:
            args += ["--as-of", as_of]
        return args


# Run in this order. Data timeframe and signal profile are the same mode.
STAGES: tuple[Stage, ...] = (
    Stage("finance_vibe.data.tickers", pass_mode=False, ingest=True),
    Stage("finance_vibe.data.ingest", modes=("weekly",), ingest=True),
    # Daily has its own ingest wrapper (drops today's in-progress bar).
    Stage("finance_vibe.data.daily_ingest", modes=("daily",), pass_mode=False, ingest=True),
    Stage("finance_vibe.scanners.coiled_cobra", as_of=True),
    Stage("finance_vibe.scanners.breakout", as_of=True),
)


def _run_in_process(stage: Stage, argv: list[str]) -> bool:
    """Import the stage and call its ``main(argv)``; True on exit code 0."""
    try:
        module = importlib.import_module(stage.module)
        return int(module.main(argv) or 0) == 0
    except SystemExit as exc:
        return exc.code in (None, 0)
    except Exception:  # stage boundary: report and halt like a failed subprocess
        logger.exception(f"{stage.module} raised")
        return False


def clean_raw_folder(root_dir: str, mode: str) -> None:
    """Remove all files in data/raw/{mode}/ before a fresh ingestion run."""
    raw_dir = Path(root_dir) / "data" / "raw" / mode
    if not raw_dir.exists():
        logger.warning(f"Raw '{mode}' folder does not exist. Skipping cleanup.")
        return

    for item in raw_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except OSError as e:
            logger.error(f"Failed to delete {item}: {e}")

    logger.info(f"🧹 Raw '{mode}' folder cleaned.")


def run_workflow(argv: list[str] | None = None) -> None:
    """Parse CLI args and run each pipeline stage (subprocess by default)."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Finance Vibe pipeline")
    parser.add_argument(
        "--mode",
        choices=["weekly", "daily"],
        default="weekly",
        help="Execution profile (weekly or daily)",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help=(
            "Keep existing data/raw/{mode} files; skip wipe, ticker refresh, "
            "and yfinance ingest. Scanners still run on the files already on disk."
        ),
    )
    parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        help=(
            "Replay a past date: scanners see only bars completed on/before it and "
            "outputs are stamped with it. Implies --reuse-raw (no wipe/re-download). "
            "Uses today's ticker list and today's split/dividend-adjusted prices."
        ),
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help=(
            "Run stages as function calls in this interpreter instead of subprocesses "
            "(debugging / faster local runs). Cron keeps the default subprocess mode."
        ),
    )
    args = parser.parse_args(argv)
    mode = args.mode.lower()

    root_dir = config.PROJECT_ROOT
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(root_dir, "src") + os.pathsep + env.get("PYTHONPATH", "")

    # Validate --as-of once, up front, with the same parser every stage uses.
    as_of = None
    if args.as_of:
        try:
            as_of = config.parse_as_of(["--as-of", args.as_of])
        except ValueError as exc:
            parser.error(str(exc))
    reuse_raw = args.reuse_raw or as_of is not None

    logger.info(f"🚀 Starting Finance Vibe pipeline [{mode.upper()} MODE]...")
    logger.info(f"📍 Project Root: {root_dir}")
    if as_of:
        logger.info(f"⏪ AS-OF replay: {as_of} (bars completed on/before this date only)")
        if mode == "weekly" and date.fromisoformat(as_of).weekday() != 4:
            logger.info("   Note: as-of is not a Friday, so the week in progress is excluded.")
        logger.info("   Uses today's ticker list and split/dividend-adjusted prices.")
    if args.in_process:
        logger.info("🧪 In-process mode: stages run as function calls in this interpreter.")

    # Clean the shared raw silo unless the caller wants to reuse existing OHLCV.
    if reuse_raw:
        logger.info(f"♻️  Reusing existing raw files in data/raw/{mode}/")
    else:
        clean_raw_folder(root_dir, mode)

    for stage in STAGES:
        if mode not in stage.modes:
            continue
        if reuse_raw and stage.ingest:
            logger.info(f"⏭️  Skipping {stage.name} (--reuse-raw).")
            continue

        logger.info(f"🔹 Running: {stage.name}...")
        stage_argv = stage.argv(mode, as_of)
        if args.in_process:
            ok = _run_in_process(stage, stage_argv)
        else:
            cmd = [sys.executable, "-m", stage.module, *stage_argv]
            try:
                subprocess.run(cmd, check=True, env=env, cwd=root_dir)
                ok = True
            except subprocess.CalledProcessError:
                ok = False
        if not ok:
            logger.error(f"Error in {stage.name}. Pipeline halted.")
            sys.exit(1)
        logger.info(f"✅ Finished: {stage.name}")

    logger.info("🏁 Workflow Complete!")
    logger.info(f"📁 Reports saved to: {os.path.join(root_dir, 'data', 'logs', mode)}")


if __name__ == "__main__":
    run_workflow()
