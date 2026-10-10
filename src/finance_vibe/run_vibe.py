"""Finance Vibe pipeline orchestrator.

Runs ingestion, macro scoring, tactical scanning, and trade plan generation
in sequence for a given timeframe profile (weekly or daily).

``--as-of YYYY-MM-DD`` replays a past date from the raw data already on disk
(implies ``--reuse-raw``): every stage sees only bars completed on that date and
writes its usual dated files stamped with it, e.g.
``python src/finance_vibe/run_vibe.py --as-of 2025-11-07``.
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

from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    """One pipeline stage: ``src/finance_vibe/<module>.py``, entry point ``main(argv)``."""

    module: str
    modes: tuple[str, ...] = ("weekly", "daily")
    pass_mode: bool = True  # pass the mode as the first CLI argument
    as_of: bool = False  # pass --as-of on a replay
    ingest: bool = False  # refreshes data/raw; skipped with --reuse-raw / --as-of

    @property
    def path(self) -> str:
        return f"src/finance_vibe/{self.module}.py"

    def argv(self, mode: str, as_of: str | None) -> list[str]:
        args = [mode] if self.pass_mode else []
        if as_of and self.as_of:
            args += ["--as-of", as_of]
        return args


# Run in this order. Data timeframe and signal profile are the same mode.
STAGES: tuple[Stage, ...] = (
    Stage("ticker_provider", pass_mode=False, ingest=True),
    Stage("data_ingestor", modes=("weekly",), ingest=True),
    # Daily has its own ingest wrapper (drops today's in-progress bar).
    Stage("daily_ingest", modes=("daily",), pass_mode=False, ingest=True),
    Stage("analysis_engine", as_of=True),
    Stage("coiled_cobra", as_of=True),  # primary signal engine
    Stage("breakout_scanner", as_of=True),
    Stage("trade_planner", as_of=True),
    Stage("trade_plan_helper", as_of=True),
)


def _run_in_process(stage: Stage, argv: list[str]) -> bool:
    """Import the stage and call its ``main(argv)``; True on exit code 0."""
    try:
        module = importlib.import_module(f"finance_vibe.{stage.module}")
        return int(module.main(argv) or 0) == 0
    except SystemExit as exc:
        return exc.code in (None, 0)
    except Exception:  # stage boundary: report and halt like a failed subprocess
        logger.exception(f"{stage.path} raised")
        return False


def clean_raw_folder(root_dir, mode):
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
    parser = argparse.ArgumentParser(description="Finance-Vibe Pipeline Orchestrator")
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

    # 2. CLIMB TO ROOT
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    SRC_DIR = os.path.join(ROOT_DIR, "src")

    # 3. ENVIRONMENT SETUP
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")

    # Validate --as-of once, up front, with the same parser every stage uses.
    as_of = None
    if args.as_of:
        from finance_vibe import config

        try:
            as_of = config.parse_as_of(["--as-of", args.as_of])
        except ValueError as exc:
            parser.error(str(exc))
    reuse_raw = args.reuse_raw or as_of is not None

    logger.info(f"🚀 Starting Finance-Vibe Pipeline [{mode.upper()} MODE]...")
    logger.info(f"📍 Project Root: {ROOT_DIR}")
    if as_of:
        logger.info(f"⏪ AS-OF replay: {as_of} (bars completed on/before this date only)")
        if mode == "weekly" and date.fromisoformat(as_of).weekday() != 4:
            logger.info("   Note: as-of is not a Friday, so the week in progress is excluded.")
        logger.info(
            "   Uses today's ticker list and split/dividend-adjusted prices; ML ranking is skipped."
        )
    if args.in_process:
        logger.info("🧪 In-process mode: stages run as function calls in this interpreter.")

    # Clean the shared raw silo unless the caller wants to reuse existing OHLCV.
    if reuse_raw:
        logger.info(f"♻️  Reusing existing raw files in data/raw/{mode}/")
    else:
        clean_raw_folder(ROOT_DIR, mode)

    for stage in STAGES:
        if mode not in stage.modes:
            continue
        if reuse_raw and stage.ingest:
            logger.info(f"⏭️  Skipping {stage.path} (--reuse-raw).")
            continue

        logger.info(f"🔹 Running: {stage.path}...")
        stage_argv = stage.argv(mode, as_of)
        if args.in_process:
            ok = _run_in_process(stage, stage_argv)
        else:
            cmd = [sys.executable, os.path.join(ROOT_DIR, stage.path), *stage_argv]
            try:
                subprocess.run(cmd, check=True, env=env, cwd=ROOT_DIR)
                ok = True
            except subprocess.CalledProcessError:
                ok = False
        if not ok:
            logger.error(f"Error in {stage.path}. Pipeline halted.")
            sys.exit(1)
        logger.info(f"✅ Finished: {stage.path}")

    logger.info("🏁 Workflow Complete!")
    logger.info(f"📁 Reports saved to: {os.path.join(ROOT_DIR, 'data', 'logs', mode)}")


if __name__ == "__main__":
    run_workflow()
