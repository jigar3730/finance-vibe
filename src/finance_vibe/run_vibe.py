"""Finance Vibe pipeline orchestrator.

Runs ingestion, macro scoring, tactical scanning, and trade plan generation
in sequence for a given timeframe profile (weekly or daily).

``--as-of YYYY-MM-DD`` replays a past date from the raw data already on disk
(implies ``--reuse-raw``): every stage sees only bars completed on that date and
writes its usual dated files stamped with it, e.g.
``python src/finance_vibe/run_vibe.py --as-of 2025-11-07``.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)


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


def run_workflow():
    """Parse CLI args and execute each pipeline stage as a subprocess."""
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
    args = parser.parse_args()
    mode = args.mode.lower()

    data_mode = mode

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

    # 4. SCRIPT CONFIGURATION
    # "scope" selects the argument each stage receives:
    #   data    -> data timeframe (weekly/daily); shares raw data silo
    #   profile -> signal profile (weekly/daily); drives geometry + logs
    # Coiled Cobra is the primary signal engine and runs for every profile.
    scripts_config = [
        {
            "path": "src/finance_vibe/ticker_provider.py",
            "pass_mode": False,
            "scope": "data",
        },
        {
            # Daily has its own ingest wrapper (drops today's in-progress bar).
            "path": (
                "src/finance_vibe/daily_ingest.py"
                if data_mode == "daily"
                else "src/finance_vibe/data_ingestor.py"
            ),
            "pass_mode": data_mode != "daily",
            "scope": "data",
        },
        {
            "path": "src/finance_vibe/analysis_engine.py",
            "as_of": True,
            "pass_mode": True,
            "scope": "data",
        },
        {
            "path": "src/finance_vibe/coiled_cobra.py",
            "as_of": True,
            "pass_mode": True,
            "scope": "profile",
        },
        {
            "path": "src/finance_vibe/breakout_scanner.py",
            "as_of": True,
            "pass_mode": True,
            "scope": "profile",
        },
        {
            "path": "src/finance_vibe/trade_planner.py",
            "as_of": True,
            "pass_mode": True,
            "scope": "profile",
        },
        {
            "path": "src/finance_vibe/trade_plan_helper.py",
            "as_of": True,
            "pass_mode": True,
            "scope": "profile",
        },
    ]

    logger.info(f"🚀 Starting Finance-Vibe Pipeline [{mode.upper()} MODE]...")
    logger.info(f"📍 Project Root: {ROOT_DIR}")
    if mode != data_mode:
        logger.info(f"🧬 Data timeframe: {data_mode} | Swing profile: {mode}")
    if as_of:
        logger.info(f"⏪ AS-OF replay: {as_of} (bars completed on/before this date only)")
        if data_mode == "weekly" and date.fromisoformat(as_of).weekday() != 4:
            logger.info("   Note: as-of is not a Friday, so the week in progress is excluded.")
        logger.info(
            "   Uses today's ticker list and split/dividend-adjusted prices; ML ranking is skipped."
        )

    skip_ingest = {
        "src/finance_vibe/ticker_provider.py",
        "src/finance_vibe/data_ingestor.py",
        "src/finance_vibe/daily_ingest.py",
    }

    # Clean the shared raw silo unless the caller wants to reuse existing OHLCV.
    if reuse_raw:
        logger.info(f"♻️  Reusing existing raw files in data/raw/{data_mode}/")
    else:
        clean_raw_folder(ROOT_DIR, data_mode)

    for script in scripts_config:
        if mode in script.get("skip_modes", []):
            logger.info(f"⏭️  Skipping {script['path']} for {mode} mode.")
            continue
        if reuse_raw and script["path"] in skip_ingest:
            logger.info(f"⏭️  Skipping {script['path']} (--reuse-raw).")
            continue

        script_path = os.path.join(ROOT_DIR, script["path"])
        logger.info(f"🔹 Running: {script['path']}...")

        # Data-scope stages receive the data timeframe; profile-scope stages
        # receive the swing profile.
        arg_mode = data_mode if script.get("scope") == "data" else mode

        cmd = [sys.executable, script_path]
        if script["pass_mode"]:
            cmd.append(arg_mode)
        if as_of and script.get("as_of"):
            cmd += ["--as-of", as_of]

        try:
            subprocess.run(cmd, check=True, env=env, cwd=ROOT_DIR)
            logger.info(f"✅ Finished: {script['path']}")
        except subprocess.CalledProcessError:
            logger.error(f"Error in {script['path']}. Pipeline halted.")
            sys.exit(1)

    logger.info("🏁 Workflow Complete!")
    logger.info(f"📁 Reports saved to: {os.path.join(ROOT_DIR, 'data', 'logs', mode)}")


if __name__ == "__main__":
    run_workflow()
