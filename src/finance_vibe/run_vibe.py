"""Finance Vibe pipeline orchestrator.

Coiled Cobra v2.1 is the live path: ingest → coil scan → expansion trade plan.
Quality-swing and analysis_engine remain in the repo for offline studies only.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import os

def clean_raw_folder(root_dir, mode):
    """Remove all files in data/raw/{mode}/ before a fresh ingestion run."""
    raw_dir = Path(root_dir) / "data" / "raw" / mode
    if not raw_dir.exists():
        print(f"⚠️ Raw '{mode}' folder does not exist. Skipping cleanup.")
        return

    for item in raw_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"❌ Failed to delete {item}: {e}")

    print(f"🧹 Raw '{mode}' folder cleaned.\n")

def run_workflow():
    """Parse CLI args and execute each pipeline stage as a subprocess."""
    parser = argparse.ArgumentParser(
        description="Finance-Vibe Coiled Cobra pipeline (coil → expansion)"
    )
    parser.add_argument(
        "--mode",
        choices=["weekly", "daily", "high_beta"],
        default="daily",
        help="daily Cobra scan (default), weekly, or high_beta ingest-only",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Skip wiping data/raw/{mode}/ so existing OHLCV is reused",
    )
    args = parser.parse_args()
    mode = args.mode.lower()
    data_mode = "daily" if mode == "high_beta" else mode

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    SRC_DIR = os.path.join(ROOT_DIR, "src")

    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")

    scripts_config = [
        {"path": "src/finance_vibe/ticker_provider.py", "pass_mode": False, "scope": "data"},
        {"path": "src/finance_vibe/data_ingestor.py", "pass_mode": True, "scope": "data"},
        {"path": "src/finance_vibe/coiled_cobra.py", "pass_mode": True, "scope": "data",
         "skip_modes": ["high_beta"]},
        {"path": "src/finance_vibe/trade_planner.py", "pass_mode": True, "scope": "profile",
         "skip_modes": ["high_beta"]},
        {"path": "src/finance_vibe/trade_plan_helper.py", "pass_mode": True, "scope": "profile",
         "skip_modes": ["high_beta"]},
        {"path": "src/finance_vibe/ai_notifier.py", "pass_mode": True, "scope": "profile",
         "skip_modes": ["high_beta"]},
    ]

    print(f"🚀 Starting Coiled Cobra Pipeline [{mode.upper()} MODE]...")
    print(f"📍 Project Root: {ROOT_DIR}")
    if mode != data_mode:
        print(f"🧬 Data timeframe: {data_mode} | Profile: {mode}")
    if mode == "high_beta":
        print("ℹ️  high_beta is ingest-only. Use pipeline_backtest.py for the swing study.\n")
    print()

    if args.keep_raw:
        print(f"📦 --keep-raw: leaving data/raw/{data_mode}/ in place.\n")
    else:
        clean_raw_folder(ROOT_DIR, data_mode)

    for script in scripts_config:
        if mode in script.get("skip_modes", []):
            print(f"⏭️  Skipping {script['path']} for {mode} mode.\n")
            continue

        script_path = os.path.join(ROOT_DIR, script["path"])
        print(f"🔹 Running: {script['path']}...")
        arg_mode = data_mode if script.get("scope") == "data" else mode
        cmd = [sys.executable, script_path]
        if script["pass_mode"]:
            cmd.append(arg_mode)

        try:
            subprocess.run(cmd, check=True, env=env, cwd=ROOT_DIR)
            print(f"✅ Finished: {script['path']}\n")
        except subprocess.CalledProcessError:
            print(f"❌ Error in {script['path']}. Pipeline halted.")
            sys.exit(1)

    print("🏁 Workflow Complete!")
    print(f"📁 Reports saved to: {os.path.join(ROOT_DIR, 'data', 'logs', mode)}")

if __name__ == "__main__":
    run_workflow()
