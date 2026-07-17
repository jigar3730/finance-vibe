"""Finance Vibe pipeline orchestrator.

Runs ingestion, macro scoring, tactical scanning, and trade plan generation
in sequence for a given timeframe profile (weekly or daily).
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
    parser = argparse.ArgumentParser(description="Finance-Vibe Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["weekly", "daily", "high_beta"],
        default="weekly",
        help="Execution profile (weekly, daily, or high_beta long-only single names)",
    )
    args = parser.parse_args()
    mode = args.mode.lower()

    # high_beta reads daily OHLCV but keeps its own swing profile + log silo.
    data_mode = "daily" if mode == "high_beta" else mode

    # 2. CLIMB TO ROOT
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    SRC_DIR = os.path.join(ROOT_DIR, "src")

    # 3. ENVIRONMENT SETUP
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")

    # 4. SCRIPT CONFIGURATION
    # "scope" selects the argument each stage receives:
    #   data    -> data timeframe (weekly/daily); shares raw data silo
    #   profile -> swing profile (weekly/daily/high_beta); drives geometry + logs
    # high_beta skips Coiled Cobra (LEAPS-oriented, not part of the long-only swing).
    scripts_config = [
        {"path": "src/finance_vibe/ticker_provider.py", "pass_mode": False, "scope": "data"},
        {"path": "src/finance_vibe/data_ingestor.py", "pass_mode": True, "scope": "data"},
        #{"path": "src/finance_vibe/analysis_engine.py", "pass_mode": True, "scope": "data"},
        {"path": "src/finance_vibe/swing_scanner.py", "pass_mode": True, "scope": "profile"},
        {"path": "src/finance_vibe/coiled_cobra.py", "pass_mode": True, "scope": "data",
         "skip_modes": ["high_beta"]},
        {"path": "src/finance_vibe/trade_planner.py", "pass_mode": True, "scope": "profile"},
        {"path": "src/finance_vibe/trade_plan_helper.py", "pass_mode": True, "scope": "profile"},
    ]

    print(f"🚀 Starting Finance-Vibe Pipeline [{mode.upper()} MODE]...")
    print(f"📍 Project Root: {ROOT_DIR}")
    if mode != data_mode:
        print(f"🧬 Data timeframe: {data_mode} | Swing profile: {mode}")
    print()

    # Clean the shared raw silo for the data timeframe.
    clean_raw_folder(ROOT_DIR, data_mode)

    for script in scripts_config:
        if mode in script.get("skip_modes", []):
            print(f"⏭️  Skipping {script['path']} for {mode} mode.\n")
            continue

        script_path = os.path.join(ROOT_DIR, script["path"])
        print(f"🔹 Running: {script['path']}...")

        # Data-scope stages receive the data timeframe; profile-scope stages
        # receive the swing profile so high_beta geometry/logs stay isolated.
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