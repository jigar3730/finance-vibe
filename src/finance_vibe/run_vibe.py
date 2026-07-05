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
        choices=["weekly", "daily"], 
        default="weekly", 
        help="Execution timeframe profile (default: weekly)"
    )
    args = parser.parse_args()
    mode = args.mode.lower()

    # 2. CLIMB TO ROOT
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    SRC_DIR = os.path.join(ROOT_DIR, "src")

    # 3. ENVIRONMENT SETUP
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")

    # 4. SCRIPT CONFIGURATION
    # We define which scripts accept the mode argument
    scripts_config = [
        {"path": "src/finance_vibe/ticker_provider.py", "pass_mode": False},
        {"path": "src/finance_vibe/data_ingestor.py", "pass_mode": True},
        {"path": "src/finance_vibe/analysis_engine.py", "pass_mode": True},
        {"path": "src/finance_vibe/swing_scanner.py", "pass_mode": True},
        {"path": "src/finance_vibe/trade_planner.py", "pass_mode": True},
        {"path": "src/finance_vibe/trade_plan_helper.py", "pass_mode": True},
    ]

    print(f"🚀 Starting Finance-Vibe Pipeline [{mode.upper()} MODE]...")
    print(f"📍 Project Root: {ROOT_DIR}\n")

    # Target only the active mode's subdirectory
    clean_raw_folder(ROOT_DIR, mode)

    for script in scripts_config:
        script_path = os.path.join(ROOT_DIR, script["path"])
        print(f"🔹 Running: {script['path']}...")
        
        # Build command array dynamically
        cmd = [sys.executable, script_path]
        if script["pass_mode"]:
            cmd.append(mode)  # Injected cleanly as a positional argument ('daily' or 'weekly')

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