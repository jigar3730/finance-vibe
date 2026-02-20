import subprocess
import sys


def run_workflow():
    # Added analysis_engine_local.py to the sequence
    scripts = [
        "src/finance_vibe/ticker_provider.py",
        "src/finance_vibe/data_ingestor.py",
        "src/finance_vibe/analysis_engine.py",
        "src/finance_vibe/analysis_engine_local.py"
    ]

    print("🚀 Starting Finance-Vibe Pipeline...\n")

    for script in scripts:
        print(f"🔹 Running: {script}...")
        try:
            # check=True ensures the master script stops if any sub-script fails
            subprocess.run([sys.executable, script], check=True)
            print(f"✅ Finished: {script}\n")
        except subprocess.CalledProcessError:
            print(f"❌ Error in {script}. Pipeline halted.")
            sys.exit(1)  # Exit the master script on failure

    print("🏁 Workflow Complete!")
    print("📁 Reports saved to data/logs/")


if __name__ == "__main__":
    run_workflow()
