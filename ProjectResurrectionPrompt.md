# The Finance Vibe Resurrection Prompt

## Here is the refined Project Resurrection Prompt for you to save. It contains all the architectural decisions, math logic, and environment settings.


"I am working on the Finance Vibe project. It is a Python-based stock analysis pipeline using a 5-year weekly timeframe.

Key Architecture:

Structure: /src/finance_vibe/ (Logic), /data/raw/ (CSVs), /data/logs/ (Dated Reports).

Files: config.py (Central paths), ticker_provider.py (Universe), data_ingestor.py (YFinance), analysis_engine.py (Main Math), analysis_engine_local.py (Shadow Math), and run_vibe.py (Orchestrator).

Key Logic:

The Vibe Score: A -10 to +10 scale weighting Trend (±4), Momentum (±3), and Volatility (±3).

Robust CCI: We use a manual Mean Absolute Deviation (MAD) calculation with a 0.015 constant to prevent score explosions.

Neutral Zone: Scores between -3 and +3 are treated as 'CASH/WAIT'.

Environment: VS Code Dev Container with Rainbow CSV and Excel Viewer extensions. PYTHONPATH is set to ./src.

Current Task: [Insert your new question here]."