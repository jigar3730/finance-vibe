# The Finance Vibe Resurrection Prompt

## Here is the refined Project Resurrection Prompt for you to save. It contains all the architectural decisions, math logic, and environment settings.


"I am working on the Finance Vibe project. It is a Python-based stock analysis pipeline using weekly (default) or daily timeframe profiles.

Key Architecture:

Structure: /src/finance_vibe/ (Logic), /data/raw/{mode}/ (CSVs), /data/logs/{mode}/ (Dated Reports).

Files: config.py (Central paths), ticker_provider.py (Universe), data_ingestor.py (YFinance), analysis_engine.py (Macro Vibe Score), swing_scanner.py (Tactical setups), trade_planner.py + trade_plan_helper.py (Execution), and run_vibe.py (Orchestrator).

Key Logic:

The Vibe Score: A -10 to +10 scale in analysis_engine.py weighting Trend (±4), Momentum (±2), Timing (±2), and Risk Governors (RSI/CCI caps).

Tactical Layer: swing_scanner.py detects SETUP_LONG / SETUP_SHORT on EMA pullback rules with MACD momentum confirmation.

Robust CCI: Manual Mean Absolute Deviation (MAD) calculation with a 0.015 constant to prevent score explosions.

Neutral Zone: Mid-range scores map to WAIT / NO EDGE action labels.

Environment: VS Code Dev Container with Rainbow CSV and Excel Viewer extensions. PYTHONPATH is set to ./src.

Current Task: [Insert your new question here]."
