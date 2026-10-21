# Finance Vibe documentation

Reference manual for the Finance Vibe pipeline: macro scoring, Coiled Cobra
coil scoring (primary signal engine), the descriptive breakout-readiness scan,
and walk-forward backtests.

Project overview and run commands stay in the root [`README.md`](../README.md).

---

## Handbook — scoring rules and calculations

| File | Contents |
| ---- | -------- |
| [`handbook/scoring_logic.md`](handbook/scoring_logic.md) | Macro Vibe Score (−10 to +10) specification |
| [`handbook/coiled_cobra_rubric.md`](handbook/coiled_cobra_rubric.md) | Coil → expansion 100-point scorecard |
| [`handbook/trade_plan_calculations.md`](handbook/trade_plan_calculations.md) | Live entry / stop / target math and helper ranking |

---

## Architecture — system design and operations

| File | Contents |
| ---- | -------- |
| [`architecture/operation_manual.md`](architecture/operation_manual.md) | SOP, scheduling, environment, troubleshooting |
| [`architecture/breakout_scanner.md`](architecture/breakout_scanner.md) | Breakout readiness scanner: states, status rules, 100-pt score, dashboard |
| [`architecture/backtest_and_backfill.md`](architecture/backtest_and_backfill.md) | Data backfill, signal archives, Cobra walk-forward simulation |

---

## Code of record (quick map)

| Concern | Module |
| ------- | ------ |
| Orchestrator (`--mode`, `--reuse-raw`, `--as-of`) | `src/finance_vibe/run_vibe.py` |
| Config, modes, as-of cut, rubric version | `src/finance_vibe/config.py` |
| Universe + ingest | `src/finance_vibe/ticker_provider.py`, `src/finance_vibe/data_ingestor.py`, `src/finance_vibe/daily_ingest.py` |
| Raw OHLCV loading | `src/finance_vibe/raw_data.py` |
| Indicators | `src/finance_vibe/indicators.py` |
| Macro score | `src/finance_vibe/analysis_engine.py` |
| Coiled Cobra rubric (primary signal engine) | `src/finance_vibe/coiled_cobra.py` |
| Breakout readiness (descriptive scan) | `src/finance_vibe/breakout_scanner.py` |
| Signal levels + ranking | `src/finance_vibe/trade_planner.py`, `src/finance_vibe/trade_plan_helper.py` |
| Cobra backtest + forward returns | `src/finance_vibe/coiled_cobra_backtest.py` |
| Generic trade simulator | `src/finance_vibe/trade_simulator.py` |
| Dashboard + docs UI | `src/finance_vibe/app.py`, `src/finance_vibe/docs_routes.py` |
