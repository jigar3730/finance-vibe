# The Finance Vibe Resurrection Prompt

Session-bootstrap context. Keep this aligned with the live orchestrator and
scorecards. Handbook of record: [`QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md),
[`coiled_cobra_rubric.md`](../handbook/coiled_cobra_rubric.md),
[`scoring_logic.md`](../handbook/scoring_logic.md),
[`trade_plan_calculations.md`](../handbook/trade_plan_calculations.md).

Last aligned with the code: 2026-10-01 (Coiled Cobra rubric v4.0, `--as-of` replay).

---

I am working on the Finance Vibe project. It is a Python stock-signal
pipeline with three CLI profiles: `weekly` (default, 10y × 1wk), `daily`
(5y × 1d), and `high_beta` (daily OHLCV, same calibration as `daily`, its own
log silo). It generates ranked *signals*, not options trades.

Key architecture:

- Code: `src/finance_vibe/`; Flask UI `app.py` (+ `docs_routes.py`), templates in `templates/`
- Raw: `data/raw/{weekly|daily}/`
- Logs: `data/logs/{weekly|daily|high_beta}/`
- Orchestrator: `run_vibe.py` (`--mode`, `--reuse-raw`, `--as-of YYYY-MM-DD`)

Live `run_vibe.py` chain: wipe raw (unless `--reuse-raw`) → `ticker_provider`
→ `data_ingestor` → `analysis_engine` (Vibe report) → `coiled_cobra` (all
modes) → `breakout_scanner` → `trade_planner` → `trade_plan_helper`.
`swing_scanner.py` and `pipeline_backtest.py` were decommissioned on 2026-09-16.
`--as-of` replays a past date from the raw files on disk with no lookahead
(`config.cut_to_as_of`), stamps outputs with that date, uses a strict
trade-plan lookup, and skips ML ranking.

Key logic:

- **Vibe Score** (−10 to +10) in `analysis_engine.py`: Trend ±4, Momentum ±2,
  MomentumDecay −1, Timing −2..+2, CCI −2..+1, RSI caps, Persistence −2.
  Indicators: SMA20/50, MACD 12/26/9 histogram + 9-EMA of the hist, RSI14 +
  SMA10, CCI20 MAD. Informational only: Coiled Cobra does not gate on it, but
  it imports `load_benchmark_frame` / `relative_strength` /
  `check_coiled_cobra_market_gate` from the same module.
- **Coiled Cobra v4.0** (`coiled_cobra.py`, `config.RUBRIC_VERSION = "4.0"`):
  Stage 1 hard gates: A trend template (Close > EMA30w > EMA40w, EMA40w
  rising 8w), B market gate (Close ≥ 0.90×EMA50w, RS_13w > −15%), C coil
  integrity (vol-contraction and structure pillars each clear their own
  threshold), D breadth (Checks Met ≥ 4/6). Stage 2 is a 100-pt scorecard:
  volatility contraction (BBWidth percentile) 25, structure/EMA stack 20, RS
  vs QQQ 20, volume shelf 15, overhead clearance 10, RVOL trigger 10; MACD is
  a −8 directional penalty, not a pillar. Pass ≥ 70, Grade A ≥ 85; tiers
  Actionable / Watchlist. Daily periods are weekly × 5. Names need 160
  weeks of history for a full score.
- **Breakout scanner** (`breakout_scanner.py`): research-only, state-first
  classifier (PRE_BREAKOUT / BREAKOUT_CONFIRMED / FAILED_BREAKOUT /
  DEVELOPING / WATCH) plus a 100-pt readiness score with factor scores, using
  D/W/M resampled frames. Not consumed by the planner. The 2026-10-01
  `breakout_experiment.py` walk-forward found no status or score cut that beats
  random entries (readiness IC ≈ 0), so it is descriptive only.
- **Planner** (`trade_planner.py`): Cobra rows → entry `max(Fib 78.6%, Close −
  0.25×ATR)`, tightest of 10-bar swing low / 1.5×ATR / 5% Close stop, 2R/3R
  targets. Legacy swing geometry stays only as a fallback in
  `calculate_stock_levels`. No options output.
- **Helper** (`trade_plan_helper.py`): drops risk > 5% of Close, Checks Met
  < `MIN_CHECKS_MET/N_SCORED_PILLARS` (4/6), and R:R T1 < 2.0. Ranks by EV =
  R:R T2 × Score (×1.25 coil propensity). `ML_Pred_Return` is used only if
  `config.ML_RANKING_ENABLED` (default **off**) and every row has a prediction.
- **Backtest** (`coiled_cobra_backtest.py --backfill | --backtest`): same
  engine and planner geometry, `trade_simulator.simulate_trade`, rich research
  columns, `Rubric_Version` stamped on every CSV.
- **ML** (`coiled_cobra_ml_training.py`, `ml_ranker.py`): `FEATURE_COLS` =
  Score + four pct-from distances + ATR_Pct; `TARGET_COL` =
  `Forward_Return_2w`; rolling 26w/26w split with a 2w embargo; weekly-only;
  refuses CSVs with a mismatched rubric version. `MODEL_PARAMS` depth 4 /
  lr 0.01 / 400 trees / 0.8 bagging.
- **Research harnesses** (read-only, pre-registered, lockbox-gated):
  `coiled_cobra_ml_walkforward.py`, `coiled_cobra_ml_experiment.py`,
  `coiled_cobra_leader_experiment.py`. The 2026-09-19 leader experiment found
  no leader or relaxed-gate variant that beat the baseline or the random
  control, so there is no leader track.

Working rule: validate any gate or threshold change with a staged walk-forward
backtest, not judgment alone, and bump `RUBRIC_VERSION` when it changes
`Score` or qualification.

Environment: `PYTHONPATH=src` (or `/app/src` in Docker). The Docker default
command runs the dashboard on port 5000. Docs UI at `/docs/`, breakout
dashboard at `/breakout`. Tests: `python -m pytest tests/ -q`.

Current Task: [Insert your new question here].
