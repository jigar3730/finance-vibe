# The Finance Vibe Resurrection Prompt

Session-bootstrap context. Keep this aligned with the live orchestrator and
scorecards. Handbook of record: [`QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md),
[`coiled_cobra_rubric.md`](../handbook/coiled_cobra_rubric.md),
[`swing_setup.md`](../handbook/swing_setup.md).

---

I am working on the Finance Vibe project. It is a Python stock-analysis
pipeline with three CLI profiles: `weekly` (default, 10y × 1wk), `daily`
(5y × 1d), and `high_beta` (daily OHLCV + isolated swing profile / log silo).

Key architecture:

- Code: `src/finance_vibe/`
- Raw: `data/raw/{weekly|daily}/`
- Logs: `data/logs/{weekly|daily|high_beta}/`
- Orchestrator: `run_vibe.py` (`--mode`, `--reuse-raw`)

Live `run_vibe.py` chain: wipe raw (unless `--reuse-raw`) → `ticker_provider`
→ `data_ingestor` → `swing_scanner` → `coiled_cobra` (skipped on `high_beta`)
→ `trade_planner` → `trade_plan_helper`. `analysis_engine.py` is **commented
out** of the orchestrator; it still powers the swing soft vibe gate and
`pipeline_backtest.py`.

Key logic:

- **Vibe Score** (−10 to +10) in `analysis_engine.py`: Trend ±4, Momentum ±2,
  MomentumDecay −1, Timing −2..+2, CCI −2..+1, RSI caps, Persistence −2.
  Indicators: SMA20/50, MACD 12/26/9 histogram + 9-EMA of the hist, RSI14 +
  SMA10, CCI20 MAD.
- **Quality swing** (`swing_scanner.py`): EMA20/50/100 pullback, RSI bands,
  MACD hist early turn + 20-bar std cap, 10-bar structure, next-bar confirm.
  Geometry via `config.get_swing_params`: stop cap **1.5×ATR**, weekly T1/T2
  **1.25/2.25 ATR**, daily **0.85/1.6 ATR**, high_beta **2R/3R** and reject
  risk outside **[0.5, 1.5] ATR**. Daily/high_beta soft vibe ≥ 5; shorts off.
- **Coiled Cobra v3.1** (`coiled_cobra.py`): 100-pt coil → expansion card
  (shelf 20, coil 20, MACD squeeze 15, RS 15, structure 15, RVOL 10,
  overhead 5). Market Gate fails only on Close < EMA50 or RS 63d < 0.
  Extension / Fib / low RVOL are scorecard items, not binary drops. Pass ≥ 70.
- **Planner**: swing path uses `compute_swing_levels`; cobra path uses Fib
  78.6% entry floor + 2R/3R. Helper drops risk > 5% of Close, checklist < 5/7,
  R:R T1 < 2.0; ranks by EV / `ML_Pred_Return`.
- **ML**: `FEATURE_COLS` = Score + four pct-from distances + ATR_Pct;
  `TARGET_COL` = `Forward_Return_2w`; rolling 26w/26w split;
  `MODEL_PARAMS` depth 4 / lr 0.01 / 400 trees / 0.8 bagging; soft rank via
  `ml_ranker.py`.

Environment: `PYTHONPATH=src` (or `/app/src` in Docker). Docs UI at `/docs/`.

Current Task: [Insert your new question here].
