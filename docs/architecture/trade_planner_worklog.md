# Trade Planner — Work Log and Current Reference

Current reference for `trade_planner.py` / `trade_plan_helper.py`, followed by
historical notes from initial development (2026-02-26).

---

## Current implementation

### Role in pipeline

`trade_planner.py` is step 6 of `run_vibe.py`, after `coiled_cobra.py` and
`breakout_scanner.py`. It reads **today's** `coiled_cobra_setups_<date>.csv`
from `config.get_log_dir(mode)` and writes `trade_plan_<date>.csv`.
`trade_plan_helper.py` (step 7) applies guardrails and ranks survivors into
`trade_plan_clean_<date>.csv`.

It is a signal-ranking stage, not an execution planner. Levels are
informational stock-level context. It produces no options or LEAPS output.
The breakout scanner's output is **not** an input.

```bash
python src/finance_vibe/trade_planner.py weekly
python src/finance_vibe/trade_planner.py daily
python src/finance_vibe/trade_planner.py high_beta
python src/finance_vibe/trade_plan_helper.py weekly
python src/finance_vibe/trade_planner.py weekly --as-of 2025-11-07
```

### Paths

| File | Location |
| --- | --- |
| Cobra input | `data/logs/{mode}/coiled_cobra_setups_<YYYY-MM-DD>.csv` |
| Trade plan output | `data/logs/{mode}/trade_plan_<YYYY-MM-DD>.csv` |
| Cleaned plan | `data/logs/{mode}/trade_plan_clean_<YYYY-MM-DD>.csv` |

The planner uses **today's dated file only**, so last week's hits are never
silently reused. A zero-setup day writes a header-only CSV so the helper does
not crash. `high_beta` has its own log silo, and Coiled Cobra runs in every
mode, including `high_beta`.

### Input columns

Shared `config.SETUP_ROW_COLUMNS`, including `Source`, `Mode`, `Swing Low` /
`Swing High`, Fib levels, `Score` / `Grade` / `Tier` / `Checks Met`, `RVOL`,
`Market Gate`, and optional `ML_Pred_Return` / `ML_Rank`. A missing `Source`
column defaults to `coiled_cobra`.

### Stock level formulas (`calculate_stock_levels`)

**Coiled Cobra** (`Source` in `{coiled_cobra, cobra}`, `SETUP_LONG`, Fib 78.6% present):

| Level | Formula |
| --- | --- |
| Entry | `max(Fib 78.6%, Close − 0.25×ATR)` |
| Stop | tightest of: 10-bar swing low − 0.25×ATR, entry − 1.5×ATR, entry − 5% of Close; capped at entry − 0.25×ATR |
| T1 / T2 | **2R / 3R** of entry−stop risk |

`calculate_stock_levels()` is also used by `coiled_cobra_backtest.py` and
`coiled_cobra_leader_experiment.py`, so live, backtest, and experiment
geometry match.

**Legacy fallback.** Rows that miss the Cobra branch fall back to
`config.compute_swing_levels` / `get_swing_params` (the quality-swing
geometry left over from the decommissioned `swing_scanner.py`). The function
still returns an option side and delta band for signature compatibility, but
neither is written to any output.

Full math: [`trade_plan_calculations.md`](../handbook/trade_plan_calculations.md).

### Cleaned output (`trade_plan_helper.py`)

- Direction-aware `Risk Per Share`, `R:R T1`, `R:R T2`
- Drop risk > 5% of Close (`config.MAX_RISK_PCT_OF_CLOSE`), cobra
  `Checks Met` below coiled_cobra's Gate D breadth floor
  (`MIN_CHECKS_MET/N_SCORED_PILLARS`, currently 4/6), or R:R T1 < 2.0
- `Expected Value = R:R T2 × Score`; `Priority = Expected Value × propensity`
  (×1.25 for Cobra rows or rows with risk ≤ 3% of Close; every live row is
  Cobra, so in practice the boost is uniform and ranking follows Expected Value)
- `ML_Pred_Return` replaces Score in the priority only when
  `config.ML_RANKING_ENABLED` is on (default **off**) and every row has a
  prediction
- Prefers today's plan and falls back to the newest dated `trade_plan_*.csv`.
  With `--as-of` the lookup is strict, with no fallback.

### Offline validation

`coiled_cobra_backtest.py --backtest` reuses `calculate_stock_levels()` on
historical Cobra signals and simulates fills with
`trade_simulator.simulate_trade`. Stock simulation only, with no options P&L.
See [`backtest_and_backfill.md`](backtest_and_backfill.md).

---

## Historical work log (2026-02-26)

### Objective

Build a systematic trade planner from swing scanner output: stock entry/stop/targets
and options/LEAPS metadata per signal.

### Completed at the time

1. **Scanner integration** — MACD momentum filter tightened (two-bar histogram slope
   + std-dev overextension cap vs older weak trigger).
2. **Pullback zone** — price within 2% of EMA20; EMA50 slope required.
3. **RSI bands** — asymmetric long vs short; daily long floor later added (40 vs 45 weekly).
4. **Trade planner skeleton** — ATR targets, EMA-based entry/stop, CALL/PUT by direction.
5. **File handling** — auto-detect latest scanner CSV; mode-specific log directories.

### Still open

- Position sizing / % portfolio risk per trade
- Portfolio constraints (max open trades, sector caps)
- ~~LEAPS strike selection~~: options output was removed (2026-09-06); the project generates signals only
- ~~True R-multiple targets~~: resolved, since Coiled Cobra (the only live source) uses 2R/3R

---

## Historical sample output (pre-2026-09, swing + LEAPS era)

```csv
Symbol,Setup Type,Stock Entry,Stock Stop,Target 1,Target 2,LEAPS Type,LEAPS Expiry Min,LEAPS Expiry Max,Suggested Delta,Risk Notes
SPY,SETUP_LONG,691.16,681.57,699.13,707.1,CALL,Feb-2027,Feb-2028,0.65 – 0.8,Stop based on EMA50; adjust if invalidated
```

This format is retired. Current `trade_plan_<date>.csv` files have no LEAPS or
options columns. The export schema is `_PLAN_EXPORT_COLUMNS` in `trade_planner.py`.
