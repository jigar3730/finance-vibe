# Trade Planner — Work Log and Current Reference

Historical notes from initial development (2026-02-26), updated to match the
current codebase.

---

## Current implementation

### Role in pipeline

`trade_planner.py` runs after the swing scanner and Coiled Cobra in
`run_vibe.py`. It merges **today's** `swing_setups_<date>.csv` and
`coiled_cobra_setups_<date>.csv` from `config.get_log_dir(mode)` and writes
`trade_plan_<date>.csv`. `trade_plan_helper.py` applies guardrails and ranks
survivors into `trade_plan_clean_<date>.csv`.

```bash
python src/finance_vibe/trade_planner.py weekly
python src/finance_vibe/trade_planner.py daily
python src/finance_vibe/trade_planner.py high_beta
python src/finance_vibe/trade_plan_helper.py weekly
```

### Paths

| File | Location |
| --- | --- |
| Swing input | `data/logs/{mode}/swing_setups_<YYYY-MM-DD>.csv` |
| Cobra input | `data/logs/{mode}/coiled_cobra_setups_<YYYY-MM-DD>.csv` |
| Trade plan output | `data/logs/{mode}/trade_plan_<YYYY-MM-DD>.csv` |
| Cleaned plan | `data/logs/{mode}/trade_plan_clean_<YYYY-MM-DD>.csv` |

The planner uses **today's dated files only** (no silent reuse of last week's
hits). `high_beta` has its own log silo. Coiled Cobra is skipped in that mode.

### Input columns

Shared `config.SETUP_ROW_COLUMNS`, including `Source`, `Mode`, `Swing Low` /
`Swing High`, Fib levels, `Score` / `Grade` / `Checks Met`, `RVOL`,
`Market Gate`, and optional `ML_Pred_Return` / `ML_Rank`. Row `Mode` is
authoritative for swing geometry.

### Stock level formulas (`calculate_stock_levels`)

**Quality swing** — `config.compute_swing_levels` / `get_swing_params`:

| Level | Weekly | Daily | High-beta |
| --- | --- | --- | --- |
| Entry (long) | `max(EMA20, Close − 0.25×ATR)` | same | same |
| Stop | dual-constraint vs **1.5×ATR** + 5% Close | same | same, then reject risk ∉ **[0.5, 1.5] ATR** |
| T1 / T2 | **1.25 / 2.25 ATR** | **0.85 / 1.6 ATR** | **2R / 3R** |

**Coiled Cobra** (`Source` in `{coiled_cobra, cobra}` and Fib 78.6% present):

| Level | Formula |
| --- | --- |
| Entry | `max(Fib 78.6%, Close − 0.25×ATR)` |
| Stop | local 10-bar swing low vs 1.5×ATR vs 5% Close (tightest) |
| T1 / T2 | **2R / 3R** of entry−stop risk |

Full math: [`trade_plan_calculations.md`](../handbook/trade_plan_calculations.md).

### Options metadata

| Mode | Contract column | Expiry window | Delta |
| --- | --- | --- | --- |
| `weekly` | LEAPS Type (CALL/PUT) | 12–24 months forward | Long: 0.65–0.80, Short: −0.80 to −0.65 |
| `daily`, `high_beta` | Options Type (CALL/PUT) | 1–3 months forward | Same delta bands |

### Cleaned output (`trade_plan_helper.py`)

- Direction-aware `Risk Per Share`, `R:R T1`, `R:R T2`
- Drop risk > 5% of Close, cobra `Checks Met` below coiled_cobra's Gate D
  breadth floor (`MIN_CHECKS_MET/N_SCORED_PILLARS`, currently 4/6), or R:R T1 < 2.0
- Rank by `Expected Value = R:R T2 × Score`, or `R:R T2 × max(ML_Pred_Return, 0)`
  when the ML column is populated; ×1.25 propensity for cobra / tight-risk rows
- Prefers today's plan; falls back to the newest dated `trade_plan_*.csv`

### Offline validation

`pipeline_backtest.py` reuses `calculate_stock_levels()` on historical setups (with
an optional macro Vibe Score gate). Stock simulation only — no options P&L.

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
- LEAPS strike selection from delta (metadata only today)
- Portfolio constraints (max open trades, sector caps)
- True R-multiple targets on **all** swing profiles (high_beta and Cobra already use 2R/3R; weekly/daily remain ATR offsets)

---

## Sample weekly output (illustrative)

```csv
Symbol,Setup Type,Stock Entry,Stock Stop,Target 1,Target 2,LEAPS Type,LEAPS Expiry Min,LEAPS Expiry Max,Suggested Delta,Risk Notes
SPY,SETUP_LONG,691.16,681.57,699.13,707.1,CALL,Feb-2027,Feb-2028,0.65 – 0.8,Stop based on EMA50; adjust if invalidated
```

Column names differ in **daily** mode (`Options Type`, `Options Expiry Min/Max`).
