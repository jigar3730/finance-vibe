# Trade Planner — Work Log and Current Reference

Historical notes from initial development (2026-02-26), updated to match the
current codebase.

---

## Current implementation

### Role in pipeline

`trade_planner.py` is **step 5** in `run_vibe.py`. It reads the latest
`swing_setups_*.csv` for the active mode and writes `trade_plan_<date>.csv`.
`trade_plan_helper.py` (step 6) produces `trade_plan_clean_<date>.csv` with R:R columns.

```bash
python src/finance_vibe/trade_planner.py weekly
python src/finance_vibe/trade_plan_helper.py weekly
```

Daily mode uses the same flow with `daily` instead of `weekly`.

### Paths

| File | Location |
| --- | --- |
| Scanner input | `data/logs/{mode}/swing_setups_<YYYY-MM-DD>.csv` |
| Trade plan output | `data/logs/{mode}/trade_plan_<YYYY-MM-DD>.csv` |
| Cleaned plan | `data/logs/{mode}/trade_plan_clean_<YYYY-MM-DD>.csv` |

The planner auto-selects the **latest** `swing_setups_*.csv` in the mode directory
(by date in the filename). Output date matches the scanner file date.

### Input columns (from swing scanner)

`Symbol`, `Setup Type`, `Close`, `EMA20`, `EMA50`, `RSI`, `ATR`, `Notes`

### Stock level formulas (`calculate_stock_levels`)

**SETUP_LONG**

| Level | Formula |
| --- | --- |
| Stock Entry | `max(EMA20, Close − 0.25 × ATR)` |
| Stock Stop | `EMA50 − 0.5 × ATR` |
| Target 1 | `Entry + 1 × ATR` |
| Target 2 | `Entry + 2 × ATR` |

**SETUP_SHORT**

| Level | Formula |
| --- | --- |
| Stock Entry | `min(EMA20, Close + 0.25 × ATR)` |
| Stock Stop | `EMA50 + 0.5 × ATR` |
| Target 1 | `Entry − 1 × ATR` |
| Target 2 | `Entry − 2 × ATR` |

**Important:** Target 1 is **+1 ATR from entry**, not necessarily **1R** relative to
stop distance. Because the stop is anchored to EMA50 (often wider than 1 ATR below
entry), actual R:R at Target 1 is frequently **less than 1.0** — see `trade_plan_helper`
R:R columns and `pipeline_backtest.py` results.

### Options metadata

| Mode | Contract column | Expiry window | Delta |
| --- | --- | --- | --- |
| `weekly` | LEAPS Type (CALL/PUT) | 12–24 months forward | Long: 0.65–0.80, Short: −0.80 to −0.65 |
| `daily` | Options Type (CALL/PUT) | 1–3 months forward | Same delta bands |

Expiry labels: `LEAPS Expiry Min/Max` (weekly) or `Options Expiry Min/Max` (daily).

All plans include `Risk Notes`: *Stop based on EMA50; adjust if invalidated*.

### Cleaned output (`trade_plan_helper.py`)

Adds:

- `Risk Per Share` = `Stock Entry − Stock Stop`
- `R:R T1`, `R:R T2` = reward to target divided by risk per share
- `Delta Min`, `Delta Max` parsed from `Suggested Delta`

Expects today's `trade_plan_<date>.csv` when run standalone.

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
- True 1R/2R targets relative to stop distance (vs current ATR-offset targets)

---

## Sample weekly output (illustrative)

```csv
Symbol,Setup Type,Stock Entry,Stock Stop,Target 1,Target 2,LEAPS Type,LEAPS Expiry Min,LEAPS Expiry Max,Suggested Delta,Risk Notes
SPY,SETUP_LONG,691.16,681.57,699.13,707.1,CALL,Feb-2027,Feb-2028,0.65 – 0.8,Stop based on EMA50; adjust if invalidated
```

Column names differ in **daily** mode (`Options Type`, `Options Expiry Min/Max`).
