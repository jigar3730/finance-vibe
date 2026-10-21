# Backfill & Backtest Guide

Offline validation for Finance Vibe. This document covers **data backfill** (getting enough OHLCV history), **signal backfill** (historical setup archives), and **walk-forward backtests** (simulating trades on that history).

Coiled Cobra is the only signal engine with a backtest; generic fill/stop/target
simulation lives in `trade_simulator.py`. Neither module is part of `run_vibe.py`. They read from `data/raw/` and write under `data/logs/`.

---

## Concepts

| Term | Meaning |
| ---- | ------- |
| **Data backfill** | Download historical OHLCV into `data/raw/{weekly\|daily}/` via `data_ingestor.py` |
| **Signal backfill** | Scan every historical bar and archive setups (Coiled Cobra: `--backfill`) |
| **Walk-forward backtest** | At each bar, detect a setup using only past data, plan levels, simulate forward fills/exits |
| **Mode / profile** | CLI mode maps to a data timeframe + signal profile (see below) |

### Mode map

| CLI mode | Raw data | Signal profile | Log silo |
| -------- | -------- | -------------- | -------- |
| `weekly` | `data/raw/weekly/` (10y × 1wk) | weekly | `data/logs/weekly/` |
| `daily` | `data/raw/daily/` (5y × 1d) | daily | `data/logs/daily/` |

---

## Prerequisites

### 1. Environment

Docker is the recommended runtime (matches production deps):

```bash
docker exec -it finance_vibe bash
cd /app
export PYTHONPATH=/app/src   # if not already set in the image
```

Locally:

```bash
cd /opt/stacks/finance-vibe
export PYTHONPATH=src
python -m pip install -r requirements.txt
```

### 2. Active ticker universe

```bash
python src/finance_vibe/ticker_provider.py
# writes data/active_tickers.csv
```

Backtests default to this list unless you pass `--tickers`.

### 3. Raw OHLCV (data backfill)

```bash
# Weekly (10y, 1wk) — weekly Coiled Cobra
python src/finance_vibe/data_ingestor.py weekly

# Daily (5y, 1d) — daily profile
python src/finance_vibe/data_ingestor.py daily
```

**Important:** `run_vibe.py` **clears** `data/raw/{mode}/` before ingestion. For offline backtests, prefer running `data_ingestor.py` alone so existing longer-history files (e.g. leftover `*_2y_1d.csv` next to new `*_5y_1d.csv`) are not wiped unless you intend a full refresh.

#### File naming

```
data/raw/{mode}/<TICKER>_<period>_<interval>.csv
# examples:
#   AAPL_10y_1wk.csv
#   QQQ_5y_1d.csv
#   PLTR_5y_1d.csv
```

Required columns after ingest: `Date, Open, High, Low, Close, Volume` (`config.REQUIRED_OHLCV`). Minimum usable rows: 60 (`config.MIN_SAVE_ROWS`).

#### Multiple files per ticker

Keep **one file per symbol** per raw silo. The scanners iterate over every `*.csv` in the directory, so a leftover `*_2y_1d.csv` next to `*_5y_1d.csv` gets scored twice. After a period change, clear the silo and re-ingest.

#### Benchmarks

Coiled Cobra's relative-strength pillar and market gate need **QQQ** (and SPY) in the raw silo for the mode being tested. The backtest also records QQQ regime features (`QQQ_Pct_From_EMA50`, `QQQ_Ret_13w`) and `Excess_Return_2w` against QQQ.

```bash
ls data/raw/weekly/QQQ_*_1wk.csv data/raw/daily/QQQ_*_1d.csv
```

### 4. Sync code into the container (when developing on the host)

```bash
docker cp src/finance_vibe/. finance_vibe:/app/src/finance_vibe/
docker cp tests/. finance_vibe:/app/tests/
```

Or rebuild the image when you want a durable bake-in.

---

## Coiled Cobra backfill & backtest

**Module:** `src/finance_vibe/coiled_cobra_backtest.py`

Uses the same `evaluate_coiled_cobra()` engine as the live scanner on every eligible historical bar, the live Fib-anchored geometry (`trade_planner.calculate_stock_levels`, `Source=coiled_cobra`), and `trade_simulator.simulate_trade` (full exit at the first stop or target, no scale-out, no slippage). Tickers run in parallel worker processes.

Typically run on **weekly** data (coil → expansion horizon). Every output CSV carries a `Rubric_Version` column (`config.RUBRIC_VERSION`, currently `4.0`), so results from different rubric versions are never mixed.

### Signal backfill

Scans every eligible historical bar and archives Coiled Cobra setups (no trade simulation):

```bash
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill --tickers SPY,QQQ,IWM
```

**Output:** `data/logs/{mode}/coiled_cobra_backfill_{YYYY-MM-DD}.csv`

Columns include Symbol, Date, Setup Type, Close, EMAs, ATR, Swing Low, Fib levels, Score, Grade, Tier, Checks Met, Source, RS 63d, RVOL, Market Gate, BBWidth Pctile, Rubric_Version.

Useful for:

- Counting historical signal frequency
- Auditing grade distribution
- Ad-hoc research without re-scanning

### Walk-forward backtest

```bash
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest --tickers SPY,QQQ
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest \
  --entry-valid 4 --max-hold 12
```

If neither `--backfill` nor `--backtest` is passed, **backtest** is the default.

**Output:** `data/logs/{mode}/coiled_cobra_backtest_trades_{YYYY-MM-DD}.csv`

Outcomes: `no_fill`, `stopped`, `target1`, `target2`, `expired` with a single `R Multiple` (not blended).

**Causality and speed.** Each bar is scored as if it were the last one
(`df[:t+1]`). Indicators come from `coiled_cobra.macro_indicator_history`,
computed once per ticker. Its row t is bit-identical to recomputing on `df[:t+1]`,
including the pandas_ta ATR epsilon quirk, which would otherwise leak a later
flat bar backwards. Backtest and backfill both use it; it is about 3x faster
than per-bar recomputation, with byte-identical outputs.

#### Trade CSV columns

| Zone | Columns |
| ---- | ------- |
| Identity | `Symbol`, `Signal Date`, `Setup Type` |
| Pre-signal attributes | `Score`, `Grade`, `RVOL`, `Market Gate`, `Tier`, `Checks_N`, `RS_63d`, `BBWidth_Pctile`, pillar sub-scores (`Part_*`), causal QQQ regime `QQQ_Pct_From_EMA50`, `QQQ_Ret_13w` |
| Execution | `Stock Entry`, `Stock Stop`, `Target 1`, `Target 2`, `Outcome`, `Exit Date`, `Exit Price`, `R Multiple` |
| Forward returns | `Forward_Return_2w`, `Excess_Return_2w` (vs QQQ), `Forward_Return_5w`, `Forward_Return_13w`, `Forward_Return_26w` |

`Forward_Return_{Nw}` is `(Close[t+N] − Close[t]) / Close[t]` when enough future bars exist; otherwise `None` / NaN.

### Live scanner vs historical modules

| Task | Command |
| ---- | ------- |
| Live Coiled Cobra (latest bar) | `python src/finance_vibe/coiled_cobra.py weekly` |
| Historical signal archive | `.../coiled_cobra_backtest.py weekly --backfill` |
| Historical trade simulation | `.../coiled_cobra_backtest.py weekly --backtest` |

---

## End-to-end workflows

### A. Weekly Coiled Cobra validation

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py weekly   # do NOT use run_vibe if you need to keep existing files

python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill --tickers SPY,QQQ,IWM
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
```

### B. Daily Coiled Cobra

```bash
python src/finance_vibe/data_ingestor.py daily
python src/finance_vibe/coiled_cobra_backtest.py daily --backtest --tickers PLTR,TSLA,HOOD,NVDA
```

`coiled_cobra_backtest.py` accepts the modes in `config.TIMEFRAME_PROFILES` (`weekly`, `daily`).

### C. Docker one-liners

```bash
docker exec finance_vibe sh -c \
  'cd /app && PYTHONPATH=/app/src python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill --tickers SPY,QQQ'

docker exec finance_vibe sh -c \
  'cd /app && PYTHONPATH=/app/src python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest'
```

### D. Unit tests for simulation contracts

```bash
python -m pytest tests/test_trade_simulator.py tests/test_coiled_cobra_backtest.py -q
python -m pytest tests/ -q   # full suite
```

---

## Limitations (read before trusting numbers)

| Limitation | Detail |
| ---------- | ------ |
| Stock-only | No options premium, delta, or theta P&L |
| Daily OHLC order | Intrabar stop vs target order unknown → pessimistic stop-first |
| Universe drift | Defaults to today’s `active_tickers.csv`, not the historical membership |
| Survivorship | Ingested list is current; delisted names are missing |
| Simulator | `simulate_trade` exits fully at the first stop/target, with no slippage and no scale-out |
| Adjusted prices | Raw data is split/dividend-adjusted, so historical levels are not the quotes seen at the time |
| Lookahead | Design goal is causal windows; always verify new filters use `Date <= as_of` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `No raw directory` | Never ingested that mode | `data_ingestor.py weekly\|daily` |
| Zero / few Cobra signals | Missing QQQ benchmark, or names under the 160-bar full-score history floor | Check `QQQ_*` in the raw silo; the Gate A–D hard gates are strict by design |
| Duplicate symbols in output | Old `*_2y_1d.csv` left next to `*_5y_1d.csv` | Clear the silo and re-ingest |
| Stale results in Docker | Host edits not in container | `docker cp` or rebuild |
| Coiled Cobra empty backfill | Wrong mode / insufficient bars | Need weekly history; lookback ≈ 60+ bars |
| `ImportError: finance_vibe` | `PYTHONPATH` unset | `export PYTHONPATH=src` (or `/app/src`) |

### Quick data health checks

```bash
# Row counts and date ranges
python - <<'PY'
import pandas as pd, glob
for p in sorted(glob.glob("data/raw/daily/{QQQ,PLTR,TSLA}_*.csv")):
    df = pd.read_csv(p, parse_dates=["Date"])
    print(f"{p}: rows={len(df)} {df['Date'].min().date()} → {df['Date'].max().date()}")
PY
```

---

## Related docs

| Doc | Contents |
| --- | -------- |
| [`README.md`](../../README.md) | Project overview and quick commands |
| [`operation_manual.md`](operation_manual.md) | Day-to-day pipeline ops |
| [`scoring_logic.md`](../handbook/scoring_logic.md) | Vibe Score rubric |
| [`coiled_cobra_rubric.md`](../handbook/coiled_cobra_rubric.md) | Coiled Cobra checklist / grades |
| [`trade_plan_calculations.md`](../handbook/trade_plan_calculations.md) | Entry / stop / target math |

---

## Code map

| File | Role |
| ---- | ---- |
| `config.py` | Timeframes, backtest constants (`BACKTEST_*`), `RUBRIC_VERSION`, `resolve_pipeline_mode`, `cut_to_as_of` |
| `data_ingestor.py` | yfinance download → validated raw CSVs |
| `analysis_engine.py` | Vibe Score + `load_benchmark_frame` / `relative_strength` / `check_coiled_cobra_market_gate` |
| `coiled_cobra.py` | `evaluate_coiled_cobra()` (shared by live scan and backtest) |
| `trade_planner.py` | `calculate_stock_levels` (Cobra Fib path; swing-geometry fallback when Fib 78.6% is missing) |
| `trade_simulator.py` | `simulate_trade` |
| `coiled_cobra_backtest.py` | Cobra signal backfill + walk-forward trade simulation |
| `tests/test_trade_simulator.py` | Fill, stop, target and R-multiple contracts |
| `tests/test_coiled_cobra_backtest.py` | Cobra planner + backtest smoke tests |
