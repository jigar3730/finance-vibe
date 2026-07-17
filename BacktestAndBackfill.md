# Backfill & Backtest Guide

Offline validation for Finance Vibe. This document covers **data backfill** (getting enough OHLCV history), **signal backfill** (historical setup archives), and **walk-forward backtests** (simulating trades on that history).

Neither backtest module is part of `run_vibe.py`. They read from `data/raw/` and write under `data/logs/`.

---

## Concepts

| Term | Meaning |
| ---- | ------- |
| **Data backfill** | Download historical OHLCV into `data/raw/{weekly\|daily}/` via `data_ingestor.py` |
| **Signal backfill** | Scan every historical bar and archive setups (Coiled Cobra: `--backfill`) |
| **Walk-forward backtest** | At each bar, detect a setup using only past data, plan levels, simulate forward fills/exits |
| **Mode / profile** | CLI mode maps to a data timeframe + swing profile (see below) |

### Mode map

| CLI mode | Raw data | Swing profile | Log silo |
| -------- | -------- | ------------- | -------- |
| `weekly` | `data/raw/weekly/` (10y × 1wk) | weekly | `data/logs/weekly/` |
| `daily` | `data/raw/daily/` (5y × 1d) | daily | `data/logs/daily/` |
| `high_beta` | `data/raw/daily/` (5y × 1d) | high_beta (long-only) | `data/logs/high_beta/` |

`high_beta` shares daily OHLCV with the ETF daily pipeline but keeps its own geometry, filters, and log directory (`config.resolve_pipeline_mode` / `config.get_log_dir`).

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
# Weekly (10y, 1wk) — used by weekly swing + Coiled Cobra
python src/finance_vibe/data_ingestor.py weekly

# Daily (5y, 1d) — used by daily swing + high_beta
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

`pipeline_backtest.select_raw_paths` keeps **one file per symbol**, preferring the longest lookback (`5y` > `2y` > `1y`). After switching daily to 5y, re-ingest so `*_5y_1d.csv` exist; older `*_2y_1d.csv` files are ignored when a longer file is present.

#### Benchmark for high_beta

High-beta regime / relative-strength gates need **QQQ** in the daily raw silo:

```bash
ls data/raw/daily/QQQ_*_1d.csv
# Prefer QQQ_5y_1d.csv
```

If QQQ is missing, high_beta will warn and regime/RS gates reject all setups.

### 4. Sync code into the container (when developing on the host)

```bash
docker cp src/finance_vibe/. finance_vibe:/app/src/finance_vibe/
docker cp tests/. finance_vibe:/app/tests/
```

Or rebuild the image when you want a durable bake-in.

---

## Pipeline swing backtest

**Module:** `src/finance_vibe/pipeline_backtest.py`

Replays quality-swing detection (`detect_setup_at_bar`) + stock levels (`calculate_stock_levels`) bar-by-bar, then simulates fills with the **scaled-out** execution model.

### Quick start

```bash
# Weekly ETFs / broad universe
python src/finance_vibe/pipeline_backtest.py weekly
python src/finance_vibe/pipeline_backtest.py weekly --tickers SPY,QQQ,IWM

# Daily ETF profile (5y daily data)
python src/finance_vibe/pipeline_backtest.py daily --tickers QQQ,SPY,IWM

# High-beta long-only (daily data + high_beta profile)
python src/finance_vibe/pipeline_backtest.py high_beta --tickers PLTR,TSLA,HOOD
python src/finance_vibe/pipeline_backtest.py high_beta \
  --tickers NVDA,AMD,AVGO,COIN,CRWD,META,NFLX,AMZN
```

### CLI reference

```text
python src/finance_vibe/pipeline_backtest.py [mode] [options]

mode                  weekly | daily | high_beta   (default: weekly)

--tickers A,B,C       Limit to these symbols (default: active_tickers.csv)
--long-min-score N    Hard macro gate for SETUP_LONG
                      default: 7 (weekly), -10 (daily / high_beta — soft vibe
                      gate already lives inside the scanner)
--short-max-score N   Hard macro gate for SETUP_SHORT (default: -2)
--cooldown-bars N     Min bars after an exit before a new signal
                      (default: from swing profile)
```

Programmatic: `run_backtest(mode=..., tickers=..., long_min=..., ...)`.

### What happens per bar

1. **Warmup** — skip first `BACKTEST_WARMUP_BARS` (60) bars.
2. **Detect** — `detect_setup_at_bar(window, symbol, profile, benchmark_df)` using only bars through `i` (no lookahead).
3. **Long-only** — high_beta drops shorts.
4. **Hard macro gate** — `passes_macro_gate(setup, vibe_score, long_min, short_max)`.
5. **Cooldown** — skip if still within `cooldown_bars` of the **last exit** (unfilled orders never start cooldown).
6. **Levels** — `calculate_stock_levels(row, mode=profile)` (row `Mode` is authoritative in the live planner; backtest passes the profile explicitly).
7. **Simulate** — `simulate_scaled_trade(...)` from the next bar forward.
8. **No overlap** — while a position is open, no new entries on that ticker.

### Profile parameters (geometry & filters)

From `config.get_swing_params(mode)`:

| Parameter | Weekly | Daily | High-beta |
| --------- | ------ | ----- | --------- |
| Direction | long + short | long + short | **long-only** |
| Soft Vibe | none | ≥ 5 (long) | ≥ 5 (long) |
| EMA proximity | 1.5% | 2% | **0.5 × ATR** |
| RSI long | 45–55 | 40–55 | **35–58** |
| Confirm slack | 0 | 0 | **0.35 × ATR** |
| Structure tolerance | 0.2% | 0.2% | **0.25 × ATR** |
| Stop | structural, capped 1.25 ATR | capped 1.5 ATR | **uncapped structural; reject if risk ∉ [0.5, 2.5] ATR** |
| T1 / T2 | 1.25 / 2.25 ATR | 0.85 / 1.6 ATR | **1R / 2R** (stop distance) |
| Entry valid / max hold / cooldown | 4 / 12 / 4 | 6 / 20 / 8 | 6 / 20 / 10 |
| Market regime | — | — | QQQ close > EMA50 & EMA100, EMA50 rising |
| Relative strength | — | — | stock/QQQ ratio > 20d MA **and** +63d relative return |
| EMA stack required | no | no | yes (20 > 50 > 100) |

High-beta is **experimental** until it clears the promotion gates below.

### Execution model (scaled-out simulator)

Used by the pipeline backtest for all modes. Legacy `simulate_trade()` (full exit at first target/stop, no slippage) remains for **Coiled Cobra** compatibility.

| Rule | Behavior |
| ---- | -------- |
| Entry | Limit at planned entry; fill when Low ≤ entry (long) / High ≥ entry (short) within `entry_valid_bars` |
| Gap entry | If Open gaps through the limit, fill at Open (flagged `Gap Entry`) |
| Slippage | Adverse `BACKTEST_SLIPPAGE_PCT` (default **0.05%**) on entry and stop exits |
| Partial | **50%** off at T1 (1R when using R-targets) |
| Runner | Remaining stop → **breakeven**; aim for T2 (2R) |
| Same-bar ambiguity | Pessimistic: stop assumed before target |
| Gap through stop | Exit at the worse Open (can be worse than −1R) |
| Max hold | Mark-to-market at Close (`partial_expired` / `expired_no_partial`) |

**Blended R outcomes (50/50):**

| Path | Blended R (approx.) |
| ---- | ------------------- |
| Full stop before T1 | ≈ −1R (worse with gap/slippage) |
| Partial @ 1R, runner @ BE | ≈ +0.5R |
| Partial @ 1R, runner @ 2R | ≈ +1.5R |

### Outcomes & counters

| Outcome | Meaning |
| ------- | ------- |
| `stopped_full` | Stopped before any partial |
| `partial_be` | Took 1R, runner stopped at breakeven |
| `partial_t2` | Took 1R, runner hit 2R |
| `partial_expired` | Took 1R, runner hit max hold |
| `expired_no_partial` | Never reached T1; closed at max hold |
| `no_fill` | Entry never touched (not written as a filled trade row) |

### Output files

```
data/logs/{weekly|daily|high_beta}/backtest_trades_{tag}_{YYYY-MM-DD}.csv
# tag = swing profile (e.g. high_beta, daily, weekly)
```

#### Trade CSV columns (filled trades)

| Column | Description |
| ------ | ----------- |
| Symbol, Signal Date, Setup Type, Mode | Identity |
| Vibe Score | Soft/hard gate score on signal bar |
| Stock Entry / Stop / Risk Per Share | Planned levels |
| Target 1R / Target 2R | Planned targets (R or ATR geometry) |
| Fill Date / Fill Price / Gap Entry | Execution |
| Stop Moved BE | True after partial |
| Partial Exit Date / Price / R | First leg |
| Runner Exit Date / Price / R | Second leg |
| Outcome | See table above |
| Blended R Multiple | Position-weighted R |
| Bars Held, MAE R, MFE R | Path metrics |
| Regime OK, RS 63d | High-beta audit fields (else empty) |

### Interpreting the summary

Stdout centers on **filled** trades:

- **Win rate** — share of fills with blended R > 0 (Wilson 95% CI)
- **Expectancy** — mean blended R
- **Total R** — sum of blended R
- **Profit factor** — gross wins / gross losses
- **Avg winner / loser**, **MAE / MFE**

Win rate is secondary to expectancy and profit factor.

### High-beta promotion gates (frozen defaults)

Before treating `high_beta` as production-ready:

| Gate | Threshold |
| ---- | --------- |
| Out-of-sample fills | ≥ 100 |
| Win rate | ≥ 55% |
| Expectancy | ≥ +0.20R |
| Profit factor | ≥ 1.3 |
| Concentration | No single ticker > 20% of total profit |

If any gate fails, keep the profile **experimental** and use filter ablation (count setups after each gate: base → confirmed → vibe → regime → RS → risk) to see what limited sample size.

**Do not** reuse the PLTR/TSLA/HOOD tuning set as proof; use a predeclared holdout basket.

### Common recipes

```bash
# Regression: high-beta names after a code change
python src/finance_vibe/pipeline_backtest.py high_beta --tickers PLTR,TSLA,HOOD

# Daily ETF sanity check
python src/finance_vibe/pipeline_backtest.py daily --tickers QQQ,SPY

# Stricter weekly macro gate
python src/finance_vibe/pipeline_backtest.py weekly --long-min-score 8 --short-max-score -3

# Wider spacing between trades
python src/finance_vibe/pipeline_backtest.py daily --cooldown-bars 12
```

---

## Coiled Cobra backfill & backtest

**Module:** `src/finance_vibe/coiled_cobra_backtest.py`

Separate from the quality-swing path. Uses Fib-anchored geometry (`Source=coiled_cobra`) and the **legacy** `simulate_trade` (full exit at first stop/target; no scale-out / slippage).

Typically run on **weekly** data (macro reversal horizon).

### Signal backfill

Scans every eligible historical bar and archives Coiled Cobra setups (no trade simulation):

```bash
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill --tickers SPY,QQQ,IWM
```

**Output:** `data/logs/{mode}/coiled_cobra_backfill_{YYYY-MM-DD}.csv`

Columns include Symbol, Date, Setup Type, Close, EMAs, ATR, Fib 61.8%/78.6%, Score, Grade, Checks Met, Source.

Useful for:

- Counting historical signal frequency
- Auditing grade distribution
- Feeding research notebooks without re-scanning

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

#### Trade CSV columns (ML-relevant)

| Zone | Columns |
| ---- | ------- |
| Identity | `Symbol`, `Signal Date`, `Setup Type` |
| Pre-signal features | `Score`, `Grade`, `Pct_From_EMA20`, `Pct_From_EMA50`, `Pct_From_Fib618`, `Pct_From_Fib786`, `ATR_Pct` |
| Execution / leakage | `Stock Entry`, `Stock Stop`, `Target 1`, `Target 2`, `Outcome`, `Exit Date`, `Exit Price`, `R Multiple`, `Target_Label`, `Target_R_Mult` |
| Continuous targets | `Forward_Return_5w`, `Forward_Return_13w`, `Forward_Return_26w` |

`Forward_Return_{Nw}` is `(Close[t+N] − Close[t]) / Close[t]` when enough future bars exist; otherwise `None` / NaN.

### ML baseline (downstream of backtest)

**Module:** `src/finance_vibe/coiled_cobra_ml_training.py`  
**Doc:** **`CoiledCobraML.md`**

Consumes `coiled_cobra_backtest_trades_*.csv` to train XGBoost + LightGBM regressors on **`Forward_Return_13w`** with:

- 6 pre-signal features (`Grade` excluded — collinear with `Score`)
- Strict temporal split (train ≤2023 / val 2024 / test 2025–Jul 2026) — **no random K-fold**
- Leakage columns dropped; `no_fill` rows kept
- MAE objectives (`reg:absoluteerror` / `regression_l1`) + `ATR_Pct` sample weights

```bash
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_2026-07-17.csv
```

### Live scanner vs historical modules

| Task | Command |
| ---- | ------- |
| Live Coiled Cobra (latest bar) | `python src/finance_vibe/coiled_cobra.py weekly` |
| Historical signal archive | `.../coiled_cobra_backtest.py weekly --backfill` |
| Historical trade simulation | `.../coiled_cobra_backtest.py weekly --backtest` |
| ML baseline training | `.../coiled_cobra_ml_training.py [--csv PATH]` |

---

## End-to-end workflows

### A. Fresh daily / high_beta study

```bash
# 1. Universe
python src/finance_vibe/ticker_provider.py

# 2. 5y daily OHLCV (do NOT use run_vibe if you need to keep existing files)
python src/finance_vibe/data_ingestor.py daily

# 3. Confirm QQQ + study names exist
ls data/raw/daily/QQQ_5y_1d.csv
ls data/raw/daily/{PLTR,TSLA,HOOD,NVDA}_5y_1d.csv

# 4. Backtest
python src/finance_vibe/pipeline_backtest.py high_beta --tickers PLTR,TSLA,HOOD,NVDA
python src/finance_vibe/pipeline_backtest.py daily --tickers QQQ,SPY,IWM
```

### B. Weekly + Coiled Cobra validation

```bash
python src/finance_vibe/data_ingestor.py weekly

python src/finance_vibe/pipeline_backtest.py weekly --tickers SPY,QQQ,IWM
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill --tickers SPY,QQQ,IWM
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest --tickers SPY,QQQ,IWM
```

### C. Docker one-liners

```bash
docker exec finance_vibe sh -c \
  'cd /app && PYTHONPATH=/app/src python src/finance_vibe/pipeline_backtest.py high_beta --tickers PLTR,TSLA,HOOD'

docker exec finance_vibe sh -c \
  'cd /app && PYTHONPATH=/app/src python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill --tickers SPY,QQQ'
```

### D. Unit tests for simulation contracts

```bash
python -m pytest tests/test_pipeline_backtest.py tests/test_coiled_cobra_backtest.py -q
python -m pytest tests/ -q   # full suite
```

---

## Limitations (read before trusting numbers)

| Limitation | Detail |
| ---------- | ------ |
| Stock-only | No options premium, delta, or theta P&L |
| Daily OHLC order | Intrabar stop vs target order unknown → pessimistic stop-first |
| Slippage model | Flat adverse %; not volume- or volatility-scaled |
| Universe drift | Defaults to today’s `active_tickers.csv`, not the historical membership |
| Survivorship | Ingested list is current; delisted names are missing |
| Soft vs hard vibe | Daily/high_beta soft gate is inside the scanner; weekly hard gate is in the backtest CLI |
| Coiled Cobra sim | Still uses legacy full-exit simulator (not the 50% scale-out model) |
| Lookahead | Design goal is causal windows; always verify new filters use `Date <= as_of` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `No raw directory` | Never ingested that mode | `data_ingestor.py weekly\|daily` |
| Zero signals (high_beta) | Missing QQQ, regime/RS too strict, or risk band rejecting all | Check `QQQ_5y_1d.csv`; run filter ablation; inspect `max_risk_atr` |
| Uses short history | Only `*_2y_1d.csv` present after period change | Re-ingest daily → `*_5y_1d.csv` |
| Stale results in Docker | Host edits not in container | `docker cp` or rebuild |
| ML `FileNotFoundError` for trades CSV | Backtest CSV missing on volume | Run cobra `--backtest`; see **`CoiledCobraML.md`** |
| Coiled Cobra empty backfill | Wrong mode / insufficient bars | Need weekly history; lookback ≈ 60+ bars |
| `ImportError: finance_vibe` | `PYTHONPATH` unset | `export PYTHONPATH=src` (or `/app/src`) |
| Unexpected short trades in high_beta | Old code without `long_only` | Sync latest `swing_scanner` / `pipeline_backtest` |
| Outputs in wrong folder | Expected daily logs for high_beta | high_beta writes to `data/logs/high_beta/` |

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
| `README.md` | Project overview and quick commands |
| `OperationManual.md` | Day-to-day pipeline ops |
| `CoiledCobraML.md` | Coiled Cobra ML baseline (features, splits, metrics) |
| `swing_setup_readme.md` | Quality-swing rules and geometry table |
| `src/finance_vibe/Scoring_Logic.md` | Vibe Score rubric |
| `Coiled Cobra Rubric .MD` | Coiled Cobra checklist / grades |
| `Trade Plan Calculations.md` | Entry / stop / target math |

---

## Code map

| File | Role |
| ---- | ---- |
| `config.py` | Timeframes, swing profiles, backtest constants, `resolve_pipeline_mode`, `get_log_dir`, `compute_swing_levels` |
| `data_ingestor.py` | yfinance download → validated raw CSVs |
| `swing_scanner.py` | `detect_setup_at_bar`, long-only / regime / RS / risk rejection |
| `analysis_engine.py` | Vibe Score + `load_benchmark_frame` / `market_regime_ok` / `relative_strength` |
| `trade_planner.py` | `calculate_stock_levels` (row Mode authoritative; cobra Fib path preserved) |
| `pipeline_backtest.py` | Swing walk-forward + scaled simulator + reporting |
| `coiled_cobra_backtest.py` | Cobra signal backfill + legacy trade simulation |
| `coiled_cobra_ml_training.py` | XGBoost/LightGBM baseline on `Forward_Return_13w` |
| `tests/test_pipeline_backtest.py` | Scale-out, gap, slippage, long-only, cooldown contracts |
| `tests/test_coiled_cobra_backtest.py` | Cobra planner + backtest smoke tests |
