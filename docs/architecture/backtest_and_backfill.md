# Backfill & Backtest Guide

Offline validation for Finance Vibe. This document covers **data backfill** (getting enough OHLCV history), **signal backfill** (historical setup archives), and **walk-forward backtests** (simulating trades on that history).

Coiled Cobra is the only signal engine with a backtest. The quality-swing
`pipeline_backtest.py` was removed with `swing_scanner.py` (2026-09-16), and its
generic fill/stop/target primitives now live in `trade_simulator.py`. None of
these modules are part of `run_vibe.py`. They read from `data/raw/` and write under `data/logs/`.

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
| `high_beta` | `data/raw/daily/` (5y × 1d) | high_beta | `data/logs/high_beta/` |

`high_beta` reads the same daily OHLCV as `daily` and uses the same Coiled Cobra calibration. Only the log directory differs (`config.resolve_pipeline_mode` / `config.get_log_dir`).

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
# Weekly (10y, 1wk) — weekly Coiled Cobra, ML baseline, leader experiment
python src/finance_vibe/data_ingestor.py weekly

# Daily (5y, 1d) — daily + high_beta profiles
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

Typically run on **weekly** data (coil → expansion horizon). Every output CSV carries a `Rubric_Version` column (`config.RUBRIC_VERSION`, currently `4.0`), which ML training checks before it will use the file.

### Signal backfill

Scans every eligible historical bar and archives Coiled Cobra setups (no trade simulation):

```bash
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill
python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill --tickers SPY,QQQ,IWM
```

**Output:** `data/logs/{mode}/coiled_cobra_backfill_{YYYY-MM-DD}.csv`

Columns include Symbol, Date, Setup Type, Close, EMAs, ATR, Swing Low, Fib distances, ATR_Pct, Score, Grade, Tier, Checks Met, Source, RS 63d, RVOL, Market Gate, BBWidth Pctile, Rubric_Version.

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
| Research features (pre-signal) | `RVOL`, `Market Gate`, `Tier`, `Checks_N`, `RS_63d`, `BBWidth_Pctile`, pillar sub-scores (`Part_*`), causal QQQ regime `QQQ_Pct_From_EMA50`, `QQQ_Ret_13w` |
| Execution / leakage | `Stock Entry`, `Stock Stop`, `Target 1`, `Target 2`, `Outcome`, `Exit Date`, `Exit Price`, `R Multiple`, `Target_Label`, `Target_R_Mult` |
| Continuous targets | `Forward_Return_2w` (baseline $Y$), `Excess_Return_2w` (vs QQQ), `Forward_Return_5w`, `Forward_Return_13w`, `Forward_Return_26w` |

`Forward_Return_{Nw}` is `(Close[t+N] − Close[t]) / Close[t]` when enough future bars exist; otherwise `None` / NaN.

### ML baseline (downstream of backtest)

**Module:** `src/finance_vibe/coiled_cobra_ml_training.py`  
**Doc:** **[`coiled_cobra_ml.md`](coiled_cobra_ml.md)**

Consumes `coiled_cobra_backtest_trades_*.csv` to train XGBoost + LightGBM regressors on **`Forward_Return_2w`** (code of record in `coiled_cobra_ml_training.py`) with:

- 6 pre-signal features (`Grade` excluded — collinear with `Score`)
- Strict temporal split: rolling 26-week test / 26-week val / rest train on `Signal Date` with a 2-week embargo — **no random K-fold**
- Leakage columns dropped; `no_fill` rows kept
- MAE objectives (`reg:absoluteerror` / `regression_l1`) + `ATR_Pct` sample weights

```bash
python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv data/logs/weekly/coiled_cobra_backtest_trades_YYYY-MM-DD.csv
```

### Live scanner vs historical modules

| Task | Command |
| ---- | ------- |
| Live Coiled Cobra (latest bar) | `python src/finance_vibe/coiled_cobra.py weekly` |
| Historical signal archive | `.../coiled_cobra_backtest.py weekly --backfill` |
| Historical trade simulation | `.../coiled_cobra_backtest.py weekly --backtest` |
| ML baseline training | `.../coiled_cobra_ml_training.py [--csv PATH]` |
| ML vs Score walk-forward | `python -m finance_vibe.coiled_cobra_ml_walkforward [--csv PATH]` |
| Pre-registered ML experiment | `python -m finance_vibe.coiled_cobra_ml_experiment --csv PATH` |
| Leader-Expansion experiment | `python -m finance_vibe.coiled_cobra_leader_experiment` (below) |

ML walk-forward and experiment details: [`coiled_cobra_ml.md`](coiled_cobra_ml.md).

---

## End-to-end workflows

### A. Weekly Coiled Cobra validation

```bash
python src/finance_vibe/ticker_provider.py
python src/finance_vibe/data_ingestor.py weekly   # do NOT use run_vibe if you need to keep existing files

python src/finance_vibe/coiled_cobra_backtest.py weekly --backfill --tickers SPY,QQQ,IWM
python src/finance_vibe/coiled_cobra_backtest.py weekly --backtest
python src/finance_vibe/coiled_cobra_ml_training.py
```

### B. Daily / high_beta Coiled Cobra

```bash
python src/finance_vibe/data_ingestor.py daily
python src/finance_vibe/coiled_cobra_backtest.py daily --backtest --tickers PLTR,TSLA,HOOD,NVDA
```

`coiled_cobra_backtest.py` accepts the modes in `config.TIMEFRAME_PROFILES` (`weekly`, `daily`). ML training is weekly-only, because `Forward_Return_2w` is measured in bars.

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
| Slippage model | Flat adverse %; not volume- or volatility-scaled |
| Universe drift | Defaults to today’s `active_tickers.csv`, not the historical membership |
| Survivorship | Ingested list is current; delisted names are missing |
| Simulator | `simulate_trade` exits fully at the first stop/target with no slippage. `simulate_scaled_trade` (50% scale-out, slippage) exists in `trade_simulator.py` but no live tool uses it |
| Adjusted prices | Raw data is split/dividend-adjusted, so historical levels are not the quotes seen at the time |
| Lookahead | Design goal is causal windows; always verify new filters use `Date <= as_of` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `No raw directory` | Never ingested that mode | `data_ingestor.py weekly\|daily` |
| Zero / few Cobra signals | Missing QQQ benchmark, or names under the 160-bar full-score history floor | Check `QQQ_*` in the raw silo; the Gate A–D hard gates are strict by design |
| Duplicate symbols in output | Old `*_2y_1d.csv` left next to `*_5y_1d.csv` | Clear the silo and re-ingest |
| ML refuses the trades CSV | `Rubric_Version` missing or not equal to `config.RUBRIC_VERSION` | Regenerate with `--backtest` |
| Stale results in Docker | Host edits not in container | `docker cp` or rebuild |
| ML `FileNotFoundError` for trades CSV | Backtest CSV missing on volume | Run cobra `--backtest`; see **[`coiled_cobra_ml.md`](coiled_cobra_ml.md)** |
| Coiled Cobra empty backfill | Wrong mode / insufficient bars | Need weekly history; lookback ≈ 60+ bars |
| `ImportError: finance_vibe` | `PYTHONPATH` unset | `export PYTHONPATH=src` (or `/app/src`) |
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
| [`README.md`](../../README.md) | Project overview and quick commands |
| [`operation_manual.md`](operation_manual.md) | Day-to-day pipeline ops |
| [`coiled_cobra_ml.md`](coiled_cobra_ml.md) | Coiled Cobra ML baseline (features, splits, metrics) |
| [`swing_setup.md`](../handbook/swing_setup.md) | Quality-swing rules (historical; scanner decommissioned) |
| [`scoring_logic.md`](../handbook/scoring_logic.md) | Vibe Score rubric |
| [`coiled_cobra_rubric.md`](../handbook/coiled_cobra_rubric.md) | Coiled Cobra checklist / grades |
| [`trade_plan_calculations.md`](../handbook/trade_plan_calculations.md) | Entry / stop / target math |
| [`QUANT_ML_MANUAL.md`](../handbook/QUANT_ML_MANUAL.md) | Quant / ML theory mapped to this repo |

---

## Code map

| File | Role |
| ---- | ---- |
| `config.py` | Timeframes, backtest constants (`BACKTEST_*`), `RUBRIC_VERSION`, `resolve_pipeline_mode`, `get_log_dir`, `cut_to_as_of` |
| `data_ingestor.py` | yfinance download → validated raw CSVs |
| `analysis_engine.py` | Vibe Score + `load_benchmark_frame` / `relative_strength` / `check_coiled_cobra_market_gate` |
| `coiled_cobra.py` | `evaluate_coiled_cobra()` (shared by live scan, backtest, experiments) |
| `trade_planner.py` | `calculate_stock_levels` (Cobra Fib path; legacy swing fallback) |
| `trade_simulator.py` | `simulate_trade`, `simulate_scaled_trade`, `passes_macro_gate` |
| `coiled_cobra_backtest.py` | Cobra signal backfill + walk-forward trade simulation |
| `coiled_cobra_ml_training.py` | XGBoost/LightGBM baseline on `Forward_Return_2w` |
| `coiled_cobra_ml_walkforward.py` / `coiled_cobra_ml_experiment.py` | ML-vs-Score research harnesses |
| `coiled_cobra_leader_experiment.py` | Leader-Expansion vs Coiled Cobra walk-forward |
| `tests/test_trade_simulator.py` | Fill, gap, slippage, scale-out, macro-gate contracts |
| `tests/test_coiled_cobra_backtest.py` | Cobra planner + backtest smoke tests |
| `tests/test_leader_experiment.py` | Leader experiment protocol tests |

---

## Leader-Expansion vs Coiled Cobra experiment (research only)

`coiled_cobra_leader_experiment.py` is a pre-registered walk-forward comparison that never
touches the live scanner or its thresholds. Motivation: a review of tickers with monster runs
(PLTR, HOOD, ASTS, MU, MRVL, COIN, MSTR, STX, ...) showed the coil rubric mostly missed them --
`vol_contraction` scored 0 through most of those runs (Gates C/D fail) and several runs began
before the 160-bar history floor. The experiment asks whether a separate **leader** rule (Gate A
+ Gate B + strong RS, no coil requirement) catches them without giving back edge, and whether
simply loosening the existing gates (`R1_relaxed_gates`) does better.

```bash
docker exec -it finance_vibe bash
cd /app && export PYTHONPATH=/app/src
python -m finance_vibe.coiled_cobra_leader_experiment                     # whole raw dir (~20 min on 4 cores)
python -m finance_vibe.coiled_cobra_leader_experiment --tickers MU,MSTR --out-dir /tmp/lx   # smoke test
python -m finance_vibe.coiled_cobra_leader_experiment --from-bars <bars.csv.gz> --runs <runs.csv>   # re-analyse
```

Outputs (in `data/logs/weekly/`, or `--out-dir`): `leader_experiment_bars_<date>.csv.gz` (every
scoreable bar with its gate profile, variant flags, forward stats and both exit outcomes),
`leader_experiment_runs_<date>.csv` (monster runs), `leader_experiment_<date>.json` (analysis).
Names deliberately avoid `coiled_cobra_backtest_trades_*`, which the ML training script globs.

Protocol (fixed in the module docstring and constants before results were read):

| Element | Definition |
| ------- | ---------- |
| Variants | `B0_baseline` (deployed rubric), `R1_relaxed_gates` (Gate A close>slow EMA & rising, Gate C structure-only, score >= 60), `L1`-`L4` leader rules, `C0_random` control |
| Discovery vs holdout | The tickers the leader rules were derived from are reported but **never** used for qualification; decisions use the rest of the universe |
| Episodes | Consecutive weekly flags per ticker collapse to one entry (first bar) |
| Exits | `planner` = deployed Fib-78.6% entry, 2R/3R via `calculate_stock_levels` + `simulate_trade`; `trail` = next-open entry, 2.5 ATR stop, 3 ATR chandelier off the highest close, 52-bar time exit. **Primary outcome = trail R** |
| Periods | `dev` -> `lockbox` (52 wks ending 52 wks before the last bar) -> `live` (censored, reported only) |
| Qualification | Q0 >= 100 episodes; Q1 week-cluster bootstrap CI of mean trail-R > 0; Q2 monster capture (>= 100% still left) >= baseline + 15 pts; Q3 share with a >= 25% drawdown inside 26 wks <= baseline + 10 pts; Q4 paired mean-R difference vs `C0_random` CI > 0 |
| Lockbox | The single best qualifier is evaluated **once**; pass = mean R > 0 and cluster t > 1.645. If nothing qualifies it stays unspent |

`C0_random` is the bar to beat, not a "no edge" check: long-only entries in a rising market with
a trailing exit earn positive R by themselves, so a variant only counts if it beats the control.
Tests: `tests/test_leader_experiment.py`.


---

## Breakout Readiness experiment (research only)

`breakout_experiment.py` is a pre-registered walk-forward test of `breakout_scanner.py`. It
never changes the live scanner. Question: do any of the scanner's states, or its 100-point
readiness score, predict forward trade outcomes better than random entries?

```bash
docker exec -it finance_vibe bash
cd /app && export PYTHONPATH=/app/src
python -m finance_vibe.breakout_experiment                                  # whole weekly raw dir (~40 min on 4 cores)
python -m finance_vibe.breakout_experiment --tickers AAPL,MU --out-dir /tmp/bx   # smoke test
python -m finance_vibe.breakout_experiment --from-bars <bars.csv.gz>        # re-analyse
```

Outputs (in `data/logs/weekly/`, or `--out-dir`): `breakout_experiment_bars_<date>.csv.gz`
(every scoreable bar with status, scores, variant flags, forward stats and trail outcome) and
`breakout_experiment_<date>.json` (analysis).

Protocol (fixed in the module docstring and constants before results were read). It reuses
the leader experiment's exits, periods, deduplication and statistics:

| Element | Definition |
| ------- | ---------- |
| Signals | Every weekly bar from the scanner's 80-bar floor onward is re-scanned with the live `FeatureEngine` + `ScoringEngine` on data cut at that bar (identical to an `--as-of` run) |
| Variants | `BK_PRE`, `BK_CONFIRMED`, `BK_PRE_OR_CONF`, `BK_DISPLAY` (dashboard candidate rule minus FAILED), `BK_READY70` (readiness ≥ 70, any non-failed status); `BK_FAILED` is a negative check and never qualifies; `C0_random` is the control |
| Exit | `trail` only: next-open entry, 2.5 ATR stop, 3 ATR chandelier, 52-bar time exit. There is no planner exit because breakout rows have no Fib levels |
| Periods | `dev` → `lockbox` (52 weeks ending 52 weeks before the last bar) → `live` (censored, reported only) |
| Qualification (dev, all tickers) | Q0 ≥ 100 episodes; Q1 cluster-bootstrap CI of mean trail-R > 0; Q2 paired mean-R difference vs `C0_random` CI > 0; Q3 share of episodes with a ≥ 25% drawdown inside 26 weeks ≤ control + 10 pts |
| Lockbox | The best qualifier (highest Q2 lower bound) is evaluated **once**. It passes if its difference vs random is > 0 and its mean R > 0 with cluster t > 1.645. If nothing qualifies, the lockbox stays unspent |
| Report only | Weekly cross-sectional rank IC of readiness and each pillar score vs the 13-week return; forward outcomes per status; Coiled Cobra `B0_baseline` on shared ticker-weeks, from the newest `leader_experiment_bars_*.csv.gz` |

Tests: `tests/test_breakout_experiment.py` (variant boundaries, no-lookahead check, match with
the live scanner on cut data, IC math, protocol plumbing).

### Result (2026-10-01 run, last bar 2026-09-07, 259 tickers, 97,463 scoreable bars)

**No variant qualified, so the lockbox stays unspent.** Every breakout variant earns positive
trail-R, as random long entries do in this sample, but none beats `C0_random` (Q2):

| Variant (dev) | Episodes | Mean trail-R [95% CI] | vs random [95% CI] |
| ------------- | -------: | --------------------- | ------------------ |
| `BK_PRE` | 366 | 0.29 [0.14, 0.43] | −0.11 [−0.28, 0.05] |
| `BK_CONFIRMED` | 876 | 0.38 [0.20, 0.62] | −0.01 [−0.20, 0.22] |
| `BK_PRE_OR_CONF` | 1,219 | 0.36 [0.22, 0.53] | −0.03 [−0.19, 0.14] |
| `BK_DISPLAY` | 5,242 | 0.33 [0.24, 0.43] | −0.06 [−0.16, 0.03] |
| `BK_READY70` | 2,042 | 0.27 [0.17, 0.37] | **−0.13 [−0.23, −0.02]** (worse than random) |
| `BK_FAILED` (negative check) | 3,863 | 0.33 [0.22, 0.45] | — |
| `C0_random` | 3,297 | 0.39 [0.31, 0.48] | — |

Report-only findings:

- **Readiness score has no cross-sectional signal.** Mean weekly rank IC vs the 13-week return
  is −0.012 [−0.028, 0.002]. The Volume (−0.026), Momentum (−0.015) and Compression (−0.014)
  pillar ICs are slightly *negative*, with CIs below 0. Trend is the only positive pillar (+0.011,
  CI spans 0).
- **The status ladder is not ordered.** `PRE_BREAKOUT`, the headline status, has the weakest
  13-week mean return (+2% vs +6% for `WATCH`). `FAILED_BREAKOUT` does not underperform.
  `BREAKOUT_CONFIRMED` has the most upside (+9%, 15% doubled within 26w) but also the most
  ≥ 25% drawdowns (35%), which matches its random-like R.
- On shared ticker-weeks, Coiled Cobra `B0_baseline` (0.22R) is also below random (0.48R) under
  this trail exit, which agrees with the 2026-09-19 leader experiment.

Conclusion: the breakout scanner's states and score add nothing over random entries on weekly
data. Do not use them for ranking or entries, and do not tune their weights from this data,
since there is no signal to tune toward.

---

## Coiled Cobra vs random entries (paired test, 2026-10-01)

The leader experiment never tested the deployed rubric itself against its random control. This
follow-up does, using the saved `leader_experiment_bars_2026-09-19.csv.gz` with no new
backtest. The rule was set before the numbers were read: Cobra has an edge only if the paired
(same-week) difference in mean R vs `C0_random` has a 95% week-cluster bootstrap CI above 0 on
`dev`, **and** that is confirmed on the `lockbox` year.

Method: deduplicated `B0_baseline` and `C0_random` episodes (`dedup_episodes`), filled trades
only, compared with `paired_diff` (2,000 draws). The primary exit is Cobra's own planner
geometry (`pl_r`: Fib 78.6% limit entry, structural stop, 2R/3R via `calculate_stock_levels` +
`simulate_trade`). The trail exit (`tr_r`) is shown for reference. Rows are holdout tickers, and
including the 11 discovery tickers changes nothing material.

| Period | Exit | Cobra mean R [95% CI] | Random mean R | Cobra − random [95% CI] |
| ------ | ---- | --------------------- | ------------- | ----------------------- |
| dev | planner | +0.27 [+0.15, +0.40] (n 793) | −0.01 (n 1,987) | **+0.29 [+0.14, +0.42]** |
| lockbox | planner | +0.03 [−0.21, +0.29] (n 179) | +0.01 (n 532) | **+0.02 [−0.22, +0.26]** |
| dev | trail | +0.21 (n 969) | +0.48 (n 2,349) | −0.27 [−0.41, −0.12] |
| lockbox | trail | +0.06 (n 220) | +0.44 (n 624) | −0.38 [−0.59, −0.16] |

Planner-exit detail (dev): fill rate 82% vs 83%; win rate 41% vs 32%; Cobra reaches T1 26% of
the time vs 18% for random, and stops out 46% vs 56%.

**Result: no confirmed edge.** On its own exit, Cobra clearly beat random on the development
years, mainly by hitting the first target more often rather than by skipping entries. In the
held-out year the difference fell to about zero. Under the trail exit, Cobra trails random in
both periods: random's trail profit is the market's uptrend, which Cobra's 3R target cap
cuts off.

**The lockbox is now spent.** The 52 weeks ending 52 weeks before 2026-09-19 have been looked
at and are no longer a clean holdout. Any further claim of a Cobra edge must come from signals
dated after the 2026-09 data end, judged by the same paired rule.
