# Finance Vibe — Holistic Code Review

**Review date:** 2026-07-12  
**Scope:** Full workspace scan (`/src/finance_vibe/`, tests, config, docs)  
**Reviewer role:** Senior Python Engineer / Technical Architect

**Scope notes:**

- `analysis_engine_local.py` **does not exist** in this repo. The closest “shadow/alternate math” artifact is `coiled_cobra_v2.py` (quant factor refactor of `coiled_cobra.py`).
- Weekly ingestion is configured as **10 years** (`period: "10y"`), not 5 years.

---

## 1. Process Documentation — End-to-End Pipeline

### High-level architecture

```mermaid
flowchart TD
    A[run_vibe.py --mode weekly|daily] --> B[clean_raw_folder]
    B --> C[ticker_provider.py]
    C --> D[data/active_tickers.csv]
    C --> E[data_ingestor.py]
    E --> F[data/raw/{mode}/*.csv]
    F --> G[swing_scanner.py]
    F --> H[coiled_cobra.py]
    G --> I[swing_setups_{date}.csv]
    H --> J[coiled_cobra_setups_{date}.csv]
    I --> K[trade_planner.py]
    J --> K
    K --> L[trade_plan_{date}.csv]
    L --> M[trade_plan_helper.py]
    M --> N[trade_plan_clean_{date}.csv]

    F -.->|NOT in live pipeline| AE[analysis_engine.py]
    AE -.-> VR[vibe_report_{date}.csv]

    O[app.py Flask UI] --> N
    O --> L

    P[pipeline_backtest.py offline] --> AE
    P --> G
```

### Stage-by-stage flow

| Step | Module | Input | Output | Notes |
|------|--------|-------|--------|-------|
| 0 | `run_vibe.py` | CLI `--mode` | subprocess chain | Wipes `data/raw/{mode}/` before every run |
| 1 | `ticker_provider.py` | `ticker_manifest.csv` + Yahoo screener | `data/active_tickers.csv` | Merges manifest + `STATIC_TICKERS`, caps at 250 |
| 2 | `data_ingestor.py` | `active_tickers.csv` | `data/raw/{mode}/{TICKER}_{period}_{interval}.csv` | Sequential `yf.download()` per ticker |
| 3 | `analysis_engine.py` | Raw CSVs | `vibe_report_{date}.csv` | **Commented out** in `run_vibe.py` (line 57) |
| 4 | `swing_scanner.py` | Raw CSVs + active tickers | `swing_setups_{date}.csv` | EMA pullback tactical layer |
| 5 | `coiled_cobra.py` | Raw CSVs + active tickers | `coiled_cobra_setups_{date}.csv` | Macro reversal scanner (100-pt rubric) |
| 6 | `trade_planner.py` | Swing + Cobra setup CSVs | `trade_plan_{date}.csv` | Merges both sources |
| 7 | `trade_plan_helper.py` | Same-day trade plan | `trade_plan_clean_{date}.csv` | Adds R:R columns |

**Docker default:** `CMD` runs `app.py` (dashboard on port 5000), not the pipeline.

---

### Macro Vibe Score path (`analysis_engine.py`)

Even though it is **not wired into `run_vibe.py`**, it is the canonical macro math layer and is used by `pipeline_backtest.py`.

**Per-ticker flow:**

1. `iter_raw_csv_paths()` → discover CSVs in `data/raw/{mode}/`
2. `load_ohlc_csv()` → normalize Date/Close/High/Low
3. `build_features()` → compute indicators on full history
4. `score_last_row()` → `_compute_score()` on **latest bar only**
5. `sentiment_action()` → map score to labels
6. `ProcessPoolExecutor` parallelizes `scan_one_file()` across tickers
7. Write sorted `vibe_report_{date}.csv`

**Indicator stack (`build_features`):**

| Indicator | Implementation | Window |
|-----------|----------------|--------|
| SMA20, SMA50 | `rolling().mean()` | 20 / 50 |
| MACD_H | Manual EMA(12/26/9) histogram | 12/26/9 |
| MACD_S | EMA of MACD_H | 9 |
| RSI | Wilder EMA smoothing | 14 |
| RSI_S | SMA of RSI | 10 |
| CCI | **Vectorized MAD** via `sliding_window_view` | 20 |
| CCI_S | SMA of CCI | 10 |

**Scoring rubric (`_compute_score`) — actual weights:**

| Component | Range | Logic |
|-----------|-------|-------|
| Trend | ±4 | Full alignment: `Close > SMA20 > SMA50` or inverse |
| Momentum | ±2 | `MACD_H` vs `MACD_S` **and** `RSI` vs `RSI_S` |
| MomentumDecay | −1 | Bearish MACD while price above SMA20 |
| Timing | −2 to +2 | Distance from SMA20 (pullback band) |
| CCI | −2 to +1 | Cyclical band / exhaustion rules |
| RSI_Risk | variable | Cap at 5 if RSI>80; ±1 adjustments |
| Persistence | −2 | Scores ≥7 require `MACD_H > 0` and `RSI > 50` |
| Final | clip | `[-10, +10]` integer |

There is **no explicit Volatility (±3) bucket** in `analysis_engine.py`. Volatility appears only in downstream scanners (ATR, Bollinger width in `coiled_cobra_v2.py`).

---

### Tactical swing path (`swing_scanner.py`)

1. Load raw CSVs; filter to `active_tickers.csv`
2. `add_indicators()` via **pandas_ta**: EMA20/50, MACD histogram, RSI, ATR
3. `evaluate_setup()` on latest 2 bars:
   - **SETUP_LONG:** `EMA20 > EMA50`, rising EMA50, price within 0–2% above EMA20, RSI band, 2-bar MACD hist rise
   - **SETUP_SHORT:** mirror logic
4. `momentum_ready_long/short()` — 2 consecutive MACD hist moves + std-dev cap
5. Export matching rows to `swing_setups_{date}.csv`

Uses **EMA** (tactical), deliberately separate from macro **SMA** layer.

---

### Coiled Cobra path (`coiled_cobra.py`)

1. Requires `LOOKBACK + 15` bars (67 weekly / 267 daily)
2. `add_macro_indicators()` — EMA20/50, MACD line/signal, Fib 61.8%/78.6%, ATR
3. `evaluate_coiled_cobra()` — additive 100-point rubric:
   - Fib confluence (30)
   - Volume profile shelf via histogram (25)
   - Deep markdown vs EMA20 (15)
   - MACD compression (20)
   - Bullish MACD crossover (10)
4. Grade A (≥85) or B (≥70); else rejected
5. Output `coiled_cobra_setups_{date}.csv` (SETUP_LONG only today)

---

### Trade planning path

`trade_planner.py` merges latest swing + cobra CSVs, calls `calculate_stock_levels()` per row, writes `trade_plan_{date}.csv`. `trade_plan_helper.py` adds `Risk Per Share`, `R:R T1`, `R:R T2`.

---

### Offline validation (`pipeline_backtest.py`)

Walk-forward replay **not** in live pipeline:

1. For each bar after warmup (60): `detect_setup_at_bar()` (swing logic)
2. `build_features()` + `score_last_row()` (macro gate)
3. Long requires score ≥ 7; short requires score ≤ −2
4. `simulate_trade()` on High/Low for fill/stop/targets
5. Output `backtest_trades_{date}.csv`

---

## 2. Dead Code Identification

### Removed / missing artifacts

| Item | Status |
|------|--------|
| `analysis_engine_local.py` | **Not in workspace** — no imports or references |
| `coiled_cobra_v2.py` | Exists but **never called** by `run_vibe.py` or any module; outputs `quant_cobra_setups_*.csv` (orphan) |

### Disabled live pipeline stage

In `run_vibe.py` line 57:

```python
#{"path": "src/finance_vibe/analysis_engine.py", "pass_mode": True},
```

Macro scoring runs only manually or via backtest. Docs (`README.md`, `Scoring_Logic.md`, `OperationManual.md`) still describe it as step 3/4.

### Unused config helpers

`get_raw_filename()` and `get_raw_path()` in `config.py` are defined but **never imported** elsewhere. `data_ingestor.py` duplicates filename construction inline.

### Computed-but-discarded in `trade_planner.py`

- `calculate_options_expiry()` — called every row, results **never written** to output
- `DELTA_LONG` / `DELTA_SHORT` / `delta_range` / `options_type` — computed, **never exported**
- README promises “options/LEAPS metadata” in `trade_plan_*.csv`; actual columns omit them entirely

### Duplicate / parallel calculation pipelines

| Concern | Files | Overlap |
|---------|-------|---------|
| Macro reversal scoring | `coiled_cobra.py` vs `coiled_cobra_v2.py` | Same purpose, different math (discrete 100-pt vs continuous factor model) |
| MACD / RSI / EMA | `analysis_engine.py` (manual) vs `swing_scanner.py` / `coiled_cobra.py` (pandas_ta) | Same indicators, **different implementations** → numerically divergent |
| Path resolution | `swing_scanner`, `coiled_cobra`, `trade_planner` | Each rebuilds `BASE_DIR` paths manually instead of `config.get_mode_config()` |
| Volume profile | `coiled_cobra.evaluate_volume_profile_shelf` (30 bins, 0–25 score) vs `coiled_cobra_v2` (10 bins, boolean HVN check) | Same concept, incompatible APIs |

### Vestigial / misleading variables

- `trade_planner.py` docstring says `Source: 'swing' or 'coiled_cobra'`, but code sets `"Swing"` and `"Cobra"` — **branch for Cobra levels never fires**
- `expiry_label_min`, `expiry_label_max`, `contract_label` in `trade_planner.py` — assigned, never used
- `tabulate` in `requirements.txt` — not imported anywhere (pandas `.to_markdown()` used instead)
- `calculate_vibe_score()` `ticker` parameter — documented as unused (API compatibility vestige)

### Import path inconsistency (not dead, but friction)

Three different import strategies across modules:

- `data_ingestor.py`: `src.finance_vibe` → `finance_vibe` → manual `sys.path`
- `ticker_provider.py`: `finance_vibe` → bare `config`
- `analysis_engine.py`: `finance_vibe` → `sys.path` append

---

## 3. Key Issues & Technical Debt

### Critical logic bugs

**1. Coiled Cobra trade levels never applied**

In `trade_planner.py`, the branch checks `source == "coiled_cobra"`, but sources are assigned as `"Swing"` and `"Cobra"`. All Cobra setups fall through to swing EMA-based entry/stop logic, ignoring Fib 78.6% structural levels.

**2. `trade_planner.generate_trade_plan(scanner_csv_path=...)` will crash**

If `scanner_csv_path` is passed, `swing_csv_path` / `cobra_csv_path` are never defined, but referenced immediately after the `if scanner_csv_path is None` block → `NameError`.

**3. Macro layer disconnected from live pipeline**

`analysis_engine.py` is commented out in the orchestrator. Swing and Cobra scanners run **without** macro Vibe Score gating, while `pipeline_backtest.py` assumes that gate exists. Live output and backtest assumptions are misaligned.

---

### Vibe Score design issues

**Documented vs implemented weight model**

A common mental model cites Trend (±4), Momentum (±3), Volatility (±3). The implementation is:

- Trend ±4
- Momentum ±2 (not ±3)
- Timing ±2 (pullback distance — not labeled “Volatility”)
- CCI / RSI governors (not a ±3 volatility bucket)

**Asymmetry and interaction effects**

- Trend is binary (±4 or 0): no partial credit for `Close > SMA20` with SMA20 < SMA50
- MomentumDecay (−1) stacks with negative Momentum (−2) on the same bar
- RSI > 80 hard-caps to 5 **before** Persistence check — a score of 9 can collapse to 3 in one step
- CCI `< -200` awards +1 (contrarian) while `CCI > 200` penalizes −2 — intentional but asymmetric; boundary at exactly ±200 is exclusive for the +1 band
- Theoretical max before clip is ~10, but RSI cap + Persistence make scores ≥9 rare by design

**No live macro → tactical linkage**

`swing_scanner.py` docstring says macro context comes from `analysis_engine.py`, but nothing enforces it in production.

---

### Data ingestion edge cases

| Risk | Detail |
|------|--------|
| Sequential downloads | 250 tickers × individual `yf.download()` — slow, no batching, no retry/backoff |
| Broad `except Exception` | Failures logged as string only; no distinction between rate-limit, delisted, network |
| Empty DataFrame | Skipped with warning; ticker silently absent from downstream |
| Incomplete week trim | `weekday() != 4` assumes Friday close; ignores holidays, timezone, non-US listings |
| No validation gate | Missing High/Low/Volume not checked before save; breaks `coiled_cobra` volume profile |
| `auto_adjust=True` | Good for splits, but Volume is **not** split-adjusted by yfinance — volume profile math may be distorted historically |

---

### Performance bottlenecks

**Weekly 10y ≈ ~520 bars per ticker — generally fine for single-pass scans**, but:

| Hotspot | Complexity | Location |
|---------|------------|----------|
| `pipeline_backtest.backtest_ticker()` | **O(n²)** — `df.iloc[:i+1].copy()` every bar + full `build_features()` + `add_indicators()` | `pipeline_backtest.py:130-138` |
| `ProcessPoolExecutor` for ~250 small CSVs | Fork + IPC overhead may **exceed** benefit for ~500-row files | `analysis_engine.py:374` |
| `coiled_cobra` volume histogram | Per-ticker 30-bin histogram over 52/252 bars — negligible at current scale |
| `app.py` live prices | Per-symbol `fast_info` loop — N sequential network calls on dashboard load |

At current universe size (~250 tickers), production pipeline runtime is dominated by **yfinance ingestion**, not indicator math.

---

### Architectural drift

| Area | Drift |
|------|-------|
| Step numbering | `trade_planner` and `coiled_cobra` both print “STEP 5” |
| README vs `run_vibe.py` | README lists `analysis_engine` as step 4; orchestrator skips it |
| Docker CMD | Runs dashboard, not pipeline |
| Tests | Cover `pipeline_backtest` helpers and `trade_plan_helper` paths only — **zero tests** for Vibe Score math, CCI MAD, swing setup rules, or coiled cobra rubric |
| `coiled_cobra_v2.py` | Production-quality refactor (validation, argparse, factor weights) sitting unused beside legacy `coiled_cobra.py` |

---

### `coiled_cobra.py`-specific risks

- **Volume required** but not validated at ingest — `KeyError` caught generically as `execution_error`
- **Long-only output** — `SETUP_TYPE` always `SETUP_LONG`; no short-side macro reversal symmetry
- `evaluate_coiled_cobra` returns `None` for scores < 70 with no partial diagnostics
- `MARKDOWN_THRESHOLD` 0.85 weekly means price must be 15% below EMA20 — very restrictive

---

## 4. Actionable Next Steps (Prioritized)

### P0 — Correctness (do first)

1. **Fix Cobra source routing in `trade_planner.py`**  
   Align `Source` values (`"coiled_cobra"` / `"swing"`) or normalize with `.lower()` before branching so Fib-based levels actually apply.

2. **Re-enable or formally deprecate `analysis_engine.py` in `run_vibe.py`**  
   Either uncomment step 3 and optionally gate swing/Cobra on `vibe_report`, or update all docs/backtests to state macro is offline-only.

3. **Fix `trade_planner` `scanner_csv_path` code path**  
   Initialize `swing_csv_path` / `cobra_csv_path` when a single path is passed, or remove the unused parameter.

4. **Export options metadata or stop computing it**  
   Either add `LEAPS Expiry Min/Max`, `LEAPS Type`, `Delta Range` to `plan_rows`, or delete dead `calculate_options_expiry()` / delta logic to reduce confusion.

---

### P1 — Reliability

5. **Harden `data_ingestor.py`**  
   - Batch download: `yf.download(tickers, group_by='ticker')`  
   - Retry with exponential backoff on transient failures  
   - Validate required columns (`Date, Open, High, Low, Close, Volume`) before save  
   - Log structured failures (ticker, exception type, timestamp) to `data/logs/{mode}/ingest_errors_{date}.csv`

6. **Unify path resolution**  
   Refactor `swing_scanner`, `coiled_cobra`, `trade_planner` to use `config.get_mode_config(mode)` and `get_raw_path()` — single source of truth.

7. **Standardize imports**  
   One pattern: `from finance_vibe import config` with `PYTHONPATH=src` (already set in `run_vibe.py` and Dockerfile).

---

### P2 — Architecture cleanup

8. **Consolidate Coiled Cobra implementations**  
   Pick `coiled_cobra.py` (live) or `coiled_cobra_v2.py` (cleaner factor model). Delete or archive the other; wire the winner in `run_vibe.py` and `trade_planner` prefix constants.

9. **Extract shared indicator module**  
   `indicators.py` with one MACD/RSI/ATR implementation (prefer manual numpy for CCI MAD + pandas_ta or pure pandas for the rest). Eliminates macro vs tactical numeric drift and aids testing.

10. **Introduce a `scoring/` or `pipeline/` package**  
    Replace subprocess chaining in `run_vibe.py` with importable stage functions (`run_ticker_refresh()`, `run_ingest()`, `run_macro_scan()`, …). Enables unit testing, shared logging, and partial reruns without wiping raw data.

---

### P3 — Performance & validation

11. **Optimize `pipeline_backtest.py`**  
    - Incremental indicator updates instead of `iloc[:i+1].copy()` per bar  
    - Or vectorize setup detection where possible  
    Target: O(n) per ticker for walk-forward

12. **Revisit `ProcessPoolExecutor` threshold**  
    Benchmark: sequential vs parallel at 50/150/250 tickers. For small weekly frames, `ThreadPoolExecutor` or sequential may be faster.

13. **Add golden-file tests for Vibe Score**  
    Fixture CSVs with known indicator values → assert `Components` dict and final `Score`. Protects the MAD CCI and rubric ordering (RSI cap before Persistence).

14. **Add integration test for full pipeline**  
    Mock `yf.download` / screener; run stages in-process; assert output schemas and row counts.

---

### P4 — Product alignment

15. **Decide macro gating policy for live runs**  
    If backtest gate (long ≥7, short ≤−2) is validated, add optional `--macro-gate` to swing scanner reading latest `vibe_report_{date}.csv`.

16. **Align timeframe narrative**  
    Update external docs from “5-year weekly” to `10y / 1wk` per `config.py`, or change config if 5y is the intended lookback.

17. **Dashboard enhancements**  
    Surface `vibe_report`, `swing_setups`, and `coiled_cobra_setups` alongside trade plans; batch `yf.Tickers` price fetch.

---

## Summary Assessment

Finance Vibe has a **clear layered design** (universe → ingest → macro/tactical → execution → clean export) and several strong choices: vectorized MAD CCI, explicit scoring documentation (`Scoring_Logic.md`), mode-isolated data silos, and an offline backtest path.

The biggest risks today are **integration gaps**, not math complexity:

- Macro scoring is implemented well but **not run** in production
- Cobra setups are scanned but **trade levels ignore Cobra math** due to a source-string bug
- Two parallel Coiled Cobra engines and duplicate indicator stacks create **maintenance drag**
- Documentation, orchestrator, and backtest **disagree on what “the pipeline” actually is**

Fixing P0 items and re-enabling (or explicitly retiring) the macro layer would give the highest reliability return before any performance work.

---

## Related documentation

- `README.md` — project overview and pipeline flow
- `OperationManual.md` — operations and troubleshooting
- `src/finance_vibe/Scoring_Logic.md` — macro Vibe Score specification
- `swing_setup_readme.md` — tactical scanner reference
- `Planned future enhancements.md` — roadmap items
