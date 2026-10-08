# Finance Vibe: Modernization Plan

Status: **proposal** (2026-10-08). Nothing in this document has been implemented
yet. Each phase is a separate, reviewable change.

## 1. Summary and reality check

The brief asked for an upgrade to Python 3.12+, `uv`, and a high-performance
data stack (polars/pyarrow/duckdb), with ruff/mypy and async where useful.
After auditing the repo, the honest picture is:

| Assumption in the brief | What the repo actually has |
|---|---|
| Needs a Python 3.12 upgrade | **Already on 3.12.** `Dockerfile` uses `python:3.12-slim` (container runs 3.12.14), and `.devcontainer` uses the 3.12 image. Only the *code idioms* lag behind. |
| `analysis_engine_local.py` exists | **It doesn't.** The scoring engines are `analysis_engine.py` (macro Vibe Score) and `coiled_cobra.py` (setup rubric). |
| "Robust CCI with MAD" | `_cci_fast` (`analysis_engine.py:161`) is the **standard Lambert CCI** using *mean* absolute deviation, already vectorized with `numpy.sliding_window_view`. It is not a median-based "robust" CCI. Switching to median-AD would change Vibe Scores, so it belongs in a rubric/score decision, not a modernization pass. |
| CSV IO is a bottleneck that needs polars/duckdb | Raw data is **~12 MB weekly + ~30 MB daily across ~270 tickers**. A full CSV re-parse takes seconds. Wall-clock is dominated by **Yahoo network latency**, which is already batched (50 tickers per call, yfinance threads, retry and backoff; `data_ingestor.py:43-71`). |
| Async ingestion would help | yfinance already fetches in threads per batch, and Yahoo rate-limits aggressively. `asyncio` would add complexity and throttling risk for little gain. |

**Where modernization pays off is reproducibility, packaging and
maintainability, not raw speed.** Today's biggest real risk is that
dependencies are unpinned. The running container already floated to
**pandas 3.0.5 and numpy 2.2.6** without anyone choosing that, and the next
`docker compose build` can pull something different again.

**Not recommended now** (with triggers to revisit, see §5):
- rewriting the pipeline in polars;
- adding duckdb;
- asyncio ingestion;
- replacing CSV raw storage before a shared IO layer exists.

### Non-negotiables for every phase
1. **`Score` doesn't change.** Any phase whose output differs from the golden
   baseline (§3, Phase 0) stops. A deliberate score change needs a
   `config.RUBRIC_VERSION` bump, a rubric doc update and backtest validation,
   per `CLAUDE.md`.
2. **No lookahead.** `tests/test_as_of.py` stays green.
3. **Data contracts hold:** `<TICKER>_<period>_<interval>.csv` naming,
   `REQUIRED_OHLCV` and `SETUP_ROW_COLUMNS`. The cron health checks
   (`scripts/check_*_outputs.py`) and as-of replay depend on them.
4. **Daily and weekly paths stay separate**, as they are today.

---

## 2. Current-state inventory

### 2.1 Dependencies (`requirements.txt`, versions installed in the container)

| Package | Pin | Installed | Role | Risk / note |
|---|---|---|---|---|
| pandas | none | 3.0.5 | core | Major-version jump happened implicitly. Copy-on-write semantics are now default. |
| numpy | none | 2.2.6 | core | numpy 2 ABI; fine with the current wheels. |
| pandas_ta | none | 0.4.71b0 | indicators (`coiled_cobra.py`, `breakout_scanner.py`) | **Beta release with uncertain upkeep.** Only uses ema/sma/rsi/macd/atr/bbands/kc/obv. |
| yfinance | none | 1.7.0 | ingestion | Yahoo API churn is the top external risk. |
| yahooquery | none | 2.4.1 | screener (`ticker_provider.py`) | Second Yahoo client to keep working. |
| Flask, markdown, Pygments | none | 3.1.3 / 3.10.3 / 2.21.0 | UI + in-app handbook | Low. |
| tabulate | none | 0.10.0 | `DataFrame.to_markdown` console output | Low. |
| xgboost, lightgbm, scikit-learn, matplotlib | none | 3.4.1 / 4.7.0 / 1.9.1 / 3.11.2 | ML lab only (`ML_RANKING_ENABLED = False`) | Heavy. Installed in the production UI image even though ranking is off. |
| pytest | `>=8.0.0` | 9.1.1 | tests | Dev dependency shipped in the runtime image. Not installed on the host. |

Other gaps:
- no `pyproject.toml`, no lockfile;
- runtime, ML and dev dependencies are mixed together;
- the `Dockerfile` installs `build-essential`/`gcc`, which are probably unneeded now that every dependency ships wheels (`libgomp1` is still needed for lightgbm/xgboost);
- `.devcontainer` installs a *different*, partial dependency set (`postCreateCommand`).

### 2.2 Packaging and imports
- `src/finance_vibe/` has **no `__init__.py`**. It only imports as a namespace
  package via `PYTHONPATH=src`.
- **15 modules** carry `sys.path` fallbacks. Import order is inconsistent:
  `data_ingestor.py:10-22` tries `from src.finance_vibe import config` first,
  while the others try `finance_vibe` first.
- `run_vibe.py:183-190` runs each stage as `subprocess.run([sys.executable,
  script_path])`. This is good for crash isolation, but every stage re-imports
  pandas and re-reads raw CSVs.
- Import-time side effects: `coiled_cobra.py:32-40` parses `sys.argv` and sets
  module globals at import, which `apply_timeframe()` / `_calibrate()` then
  mutate. Library callers and tests have to know to call `apply_timeframe`.

### 2.3 Code quality

| Signal | Count / location |
|---|---|
| Lint / type / format config | none (no ruff, mypy, pre-commit) |
| `print(` vs `logging.getLogger` | 154 vs 3 (heaviest: `coiled_cobra_ml_training.py`, `run_vibe.py`, `trade_plan_helper.py`) |
| Broad `except Exception` | 41 |
| `from __future__ import annotations` | 16 files. `Optional[...]` still used 38× (UP007 would modernize these to `X \| None`) |
| `os.path` calls | 109 (`pathlib` is not used) |
| Largest module | `breakout_scanner.py`, 1,321 lines |
| Row loops | `trade_planner.py:245` `iterrows` (small frames); `breakout_scanner.py:1305` `apply(axis=1)` |

The numeric core is already in good shape:
- the indicators in `analysis_engine.py:128-200` are vectorized pandas/numpy;
- the RSI edge cases are documented and handled;
- per-ticker scans and backtests already use `ProcessPoolExecutor` (`analysis_engine.py:583`, `coiled_cobra_backtest.py:147,451`, the experiment modules).

### 2.4 Data pipeline and storage
- Ingestion (`data_ingestor.py`):
  - batch download;
  - drops the in-progress weekly bar using market-time rules (`weekly_bar_is_complete`);
  - validates against `config.validate_and_clean_ohlcv`;
  - writes one CSV per ticker;
  - logs failures to `ingest_errors_<date>.csv`.

  This works and is well-guarded. `daily_ingest.py` is the separate daily path.
- **Readers are fragmented.** About 13 modules call `pd.read_csv` on raw
  files, each with its own normalization (`analysis_engine.load_ohlc_csv`
  versus `config.validate_and_clean_ohlcv` versus ad-hoc code in the
  backtest/experiment modules). That makes it easy to get dtype, date or
  timezone handling subtly different across stages, which is a
  correctness risk more than a speed one.
- Storage size: 12 MB weekly + 30 MB daily raw, about 21 MB of weekly logs.
  pyarrow is **not installed**.

---

## 3. Phased roadmap

Effort is given in relative sizes: S under a day, M a few days, L about a week.

### Phase 0: Safety net (S, low risk). *Do this first.*
**Goal:** make every later phase provably behaviour-neutral.
1. **Freeze today's working environment.** Snapshot `pip freeze` from the
   running `finance_vibe` container into `constraints/2026-10-baseline.txt`.
   This is the known-good set for pandas 3.0.5 / numpy 2.2.6 / yfinance 1.7.0.
2. **Golden-output harness** (`scripts/golden_compare.py`):
   - copy a frozen snapshot of `data/raw/{weekly,daily}` into a fixture directory;
   - run each stage with `--as-of <fixed date>` against it (no network);
   - store the resulting `coiled_cobra_setups_*.csv`, Vibe Score and planner CSVs as the baseline;
   - diff with a float tolerance of 1e-9 on `Score` and the numeric columns.
3. Make the container test command a script (`scripts/test_in_container.sh`)
   so host machines without pytest have a one-liner.

**Exit:** the baseline is captured, the compare script passes on unchanged code, and pytest is green.

### Phase 1: Dependencies and packaging with `uv` (M, medium risk)
**Goal:** reproducible builds and normal imports.
1. Add `pyproject.toml`:
   - `requires-python = ">=3.12"`;
   - `[project.dependencies]` for runtime (pandas, numpy, pandas-ta, yfinance, yahooquery, Flask, markdown, Pygments, tabulate);
   - optional/group dependencies `ml` (xgboost, lightgbm, scikit-learn, matplotlib) and `dev` (pytest, ruff, mypy, pandas-stubs, pre-commit);
   - lower bounds taken from the Phase 0 baseline, e.g. `pandas>=3.0,<4`, `numpy>=2.2,<3`.
2. `uv lock`, then commit `uv.lock`. CI and Docker install with
   `uv sync --frozen`.
3. Add `src/finance_vibe/__init__.py` and a src-layout build backend, so the
   package installs (`uv sync` gives an editable install). Then delete the
   `sys.path` fallbacks module by module and unify on
   `from finance_vibe import ...`.
4. Console scripts (`finance-vibe = finance_vibe.run_vibe:main`,
   `finance-vibe-app = finance_vibe.app:main`). **Keep**
   `python src/finance_vibe/<stage>.py` working, because `run_vibe.py` and the
   cron runners call stages by path.
5. Dockerfile:
   - multi-stage build with `ghcr.io/astral-sh/uv` copied in;
   - `uv sync --frozen --no-dev`;
   - drop `build-essential`/`gcc` if the build passes without them, keeping `libgomp1`;
   - decide whether the UI image needs the `ml` group (the `trade_plan_helper` ML paths are gated off, so it may be able to skip it, which cuts several hundred MB).
6. `.devcontainer`: replace `postCreateCommand` with `uv sync --all-groups`.
7. Keep `requirements.txt` for one release, generated with
   `uv export --no-hashes`, then remove it.

**Verify:**
- pytest green;
- golden compare identical;
- `docker compose build && docker compose up -d` serves the UI;
- `scripts/run_weekly_pipeline.sh --now` with `--reuse-raw` succeeds inside the new image.

**Exit:** a fresh clone gets `uv sync && uv run pytest` working without `PYTHONPATH`.

### Phase 2: Code quality tooling (M, low risk to behaviour)
1. **ruff**, configured in `pyproject.toml`:
   - `target-version = "py312"`;
   - rules `E, F, I, UP, B, SIM, PD, RUF`, plus `PTH` later.
   - Roll it out as: `ruff check --fix` for the safe fixes in one commit, then `ruff format` in a separate formatting-only commit so `git blame` can ignore it via `.git-blame-ignore-revs`. Keep any remaining rules in `per-file-ignores` and burn them down over time.
2. **mypy**, gradually:
   - global `ignore_missing_imports` for yfinance/pandas_ta/yahooquery;
   - `strict = true` only for `config.py`, `analysis_engine.py`, `coiled_cobra.py` and `trade_planner.py`, the scoring and contract core;
   - add `pandas-stubs`;
   - widen the strict set one module per PR.
3. **pre-commit:** ruff, ruff-format, end-of-file and trailing whitespace. mypy
   runs in CI or the container, not on commit, because it's too slow with pandas stubs.
4. **Logging:** add a `finance_vibe/log.py` helper (one format, level from
   env). Replace `print` in library and stage modules. Keep user-facing
   console tables (`to_markdown`) as explicit output.
5. **Exceptions:** narrow the 41 broad `except Exception` at data-contract
   boundaries to `ValueError`/`KeyError`/`OSError`. Keep the outer per-ticker
   catch-all in the scan loops, but log it with `logger.exception` so
   tracebacks survive.

**Verify:** ruff and mypy are clean on the configured scope, pytest is green, and the golden compare is identical.

### Phase 3: Data layer (M-L, medium risk)
**Goal:** one correct way to read raw data. Faster storage is optional.
1. **`finance_vibe/io.py`:**
   - `load_raw(ticker, mode, *, as_of=None) -> pd.DataFrame` combines `config.get_raw_path`, `config.validate_and_clean_ohlcv` and `config.cut_to_as_of`;
   - `iter_raw(mode, *, as_of=None)` for scans.

   Migrate the ~13 `read_csv` call sites to it one by one. This is where
   lookahead and dtype bugs get removed, so it is worth doing even without Parquet.
2. **Parquet (optional, gated on measurement):**
   - add `pyarrow`;
   - have the ingestor write `<TICKER>_<period>_<interval>.parquet` **alongside** the CSV for one release, and have `io.load_raw` prefer Parquet when present;
   - switch the CSV to optional only after the golden compare and `check_*_outputs.py` both pass, and after updating `REQUIRED_OHLCV` docs and the health checks to understand Parquet.

   Expected gain is a few seconds per run. It is mainly worthwhile for the
   backtest/walk-forward jobs, which re-read the universe many times.
3. **dtypes:** with pyarrow present, pandas 3 can use Arrow-backed strings.
   Keep float64 for prices, because Arrow or float32 numerics would change
   indicator results in the last bits and so `Score`.
4. **Deferred, with triggers:**
   - **polars**: adopt for new research code only if the universe grows past roughly 5k tickers, intraday bars are added, or a profiled backtest shows pandas time dominates;
   - **duckdb**: adopt if ad-hoc cross-run analytics over `data/logs/` (backfills, experiment CSVs) become a regular workflow. It reads CSV and Parquet in place, so it can be added later without migrating storage.

**Verify:**
- golden compare identical (byte-for-byte on the setup CSV columns);
- `tests/test_as_of.py`;
- new `tests/test_io.py` covering the contract rejection, the as-of cut and CSV/Parquet parity.

### Phase 4: Architecture and performance (L, medium risk)
1. **No import-time CLI parsing.**
   - Give each stage `def main(argv: list[str] | None = None) -> int`.
   - Replace `coiled_cobra`'s mutable module globals (`LOOKBACK`, `COIL_BARS`, `TT_EMA_*`, and others) with a frozen `Timeframe` dataclass built by `Timeframe.for_mode("weekly"|"daily")` and passed explicitly.
   - Keep `apply_timeframe()` as a thin shim until callers migrate.
2. **Orchestration:** `run_vibe.py` gets a declarative stage list
   (`Stage(name, module, modes)`). It keeps subprocess execution by default,
   for isolation and identical cron behaviour, and adds an `--in-process`
   flag for debugging and faster local runs.
3. **Split `breakout_scanner.py`** into indicators, state classification and
   output modules. No logic change; the golden compare guards it.
4. **Indicator ownership:** add a parity test of `pandas_ta` versus in-house
   implementations for the 8 functions used. If they match within 1e-12,
   drop the `pandas-ta` beta dependency. If they don't, keep it pinned.
   **Any drift is a score change** and needs a `RUBRIC_VERSION` bump.
5. **Performance, profile first:** run `py-spy`/`cProfile` on
   `coiled_cobra_backtest` and `coiled_cobra_ml_walkforward`, the only
   CPU-heavy paths, and optimize what the profile shows. Likely candidates:
   - caching indicator frames per ticker across as-of dates instead of recomputing per date;
   - vectorizing `trade_planner.py:245` (`iterrows`) and `breakout_scanner.py:1305` (`apply(axis=1)`), which is cheap and safe.
6. **Async: still no** for ingestion. Revisit only if moving off yfinance to
   an HTTP API with documented rate limits.

**Verify:** golden compare, pytest, and timing before/after recorded in the PR.

---

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Unpinned dependencies change on rebuild | **High (already happened)** | Silent score or behaviour changes | Phase 0 constraints file, then the Phase 1 `uv.lock` |
| yfinance/Yahoo API changes (empty bars, NaN latest bar, schema) | High | Ingestion failures, stale data | Existing contract validation and health checks; pin yfinance and upgrade deliberately |
| pandas 3 copy-on-write / dtype semantics | Medium | Chained-assignment code silently stops mutating | ruff `PD` rules; pytest; golden compare |
| Indicator numeric drift (pandas_ta swap, float32/Arrow numerics) | Medium | `Score` changes, mixing rubric vintages in training data | Float64 everywhere; parity tests; `RUBRIC_VERSION` rule |
| Container / cron coupling (`scripts/run_*_pipeline.sh` `docker exec` stage paths) | Medium | Scheduled runs break after a Docker or packaging change | Keep path-based stage invocation; run `--now --reuse-raw` after every image change |
| Large mechanical diffs (ruff format, sys.path removal) | Low | Review fatigue, hidden logic edits | Format-only commits, `.git-blame-ignore-revs`, one concern per PR |

## 5. Suggested sequencing

```
Phase 0 (S) ─▶ Phase 1 (M) ─▶ Phase 2 (M) ─▶ Phase 3.1 io.py (M) ─▶ Phase 4.1/4.2 (M)
                                               └─▶ Phase 3.2 Parquet (S, optional, measured)
                                                   Phase 4.4 pandas_ta parity (S-M)
                                                   Phase 4.5 profiling-driven perf (as needed)
```

Phases 0-2 give most of the value: reproducible builds, normal imports and
guardrails. Phases 3-4 are maintainability work, to be scheduled when the
pipeline needs to grow.
