# CLAUDE.md

Guidance for Claude when working in this repo. Keep it short and current; deep
detail lives in `docs/`.

## What this is

Finance Vibe runs **two stock scanners**, Coiled Cobra (coil → expansion
scorecard) and breakout readiness (descriptive states), over a Yahoo OHLCV
universe in weekly and daily modes, and serves their dated CSVs in a Flask
dashboard. It produces *signals* for review, not trades.

## Layout

- `src/finance_vibe/` — `config`, `log`, `indicators`, `market` (QQQ benchmark,
  RS, Cobra market gate), `pipeline` (orchestrator)
- `src/finance_vibe/data/` — `tickers`, `ingest` (Yahoo download, both modes),
  `daily_ingest`, `raw` (the raw CSV loader)
- `src/finance_vibe/scanners/` — `coiled_cobra`, `breakout` (`scan_history()` gives
  every bar's breakout states/status/score, same rules as the scan)
- `src/finance_vibe/web/` — Flask app, `playbook.py` (Top Setups ranking, Market tab bias, long/short levels), `templates/`, `static/`
- `tests/` — pytest suite
- `docs/` — `coiled_cobra.md`, `breakout.md` (scanner specs), `operations.md`
- `data/` — gitignored. `data/raw/{weekly|daily}/`, `data/logs/{weekly|daily}/`, `data/active_tickers.csv`
- `scripts/` — cron runners, output health checks, test/lint/golden helpers

## Commands

```bash
uv sync                                    # deps from uv.lock + editable install (dev group by default)
uv lock                                    # after editing pyproject.toml deps; commit uv.lock
# without uv: pip install -r requirements.txt && export PYTHONPATH=src
# commands below assume the .venv is active (or prefix with `uv run`)

python -m pytest -q                        # full suite, ~70s (or: uv run pytest -q)
python -m pytest -q -m "not slow"          # skip the brute-force forming-SMA oracle, ~50s
python -m pytest tests/test_coiled_cobra.py -q

python -m finance_vibe.pipeline                         # weekly (default)
python -m finance_vibe.pipeline --mode daily
python -m finance_vibe.pipeline --reuse-raw             # skip wipe + ticker refresh + ingest
python -m finance_vibe.pipeline --as-of 2025-11-07      # replay from raw on disk (implies --reuse-raw)
python -m finance_vibe.pipeline --reuse-raw --in-process  # stages as function calls (debugging)

python -m finance_vibe.web.app                          # UI at http://127.0.0.1:5000
```

Every stage also runs standalone, e.g. `python -m finance_vibe.scanners.coiled_cobra weekly`.

## Pipeline (`pipeline.py`, the source of truth)

Stages are declared in `pipeline.STAGES` (dotted module names) and run as
`python -m <module>` subprocesses (or `--in-process` via each stage's
`main(argv)`), in order: wipe `data/raw/{mode}/` (unless `--reuse-raw`/`--as-of`)
→ `data.tickers` → `data.ingest` (weekly) or `data.daily_ingest` (daily; drops
today's bar before 17:00 ET) → `scanners.coiled_cobra` → `scanners.breakout`.
Outputs land in `data/logs/{mode}/` with a `_<date>` suffix.

If a doc disagrees with `pipeline.py` or `config.py`, the code wins.

## Rules that matter

- **Never run the full pipeline casually.** Without `--reuse-raw` it deletes
  `data/raw/{mode}/` and re-downloads hundreds of tickers from Yahoo. For
  experiments use `--reuse-raw` or call a single stage.
- **No lookahead.** Anything touching `--as-of` must only use bars completed
  on/before the as-of date. Weekly
  as-of on a non-Friday excludes the week in progress. `tests/test_as_of.py`
  guards this; keep it green.
- **Config lives in `config.py`.** Profiles (`TIMEFRAME_PROFILES`), schemas (`REQUIRED_OHLCV`, `SETUP_ROW_COLUMNS`) and paths
  belong there, not hard-coded in stage modules.
- **Rubric versioning.** If you change Coiled Cobra gate/pillar logic or
  thresholds in a way that changes `Score` or which setups qualify, bump
  `config.RUBRIC_VERSION` and update `docs/coiled_cobra.md`.
  Breakout scanner state/status/score changes update `docs/breakout.md`. `tests/test_rubric_docs_sync.py`
  fails when a quoted threshold in either doc drifts from the code.
- **Data contracts.** Raw CSVs are named `<TICKER>_<period>_<interval>.csv`
  and must satisfy `REQUIRED_OHLCV`; setup rows must match
  `SETUP_ROW_COLUMNS`. Reject malformed input loudly rather than mis-scoring.
  Read raw files only through `data.raw.load_raw` (contract, date cleanup,
  float64 prices and the as-of cut in one place), never a bare `read_csv`.
- **Signals, not advice.** Keep scanner output informational. The dashboard's
  Top Setups page (`web/playbook.py`) shows rule-based trigger / stop / 2R-3R
  levels; don't add options metadata or position sizing without being asked.

## Coding standards

- Python 3.12 (the container is the reference environment: pandas 3.x,
  numpy 2.x). Dependencies live in `pyproject.toml` and are pinned in
  `uv.lock`; `requirements.txt` is a `uv export` of it, regenerate it after a
  lock change. Upgrading a pinned package is a golden-compare event.
- Use `from __future__ import annotations`, builtin generics and `X | None`;
  prefer `pathlib` in new code.
- Use `logging`, not `print`, in library code. Console tables
  (`to_markdown`) in stage `__main__` paths are fine.
- Import as `from finance_vibe import ...`. Don't add new `sys.path`
  fallbacks.
- Keep numeric code vectorized (pandas/numpy, float64 prices). Don't use
  `iterrows`/`apply(axis=1)` in per-ticker hot paths. Reuse `indicators.py`
  (EMA/SMA/RSI/ATR/MACD/BB/KC/OBV, frozen against pandas-ta by
  `tests/test_indicators_parity.py`) and `market.py` instead of writing new
  ones; numeric drift changes `Score`.
- Catch specific exceptions (`ValueError`, `KeyError`, `OSError`) at IO and
  data-contract boundaries. Broad catches only in per-ticker scan loops, and
  log them with the traceback.
- Don't add heavy dependencies (polars, duckdb, async frameworks) without a
  measured need.
- ruff (lint + format, config in `pyproject.toml`) and mypy must stay clean.
  mypy is strict for `config`, `market`, `scanners.coiled_cobra`; widen that
  list one module at a time. Prefer a targeted
  `# type: ignore[code]` over `cast()`/runtime coercion for pandas-stubs noise.
- Logging: `logger = logging.getLogger(__name__)` in modules;
  `finance_vibe.log.setup_logging()` once per `__main__`. Never configure
  logging at import time.

## Testing without host pytest

The host may not have pytest. These scripts copy the working tree into the
`finance_vibe` container and run there:

```bash
scripts/test_in_container.sh                    # pytest (any pytest args pass through)
scripts/test_in_container.sh -m "not slow"      # fast run; run the full suite before committing
scripts/golden_in_container.sh compare          # golden-output check, ~1.5 min
scripts/lint_in_container.sh                    # ruff check + ruff format --check + mypy
scripts/lint_in_container.sh ruff check --fix   # or any command in the lint venv
```

The lint tools live in the uv `lint` group, which the production image does not
install; `lint_in_container.sh` runs them in the uv image with a venv cached in
`~/.cache/finance-vibe-lint`. `.pre-commit-config.yaml` has ruff + whitespace
hooks (no mypy). `git config blame.ignoreRevsFile .git-blame-ignore-revs`
hides the format commit from blame.

`golden_in_container.sh` replays the weekly and daily pipelines with `--as-of` on
a frozen raw-data fixture (`/mnt/fast/finance-vibe-data/golden/`). It diffs every
output CSV, plus a scorecard of every ticker including rejects, against the
stored baseline. Run `compare` before and after any refactor, dependency bump or
data-layer change; it must print `IDENTICAL`. Only after a deliberate,
rubric-versioned score change should you run `baseline --force` to re-record.
`snapshot --force` re-freezes the raw data; always re-baseline right after it.

## Working style

- Before deleting or "cleaning up" code, search all of `src/`, `tests/`,
  `scripts/`, Docker files and docs for references (stages are invoked by
  module name from `pipeline.STAGES`, so imports alone won't show usage).
  Present findings and wait for confirmation before removing things
  (mirrors `.cursorrules`).
- Add or update a test in `tests/` with any logic change, and run the
  relevant tests before calling work done.
- When behaviour changes, update the matching doc in `docs/` and the README
  tables in the same change.

## Deployment

`docker compose up -d` serves the Flask UI on port 5000 (container default
command is `python -m finance_vibe.web.app`, not the pipeline). Host data
volume: `/mnt/fast/finance-vibe-data` → `/app/data`. TZ is `America/New_York`.
Cron runs the pipeline in the container via `scripts/run_{daily,weekly}_pipeline.sh`
(see `docs/operations.md`); rebuild the image after code changes.
