# CLAUDE.md

Guidance for Claude when working in this repo. Keep it short and current; deep
detail lives in `docs/`.

## What this is

Finance Vibe is a Python pipeline for **coil → expansion stock signals**: build a
ticker universe, ingest OHLCV from Yahoo, score Coiled Cobra setups, and attach
informational entry / stop / target context. It produces *signals*, not trade
or options positions. It is also a quant/ML lab (see `docs/labs/`).

## Layout

- `src/finance_vibe/` — all application code (package `finance_vibe`)
- `templates/`, `src/finance_vibe/static/` — Flask UI
- `tests/` — pytest suite (imports `from finance_vibe import ...`)
- `docs/handbook/` — theory and rubrics; `docs/architecture/` — pipeline, ops, ML; `docs/labs/` — experiments
- `data/` — gitignored. `data/raw/{weekly|daily}/`, `data/logs/{weekly|daily}/`, `data/active_tickers.csv`
- `notebooks/` — exploration only, not part of the pipeline

## Commands

```bash
uv sync                                    # deps from uv.lock + editable install (ml, dev groups by default)
uv lock                                    # after editing pyproject.toml deps; commit uv.lock
# without uv: pip install -r requirements.txt && export PYTHONPATH=src
# commands below assume the .venv is active (or prefix with `uv run`)

python -m pytest -q                        # full suite, ~30s (or: uv run pytest -q)
python -m pytest tests/test_coiled_cobra.py -q

python src/finance_vibe/run_vibe.py                        # weekly (default)
python src/finance_vibe/run_vibe.py --mode daily
python src/finance_vibe/run_vibe.py --reuse-raw            # skip wipe + ticker refresh + ingest
python src/finance_vibe/run_vibe.py --as-of 2025-11-07     # replay from raw on disk (implies --reuse-raw)

python src/finance_vibe/app.py                             # UI at http://127.0.0.1:5000
```

Every stage script also runs standalone with a mode argument, e.g.
`python src/finance_vibe/coiled_cobra.py weekly`.

## Pipeline (`run_vibe.py`, the source of truth)

Stages run as subprocesses, in order: wipe `data/raw/{mode}/` (unless
`--reuse-raw`/`--as-of`) → `ticker_provider` → `data_ingestor` (weekly) or
`daily_ingest` (daily; drops today's bar before 17:00 ET) →
`analysis_engine` (macro Vibe Score) → `coiled_cobra` → `breakout_scanner` →
`trade_planner` → `trade_plan_helper`. All outputs land in
`data/logs/{mode}/` with a `_<date>` suffix.

If a doc disagrees with `run_vibe.py` or `config.py`, the code wins. Some
architecture docs are stale (e.g. `project_resurrection_prompt.md` and
`code_review.md` still describe a `swing_scanner.py` that no longer exists).
`docs/handbook/coiled_cobra_rubric.md` was re-audited against
`coiled_cobra.py` on 2026-10-08.

## Rules that matter

- **Never run the full pipeline casually.** Without `--reuse-raw` it deletes
  `data/raw/{mode}/` and re-downloads hundreds of tickers from Yahoo. For
  experiments use `--reuse-raw` or call a single stage.
- **No lookahead.** Anything touching `--as-of`, backtests, backfills or ML
  features must only use bars completed on/before the as-of date. Weekly
  as-of on a non-Friday excludes the week in progress. `tests/test_as_of.py`
  guards this; keep it green.
- **Config lives in `config.py`.** Thresholds, profiles (`TIMEFRAME_PROFILES`,
  `SWING_PROFILES`), schemas (`REQUIRED_OHLCV`, `SETUP_ROW_COLUMNS`) and paths
  belong there, not hard-coded in stage modules.
- **Rubric versioning.** If you change Coiled Cobra gate/pillar logic or
  thresholds in a way that changes `Score` or which setups qualify, bump
  `config.RUBRIC_VERSION` and update `docs/handbook/coiled_cobra_rubric.md`.
  Training data, model artifacts and inference refuse to mix vintages.
- **ML ranking stays off.** `ML_RANKING_ENABLED = False` because walk-forward
  found no out-of-sample edge over raw Score. Don't flip it without the
  paired ML-minus-Score IC interval excluding 0 (see
  `docs/architecture/coiled_cobra_ml.md`).
- **Data contracts.** Raw CSVs are named `<TICKER>_<period>_<interval>.csv`
  and must satisfy `REQUIRED_OHLCV`; setup rows must match
  `SETUP_ROW_COLUMNS`. Reject malformed input loudly rather than mis-scoring.
- **Signals, not advice.** Keep planner output informational (entry/stop/2R/3R
  targets); don't add options metadata or position sizing without being asked.

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
  `iterrows`/`apply(axis=1)` in per-ticker or backtest hot paths. Reuse the
  indicator helpers in `analysis_engine.py` / `coiled_cobra.py` instead of
  writing new ones; numeric drift changes `Score`.
- Catch specific exceptions (`ValueError`, `KeyError`, `OSError`) at IO and
  data-contract boundaries. Broad catches only in per-ticker scan loops, and
  log them with the traceback.
- Don't add heavy dependencies (polars, duckdb, async frameworks) without a
  measured need. See `MODERNIZATION_PLAN.md` for the phased roadmap. Add
  `uv`/ruff/mypy commands to this file only once those phases land.

## Testing without host pytest

The host may not have pytest. These scripts copy the working tree into the
`finance_vibe` container and run there:

```bash
scripts/test_in_container.sh                    # pytest (any pytest args pass through)
scripts/golden_in_container.sh compare          # golden-output check, ~1.5 min
```

`golden_in_container.sh` replays the weekly and daily pipelines with `--as-of` on
a frozen raw-data fixture (`/mnt/fast/finance-vibe-data/golden/`). It diffs every
output CSV, plus a scorecard of every ticker including rejects, against the
stored baseline. Run `compare` before and after any refactor, dependency bump or
data-layer change; it must print `IDENTICAL`. Only after a deliberate,
rubric-versioned score change should you run `baseline --force` to re-record.
`snapshot --force` re-freezes the raw data; always re-baseline right after it.

## Working style

- Before deleting or "cleaning up" code, search all of `src/`, `tests/`,
  `templates/`, `scripts/`, Docker files and docs for references (stages are
  invoked by path from `run_vibe.py`, so imports alone won't show usage).
  Present findings and wait for confirmation before removing things
  (mirrors `.cursorrules`).
- Add or update a test in `tests/` with any logic change, and run the
  relevant tests before calling work done.
- When behaviour changes, update the matching doc in `docs/` and the README
  tables in the same change.

## Deployment

`docker compose up -d` serves the Flask UI on port 5000 (container default
command is `app.py`, not the pipeline). Host data volume:
`/mnt/fast/finance-vibe-data` → `/app/data`; `docs/` is mounted read-only for
the in-UI handbook. TZ is `America/New_York`.
