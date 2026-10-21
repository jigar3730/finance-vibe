# Operations

How to run, schedule and maintain the Finance Vibe pipeline: ticker universe,
Yahoo OHLCV ingest, the two scanners (Coiled Cobra and breakout readiness), and
the dashboard.

## Environment

- Python 3.12 (the Docker image) or 3.13
- Install from repo root with `uv sync` (or `pip install -r requirements.txt` and `export PYTHONPATH=src`)
- Core packages: `pandas`, `numpy`, `yfinance`, `yahooquery`, `Flask`. Indicators are in-house (`finance_vibe/indicators.py`).

## Layout

| Path | Contents |
| ---- | -------- |
| `src/finance_vibe/` | `config`, `log`, `indicators`, `market` (benchmark, RS, market gate), `pipeline` |
| `src/finance_vibe/data/` | `tickers` (universe), `ingest` (Yahoo download), `daily_ingest`, `raw` (raw CSV loader) |
| `src/finance_vibe/scanners/` | `coiled_cobra`, `breakout` |
| `src/finance_vibe/web/` | Flask dashboard (`app`, `templates/`, `static/`) |
| `scripts/` | Cron runners, output health checks, email alerts, test/lint/golden helpers |
| `data/active_tickers.csv` | Ticker universe |
| `data/raw/{weekly,daily}/` | Raw OHLCV CSVs (`*_10y_1wk.csv`, `*_5y_1d.csv`) |
| `data/logs/{weekly,daily}/` | Dated scan outputs and ingest errors |

## Pipeline

```bash
python -m finance_vibe.pipeline                          # weekly (default)
python -m finance_vibe.pipeline --mode daily
python -m finance_vibe.pipeline --mode daily --reuse-raw # keep raw OHLCV; skip wipe + ingest
python -m finance_vibe.pipeline --as-of 2025-11-07       # replay a past date
```

| Step | Module | Output |
| ---- | ------ | ------ |
| 0 | (pipeline) | wipes `data/raw/{mode}/` unless `--reuse-raw` / `--as-of` |
| 1 | `data.tickers` | `data/active_tickers.csv` (skipped with `--reuse-raw`) |
| 2 | `data.ingest` (weekly) / `data.daily_ingest` (daily) | raw CSVs in `data/raw/{mode}/` (skipped with `--reuse-raw`) |
| 3 | `scanners.coiled_cobra` | `coiled_cobra_setups_<date>.csv` |
| 4 | `scanners.breakout` | `breakout_setups_<date>.csv` |

Each stage also runs on its own:

```bash
python -m finance_vibe.data.tickers
python -m finance_vibe.data.ingest weekly
python -m finance_vibe.data.daily_ingest                 # takes no mode
python -m finance_vibe.scanners.coiled_cobra weekly
python -m finance_vibe.scanners.breakout weekly
```

### Stages

- **`data.tickers`** merges `STATIC_TICKERS` (`config.py`),
  `data/ticker_manifest.csv` and Yahoo screeners (`SCREENER_IDS`, `SCREENER_COUNT`
  each) into `data/active_tickers.csv`, capped at `ACTIVE_TICKER_CAP` (1000). If the
  screeners fail, the static baseline is saved so the pipeline is never blocked.
- **`data.ingest`** downloads every active ticker with `yfinance` per
  `TIMEFRAME_PROFILES`, split/dividend-adjusted (`auto_adjust=True`), in batches with
  retry and backoff. It drops the current weekly candle while its week is still
  trading (complete from Friday 17:00 ET, `ingest.drop_incomplete_weeks` drops every
  trailing row of the week in progress) and logs
  per-ticker failures to `ingest_errors_<date>.csv`.
- **`data.daily_ingest`** runs `ingest.ingest_market_data("daily")`, then drops
  today's bar from every daily CSV while the session is still trading (final from
  17:00 ET, `DAY_FINAL_HOUR_ET`), so a mid-session run scans yesterday's close. It also
  holds the NYSE holiday calendar (`last_complete_session`) used by the daily health check.
- **`scanners.coiled_cobra`**: 100-point coil → expansion scorecard with four hard
  gates. Specification: [`coiled_cobra.md`](coiled_cobra.md).
- **`scanners.breakout`**: breakout readiness states, statuses and a 100-point score
  for every ticker. Descriptive only. Specification: [`breakout.md`](breakout.md).

### Replaying a past date (`--as-of`)

`--as-of YYYY-MM-DD` re-runs both scanners as if it were that date, from the raw data
already on disk. It implies `--reuse-raw`, passes the date to each scanner, and stamps
every output with it, so the dashboard shows the replayed run like any other.

- **No lookahead.** Only bars *complete* on the as-of date are used. Weekly bars are
  Monday-dated but hold the whole week, so a weekly bar counts only once its Friday is
  on or before the as-of date: `--as-of 2025-11-07` scans the bar dated `2025-11-03`,
  while a mid-week date such as `2025-11-05` scans the previous week. Daily bars count
  on their own date. The QQQ benchmark is cut the same way.
- **Limits.** It uses *today's* `active_tickers.csv` (not the historical universe) and
  *today's* split/dividend-adjusted prices. Names under the history floor as of that
  date are not scored.
- The scanners accept the same flag, e.g.
  `python -m finance_vibe.scanners.coiled_cobra weekly --as-of 2025-11-07`.
  Tests: `tests/test_as_of.py`.

## Scheduling

Two cron jobs (host crontab, user `jigar`) run the pipeline inside the `finance_vibe`
container through wrapper scripts. Each wrapper exits unless it is the right hour in
New York, so the cron line only has to cover that hour in the host's clock:

```cron
# host clock America/New_York (mediabox)
30 18,22 * * 1-5 /opt/stacks/finance-vibe/scripts/run_daily_pipeline.sh
0 18 * * 5       /opt/stacks/finance-vibe/scripts/run_weekly_pipeline.sh
# host clock UTC: fire at every UTC hour those slots can map to (EDT and EST)
30 22,23,2,3 * * * /opt/stacks/finance-vibe/scripts/run_daily_pipeline.sh
0 22,23 * * 5      /opt/stacks/finance-vibe/scripts/run_weekly_pipeline.sh
```

- **Weekly: Fridays 18:00 ET.** The ingestor keeps the week's bar once that Friday's
  close has passed, so a Friday-evening run scans the week that just closed.
- **Daily: Monday–Friday 18:30 ET, retry at 22:30 ET.** Daily bars are final from
  17:00 ET, and 18:30 starts after the Friday weekly run. Yahoo sometimes publishes the
  close late (on 2026-10-07 it was still blank at 21:00 ET), so the 22:30 slot re-runs
  the pipeline unless the 18:30 run passed its health check (it leaves
  `daily_ok_<date>`). Weekends are skipped; market holidays still run, and the health
  check then expects the previous session's bar.
- **Locks.** Each runner holds a lock so runs cannot overlap; the daily runner also
  waits for the weekly lock (up to 30 min), since both refresh `active_tickers.csv`.
- **Logs.** One file per run in `~/.local/state/finance-vibe/`
  (`daily_<date>_<time>.log`, `weekly_<date>_<time>.log`). Stage status lines are
  `logging` records on stderr and console tables go to stdout; the log captures both.
  Set `FINANCE_VIBE_LOG_LEVEL=DEBUG` on the `docker exec` for more detail.
- **Manual runs.** `scripts/run_daily_pipeline.sh --now` / `run_weekly_pipeline.sh --now`
  skip the time check; a passing manual daily run also satisfies that day's schedule.

### Health checks and alerts

After the pipeline exits 0, `scripts/check_{daily,weekly}_outputs.py` checks the run's
outputs: both scanners' CSVs for today exist, SPY/QQQ's newest bar is the last closed
session (daily) or the week that just closed (weekly), and at most 20% of tickers failed
to ingest. Run one by hand with
`docker exec -i finance_vibe python - < scripts/check_daily_outputs.py`.

`scripts/notify_email.py` (stdlib only, runs on the host) emails an alert with the end
of the run log when the pipeline fails or times out (daily: 25 minutes), when a run is
skipped because a lock is held, or when the health check fails (daily: only if the
22:30 retry or a `--now` run also fails). SMTP settings live in
`~/.config/finance-vibe/notify.env` (mode 600, not in the repo; same `SMTP_*` /
`EMAIL_*` names as quant-hub). Without that file the alert is logged as not sent.

## Dashboard

```bash
python -m finance_vibe.web.app      # http://127.0.0.1:5000
```

`docker compose up -d` runs it in the `finance_vibe` container on port 5000 with
`/mnt/fast/finance-vibe-data` mounted at `/app/data`. The container runs the code baked
into its image: after a code change, rebuild with `docker compose up -d --build` so the
dashboard and the next scheduled run pick it up.

| Route | Contents |
| ----- | -------- |
| `/` | Run dates per mode, linking each date's Coiled Cobra and breakout pages |
| `/cobra/<mode>/<date>` | Coiled Cobra setups, highest Score first, live price vs Close |
| `/breakout/<mode>/<date>` | Breakout scan: KPIs, status chips, highlights (or `?show=all` / `?status=`), failed breakouts collapsed |

Live prices come from `yfinance` `fast_info`, fetched 16 at a time and cached for 60
seconds. A Coiled Cobra row whose live price is more than 5% above its close is
highlighted.

## Maintenance

- **Re-download one mode:** `python -m finance_vibe.pipeline --mode weekly` (the run
  wipes `data/raw/weekly/` first).
- **Clear old data:** `scripts/clean_data_volume.sh --dry-run`, then
  `--target raw|logs|all`. Raw files come back on the next run; logs do not.
- **Add tickers:** edit `STATIC_TICKERS` in `config.py` or add rows to
  `src/finance_vibe/data/ticker_manifest.csv`.
- **Change lookback or cadence:** edit `TIMEFRAME_PROFILES` in `config.py`.
- **Change scanner rules:** edit `evaluate_coiled_cobra()` in `scanners/coiled_cobra.py`
  (update [`coiled_cobra.md`](coiled_cobra.md) and bump `config.RUBRIC_VERSION` when
  `Score` or qualification changes) or `classify_status()` / `score_readiness()` in
  `scanners/breakout.py` (update [`breakout.md`](breakout.md)).
  `tests/test_rubric_docs_sync.py` fails when a quoted threshold drifts from the code.

## Troubleshooting

| Symptom | Action |
| ------- | ------ |
| Missing `active_tickers.csv` | `python -m finance_vibe.data.tickers` |
| Ingest skips a symbol | No yfinance data for that ticker; check `ingest_errors_<date>.csv` |
| Empty `coiled_cobra_setups_*.csv` | No ticker passed the gates (expected in quiet markets) |
| Daily health check: stale SPY/QQQ bar | Yahoo has not published the close yet; the 22:30 retry handles it |
| Dashboard shows old behaviour | The container runs its image; rebuild with `docker compose up -d --build` |
