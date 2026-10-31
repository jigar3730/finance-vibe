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
  trailing row of the week in progress, and `one_row_per_week` keeps only the
  Monday-dated bar when yfinance adds a same-week row) and logs
  per-ticker failures to `ingest_errors_<date>.csv`.
- **`data.daily_ingest`** runs `ingest.ingest_market_data("daily")`, then drops
  today's bar from every daily CSV while the session is still trading (final from
  17:00 ET, `DAY_FINAL_HOUR_ET`), so a mid-session run scans yesterday's close. It also
  holds the NYSE holiday calendar (`last_complete_session`) used by the daily health check.
- **Bar-quality report (`data.quality`).** After each ingest (weekly: `ingest.main`;
  daily: `daily_ingest.ingest_daily`, after today's bar is dropped) every raw CSV is
  checked and each issue is written to `data_quality_<date>.csv`
  (`Symbol, Check, Severity, Date, Detail`; header only when clean). It flags and never
  changes what is scanned. Thresholds are the `DQ_*` constants in `config.py`:
  - `stale` (error): the newest bar is older than the last closed session / week.
  - `ohlc_incoherent` (error): in the last 20 bars, High below max(Open, Close), Low
    above min(Open, Close), or a price ≤ 0. Usually Yahoo's provisional last bar.
  - `zero_volume` (warn): a bar with no volume in the last 20.
  - `flat_closes` (warn): 5+ identical closes in a row (placeholder bars).
  - `big_move` (warn): a one-bar close move of 40% or more (a split or bad print, or a
    real gap).
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
session (daily) or the week that just closed (weekly), at most 20% of tickers failed
to ingest, and `data_quality_<date>.csv` exists with no error rows for the Market tab
ETFs (other tickers' errors are only counted). Run one by hand with
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
| `/` | Top setups by evidence: weekly BREAKOUT_CONFIRMED cards (unproven), untested Coiled Cobra setups (collapsed), counts of signals that failed in-sample |
| `/market` | SPY / QQQ / SCHG / SCHD: relative view (returns, RS vs SPY, growth-vs-dividend label), long / short / neutral bias with the weekly and daily states, a daily chart with Support / Resistance, row freshness |
| `/changes` | What changed since the previous run, per mode: breakout status moves, Coiled Cobra setups new / dropped / kept, Market ETF bias |
| `/runs` | Run dates per mode, linking each date's Coiled Cobra and breakout pages |
| `/api/bars/<mode>/<SYMBOL>` | JSON OHLC bars from the raw silo (last 60 daily / 52 weekly) for the level charts |
| `/cobra/<mode>/<date>` | Coiled Cobra setups, highest Score first, live price vs Close |
| `/breakout/<mode>/<date>` | Breakout scan: KPIs, status chips, highlights (or `?show=all` / `?status=`), failed breakouts collapsed |

Live prices come from `yfinance` `fast_info`, fetched 16 at a time and cached for 60
seconds. A Coiled Cobra row whose live price is more than 5% above its close is
highlighted.

### Top setups (`/`, `web/playbook.py`)

The page sorts the newest runs by what the backtest found (`RESEARCH.md` §7, §11). Nothing
is ranked or scored, and there are no trade levels.

1. **Weekly CONFIRMED** (`playbook.weekly_confirmed`): one card per weekly
   `BREAKOUT_CONFIRMED` row, by symbol. Each says "Unproven: passed old-data test only"
   (it passed in-sample; the holdout runs on 2035-09-07) and "Tested as: buy next open,
   hold 4 weeks, no stop." A card also says "Also CONFIRMED last week" when the previous
   weekly breakout CSV had the symbol CONFIRMED: a continuation, not the first-week event
   that was tested. A symbol missing from the previous file (universe churn) is not marked.
   Cards show the close or live price, ATR, `Resistance`, `Support`, extension, RVOL20,
   SMA50 and MTF.
2. **Untested: Coiled Cobra setups** (`playbook.untested_cobra`): collapsed by default; a
   table per mode with Score, Grade, Close and the ticker's breakout status.
3. **Failed the old-data test** (`playbook.failed_counts`): counts only, each linking to
   the breakout page filtered to that status. Daily: `PRE_BREAKOUT`, `BREAKOUT_TRIGGERED`,
   `BREAKOUT_CONFIRMED`. Weekly: `PRE_BREAKOUT`, `BREAKOUT_TRIGGERED`.

Each card has a level chart: recent weekly candles from `/api/bars` with `Resistance` and
`Support` drawn as price lines (`web/static/js/level-charts.js`). It uses
TradingView Lightweight Charts v4.2.3 (Apache 2.0), vendored at
`web/static/vendor/lightweight-charts-4.2.3.standalone.production.js` so the dashboard
needs no internet access; to upgrade, replace that file and update the version in
`top.html`. Its attribution logo stays on, as the licence notice asks.

Signals to review, not advice.

### Market tab (`/market`, `playbook.market_bias`)

SPY, QQQ, SCHG and SCHD are in `STATIC_TICKERS`, so every run scans them. Their bias comes
from the breakout scan's states:

- **LONG:** the weekly `Trend` is BULLISH, the weekly `MTF` is ALIGNED or PARTIAL, and
  the daily close is above its SMA50.
- **SHORT:** the weekly `Trend` or `MTF` is BEARISH, and the daily close is below its
  SMA50.
- **NEUTRAL:** anything else, or a missing row.

Each card shows the weekly and daily states, a daily chart with the scan's
`Resistance` and `Support` lines, and the daily close, SMAs, ATR and RVOL20. There are no
trade plans (no trigger, stop or targets) on this page. When SPY or QQQ is SHORT, Top
Setups shows a "market headwind" banner. The ranking is unchanged.

**Market strip (`web/insights.market_strip`, every page).** A bar under the nav with,
for each of SPY / QQQ / SCHG / SCHD:
- the last daily % change (last two closes in the daily raw file),
- the daily close above or below its SMA50 and SMA200 (latest daily breakout row),
- "closed below support on heavy volume" when the daily close is below `Support` and
  RVOL20 ≥ `config.BREAKOUT_RVOL_CONFIRM` (1.5),

then the growth-vs-dividend label. Facts only: it shows no bias and adds no rule.

**Relative view (`web/insights.etf_relative`).** A table above the cards, from each
ETF's daily raw closes:
- returns over 1W / 1M / 3M / 6M (5 / 21 / 63 / 126 bars),
- 63-day relative strength vs SPY (`market.relative_strength`, "strong" when the ratio
  is above its 20-bar MA and the return is ahead),
- a growth-vs-dividend label from the SCHG / SCHD close ratio
  (`config.ROTATION_*`):
  - *Growth leading*: the ratio is above its 50-day SMA and SCHG's 63-day return beats
    SCHD's.
  - *Dividend/value leading*: below the SMA and SCHD beats SCHG.
  - *Mixed*: anything else, or too little history.

Raw prices are Yahoo adjusted closes (`auto_adjust=True`), so returns include
dividends and SCHD's yield is not counted against it. The label is unvalidated
context, not a signal.

**Freshness (`web/insights.etf_freshness`).** Under each card's header a line gives the
daily and weekly rows' `AsOf Date`, with a warning next to a mode when:
- its row is `missing`,
- the row is `stale` (older than the last closed session, or the Monday of the last
  closed week),
- the ETF's raw file has a bar-quality issue (`data.quality`, checked live).

**Data warnings on Top Setups.** A weekly CONFIRMED card shows "⚠ data: …" when its
ticker has an error-severity row in the newest weekly `data_quality_<date>.csv`. Warnings
(`big_move`, …) are not shown there.

### Changes (`/changes`, `web/insights.run_changes`)

Compares each mode's newest breakout run with the one before it:

- **Status changes**, for tickers in both runs. Tickers that entered or left the
  universe are not listed.
  - *Moved toward a breakout*: forward along PRE_BREAKOUT → BREAKOUT_TRIGGERED →
    BREAKOUT_CONFIRMED, from any other status into one of them.
  - *Failed breakouts*: into FAILED_BREAKOUT.
  - *Other*: every other change.
  - Each row shows the Breakout Readiness and its change.
- **Coiled Cobra setups**: new since the previous run, dropped, and kept with their
  Score change.
- **Market bias**: each Market tab ETF's bias at the newest daily run vs the previous
  one. Each daily run is paired with the newest weekly run on or before its date.

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
| Health check: `<ETF> data quality ohlc_incoherent` | Yahoo's provisional bar; re-run with `--now` later, or check the bar on Yahoo |
| Daily health check: stale SPY/QQQ bar | Yahoo has not published the close yet; the 22:30 retry handles it |
| Dashboard shows old behaviour | The container runs its image; rebuild with `docker compose up -d --build` |
