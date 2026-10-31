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
| `src/finance_vibe/data/` | `tickers` (universe), `ingest` (Yahoo download), `daily_ingest`, `market_ingest` (market-context funds), `raw` (raw CSV loader), `quality` |
| `src/finance_vibe/scanners/` | `coiled_cobra`, `breakout` |
| `src/finance_vibe/web/` | Flask dashboard (`app`, `templates/`, `static/`) |
| `scripts/` | Cron runners, output health checks, email alerts, test/lint/golden helpers |
| `data/active_tickers.csv` | Ticker universe |
| `data/raw/{weekly,daily}/` | Raw OHLCV CSVs (`*_10y_1wk.csv`, `*_5y_1d.csv`) |
| `data/logs/{weekly,daily}/` | Dated scan outputs and ingest errors |
| `data/raw/market/`, `data/logs/market/` | Market-context raw CSVs (`*_2y_1d.csv`), their ingest errors and quality reports |

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
| 5 | `data.market_ingest` (daily only) | raw CSVs in `data/raw/market/` and `data/logs/market/data_quality_<date>.csv` (skipped with `--reuse-raw`) |

Each stage also runs on its own:

```bash
python -m finance_vibe.data.tickers
python -m finance_vibe.data.ingest weekly
python -m finance_vibe.data.daily_ingest                 # takes no mode
python -m finance_vibe.data.market_ingest                # takes no mode; 14 tickers
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
- **`data.market_ingest`** (daily, last stage) downloads `config.MARKET_CONTEXT_TICKERS`
  for the market home page: VIX (Yahoo `^VIX`, saved as `VIX_2y_1d.csv`), HYG and IEF
  (credit), and the 11 sector funds in `MARKET_SECTOR_FUNDS` (breadth). They are not in
  the ticker list and live in `data/raw/market/`, so neither scanner sees them. Same
  download, OHLCV contract, today's-bar rule and `MIN_SAVE_ROWS` as `daily_ingest`.
  The folder is not wiped: a ticker that fails keeps its old file, which the quality
  report then flags `stale`. Failures go to `data/logs/market/ingest_errors_<date>.csv`;
  the stage always exits 0, so it never fails the pipeline. Its quality report
  (`data/logs/market/data_quality_<date>.csv`) skips `zero_volume` for VIX
  (`MARKET_NO_VOLUME`; an index has no volume).
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
to ingest, and `data_quality_<date>.csv` exists with no error rows for the market
ETFs (other tickers' errors are only counted). The daily check also fails when a
market-context file is missing, or `data/logs/market/data_quality_<date>.csv` is missing
or has an error row (e.g. a stale VIX file). Run one by hand with
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
| `/` | Market home: a "Today" table for SPY / QQQ / SCHG / SCHD (daily change, close vs 50- and 200-day averages, heavy-volume support break, growth vs dividend), Market health (VIX, HYG/IEF credit ratio, sectors above their 50-day average), the relative view (returns, RS vs SPY), then a card per ETF (weekly trend / MTF and close vs 50-day average, weekly and daily states, a daily chart with Support / Resistance, row freshness) |
| `/market` | Redirects (301) to `/` |
| `/setups` | Setups (unproven), by evidence: weekly BREAKOUT_CONFIRMED cards (unproven), untested Coiled Cobra setups (collapsed), since last run (new weekly CONFIRMED, stock list in / out, status-change counts), counts of signals that failed in-sample |
| `/changes` | Market facts that changed since the previous daily run (SMA50 / SMA200 crossings, new heavy-volume support breaks, growth-vs-dividend label) |
| `/runs` | Run dates per mode, linking each date's Coiled Cobra and breakout pages |
| `/api/bars/<mode>/<SYMBOL>` | JSON OHLC bars from the raw silo (last 60 daily / 52 weekly) for the level charts |
| `/cobra/<mode>/<date>` | Coiled Cobra setups, highest Score first, live price vs Close (green above / red below; no row highlight) |
| `/breakout/<mode>/<date>` | Breakout scan: KPIs, status chips, highlights (or `?show=all` / `?status=`), failed breakouts collapsed |

Live prices (scanner run pages only) come from `yfinance` `fast_info`, fetched 16 at a
time and cached for 60 seconds. Both run pages have one badge (mode and run date), and
their caveats are `?` tooltips on the title.

### Setups (unproven) (`/setups`, `web/playbook.py`)

The page sorts the newest runs by what the backtest found (`RESEARCH.md` §7, §11). Nothing
is ranked or scored, and there are no trade levels. Run closes only (no live prices). One
badge at the top: "as of <date>" when the weekly and daily runs share a date, otherwise
both run dates. How far each section can be trusted is a `?` tooltip on its heading.
It opens with the SPY / QQQ facts box (`playbook.market_facts`: weekly `Trend`, weekly
`MTF`, daily close vs SMA50; scanner names are allowed in the Setups area).

1. **Weekly CONFIRMED** (`playbook.weekly_confirmed`): one card per weekly
   `BREAKOUT_CONFIRMED` row, by symbol. Tooltip: passed in-sample (+2.3% vs SPY over 4
   weeks, p 0.049; p 0.40 without the split-flagged events), unproven until the
   2035-09-07 holdout, tested as buy next open, hold 4 weeks, no stop. A card says "Also
   CONFIRMED last week" when the previous weekly breakout CSV had the symbol CONFIRMED (a
   continuation, not the tested first-week event; a symbol missing from the previous file
   is not marked). Each card shows the close, the weekly position bar
   (`playbook.position_bar`, `Support` to `Resistance`, as on the market home), then
   SMA50, ATR, extension, RVOL20 and MTF.
2. **Coiled Cobra setups** (`playbook.untested_cobra`), tooltip "Never backtested":
   collapsed by default; a table per mode with Score, Grade, Close and the ticker's
   breakout status.
3. **Since last run** (`templates/_since_last_run.html`, `insights.run_changes`), the two
   newest runs per mode (dates in the heading tooltip):
   - **New weekly CONFIRMED** (`insights.new_confirmed`). *New this week*: CONFIRMED now
     and scanned last week with another status, the first-week event that was tested.
     *Confirmed now, but not scanned last week*: no row in last week's file, so the
     first week is unknown. Names CONFIRMED both weeks are continuations and not listed.
   - **Stock list changes** (`insights.universe_changes`): per mode, tickers that
     entered or left the scanned list.
   - **Everything else**, collapsed: per mode, counts of status changes (moved toward a
     breakout, failed, other) and Coiled Cobra setups new / dropped. `?show=all` opens it
     with the full tables (`insights.status_changes`, `cobra_changes`; Readiness shown,
     no Readiness change column, since Readiness falls when a breakout happens).
4. **Failed the old-data test** (`playbook.failed_counts`), tooltip "did no better than
   SPY or random entries": counts only, each linking to the breakout page filtered to
   that status. Daily: `PRE_BREAKOUT`, `BREAKOUT_TRIGGERED`, `BREAKOUT_CONFIRMED`.
   Weekly: `PRE_BREAKOUT`, `BREAKOUT_TRIGGERED`.

No page loads a chart any more. `web/static/js/level-charts.js`, the vendored
TradingView Lightweight Charts v4.2.3 (`web/static/vendor/`) and `/api/bars` are still in
the tree, unused, until their removal is confirmed.

### Market home (`/`)

Daily closes only (no live prices), laid out per `docs/UX_STANDARDS.md`. One "as of"
badge next to the title: the newest daily close the page uses. Under it one fact line:
"N of 4 funds closed below support on heavy volume · VIX ±X.XX · N of 11 sectors above
50-day". Top to bottom:

1. **One table** (`insights.market_table`) for SPY / QQQ / SCHG / SCHD: daily % change
   (last two closes in the daily raw file), returns over 1W / 1M / 3M / 6M (5 / 21 / 63 /
   126 bars) and RS vs SPY (`market.relative_strength`: 63-day return minus SPY's), all
   with a sign and 2 decimals, right-aligned, green / red for up / down. Full width, in
   the regular font. A note column shows "⚠ below support · heavy vol" (daily close below `Support` with RVOL20 ≥
   `config.BREAKOUT_RVOL_CONFIRM`, 1.5; `insights.heavy_break`) and "⚠ stale" / "⚠ no
   data" when a fund's newest close is older than the badge or missing. Under it, the
   growth-vs-dividend label (`insights.rotation`, `config.ROTATION_*`):
   - *Growth outperforming (last 63 days)*: the SCHG / SCHD close ratio is above its
     50-day SMA and SCHG's 63-day return beats SCHD's.
   - *Dividend/value outperforming (last 63 days)*: below the SMA and SCHD beats SCHG.
   - *Mixed*: anything else, or too little history.

   Raw prices are Yahoo adjusted closes (`auto_adjust=True`), so returns include
   dividends.
2. **Market health** (`web/health.market_health`), three tiles with a large number, from
   the market-context files (`data/raw/market/`):
   - **VIX:** the close, its change in points since the previous close, and above or
     below its 20-day average.
   - **Credit: HYG / IEF:** the close ratio (high-yield company bonds vs 7–10 year
     Treasuries), above or below its 50-day average, and its % change over 21 sessions.
     IEF rather than TLT: its bonds are closer in length to HYG's, so the ratio moves
     mostly with credit worry, not with long-term interest rates.
   - **Breadth:** "N of 11" sector funds (`config.MARKET_SECTOR_FUNDS`) closed above
     their 50-day average. Inside the tile, 11 small tiles (green above / red below /
     grey no data), each with the ticker, a short name (`config.MARKET_SECTOR_SHORT`,
     "Tech") and its % from the 50-day average, sorted by that %, highest first; the full
     name, close and average on hover. A fund without a file or enough history is left
     out of the count and sorted last.

   A tile shows "⚠ HYG stale" (or missing, or a bar-quality check) when a file is
   missing, older than the last closed session, or has an error row in the newest
   `data/logs/market/data_quality_<date>.csv`. The tiles themselves stay uncoloured.
3. **Support and resistance**, a card per ETF (`playbook.position_bar`, macro in
   `templates/_macros.html`): the daily close, and a bar from the daily scan's `Support`
   (left end) to `Resistance` (right end) with a dot at the close. Every card has the
   same layout: the bar, Support and Resistance under its ends, then the close on its
   own line. A close out of range draws a hollow dot just outside that end of the bar
   and adds "below support by 1.20%" (line in red) or "above resistance by …" (green).
   SMA 20 / 50 / 200 underneath. A card
   header shows the ETF's daily freshness flags (`insights.etf_freshness`: missing or
   stale scan row, raw bar-quality issue).

Explanations are `?` tooltips only. No charts, notes, scanner states or status names on
this page.

**Market strip (`web/insights.market_strip`).** A bar under the nav (0.9rem, facts in white, labels grey) on every page except
`/`, where the same facts are in its table. For each of SPY / QQQ / SCHG / SCHD:
- the last daily % change (last two closes in the daily raw file),
- the daily close above or below its SMA50 and SMA200 (latest daily breakout row),
- "closed below support on heavy volume" when the daily close is below `Support` and
  RVOL20 ≥ `config.BREAKOUT_RVOL_CONFIRM` (1.5),

then the growth-vs-dividend label. Facts only: it adds no rule.

**Data warnings on Setups.** A weekly CONFIRMED card shows "⚠ data: …" when its
ticker has an error-severity row in the newest weekly `data_quality_<date>.csv`. Warnings
(`big_move`, …) are not shown there.

### Changes (`/changes`, `web/insights.market_changes`)

Market facts only, newest daily run vs the previous one; the badge names both dates
("Fri 2026-10-30 vs Thu 2026-10-29"). For SPY / QQQ / SCHG / SCHD, only ETFs where
something changed: the daily close crossing its SMA50 or SMA200, or a *new* close below
`Support` with RVOL20 ≥ 1.5 (`insights.heavy_break`; already broken the run before does
not count). Then whether the growth-vs-dividend label changed, from daily closes up to
each run's date. No scanner states or stock lists: those are on Setups ("Since last
run"). The old `/changes?show=all` link redirects to `/setups?show=all#since-last-run`.

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
