# Runbook

What to do, step by step. Details and the reasons behind them are in
[`operations.md`](operations.md). Commands run on the host (mediabox) from
`/opt/stacks/finance-vibe` unless they start with `docker exec`.

## Where things are

| What | Where |
| ---- | ----- |
| Dashboard | http://mediabox:5000 (container `finance_vibe`); these docs are under **Docs** |
| Run logs, locks, markers | `~/.local/state/finance-vibe/` on the host |
| Data volume | `/mnt/fast/finance-vibe-data` on the host, `/app/data` in the container (files are root-owned: change them through `docker exec`) |
| Schedule | `crontab -l` (user `jigar`; host clock is New York time) |
| Alert email settings | `~/.config/finance-vibe/notify.env` (not set up on mediabox yet, so alerts are only written to the log) |

| When (ET) | Job | Log |
| --------- | --- | --- |
| Mon–Fri every 5 min, 08:00–16:15 | Live quotes for the market home | `snapshot_<date>.log` |
| Mon–Fri 17:05 | Market closes for the market home | `market_close_<date>.log` |
| Mon–Fri 18:30, retry 22:30 | Daily pipeline, database rebuild, health check | `daily_<date>_<time>.log` |
| Sat 09:00 | Weekly pipeline, database rebuild, health check | `weekly_<date>_<time>.log` |

## Daily check (a minute)

1. Open the market home: the "as of" badge should show the last session's close.
   Any ⚠ note names what is stale or missing.
2. Check that the daily run passed: a `daily_ok_<today>` file means the health check
   passed.

   ```bash
   ls ~/.local/state/finance-vibe/ | grep "$(date +%F)"
   grep -h -A5 "HEALTH CHECK FAILED" ~/.local/state/finance-vibe/daily_$(date +%F)_*.log
   ```

   A `daily_market_retry_<date>` file with no `daily_ok_<date>` means only the market
   context data failed, and the retry did not fix it either (see below).
3. Saturday: the same for `weekly_<date>_*.log`, and the Runs page should list the new
   weekly date.

Until `notify.env` exists, this check is the only way to learn that a run failed.

## When something fails

Find the newest log for the run, then read its end: `tail -40 <log>`.

| What the log says | What it means | What to do |
| ----------------- | ------------- | ---------- |
| `health check failed; will retry at 22:30 ET` | 18:30 run: Yahoo had not published the close yet, or too many downloads failed | Nothing; the 22:30 retry handles it. Check the next morning |
| `health check failed on market data only` | 18:30 run: the scans are fine, a market-context file (VIX, DXY, HYG, …) failed | Nothing; 22:30 re-fetches only that data |
| `HEALTH CHECK FAILED` after the 22:30 retry | Still wrong after the retry | Read the listed problems and use the rows below, then re-run with `scripts/run_daily_pipeline.sh --now` |
| `<SYM> newest daily bar is …, expected …` | Yahoo still has no close for that day | Re-run with `--now` the next morning |
| `<SYM> … data quality ohlc_incoherent` | Yahoo's provisional bar (e.g. high equal to open on DXY) | Usually fixes itself the next day. Re-run with `--now` later or check the bar on Yahoo. For a market-context symbol only that tile shows a ⚠; the scans are not affected |
| `ingest errors N/M tickers (> 20%)` | Yahoo throttled or was down | Wait an hour, then `--now` |
| `missing output: …` | A stage did not write its file | Look higher up the log for the stage's traceback |
| `daily pipeline FAILED (exit N)` | A stage crashed | Find the traceback in the log. Reproduce with `docker exec finance_vibe python -m finance_vibe.pipeline --mode daily --reuse-raw --in-process` |
| `daily pipeline TIMED OUT after 25m` | Yahoo very slow, or something hung | Re-run with `--now`. If it repeats, time the stages separately |
| `run skipped: previous run still holds the lock` | An earlier run is still going, or died while holding the lock | Check with `docker exec finance_vibe ps aux`. If nothing is running, the next run takes the lock as normal (`flock` locks are freed when the process ends) |
| `database rebuild FAILED` | The SQLite copy could not be rebuilt; scans are fine | Run `docker exec finance_vibe python -m finance_vibe.data.database rebuild` and read the error. Only `/scan-stats` is affected (it shows a ⚠) |
| `universe rebuild failed` | More than 10% of the quarterly rebuild downloads failed; the previous members were scanned | The 22:30 retry tries again, or run `docker exec finance_vibe python -m finance_vibe.data.universe --rebuild` later |
| Market home "⚠ quotes N min old" | Live quote fetches are failing | `tail ~/.local/state/finance-vibe/snapshot_$(date +%F).log`. Closes and scans are not affected |
| Dashboard down | Container stopped | `docker compose ps`, then `docker compose up -d`, then `docker compose logs --tail 50` |

Manual re-runs:

```bash
scripts/run_daily_pipeline.sh --now     # whole daily run; a pass also satisfies today's schedule
scripts/run_weekly_pipeline.sh --now
docker exec -i finance_vibe python - < scripts/check_daily_outputs.py   # health check only
docker exec finance_vibe python -m finance_vibe.data.market_ingest     # market-context data only
```

Never run `python -m finance_vibe.pipeline` without `--reuse-raw` just to test something: it deletes
`data/raw/<mode>/` and downloads everything again.

## Deploying a change

The cron scripts run from this checkout but `docker exec` into the running image, so the
two must hold the same code.

1. Tests: `scripts/test_in_container.sh -q` (the full suite, not `-m "not slow"`).
2. Lint: `scripts/lint_in_container.sh`.
3. Golden check: `scripts/golden_in_container.sh compare` must print `IDENTICAL`. When a
   change is meant to move scores, bump the rubric version first and re-record with
   `baseline --force`.
4. Commit and merge to `main`.
5. Rebuild and restart: `docker compose up -d --build`.
6. Check in the browser: the market home, Setups, one run page, and whatever page changed.
   Avoid rebuilding during a scheduled run (18:30–19:00 and 22:30 on weekdays, 09:00 on Saturday).

## Recovery

| Lost | How to get it back |
| ---- | ------------------ |
| The container or image | `docker compose up -d --build` |
| `data/raw/*` | Comes back on the next scheduled run, or `scripts/run_daily_pipeline.sh --now` |
| `data/active_tickers.csv` | `docker exec finance_vibe python -m finance_vibe.data.universe` |
| `data/universe/` | `docker exec finance_vibe python -m finance_vibe.data.universe --rebuild` (the members can differ from before: the join rules apply again, with no stay rule) |
| `data/db/` | `docker exec finance_vibe python -m finance_vibe.data.database rebuild` |
| `data/live/` | The next snapshot run during the session |
| `data/logs/` (scan history) | **Cannot be rebuilt.** Past runs can only be replayed with `--as-of` from raw bars that are still on disk, and Yahoo restates adjusted prices, so those are not the original results |
| `data/research/backfill/snapshots/` | **Cannot be rebuilt** byte for byte (Yahoo restates prices). A new download is a new snapshot |
| The golden fixture (`/mnt/fast/finance-vibe-data/golden/`) | `snapshot --force` then `baseline --force`; the new baseline only shows that the code still behaves as it does today |
| The crontab | The lines in [`operations.md`](operations.md#scheduling); an old copy is in `~/.local/state/finance-vibe/crontab.bak-2026-10-20` |
| Alert email | Create `~/.config/finance-vibe/notify.env` (mode 600) with `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_USE_TLS`, `EMAIL_FROM`, `EMAIL_TO` (read by `scripts/notify_email.py`), then send a test: `python3 scripts/notify_email.py --subject test --body ok` |

Back up `data/logs/` and `data/research/` from time to time, since they cannot be downloaded again:

```bash
sudo tar czf /mnt/fast/finance-vibe-backup-$(date +%F).tar.gz -C /mnt/fast/finance-vibe-data logs research universe
```

## Recurring tasks

| When | Task | How |
| ---- | ---- | ---- |
| Before the third Friday of Mar / Jun / Sep / Dec (next: 2026-12-18) | Refresh the S&P 500 / 400 lists before the quarterly universe rebuild | `python3 scripts/refresh_index_lists.py --dry-run`, review it, run it, commit the lists ([details](operations.md#maintenance)) |
| The first run after that Friday | Check the universe rebuild | `/universe` page: no "⚠ last rebuild failed", the joins and leaves look sensible |
| When the Fed or BLS changes a date, and before 2027-12 | Update the market event dates | `config.MARKET_EVENTS` (the lists run through December 2027); the market home shows "⚠ … dates need updating" when one runs out |
| Monthly | Clear old raw files if space is low | `scripts/clean_data_volume.sh --dry-run`, then `--target raw` (never `logs` without a backup) |
| Monthly | Back up `data/logs/` and `data/research/` | The `tar` command above |
| After a dependency bump | Golden check | `scripts/golden_in_container.sh compare` |
