#!/usr/bin/env bash
# Run the daily Finance Vibe pipeline (Coiled Cobra + breakout scanner on daily
# bars) inside the finance_vibe container.
#
# Scheduled from the host crontab for every trading day (Mon-Fri, America/New_York):
#
#   18:30 ET  first attempt. Yahoo has usually published the day's close by then;
#             it also starts after the Friday 18:00 weekly run.
#   22:30 ET  retry, only if the 18:30 run did not pass its health check (Yahoo
#             sometimes publishes the daily close hours late; on 2026-10-07 it was
#             still blank at 21:00 ET).
#
# The host clock is UTC, so cron fires at the UTC hours covering both slots in
# EDT and EST, and this script exits unless it is 18:xx or 22:xx on a weekday in
# New York:
#
#   30 22,23,2,3 * * * /opt/stacks/finance-vibe/scripts/run_daily_pipeline.sh
#
# Weekends are skipped (no new daily bars). Market holidays still run; the health
# check then expects the previous session's bar.
#
# Usage:
#   scripts/run_daily_pipeline.sh            # scheduled run (checks the ET time)
#   scripts/run_daily_pipeline.sh --now      # run immediately, skip the time checks
#
# Logs: $FINANCE_VIBE_CRON_LOG_DIR (default ~/.local/state/finance-vibe),
# one file per run: daily_<YYYY-MM-DD_HHMM>.log. A run that passes the health check
# leaves daily_ok_<YYYY-MM-DD> there, which makes the 22:30 retry a no-op.
#
# Alerts (scripts/notify_email.py, ~/.config/finance-vibe/notify.env):
#   - pipeline failure or timeout: emailed at once;
#   - health check failure (missing outputs, stale newest bar, > 20% ingest errors):
#     emailed only if the 22:30 retry (or a --now run) also fails;
#   - skipped because another daily run holds the lock.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${FINANCE_VIBE_CONTAINER:-finance_vibe}"
FIRST_HOUR_ET="18"
RETRY_HOUR_ET="22"
RUN_TIMEOUT="25m"
WEEKLY_WAIT_SECONDS=1800
LOG_DIR="${FINANCE_VIBE_CRON_LOG_DIR:-$HOME/.local/state/finance-vibe}"
LOCK_FILE="$LOG_DIR/daily.lock"
WEEKLY_LOCK_FILE="$LOG_DIR/weekly.lock"
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

now_hour="$(TZ=America/New_York date +%H)"
now_dow="$(TZ=America/New_York date +%u)"   # 1 = Monday .. 7 = Sunday
today="$(TZ=America/New_York date +%Y-%m-%d)"
ok_marker="$LOG_DIR/daily_ok_$today"

attempt="manual"
if [[ "${1:-}" != "--now" ]]; then
    (( now_dow <= 5 )) || exit 0
    case "$now_hour" in
        "$FIRST_HOUR_ET") attempt="first" ;;
        "$RETRY_HOUR_ET") attempt="retry" ;;
        *) exit 0 ;;
    esac
    # Already done today (e.g. a manual --now run before 18:30 still counts).
    [[ -e "$ok_marker" ]] && exit 0
fi

mkdir -p "$LOG_DIR"
log="$LOG_DIR/daily_$(TZ=America/New_York date +%Y-%m-%d_%H%M).log"

alert() {  # alert <subject> <first line of body>
    python3 "$SCRIPT_DIR/notify_email.py" --subject "[finance-vibe] $1" \
        --body "$2"$'\n\n'"Host: $(hostname)   Log: $log" --body-file "$log" >> "$log" 2>&1 \
        || echo "$(date -Is) alert email not sent (see notify_email output above)" >> "$log"
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date -Is) another daily run holds $LOCK_FILE; skipping" >> "$log"
    alert "daily run skipped: previous run still holds the lock" \
          "The scheduled daily pipeline did not start because another run is still in progress."
    exit 1
fi

# Both pipelines refresh data/active_tickers.csv, so never overlap a weekly run
# (e.g. a slow Friday 18:00 run): wait for its lock and hold it while we run.
exec 8>"$WEEKLY_LOCK_FILE"
if ! flock -w "$WEEKLY_WAIT_SECONDS" 8; then
    echo "$(date -Is) weekly run still holds $WEEKLY_LOCK_FILE after ${WEEKLY_WAIT_SECONDS}s; skipping" >> "$log"
    alert "daily run skipped: weekly run still in progress" \
          "The daily pipeline waited ${WEEKLY_WAIT_SECONDS}s for the weekly run to finish and gave up."
    exit 1
fi

status=0
{
    echo "=== daily pipeline start ($attempt attempt) $(TZ=America/New_York date -Is) ==="
    # timeout runs inside the container so it kills run_vibe.py and its stage
    # subprocesses (its whole process group), not just the docker client.
    docker exec "$CONTAINER" timeout "$RUN_TIMEOUT" python src/finance_vibe/run_vibe.py --mode daily || status=$?
    echo "=== daily pipeline end $(TZ=America/New_York date -Is) exit=$status ==="
} >> "$log" 2>&1

if [[ "$status" -eq 124 ]]; then
    alert "daily pipeline TIMED OUT after $RUN_TIMEOUT" \
          "run_vibe.py --mode daily was stopped after $RUN_TIMEOUT ($attempt attempt)."
    exit "$status"
fi
if [[ "$status" -ne 0 ]]; then
    alert "daily pipeline FAILED (exit $status)" \
          "run_vibe.py --mode daily exited with status $status ($attempt attempt)."
    exit "$status"
fi

check=0
docker exec -i "$CONTAINER" python - < "$SCRIPT_DIR/check_daily_outputs.py" >> "$log" 2>&1 || check=$?
if [[ "$check" -ne 0 ]]; then
    if [[ "$attempt" == "first" ]]; then
        echo "$(date -Is) health check failed; will retry at ${RETRY_HOUR_ET}:30 ET" >> "$log"
    else
        alert "daily pipeline finished but its outputs look wrong" \
              "run_vibe.py exited 0, but the post-run health check failed ($attempt attempt; details at the end of the log)."
    fi
    exit 3
fi

touch "$ok_marker"
find "$LOG_DIR" -maxdepth 1 -name 'daily_ok_*' -mtime +14 -delete
exit 0
