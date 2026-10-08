#!/usr/bin/env bash
# Run the daily Finance Vibe pipeline inside the finance_vibe container.
#
# Scheduled from the host crontab for weekdays 17:30 America/New_York, after
# daily bars are final (17:00 ET, daily_ingest.DAY_FINAL_HOUR_ET). The host
# clock is UTC, so cron fires at both 21:30 and 22:30 UTC and this script exits
# unless it is the 17:00 hour in New York (EDT vs EST):
#
#   30 21,22 * * 1-5 /opt/stacks/finance-vibe/scripts/run_daily_pipeline.sh
#
# A run is capped at RUN_TIMEOUT so it always ends before the Friday 18:00 ET
# weekly run (both refresh data/active_tickers.csv). Market holidays still run;
# the health check expects the previous session's bar on those days.
#
# Usage:
#   scripts/run_daily_pipeline.sh            # scheduled run (checks the ET hour)
#   scripts/run_daily_pipeline.sh --now      # run immediately, skip the hour check
#
# Logs: $FINANCE_VIBE_CRON_LOG_DIR (default ~/.local/state/finance-vibe),
# one file per run: daily_<YYYY-MM-DD_HHMM>.log. A lock prevents overlapping runs.
#
# Alerts: if the pipeline fails or times out, or it exits 0 but
# scripts/check_daily_outputs.py finds a problem (missing outputs, stale newest bar,
# > 20% ingest errors), an email is sent by scripts/notify_email.py using
# ~/.config/finance-vibe/notify.env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${FINANCE_VIBE_CONTAINER:-finance_vibe}"
RUN_HOUR_ET="17"
RUN_TIMEOUT="25m"
LOG_DIR="${FINANCE_VIBE_CRON_LOG_DIR:-$HOME/.local/state/finance-vibe}"
LOCK_FILE="$LOG_DIR/daily.lock"
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

if [[ "${1:-}" != "--now" ]] && [[ "$(TZ=America/New_York date +%H)" != "$RUN_HOUR_ET" ]]; then
    exit 0
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

status=0
{
    echo "=== daily pipeline start $(TZ=America/New_York date -Is) ==="
    # timeout runs inside the container so it kills run_vibe.py and its stage
    # subprocesses (its whole process group), not just the docker client.
    docker exec "$CONTAINER" timeout "$RUN_TIMEOUT" python src/finance_vibe/run_vibe.py --mode daily || status=$?
    echo "=== daily pipeline end $(TZ=America/New_York date -Is) exit=$status ==="
} >> "$log" 2>&1

if [[ "$status" -eq 124 ]]; then
    alert "daily pipeline TIMED OUT after $RUN_TIMEOUT" \
          "run_vibe.py --mode daily was stopped after $RUN_TIMEOUT."
    exit "$status"
fi
if [[ "$status" -ne 0 ]]; then
    alert "daily pipeline FAILED (exit $status)" \
          "run_vibe.py --mode daily exited with status $status."
    exit "$status"
fi

check=0
docker exec -i "$CONTAINER" python - < "$SCRIPT_DIR/check_daily_outputs.py" >> "$log" 2>&1 || check=$?
if [[ "$check" -ne 0 ]]; then
    alert "daily pipeline finished but its outputs look wrong" \
          "run_vibe.py exited 0, but the post-run health check failed (details at the end of the log)."
    exit 3
fi
exit 0
