#!/usr/bin/env bash
# Run the weekly Finance Vibe pipeline inside the finance_vibe container.
#
# Scheduled from the host crontab for Fridays 18:00 America/New_York. The host
# clock is UTC, so cron fires at both 22:00 and 23:00 UTC on Fridays and this
# script exits unless it is the 18:00 hour in New York (EDT vs EST):
#
#   0 22,23 * * 5 /opt/stacks/finance-vibe/scripts/run_weekly_pipeline.sh
#
# Usage:
#   scripts/run_weekly_pipeline.sh            # scheduled run (checks the ET hour)
#   scripts/run_weekly_pipeline.sh --now      # run immediately, skip the hour check
#
# Logs: $FINANCE_VIBE_CRON_LOG_DIR (default ~/.local/state/finance-vibe),
# one file per run: weekly_<YYYY-MM-DD_HHMM>.log. A lock prevents overlapping runs.
#
# Alerts: if the pipeline fails, or it exits 0 but scripts/check_weekly_outputs.py
# finds a problem (missing outputs, stale newest bar, > 20% ingest errors), an email
# is sent by scripts/notify_email.py using ~/.config/finance-vibe/notify.env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${FINANCE_VIBE_CONTAINER:-finance_vibe}"
RUN_HOUR_ET="18"
LOG_DIR="${FINANCE_VIBE_CRON_LOG_DIR:-$HOME/.local/state/finance-vibe}"
LOCK_FILE="$LOG_DIR/weekly.lock"
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

if [[ "${1:-}" != "--now" ]] && [[ "$(TZ=America/New_York date +%H)" != "$RUN_HOUR_ET" ]]; then
    exit 0
fi

mkdir -p "$LOG_DIR"
log="$LOG_DIR/weekly_$(TZ=America/New_York date +%Y-%m-%d_%H%M).log"

alert() {  # alert <subject> <first line of body>
    python3 "$SCRIPT_DIR/notify_email.py" --subject "[finance-vibe] $1" \
        --body "$2"$'\n\n'"Host: $(hostname)   Log: $log" --body-file "$log" >> "$log" 2>&1 \
        || echo "$(date -Is) alert email not sent (see notify_email output above)" >> "$log"
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date -Is) another weekly run holds $LOCK_FILE; skipping" >> "$log"
    alert "weekly run skipped: previous run still holds the lock" \
          "The scheduled weekly pipeline did not start because another run is still in progress."
    exit 1
fi

status=0
{
    echo "=== weekly pipeline start $(TZ=America/New_York date -Is) ==="
    docker exec "$CONTAINER" python src/finance_vibe/run_vibe.py --mode weekly || status=$?
    echo "=== weekly pipeline end $(TZ=America/New_York date -Is) exit=$status ==="
} >> "$log" 2>&1

if [[ "$status" -ne 0 ]]; then
    alert "weekly pipeline FAILED (exit $status)" \
          "run_vibe.py --mode weekly exited with status $status."
    exit "$status"
fi

check=0
docker exec -i "$CONTAINER" python - < "$SCRIPT_DIR/check_weekly_outputs.py" >> "$log" 2>&1 || check=$?
if [[ "$check" -ne 0 ]]; then
    alert "weekly pipeline finished but its outputs look wrong" \
          "run_vibe.py exited 0, but the post-run health check failed (details at the end of the log)."
    exit 3
fi
exit 0
