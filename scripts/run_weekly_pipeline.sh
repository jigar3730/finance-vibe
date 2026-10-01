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
set -euo pipefail

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

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date -Is) another weekly run holds $LOCK_FILE; skipping" >> "$log"
    exit 1
fi

{
    echo "=== weekly pipeline start $(TZ=America/New_York date -Is) ==="
    status=0
    docker exec "$CONTAINER" python src/finance_vibe/run_vibe.py --mode weekly || status=$?
    echo "=== weekly pipeline end $(TZ=America/New_York date -Is) exit=$status ==="
} >> "$log" 2>&1

exit "$status"
