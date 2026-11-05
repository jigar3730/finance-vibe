#!/usr/bin/env bash
# Take one live-quote snapshot for the market home, inside the finance_vibe container.
#
# Scheduled from the host crontab every 5 minutes on weekdays:
#
#   */5 8-16 * * 1-5 /opt/stacks/finance-vibe/scripts/run_market_snapshot.sh
#
# data.market_snapshot checks the New York clock and the NYSE calendar itself and does
# nothing outside its slots: every 15 minutes from 08:00 ET, every 5 from 09:30 to 16:15.
#
# Usage:
#   scripts/run_market_snapshot.sh          # scheduled run (checks the clock)
#   scripts/run_market_snapshot.sh --now    # fetch regardless of the clock
#
# Logs: $FINANCE_VIBE_CRON_LOG_DIR (default ~/.local/state/finance-vibe), one file per
# day: snapshot_<YYYY-MM-DD>.log. A failed fetch keeps the previous quotes, and the page
# shows their age, so there is no email alert.
set -euo pipefail

CONTAINER="${FINANCE_VIBE_CONTAINER:-finance_vibe}"
RUN_TIMEOUT="60s"
LOG_DIR="${FINANCE_VIBE_CRON_LOG_DIR:-$HOME/.local/state/finance-vibe}"
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$LOG_DIR"
log="$LOG_DIR/snapshot_$(TZ=America/New_York date +%Y-%m-%d).log"

exec 9> "$LOG_DIR/snapshot.lock"
if ! flock -n 9; then
    echo "$(date -Is) previous snapshot still running; skipped" >> "$log"
    exit 0
fi

docker exec "$CONTAINER" timeout "$RUN_TIMEOUT" \
    python -m finance_vibe.data.market_snapshot "$@" >> "$log" 2>&1 \
    || echo "$(date -Is) snapshot run failed: exit $?" >> "$log"
