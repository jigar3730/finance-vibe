#!/usr/bin/env bash
# Save the day's closes for the market home, without waiting for the 18:30 scan.
#
# Scheduled from the host crontab on weekdays, once daily bars are final (17:00 ET):
#
#   5 17 * * 1-5 /opt/stacks/finance-vibe/scripts/run_market_close.sh
#
# The 18:30 pipeline downloads the same files again and runs the health check and
# alerts, so a failure here is only logged.
#
# Logs: $FINANCE_VIBE_CRON_LOG_DIR (default ~/.local/state/finance-vibe), one file per
# day: market_close_<YYYY-MM-DD>.log.
set -euo pipefail

CONTAINER="${FINANCE_VIBE_CONTAINER:-finance_vibe}"
RUN_TIMEOUT="120s"
LOG_DIR="${FINANCE_VIBE_CRON_LOG_DIR:-$HOME/.local/state/finance-vibe}"
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$LOG_DIR"
log="$LOG_DIR/market_close_$(TZ=America/New_York date +%Y-%m-%d).log"

exec 9> "$LOG_DIR/market_close.lock"
if ! flock -n 9; then
    echo "$(date -Is) previous run still going; skipped" >> "$log"
    exit 0
fi

docker exec "$CONTAINER" timeout "$RUN_TIMEOUT" \
    python -m finance_vibe.data.market_ingest >> "$log" 2>&1 \
    || echo "$(date -Is) market close run failed: exit $?" >> "$log"
