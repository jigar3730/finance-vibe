#!/usr/bin/env bash
# Run pytest inside the finance_vibe container against the working tree (the
# host may not have pytest; the image's /app/src may be stale). docs/ and
# templates/ are copied because tests/test_docs_routes.py reads them.
#
#   scripts/test_in_container.sh                       # full suite
#   scripts/test_in_container.sh tests/test_as_of.py   # any pytest args
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${FINANCE_VIBE_CONTAINER:-finance_vibe}"
WORK=/tmp/test_repo

docker exec "$CONTAINER" rm -rf "$WORK"
docker exec "$CONTAINER" mkdir -p "$WORK"
for d in src tests docs templates; do docker cp "$REPO/$d" "$CONTAINER:$WORK/$d" >/dev/null; done
trap 'docker exec "$CONTAINER" rm -rf "$WORK"' EXIT

docker exec -w "$WORK" -e PYTHONPATH="$WORK/src" "$CONTAINER" \
    python -m pytest -q -p no:cacheprovider "$@"
