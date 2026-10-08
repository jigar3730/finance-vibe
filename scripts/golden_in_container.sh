#!/usr/bin/env bash
# Run scripts/golden_compare.py inside the finance_vibe container against the
# *working tree* (the image's /app/src may be stale). Fixture and baseline live
# on the data volume at /app/data/golden (host: /mnt/fast/finance-vibe-data/golden).
#
#   scripts/golden_in_container.sh snapshot            # once: freeze raw data
#   scripts/golden_in_container.sh baseline            # record outputs of this tree
#   scripts/golden_in_container.sh compare [--keep]    # diff this tree vs baseline
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${FINANCE_VIBE_CONTAINER:-finance_vibe}"
WORK=/tmp/golden_repo

docker exec "$CONTAINER" rm -rf "$WORK"
docker exec "$CONTAINER" mkdir -p "$WORK"
for d in src scripts; do docker cp "$REPO/$d" "$CONTAINER:$WORK/$d" >/dev/null; done
trap 'docker exec "$CONTAINER" rm -rf "$WORK"' EXIT

cmd="$1"; shift
extra=()
[[ "$cmd" == "snapshot" ]] && extra=(--source-data /app/data)
rev="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
git -C "$REPO" diff --quiet HEAD -- src 2>/dev/null || rev="$rev+dirty"
docker exec -e GOLDEN_GIT_REV="$rev" "$CONTAINER" python "$WORK/scripts/golden_compare.py" \
    --repo "$WORK" --golden-dir /app/data/golden "$cmd" "${extra[@]}" "$@"
