#!/usr/bin/env bash
# Run ruff and mypy (the uv `lint` group) in a throwaway uv container against the
# working tree. The host has no Python tooling, and the finance_vibe image does not
# install the lint group.
#
#   scripts/lint_in_container.sh                       # ruff check + ruff format --check + mypy
#   scripts/lint_in_container.sh ruff check --fix      # any command, run in the lint venv
#   scripts/lint_in_container.sh ruff format
#
# The venv and caches persist in $FINANCE_VIBE_LINT_HOME (default
# ~/.cache/finance-vibe-lint), so only the first run downloads anything.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_IMAGE="ghcr.io/astral-sh/uv:0.8.22-python3.12-bookworm-slim"  # matches the Dockerfile's uv
LINT_HOME="${FINANCE_VIBE_LINT_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/finance-vibe-lint}"
mkdir -p "$LINT_HOME"

if (( $# == 0 )); then
    set -- sh -c 'rc=0; ruff check . || rc=1; ruff format --check . || rc=1; mypy || rc=1; exit $rc'
fi

tty_flag=()
[[ -t 1 ]] && tty_flag=(-t)

docker run --rm "${tty_flag[@]}" -u "$(id -u):$(id -g)" \
    -v "$REPO:/w" -w /w -v "$LINT_HOME:/lint" \
    -e HOME=/tmp -e UV_CACHE_DIR=/lint/uv -e UV_PROJECT_ENVIRONMENT=/lint/venv \
    -e UV_PYTHON_DOWNLOADS=never -e RUFF_CACHE_DIR=/lint/ruff -e MYPY_CACHE_DIR=/lint/mypy \
    "$UV_IMAGE" \
    sh -c 'uv sync -q --frozen --no-default-groups --group lint && exec uv run -q --frozen --no-sync "$@"' -- "$@"
