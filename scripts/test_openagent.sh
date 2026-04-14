#!/usr/bin/env bash
# OpenAgent end-to-end test runner.
# Wraps scripts/test_openagent.py with sensible defaults and the right
# Python interpreter (the venv).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "Error: ${PY} not found. Run 'uv pip install -e .' from the repo root first." >&2
    exit 1
fi

exec "$PY" "${REPO_ROOT}/scripts/test_openagent.py" "$@"
