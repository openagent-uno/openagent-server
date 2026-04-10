#!/bin/bash
# First-time bootstrap for OpenAgent on a fresh machine.
#
# This script only handles what `openagent setup` cannot do itself — create
# the Python venv and install the package — and then delegates everything
# else (Docker, OS service registration, image pulls, checks) to
# `openagent setup --full`.
#
# Usage:
#   ./scripts/setup.sh            # minimal: venv + pip install + doctor
#   ./scripts/setup.sh --full     # also run `openagent setup --full`
#
# Prerequisites: Python 3.11+

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODE="${1:-}"

cd "$PROJECT_DIR"

echo "=== OpenAgent bootstrap ==="

# 1. Python venv
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Upgrading pip/setuptools/wheel..."
venv/bin/pip install --quiet --upgrade pip setuptools wheel

echo "Installing openagent-framework..."
if [ -f "pyproject.toml" ]; then
    venv/bin/pip install --quiet -e ".[all]"
else
    venv/bin/pip install --quiet "openagent-framework[all]"
fi

# 2. Memories dir
mkdir -p memories

# 3. Config check
if [ ! -f "openagent.yaml" ]; then
    echo ""
    echo "WARNING: no openagent.yaml found."
    echo "Create one before running 'openagent serve'."
fi

echo ""
echo "=== Running openagent doctor ==="
venv/bin/openagent doctor || true

# 4. Optional: full platform setup (Docker, OS service, image pulls)
if [ "$MODE" = "--full" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "=== Running openagent setup --full ==="
    venv/bin/openagent setup --full || {
        echo "openagent setup --full reported errors — see above."
    }
fi

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  openagent doctor            # verify environment"
echo "  openagent setup --full      # install Docker + OS service + pull images"
echo "  ./scripts/start.sh          # start OpenAgent in a screen session"
