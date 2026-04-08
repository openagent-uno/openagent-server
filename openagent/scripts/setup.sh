#!/bin/bash
# First-time setup for OpenAgent on a fresh VPS.
#
# Usage: ./scripts/setup.sh
#
# Prerequisites: Python 3.11+, Node.js 20+, npm

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== OpenAgent Setup ==="

cd "$PROJECT_DIR"

# 1. Python venv
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Installing Python dependencies..."
venv/bin/pip install --upgrade pip setuptools wheel > /dev/null 2>&1
venv/bin/pip install -e ".[all]" croniter > /dev/null 2>&1
echo "Python deps installed."

# 2. Build bundled MCPs
for mcp_dir in mcps/*/; do
    mcp_name=$(basename "$mcp_dir")

    # Skip Python-based MCPs (no npm)
    if [ -f "$mcp_dir/requirements.txt" ] && [ ! -f "$mcp_dir/package.json" ]; then
        echo "Skipping Python MCP: $mcp_name"
        continue
    fi

    if [ -f "$mcp_dir/package.json" ]; then
        if [ ! -d "$mcp_dir/node_modules" ]; then
            echo "Installing MCP: $mcp_name..."
            (cd "$mcp_dir" && npm install > /dev/null 2>&1)
        fi
        if [ ! -d "$mcp_dir/dist" ] && grep -q '"build"' "$mcp_dir/package.json"; then
            echo "Building MCP: $mcp_name..."
            (cd "$mcp_dir" && npm run build > /dev/null 2>&1)
        fi
        echo "MCP ready: $mcp_name"
    fi
done

# 3. Create memories dir
mkdir -p memories

# 4. Config check
if [ ! -f "openagent.yaml" ]; then
    echo ""
    echo "WARNING: No openagent.yaml found."
    echo "Copy the example and edit it:"
    echo "  cp openagent.yaml.example openagent.yaml"
    echo "  nano openagent.yaml"
fi

echo ""
echo "=== Setup complete ==="
echo "Start with: ./scripts/start.sh"
