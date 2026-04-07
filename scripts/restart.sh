#!/bin/bash
# Restart OpenAgent: stop + start with Telegram polling cooldown.
#
# Usage: ./scripts/restart.sh [config_path]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== OpenAgent Restart ==="
"$SCRIPT_DIR/stop.sh"
"$SCRIPT_DIR/start.sh" "$@"
