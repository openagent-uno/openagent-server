#!/bin/bash
# Start OpenAgent in a screen session.
# Kills any existing instance first, waits for Telegram polling release.
#
# Usage: ./scripts/start.sh [config_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$PROJECT_DIR/openagent.yaml}"
VENV="$PROJECT_DIR/venv/bin/openagent"
LOG="$PROJECT_DIR/openagent.log"
SCREEN_NAME="openagent"

echo "=== OpenAgent Start ==="

# 1. Kill any existing instance
if pgrep -f "openagent serve" > /dev/null 2>&1; then
    echo "Killing existing OpenAgent processes..."
    pkill -9 -f "openagent serve" 2>/dev/null || true
    sleep 2
fi

# Kill stale screen sessions
screen -ls 2>/dev/null | grep "$SCREEN_NAME" && screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
screen -wipe 2>/dev/null || true

# 2. Verify clean state
if pgrep -f "openagent serve" > /dev/null 2>&1; then
    echo "ERROR: Could not kill existing OpenAgent process"
    exit 1
fi
echo "Clean state confirmed."

# 3. Wait for Telegram polling release
echo "Waiting 45s for Telegram polling release..."
sleep 45

# 4. Double-check no stale process appeared
if pgrep -f "openagent serve" > /dev/null 2>&1; then
    echo "ERROR: Stale OpenAgent process appeared during cooldown"
    exit 1
fi

# 5. Start in screen
echo "Starting OpenAgent..."
export DISPLAY="${DISPLAY:-:1}"
rm -f "$LOG"

screen -dmS "$SCREEN_NAME" bash -c "exec $VENV -c $CONFIG serve --channel telegram > $LOG 2>&1"

sleep 15

# 6. Verify
if pgrep -f "openagent serve" > /dev/null 2>&1; then
    PID=$(pgrep -f "openagent serve" | head -1)
    INSTANCES=$(pgrep -f "openagent serve" | wc -l)
    echo "OpenAgent running (PID: $PID, instances: $INSTANCES)"
    if [ "$INSTANCES" -gt 1 ]; then
        echo "WARNING: Multiple instances detected!"
    fi
    echo "Log: $LOG"
    tail -5 "$LOG"
else
    echo "ERROR: OpenAgent failed to start. Log:"
    cat "$LOG"
    exit 1
fi
