#!/bin/bash
# Stop OpenAgent cleanly.
#
# Usage: ./scripts/stop.sh

echo "=== OpenAgent Stop ==="

# Kill processes
if pgrep -f "openagent serve" > /dev/null 2>&1; then
    echo "Killing OpenAgent processes..."
    pkill -9 -f "openagent serve" 2>/dev/null || true
    sleep 2
fi

# Kill screen session
screen -ls 2>/dev/null | grep "openagent" && screen -S openagent -X quit 2>/dev/null || true
screen -wipe 2>/dev/null || true

# Verify
if pgrep -f "openagent serve" > /dev/null 2>&1; then
    echo "WARNING: Some processes still running:"
    ps aux | grep "openagent serve" | grep -v grep
else
    echo "OpenAgent stopped."
fi
