#!/bin/bash
# Check OpenAgent status.
#
# Usage: ./scripts/status.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG="$PROJECT_DIR/openagent.log"

echo "=== OpenAgent Status ==="

# Process
if pgrep -f "openagent serve" > /dev/null 2>&1; then
    INSTANCES=$(pgrep -f "openagent serve" | wc -l)
    echo "Status: RUNNING ($INSTANCES instance(s))"
    ps aux | grep "openagent serve" | grep -v grep
else
    echo "Status: STOPPED"
fi

echo ""

# Screen
echo "Screen sessions:"
screen -ls 2>/dev/null | grep openagent || echo "  None"

echo ""

# Last log lines
if [ -f "$LOG" ]; then
    echo "Last log:"
    tail -10 "$LOG"
else
    echo "No log file found."
fi
