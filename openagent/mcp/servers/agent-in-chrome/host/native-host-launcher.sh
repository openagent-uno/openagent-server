#!/bin/sh
# Native messaging host launcher for OpenAgent in Chrome.
#
# Chrome/Chromium launches this script (manifest "path", no "args").
# We locate the Node.js binary and delegate to native-host.js.
#
# Changes to native-host.js take effect immediately on the next
# connectNative() call — no browser restart needed.

die() { echo "$@" >&2; exit 1; }

DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
BRIDGE="$DIR/native-host.js"

# Find Node — try common locations
for node in \
  /opt/homebrew/bin/node \
  /usr/local/bin/node \
  /usr/bin/node \
  /snap/bin/node \
  "$HOME/.nvm/versions/node"/*/bin/node; do
  [ -x "$node" ] || continue
  exec "$node" "$BRIDGE"
done

# Windows: check Program Files
if [ -d "$ProgramFiles/nodejs" ]; then
  [ -x "$ProgramFiles/nodejs/node.exe" ] && exec "$ProgramFiles/nodejs/node.exe" "$BRIDGE"
fi
if [ -d "${ProgramFiles(x86)}/nodejs" ]; then
  [ -x "${ProgramFiles(x86)}/nodejs/node.exe" ] && exec "${ProgramFiles(x86)}/nodejs/node.exe" "$BRIDGE"
fi

die "Error: Node.js not found. Install Node.js from https://nodejs.org"
