#!/bin/bash
# Auto-setup script for OpenAgent in Chrome.
# Installs native messaging manifests for all detected Chromium browsers.
# The extension ID is deterministic (baked into manifest.json) so no user input needed.
#
# Called by OpenAgent on startup, or can be run manually.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTENSION_DIR="$SCRIPT_DIR/extension"
HOST_DIR="$SCRIPT_DIR/host"
HOST_NAME="com.openagent.agent_in_chrome"
NATIVE_HOST_PATH="$HOST_DIR/native-host-wrapper.sh"

if [ ! -f "$EXTENSION_DIR/extension-id.txt" ]; then
  echo "Error: extension-id.txt not found. Run generate-key.py first." >&2
  exit 1
fi

EXTENSION_ID="$(cat "$EXTENSION_DIR/extension-id.txt" | tr -d '\n')"
ORIGIN="chrome-extension://${EXTENSION_ID}/"

echo "OpenAgent in Chrome auto-setup"
echo "Extension ID: $EXTENSION_ID"
echo ""

# Ensure wrapper is executable
chmod +x "$NATIVE_HOST_PATH" 2>/dev/null || true

# Generate native messaging host manifest
generate_manifest() {
  cat << EOF
{
  "name": "$HOST_NAME",
  "description": "OpenAgent in Chrome Native Messaging Host",
  "path": "$NATIVE_HOST_PATH",
  "type": "stdio",
  "allowed_origins": [
    "$ORIGIN"
  ]
}
EOF
}

install_for_browser() {
  local name="$1"
  local dir="$2"

  if [ ! -d "$(dirname "$dir")" ]; then
    return 0  # Browser not installed, silently skip
  fi

  mkdir -p "$dir"
  local dst="$dir/$HOST_NAME.json"
  generate_manifest > "$dst"
  echo "  [$name] Installed: $dst"
}

echo "Installing native messaging hosts..."

case "$(uname)" in
  Darwin)
    install_for_browser "Chrome"     "$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts"
    install_for_browser "Chrome Beta" "$HOME/Library/Application Support/Google/Chrome Beta/NativeMessagingHosts"
    install_for_browser "Chrome Canary" "$HOME/Library/Application Support/Google/Chrome Canary/NativeMessagingHosts"
    install_for_browser "Chromium"   "$HOME/Library/Application Support/Chromium/NativeMessagingHosts"
    install_for_browser "Edge"       "$HOME/Library/Application Support/Microsoft Edge/NativeMessagingHosts"
    install_for_browser "Brave"      "$HOME/Library/Application Support/BraveSoftware/Brave-Browser/NativeMessagingHosts"
    install_for_browser "Arc"        "$HOME/Library/Application Support/Arc/NativeMessagingHosts"
    install_for_browser "Opera"      "$HOME/Library/Application Support/com.operasoftware.Opera/NativeMessagingHosts"
    install_for_browser "Vivaldi"    "$HOME/Library/Application Support/Vivaldi/NativeMessagingHosts"
    ;;
  Linux)
    install_for_browser "Chrome"     "$HOME/.config/google-chrome/NativeMessagingHosts"
    install_for_browser "Chrome Beta" "$HOME/.config/google-chrome-beta/NativeMessagingHosts"
    install_for_browser "Chromium"   "$HOME/.config/chromium/NativeMessagingHosts"
    install_for_browser "Edge"       "$HOME/.config/microsoft-edge/NativeMessagingHosts"
    install_for_browser "Brave"      "$HOME/.config/BraveSoftware/Brave-Browser/NativeMessagingHosts"
    install_for_browser "Vivaldi"    "$HOME/.config/vivaldi/NativeMessagingHosts"
    install_for_browser "Opera"      "$HOME/.config/opera/NativeMessagingHosts"
    ;;
  *)
    echo "Warning: unsupported platform $(uname)" >&2
    exit 1
    ;;
esac

echo ""
echo "Native messaging hosts installed."
echo ""

# Create launch wrappers
create_launcher() {
  local name="$1"
  local app_path="$2"
  local mac_app="$3"

  local launcher_path="$HOST_DIR/launch-$name.sh"

  if [ "$(uname)" = "Darwin" ] && [ -n "$mac_app" ] && [ -d "$mac_app" ]; then
    cat > "$launcher_path" << LAUNCHEOF
#!/bin/bash
# Launch $name with OpenAgent in Chrome extension loaded.
# Close all $name windows first for the flag to take effect.
osascript -e 'quit app "$name"' 2>/dev/null
sleep 1
open -a "$mac_app" --args --load-extension="$EXTENSION_DIR" --no-first-run --no-default-browser-check
echo "Launching $name with OpenAgent extension..."
LAUNCHEOF
    chmod +x "$launcher_path"
    echo "  Created launcher: $launcher_path"
  elif [ "$(uname)" = "Linux" ] && command -v "$app_path" &>/dev/null; then
    cat > "$launcher_path" << LAUNCHEOF
#!/bin/bash
# Launch $name with OpenAgent in Chrome extension loaded.
pkill -f "$app_path" 2>/dev/null || true
sleep 1
exec "$app_path" --load-extension="$EXTENSION_DIR" --no-first-run --no-default-browser-check &
echo "Launching $name with OpenAgent extension..."
LAUNCHEOF
    chmod +x "$launcher_path"
    echo "  Created launcher: $launcher_path"
  fi
}

echo "Creating browser launchers..."
echo "  (Use these to launch your browser with the extension auto-loaded)"
echo ""

create_launcher "chrome" "google-chrome" "/Applications/Google Chrome.app"
create_launcher "chrome-beta" "google-chrome-beta" "/Applications/Google Chrome Beta.app"
create_launcher "chromium" "chromium-browser" "/Applications/Chromium.app"
create_launcher "edge" "microsoft-edge" "/Applications/Microsoft Edge.app"
create_launcher "brave" "brave-browser" "/Applications/Brave Browser.app"
create_launcher "arc" "" "/Applications/Arc.app"

echo ""
echo "Done! To start using OpenAgent in Chrome:"
echo ""
echo "  1. Launch your browser using one of the launchers above:"
echo "     $HOST_DIR/launch-chrome.sh"
echo ""
echo "  2. Or manually launch your browser with:"
echo "     <browser> --load-extension=$EXTENSION_DIR"
echo ""
echo "  3. Restart OpenAgent (or it will connect on next startup)"
echo ""
