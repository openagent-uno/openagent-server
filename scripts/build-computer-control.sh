#!/usr/bin/env bash
# Build the Rust computer-control MCP for the host platform and stage the
# binary into bin/<target>/ for builtins.py to discover.
#
# Usage:
#   ./scripts/build-computer-control.sh
#
# In CI release builds, this script is bypassed — GitHub Actions builds per
# target on matching runners, uploads artifacts, and the executable job
# drops them into bin/<target>/ directly.

set -euo pipefail

cd "$(dirname "$0")/../openagent/mcp/servers/computer-control"

TARGET="$(rustc -vV | sed -n 's|host: ||p')"
case "$TARGET" in
  aarch64-apple-darwin)      OUT=darwin-arm64 ; EXT='' ;;
  x86_64-unknown-linux-gnu)  OUT=linux-x64    ; EXT='' ;;
  x86_64-pc-windows-msvc)    OUT=windows-x64  ; EXT='.exe' ;;
  *) echo "Unsupported host target: $TARGET" >&2 ; exit 1 ;;
esac

cargo build --release --target "$TARGET"

mkdir -p "bin/$OUT"
cp "target/$TARGET/release/openagent-computer-control$EXT" "bin/$OUT/"
chmod +x "bin/$OUT/openagent-computer-control$EXT" 2>/dev/null || true

echo "Staged: bin/$OUT/openagent-computer-control$EXT"
