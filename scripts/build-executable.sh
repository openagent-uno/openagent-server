#!/usr/bin/env bash
# Build the OpenAgent standalone executable.
#
# Usage:
#   ./scripts/build-executable.sh
#
# Prerequisites:
#   - Python 3.11+
#   - Node.js 18+ (for built-in Node MCPs)
#   - pip install pyinstaller
#
# Output:
#   dist/openagent/          (onedir bundle)
#   dist/openagent-<os>-<arch>.tar.gz  (or .zip on Windows)
#   dist/openagent-<os>-<arch>.sha256  (checksum)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "=== OpenAgent Executable Builder ==="
echo ""

# ── Step 1: Install Python dependencies ──
echo "→ Installing Python dependencies..."
pip install -e ".[all]" --quiet
pip install pyinstaller --quiet

# ── Step 2: Build Node.js MCPs ──
echo "→ Building built-in Node MCPs..."

NODE_MCPS=(computer-control shell web-search editor chrome-devtools messaging)
for mcp in "${NODE_MCPS[@]}"; do
    mcp_dir="openagent/mcps/$mcp"
    if [ ! -d "$mcp_dir" ]; then
        echo "  ⚠ Skipping $mcp (directory not found)"
        continue
    fi

    if [ ! -d "$mcp_dir/node_modules" ]; then
        echo "  Installing $mcp..."
        (cd "$mcp_dir" && npm install --silent 2>/dev/null)
    fi

    # Build if package.json has a build script and dist/ doesn't exist
    if [ ! -d "$mcp_dir/dist" ] && grep -q '"build"' "$mcp_dir/package.json" 2>/dev/null; then
        echo "  Building $mcp..."
        (cd "$mcp_dir" && npm run build --silent 2>/dev/null)
    fi

    echo "  ✓ $mcp"
done

# ── Step 3: Run PyInstaller ──
echo ""
echo "→ Running PyInstaller..."
pyinstaller openagent.spec --clean --noconfirm

# ── Step 4: Package ──
echo ""
echo "→ Packaging..."

# Detect platform and architecture
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$OS" in
    darwin) OS_NAME="macos" ;;
    linux)  OS_NAME="linux" ;;
    *)      OS_NAME="$OS" ;;
esac

case "$ARCH" in
    x86_64)  ARCH_NAME="x64" ;;
    aarch64|arm64) ARCH_NAME="arm64" ;;
    *)       ARCH_NAME="$ARCH" ;;
esac

VERSION=$(python -c "import openagent; print(openagent.__version__)")
ARCHIVE_NAME="openagent-${VERSION}-${OS_NAME}-${ARCH_NAME}"

cd dist
tar czf "${ARCHIVE_NAME}.tar.gz" openagent/
shasum -a 256 "${ARCHIVE_NAME}.tar.gz" > "${ARCHIVE_NAME}.tar.gz.sha256"
cd ..

echo ""
echo "✓ Build complete!"
echo "  Archive: dist/${ARCHIVE_NAME}.tar.gz"
echo "  Checksum: dist/${ARCHIVE_NAME}.tar.gz.sha256"
