#!/bin/bash
set -euo pipefail

# ── OpenAgent Release ──
# Bumps version everywhere, tags, pushes → GitHub Actions builds:
#   - Python package → PyPI
#   - Electron app (macOS, Windows, Linux) → GitHub Release assets
#
# Usage:
#   ./release.sh patch    # 0.1.0 → 0.1.1
#   ./release.sh minor    # 0.1.0 → 0.2.0
#   ./release.sh major    # 0.1.0 → 1.0.0
#   ./release.sh 0.3.0    # explicit version

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUMP="${1:-patch}"

# ── Get current version ──
CURRENT=$(grep 'version = "' "$ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "Current version: $CURRENT"

# ── Calculate new version ──
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "$BUMP" in
  patch) NEW="$MAJOR.$MINOR.$((PATCH + 1))" ;;
  minor) NEW="$MAJOR.$((MINOR + 1)).0" ;;
  major) NEW="$((MAJOR + 1)).0.0" ;;
  *)     NEW="$BUMP" ;;
esac

echo "New version: $NEW"
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# ── Bump Python package ──
echo "📦 Bumping Python package..."
sed -i.bak "s/version = \"$CURRENT\"/version = \"$NEW\"/" "$ROOT/pyproject.toml"
sed -i.bak "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW\"/" "$ROOT/openagent/__init__.py"
rm -f "$ROOT/pyproject.toml.bak" "$ROOT/openagent/__init__.py.bak"

# ── Bump Electron app ──
echo "📦 Bumping desktop app..."
DESKTOP_PKG="$ROOT/app/desktop/package.json"
if [ -f "$DESKTOP_PKG" ]; then
    # Use node to update version in package.json (handles JSON properly)
    node -e "
      const fs = require('fs');
      const pkg = JSON.parse(fs.readFileSync('$DESKTOP_PKG', 'utf8'));
      pkg.version = '$NEW';
      fs.writeFileSync('$DESKTOP_PKG', JSON.stringify(pkg, null, 2) + '\n');
    "
    echo "  desktop/package.json → $NEW"
fi

UNIVERSAL_PKG="$ROOT/app/universal/package.json"
if [ -f "$UNIVERSAL_PKG" ]; then
    node -e "
      const fs = require('fs');
      const pkg = JSON.parse(fs.readFileSync('$UNIVERSAL_PKG', 'utf8'));
      pkg.version = '$NEW';
      fs.writeFileSync('$UNIVERSAL_PKG', JSON.stringify(pkg, null, 2) + '\n');
    "
    echo "  universal/package.json → $NEW"
fi

# ── Commit + tag + push ──
echo ""
echo "📤 Committing and pushing..."
cd "$ROOT"
git add pyproject.toml openagent/__init__.py app/desktop/package.json app/universal/package.json
git commit -m "release: v$NEW"
git tag "v$NEW"
git push origin main
git push origin "v$NEW"

echo ""
echo "=== Released v$NEW ==="
echo "GitHub Actions will now:"
echo "  1. Build Python package → PyPI"
echo "  2. Build Electron app (macOS, Windows, Linux) → GitHub Release"
echo "  3. Create GitHub Release with all assets"
echo ""
echo "Track: https://github.com/geroale/OpenAgent/actions"
echo "PyPI:  https://pypi.org/project/openagent-framework/"
