#!/bin/bash
set -euo pipefail

# ── OpenAgent Release ──
# Bumps version in ALL projects, tags, pushes → GitHub Actions builds:
#   - openagent server standalone executables (macOS, Linux, Windows) → GitHub Release
#   - openagent-cli standalone executables (macOS, Linux, Windows)    → GitHub Release
#   - OpenAgent Desktop (macOS, Windows, Linux)                        → GitHub Release
#
# All artifacts ship via GitHub Releases only — no PyPI publishing.
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
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

# ── 1. openagent-framework ──
echo "📦 openagent-framework → $NEW"
sed -i.bak "s/version = \"$CURRENT\"/version = \"$NEW\"/" "$ROOT/pyproject.toml"
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW\"/" "$ROOT/openagent/__init__.py"
rm -f "$ROOT/pyproject.toml.bak" "$ROOT/openagent/__init__.py.bak"

# ── 2. openagent-cli ──
echo "📦 openagent-cli → $NEW"
sed -i.bak "s/version = \".*\"/version = \"$NEW\"/" "$ROOT/cli/pyproject.toml"
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW\"/" "$ROOT/cli/openagent_cli/__init__.py"
rm -f "$ROOT/cli/pyproject.toml.bak" "$ROOT/cli/openagent_cli/__init__.py.bak"

# ── 3. Desktop app ──
echo "📦 desktop app → $NEW"
for pkg in "$ROOT/app/desktop/package.json" "$ROOT/app/universal/package.json"; do
  [ -f "$pkg" ] && node -e "
    const fs = require('fs');
    const p = JSON.parse(fs.readFileSync('$pkg','utf8'));
    p.version = '$NEW';
    fs.writeFileSync('$pkg', JSON.stringify(p,null,2)+'\n');
  " && echo "  $(basename $(dirname $pkg))/package.json → $NEW"
done

# ── Commit + tag + push ──
echo ""
echo "📤 Committing..."
cd "$ROOT"
git add pyproject.toml openagent/__init__.py \
       cli/pyproject.toml cli/openagent_cli/__init__.py \
       app/desktop/package.json app/universal/package.json
git commit -m "release: v$NEW"
git tag "v$NEW"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$BRANCH" "v$NEW"

echo ""
echo "=== Released v$NEW ==="
echo ""
echo "GitHub Actions will now build & publish to GitHub Releases:"
echo "  1. openagent server executables → macOS / Linux / Windows"
echo "  2. openagent-cli executables    → macOS / Linux / Windows"
echo "  3. OpenAgent Desktop            → macOS / Windows / Linux"
echo ""
echo "Track: https://github.com/geroale/OpenAgent/actions"
