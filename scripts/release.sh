#!/bin/bash
set -euo pipefail

# ── OpenAgent Server Release ──
# Bumps version in pyproject.toml + src/__init__.py, tags, pushes.
# CI (.github/workflows/release.yml) builds standalone executables for
# macOS / Linux / Windows and publishes them to GitHub Releases.
#
# Usage:
#   ./release.sh patch    # 0.13.8 → 0.13.9
#   ./release.sh minor    # 0.13.8 → 0.14.0
#   ./release.sh major    # 0.13.8 → 1.0.0
#   ./release.sh 0.14.0   # explicit version

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

BUMP="${1:-patch}"

CURRENT=$(grep 'version = "' "$ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "Current version: $CURRENT"

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"
case "$BUMP" in
  patch) NEW="$MAJOR.$MINOR.$((PATCH + 1))" ;;
  minor) NEW="$MAJOR.$((MINOR + 1)).0" ;;
  major) NEW="$((MAJOR + 1)).0.0" ;;
  *)     NEW="$BUMP" ;;
esac

if [ "$CURRENT" = "$NEW" ]; then
  echo "Version unchanged ($CURRENT). Nothing to do."
  exit 0
fi

echo "New version: $NEW"

# ── Check clean working tree ──
if ! git diff-index --quiet HEAD --; then
  echo "ERROR: working tree is dirty. Commit or stash changes first." >&2
  exit 1
fi

read -p "Continue? [y/N] " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

# ── Bump version ──
echo "📦 Bumping $CURRENT → $NEW"

sed -i.bak "s/version = \"$CURRENT\"/version = \"$NEW\"/" "$ROOT/pyproject.toml"
sed -i.bak "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW\"/" "$ROOT/src/__init__.py"
rm -f "$ROOT/pyproject.toml.bak" "$ROOT/src/__init__.py.bak"

# ── Commit + tag + push ──
echo ""
echo "📤 Committing and tagging v$NEW..."

git add pyproject.toml src/__init__.py
git commit -m "release: v$NEW"
git tag "v$NEW" -m "v$NEW"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$BRANCH" "v$NEW"

echo ""
echo "=== Released v$NEW ==="
echo ""
echo "GitHub Actions will now build & publish to GitHub Releases:"
echo "  https://github.com/openagent-uno/openagent-server/releases"
echo ""
echo "Track: https://github.com/openagent-uno/openagent-server/actions"
