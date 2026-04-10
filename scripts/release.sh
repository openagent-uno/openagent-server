#!/bin/bash
# Release OpenAgent: bump version, tag, push → GitHub Actions publishes to PyPI + GitHub Release
#
# Usage:
#   ./release.sh patch    # 0.1.0 → 0.1.1
#   ./release.sh minor    # 0.1.0 → 0.2.0
#   ./release.sh major    # 0.1.0 → 1.0.0
#   ./release.sh 0.3.0    # explicit version

set -e

BUMP="${1:-patch}"

# Get current version
CURRENT=$(grep 'version = ' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "Current version: $CURRENT"

# Calculate new version
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "$BUMP" in
  patch) NEW="$MAJOR.$MINOR.$((PATCH + 1))" ;;
  minor) NEW="$MAJOR.$((MINOR + 1)).0" ;;
  major) NEW="$((MAJOR + 1)).0.0" ;;
  *)     NEW="$BUMP" ;;  # explicit version
esac

echo "New version: $NEW"
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Update version in pyproject.toml and __init__.py
sed -i.bak "s/version = \"$CURRENT\"/version = \"$NEW\"/" pyproject.toml
sed -i.bak "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW\"/" openagent/__init__.py
rm -f pyproject.toml.bak openagent/__init__.py.bak

# Commit + tag + push
git add pyproject.toml openagent/__init__.py
git commit -m "release: v$NEW"
git tag "v$NEW"
git push origin main
git push origin "v$NEW"

echo ""
echo "=== Released v$NEW ==="
echo "GitHub Actions will now:"
echo "  1. Build the package"
echo "  2. Publish to PyPI"
echo "  3. Create GitHub Release"
echo ""
echo "Track: https://github.com/geroale/OpenAgent/actions"
echo "PyPI:  https://pypi.org/project/openagent/"
