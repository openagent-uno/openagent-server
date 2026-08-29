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

sed -i.bak "s/version = \"[^\"]*\"/version = \"$NEW\"/" "$ROOT/pyproject.toml"
sed -i.bak "s/__version__ = \"[^\"]*\"/__version__ = \"$NEW\"/" "$ROOT/src/__init__.py"
rm -f "$ROOT/pyproject.toml.bak" "$ROOT/src/__init__.py.bak"

# uv.lock records the root package's own version, and the release test suite
# asserts it matches pyproject. Bumping two files and not the third made every
# tag fail in CI, AFTER the tag had already been pushed. Only the root package
# block is touched: re-resolving the whole lock during a release would drag in
# unrelated dependency updates that nothing has tested.
python3 - "$ROOT/uv.lock" "$NEW" <<'PYBUMP'
import pathlib, re, sys

path, version = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text(encoding="utf-8")
pattern = re.compile(
    r'(\[\[package\]\]\nname = "openagent-framework"\nversion = ")([^"]+)(")'
)
if not pattern.search(text):
    raise SystemExit("uv.lock: openagent-framework package block not found")
path.write_text(
    pattern.sub(rf"\g<1>{version}\g<3>", text, count=1), encoding="utf-8",
)
PYBUMP

# ── Commit + tag + push ──
echo ""
echo "📤 Committing and tagging v$NEW..."

git add pyproject.toml src/__init__.py uv.lock
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
