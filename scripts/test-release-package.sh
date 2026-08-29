#!/usr/bin/env bash
# Run platform-specific smoke checks against a packaged server release.

set -euo pipefail

DIST="${1:-dist}"
VERSION="$(python -c 'from src import __version__; print(__version__)')"
TMP_ROOT="$(mktemp -d "${RUNNER_TEMP:-/tmp}/openagent-release-smoke.XXXXXX")"
trap 'rm -r -- "$TMP_ROOT"' EXIT

case "${RUNNER_OS:-$(uname -s)}" in
    macOS|Darwin)
        ARCH_RAW="$(uname -m)"
        case "$ARCH_RAW" in
            arm64|aarch64) ARCH="arm64" ;;
            x86_64|amd64) ARCH="x64" ;;
            *) ARCH="$ARCH_RAW" ;;
        esac
        PACKAGE="openagent-${VERSION}-macos-${ARCH}.pkg"
        (cd "$DIST" && shasum -a 256 -c "${PACKAGE}.sha256")
        pkgutil --check-signature "$DIST/$PACKAGE"
        xcrun stapler validate "$DIST/$PACKAGE"
        pkgutil --expand-full "$DIST/$PACKAGE" "$TMP_ROOT/pkg"
        BINARY="$(find "$TMP_ROOT/pkg" -type f -path '*/openagent.app/Contents/MacOS/openagent' -print -quit)"
        SIDECAR="$(find "$TMP_ROOT/pkg" -type f -path '*/openagent.app/Contents/MacOS/openagent-computer-control' -print -quit)"
        test -n "$BINARY"
        test -n "$SIDECAR"
        test "$(find "$TMP_ROOT/pkg" -type f -path '*/openagent.app/Contents/MacOS/openagent' | wc -l | tr -d ' ')" = 1
        test "$(find "$TMP_ROOT/pkg" -type f -path '*/openagent.app/Contents/MacOS/openagent-computer-control' | wc -l | tr -d ' ')" = 1
        APP_BUNDLE="${BINARY%/Contents/MacOS/openagent}"
        codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"
        ;;
    Linux)
        PACKAGE="openagent-${VERSION}-linux-x64.tar.gz"
        (cd "$DIST" && sha256sum -c "${PACKAGE}.sha256")
        tar -tzf "$DIST/$PACKAGE" | sort > "$TMP_ROOT/archive-files.txt"
        printf '%s\n' openagent openagent-computer-control | sort > "$TMP_ROOT/expected-files.txt"
        diff -u "$TMP_ROOT/expected-files.txt" "$TMP_ROOT/archive-files.txt"
        tar -xzf "$DIST/$PACKAGE" -C "$TMP_ROOT"
        BINARY="$TMP_ROOT/openagent"
        ;;
    Windows|MINGW*|CYGWIN*|MSYS*)
        PACKAGE="openagent-${VERSION}-windows-x64.zip"
        (cd "$DIST" && sha256sum -c "${PACKAGE}.sha256")
        python - "$DIST/$PACKAGE" "$TMP_ROOT" <<'PY'
import sys
import zipfile
from pathlib import PurePosixPath

archive_path, destination = sys.argv[1:]
expected = {"openagent.exe", "openagent-computer-control.exe"}
with zipfile.ZipFile(archive_path) as archive:
    names = []
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        normalized = entry.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe ZIP entry: {entry.filename!r}")
        names.append(normalized)
    if set(names) != expected or len(names) != len(expected):
        raise SystemExit(f"unexpected ZIP entries: {sorted(names)!r}")
    archive.extractall(destination)
PY
        chmod +x "$TMP_ROOT/openagent.exe"
        BINARY="$TMP_ROOT/openagent.exe"
        ;;
    *)
        echo "Unsupported OS: ${RUNNER_OS:-$(uname -s)}" >&2
        exit 1
        ;;
esac

test -x "$BINARY" || test -f "$BINARY"
VERSION_OUTPUT="$($BINARY --version)"
case "$VERSION_OUTPUT" in
    *"$VERSION"*) ;;
    *) echo "unexpected --version output: $VERSION_OUTPUT" >&2; exit 1 ;;
esac

mkdir -p "$TMP_ROOT/agent"
"$BINARY" --agent-dir "$TMP_ROOT/agent" selfcheck --quiet --expect "$VERSION"
echo "release package smoke passed for $PACKAGE"
