#!/usr/bin/env bash
# Sign + notarize a PyInstaller onefile binary for macOS distribution, and
# optionally wrap the signed binary in a notarized + stapled .pkg installer.
#
# Usage:
#   scripts/sign-notarize-macos.sh <binary>
#   scripts/sign-notarize-macos.sh <binary> <pkg-identifier> <install-path>
#   scripts/sign-notarize-macos.sh <binary> <pkg-identifier> <install-path> <extra-sidecar-binary>
#
# Examples:
#   # Sign + notarize the bare binary only (no .pkg):
#   scripts/sign-notarize-macos.sh dist/openagent
#
#   # Sign + notarize AND produce a .pkg that installs into /usr/local/bin.
#   # Output path is derived as
#   #   dist/<binary-basename>-<version>-macos-<arch>.pkg
#   # where <version> comes from the Python module matching the binary name.
#   scripts/sign-notarize-macos.sh dist/openagent \
#       com.openagent.server /usr/local/bin
#
#   # Same as above, PLUS bundle an already-signed sidecar binary into the
#   # .pkg at the same install path. Used for openagent-computer-control
#   # (signed with a stable TCC identifier in its own CI job). The sidecar
#   # is NOT re-signed here — its existing Developer-ID signature must be
#   # preserved verbatim so macOS can record a persistent TCC grant.
#   scripts/sign-notarize-macos.sh dist/openagent \
#       com.openagent.server /usr/local/bin \
#       dist/openagent-computer-control
#
# Env vars (identical to the desktop electron-builder job so both flows
# share one set of GitHub Actions secrets):
#
#   # Required for binary signing:
#   CSC_LINK                         base64 Developer ID Application .p12
#   CSC_KEY_PASSWORD                 password for the .p12
#
#   # Required for notarization:
#   APPLE_ID                         Apple Developer account email
#   APPLE_APP_SPECIFIC_PASSWORD      app-specific password for notarytool
#   APPLE_TEAM_ID                    Apple Developer Team ID
#
#   # Required only for the .pkg flow (ignored otherwise):
#   CSC_LINK_INSTALLER               base64 Developer ID Installer .p12
#   CSC_KEY_PASSWORD_INSTALLER       password for the Installer .p12
#                                    (falls back to CSC_KEY_PASSWORD if unset)
#
# Behaviour when secrets are missing:
#
#   - CSC_LINK missing          → script exits 0, binary left unsigned
#   - APPLE_ID missing          → binary signed but not notarized
#   - CSC_LINK_INSTALLER missing + pkg requested → pkg step skipped (no error)
#
# Why we also ship a .pkg:
#
#   Bare executables can't have Apple's notarization ticket *stapled* to
#   them (stapler only supports .app, .pkg, .dmg). On modern macOS, Finder
#   double-click of a browser-downloaded (quarantined) bare binary shows
#   the scary "Apple cannot verify" dialog even when the binary is signed
#   + notarized, because Gatekeeper's runtime ticket lookup for bare
#   executables intentionally prompts. A .pkg with a stapled ticket shows
#   zero warnings on first launch. For terminal users the tar.gz is fine;
#   for anyone who downloads through Safari and double-clicks, the .pkg
#   is the "just works" install flow.

set -euo pipefail

BINARY="${1:-}"
PKG_IDENTIFIER="${2:-}"
PKG_INSTALL_PATH="${3:-}"
EXTRA_SIDECAR="${4:-}"

if [ -z "$BINARY" ]; then
    echo "usage: $0 <binary> [pkg-identifier pkg-install-path [extra-sidecar-binary]]" >&2
    exit 2
fi
if [ ! -f "$BINARY" ]; then
    echo "not a file: $BINARY" >&2
    exit 2
fi
if [ -n "$EXTRA_SIDECAR" ] && [ ! -f "$EXTRA_SIDECAR" ]; then
    echo "sidecar binary not a file: $EXTRA_SIDECAR" >&2
    exit 2
fi

WANT_PKG=false
if [ -n "$PKG_IDENTIFIER" ] || [ -n "$PKG_INSTALL_PATH" ]; then
    if [ -z "$PKG_IDENTIFIER" ] || [ -z "$PKG_INSTALL_PATH" ]; then
        echo "pkg mode requires both: identifier and install-path" >&2
        exit 2
    fi
    WANT_PKG=true
fi

# Compute the pkg output path here so callers don't have to duplicate
# version/arch detection. Filename follows the release convention:
#   <binary-dir>/<binary-basename>-<version>-macos-<arch>.pkg
# e.g. dist/openagent-0.5.7-macos-arm64.pkg.
if [ "$WANT_PKG" = true ]; then
    BINARY_DIR="$(dirname "$BINARY")"
    BINARY_NAME="$(basename "$BINARY")"
    MODULE="${BINARY_NAME//-/_}"
    PKG_VERSION="$(python -c "import ${MODULE}; print(${MODULE}.__version__)")"
    PKG_ARCH_RAW="$(uname -m)"
    case "$PKG_ARCH_RAW" in
        x86_64|amd64)  PKG_ARCH="x64" ;;
        aarch64|arm64) PKG_ARCH="arm64" ;;
        *) PKG_ARCH="$PKG_ARCH_RAW" ;;
    esac
    PKG_OUTPUT="${BINARY_DIR}/${BINARY_NAME}-${PKG_VERSION}-macos-${PKG_ARCH}.pkg"
fi

# ── Skip cleanly when the binary-signing secrets are missing ──────────

if [ -z "${CSC_LINK:-}" ] || [ -z "${CSC_KEY_PASSWORD:-}" ]; then
    echo "⚠️  CSC_LINK / CSC_KEY_PASSWORD not set — skipping macOS signing"
    exit 0
fi

# ── Build a throwaway keychain + import the Application cert ──────────

KEYCHAIN_PATH="${RUNNER_TEMP:-/tmp}/openagent-build.keychain-db"
KEYCHAIN_PASSWORD="build-$(uuidgen)"
CERT_FILE="${RUNNER_TEMP:-/tmp}/openagent-cert.p12"

echo "→ Importing Application cert into keychain $KEYCHAIN_PATH"
echo -n "$CSC_LINK" | base64 --decode > "$CERT_FILE"

if [ -f "$KEYCHAIN_PATH" ]; then
    security delete-keychain "$KEYCHAIN_PATH" 2>/dev/null || true
fi
security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH"
security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
security import "$CERT_FILE" \
    -P "$CSC_KEY_PASSWORD" \
    -A -t cert -f pkcs12 \
    -k "$KEYCHAIN_PATH"
# Keep the login keychain in the search list (don't evict it — that
# breaks SSH on self-hosted runners) but make our new keychain default
# for codesign / productsign lookups.
security list-keychains -d user -s "$KEYCHAIN_PATH" $(security list-keychains -d user | sed 's/"//g')
security set-key-partition-list \
    -S apple-tool:,apple:,codesign:,productsign: \
    -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"

# ── Also import the Installer cert (same keychain) if we're building pkg ─
#
# When the caller asked for a .pkg but the Installer cert secret is
# missing we fail HARD rather than silently producing only the tar.gz
# artifact — the downstream workflow expects a .pkg and dropping to
# "tar.gz only" would confuse everyone. The error message names the
# two secrets so whoever's configuring the repo knows exactly what to
# add.

INSTALLER_CERT_FILE="${RUNNER_TEMP:-/tmp}/openagent-installer-cert.p12"
HAVE_INSTALLER_CERT=false
if [ "$WANT_PKG" = true ]; then
    if [ -z "${CSC_LINK_INSTALLER:-}" ]; then
        echo "ERROR: .pkg build requested but CSC_LINK_INSTALLER is not set." >&2
        echo "  Add these two GitHub Actions repository secrets:" >&2
        echo "    CSC_LINK_INSTALLER          base64-encoded Developer ID Installer .p12" >&2
        echo "    CSC_KEY_PASSWORD_INSTALLER  password for the .p12" >&2
        exit 1
    fi
    INSTALLER_PASSWORD="${CSC_KEY_PASSWORD_INSTALLER:-$CSC_KEY_PASSWORD}"
    echo "→ Importing Installer cert into same keychain"
    echo -n "$CSC_LINK_INSTALLER" | base64 --decode > "$INSTALLER_CERT_FILE"
    security import "$INSTALLER_CERT_FILE" \
        -P "$INSTALLER_PASSWORD" \
        -A -t cert -f pkcs12 \
        -k "$KEYCHAIN_PATH"
    security set-key-partition-list \
        -S apple-tool:,apple:,codesign:,productsign: \
        -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
    HAVE_INSTALLER_CERT=true
fi

# ── Resolve signing identities ────────────────────────────────────────

APP_IDENTITY=$(security find-identity -v -p codesigning "$KEYCHAIN_PATH" \
    | grep "Developer ID Application" \
    | head -1 \
    | awk -F'"' '{print $2}')
if [ -z "$APP_IDENTITY" ]; then
    echo "No Developer ID Application identity found in cert" >&2
    security find-identity -v "$KEYCHAIN_PATH" >&2
    exit 1
fi
echo "→ App identity: $APP_IDENTITY"

INSTALLER_IDENTITY=""
if [ "$HAVE_INSTALLER_CERT" = true ]; then
    # The Installer cert isn't a codesigning identity — search with -p basic
    # which includes installer identities in the output.
    INSTALLER_IDENTITY=$(security find-identity -v -p basic "$KEYCHAIN_PATH" \
        | grep "Developer ID Installer" \
        | head -1 \
        | awk -F'"' '{print $2}')
    if [ -z "$INSTALLER_IDENTITY" ]; then
        echo "⚠️  CSC_LINK_INSTALLER was set but no Developer ID Installer identity resolved"
        HAVE_INSTALLER_CERT=false
    else
        echo "→ Installer identity: $INSTALLER_IDENTITY"
    fi
fi

# ── Sign the binary ───────────────────────────────────────────────────

# macOS TCC (Accessibility, Screen Recording, etc.) keys each permission
# grant to the binary's CODE-SIGN IDENTIFIER. Without an explicit
# ``--identifier`` flag, codesign defaults to the binary's filename —
# e.g. ``Identifier=openagent``. That's a valid signature but it's NOT
# a valid bundle identifier, and TCC refuses to record a persistent
# entry for non-reverse-DNS identifiers: the permission dialog fires
# correctly, the user clicks "Open System Settings", but no toggle
# ever appears in the Accessibility pane because TCC silently dropped
# the attempted row insertion. Observed on v0.6.7 — the user got the
# prompt, got redirected to Settings, saw no openagent entry at all.
#
# ``$PKG_IDENTIFIER`` (2nd arg) is already a valid reverse-DNS string
# (e.g. ``com.openagent.server``) that the caller passes to pkgbuild;
# we reuse it here so the binary's codesign identifier stays
# consistent with the installer package. When the caller didn't ask
# for a .pkg (pkg-identifier not set), fall back to prefixing the
# binary name with ``com.openagent.``.
if [ -n "$PKG_IDENTIFIER" ]; then
    SIGN_IDENTIFIER="$PKG_IDENTIFIER"
else
    SIGN_IDENTIFIER="com.openagent.$(basename "$BINARY" | tr -d '[:space:]')"
fi
echo "→ Signing $BINARY with identifier $SIGN_IDENTIFIER"
codesign --force \
    --sign "$APP_IDENTITY" \
    --identifier "$SIGN_IDENTIFIER" \
    --options runtime \
    --timestamp \
    --entitlements buildResources/entitlements.mac.plist \
    "$BINARY"
codesign --verify --strict --verbose=2 "$BINARY"
codesign -dvv "$BINARY" 2>&1 | grep -E '^(Identifier|TeamIdentifier)=' || true
echo "✓ Binary signed"

# ── Notarize the bare binary (ticket goes into Apple's online DB) ────

if [ -z "${APPLE_ID:-}" ] || [ -z "${APPLE_APP_SPECIFIC_PASSWORD:-}" ] || [ -z "${APPLE_TEAM_ID:-}" ]; then
    echo "⚠️  APPLE_ID / APPLE_APP_SPECIFIC_PASSWORD / APPLE_TEAM_ID not set"
    echo "   — binary is signed but NOT notarized. Manual downloads will"
    echo "     still trigger Gatekeeper on first launch."
    exit 0
fi

BINARY_ZIP="${RUNNER_TEMP:-/tmp}/$(basename "$BINARY")-notarize.zip"
echo "→ Notarizing bare binary (ticket recorded online — bare binaries can't be stapled)"
rm -f "$BINARY_ZIP"
ditto -c -k --keepParent "$BINARY" "$BINARY_ZIP"
xcrun notarytool submit "$BINARY_ZIP" \
    --apple-id "$APPLE_ID" \
    --password "$APPLE_APP_SPECIFIC_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" \
    --wait
echo "✓ Bare binary notarized"

# ── Build signed + notarized + stapled .pkg ───────────────────────────

if [ "$WANT_PKG" = false ]; then
    exit 0
fi
if [ "$HAVE_INSTALLER_CERT" = false ]; then
    echo "⚠️  Installer cert unavailable — skipping .pkg build"
    exit 0
fi

# Lay out the filesystem tree the installer should write. pkgbuild picks
# up the subtree under --root and maps it 1:1 onto the user's disk.
PKG_ROOT="${RUNNER_TEMP:-/tmp}/openagent-pkg-root-$(uuidgen)"
rm -rf "$PKG_ROOT"
mkdir -p "$PKG_ROOT$PKG_INSTALL_PATH"
cp "$BINARY" "$PKG_ROOT$PKG_INSTALL_PATH/$(basename "$BINARY")"
chmod +x "$PKG_ROOT$PKG_INSTALL_PATH/$(basename "$BINARY")"

# Sidecar binary (optional, already-signed). Copied verbatim so its
# Developer-ID signature with ``com.openagent.computer-control`` stays
# intact — that stable identifier is what makes macOS TCC grants for
# Accessibility / Screen Recording survive openagent updates. The
# .pkg notarization below covers this binary too (notarytool walks
# the pkg payload and notarizes every embedded Mach-O).
if [ -n "$EXTRA_SIDECAR" ]; then
    echo "→ Adding sidecar to pkg payload: $(basename "$EXTRA_SIDECAR")"
    cp "$EXTRA_SIDECAR" "$PKG_ROOT$PKG_INSTALL_PATH/$(basename "$EXTRA_SIDECAR")"
    chmod +x "$PKG_ROOT$PKG_INSTALL_PATH/$(basename "$EXTRA_SIDECAR")"
    codesign -dvv "$PKG_ROOT$PKG_INSTALL_PATH/$(basename "$EXTRA_SIDECAR")" 2>&1 \
        | grep -E '^(Identifier|TeamIdentifier|Authority)=' || true
fi

UNSIGNED_PKG="${RUNNER_TEMP:-/tmp}/$(basename "$PKG_OUTPUT" .pkg)-unsigned.pkg"
echo "→ Building unsigned .pkg with identifier $PKG_IDENTIFIER"
# ``pkgbuild`` produces a "component" pkg. --install-location / tells it
# to preserve the PKG_ROOT layout; we've laid out the absolute path
# already so the installer writes into $PKG_INSTALL_PATH.
pkgbuild \
    --identifier "$PKG_IDENTIFIER" \
    --version "$PKG_VERSION" \
    --install-location / \
    --root "$PKG_ROOT" \
    "$UNSIGNED_PKG"

echo "→ Signing .pkg with $INSTALLER_IDENTITY"
productsign \
    --sign "$INSTALLER_IDENTITY" \
    --keychain "$KEYCHAIN_PATH" \
    "$UNSIGNED_PKG" \
    "$PKG_OUTPUT"
pkgutil --check-signature "$PKG_OUTPUT"

echo "→ Notarizing .pkg"
xcrun notarytool submit "$PKG_OUTPUT" \
    --apple-id "$APPLE_ID" \
    --password "$APPLE_APP_SPECIFIC_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" \
    --wait

echo "→ Stapling notarization ticket to .pkg"
xcrun stapler staple "$PKG_OUTPUT"
xcrun stapler validate "$PKG_OUTPUT"

echo "✓ Built + signed + notarized + stapled: $PKG_OUTPUT"
