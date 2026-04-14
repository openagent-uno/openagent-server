#!/usr/bin/env bash
# Sign + notarize a PyInstaller onefile binary for distribution on macOS.
#
# Usage: scripts/sign-notarize-macos.sh <path-to-binary>
#
# Reads these env vars (identical to the desktop job's signing setup so both
# flows share one set of GitHub Actions secrets):
#
#   CSC_LINK                       base64-encoded Developer ID .p12 cert
#   CSC_KEY_PASSWORD               password for the .p12
#   APPLE_ID                       Apple Developer account email
#   APPLE_APP_SPECIFIC_PASSWORD    app-specific password for notarytool
#   APPLE_TEAM_ID                  Apple Developer Team ID
#
# If any of the secrets is missing the script exits 0 (no-op — build will
# produce an unsigned binary). Failure to sign or notarize exits non-zero.
#
# Why the zip dance for notarization:
#   The Apple notary service accepts .app / .pkg / .dmg / .zip submissions.
#   A bare executable has to be wrapped in a zip. The notarization ticket
#   is recorded online by Apple — bare binaries can't be stapled, so
#   Gatekeeper will fetch the ticket from Apple's service on first launch
#   (hence the "Developer ID" badge + no quarantine prompt for users with
#   internet connectivity).

set -euo pipefail

BINARY="${1:-}"
if [ -z "$BINARY" ]; then
    echo "usage: $0 <path-to-binary>" >&2
    exit 2
fi
if [ ! -f "$BINARY" ]; then
    echo "not a file: $BINARY" >&2
    exit 2
fi

# ── Skip cleanly when secrets are missing ─────────────────────────────

if [ -z "${CSC_LINK:-}" ] || [ -z "${CSC_KEY_PASSWORD:-}" ]; then
    echo "⚠️  CSC_LINK / CSC_KEY_PASSWORD not set — skipping macOS signing"
    exit 0
fi

# ── Import the signing cert into a throwaway keychain ─────────────────

KEYCHAIN_PATH="${RUNNER_TEMP:-/tmp}/openagent-build.keychain-db"
KEYCHAIN_PASSWORD="build-$(uuidgen)"
CERT_FILE="${RUNNER_TEMP:-/tmp}/openagent-cert.p12"

echo "→ Importing signing certificate"
echo -n "$CSC_LINK" | base64 --decode > "$CERT_FILE"

# Fresh keychain each run so repeated invocations don't accumulate state.
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
# Put our keychain in the search list so codesign can find it, but don't
# evict the login keychain (breaks SSH auth on self-hosted runners).
security list-keychains -d user -s "$KEYCHAIN_PATH" $(security list-keychains -d user | sed 's/"//g')
security set-key-partition-list \
    -S apple-tool:,apple:,codesign: \
    -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"

# ── Resolve the Developer ID identity ────────────────────────────────

IDENTITY=$(security find-identity -v -p codesigning "$KEYCHAIN_PATH" \
    | grep "Developer ID Application" \
    | head -1 \
    | awk -F'"' '{print $2}')
if [ -z "$IDENTITY" ]; then
    echo "No Developer ID Application identity found in cert" >&2
    security find-identity -v "$KEYCHAIN_PATH" >&2
    exit 1
fi
echo "→ Signing with identity: $IDENTITY"

# ── Sign the binary ───────────────────────────────────────────────────

codesign --force \
    --sign "$IDENTITY" \
    --options runtime \
    --timestamp \
    --entitlements buildResources/entitlements.mac.plist \
    "$BINARY"
codesign --verify --strict --verbose=2 "$BINARY"

# ── Notarize ──────────────────────────────────────────────────────────

if [ -z "${APPLE_ID:-}" ] || [ -z "${APPLE_APP_SPECIFIC_PASSWORD:-}" ] || [ -z "${APPLE_TEAM_ID:-}" ]; then
    echo "⚠️  APPLE_ID / APPLE_APP_SPECIFIC_PASSWORD / APPLE_TEAM_ID not set"
    echo "   — binary is signed but NOT notarized. Users will still see a"
    echo "     Gatekeeper warning on first launch."
    exit 0
fi

NOTARIZE_ZIP="${RUNNER_TEMP:-/tmp}/$(basename "$BINARY")-notarize.zip"
echo "→ Packaging for notarization: $NOTARIZE_ZIP"
rm -f "$NOTARIZE_ZIP"
# ditto --keepParent preserves the filename so notarytool sees the binary
# not just a raw blob.
ditto -c -k --keepParent "$BINARY" "$NOTARIZE_ZIP"

echo "→ Submitting to Apple notary service (this can take a few minutes)"
xcrun notarytool submit "$NOTARIZE_ZIP" \
    --apple-id "$APPLE_ID" \
    --password "$APPLE_APP_SPECIFIC_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" \
    --wait

# Stapling is only possible on .app / .pkg / .dmg targets. For bare
# binaries Gatekeeper fetches the ticket from Apple on first launch
# (requires internet; works for ~99% of users). Nothing to staple.

echo "✓ Signed + notarized $BINARY"
