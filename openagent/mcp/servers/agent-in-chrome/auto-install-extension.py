#!/usr/bin/env python3
"""
Launch a dedicated Chromium browser with the OpenAgent extension.

Google Chrome stable blocks ``--load-extension``, so Chromium is
auto-downloaded (~150 MB, cached).  The manifest uses a direct node
binary path (with ``args``) so Chromium can launch the native host
reliably.

macOS: Chromium is ad-hoc re-signed and launched with ``--in-process-gpu``
       to work around Apple Silicon sandbox issues.
Linux & Windows: native binaries work without extra steps.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


EXTENSION_ID = "npejdhmhoommpolfaphlggfjmhaipeib"
SCRIPT_DIR = Path(__file__).resolve().parent
EXTENSION_DIR = SCRIPT_DIR / "extension"
HOST_DIR = SCRIPT_DIR / "host"
PIDFILE = HOST_DIR / ".agent-chrome.pid"
SYSTEM = platform.system()  # Darwin | Linux | Windows

# Chromium cache location (prefer /Applications, fall back to ~/.openagent)
if SYSTEM == "Darwin":
    CHROMIUM_APP = "/Applications/Chromium.app"
    CHROMIUM_BINARY = Path(CHROMIUM_APP) / "Contents" / "MacOS" / "Chromium"
    CHROMIUM_DIR = Path.home() / ".openagent" / "chromium"
elif SYSTEM == "Linux":
    CHROMIUM_DIR = Path.home() / ".openagent" / "chromium"
    CHROMIUM_BINARY = CHROMIUM_DIR / "chrome"
elif SYSTEM == "Windows":
    CHROMIUM_DIR = Path.home() / ".openagent" / "chromium"
    CHROMIUM_BINARY = CHROMIUM_DIR / "chrome.exe"
else:
    CHROMIUM_DIR = Path.home() / ".openagent" / "chromium"
    CHROMIUM_BINARY = CHROMIUM_DIR / "chrome"

# ============================================================
#  Chromium download
# ============================================================

def _chromium_url() -> tuple[str, str]:
    base = "https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o"
    if SYSTEM == "Darwin":
        arch = "Mac_Arm" if platform.machine() in ("arm64", "aarch64") else "Mac"
        pos = _http_get(f"{base}/{arch}%2FLAST_CHANGE?alt=media")
        return f"{base}/{arch}%2F{pos}%2Fchrome-mac.zip?alt=media", "chrome-mac.zip"
    elif SYSTEM == "Linux":
        pos = _http_get(f"{base}/Linux_x64%2FLAST_CHANGE?alt=media")
        return f"{base}/Linux_x64%2F{pos}%2Fchrome-linux.zip?alt=media", "chrome-linux.zip"
    elif SYSTEM == "Windows":
        arch = "Win_Arm" if os.environ.get("PROCESSOR_ARCHITECTURE","") in ("ARM64","ARM") else "Win_x64"
        pos = _http_get(f"{base}/{arch}%2FLAST_CHANGE?alt=media")
        return f"{base}/{arch}%2F{pos}%2Fchrome-win.zip?alt=media", "chrome-win.zip"
    raise RuntimeError(f"Unsupported: {SYSTEM}")

def _http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=15) as r:
        return r.read().decode().strip()

def _download(url: str, dest: Path):
    print("  Downloading Chromium (~150 MB)...", end="", flush=True)
    def _cb(n, bs, t):
        if t > 0 and n * bs * 100 // t % 20 == 0:
            print(f"{n*bs*100//t}%", end="", flush=True)
    urllib.request.urlretrieve(url, str(dest), reporthook=_cb)
    print(" done.")

def _ensure_chromium() -> Path:
    if CHROMIUM_BINARY.is_file():
        age_days = (time.time() - CHROMIUM_BINARY.stat().st_mtime) / 86400
        if age_days < 30:
            return CHROMIUM_BINARY
        print("  Chromium >30 days old, re-downloading...")

    CHROMIUM_DIR.mkdir(parents=True, exist_ok=True)
    url, archive = _chromium_url()
    zip_path = CHROMIUM_DIR / archive
    _download(url, zip_path)
    print("  Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CHROMIUM_DIR)
    zip_path.unlink()

    # --- macOS ----------------------------------------------------------
    if SYSTEM == "Darwin":
        src = CHROMIUM_DIR / "chrome-mac" / "Chromium.app"
        dst = CHROMIUM_DIR / "Chromium.app"
        if src.is_dir():
            if dst.is_dir():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            shutil.rmtree(CHROMIUM_DIR / "chrome-mac", ignore_errors=True)
        subprocess.run(["xattr", "-cr", str(dst)], check=False, capture_output=True)
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(dst)],
                       check=False, capture_output=True)
        CHROMIUM_BINARY.chmod(CHROMIUM_BINARY.stat().st_mode | stat.S_IEXEC)

    # --- Linux ----------------------------------------------------------
    elif SYSTEM == "Linux":
        src = CHROMIUM_DIR / "chrome-linux"
        if src.is_dir():
            for f in src.iterdir():
                d = CHROMIUM_DIR / f.name
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                shutil.move(str(f), str(d))
            shutil.rmtree(src, ignore_errors=True)
        CHROMIUM_BINARY.chmod(0o755)

    # --- Windows --------------------------------------------------------
    elif SYSTEM == "Windows":
        src = CHROMIUM_DIR / "chrome-win"
        if src.is_dir():
            for f in src.iterdir():
                d = CHROMIUM_DIR / f.name
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                shutil.move(str(f), str(d))
            shutil.rmtree(src, ignore_errors=True)

    print(f"  Chromium ready: {CHROMIUM_BINARY}")
    return CHROMIUM_BINARY

# ============================================================
#  Native messaging manifest  (uses `args` for direct node launch)
# ============================================================

def _install_manifests():
    """Install native messaging manifests (idempotent).

    Uses ``native-host-launcher.sh`` as the single ``path`` (no ``args``).
    This is the most robust format — Chromium launches the launcher
    directly, and the launcher finds Node.js on its own.
    """
    origin = f"chrome-extension://{EXTENSION_ID}/"
    launcher = str((HOST_DIR / "native-host-launcher.sh").resolve())

    manifest = {
        "name": "com.openagent.agent_in_chrome",
        "description": "OpenAgent in Chrome Native Messaging Host",
        "path": launcher,
        "type": "stdio",
        "allowed_origins": [origin],
    }
    content = json.dumps(manifest, indent=2) + "\n"

    if SYSTEM == "Darwin":
        base = Path.home() / "Library" / "Application Support"
        dirs = [
            base / "Google" / "Chrome" / "NativeMessagingHosts",
            base / "Chromium" / "NativeMessagingHosts",
            base / "org.chromium.Chromium" / "NativeMessagingHosts",
            base / "Microsoft Edge" / "NativeMessagingHosts",
            base / "BraveSoftware" / "Brave-Browser" / "NativeMessagingHosts",
        ]
    elif SYSTEM == "Linux":
        base = Path.home() / ".config"
        dirs = [
            base / d / "NativeMessagingHosts"
            for d in ("google-chrome", "chromium", "microsoft-edge")
        ]
    elif SYSTEM == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", ""))
        dirs = [
            base / d / "NativeMessagingHosts"
            for d in (r"Google\Chrome\User Data", r"Microsoft\Edge\User Data")
        ]
    else:
        return

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        mf = d / "com.openagent.agent_in_chrome.json"
        mf.write_text(content)
        # macOS adds com.apple.provenance to new files, which blocks
        # Chromium from reading the manifest.  Strip it.
        if SYSTEM == "Darwin":
            subprocess.run(["xattr", "-c", str(mf)], check=False, capture_output=True)


# ============================================================
#  Profile & launch
# ============================================================

def _profile_dir() -> Path:
    if SYSTEM == "Darwin":
        return Path.home() / "Library" / "Application Support" / "OpenAgent" / "chrome-profile"
    elif SYSTEM == "Linux":
        return Path.home() / ".config" / "openagent" / "chrome-profile"
    elif SYSTEM == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "OpenAgent" / "chrome-profile"
    return Path.home() / ".openagent" / "chrome-profile"


def _is_running(profile: str) -> bool:
    if PIDFILE.exists():
        try:
            os.kill(int(PIDFILE.read_text().strip()), 0)
            return True
        except (ValueError, OSError):
            PIDFILE.unlink(missing_ok=True)
    return False


def launch() -> bool:
    _install_manifests()

    if not CHROMIUM_BINARY.is_file():
        print("Chromium not found — downloading...")
        try:
            _ensure_chromium()
        except Exception as e:
            print(f"  Failed: {e}", file=sys.stderr)
            print("  Install Chromium manually:  brew install chromium", file=sys.stderr)
            return False

    profile = str(_profile_dir())
    ext_path = str(EXTENSION_DIR.resolve())

    if _is_running(profile):
        print(f"Agent browser already running.")
        return True

    args = [
        str(CHROMIUM_BINARY),
        f"--user-data-dir={profile}",
        f"--load-extension={ext_path}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
        "--disable-background-networking",
        "--new-window", "about:blank",
    ]

    print(f"Launching Chromium with OpenAgent extension...")
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)

    try:
        r = subprocess.run(["pgrep", "-f", f"user-data-dir={profile}"],
                           capture_output=True, text=True)
        pids = [int(p) for p in r.stdout.strip().split("\n") if p]
        if pids:
            PIDFILE.write_text(str(min(pids)))
            print("  Agent browser started.")
            return True
    except Exception:
        pass

    print("  WARNING: Could not confirm browser started.", file=sys.stderr)
    return False


def main():
    ok = launch()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
