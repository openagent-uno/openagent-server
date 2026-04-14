"""Self-update for frozen (PyInstaller) executables.

Downloads the latest release from GitHub, verifies the checksum, and
replaces the running executable in place. The update is applied by:

- macOS/Linux: rename current → .old, move new → current
- Windows: save as .pending.exe, swap at next startup

After replacement the caller should exit with code 75 so the OS service
manager restarts the process with the new binary.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import NamedTuple
from urllib.request import urlopen, Request

logger = logging.getLogger(__name__)

# GitHub repository for release lookups
GITHUB_REPO = "geroale/OpenAgent"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


class UpdateInfo(NamedTuple):
    current_version: str
    new_version: str
    download_url: str
    checksum_url: str | None


def _asset_suffix() -> str:
    """Return the expected archive suffix for this platform/arch."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        os_name = "macos"
    elif system == "linux":
        os_name = "linux"
    elif system == "windows":
        os_name = "windows"
    else:
        os_name = system

    if machine in ("x86_64", "amd64"):
        arch = "x64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    else:
        arch = machine

    ext = "zip" if os_name == "windows" else "tar.gz"
    return f"{os_name}-{arch}.{ext}"


def check_for_update() -> UpdateInfo | None:
    """Query GitHub Releases for a newer version.

    Returns UpdateInfo if a newer version is available, else None.
    """
    import json
    import openagent

    current = getattr(openagent, "__version__", "0.0.0")

    try:
        req = Request(GITHUB_API, headers={"Accept": "application/vnd.github+json"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.error("Failed to check for updates: %s", e)
        return None

    tag = data.get("tag_name", "")
    new_version = tag.lstrip("v")

    # Compare versions
    from packaging.version import Version
    try:
        if Version(new_version) <= Version(current):
            return None
    except Exception:
        # If version parsing fails, skip update
        return None

    # Find matching asset
    suffix = _asset_suffix()
    download_url = None
    checksum_url = None

    for asset in data.get("assets", []):
        name = asset.get("name", "")
        url = asset.get("browser_download_url", "")
        if name.endswith(suffix):
            download_url = url
        elif name.endswith(f"{suffix}.sha256"):
            checksum_url = url

    if not download_url:
        logger.warning("No matching release asset for %s", suffix)
        return None

    return UpdateInfo(
        current_version=current,
        new_version=new_version,
        download_url=download_url,
        checksum_url=checksum_url,
    )


def download_update(url: str, checksum_url: str | None = None) -> Path:
    """Download and verify the update archive. Returns the path to the new
    executable file (onefile format — a single binary, not a directory).

    Since v0.5.2 the release archives contain ONE executable each:
        openagent-<ver>-<platform>-<arch>.tar.gz → openagent (or .exe)
    We pick that one file out of the archive and return its path.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="openagent_update_"))
    archive_path = tmp_dir / "update_archive"

    logger.info("Downloading update from %s", url)
    req = Request(url)
    with urlopen(req, timeout=120) as resp:
        archive_path.write_bytes(resp.read())

    # Verify checksum
    if checksum_url:
        try:
            with urlopen(Request(checksum_url), timeout=15) as resp:
                expected = resp.read().decode().strip().split()[0]
            actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"Checksum mismatch: expected {expected}, got {actual}"
                )
            logger.info("Checksum verified OK")
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("Could not verify checksum: %s", e)

    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir()

    if str(archive_path).endswith(".zip") or url.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)

    # Find the new binary. onefile archives contain a single executable
    # (``openagent`` on macOS/Linux, ``openagent.exe`` on Windows) — larger
    # than ~10 MB, never inside a nested directory. If an older onedir
    # archive is encountered (pre-v0.5.2), fall back to the bundled binary.
    candidates = sorted(
        (p for p in extract_dir.rglob("openagent*")
         if p.is_file() and not p.name.endswith(".sha256")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise RuntimeError("Could not locate executable in downloaded archive")


def apply_update(new_exe: Path) -> None:
    """Replace the running executable with the new onefile binary.

    - macOS/Linux: rename current → .old, move new into place, chmod +x
    - Windows: save as ``<name>.pending.exe`` next to the current binary;
      the startup hook (see ``_frozen.swap_pending_if_any``) promotes it on
      the next launch since a running .exe can't be overwritten on Windows.
    """
    from openagent._frozen import executable_path

    current_exe = executable_path()
    system = platform.system()

    if system == "Windows":
        pending = current_exe.with_name(current_exe.stem + ".pending.exe")
        if pending.exists():
            pending.unlink()
        shutil.copy2(str(new_exe), str(pending))
        logger.info("Update staged at %s (will apply on next restart)", pending)
    else:
        # Rename the running binary to .old — the OS keeps the file open for
        # the live process, and the new file is installed in its place so
        # the next launch picks up the upgrade.
        old = current_exe.with_suffix(current_exe.suffix + ".old")
        if old.exists():
            old.unlink()
        current_exe.rename(old)
        shutil.copy2(str(new_exe), str(current_exe))
        current_exe.chmod(0o755)
        logger.info("Update applied. Old version at %s", old)


def perform_self_update_sync() -> tuple[str, str]:
    """Synchronous self-update: check → download → apply.

    Returns (old_version, new_version). If already up-to-date,
    old == new.
    """
    info = check_for_update()
    if info is None:
        import openagent
        v = getattr(openagent, "__version__", "unknown")
        return v, v

    logger.info(
        "Update available: %s → %s", info.current_version, info.new_version
    )

    new_exe = download_update(info.download_url, info.checksum_url)
    apply_update(new_exe)

    return info.current_version, info.new_version
