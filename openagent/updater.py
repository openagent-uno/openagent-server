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
    """Download the update archive and verify its checksum.

    Returns the path to the extracted directory containing the new executable.
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

    # Extract
    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir()

    if str(archive_path).endswith(".zip") or url.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)

    # Find the openagent executable in the extracted directory
    for candidate in extract_dir.rglob("openagent*"):
        if candidate.is_file() and candidate.stat().st_size > 1_000_000:
            return candidate.parent

    # Fallback: look for the openagent/ directory
    openagent_dir = extract_dir / "openagent"
    if openagent_dir.is_dir():
        return openagent_dir

    return extract_dir


def apply_update(new_dir: Path) -> None:
    """Replace the running executable with the new version.

    - macOS/Linux: rename current to .old, copy new into place
    - Windows: save as .pending.exe (applied at next startup)
    """
    from openagent._frozen import executable_path

    current_exe = executable_path()
    current_dir = current_exe.parent
    system = platform.system()

    if system == "Windows":
        # Can't replace a running .exe on Windows.
        # Copy the new directory alongside and mark for swap at startup.
        pending_dir = current_dir.parent / (current_dir.name + ".pending")
        if pending_dir.exists():
            shutil.rmtree(pending_dir)
        shutil.copytree(str(new_dir), str(pending_dir))

        # Create a marker so startup knows to swap
        pending_exe = current_dir / (current_exe.stem + ".pending.exe")
        # Copy just the main executable as the marker
        new_exe = new_dir / current_exe.name
        if new_exe.exists():
            shutil.copy2(str(new_exe), str(pending_exe))
        logger.info("Update staged at %s (will apply on next restart)", pending_dir)
    else:
        # macOS/Linux: rename current dir to .old, move new into place
        old_dir = current_dir.parent / (current_dir.name + ".old")
        if old_dir.exists():
            shutil.rmtree(old_dir)

        # Rename current to .old (running process keeps file descriptors)
        current_dir.rename(old_dir)

        # Move new into place
        shutil.copytree(str(new_dir), str(current_dir))
        # Ensure executable permission
        new_exe = current_dir / current_exe.name
        if new_exe.exists():
            new_exe.chmod(0o755)

        logger.info("Update applied. Old version at %s", old_dir)


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

    new_dir = download_update(info.download_url, info.checksum_url)
    apply_update(new_dir)

    return info.current_version, info.new_version
