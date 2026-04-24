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
import subprocess
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


def _ssl_context():
    """Return an SSLContext that uses certifi's CA bundle when available.

    PyInstaller-frozen binaries on macOS/Linux don't ship the OS CA bundle,
    so ``urlopen`` against github.com fails with ``CERTIFICATE_VERIFY_FAILED:
    unable to get local issuer certificate``. Fall back to the system
    context when certifi isn't bundled (e.g. pip installs).
    """
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class UpdateInfo(NamedTuple):
    current_version: str
    new_version: str
    download_url: str
    checksum_url: str | None


def _expected_asset_name(version: str) -> str:
    """Return the exact server asset filename for this platform."""
    return f"openagent-{version}-{_asset_suffix()}"


def _select_release_assets(
    assets: list[dict[str, object]],
    *,
    version: str,
) -> tuple[str | None, str | None]:
    """Pick the server archive + checksum from a GitHub release asset list.

    Prefer an exact match like ``openagent-0.5.17-linux-x64.tar.gz`` so we
    never confuse the server binary with sibling artifacts such as
    ``openagent-cli-*`` or ``openagent-app-*`` that happen to share the same
    platform suffix.
    """
    exact_name = _expected_asset_name(version)
    checksum_name = f"{exact_name}.sha256"

    download_url = None
    checksum_url = None

    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if name == exact_name:
            download_url = url
        elif name == checksum_name:
            checksum_url = url

    if download_url:
        return download_url, checksum_url

    # Backward-compatible fallback for older release layouts: keep the server
    # prefix explicit so ``openagent-cli`` / ``openagent-app`` are ignored.
    suffix = _asset_suffix()
    server_prefix = "openagent-"
    excluded_prefixes = ("openagent-cli-", "openagent-app-")
    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if (
            name.startswith(server_prefix)
            and not name.startswith(excluded_prefixes)
            and name.endswith(suffix)
        ):
            download_url = url
        elif (
            name.startswith(server_prefix)
            and not name.startswith(excluded_prefixes)
            and name.endswith(f"{suffix}.sha256")
        ):
            checksum_url = url

    return download_url, checksum_url


def _asset_suffix() -> str:
    """Return the expected archive suffix for this platform/arch.

    - macOS → .pkg (signed + notarized + stapled; we extract the binary
      out of it with ``pkgutil --expand-full`` — no sudo needed)
    - Linux → .tar.gz
    - Windows → .zip
    """
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

    if os_name == "macos":
        ext = "pkg"
    elif os_name == "windows":
        ext = "zip"
    else:
        ext = "tar.gz"
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
        with urlopen(req, timeout=15, context=_ssl_context()) as resp:
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

    # Find matching server asset. Releases also ship desktop and CLI artifacts,
    # so matching on platform suffix alone is not enough.
    download_url, checksum_url = _select_release_assets(
        list(data.get("assets", [])),
        version=new_version,
    )

    if not download_url:
        logger.warning("No matching release asset for %s", _expected_asset_name(new_version))
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
    # Generous timeout because release assets are large and residential
    # networks/VPNs occasionally cap throughput well below GitHub Releases'
    # CDN speed, stretching a 100 MB archive past 2 minutes.
    ctx = _ssl_context()
    with urlopen(req, timeout=600, context=ctx) as resp:
        archive_path.write_bytes(resp.read())

    # Verify checksum
    if checksum_url:
        try:
            with urlopen(Request(checksum_url), timeout=15, context=ctx) as resp:
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

    lower = str(archive_path).lower() + " " + url.lower()
    if ".pkg" in lower:
        # macOS distribution. ``pkgutil --expand-full`` unpacks the xar +
        # Payload tree into a directory — no sudo needed. The binary we
        # want sits at <expanded>/<component>.pkg/Payload/<install-path>/<name>.
        subprocess.run(
            ["pkgutil", "--expand-full", str(archive_path), str(extract_dir / "pkg")],
            check=True,
        )
    elif ".zip" in lower:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)

    # Find the server binary, not any sibling artifact such as openagent-cli.
    # Releases are supposed to contain one executable per archive, but we keep
    # the selection exact as defence-in-depth.
    candidates = sorted(
        (
            p
            for p in extract_dir.rglob("openagent*")
            if p.is_file()
            and not p.name.endswith(".sha256")
            and p.suffix != ".plist"
        ),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    expected_names = {"openagent.exe"} if platform.system() == "Windows" else {"openagent"}
    exact = [p for p in candidates if p.name in expected_names]
    if exact:
        return exact[0]
    if candidates:
        found = ", ".join(sorted({p.name for p in candidates[:5]}))
        raise RuntimeError(
            "Downloaded archive did not contain the OpenAgent server executable "
            f"(found: {found})"
        )
    raise RuntimeError("Could not locate executable in downloaded archive")


def _find_app_bundle(path: Path) -> "Path | None":
    """Walk up from path to find the enclosing .app bundle directory, if any."""
    p = path.parent
    while p != p.parent:
        if p.suffix == ".app" and p.is_dir():
            return p
        p = p.parent
    return None


def apply_update(new_exe: Path) -> None:
    """Replace the running executable with the new onefile binary.

    - macOS inside .app bundle: rename the whole bundle from its parent
      directory (which the service user owns) to .app.old, then copy the
      new bundle in place. Renaming the inner binary directly fails when
      the Contents/MacOS/ directory is root-owned (e.g. after a pkg install),
      but the user-owned Applications/ parent always allows the bundle rename.
    - macOS/Linux bare binary: rename current → .old, move new into place, chmod +x
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
        return

    if system == "Darwin":
        current_bundle = _find_app_bundle(current_exe)
        if current_bundle is not None:
            new_bundle = _find_app_bundle(new_exe)
            if new_bundle is None:
                raise RuntimeError(
                    "Current executable is inside an .app bundle but the "
                    "downloaded archive did not contain an .app bundle. "
                    f"(new_exe={new_exe})"
                )
            parent_dir = current_bundle.parent
            old = parent_dir / (current_bundle.stem + ".app.old")
            if old.exists():
                shutil.rmtree(str(old))
            current_bundle.rename(old)
            shutil.copytree(str(new_bundle), str(current_bundle))
            logger.info("Update applied (bundle swap). Old version at %s", old)
            return

    # Bare binary (Linux, or macOS without .app bundle).
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
