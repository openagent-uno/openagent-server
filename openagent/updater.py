"""Self-update for frozen (PyInstaller) executables.

Downloads the latest release from GitHub, verifies the checksum, and
replaces the running executable in place. The update is applied by:

- macOS/Linux: rename current → .old, move new → current
- Windows: save as .pending.exe, swap at next startup

After replacement the caller should exit with code 75 so the OS service
manager restarts the process with the new binary.
"""

from __future__ import annotations

import contextlib
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

# GitHub repository for release lookups. The previous owner ``geroale``
# still resolves via GitHub's rename redirect, but pinning to the
# canonical owner removes a silent single point of failure: if the
# redirect is ever revoked (rename loop, namespace conflict) every
# deployed agent would silently stop receiving updates.
GITHUB_REPO = "openagent-uno/OpenAgent"
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


def _try_elog(event: str, level: str = "info", **data) -> None:
    """Best-effort wrapper around :func:`elog` so importing the events
    sink can never block a self-update flow that runs before logging is
    fully wired (e.g. during tests, or pre-``setup_logging`` startup).
    """
    try:
        from openagent.core.logging import elog
        elog(event, level=level, **data)
    except Exception:
        pass


def check_for_update() -> UpdateInfo | None:
    """Query GitHub Releases for a newer version.

    Returns UpdateInfo if a newer version is available, else None.

    Every "no update" path emits a structured event so an operator
    watching ``events.jsonl`` can tell "really up-to-date" apart from
    "GitHub unreachable", "tag malformed", or "no asset for this
    platform". Without those events all four cases looked identical.
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
        _try_elog("update.check_failed", level="warning", error=str(e) or type(e).__name__)
        return None

    # Defence-in-depth: GitHub's ``/releases/latest`` filters prereleases
    # server-side, but a future migration to ``/releases`` (or someone
    # marking a release as ``latest=true`` manually) would otherwise
    # auto-deploy an RC build to every production agent on the next 6 h
    # check.
    if data.get("prerelease"):
        tag = data.get("tag_name", "")
        logger.info("Skipping prerelease %s", tag)
        _try_elog("update.skipped_prerelease", tag=tag)
        return None

    tag = data.get("tag_name", "")
    new_version = tag.lstrip("v")

    # Compare versions
    from packaging.version import Version, InvalidVersion
    try:
        if Version(new_version) <= Version(current):
            return None
    except InvalidVersion as e:
        # Without this log path the agent silently stayed on the old
        # version forever: the caller logs ``update.check updated=false``
        # which is indistinguishable from a healthy no-op.
        logger.warning("Could not parse release tag %r: %s", tag, e)
        _try_elog(
            "update.tag_parse_failed",
            level="warning",
            tag=tag,
            error=str(e) or type(e).__name__,
        )
        return None

    # Find matching server asset. Releases also ship desktop and CLI artifacts,
    # so matching on platform suffix alone is not enough.
    download_url, checksum_url = _select_release_assets(
        list(data.get("assets", [])),
        version=new_version,
    )

    if not download_url:
        expected = _expected_asset_name(new_version)
        logger.warning("No matching release asset for %s", expected)
        _try_elog(
            "update.no_asset",
            level="warning",
            tag=tag,
            expected=expected,
        )
        return None

    return UpdateInfo(
        current_version=current,
        new_version=new_version,
        download_url=download_url,
        checksum_url=checksum_url,
    )


_DOWNLOAD_CHUNK_SIZE = 64 * 1024


def download_update(url: str, checksum_url: str | None = None) -> Path:
    """Download and verify the update archive. Returns the path to the new
    executable file (onefile format — a single binary, not a directory).

    Since v0.5.2 the release archives contain ONE executable each:
        openagent-<ver>-<platform>-<arch>.tar.gz → openagent (or .exe)
    We pick that one file out of the archive and return its path.

    The body is streamed to disk and hashed incrementally. ``resp.read()``
    used to load the entire archive (200 MB+) into memory before writing
    a single byte — on the performa-agent VPS (7.8 GiB RAM, 3 OpenAgent
    services, no swap) the kernel OOM-killed openagent + systemd itself
    mid-download. Streaming caps peak memory at one chunk.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="openagent_update_"))
    archive_path = tmp_dir / "update_archive"

    ctx = _ssl_context()

    # Fetch the expected checksum FIRST, so we can verify as we stream
    # rather than reading the file back from disk for a second pass.
    expected: str | None = None
    if checksum_url:
        try:
            with urlopen(Request(checksum_url), timeout=15, context=ctx) as resp:
                expected = resp.read().decode().strip().split()[0]
        except Exception as e:
            logger.warning("Could not fetch checksum: %s", e)

    logger.info("Downloading update from %s", url)
    # Generous timeout because release assets are large and residential
    # networks/VPNs occasionally cap throughput well below GitHub Releases'
    # CDN speed, stretching a 100 MB archive past 2 minutes.
    h = hashlib.sha256()
    bytes_read = 0
    with urlopen(Request(url), timeout=600, context=ctx) as resp, \
         open(archive_path, "wb") as f:
        while True:
            chunk = resp.read(_DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            bytes_read += len(chunk)
    logger.info("Downloaded %d bytes", bytes_read)

    if expected is not None:
        actual = h.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch: expected {expected}, got {actual}"
            )
        logger.info("Checksum verified OK")

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
        # tarfile.open has been seen raising ``zlib.error: Error -3 ...
        # incorrect header check`` on a downloaded archive whose sha256
        # matches the published checksum and which extracts cleanly with
        # system tar. Capture archive size + magic bytes + the full
        # traceback so the next reproduction has the data to root-cause
        # whether the failure is in the bundled zlib, a truncated read,
        # or something else.
        try:
            with tarfile.open(archive_path) as tf:
                tf.extractall(extract_dir)
        except Exception:
            try:
                stat = archive_path.stat()
                with open(archive_path, "rb") as _f:
                    head = _f.read(32)
                logger.exception(
                    "tarfile.open failed for %s (size=%d, head_hex=%s)",
                    archive_path, stat.st_size, head.hex(),
                )
            except Exception:
                logger.exception("tarfile.open failed for %s (stat unavailable)", archive_path)
            raise

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


def _swap_lock_path(target: Path) -> Path:
    """Path to the cross-process lockfile next to the binary or bundle.

    Multi-tenant boxes (e.g. ``openagent.service`` + ``yoanna-agent.service``
    + ``friday-agent.service`` sharing ``/home/ubuntu/.local/bin/openagent-stable``)
    can otherwise race in apply_update: A renames current→.old, then B's
    rename(current→.old) hits FileNotFoundError because A's already moved
    it. The default 4 AM auto-update cron makes that race very likely.
    """
    return target.with_name(target.name + ".swap-lock")


@contextlib.contextmanager
def _swap_lock(target: Path):
    """Hold an exclusive flock on the swap-lock file for the duration.

    Best-effort: on platforms without ``fcntl`` (Windows) we just no-op,
    since the Windows path uses a side-by-side ``.pending.exe`` and
    doesn't need the lock anyway.
    """
    lock_path = _swap_lock_path(target)
    try:
        import fcntl
    except ImportError:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fd.close()


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

    The rename + copy pair is wrapped in a try/except: if the copy fails
    (disk full, permission error) the previous binary is renamed back so
    a subsequent launch can still find it. Without this rollback a
    half-completed swap left the .app bundle missing entirely and any
    later restart would fail to exec.
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
            with _swap_lock(current_bundle):
                if old.exists():
                    shutil.rmtree(str(old))
                current_bundle.rename(old)
                try:
                    shutil.copytree(str(new_bundle), str(current_bundle))
                except Exception:
                    # Roll back so launchd can still find the bundle.
                    if current_bundle.exists():
                        shutil.rmtree(str(current_bundle), ignore_errors=True)
                    old.rename(current_bundle)
                    raise
            logger.info("Update applied (bundle swap). Old version at %s", old)
            return

    # Bare binary (Linux, or macOS without .app bundle).
    # Rename the running binary to .old — the OS keeps the file open for
    # the live process, and the new file is installed in its place so
    # the next launch picks up the upgrade.
    old = current_exe.with_suffix(current_exe.suffix + ".old")
    with _swap_lock(current_exe):
        if old.exists():
            old.unlink()
        current_exe.rename(old)
        try:
            shutil.copy2(str(new_exe), str(current_exe))
            current_exe.chmod(0o755)
        except Exception:
            if current_exe.exists():
                current_exe.unlink()
            old.rename(current_exe)
            raise
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
