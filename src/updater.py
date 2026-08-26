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
import os
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

# GitHub repository for release lookups. Releases are cut from
# ``openagent-uno/openagent-server`` since the 2026-04 monorepo split;
# the historical ``geroale/OpenAgent`` host still resolves via GitHub's
# rename redirect but we never depend on that — a silent revocation
# (rename loop, namespace conflict) would otherwise stop every deployed
# agent from receiving updates with no surfaced error.
GITHUB_REPO = "openagent-uno/openagent-server"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases?per_page=30"


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
    # SHA-256 the GitHub API reports for the asset itself, on the SAME
    # authenticated response as the version/tag (``"sha256:<hex>"``).
    # Preferred over the separate ``.sha256`` asset because it can't go
    # missing or fail its own second request — see ``download_update``.
    expected_digest: str | None = None


def _expected_asset_name(version: str) -> str:
    """Return the exact server asset filename for this platform."""
    return f"openagent-{version}-{_asset_suffix()}"


def _select_release_assets(
    assets: list[dict[str, object]],
    *,
    version: str,
) -> tuple[str | None, str | None, str | None]:
    """Pick the server archive + checksum from a GitHub release asset list.

    Returns ``(download_url, checksum_url, expected_digest)`` where
    ``expected_digest`` is the chosen asset's own ``digest`` field
    (``"sha256:<hex>"``) when GitHub reports it — the strongest integrity
    anchor available because it rides the same authenticated API response
    as the version, with no fragile second request.

    Prefer an exact match like ``openagent-0.5.17-linux-x64.tar.gz`` so we
    never confuse the server binary with sibling artifacts such as
    ``openagent-cli-*`` or ``openagent-app-*`` that happen to share the same
    platform suffix.
    """
    exact_name = _expected_asset_name(version)
    checksum_name = f"{exact_name}.sha256"

    download_url = None
    checksum_url = None
    digest = None

    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if name == exact_name:
            download_url = url
            digest = str(asset.get("digest") or "") or None
        elif name == checksum_name:
            checksum_url = url

    if download_url:
        return download_url, checksum_url, digest
    # Release metadata and downloaded bytes must agree on one exact version.
    # A same-platform fallback could silently install an older/newer sibling
    # archive when a release was assembled incompletely, so fail closed.
    return None, None, None


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
        from src.core.logging import elog
        elog(event, level=level, **data)
    except Exception:
        pass


def _effective_update_channel(current: str, configured: str | None) -> str:
    """Return stable/beta without ever opting a stable build into prereleases."""

    from packaging.version import InvalidVersion, Version

    explicit = (configured or os.environ.get("OPENAGENT_UPDATE_CHANNEL", "")).strip().lower()
    if explicit:
        if explicit not in {"stable", "beta"}:
            logger.warning("Unknown update channel %r; defaulting to stable", explicit)
            return "stable"
        return explicit
    try:
        parsed = Version(current)
    except InvalidVersion:
        return "stable"
    raw_pre = "".join(str(part).lower() for part in (parsed.pre or ()))
    return "beta" if parsed.is_prerelease and ("b" in raw_pre or "beta" in current.lower()) else "stable"


def _select_channel_release(payload: object, *, current: str, channel: str) -> dict | None:
    """Select the newest allowed release from a synthetic/GitHub feed."""

    from packaging.version import InvalidVersion, Version

    releases = payload if isinstance(payload, list) else [payload]
    try:
        current_version = Version(current)
    except InvalidVersion:
        return None
    candidates: list[tuple[Version, dict]] = []
    for raw in releases:
        if not isinstance(raw, dict) or raw.get("draft"):
            continue
        tag = str(raw.get("tag_name") or "")
        try:
            parsed = Version(tag.lstrip("v"))
        except InvalidVersion:
            continue
        if parsed <= current_version:
            continue
        prerelease = bool(raw.get("prerelease")) or parsed.is_prerelease
        if prerelease:
            if channel != "beta":
                continue
            # Beta means beta — never silently consume alpha/rc/nightly. Keep
            # prereleases on the current major/minor compatibility line; a
            # semantically newer stable is allowed independently below.
            pre_name = str((parsed.pre or ("",))[0]).lower()
            if pre_name not in {"b", "beta"} or parsed.release[:2] != current_version.release[:2]:
                continue
        candidates.append((parsed, raw))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def check_for_update(channel: str | None = None) -> UpdateInfo | None:
    """Query GitHub Releases for a newer version.

    Returns UpdateInfo if a newer version is available, else None.

    Every "no update" path emits a structured event so an operator
    watching ``events.jsonl`` can tell "really up-to-date" apart from
    "GitHub unreachable", "tag malformed", or "no asset for this
    platform". Without those events all four cases looked identical.
    """
    import json
    import src

    current = getattr(src, "__version__", "0.0.0")

    effective_channel = _effective_update_channel(current, channel)
    api_url = GITHUB_RELEASES_API if effective_channel == "beta" else GITHUB_API
    try:
        req = Request(api_url, headers={"Accept": "application/vnd.github+json"})
        with urlopen(req, timeout=15, context=_ssl_context()) as resp:
            payload = json.loads(resp.read())
    except Exception as e:
        logger.error("Failed to check for updates: %s", e)
        _try_elog("update.check_failed", level="warning", error=str(e) or type(e).__name__)
        return None

    data = _select_channel_release(payload, current=current, channel=effective_channel)
    if data is None:
        if effective_channel == "stable" and isinstance(payload, dict) and payload.get("prerelease"):
            tag = payload.get("tag_name", "")
            logger.info("Skipping prerelease %s", tag)
            _try_elog("update.skipped_prerelease", tag=tag)
        elif isinstance(payload, dict) and payload.get("tag_name"):
            from packaging.version import InvalidVersion, Version

            tag = str(payload.get("tag_name") or "")
            try:
                Version(tag.lstrip("v"))
            except InvalidVersion as exc:
                logger.warning("Could not parse release tag %r: %s", tag, exc)
                _try_elog(
                    "update.tag_parse_failed", level="warning", tag=tag,
                    error=str(exc) or type(exc).__name__,
                )
        return None

    tag = data.get("tag_name", "")
    new_version = tag.lstrip("v")

    # Never re-install a version we already proved is broken and rolled
    # back from. Without this, a bad release that stays GitHub-``latest``
    # would re-download → re-apply → re-rollback on every 6 h / nightly
    # poll, a perpetual brick-and-recover loop.
    try:
        from src.update_guard import rolled_back_versions
        bad = rolled_back_versions()
    except Exception:  # noqa: BLE001
        bad = set()
    if new_version in bad:
        logger.warning("Skipping previously rolled-back version %s", new_version)
        _try_elog("update.skipped_rolled_back", level="warning", tag=tag)
        return None

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
    download_url, checksum_url, expected_digest = _select_release_assets(
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
        expected_digest=expected_digest,
    )


_DOWNLOAD_CHUNK_SIZE = 64 * 1024

# When True (the default for self-update), a verified SHA-256 is
# MANDATORY: if neither the GitHub asset digest nor a fetchable
# ``.sha256`` is available, the download is refused rather than
# installed unverified. On a box the operator can't reach, an
# unverified 150 MB binary the supervisor will immediately exec is
# exactly the brick we must never risk. Tests flip this off to exercise
# the legacy no-checksum extraction paths.
REQUIRE_CHECKSUM = True


def _normalize_sha256(value: str | None) -> str | None:
    """Return a bare lowercase 64-hex sha256 from a digest string.

    Accepts ``"sha256:<hex>"`` (GitHub API digest) or ``"<hex>  name"``
    (a ``.sha256`` file's first field). Returns None when the value
    isn't a usable sha256 so the caller can fall back / fail closed.
    """
    if not value:
        return None
    token = value.strip().split()[0]
    if token.lower().startswith("sha256:"):
        token = token[len("sha256:"):]
    token = token.strip().lower()
    if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
        return token
    return None


def download_update(
    url: str,
    checksum_url: str | None = None,
    expected_digest: str | None = None,
    *,
    dest_dir: Path | None = None,
    require_checksum: bool | None = None,
) -> Path:
    """Download and verify the update archive. Returns the path to the new
    executable file (onefile format — a single binary, not a directory).

    Since v0.5.2 the release archives contain ONE executable each:
        openagent-<ver>-<platform>-<arch>.tar.gz → openagent (or .exe)
    We pick that one file out of the archive and return its path.

    Integrity is **fail-closed** (see :data:`REQUIRE_CHECKSUM`). The
    expected SHA-256 is resolved in order of trust:

    1. ``expected_digest`` — the GitHub API's per-asset ``digest``, which
       arrives on the same authenticated response as the version and
       cannot silently go missing or fail a second request.
    2. ``checksum_url`` — the sibling ``.sha256`` asset, fetched here.

    If neither yields a usable checksum and ``require_checksum`` is set,
    the download is REFUSED. A corrupt/truncated/tampered archive must
    never reach :func:`apply_update`.

    The body is streamed to disk and hashed incrementally. ``resp.read()``
    used to load the entire archive (200 MB+) into memory before writing
    a single byte — on a small multi-tenant VPS the kernel OOM-killed
    openagent plus systemd itself mid-download. Streaming caps peak
    memory at one chunk. The total is also checked against the HTTP
    ``Content-Length`` so a mid-stream disconnect can't masquerade as a
    complete download.

    ``dest_dir`` lets the caller own the temp-dir lifecycle (and clean it
    up); when omitted a fresh ``mkdtemp`` is used.
    """
    if require_checksum is None:
        require_checksum = REQUIRE_CHECKSUM
    if dest_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="openagent_update_"))
    else:
        tmp_dir = Path(dest_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / "update_archive"

    ctx = _ssl_context()

    # Resolve the expected checksum BEFORE downloading so we can verify
    # the stream and fail closed early. Digest (same-response, can't
    # vanish) is preferred over the separate .sha256 request.
    expected = _normalize_sha256(expected_digest)
    digest_source = "api-digest" if expected else None
    if expected is None and checksum_url:
        try:
            with urlopen(Request(checksum_url), timeout=15, context=ctx) as resp:
                expected = _normalize_sha256(resp.read().decode())
                digest_source = "sha256-asset" if expected else None
        except Exception as e:
            logger.warning("Could not fetch checksum: %s", e)

    if expected is None and require_checksum:
        raise RuntimeError(
            "Refusing to apply update: no verifiable SHA-256 available "
            "(GitHub asset digest missing and .sha256 unavailable). "
            "Installing an unverified binary the service manager will "
            "immediately exec is not safe on an unattended host."
        )

    logger.info("Downloading update from %s", url)
    # Generous timeout because release assets are large and residential
    # networks/VPNs occasionally cap throughput well below GitHub Releases'
    # CDN speed, stretching a 100 MB archive past 2 minutes.
    h = hashlib.sha256()
    bytes_read = 0
    with urlopen(Request(url), timeout=600, context=ctx) as resp, \
         open(archive_path, "wb") as f:
        content_length = resp.getheader("Content-Length")
        while True:
            chunk = resp.read(_DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            bytes_read += len(chunk)
    logger.info("Downloaded %d bytes", bytes_read)

    # Truncation guard: a clean-enough socket close mid-transfer ends the
    # read loop normally with a short body. Without a checksum this is the
    # only thing standing between a half-download and an install.
    if content_length is not None:
        try:
            expected_len = int(content_length)
        except (TypeError, ValueError):
            expected_len = None
        if expected_len is not None and bytes_read != expected_len:
            raise RuntimeError(
                f"Truncated download: got {bytes_read} bytes, "
                f"Content-Length was {expected_len}"
            )

    if expected is not None:
        actual = h.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch: expected {expected}, got {actual}"
            )
        logger.info("Checksum verified OK (%s)", digest_source or "checksum")

    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

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
                # filter="data" rejects absolute paths, ``..`` traversal,
                # and unsafe member types — defence-in-depth even though
                # we control the release archives. Falls back gracefully
                # on the rare runtime without the 3.12 extraction filters.
                try:
                    tf.extractall(extract_dir, filter="data")
                except TypeError:
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

    Multi-tenant boxes that run several systemd units sharing a single
    ``openagent`` (or ``openagent-stable``) binary can otherwise race in
    apply_update: A renames current→.old, then B's rename(current→.old)
    hits FileNotFoundError because A has already moved it. The default
    4 AM auto-update cron makes that race very likely.
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
    from src._frozen import executable_path

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
            staged = parent_dir / (current_bundle.stem + ".app.new")
            with _swap_lock(current_bundle):
                # Stage the new bundle ALONGSIDE the live one first. The
                # running binary stays valid the whole time this (slow)
                # copytree runs — the supervisor would find it on relaunch.
                if staged.exists():
                    shutil.rmtree(str(staged), ignore_errors=True)
                shutil.copytree(str(new_bundle), str(staged))
                if old.exists():
                    shutil.rmtree(str(old))
                # Commit with two adjacent renames. The window where the
                # bundle path doesn't exist shrinks from the multi-second
                # copytree to two metadata ops (microseconds) — and the
                # live process is still running, so the supervisor isn't
                # relaunching during it.
                current_bundle.rename(old)
                try:
                    staged.rename(current_bundle)
                except Exception:
                    # Couldn't put the new bundle in place — restore the
                    # known-good so launchd always has something to exec.
                    old.rename(current_bundle)
                    raise
            logger.info("Update applied (bundle swap). Old version at %s", old)
            return

    # Bare binary (Linux, or macOS without .app bundle).
    #
    # Fully atomic: back up the live binary to .old (the OS keeps the
    # running process's file open via its inode), stage the new bytes to
    # a sibling temp file, fsync, then os.replace() over the target.
    # os.replace is an atomic rename within the filesystem, so there is
    # NEVER a moment where the executable path is missing — a SIGKILL or
    # power loss at any instant leaves either the old or the new binary
    # in place, never a truncated file or a hole.
    old = current_exe.with_suffix(current_exe.suffix + ".old")
    staged = current_exe.with_name(current_exe.name + ".new")
    with _swap_lock(current_exe):
        try:
            shutil.copy2(str(current_exe), str(old))  # last-known-good backup
        except Exception:
            # Non-fatal: without a backup we lose rollback, but the swap
            # itself can still proceed. Log loudly so it's visible.
            logger.warning("Could not back up current binary to %s", old)
        if staged.exists():
            staged.unlink()
        try:
            shutil.copy2(str(new_exe), str(staged))
            os.chmod(str(staged), 0o755)
            with open(staged, "rb") as _sf:
                os.fsync(_sf.fileno())
            os.replace(str(staged), str(current_exe))  # atomic
        except Exception:
            if staged.exists():
                try:
                    staged.unlink()
                except Exception:  # noqa: BLE001
                    pass
            # current_exe is untouched by a failed os.replace, so the
            # running binary on disk is still the old one — nothing to
            # restore. Re-raise so the caller reports the failure.
            raise
    logger.info("Update applied. Old version at %s", old)


_SELFCHECK_TIMEOUT_S = 60


def verify_new_binary(new_exe: Path, expected_version: str | None = None) -> None:
    """Pre-swap execution gate: prove the freshly-downloaded binary can
    actually start BEFORE we replace the live one.

    This is the first and most important line of defence on a box the
    operator can't reach: it runs in the CURRENT, known-good process, so
    a download that can't even start Python (wrong arch, missing dylib,
    corrupt Mach-O, truncated extract) is rejected *without ever touching
    the running binary* — no swap, no rollback needed, zero risk.

    Runs ``<new_exe> selfcheck`` (a cheap, side-effect-free command that
    boots the import graph and prints the version). Falls back to
    ``--version`` then ``--help`` for binaries that predate ``selfcheck``.
    A non-zero exit, a timeout, or a version mismatch raises.

    NOTE: this runs under the updater's process environment, which may
    differ from the supervisor's (launchd/systemd) env. It catches
    "can't start at all"; the post-restart boot guard
    (:mod:`src.update_guard`) catches "starts here but unhealthy under
    the supervisor". The two layers are complementary.
    """
    if expected_version:
        try:
            proc = subprocess.run(
                [
                    str(new_exe),
                    "selfcheck",
                    "--quiet",
                    "--expect",
                    expected_version,
                ],
                capture_output=True,
                timeout=_SELFCHECK_TIMEOUT_S,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Downloaded binary is not executable: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Downloaded binary version self-check timed out") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Downloaded binary version self-check could not run") from exc
        reported = proc.stdout.decode("utf-8", "replace").strip().splitlines()
        reported_version = reported[-1].strip() if reported else ""
        if proc.returncode != 0 or reported_version != expected_version:
            raise RuntimeError(
                "Downloaded binary version does not exactly match the selected release"
            )
        logger.info("Pre-swap version self-check passed")
        return

    # Probes from strongest to weakest. The FIRST one that exits cleanly
    # proves the binary can start, and we accept. We deliberately do NOT
    # hard-fail on an individual probe's non-zero exit: ``selfcheck`` can
    # fail for an ENVIRONMENTAL reason (e.g. a malformed ``openagent.yaml``
    # in the service's cwd that the CLI group callback loads) on a binary
    # that is otherwise perfectly healthy. ``--help`` is eager in Click —
    # it builds (hence imports) the whole command tree but runs no
    # callback body / config load — so it's the robust floor: if it
    # exits 0, the bundle extracted, Python started, and the import graph
    # is intact. We only reject when EVERY probe fails (i.e. the binary
    # genuinely can't start at all).
    errors: list[str] = []
    for args in (["selfcheck"], ["--version"], ["--help"]):
        try:
            proc = subprocess.run(
                [str(new_exe), *args],
                capture_output=True,
                timeout=_SELFCHECK_TIMEOUT_S,
            )
        except FileNotFoundError as e:
            # Not executable at all — no probe can succeed; fail fast.
            raise RuntimeError(f"Downloaded binary is not executable: {e}") from e
        except subprocess.TimeoutExpired:
            errors.append(f"`{' '.join(args)}` timed out (>{_SELFCHECK_TIMEOUT_S}s)")
            continue
        except Exception as e:  # noqa: BLE001
            errors.append(f"`{' '.join(args)}` could not run: {e}")
            continue

        out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
        if proc.returncode != 0:
            errors.append(f"`{' '.join(args)}` exit {proc.returncode}: {out.strip()[:200]}")
            continue

        logger.info("Pre-swap self-check passed (`%s`)", " ".join(args))
        return

    raise RuntimeError(
        "Downloaded binary failed every self-check probe — refusing to "
        "install (it cannot start). Tried: " + "; ".join(errors)
    )


def perform_self_update_sync(channel: str | None = None) -> tuple[str, str]:
    """Synchronous self-update: check → download → verify → apply.

    Returns (old_version, new_version). If already up-to-date,
    old == new.

    Owns the temp-dir lifecycle so the ~150-300 MB download+extract tree
    is always cleaned up (previously leaked on every run). Verifies the
    new binary actually runs before swapping, and records the swap in the
    update journal so the post-restart boot guard can roll back if the
    new binary turns out to be unhealthy under the supervisor.
    """
    info = check_for_update(channel=channel)
    if info is None:
        import src
        v = getattr(src, "__version__", "unknown")
        return v, v

    logger.info(
        "Update available: %s → %s", info.current_version, info.new_version
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="openagent_update_"))
    try:
        new_exe = download_update(
            info.download_url,
            info.checksum_url,
            info.expected_digest,
            dest_dir=tmp_dir,
        )
        verify_new_binary(new_exe, expected_version=info.new_version)
        apply_update(new_exe)
    finally:
        # Always reclaim the temp tree, success or failure.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Record the pending swap so the next boot's guard can confirm or
    # roll it back. Best-effort: a journal hiccup must not undo a
    # successful update.
    try:
        from src.update_guard import record_pending
        record_pending(info.new_version, prev_version=info.current_version)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record pending update: %s", e)

    return info.current_version, info.new_version
