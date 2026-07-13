"""Frozen-environment detection for PyInstaller bundles.

Provides helpers to detect whether OpenAgent is running from a frozen
executable (PyInstaller) and resolve paths accordingly.

Since v0.5.2 the server ships as a PyInstaller **onefile** build: the
bundle extracts to ``sys._MEIPASS`` on first launch and the executable
lives at ``sys.executable`` outside of that temp tree.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def is_frozen() -> bool:
    """Return True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def _stable_ssl_cert_cache_dir() -> Path:
    """Return a writable cache dir outside PyInstaller's ``_MEI`` tree."""
    override = os.environ.get("OPENAGENT_SSL_CERT_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    try:
        from src.core import paths

        return paths.data_dir() / ".runtime"
    except Exception:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        return base / "openagent"


def _cached_ssl_cert_path() -> Path:
    return _stable_ssl_cert_cache_dir() / "certifi-cacert.pem"


def _copy_ssl_cert_to_stable_cache(source: Path) -> Path | None:
    """Copy certifi's CA bundle out of ``_MEI`` and return the stable path."""
    if not source.is_file():
        cached = _cached_ssl_cert_path()
        return cached if cached.is_file() else None

    dest = _cached_ssl_cert_path()
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, tmp)
        os.replace(tmp, dest)
        return dest
    except OSError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return source if source.is_file() else None


def patch_ssl_for_frozen() -> None:
    """Point Python's SSL at a stable copy of certifi's CA bundle.

    Without this, any library that uses Python's default
    ``ssl.create_default_context()`` (aiohttp → discord.py,
    urllib3, etc.) fails inside the onefile bundle with::

        ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
        certificate verify failed: unable to get local issuer certificate

    because OpenSSL's compiled-in CA path doesn't exist in the frozen
    runtime. Setting ``SSL_CERT_FILE`` at process start — before any
    bridge or MCP opens an HTTPS connection — makes every downstream
    consumer inherit the right CA bundle.

    The bundled certifi file originally lives under PyInstaller's
    disposable ``$TMPDIR/_MEI_xxxxx`` extraction tree. Long-running
    container agents may run cleanup jobs that delete old temp dirs while
    the process is still alive; if ``SSL_CERT_FILE`` keeps pointing into
    that deleted tree, later model calls and Telegram polling fail with
    ``FileNotFoundError`` while creating SSL contexts. Copy the CA bundle
    into a stable cache outside ``_MEI`` and point SSL there instead.

    Libraries like ``litellm`` (OpenAI SDK, Anthropic SDK) also benefit:
    they use ``httpx`` too, but having ``SSL_CERT_FILE`` set is a
    belt-and-braces safety net for any future transitive dep that uses
    plain ``urllib3`` or ``aiohttp``.

    No-op when certifi isn't bundled (pip-installed dev setups use the
    system CA store, which works fine).
    """
    if not is_frozen():
        return
    configured = os.environ.get("SSL_CERT_FILE")
    if configured and Path(configured).is_file():
        return  # caller already set a live CA bundle — don't override
    try:
        import certifi

        ca = Path(certifi.where())
        stable = _copy_ssl_cert_to_stable_cache(ca)
        if stable is not None and stable.is_file():
            os.environ["SSL_CERT_FILE"] = str(stable)
    except ImportError:
        pass


_METADATA_PATCHED = False

# Packages whose dist-info can go missing inside the PyInstaller onefile
# extract tree (sibling-swap, partial copies, race with cleanup) but
# whose Python module is always already importable from the bundle. For
# these, falling back to ``module.__version__`` is safe and matches what
# ``importlib.metadata.version`` would have returned.
_METADATA_FALLBACK_PACKAGES = frozenset({
    "pydantic",
    "pydantic_core",
    "email_validator",
})


def patch_importlib_metadata_for_frozen() -> None:
    """Make ``importlib.metadata.version`` survive a missing dist-info
    for a small allowlist of packages (pydantic, pydantic_core,
    email_validator).

    Without this, every runtime ``Team`` run in the PyInstaller onefile
    bundle dies with::


        ERROR Error in Team run: No package metadata was found for pydantic

    because ``the runtime's tool-wrapping function`` calls
    ``importlib.metadata.version("pydantic")`` on every Team run and
    the bundled ``pydantic-*.dist-info`` directory under
    ``sys._MEIPASS`` is fragile — a sibling-swap during self-update
    (the same failure mode patched for zlib in commit b394176) is
    enough to invalidate it.

    ``openagent.spec`` already ships the dist-info via
    ``copy_metadata``; this is belt-and-braces defense for the cases
    where the build-time fix doesn't survive runtime.

    The wrapper only fires for the allowlist and only when the
    underlying ``PackageNotFoundError`` is raised — pip-installed dev
    setups hit the original ``version()`` path untouched.

    No-op outside a frozen bundle. Idempotent.
    """
    global _METADATA_PATCHED
    if not is_frozen():
        return
    if _METADATA_PATCHED:
        return

    import importlib
    import importlib.metadata as _metadata

    _original_version = _metadata.version

    def _version_with_fallback(distribution_name: str) -> str:
        try:
            return _original_version(distribution_name)
        except _metadata.PackageNotFoundError:
            normalized = distribution_name.lower().replace("-", "_")
            if normalized not in _METADATA_FALLBACK_PACKAGES:
                raise
            module = importlib.import_module(normalized)
            module_version = getattr(module, "__version__", None)
            if not isinstance(module_version, str):
                raise
            return module_version

    _metadata.version = _version_with_fallback
    _METADATA_PATCHED = True


def bundle_dir() -> Path:
    """Return the directory containing bundled data files.

    In onefile mode this is ``sys._MEIPASS`` (the freshly-extracted temp
    tree). In onedir mode — not what we ship any more, but kept for dev
    setups — it's the directory next to the executable.
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(sys.executable).resolve().parent


def executable_path() -> Path:
    """Return the path to the running executable on disk.

    Important distinction in onefile mode: ``sys.executable`` points at
    the on-disk binary (e.g. ``~/bin/openagent``), NOT into the
    ``sys._MEIPASS`` extract tree. That's what the self-updater needs to
    swap, so resolve here without touching ``_MEIPASS``.
    """
    return Path(sys.executable).resolve()


def swap_pending_if_any() -> bool:
    """On Windows, promote a staged ``*.pending.exe`` file into place.

    The self-updater can't overwrite a running executable on Windows, so
    ``apply_update`` drops the new binary next to the current one with a
    ``.pending.exe`` suffix. The OS service manager restarts us on exit 75;
    the very next launch calls this from the CLI entry point BEFORE
    importing the rest of the package, moves the pending file into place,
    and re-execs so the user ends up running the new code.

    Returns True if a swap was performed (caller should re-exec or exit).
    """
    if not is_frozen():
        return False
    if sys.platform != "win32":
        return False
    exe = executable_path()
    pending = exe.with_name(exe.stem + ".pending.exe")
    if not pending.exists():
        return False
    old = exe.with_suffix(exe.suffix + ".old")
    try:
        if old.exists():
            old.unlink()
        exe.rename(old)
        shutil.move(str(pending), str(exe))
        # Re-exec so the user ends up running the new binary immediately.
        os.execv(str(exe), [str(exe)] + sys.argv[1:])
    except Exception:
        # Swap failed — keep running the current binary so the user isn't
        # stuck in a broken state. The next update attempt will retry.
        return False
    return True
