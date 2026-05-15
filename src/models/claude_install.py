"""Claude Code CLI binary auto-install, auth check, and login launch.

Mirrors the Chromium auto-install pattern (``agent-in-chrome/auto-install-extension.py``)
but for the Anthropic ``claude`` binary needed by the claude-cli framework.

The SDK's ``SubprocessCLITransport._find_cli()`` already searches:
    1. Bundled path: ``claude_agent_sdk/_bundled/claude``
    2. ``PATH`` (``shutil.which("claude")``)
    3. Known fallback locations (``~/.npm-global/bin/claude``, etc.)

This module adds a *proactive* layer that runs at OpenAgent startup / first use:
    1. Find the binary (using the same search as the SDK).
    2. If missing, auto-download via the official install script or platform
       package manager (brew / apt / etc.).
    3. Check auth status (``claude auth status``).
    4. Optionally launch the OAuth login flow (``claude auth login``).

All operations are idempotent — calling them repeatedly is safe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform as _platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_INSTALL_SCRIPT_URL = "https://claude.ai/install.sh"
_INSTALL_SCRIPT_URL_WIN = "https://claude.ai/install.ps1"

SYSTEM = _platform.system()  # Darwin | Linux | Windows
MACHINE = _platform.machine().lower()


# ── binary search (same logic as the SDK) ──────────────────────────────

def find_claude_binary() -> str | None:
    """Locate the ``claude`` binary using the same search as the SDK.

    Returns the absolute path as a string, or ``None`` if not found.
    """
    cli_name = "claude.exe" if SYSTEM == "Windows" else "claude"

    # 1. SDK bundled path
    try:
        from claude_agent_sdk._internal.transport.subprocess_cli import (  # type: ignore[import-untyped]
            SubprocessCLITransport,
        )
        # Instantiate a dummy transport to reach _find_bundled_cli.
        # We don't actually connect — just call the private helper.
        bundled = _find_bundled_path_internal()
        if bundled and Path(bundled).is_file():
            return str(bundled)
    except Exception:
        pass

    # 2. PATH
    if found := shutil.which("claude"):
        return found

    # 3. Known fallback locations
    locations = [
        Path.home() / ".npm-global/bin" / cli_name,
        Path("/usr/local/bin") / cli_name,
        Path.home() / ".local/bin" / cli_name,
        Path.home() / "node_modules/.bin" / cli_name,
        Path.home() / ".yarn/bin" / cli_name,
        Path.home() / ".claude/local" / cli_name,
    ]
    for path in locations:
        if path.exists() and path.is_file():
            return str(path)

    return None


def _find_bundled_path_internal() -> str | None:
    """Replicate the SDK's ``_find_bundled_cli()`` without importing the class."""
    cli_name = "claude.exe" if SYSTEM == "Windows" else "claude"
    # The _bundled directory lives at claude_agent_sdk/_bundled/
    try:
        from claude_agent_sdk._internal.transport import subprocess_cli as sdk_mod  # type: ignore[import-untyped]
        bundled_path = Path(sdk_mod.__file__).resolve().parent.parent.parent / "_bundled" / cli_name
        if bundled_path.exists() and bundled_path.is_file():
            return str(bundled_path)
    except Exception:
        pass
    return None


# ── auth status ────────────────────────────────────────────────────────

@dataclass
class AuthStatus:
    logged_in: bool
    email: str | None = None
    account_type: str | None = None  # "pro" | "max" | "console"
    raw: str | None = None


def check_claude_auth(binary_path: str | None = None) -> AuthStatus:
    """Run ``claude auth status`` and parse the result.

    Returns ``AuthStatus(logged_in=False)`` if the binary is missing or
    the command fails.
    """
    cli = binary_path or find_claude_binary()
    if cli is None:
        return AuthStatus(logged_in=False)

    try:
        proc = subprocess.run(
            [cli, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return AuthStatus(logged_in=False)
        # The command exits 0 when logged in and prints JSON.
        try:
            data = json.loads(proc.stdout.strip())
        except (json.JSONDecodeError, AttributeError):
            # --text output or unexpected format — treat as logged_out
            return AuthStatus(logged_in=False, raw=proc.stdout.strip())
        return AuthStatus(
            logged_in=True,
            email=data.get("email") or data.get("accountEmail"),
            account_type=data.get("subscriptionType") or data.get("accountType") or data.get("type"),
            raw=proc.stdout.strip(),
        )
    except FileNotFoundError:
        return AuthStatus(logged_in=False)
    except Exception as exc:
        logger.debug("claude auth status failed: %s", exc)
        return AuthStatus(logged_in=False)


# ── install ────────────────────────────────────────────────────────────

def _try_brew_install() -> bool:
    """Attempt to install via Homebrew (macOS only)."""
    if SYSTEM != "Darwin":
        return False
    if shutil.which("brew") is None:
        logger.debug("Homebrew not found, skipping brew install")
        return False
    try:
        subprocess.run(
            ["brew", "install", "--cask", "claude-code"],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return find_claude_binary() is not None
    except Exception as exc:
        logger.debug("brew install claude-code failed: %s", exc)
        return False


def _try_install_script() -> bool:
    """Download and run the official install script.

    The script is idempotent — it detects an existing install and skips.
    """
    import tempfile
    import urllib.request as _urllib

    if SYSTEM == "Windows":
        url = _INSTALL_SCRIPT_URL_WIN
        script_suffix = ".ps1"
        interpreter = ["powershell", "-ExecutionPolicy", "Bypass", "-File"]
    else:
        url = _INSTALL_SCRIPT_URL
        script_suffix = ".sh"
        interpreter = ["bash"]

    try:
        with tempfile.NamedTemporaryFile(
            suffix=script_suffix, mode="wb", delete=False
        ) as tmp:
            with _urllib.request.urlopen(url, timeout=30) as resp:
                tmp.write(resp.read())
            script_path = tmp.name

        # Make executable on Unix
        if SYSTEM != "Windows":
            os.chmod(script_path, 0o755)

        env = os.environ.copy()
        env.setdefault("CI", "true")  # Many installers skip prompts in CI mode.

        subprocess.run(
            interpreter + [script_path],
            env=env,
            check=True,
            capture_output=True,
            timeout=180,
        )
        os.unlink(script_path)
        return find_claude_binary() is not None
    except Exception as exc:
        logger.debug("install script failed: %s", exc)
        return False


def ensure_claude_binary() -> str | None:
    """Idempotent: find the Claude binary, auto-installing if missing.

    Returns the absolute path to the binary, or ``None`` if installation
    could not be completed.
    """
    existing = find_claude_binary()
    if existing is not None:
        logger.info("Claude Code binary found at %s", existing)
        return existing

    logger.info("Claude Code binary not found — attempting auto-install…")

    if _try_brew_install():
        if found := find_claude_binary():
            logger.info("Claude Code installed via Homebrew at %s", found)
            return found

    if _try_install_script():
        if found := find_claude_binary():
            logger.info("Claude Code installed via official script at %s", found)
            return found

    logger.warning(
        "Could not auto-install Claude Code. "
        "Install manually: curl -fsSL https://claude.ai/install.sh | bash"
    )
    return None


# ── login launch ───────────────────────────────────────────────────────

def launch_claude_login(
    email: str | None = None,
    sso: bool = False,
    console: bool = False,
    binary_path: str | None = None,
) -> subprocess.Popen | None:
    """Launch ``claude auth login`` as a subprocess (opens browser for OAuth).

    Returns the ``Popen`` handle so the caller can wait or poll.
    Returns ``None`` if the binary is not available.
    """
    cli = binary_path or find_claude_binary()
    if cli is None:
        logger.error("Cannot launch claude auth login — binary not found")
        return None

    cmd: list[str] = [cli, "auth", "login"]
    if email:
        cmd.extend(["--email", email])
    if sso:
        cmd.append("--sso")
    if console:
        cmd.append("--console")

    logger.info("Launching claude auth login: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


# ── convenience ─────────────────────────────────────────────────────────

@dataclass
class ClaudeSetupResult:
    binary_path: str | None = None
    binary_ok: bool = False
    auth_ok: bool = False
    auth_email: str | None = None
    auth_type: str | None = None
    error: str | None = None


def full_status(binary_path: str | None = None) -> ClaudeSetupResult:
    """One-shot check: find/install binary, then check auth.

    Does NOT auto-install — that happens lazily on first use.
    Use ``ensure_claude_binary()`` for the auto-install path.
    """
    path = binary_path or find_claude_binary()
    if path is None:
        return ClaudeSetupResult(binary_ok=False, error="Claude Code CLI not found")

    auth = check_claude_auth(binary_path=path)
    return ClaudeSetupResult(
        binary_path=path,
        binary_ok=True,
        auth_ok=auth.logged_in,
        auth_email=auth.email,
        auth_type=auth.account_type,
    )
