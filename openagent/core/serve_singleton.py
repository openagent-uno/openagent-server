"""Kill leftover ``openagent serve`` processes bound to the same agent folder.

Why this module exists
----------------------
A service crash or ungraceful stop can leave an ``openagent serve <dir>``
process alive with the gateway port still bound. The next start (systemd,
launchd, or manual) then either fails with ``EADDRINUSE`` or — worse —
succeeds but with the old process still routing Telegram / Discord traffic
in the background, so the user sees stale cached replies.

The triage for this is always the same: SSH in, ``ps``, ``kill -TERM``,
restart. This module moves that into the ``serve`` startup path so the
fresh process cleans up its stale twins before binding. Linux + macOS only;
Windows falls through silently because ``ps`` won't be available.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _scan_ps() -> list[tuple[int, str]]:
    """Return ``[(pid, cmdline)]`` for every process visible to ``ps``."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("serve_singleton: ps scan failed: %s", e)
        return []

    result: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_str, cmd = line.split(maxsplit=1)
            result.append((int(pid_str), cmd))
        except ValueError:
            continue
    return result


def _matches_serve(cmd: str, agent_dir_abs: str, agent_dir_name: str) -> bool:
    """Whether ``cmd`` looks like an ``openagent serve`` of ``agent_dir``.

    The match intentionally allows for both the absolute path (most cases)
    and the bare directory name (shell that passed ``.`` or a relative
    path), because ``ps`` shows argv verbatim — we don't re-resolve it.
    """
    if "openagent" not in cmd or "serve" not in cmd:
        return False
    if agent_dir_abs in cmd:
        return True
    # Require a boundary before the name so "foo" doesn't match "foobar".
    return f" {agent_dir_name}" in cmd or cmd.endswith(f" {agent_dir_name}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def find_stale_serve_pids(agent_dir: Path, *, self_pid: int | None = None) -> list[int]:
    """PIDs (other than ``self_pid`` and the parent PID) serving ``agent_dir``.

    The parent PID is excluded because PyInstaller's onefile bootloader on
    Linux forks a Python child whose os.getpid() differs from the parent
    bootloader PID. Without this exclusion the singleton would SIGTERM its
    own parent on every startup, causing an immediate crash loop.
    """
    me = self_pid if self_pid is not None else os.getpid()
    parent = os.getppid()
    abs_dir = str(agent_dir)
    name = agent_dir.name
    return [
        pid
        for pid, cmd in _scan_ps()
        if pid != me and pid != parent and _matches_serve(cmd, abs_dir, name)
    ]


def kill_stale_serve_processes(
    agent_dir: Path,
    *,
    grace_seconds: float = 5.0,
) -> list[int]:
    """Terminate other ``openagent serve`` processes on ``agent_dir``.

    Returns the PIDs that were signalled (whether or not they actually
    died). Safe to call on platforms without ``ps``: the scan returns
    empty and we no-op.
    """
    pids = find_stale_serve_pids(agent_dir)
    if not pids:
        return []

    for pid in pids:
        logger.warning(
            "serve_singleton: killing stale openagent serve pid=%d on %s",
            pid,
            agent_dir,
        )
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as e:
            logger.warning("serve_singleton: SIGTERM pid=%d failed: %s", pid, e)

    deadline = time.monotonic() + grace_seconds
    survivors = list(pids)
    while survivors and time.monotonic() < deadline:
        time.sleep(0.25)
        survivors = [pid for pid in survivors if _pid_alive(pid)]

    for pid in survivors:
        logger.warning("serve_singleton: pid=%d ignored SIGTERM; sending SIGKILL", pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as e:
            logger.warning("serve_singleton: SIGKILL pid=%d failed: %s", pid, e)

    return pids
