"""Pure-Python handlers for the six shell tools.

Provider-agnostic: Claude SDK adapter and Agno adapter both wrap these
functions with their own decorators. All state lives on the ShellHub
singleton plus per-BackgroundShell buffers. No subprocess shenanigans
here beyond asyncio.create_subprocess_exec (in BackgroundShell).
"""
from __future__ import annotations

import asyncio
import logging
import re as _re
import secrets
import shutil
import time
from typing import Any

from openagent.mcp.servers.shell.events import ShellEvent
from openagent.mcp.servers.shell.hub import ShellHub
from openagent.mcp.servers.shell.shells import BackgroundShell

logger = logging.getLogger(__name__)

# Defaults — match the spec § Tool surface and the v0.6 TS MCP.
DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 1_800_000  # 30 min


_hub_singleton: ShellHub | None = None


def get_hub() -> ShellHub:
    """Return the process-wide ShellHub singleton, creating on demand."""
    global _hub_singleton
    if _hub_singleton is None:
        _hub_singleton = ShellHub()
    return _hub_singleton


def _reset_hub_for_tests() -> None:
    """Test-only helper: replace the singleton with a fresh instance."""
    global _hub_singleton
    _hub_singleton = ShellHub()


def _new_shell_id() -> str:
    return f"sh_{secrets.token_hex(3)}"


def _clamp_timeout(ms: int | None) -> float:
    if ms is None:
        ms = DEFAULT_TIMEOUT_MS
    ms = max(1, min(ms, MAX_TIMEOUT_MS))
    return ms / 1000.0


# ── shell_exec ──────────────────────────────────────────────────────

async def shell_exec(
    command: str,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    run_in_background: bool = False,
    stdin: str | None = None,
    description: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Foreground or background shell command.

    Returns a dict. Foreground: exit_code / stdout / stderr /
    duration_ms / timed_out / signal / truncated_stdout /
    truncated_stderr. Background: shell_id / started_at.
    """
    if not command or not command.strip():
        raise ValueError("command must be a non-empty string")
    timeout_s = _clamp_timeout(timeout)
    shell_id = _new_shell_id()
    bg = BackgroundShell(
        shell_id=shell_id,
        command=command,
        cwd=cwd,
        env=env,
    )
    hub = get_hub()

    if run_in_background:
        await bg.start()
        rec = hub.register(
            shell_id=shell_id,
            session_id=session_id,
            command=command,
            shell=bg,
        )
        if stdin:
            await bg.write_stdin(stdin, press_enter=False)
        # Schedule a watcher task to detect completion and post event.
        asyncio.create_task(_watch_background(bg, session_id))
        return {
            "shell_id": shell_id,
            "started_at": rec.created_at,
            "description": description,
        }

    # Foreground path — no hub registration.
    result = await bg.run_with_timeout(
        timeout_seconds=timeout_s, stdin_data=stdin
    )
    return {
        "exit_code": result.exit_code,
        "signal": result.signal,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": result.duration_ms,
        "timed_out": result.timed_out,
        "truncated_stdout": result.stdout_dropped > 0,
        "truncated_stderr": result.stderr_dropped > 0,
    }


async def _watch_background(bg: BackgroundShell, session_id: str | None) -> None:
    """Wait for ``bg`` to exit and post a terminal event to the hub."""
    try:
        assert bg._proc is not None
        await bg._proc.wait()
    except Exception as e:  # noqa: BLE001
        logger.debug("_watch_background %s failed: %s", bg.shell_id, e)
        return
    await bg.finalise()
    hub = get_hub()
    hub.mark_completed(
        bg.shell_id, exit_code=bg.exit_code, signal=bg.signal,
    )
    kind = "completed" if bg.exit_code is not None else "killed"
    event = ShellEvent(
        shell_id=bg.shell_id,
        kind=kind,
        exit_code=bg.exit_code,
        signal=bg.signal,
        bytes_stdout=bg.stdout_bytes_total,
        bytes_stderr=bg.stderr_bytes_total,
        at=time.time(),
    )
    hub.post_event(session_id, event)


# ── shell_output ────────────────────────────────────────────────────

async def shell_output(
    shell_id: str,
    *,
    filter: str | None = None,
    since_last: bool = True,
) -> dict[str, Any]:
    hub = get_hub()
    rec = hub.get(shell_id)
    if rec is None:
        raise ValueError(f"unknown shell_id: {shell_id}")
    bg = rec.shell
    if bg is None:
        # Race: registered without a shell (tests). Fall through as empty.
        return {
            "stdout_delta": "",
            "stderr_delta": "",
            "still_running": False,
            "exit_code": rec.exit_code,
            "signal": rec.signal,
            "stdout_bytes_total": 0,
            "stderr_bytes_total": 0,
            "truncated_stdout": False,
            "truncated_stderr": False,
        }
    since_stdout = rec.last_read_stdout if since_last else 0
    since_stderr = rec.last_read_stderr if since_last else 0
    stdout_delta, stderr_delta = bg.read(
        since_stdout=since_stdout, since_stderr=since_stderr,
    )
    if filter:
        pattern = _re.compile(filter)
        stdout_delta = "\n".join(
            l for l in stdout_delta.splitlines() if pattern.search(l)
        )
        stderr_delta = "\n".join(
            l for l in stderr_delta.splitlines() if pattern.search(l)
        )
    if since_last:
        rec.last_read_stdout = bg.stdout_bytes_total
        rec.last_read_stderr = bg.stderr_bytes_total
    return {
        "stdout_delta": stdout_delta,
        "stderr_delta": stderr_delta,
        "still_running": bg.is_running,
        "exit_code": bg.exit_code,
        "signal": bg.signal,
        "stdout_bytes_total": bg.stdout_bytes_total,
        "stderr_bytes_total": bg.stderr_bytes_total,
        "truncated_stdout": bg.stdout_dropped > 0,
        "truncated_stderr": bg.stderr_dropped > 0,
    }


# ── shell_which ─────────────────────────────────────────────────────

async def shell_which(command: str) -> dict[str, Any]:
    if not command or "/" in command or "\\" in command:
        # shutil.which handles "/" / "\\" differently across platforms;
        # reject anything that looks like a path so the model gets an
        # unambiguous error.
        raise ValueError("command must be a bare program name (no path separator)")
    path = shutil.which(command)
    if path is None:
        return {"available": False}
    return {"available": True, "path": path}


# ── shell_input ─────────────────────────────────────────────────────

async def shell_input(
    shell_id: str,
    *,
    text: str,
    press_enter: bool = True,
) -> dict[str, Any]:
    rec = get_hub().get(shell_id)
    if rec is None:
        raise ValueError(f"unknown shell_id: {shell_id}")
    if rec.shell is None:
        raise RuntimeError(f"shell {shell_id} has no spawned process")
    n = await rec.shell.write_stdin(text, press_enter=press_enter)
    return {"bytes_written": n}


# ── shell_kill ──────────────────────────────────────────────────────

async def shell_kill(
    shell_id: str,
    *,
    signal: str = "TERM",
) -> dict[str, Any]:
    rec = get_hub().get(shell_id)
    if rec is None:
        raise ValueError(f"unknown shell_id: {shell_id}")
    if rec.shell is None:
        raise RuntimeError(f"shell {shell_id} has no spawned process")
    sig_name = signal.upper()
    if sig_name not in ("TERM", "INT", "KILL"):
        raise ValueError(f"unsupported signal: {signal}")
    await rec.shell.kill(signal_name=sig_name)  # type: ignore[arg-type]
    return {
        "killed": True,
        "exit_code": rec.shell.exit_code,
        "signal": rec.shell.signal,
    }


# ── shell_list ──────────────────────────────────────────────────────

async def shell_list(session_id: str | None = None) -> list[dict[str, Any]]:
    hub = get_hub()
    records = hub.list_for_session(session_id)
    now = time.time()
    out: list[dict[str, Any]] = []
    for rec in records:
        bg = rec.shell
        if bg is None:
            state = "completed" if rec.is_completed else "running"
            started_at = rec.created_at
            runtime_ms = int((now - started_at) * 1000)
            stdout_bytes = 0
            stderr_bytes = 0
        else:
            if bg.is_running:
                state = "running"
            elif bg.signal is not None:
                state = "killed"
            else:
                state = "completed"
            started_at = bg.started_at or rec.created_at
            completed = bg.completed_at or now
            runtime_ms = int((completed - started_at) * 1000)
            stdout_bytes = bg.stdout_bytes_total
            stderr_bytes = bg.stderr_bytes_total
        out.append({
            "shell_id": rec.shell_id,
            "command": rec.command,
            "state": state,
            "started_at": started_at,
            "runtime_ms": runtime_ms,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
            "exit_code": rec.exit_code,
            "session_id": rec.session_id,
        })
    return out
