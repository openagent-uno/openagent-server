"""One running background shell: subprocess + output buffers +
lifecycle control (timeout, kill, stdin). State lives here; the
ShellHub holds references for lookup but delegates every real operation
to this class.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal as signal_module
import time
from typing import Literal

logger = logging.getLogger(__name__)

# Per-stream output cap (spec § Buffering and truncation).
MAX_STREAM_BYTES = 1_000_000

# Grace between SIGTERM and SIGKILL during kill.
DEFAULT_KILL_GRACE = 5.0


SignalName = Literal["TERM", "INT", "KILL"]


def _pick_shell() -> tuple[str, str]:
    """Return (shell_path, '-c' flag). Same logic as the old TS MCP."""
    import platform as _platform
    sysname = _platform.system().lower()
    if sysname == "windows":
        return (os.environ.get("COMSPEC", "cmd.exe"), "/c")
    if sysname == "darwin":
        return (os.environ.get("SHELL", "/bin/zsh"), "-c")
    return (os.environ.get("SHELL", "/bin/bash"), "-c")


from dataclasses import dataclass as _dc


@_dc
class ForegroundResult:
    exit_code: int | None
    signal: str | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool
    stdout_dropped: int
    stderr_dropped: int


class BackgroundShell:
    """One spawned subprocess, tracked by its ``shell_id``.

    Buffers are simple ``bytearray`` with a 1 MB cap per stream; once
    full, oldest bytes are dropped and ``_stdout_dropped`` /
    ``_stderr_dropped`` counters track how many. Callers see
    truncation via the ``stdout_dropped`` / ``stderr_dropped``
    properties (and the ``truncated_stdout`` / ``truncated_stderr``
    flags Task 10's ``shell_output`` surfaces).

    Cursors are raw byte offsets into the *original* output stream
    (not the buffer), so ``read(since=N)`` is stable even after
    truncation — bytes past the drop boundary are simply gone and
    are skipped by ``_slice``.
    """

    def __init__(
        self,
        *,
        shell_id: str,
        command: str,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> None:
        self.shell_id = shell_id
        self.command = command
        self.cwd = cwd
        self.env = env
        self._proc: asyncio.subprocess.Process | None = None
        self._stdout_buf = bytearray()
        self._stderr_buf = bytearray()
        self._stdout_total = 0  # total bytes ever written (including dropped)
        self._stderr_total = 0
        self._stdout_dropped = 0
        self._stderr_dropped = 0
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._started_at: float | None = None
        self._completed_at: float | None = None
        self._exit_code: int | None = None
        self._signal: str | None = None

    # ── Spawn ───────────────────────────────────────────────────────

    async def start(self) -> None:
        shell, flag = _pick_shell()
        proc_env = os.environ.copy()
        if self.env:
            proc_env.update(self.env)
        self._proc = await asyncio.create_subprocess_exec(
            shell, flag, self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=proc_env,
            start_new_session=True,  # own process group → killpg on kill
        )
        self._started_at = time.time()
        self._stdout_task = asyncio.create_task(self._drain(self._proc.stdout, is_stderr=False))
        self._stderr_task = asyncio.create_task(self._drain(self._proc.stderr, is_stderr=True))

    # ── Output accounting ───────────────────────────────────────────

    async def _drain(self, stream: asyncio.StreamReader | None, *, is_stderr: bool) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            self._append(chunk, is_stderr=is_stderr)

    def _append(self, chunk: bytes, *, is_stderr: bool) -> None:
        buf = self._stderr_buf if is_stderr else self._stdout_buf
        buf.extend(chunk)
        if is_stderr:
            self._stderr_total += len(chunk)
        else:
            self._stdout_total += len(chunk)
        # Truncate from the front if past cap.
        if len(buf) > MAX_STREAM_BYTES:
            dropped = len(buf) - MAX_STREAM_BYTES
            del buf[:dropped]
            if is_stderr:
                self._stderr_dropped += dropped
            else:
                self._stdout_dropped += dropped

    # ── Public read API ─────────────────────────────────────────────

    def read(
        self, *, since_stdout: int, since_stderr: int
    ) -> tuple[str, str]:
        """Return (stdout_delta, stderr_delta) starting from the given
        byte cursors on the *original* stream (not the buffer). Bytes
        that have been dropped due to truncation are simply skipped.
        """
        return (
            self._slice(self._stdout_buf, since_stdout, self._stdout_total, self._stdout_dropped),
            self._slice(self._stderr_buf, since_stderr, self._stderr_total, self._stderr_dropped),
        )

    @staticmethod
    def _slice(
        buf: bytearray, since: int, total: int, dropped: int,
    ) -> str:
        """Return the buffer slice starting at stream-offset ``since``.

        Stream math: the buffer holds bytes [dropped, total). A caller
        asking for ``since < dropped`` only gets what's still present
        (i.e. starting from ``dropped``). A caller asking for
        ``since >= total`` gets an empty string.
        """
        if since >= total:
            return ""
        start = max(0, since - dropped)
        return bytes(buf[start:]).decode("utf-8", errors="replace")

    # ── State accessors ─────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def exit_code(self) -> int | None:
        return self._exit_code if not self.is_running else None

    @property
    def signal(self) -> str | None:
        return self._signal

    @property
    def stdout_bytes_total(self) -> int:
        return self._stdout_total

    @property
    def stderr_bytes_total(self) -> int:
        return self._stderr_total

    @property
    def started_at(self) -> float | None:
        return self._started_at

    @property
    def completed_at(self) -> float | None:
        return self._completed_at

    @property
    def stdout_dropped(self) -> int:
        return self._stdout_dropped

    @property
    def stderr_dropped(self) -> int:
        return self._stderr_dropped

    # Triggered by handlers once the process has exited — drains
    # remaining buffered output and finalises exit_code / signal.
    async def finalise(self) -> None:
        if self._proc is None:
            return
        rc = await self._proc.wait()
        if self._stdout_task:
            try:
                await self._stdout_task
            except Exception as e:  # noqa: BLE001
                logger.warning("stdout drain error for %s: %s", self.shell_id, e)
        if self._stderr_task:
            try:
                await self._stderr_task
            except Exception as e:  # noqa: BLE001
                logger.warning("stderr drain error for %s: %s", self.shell_id, e)
        # Signal naming: negative returncodes = killed by signal on
        # POSIX. Translate back to a name.
        if rc is not None and rc < 0:
            sig = -rc
            try:
                self._signal = signal_module.Signals(sig).name.removeprefix("SIG")
            except ValueError:
                self._signal = str(sig)
            self._exit_code = None
        else:
            self._exit_code = rc
        self._completed_at = time.time()

    # Placeholder — next task implements these.
    async def write_stdin(self, text: str, *, press_enter: bool = True) -> int:
        raise NotImplementedError

    async def kill(self, *, signal_name: SignalName = "TERM", grace_seconds: float = DEFAULT_KILL_GRACE) -> None:
        raise NotImplementedError
