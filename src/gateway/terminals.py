"""PTY-backed interactive terminals exposed through the gateway.

This is the *user-facing* "SSH terminal" surface: a real pseudo-terminal
on the host running the OpenAgent server, driven live by a client (the
desktop app's System tab, the CLI's ``terminal`` command). It is
deliberately separate from the agent-facing ``shell`` MCP — that one is
a pipe-based, one-shot/background command runner the *model* calls;
this one is a full PTY a *human* types into, so interactive programs
(``vim``, ``top``, ``htop``, ``ssh`` onward, REPLs) render and respond
exactly as they would over SSH.

Wire shape (see :mod:`src.gateway.protocol`)::

    client → gateway   terminal_open   {terminal_id, cols, rows, cwd?, shell?}
                       terminal_input  {terminal_id, data}      # base64 bytes
                       terminal_resize {terminal_id, cols, rows}
                       terminal_close  {terminal_id}
                       terminal_signal {terminal_id, signal}    # INT|TERM|...

    gateway → client   terminal_ready  {terminal_id, pid, shell, cols, rows}
                       terminal_output {terminal_id, data}      # base64 bytes
                       terminal_exit   {terminal_id, exit_code, signal}
                       terminal_error  {terminal_id, error}

Authentication is the gateway's existing per-device certificate — by the
time a ``terminal_open`` frame reaches here the connection is already a
verified member of the agent's network, so terminals inherit the same
trust boundary as every other gateway action (vision §10, §11). A
terminal is bound to the websocket that opened it and is torn down when
that connection drops, matching SSH session semantics.

POSIX only. The ``pty`` module is Unix-only; on Windows ``open`` returns
a ``terminal_error`` rather than raising — the server host in every
shipped deployment is macOS or Linux.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal as signal_module
import struct
from typing import Awaitable, Callable

from src.core.logging import elog

logger = logging.getLogger(__name__)

# Whether this host can host a PTY at all. ``pty``/``termios``/``fcntl``
# are POSIX-only; importing them up front lets ``open`` fail cleanly
# with a ``terminal_error`` on Windows instead of throwing mid-spawn.
try:  # pragma: no cover - platform gate
    import fcntl
    import pty
    import termios

    PTY_SUPPORTED = True
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    pty = None  # type: ignore[assignment]
    termios = None  # type: ignore[assignment]
    PTY_SUPPORTED = False

DEFAULT_COLS = 80
DEFAULT_ROWS = 24
# Cap a single ``os.read`` off the PTY master. 64 KiB is plenty per
# tick — a flood (``yes``, ``cat bigfile``) just produces many frames.
READ_CHUNK = 65536
# Bound the wait for the child to reap after we close the master fd, so
# a wedged grandchild can't pin the close coroutine forever.
REAP_TIMEOUT = 5.0

# Signals a client may deliver to the foreground process group via
# ``terminal_signal``. Mapped by name so the wire stays OS-agnostic.
_SIGNALS = {
    "INT": signal_module.SIGINT,
    "TERM": signal_module.SIGTERM,
    "HUP": signal_module.SIGHUP,
    "QUIT": signal_module.SIGQUIT,
    "KILL": signal_module.SIGKILL,
}


OutputCb = Callable[[bytes], Awaitable[None]]
ExitCb = Callable[[int | None, str | None], Awaitable[None]]


def pick_shell() -> str:
    """Return the absolute path to a sensible interactive login shell.

    Honours ``$SHELL`` (what the user actually runs), falling back to a
    per-platform default and finally to whatever ``/bin/sh`` resolves
    to. Mirrors the agent ``shell`` MCP's ``_pick_shell`` so both
    surfaces land on the same program.
    """
    env_shell = os.environ.get("SHELL", "").strip()
    if env_shell and os.path.exists(env_shell):
        return env_shell
    import platform as _platform

    sysname = _platform.system().lower()
    candidates = (
        ["/bin/zsh", "/bin/bash", "/bin/sh"]
        if sysname == "darwin"
        else ["/bin/bash", "/bin/sh"]
    )
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return shutil.which("sh") or "/bin/sh"


def _winsize(cols: int, rows: int) -> bytes:
    """Pack a ``struct winsize`` (rows, cols, xpixel, ypixel)."""
    cols = max(1, min(int(cols or DEFAULT_COLS), 2000))
    rows = max(1, min(int(rows or DEFAULT_ROWS), 2000))
    return struct.pack("HHHH", rows, cols, 0, 0)


class TerminalSession:
    """One live PTY + the child shell attached to it.

    Reads off the master fd are pumped onto an :class:`asyncio.Queue`
    by a non-blocking ``loop.add_reader`` callback, then drained in
    order by a single sender task. The queue is what keeps output
    *ordered* across the read→send boundary — firing a task per chunk
    would let the loop interleave sends and scramble a fast stream.
    """

    def __init__(
        self,
        *,
        terminal_id: str,
        client_id: str,
        on_output: OutputCb,
        on_exit: ExitCb,
        shell: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
    ) -> None:
        self.terminal_id = terminal_id
        self.client_id = client_id
        self._on_output = on_output
        self._on_exit = on_exit
        self.shell = shell or pick_shell()
        self.cwd = cwd if (cwd and os.path.isdir(cwd)) else os.path.expanduser("~")
        self._extra_env = env or {}
        self.cols = cols
        self.rows = rows

        self._master_fd: int | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._out_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._sender_task: asyncio.Task | None = None
        self._closing = False
        self._exited = False

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Spawn the shell on a fresh PTY and begin pumping output."""
        if not PTY_SUPPORTED:
            raise RuntimeError("PTY terminals are not supported on this host OS")

        master_fd, slave_fd = pty.openpty()
        self._master_fd = master_fd
        # Size the PTY before exec so the shell's first prompt already
        # knows the geometry (otherwise full-screen apps mis-paint until
        # the first resize).
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, _winsize(self.cols, self.rows))
        except OSError:
            pass

        proc_env = os.environ.copy()
        proc_env.update(self._extra_env)
        # A real terminal advertises itself so curses apps enable colour
        # and cursor addressing. Keep an existing TERM if the caller set
        # one; otherwise pick a widely-supported default.
        proc_env.setdefault("TERM", "xterm-256color")
        proc_env["OPENAGENT_TERMINAL"] = "1"

        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.shell,
                "-l" if self._is_login_capable() else "-i",
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=self.cwd,
                env=proc_env,
                start_new_session=True,  # own session+pgrp → job control, killpg
            )
        finally:
            # The child holds its own dup of the slave now; the parent
            # must drop its copy or the master never sees EOF on exit.
            os.close(slave_fd)

        # Non-blocking master so the reader callback never stalls the
        # event loop on a partial read.
        os.set_blocking(master_fd, False)
        loop = asyncio.get_running_loop()
        loop.add_reader(master_fd, self._on_readable)
        self._sender_task = asyncio.create_task(
            self._drain_output(), name=f"terminal-out-{self.terminal_id}"
        )
        elog(
            "terminal.start",
            terminal_id=self.terminal_id,
            client_id=self.client_id,
            pid=self.pid,
            shell=self.shell,
            cwd=self.cwd,
        )

    def _is_login_capable(self) -> bool:
        """``sh``/``bash``/``zsh`` accept ``-l``; be conservative elsewhere."""
        base = os.path.basename(self.shell)
        return base in {"bash", "zsh", "sh", "-bash", "-zsh"}

    # ── Output pump ─────────────────────────────────────────────────

    def _on_readable(self) -> None:
        """``loop.add_reader`` callback — drain whatever's ready, once."""
        if self._master_fd is None:
            return
        try:
            data = os.read(self._master_fd, READ_CHUNK)
        except BlockingIOError:
            return
        except OSError:
            # EIO on Linux when the slave side closes (child exited) —
            # the canonical "PTY hung up" signal. Finalise.
            self._schedule_finalize()
            return
        if not data:
            # EOF — child closed the PTY.
            self._schedule_finalize()
            return
        self._out_queue.put_nowait(data)

    async def _drain_output(self) -> None:
        """Single consumer: forward queued PTY bytes to the client in order."""
        try:
            while True:
                data = await self._out_queue.get()
                if data is _SENTINEL:
                    return
                try:
                    await self._on_output(data)
                except Exception as e:  # noqa: BLE001 — one bad send mustn't kill the pump
                    logger.debug("terminal output send failed (%s): %s", self.terminal_id, e)
        except asyncio.CancelledError:
            raise

    def _schedule_finalize(self) -> None:
        if self._exited or self._closing:
            return
        self._exited = True
        asyncio.create_task(
            self._finalize(), name=f"terminal-exit-{self.terminal_id}"
        )

    async def _finalize(self) -> None:
        """Child exited on its own — reap it and notify the client."""
        self._detach_reader()
        exit_code, sig = await self._reap()
        await self._flush_and_stop_sender()
        elog(
            "terminal.exit",
            terminal_id=self.terminal_id,
            exit_code=exit_code,
            signal=sig,
        )
        try:
            await self._on_exit(exit_code, sig)
        except Exception as e:  # noqa: BLE001
            logger.debug("terminal exit callback failed (%s): %s", self.terminal_id, e)

    def rebind(self, on_output: OutputCb, on_exit: ExitCb) -> None:
        """Point output/exit at a new client websocket (re-attach).

        Used when a detached terminal window is re-opened: the shell is
        still alive (its window closed without an explicit
        ``terminal_close``, e.g. an OS window-destroy that skipped the
        renderer cleanup), so instead of respawning we just swap where
        its bytes go. ``_drain_output`` reads ``self._on_output`` fresh
        each iteration, so the swap takes effect on the next chunk.
        """
        self._on_output = on_output
        self._on_exit = on_exit

    # ── Input / control ─────────────────────────────────────────────

    def write(self, data: bytes) -> None:
        """Feed bytes to the PTY (the user's keystrokes)."""
        if self._master_fd is None or self._closing:
            return
        try:
            os.write(self._master_fd, data)
        except OSError as e:
            logger.debug("terminal write failed (%s): %s", self.terminal_id, e)

    def resize(self, cols: int, rows: int) -> None:
        """Apply a new window size and signal SIGWINCH to the foreground app."""
        self.cols, self.rows = cols, rows
        if self._master_fd is None:
            return
        try:
            fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, _winsize(cols, rows))
        except OSError as e:
            logger.debug("terminal resize failed (%s): %s", self.terminal_id, e)

    def send_signal(self, name: str) -> None:
        """Deliver a signal to the child's process group (e.g. Ctrl-C → INT)."""
        sig = _SIGNALS.get(name.upper())
        if sig is None or self._proc is None or self._proc.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), sig)
        except (ProcessLookupError, OSError) as e:
            logger.debug("terminal signal failed (%s): %s", self.terminal_id, e)

    async def close(self) -> None:
        """Tear the terminal down (client asked, or the connection dropped)."""
        if self._closing:
            return
        self._closing = True
        self._detach_reader()
        # SIGHUP the whole session, like closing a real terminal window.
        if self._proc is not None and self._proc.returncode is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal_module.SIGHUP)
            except (ProcessLookupError, OSError):
                pass
        await self._reap()
        await self._flush_and_stop_sender()
        elog("terminal.close", terminal_id=self.terminal_id, client_id=self.client_id)

    # ── Teardown helpers ────────────────────────────────────────────

    def _detach_reader(self) -> None:
        if self._master_fd is None:
            return
        try:
            asyncio.get_running_loop().remove_reader(self._master_fd)
        except (ValueError, RuntimeError, OSError):
            pass

    async def _reap(self) -> tuple[int | None, str | None]:
        """Wait (bounded) for the child, escalating to SIGKILL on stall.

        Returns ``(exit_code, signal_name)``; one is ``None`` depending
        on whether the child exited normally or was killed by a signal.
        """
        proc = self._proc
        if proc is None:
            return None, None
        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=REAP_TIMEOUT)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(proc.pid), signal_module.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=REAP_TIMEOUT)
                except asyncio.TimeoutError:
                    pass
        # Now close the master fd — after the child is gone so any final
        # output has already been read.
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None
        rc = proc.returncode
        if rc is not None and rc < 0:
            try:
                return None, signal_module.Signals(-rc).name.removeprefix("SIG")
            except ValueError:
                return None, str(-rc)
        return rc, None

    async def _flush_and_stop_sender(self) -> None:
        """Drain remaining queued output, then stop the sender task."""
        # Push a sentinel so the sender exits after the real bytes ahead
        # of it have been delivered — preserves trailing output (e.g. a
        # shell's goodbye line) that landed just before EOF.
        self._out_queue.put_nowait(_SENTINEL)
        task = self._sender_task
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=REAP_TIMEOUT)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                task.cancel()
            self._sender_task = None


# Unique marker enqueued to tell the sender task to stop after flushing.
_SENTINEL = object()


class TerminalManager:
    """Owns every live :class:`TerminalSession`, keyed by (client, id).

    The gateway holds one of these. Per-client teardown
    (:meth:`close_for_client`) runs from the websocket's ``finally`` so
    a dropped connection never leaves orphan shells running on the host.
    """

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], TerminalSession] = {}

    def get(self, client_id: str, terminal_id: str) -> TerminalSession | None:
        return self._sessions.get((client_id, terminal_id))

    def list_for_client(self, client_id: str) -> list[TerminalSession]:
        return [s for (cid, _), s in self._sessions.items() if cid == client_id]

    def count(self) -> int:
        return len(self._sessions)

    async def open(
        self,
        *,
        client_id: str,
        terminal_id: str,
        on_output: OutputCb,
        on_exit: ExitCb,
        shell: str | None = None,
        cwd: str | None = None,
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
    ) -> TerminalSession:
        """Create + start a terminal — or re-attach to a live one.

        If a session under this ``(client_id, terminal_id)`` is still
        running (its window was closed without an explicit
        ``terminal_close``), re-attach: rebind output to the new ws and
        resize to the new geometry rather than spawning a second shell.
        A dead session is replaced.
        """
        key = (client_id, terminal_id)
        existing = self._sessions.get(key)
        if existing is not None:
            if existing.is_running:
                existing.rebind(on_output, on_exit)
                existing.resize(cols, rows)
                elog(
                    "terminal.reattach",
                    terminal_id=terminal_id,
                    client_id=client_id,
                    pid=existing.pid,
                )
                return existing
            await existing.close()
        session = TerminalSession(
            terminal_id=terminal_id,
            client_id=client_id,
            on_output=on_output,
            on_exit=on_exit,
            shell=shell,
            cwd=cwd,
            cols=cols,
            rows=rows,
        )
        self._sessions[key] = session
        try:
            await session.start()
        except Exception:
            self._sessions.pop(key, None)
            raise
        return session

    def write(self, client_id: str, terminal_id: str, data: bytes) -> None:
        s = self._sessions.get((client_id, terminal_id))
        if s is not None:
            s.write(data)

    def resize(self, client_id: str, terminal_id: str, cols: int, rows: int) -> None:
        s = self._sessions.get((client_id, terminal_id))
        if s is not None:
            s.resize(cols, rows)

    def signal(self, client_id: str, terminal_id: str, name: str) -> None:
        s = self._sessions.get((client_id, terminal_id))
        if s is not None:
            s.send_signal(name)

    async def close(self, client_id: str, terminal_id: str) -> bool:
        """Close one terminal. Returns ``True`` if it existed."""
        s = self._sessions.pop((client_id, terminal_id), None)
        if s is None:
            return False
        await s.close()
        return True

    async def close_for_client(self, client_id: str) -> int:
        """Close every terminal a disconnecting client owned."""
        keys = [k for k in self._sessions if k[0] == client_id]
        for key in keys:
            s = self._sessions.pop(key, None)
            if s is not None:
                try:
                    await s.close()
                except Exception as e:  # noqa: BLE001
                    logger.debug("terminal close failed (%s): %s", key, e)
        return len(keys)

    async def close_all(self) -> None:
        """Shut every terminal down (gateway stop)."""
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for s in sessions:
            try:
                await s.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("terminal close_all failed: %s", e)
