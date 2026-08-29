"""Process-wide singleton that tracks background shells and the
per-session event queues the agent loop awaits.

Owned by the agent process. Tool handlers write; agent._run_inner
reads. Thread-safety: single event loop, no cross-thread access.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.mcp.servers.shell.backends import get_exec_backend
from src.mcp.servers.shell.events import ShellEvent, ShellEventKind

if TYPE_CHECKING:
    from src.mcp.servers.shell.shells import BackgroundShell

logger = logging.getLogger(__name__)

# Queue cap per session — chatty or broken session can't exhaust memory.
_MAX_QUEUED_EVENTS = 200

# Exact client capability host which owns a client-local background process.
# The generation is part of the key: a reconnect with a newer generation must
# never inherit a process/event from the replaced capability channel.
ClientHostKey = tuple[str, str, int]
EventQueueKey = tuple[str, ClientHostKey | None]


@dataclass
class ShellRecord:
    shell_id: str
    session_id: str | None
    command: str
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    exit_code: int | None = None
    signal: str | None = None
    # ``None`` is a process running on the OpenAgent server. Client-local
    # proxy records are pinned to the exact interactive capability instance.
    client_host: ClientHostKey | None = None
    # The BackgroundShell is attached after spawn (None while tests use
    # register() directly without spawning a real subprocess).
    shell: "BackgroundShell | None" = None
    # Per-caller cursor used by shell_output(since_last=True).
    last_read_stdout: int = 0
    last_read_stderr: int = 0

    @property
    def is_completed(self) -> bool:
        return self.completed_at is not None


class ShellHub:
    """Singleton (per agent process) for background-shell bookkeeping.

    Not thread-safe. Every method must be called from the single agent
    event loop. See module docstring.
    """

    def __init__(self) -> None:
        self._shells: dict[str, ShellRecord] = {}
        self._by_session: dict[str, set[str]] = {}
        self._events: dict[EventQueueKey, asyncio.Event] = {}
        self._queues: dict[EventQueueKey, deque[ShellEvent]] = {}

    # ── Registration ────────────────────────────────────────────────

    def register(
        self,
        *,
        shell_id: str,
        session_id: str | None,
        command: str,
        shell: "BackgroundShell | None" = None,
        client_host: ClientHostKey | None = None,
    ) -> ShellRecord:
        record = ShellRecord(
            shell_id=shell_id,
            session_id=session_id,
            command=command,
            shell=shell,
            client_host=client_host,
        )
        self._shells[shell_id] = record
        if session_id is not None:
            self._by_session.setdefault(session_id, set()).add(shell_id)
        return record

    def get(self, shell_id: str) -> ShellRecord | None:
        return self._shells.get(shell_id)

    def list_for_session(self, session_id: str | None) -> list[ShellRecord]:
        """Return records for ``session_id``. ``None`` means every record,
        regardless of session."""
        if session_id is None:
            return list(self._shells.values())
        ids = self._by_session.get(session_id, set())
        return [self._shells[i] for i in ids if i in self._shells]

    def has_running(
        self,
        session_id: str | None,
        *,
        client_host: ClientHostKey | None = None,
    ) -> bool:
        """Return whether this turn has a relevant running shell.

        Server shells are visible to every turn for their session. A client
        proxy is visible only to the exact host which started it; in
        particular, automated turns (``client_host=None``) cannot be woken by
        or wait on a user's computer.
        """
        for rec in self.list_for_session(session_id):
            visible = rec.client_host is None or rec.client_host == client_host
            if visible and not rec.is_completed:
                return True
        return False

    def mark_completed(
        self,
        shell_id: str,
        *,
        exit_code: int | None,
        signal: str | None,
    ) -> None:
        rec = self._shells.get(shell_id)
        if rec is None:
            return
        rec.completed_at = time.time()
        rec.exit_code = exit_code
        rec.signal = signal

    # ── Event queue ─────────────────────────────────────────────────

    def post_event(
        self,
        session_id: str | None,
        event: ShellEvent,
        *,
        client_host: ClientHostKey | None = None,
    ) -> None:
        """Push a terminal event into ``session_id``'s queue and wake any
        waiter. No-op when ``session_id`` is None — we only do active
        wake-up for shells that have a session.

        The queue is bounded to ``_MAX_QUEUED_EVENTS`` (200); when full, the
        **oldest** event is silently dropped. See module docstring.
        """
        if session_id is None:
            return
        key = (session_id, client_host)
        q = self._queues.setdefault(key, deque(maxlen=_MAX_QUEUED_EVENTS))
        q.append(event)
        ev = self._events.setdefault(key, asyncio.Event())
        ev.set()

    def drain(
        self,
        session_id: str | None,
        *,
        client_host: ClientHostKey | None = None,
    ) -> list[ShellEvent]:
        """Drain server events and this exact client's events for a session."""
        if session_id is None:
            return []
        keys = [(session_id, None)]
        if client_host is not None:
            keys.append((session_id, client_host))
        out: list[ShellEvent] = []
        for key in keys:
            q = self._queues.get(key)
            if q:
                out.extend(q)
                q.clear()
            ev = self._events.get(key)
            if ev is not None:
                ev.clear()  # Clear queue first: queue/signal stay in lockstep.
        # Events from the server and client have independent bounded queues.
        # Merge them by event timestamp so reminders remain chronological.
        out.sort(key=lambda item: item.at)
        return out

    async def wait(
        self,
        session_id: str | None,
        timeout: float,
        *,
        client_host: ClientHostKey | None = None,
    ) -> list[ShellEvent]:
        """Await up to ``timeout`` seconds for any event on ``session_id``.

        Returns the drained events (possibly empty on timeout). Safe to
        call when no shells are registered — returns [] immediately
        after the timeout. ``timeout <= 0`` short-circuits to an immediate
        drain (non-blocking poll).
        """
        if session_id is None or timeout <= 0:
            return self.drain(session_id, client_host=client_host)
        # Fast path — already something queued.
        keys = [(session_id, None)]
        if client_host is not None:
            keys.append((session_id, client_host))
        if any(self._queues.get(key) for key in keys):
            return self.drain(session_id, client_host=client_host)
        waiters = [
            asyncio.create_task(self._events.setdefault(key, asyncio.Event()).wait())
            for key in keys
        ]
        try:
            done, pending = await asyncio.wait(
                waiters,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return []
        finally:
            for task in waiters:
                if not task.done():
                    task.cancel()
            if waiters:
                await asyncio.gather(*waiters, return_exceptions=True)
        if not any(self._queues.get(key) for key in keys):
            return []
        return self.drain(session_id, client_host=client_host)

    # ── Purge ───────────────────────────────────────────────────────

    async def purge_session(self, session_id: str) -> list[str]:
        """Kill every shell for ``session_id`` and drop the session.

        Returns the list of shell_ids that were purged (for logging).
        Kills *live* shells via ``BackgroundShell.kill`` with SIGKILL
        so shutdown is bounded.
        """
        ids = list(self._by_session.pop(session_id, set()))
        killed: list[str] = []
        for sid in ids:
            rec = self._shells.pop(sid, None)
            if rec is None:
                continue
            killed.append(sid)
            if rec.shell is not None and not rec.is_completed:
                try:
                    await rec.shell.kill(signal_name="KILL", grace_seconds=0)
                except Exception as e:  # noqa: BLE001 — best-effort
                    logger.debug("purge_session kill failed for %s: %s", sid, e)
        for key in [key for key in self._events if key[0] == session_id]:
            self._events.pop(key, None)
        for key in [key for key in self._queues if key[0] == session_id]:
            self._queues.pop(key, None)
        return killed

    # ── GC / shutdown ───────────────────────────────────────────────

    def gc(self, ttl_seconds: float = 600.0) -> list[str]:
        """Drop completed shells older than ``ttl_seconds``.

        Live shells are never touched. Returns the shell_ids removed
        (for debug logging). Called by the agent's idle cleanup loop.
        """
        now = time.time()
        victims: list[str] = []
        for sid, rec in list(self._shells.items()):
            if not rec.is_completed:
                continue
            if rec.completed_at is None:
                continue
            if (now - rec.completed_at) < ttl_seconds:
                continue
            victims.append(sid)
            del self._shells[sid]
            if rec.session_id and rec.session_id in self._by_session:
                self._by_session[rec.session_id].discard(sid)
                if not self._by_session[rec.session_id]:
                    del self._by_session[rec.session_id]
        return victims

    async def shutdown(self) -> None:
        """Purge every session and clear all queues / events.

        Called from ``Agent.shutdown`` so the process can exit without
        leaking background subprocesses.
        """
        for session_id in list(self._by_session.keys()):
            await self.purge_session(session_id)
        # Drop shells that were never associated with a session.
        for sid, rec in list(self._shells.items()):
            if rec.shell is not None and not rec.is_completed:
                try:
                    await rec.shell.kill(signal_name="KILL", grace_seconds=0)
                except Exception as e:  # noqa: BLE001
                    logger.debug("shutdown kill failed for %s: %s", sid, e)
            del self._shells[sid]
        self._events.clear()
        self._queues.clear()
        # Tear down the active exec backend so a docker sandbox container does
        # not leak past process exit. No-op for the default LocalBackend; the
        # DockerBackend runs ``docker rm -f`` (and handles its own errors, so
        # this call never raises during shutdown).
        await get_exec_backend().cleanup()
