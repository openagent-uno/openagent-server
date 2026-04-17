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

from openagent.mcp.servers.shell.events import ShellEvent, ShellEventKind

if TYPE_CHECKING:
    from openagent.mcp.servers.shell.shells import BackgroundShell

logger = logging.getLogger(__name__)

# Queue cap per session — chatty or broken session can't exhaust memory.
_MAX_QUEUED_EVENTS = 200


@dataclass
class ShellRecord:
    shell_id: str
    session_id: str | None
    command: str
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    exit_code: int | None = None
    signal: str | None = None
    # The BackgroundShell is attached after spawn (None while tests use
    # register() directly without spawning a real subprocess).
    shell: "BackgroundShell | None" = None

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
        self._events: dict[str, asyncio.Event] = {}
        self._queues: dict[str, deque[ShellEvent]] = {}

    # ── Registration ────────────────────────────────────────────────

    def register(
        self,
        *,
        shell_id: str,
        session_id: str | None,
        command: str,
        shell: "BackgroundShell | None" = None,
    ) -> ShellRecord:
        record = ShellRecord(
            shell_id=shell_id,
            session_id=session_id,
            command=command,
            shell=shell,
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

    def has_running(self, session_id: str | None) -> bool:
        for rec in self.list_for_session(session_id):
            if not rec.is_completed:
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

    def post_event(self, session_id: str | None, event: ShellEvent) -> None:
        """Push a terminal event into ``session_id``'s queue and wake any
        waiter. No-op when ``session_id`` is None — we only do active
        wake-up for shells that have a session."""
        if session_id is None:
            return
        q = self._queues.setdefault(session_id, deque(maxlen=_MAX_QUEUED_EVENTS))
        q.append(event)
        ev = self._events.setdefault(session_id, asyncio.Event())
        ev.set()

    def drain(self, session_id: str | None) -> list[ShellEvent]:
        """Return every queued event for ``session_id`` and clear the queue."""
        if session_id is None:
            return []
        q = self._queues.get(session_id)
        if not q:
            return []
        out = list(q)
        q.clear()
        ev = self._events.get(session_id)
        if ev is not None:
            ev.clear()
        return out

    async def wait(self, session_id: str | None, timeout: float) -> list[ShellEvent]:
        """Await up to ``timeout`` seconds for any event on ``session_id``.

        Returns the drained events (possibly empty on timeout). Safe to
        call when no shells are registered — returns [] immediately
        after the timeout.
        """
        if session_id is None or timeout <= 0:
            return self.drain(session_id)
        # Fast path — already something queued.
        if self._queues.get(session_id):
            return self.drain(session_id)
        ev = self._events.setdefault(session_id, asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return []
        return self.drain(session_id)

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
        self._events.pop(session_id, None)
        self._queues.pop(session_id, None)
        return killed
