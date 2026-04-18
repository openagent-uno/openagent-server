"""Per-client session tracking with a FIFO message queue.

Each client can have multiple chat sessions, but queueing/stop semantics
are currently owned at the client level: one active run plus one pending
FIFO queue per client. Session metadata is RAM-only and lost on restart.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from openagent.core.logging import elog


Handler = Callable[[], Awaitable[None]]

# Maximum pending messages per client before new messages are rejected.
MAX_QUEUE_SIZE = 20


@dataclass
class Session:
    id: str
    client_id: str
    history_mode: str | None = None


@dataclass
class _QueuedItem:
    handler: Handler
    session_id: str | None = None


@dataclass
class _ClientState:
    sessions: dict[str, Session] = field(default_factory=dict)
    pending: asyncio.Queue = field(default_factory=asyncio.Queue)
    current_task: asyncio.Task | None = None
    # Session id the currently running ``current_task`` is handling (if any).
    # Tracked separately from the task so per-session ``/stop`` can decide
    # whether to cancel: telegram/discord bridges multiplex many users onto
    # a single ``client_id``, so we only want to interrupt the user who
    # actually issued the command.
    current_session_id: str | None = None
    worker_task: asyncio.Task | None = None


class SessionManager:
    """Manages sessions and message queuing per client."""

    def __init__(self, agent_name: str = "agent"):
        self.agent_name = agent_name
        self._clients: dict[str, _ClientState] = {}

    def _state(self, client_id: str) -> _ClientState:
        if client_id not in self._clients:
            self._clients[client_id] = _ClientState()
        return self._clients[client_id]

    # ── sessions ──

    def create_session(self, client_id: str) -> str:
        sid = f"{self.agent_name}:{client_id}:{uuid.uuid4().hex[:8]}"
        self._state(client_id).sessions[sid] = Session(id=sid, client_id=client_id)
        elog("session.create", client_id=client_id, session_id=sid)
        return sid

    def get_or_create_session(self, client_id: str, session_id: str) -> str:
        st = self._state(client_id)
        created = False
        if session_id not in st.sessions:
            st.sessions[session_id] = Session(id=session_id, client_id=client_id)
            created = True
        elog("session.attach", client_id=client_id, session_id=session_id, created=created)
        return session_id

    def reset_session(self, client_id: str, session_id: str) -> str:
        st = self._state(client_id)
        st.sessions.pop(session_id, None)
        return self.create_session(client_id)

    def bind_history_mode(self, client_id: str, session_id: str, history_mode: str | None) -> str | None:
        """Lock a session to a history ownership mode once it starts running."""
        if not history_mode:
            return None
        st = self._state(client_id)
        session = st.sessions.get(session_id)
        if not session:
            session = Session(id=session_id, client_id=client_id)
            st.sessions[session_id] = session
        if session.history_mode and session.history_mode != history_mode:
            raise ValueError(
                f"Session '{session_id}' is locked to {session.history_mode}-managed history "
                f"and cannot switch to {history_mode}-managed history."
            )
        if session.history_mode != history_mode:
            session.history_mode = history_mode
            elog("session.history_mode", client_id=client_id, session_id=session_id, history_mode=history_mode)
        return session.history_mode

    def list_sessions(self, client_id: str) -> list[str]:
        return list(self._state(client_id).sessions.keys())

    # ── queue ──

    def is_busy(self, client_id: str) -> bool:
        st = self._clients.get(client_id)
        if not st or st.current_task is None:
            return False
        return not st.current_task.done()

    def queue_depth(self, client_id: str) -> int:
        st = self._clients.get(client_id)
        return st.pending.qsize() if st else 0

    def clear_queue(self, client_id: str) -> int:
        st = self._clients.get(client_id)
        if not st:
            return 0
        count = 0
        while True:
            try:
                st.pending.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        if count:
            elog("queue.clear", client_id=client_id, removed=count)
        return count

    async def enqueue(self, client_id: str, handler: Handler, session_id: str | None = None) -> int:
        """Enqueue a message handler. Returns queue position (0 = running now),
        or -1 if the queue is full and the message was rejected."""
        st = self._state(client_id)
        if st.pending.qsize() >= MAX_QUEUE_SIZE:
            elog("queue.full", level="warning", client_id=client_id, max=MAX_QUEUE_SIZE)
            return -1
        running = 1 if self.is_busy(client_id) else 0
        position = st.pending.qsize() + running
        await st.pending.put(_QueuedItem(handler=handler, session_id=session_id))
        elog(
            "queue.enqueue",
            client_id=client_id,
            session_id=session_id,
            position=position,
            depth=st.pending.qsize(),
            busy=bool(running),
        )
        if st.worker_task is None or st.worker_task.done():
            st.worker_task = asyncio.create_task(self._worker(client_id))
        return position

    async def _worker(self, client_id: str) -> None:
        st = self._clients[client_id]
        while True:
            try:
                item = st.pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            elog(
                "queue.start",
                client_id=client_id,
                session_id=item.session_id,
                remaining=st.pending.qsize(),
            )
            task = asyncio.create_task(item.handler())
            st.current_task = task
            st.current_session_id = item.session_id
            try:
                await task
            except asyncio.CancelledError:
                elog("queue.cancel", client_id=client_id, session_id=item.session_id)
            except Exception as e:
                elog("queue.error", level="error",
                     client_id=client_id, session_id=item.session_id, error=str(e))
            finally:
                st.current_task = None
                st.current_session_id = None
                elog("queue.done", client_id=client_id, session_id=item.session_id, remaining=st.pending.qsize())

    def stop_current(self, client_id: str, session_id: str | None = None) -> bool:
        """Cancel the currently-running handler for ``client_id``.

        When ``session_id`` is provided, only cancels if the running task
        belongs to that session — used by bridges (telegram, discord) where
        one gateway client multiplexes many users, so ``/stop`` from user A
        must not cancel user B's turn. ``session_id=None`` preserves the
        legacy "cancel whatever is running" behaviour used by direct ws
        clients and by administrative shutdowns.
        """
        st = self._clients.get(client_id)
        if not st or st.current_task is None or st.current_task.done():
            return False
        if session_id is not None and st.current_session_id != session_id:
            elog(
                "queue.stop_skipped",
                client_id=client_id,
                session_id=session_id,
                running_session_id=st.current_session_id,
            )
            return False
        st.current_task.cancel()
        elog("queue.stop_requested", client_id=client_id, session_id=session_id)
        return True

    def clear_queue_for_session(self, client_id: str, session_id: str) -> int:
        """Drop queued handlers whose ``session_id`` matches.

        Counterpart to :meth:`stop_current`'s session scoping — leaves other
        users' queued messages alone. Returns the number of items removed.
        """
        st = self._clients.get(client_id)
        if not st:
            return 0
        kept: list[_QueuedItem] = []
        removed = 0
        while True:
            try:
                item = st.pending.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item.session_id == session_id:
                removed += 1
            else:
                kept.append(item)
        for item in kept:
            st.pending.put_nowait(item)
        if removed:
            elog(
                "queue.clear_session",
                client_id=client_id,
                session_id=session_id,
                removed=removed,
            )
        return removed

    async def shutdown(self) -> None:
        tasks: list[asyncio.Task] = []
        for st in self._clients.values():
            if st.current_task and not st.current_task.done():
                st.current_task.cancel()
                tasks.append(st.current_task)
            if st.worker_task and not st.worker_task.done():
                st.worker_task.cancel()
                tasks.append(st.worker_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._clients.clear()
