"""Per-client session management with FIFO message queue.

Each client can have multiple chat sessions. Messages within a session
are processed sequentially (one at a time) to prevent race conditions.
Sessions are RAM-only — lost on restart.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

Handler = Callable[[], Awaitable[None]]

# Maximum pending messages per client before new messages are rejected.
MAX_QUEUE_SIZE = 20


@dataclass
class Session:
    id: str
    client_id: str


@dataclass
class _ClientState:
    sessions: dict[str, Session] = field(default_factory=dict)
    pending: asyncio.Queue = field(default_factory=asyncio.Queue)
    current_task: asyncio.Task | None = None
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
        if session_id not in st.sessions:
            st.sessions[session_id] = Session(id=session_id, client_id=client_id)
        return session_id

    def reset_session(self, client_id: str, session_id: str) -> str:
        st = self._state(client_id)
        st.sessions.pop(session_id, None)
        return self.create_session(client_id)

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
        return count

    async def enqueue(self, client_id: str, handler: Handler) -> int:
        """Enqueue a message handler. Returns queue position (0 = running now),
        or -1 if the queue is full and the message was rejected."""
        st = self._state(client_id)
        if st.pending.qsize() >= MAX_QUEUE_SIZE:
            logger.warning("Queue full for %s (%d), rejecting message", client_id, MAX_QUEUE_SIZE)
            elog("queue.full", client_id=client_id, max=MAX_QUEUE_SIZE)
            return -1
        running = 1 if self.is_busy(client_id) else 0
        position = st.pending.qsize() + running
        await st.pending.put(handler)
        if st.worker_task is None or st.worker_task.done():
            st.worker_task = asyncio.create_task(self._worker(client_id))
        return position

    async def _worker(self, client_id: str) -> None:
        st = self._clients[client_id]
        while True:
            try:
                handler = st.pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            task = asyncio.create_task(handler())
            st.current_task = task
            try:
                await task
            except asyncio.CancelledError:
                logger.info("Task cancelled for %s", client_id)
            except Exception as e:
                logger.error("Handler error for %s: %s", client_id, e)
            finally:
                st.current_task = None

    def stop_current(self, client_id: str) -> bool:
        st = self._clients.get(client_id)
        if not st or st.current_task is None or st.current_task.done():
            return False
        st.current_task.cancel()
        return True

    async def shutdown(self) -> None:
        for st in self._clients.values():
            if st.current_task and not st.current_task.done():
                st.current_task.cancel()
            if st.worker_task and not st.worker_task.done():
                st.worker_task.cancel()
        self._clients.clear()
