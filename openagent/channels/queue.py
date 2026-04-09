"""Per-user FIFO message queue with cancellation support.

Every channel (Telegram, Discord, WhatsApp) wraps incoming messages through
a ``UserQueueManager``. This guarantees that for a given user only ONE
agent run is in-flight at a time — subsequent messages queue up and are
processed sequentially. It also exposes ``stop_current`` to cancel the
running task (wired up to ``/stop`` commands and Discord/Telegram stop
buttons) and ``reset_session`` to start a fresh session id (``/new``).

State is RAM-only: survives inside the process, does not persist across
restarts. A restart drops the queue and resets all sessions.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

Handler = Callable[[], Awaitable[None]]


@dataclass
class _UserState:
    session_id: str
    pending: asyncio.Queue = field(default_factory=asyncio.Queue)
    current_task: asyncio.Task | None = None
    worker_task: asyncio.Task | None = None


class UserQueueManager:
    """Per-user message queue + cancellation.

    Usage::

        queue = UserQueueManager(platform="telegram", agent_name="my-agent")

        async def handler():
            await agent.run(..., session_id=queue.get_session_id(user_id), ...)

        position = await queue.enqueue(user_id, handler)
        # position == 0 → starts immediately
        # position >  0 → N items ahead (1 running + N-1 pending, or N pending)
    """

    def __init__(self, platform: str, agent_name: str):
        self.platform = platform
        self.agent_name = agent_name
        self._users: dict[str, _UserState] = {}

    # ── session helpers ────────────────────────────────────────────────

    def _make_session_id(self, user_id: str) -> str:
        return f"{self.platform}:{self.agent_name}:{user_id}:{uuid.uuid4().hex[:8]}"

    def _state(self, user_id: str) -> _UserState:
        st = self._users.get(user_id)
        if st is None:
            st = _UserState(session_id=self._make_session_id(user_id))
            self._users[user_id] = st
        return st

    def get_session_id(self, user_id: str) -> str:
        return self._state(user_id).session_id

    def reset_session(self, user_id: str) -> str:
        """Start a fresh session for this user. Returns the new id."""
        st = self._state(user_id)
        st.session_id = self._make_session_id(user_id)
        return st.session_id

    # ── queue state ────────────────────────────────────────────────────

    def is_busy(self, user_id: str) -> bool:
        st = self._users.get(user_id)
        if not st or st.current_task is None:
            return False
        return not st.current_task.done()

    def queue_depth(self, user_id: str) -> int:
        """Number of messages waiting (excludes the one currently running)."""
        st = self._users.get(user_id)
        return st.pending.qsize() if st else 0

    def clear_queue(self, user_id: str) -> int:
        """Drop all pending messages for a user. Returns the number removed."""
        st = self._users.get(user_id)
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

    # ── enqueue / run / cancel ─────────────────────────────────────────

    async def enqueue(self, user_id: str, handler: Handler) -> int:
        """Add a handler to the user's queue. Returns the wait position.

        Position 0 means "starts immediately" (no current task, empty queue).
        Position N means "N tasks ahead of you" (running + pending).
        """
        st = self._state(user_id)
        running = 1 if self.is_busy(user_id) else 0
        position = st.pending.qsize() + running
        await st.pending.put(handler)
        if st.worker_task is None or st.worker_task.done():
            st.worker_task = asyncio.create_task(
                self._worker(user_id),
                name=f"queue-worker-{self.platform}-{user_id}",
            )
        return position

    async def _worker(self, user_id: str) -> None:
        st = self._users[user_id]
        while True:
            try:
                handler = st.pending.get_nowait()
            except asyncio.QueueEmpty:
                return

            task = asyncio.create_task(
                handler(),
                name=f"queue-task-{self.platform}-{user_id}",
            )
            st.current_task = task
            try:
                await task
            except asyncio.CancelledError:
                logger.info("Queue task cancelled for %s/%s", self.platform, user_id)
            except Exception as e:  # noqa: BLE001
                logger.error("Queue handler error for %s/%s: %s", self.platform, user_id, e)
            finally:
                st.current_task = None

    def stop_current(self, user_id: str) -> bool:
        """Cancel the currently running task for a user.

        Returns True if a task was actually cancelled. Pending items in the
        queue are NOT touched — call ``clear_queue`` for that.
        """
        st = self._users.get(user_id)
        if not st or st.current_task is None or st.current_task.done():
            return False
        st.current_task.cancel()
        return True

    async def shutdown(self) -> None:
        """Cancel all worker and in-flight tasks. Called on channel stop()."""
        for st in list(self._users.values()):
            if st.current_task and not st.current_task.done():
                st.current_task.cancel()
            if st.worker_task and not st.worker_task.done():
                st.worker_task.cancel()
        for st in list(self._users.values()):
            for t in (st.current_task, st.worker_task):
                if t is None:
                    continue
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._users.clear()
