"""Per-session FIFO message queue with concurrent execution across sessions.

Each client can host multiple chat sessions. Each session owns its own
FIFO queue and worker task: messages arriving for one session execute
in order, but different sessions on the same client run in parallel.
This matches the user-facing model where two chat tabs in the app (or
two telegram users multiplexed onto one bridge client) shouldn't block
each other — prior to this design all messages from a client serialised
through a single queue, defeating the per-session locks in ClaudeCLI
and NativeProvider.

Stop/clear semantics stay session-scoped: ``/stop`` from one tab
cancels only that tab's in-flight turn; the legacy unscoped form used
by admin shutdowns still walks every session. Session metadata is
persisted to the MemoryDB ``chat_sessions`` table when a DB reference
is wired, so the session list survives process restarts.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from src.core.logging import elog


Handler = Callable[[], Awaitable[None]]

# Maximum pending messages per SESSION before new messages are rejected.
# Applies independently to each session on a client so one noisy chat
# can't starve its siblings. A bridge client with many users therefore
# has its total backpressure = MAX_QUEUE_SIZE × len(session_states).
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
class _SessionState:
    """Per-session runtime: its own FIFO queue and worker task.

    One handler runs at a time per session (so message ordering within
    a single conversation is preserved). Different sessions' workers
    run concurrently under the asyncio scheduler, giving true parallel
    execution across independent chats on one client.
    """
    session_id: str
    client_id: str
    pending: asyncio.Queue = field(default_factory=asyncio.Queue)
    current_task: asyncio.Task | None = None
    worker_task: asyncio.Task | None = None


@dataclass
class _ClientState:
    sessions: dict[str, Session] = field(default_factory=dict)
    # Runtime queue/worker state keyed by session_id. Separate from
    # ``sessions`` (metadata) so callers that only touch metadata —
    # list_sessions, bind_history_mode, create_session — don't
    # materialise worker state until an actual message is enqueued.
    session_states: dict[str, _SessionState] = field(default_factory=dict)


class SessionManager:
    """Manages sessions and per-session message queues per client.

    When a MemoryDB reference is wired via ``set_db()``, session metadata
    is persisted to the ``chat_sessions`` table so the session list
    survives process restarts. Without a DB, behaviour is identical to
    prior releases (RAM-only, lost on restart).
    """

    def __init__(self, agent_name: str = "agent"):
        self.agent_name = agent_name
        self._db: Any = None
        self._clients: dict[str, _ClientState] = {}

    def set_db(self, db: Any) -> None:
        self._db = db

    def _fire_and_forget(self, coro) -> None:
        """Schedule a DB write on the running loop without blocking the caller."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(coro)

    def _state(self, client_id: str) -> _ClientState:
        if client_id not in self._clients:
            self._clients[client_id] = _ClientState()
        return self._clients[client_id]

    def _session_state(self, client_id: str, session_id: str) -> _SessionState:
        """Get or create the per-session runtime state."""
        st = self._state(client_id)
        ss = st.session_states.get(session_id)
        if ss is None:
            ss = _SessionState(session_id=session_id, client_id=client_id)
            st.session_states[session_id] = ss
        return ss

    # ── session persistence helpers ──

    def _persist_session(
        self,
        session_id: str,
        client_id: str,
        title: str | None = None,
        *,
        handle: str | None = None,
    ) -> None:
        """Upsert the session metadata row.

        ``handle`` (the authenticated user identity, same across all of
        their devices) is what we record as the row's primary owner
        when present, so the chat list is cross-device. ``client_id``
        is the device pubkey — kept in metadata as ``device_id`` for
        per-device routing (sticky-device retries, WS reconnection),
        but the owner column tracks the user.
        """
        if self._db is None:
            return
        owner = (handle or "").strip() or client_id
        self._fire_and_forget(
            self._db.upsert_session(
                session_id, client_id=owner, title=title, device_id=client_id,
            )
        )

    def _remove_session_from_db(self, session_id: str) -> None:
        if self._db is None:
            return
        self._fire_and_forget(self._db.delete_session(session_id))

    # ── sessions ──

    def create_session(self, client_id: str, *, handle: str | None = None) -> str:
        sid = f"{self.agent_name}:{client_id}:{uuid.uuid4().hex[:8]}"
        self._state(client_id).sessions[sid] = Session(id=sid, client_id=client_id)
        self._persist_session(sid, client_id, handle=handle)
        elog(
            "session.create",
            client_id=client_id, session_id=sid, handle=handle,
        )
        return sid

    def get_or_create_session(
        self, client_id: str, session_id: str, *, handle: str | None = None,
    ) -> str:
        st = self._state(client_id)
        created = False
        if session_id not in st.sessions:
            st.sessions[session_id] = Session(id=session_id, client_id=client_id)
            self._persist_session(session_id, client_id, handle=handle)
            created = True
        elog(
            "session.attach",
            client_id=client_id, session_id=session_id,
            created=created, handle=handle,
        )
        return session_id

    def reset_session(self, client_id: str, session_id: str) -> str:
        st = self._state(client_id)
        st.sessions.pop(session_id, None)
        self._remove_session_from_db(session_id)
        ss = st.session_states.pop(session_id, None)
        if ss is not None and ss.worker_task is not None and not ss.worker_task.done():
            ss.worker_task.cancel()
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

    async def list_sessions_persisted(self, client_id: str) -> list[dict]:
        """Return sessions for ``client_id``, merging RAM + DB.

        RAM entries carry live queue/busy state; DB entries fill in
        sessions that existed before this process started. Results
        are deduplicated by session_id.
        """
        ram_sids = set(self.list_sessions(client_id))
        results: dict[str, dict] = {}
        for sid in ram_sids:
            results[sid] = {
                "session_id": sid,
                "client_id": client_id,
                "title": None,
                "model": None,
                "framework": None,
                "created_at": None,
                "last_active_at": None,
            }

        if self._db is not None:
            try:
                db_rows = await self._db.list_all_sessions(client_id, limit=200)
                for row in db_rows:
                    sid = row.get("session_id", "")
                    if sid in results:
                        # Enrich RAM entry with DB metadata
                        results[sid].update({k: v for k, v in row.items() if v is not None})
                    else:
                        results[sid] = {
                            "session_id": sid,
                            "client_id": row.get("client_id", client_id),
                            "title": row.get("title"),
                            "model": row.get("model"),
                            "framework": row.get("framework"),
                            "created_at": row.get("created_at"),
                            "last_active_at": row.get("last_active_at"),
                        }
            except Exception:
                pass

        return sorted(
            results.values(),
            key=lambda r: r.get("last_active_at") or 0,
            reverse=True,
        )

    async def delete_session(self, client_id: str, session_id: str) -> bool:
        """Delete one session from RAM and DB. Returns True if anything was removed."""
        deleted = False
        st = self._clients.get(client_id)
        if st:
            st.sessions.pop(session_id, None)
            ss = st.session_states.pop(session_id, None)
            if ss is not None:
                if ss.current_task and not ss.current_task.done():
                    ss.current_task.cancel()
                if ss.worker_task and not ss.worker_task.done():
                    ss.worker_task.cancel()
                deleted = True
        if self._db is not None:
            try:
                await self._db.delete_session(session_id)
                deleted = True
            except Exception:
                pass
        if deleted:
            elog("session.delete", client_id=client_id, session_id=session_id)
        return deleted

    # ── queue ──

    def is_busy(self, client_id: str) -> bool:
        """Any session on this client currently running a handler."""
        st = self._clients.get(client_id)
        if not st:
            return False
        return any(
            ss.current_task is not None and not ss.current_task.done()
            for ss in st.session_states.values()
        )

    def is_busy_session(self, client_id: str, session_id: str) -> bool:
        """True iff this specific session has a running handler."""
        st = self._clients.get(client_id)
        if not st:
            return False
        ss = st.session_states.get(session_id)
        if ss is None or ss.current_task is None:
            return False
        return not ss.current_task.done()

    def queue_depth(self, client_id: str) -> int:
        """Sum of all sessions' pending queue depths on this client."""
        st = self._clients.get(client_id)
        if not st:
            return 0
        return sum(ss.pending.qsize() for ss in st.session_states.values())

    def queue_depth_for_session(self, client_id: str, session_id: str) -> int:
        st = self._clients.get(client_id)
        if not st:
            return 0
        ss = st.session_states.get(session_id)
        return ss.pending.qsize() if ss else 0

    def clear_queue(self, client_id: str) -> int:
        """Drain every session's pending queue on this client."""
        st = self._clients.get(client_id)
        if not st:
            return 0
        total = 0
        for ss in st.session_states.values():
            while True:
                try:
                    ss.pending.get_nowait()
                    total += 1
                except asyncio.QueueEmpty:
                    break
        if total:
            elog("queue.clear", client_id=client_id, removed=total)
        return total

    async def enqueue(
        self,
        client_id: str,
        handler: Handler,
        session_id: str | None = None,
    ) -> int:
        """Enqueue a message handler for a specific session.

        Returns queue position (0 = running now), or -1 if this
        session's queue is full and the message was rejected.

        ``session_id=None`` falls back to a shared pseudo-session so
        callers that never attached a session still get serialised
        work (rare in practice — the gateway always passes a real
        session_id). The cap applies per session.
        """
        sid = session_id or f"__anon__:{client_id}"
        ss = self._session_state(client_id, sid)
        if ss.pending.qsize() >= MAX_QUEUE_SIZE:
            elog(
                "queue.full", level="warning",
                client_id=client_id, session_id=session_id, max=MAX_QUEUE_SIZE,
            )
            return -1
        running_here = (
            1
            if ss.current_task is not None and not ss.current_task.done()
            else 0
        )
        position = ss.pending.qsize() + running_here
        await ss.pending.put(_QueuedItem(handler=handler, session_id=session_id))
        elog(
            "queue.enqueue",
            client_id=client_id,
            session_id=session_id,
            position=position,
            depth=ss.pending.qsize(),
            busy=bool(running_here),
        )
        if ss.worker_task is None or ss.worker_task.done():
            ss.worker_task = asyncio.create_task(self._session_worker(ss))
        return position

    async def _session_worker(self, ss: _SessionState) -> None:
        """Drain one session's queue, one handler at a time.

        Each session has its own worker so different sessions on the
        same client run in parallel under the asyncio scheduler.
        Within a session, ordering is preserved (FIFO) so a user's
        messages are replied to in order.
        """
        client_id = ss.client_id
        while True:
            try:
                item = ss.pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            elog(
                "queue.start",
                client_id=client_id,
                session_id=item.session_id,
                remaining=ss.pending.qsize(),
            )
            task = asyncio.create_task(item.handler())
            ss.current_task = task
            try:
                await task
            except asyncio.CancelledError:
                elog("queue.cancel", client_id=client_id, session_id=item.session_id)
            except Exception as e:
                elog(
                    "queue.error", level="error",
                    client_id=client_id, session_id=item.session_id, error=str(e),
                )
            finally:
                ss.current_task = None
                elog(
                    "queue.done",
                    client_id=client_id,
                    session_id=item.session_id,
                    remaining=ss.pending.qsize(),
                )

    def stop_current(self, client_id: str, session_id: str | None = None) -> bool:
        """Cancel the currently-running handler(s) on this client.

        With ``session_id``: cancel only that session's handler — used
        by bridges (telegram, discord) that multiplex many users onto
        one ``client_id``, and by app chat tabs where each tab owns a
        distinct session, so ``/stop`` from user/tab A must not cancel
        user/tab B's turn.

        Without ``session_id``: cancel every session's current handler
        on this client. Kept for admin shutdowns and single-session
        direct ws clients that never scope their /stop.
        """
        st = self._clients.get(client_id)
        if not st:
            return False
        if session_id is not None:
            ss = st.session_states.get(session_id)
            if ss is None or ss.current_task is None or ss.current_task.done():
                return False
            ss.current_task.cancel()
            elog("queue.stop_requested", client_id=client_id, session_id=session_id)
            return True
        # Unscoped: cancel everything running on this client.
        cancelled_any = False
        for ss in st.session_states.values():
            if ss.current_task is not None and not ss.current_task.done():
                ss.current_task.cancel()
                cancelled_any = True
        if cancelled_any:
            elog("queue.stop_requested", client_id=client_id, session_id=None)
        return cancelled_any

    def clear_queue_for_session(self, client_id: str, session_id: str) -> int:
        """Drop queued handlers for one session, leaving siblings alone."""
        st = self._clients.get(client_id)
        if not st:
            return 0
        ss = st.session_states.get(session_id)
        if ss is None:
            return 0
        removed = 0
        while True:
            try:
                ss.pending.get_nowait()
                removed += 1
            except asyncio.QueueEmpty:
                break
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
            for ss in st.session_states.values():
                if ss.current_task and not ss.current_task.done():
                    ss.current_task.cancel()
                    tasks.append(ss.current_task)
                if ss.worker_task and not ss.worker_task.done():
                    ss.worker_task.cancel()
                    tasks.append(ss.worker_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._clients.clear()
