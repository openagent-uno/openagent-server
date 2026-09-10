"""A session with no owner is a session nobody can reach.

Ownership lives in ``sessions.metadata.client_id``. ``list_all_sessions``
filters on exactly that field, and since the normalized resource rows arrived
the projection files an ownerless row as ``quarantined`` — a visibility
``resource_is_visible`` refuses unconditionally, so every per-session route
answers 404 for a chat that is plainly on disk. Nothing can repair it through
the API either: claiming a session is itself gated on the session being
visible.

Three ways a row lost its owner, all covered here:

* ``POST /api/chat`` never stamped one. The WebSocket gateway always has
  (``SessionManager._persist_session``); the REST path created the session and
  wrote nothing, so every REST-driven chat was born invisible.
* The runtime erased it. Its session object carries no gateway metadata, and
  the store wrote that ``None`` straight over the column on the next turn.
* It predates the stamp entirely, and no live path will ever touch it again.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from ._framework import TestContext, test


class _FakeDB:
    """The two calls the owner stamp makes, and what the row looks like."""

    def __init__(self, existing=None):
        self.rows = dict(existing or {})
        self.calls: list[dict] = []

    async def get_session(self, session_id):
        return self.rows.get(session_id)

    async def upsert_session(self, session_id, **kwargs):
        self.calls.append({"session_id": session_id, **kwargs})
        row = self.rows.setdefault(session_id, {"session_id": session_id, "client_id": ""})
        if kwargs.get("client_id"):
            row["client_id"] = kwargs["client_id"]


class _FakeStreamSession:
    def __init__(self, *_args, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.handle = kwargs.get("handle")
        self.started = False

    async def start(self):
        self.started = True


def _gateway(db):
    return SimpleNamespace(
        agent=SimpleNamespace(memory_db=db),
        _make_stream_pre_dispatch_hook=lambda _client, _session: None,
        _make_stream_post_turn_hook=lambda: None,
    )


async def _create(db, *, session_id="rest-chat", client_id="ios-rest", handle="marco"):
    from src.gateway.api import chat as chat_api
    from src.stream import session as stream_module

    chat_api._sessions.clear()
    real = stream_module.StreamSession
    stream_module.StreamSession = _FakeStreamSession
    try:
        return await chat_api._get_or_create_session(
            _gateway(db), client_id, session_id, handle=handle,
        )
    finally:
        stream_module.StreamSession = real
        chat_api._sessions.clear()


@test("session_owner_recovery", "a REST chat records its owner when it is created")
async def t_rest_chat_stamps_owner(_ctx: TestContext) -> None:
    db = _FakeDB()
    session, _lock = await _create(db)
    assert session.started
    # The handle, not the device key: the listing must be the same on every
    # device that user signs in from. The device stays for per-device routing.
    assert db.calls == [{
        "session_id": "rest-chat", "client_id": "marco", "device_id": "ios-rest",
    }], db.calls


@test("session_owner_recovery", "a session with no handle falls back to the device")
async def t_owner_falls_back_to_device(_ctx: TestContext) -> None:
    db = _FakeDB()
    await _create(db, handle=None)
    assert db.calls[0]["client_id"] == "ios-rest", db.calls


@test("session_owner_recovery", "an existing owner is never transferred")
async def t_existing_owner_is_kept(_ctx: TestContext) -> None:
    db = _FakeDB({"rest-chat": {"session_id": "rest-chat", "client_id": "giulia"}})
    await _create(db, handle="marco")
    assert db.calls == [], "attaching to someone else's session must not take it over"


@test("session_owner_recovery", "a runtime turn no longer erases the stored owner")
async def t_runtime_write_keeps_metadata(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.memory.sessions import AgentSession
    from src.memory.store.sqlite import SqliteDb

    with TemporaryDirectory(prefix="openagent-session-owner-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        await db.upsert_session("owned-chat", client_id="marco", title="Ciao")
        await db.close()

        now = int(time.time())
        runtime = SqliteDb(db_file=str(path), session_table="sessions")
        try:
            # Exactly what a turn persists: the runtime's own session object,
            # which has never read the gateway's metadata and carries none.
            assert runtime.upsert_session(AgentSession(
                session_id="owned-chat", agent_id="agent", user_id="openagent",
                metadata=None, created_at=now, updated_at=now,
            )) is not None
        finally:
            runtime.close()

        conn = sqlite3.connect(path)
        try:
            stored = conn.execute(
                "SELECT metadata FROM sessions WHERE session_id='owned-chat'"
            ).fetchone()
            owner, visibility = conn.execute(
                "SELECT owner_principal_id, visibility FROM sessions_v2 "
                "WHERE id='owned-chat'"
            ).fetchone()
        finally:
            conn.close()
        metadata = json.loads(stored[0]) if stored and stored[0] else {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert metadata.get("client_id") == "marco", metadata
        assert metadata.get("title") == "Ciao", "the whole column survives, not just the owner"
        assert (owner, visibility) == ("user:marco", "private"), (owner, visibility)


def _seed_ownerless(path: Path, *owners: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, session_type, user_id, metadata, "
            "created_at, updated_at) VALUES ('legacy-chat', 'agent', NULL, NULL, ?, ?)",
            (int(time.time()), int(time.time())),
        )
        for index, owner in enumerate(owners):
            conn.execute(
                "INSERT INTO network_users (handle, pake_record, pake_algo, status, "
                "created_at) VALUES (?, X'00', 'srp6a', 'active', ?)",
                (owner, time.time() + index),
            )
        conn.commit()
    finally:
        conn.close()


async def _reconnect_and_read(path: Path):
    from src.memory.db import MemoryDB

    db = MemoryDB(str(path))
    await db.connect()
    await db.close()
    conn = sqlite3.connect(path)
    try:
        metadata = conn.execute(
            "SELECT metadata FROM sessions WHERE session_id='legacy-chat'"
        ).fetchone()[0]
        canonical = conn.execute(
            "SELECT owner_principal_id, visibility FROM sessions_v2 WHERE id='legacy-chat'"
        ).fetchone()
    finally:
        conn.close()
    return json.loads(metadata) if metadata else {}, canonical


@test("session_owner_recovery", "sessions written before the stamp are claimed on connect")
async def t_connect_claims_ownerless_rows(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-session-claim-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        await db.close()
        _seed_ownerless(path, "marco")

        metadata, canonical = await _reconnect_and_read(path)
        assert metadata.get("client_id") == "marco", metadata
        # The canonical row was projected from ownerless metadata and is
        # quarantined; the repair has to reach it too, or the chat stays
        # unreadable while looking perfectly present on disk.
        assert canonical == ("user:marco", "private"), canonical


@test("session_owner_recovery", "with no owner on the deployment nothing is claimed")
async def t_no_owner_claims_nothing(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-session-noclaim-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        await db.close()
        _seed_ownerless(path)

        metadata, _canonical = await _reconnect_and_read(path)
        # Handing an ownerless chat to whoever connects next would be worse
        # than leaving it hidden.
        assert "client_id" not in metadata, metadata


@test("session_owner_recovery", "a shared network is left alone rather than guessed at")
async def t_several_users_claim_nothing(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-session-shared-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        await db.close()
        # The row names nobody, and here "nobody" could be either of them.
        _seed_ownerless(path, "marco", "giulia")

        metadata, _canonical = await _reconnect_and_read(path)
        assert "client_id" not in metadata, metadata


@test("session_owner_recovery", "claiming twice changes nothing")
async def t_claim_is_idempotent(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-session-idem-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        await db.close()
        _seed_ownerless(path, "marco")
        await _reconnect_and_read(path)
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                "UPDATE sessions SET metadata = json_set(json(metadata), "
                "'$.title', 'Renamed') WHERE session_id='legacy-chat'"
            )
            conn.commit()
        finally:
            conn.close()

        metadata, canonical = await _reconnect_and_read(path)
        assert metadata == {"client_id": "marco", "title": "Renamed"}, metadata
        assert canonical == ("user:marco", "private"), canonical
