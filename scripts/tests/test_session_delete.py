"""Chat-session delete — cascade to sub-agent children + chat-only guard.

A manual chat is the only directly-deletable session. Deleting it must also
remove every session it spawned (delegated sub-agents at any depth, plus any
in-chat ``run dream mode`` firing) and the per-session satellite rows that have
no FK cascade — most importantly ``conversation_embeddings``, or a deleted chat
would keep surfacing in memory-search. Run / sub-agent sessions are NOT directly
deletable (they're managed in context), so the endpoint rejects them.

Covers the new ``MemoryDB.list_descendant_sessions`` / ``purge_session`` and the
rewritten ``gateway.api.sessions.handle_delete`` (guard + cascade).
"""
from __future__ import annotations

import json
import time as _time
import uuid
from pathlib import Path
from types import SimpleNamespace

from aiohttp.test_utils import make_mocked_request

from ._framework import TestContext, test


# ── helpers ──────────────────────────────────────────────────────────


async def _open_db(db_path: Path):
    from src.memory.db import MemoryDB

    db = MemoryDB(str(db_path))
    await db.connect()
    conn = await db._ensure_connected()
    await conn.execute(
        "INSERT INTO network(singleton, role, network_id, name, created_at) "
        "VALUES(1, 'coordinator', 'test-network', 'Test', ?) "
        "ON CONFLICT(singleton) DO UPDATE SET network_id=excluded.network_id",
        (_time.time(),),
    )
    await conn.commit()
    return db


async def _seed_satellites(db, sid: str) -> None:
    """Insert one row per per-session satellite table so a purge has
    something to clean (and we can assert it did)."""
    conn = await db._ensure_connected()
    now = _time.time()
    await conn.execute(
        "INSERT OR REPLACE INTO pinned_sessions (session_id, runtime_id, pinned_at) "
        "VALUES (?, ?, ?)",
        (sid, "anthropic:claude-opus", now),
    )
    await conn.execute(
        "INSERT INTO conversation_embeddings "
        "(session_id, role, text, embedding_json, model, timestamp) "
        "VALUES (?, 'user', 'hello', ?, 'text-embedding-3-small', ?)",
        (sid, json.dumps([0.1, 0.2]), now),
    )
    await conn.execute(
        "INSERT OR REPLACE INTO user_profiles "
        "(session_id, profile_json, turn_count, created_at, updated_at) "
        "VALUES (?, '{}', 1, ?, ?)",
        (sid, now, now),
    )
    await conn.execute(
        "INSERT OR REPLACE INTO vault_save_reminders "
        "(session_id, turn_count, created_at, updated_at) "
        "VALUES (?, 1, ?, ?)",
        (sid, now, now),
    )
    await conn.commit()


async def _satellite_counts(db, sid: str) -> dict[str, int]:
    conn = await db._ensure_connected()
    out: dict[str, int] = {}
    for table in (
        "pinned_sessions",
        "conversation_embeddings",
        "user_profiles",
        "vault_save_reminders",
    ):
        cur = await conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (sid,)
        )
        out[table] = (await cur.fetchone())[0]
    return out


def _stub_gateway(db, broadcasts: list):
    """A minimal gateway the handler can drive: a ``MemoryDB`` behind
    ``agent.memory_db``, an async no-op runtime ``forget_session``, a
    ``sessions`` manager with no live clients, and a sync ``broadcast_session``
    that records every (action, id) pair so the test can assert announcements."""

    async def _forget(_sid):  # runtime forget — no live runtime in the test
        return None

    sessions = SimpleNamespace(_clients={})

    async def _del(_client_id, _sid):
        return False

    sessions.delete_session = _del

    return SimpleNamespace(
        agent=SimpleNamespace(memory_db=db, forget_session=_forget),
        sessions=sessions,
        broadcast_session=lambda action, sid: broadcasts.append((action, sid)),
    )


def _delete_request(db, sid: str, broadcasts: list, *, caller: str = "alice"):
    fake_app = {"gateway": _stub_gateway(db, broadcasts)}
    req = make_mocked_request(
        "DELETE", f"/api/sessions/{sid}",
        match_info={"session_id": sid},
        app=fake_app,
    )
    req["network_id"] = "test-network"
    req["user_handle"] = caller
    req["client_id"] = f"{caller}-device"
    req["device_cert"] = SimpleNamespace(
        network_id="test-network",
        handle=caller,
        device_pubkey_hex=f"{caller}-device",
        capabilities=[],
    )
    return req


async def _insert_double_encoded(db, sid: str, meta: dict) -> None:
    """Insert a session row whose ``metadata`` column is DOUBLE-encoded — a
    JSON string of the JSON object — exactly as the runtime's
    ``serialize_session_json_fields`` path persists it. The child-lookup must
    still match these via the double ``json_extract``."""
    import time as _t

    conn = await db._ensure_connected()
    now = int(_t.time())
    await conn.execute(
        "INSERT INTO sessions "
        "(session_id, session_type, user_id, metadata, created_at, updated_at) "
        "VALUES (?, 'agent', NULL, ?, ?, ?)",
        (sid, json.dumps(json.dumps(meta)), now, now),
    )
    await conn.commit()


# ── tests ────────────────────────────────────────────────────────────


@test("session_delete", "list_descendant_sessions walks the whole lineage")
async def t_descendants(ctx: TestContext) -> None:
    tmp = ctx.db_path.with_name(f"sd-desc-{uuid.uuid4().hex[:8]}.db")
    try:
        db = await _open_db(tmp)
        chat = f"chat-{uuid.uuid4().hex[:6]}"
        sub1 = f"{chat}::sub::m::1"
        sub2 = f"{sub1}::sub::n::2"           # grandchild (delegation nests)
        dream = f"scheduler:dream-mode:{uuid.uuid4().hex[:6]}"

        await db.upsert_session(chat, client_id="alice", title="Chat")
        await db.upsert_session(sub1, client_id="alice", title="Sub",
                                parent_session_id=chat, origin="delegation")
        await db.upsert_session(sub2, client_id="alice", title="Sub-sub",
                                parent_session_id=sub1, origin="delegation")
        # In-chat "run dream mode": scheduler-origin but parent IS the chat.
        await db.upsert_session(dream, client_id="alice", title="Dream",
                                parent_session_id=chat, origin="scheduler")

        # Noise that must NOT be collected: an unrelated chat + its sub, and a
        # genuine scheduled run hanging off a SYNTHETIC task root.
        other = f"chat-{uuid.uuid4().hex[:6]}"
        await db.upsert_session(other, client_id="alice", title="Other")
        await db.upsert_session(f"{other}::sub::m::1", client_id="alice",
                                title="OtherSub", parent_session_id=other,
                                origin="delegation")
        await db.upsert_session("scheduler:task1:run1", client_id="alice",
                                title="Run", parent_session_id="scheduler:task1",
                                origin="scheduler")

        got = set(await db.list_descendant_sessions(chat))
        assert got == {sub1, sub2, dream}, got
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("session_delete", "purge_session removes the row and all satellites")
async def t_purge(ctx: TestContext) -> None:
    tmp = ctx.db_path.with_name(f"sd-purge-{uuid.uuid4().hex[:8]}.db")
    try:
        db = await _open_db(tmp)
        sid = f"chat-{uuid.uuid4().hex[:6]}"
        await db.upsert_session(sid, client_id="alice", title="Chat")
        await _seed_satellites(db, sid)

        before = await _satellite_counts(db, sid)
        assert all(v == 1 for v in before.values()), before
        assert await db.get_session(sid) is not None

        await db.purge_session(sid)

        assert await db.get_session(sid) is None
        after = await _satellite_counts(db, sid)
        assert all(v == 0 for v in after.values()), after

        # Idempotent — a second purge of a gone session must not raise.
        await db.purge_session(sid)
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("session_delete", "handle_delete rejects run / sub-agent sessions (403)")
async def t_guard(ctx: TestContext) -> None:
    from src.gateway.api import sessions as sessions_api

    tmp = ctx.db_path.with_name(f"sd-guard-{uuid.uuid4().hex[:8]}.db")
    try:
        db = await _open_db(tmp)
        chat = f"chat-{uuid.uuid4().hex[:6]}"
        await db.upsert_session(chat, client_id="alice", title="Chat")

        cases = [
            (f"{chat}::sub::m::1", chat, "delegation"),
            ("scheduler:task1:run1", "scheduler:task1", "scheduler"),
            ("workflow:wf:run:node", "workflow:wf:run", "workflow"),
        ]
        for sid, parent, origin in cases:
            await db.upsert_session(sid, client_id="alice", title=origin,
                                    parent_session_id=parent, origin=origin)
            broadcasts: list = []
            resp = await sessions_api.handle_delete(
                _delete_request(db, sid, broadcasts)
            )
            assert resp.status == 403, (origin, resp.status)
            # Rejected → nothing deleted, nothing announced.
            assert await db.get_session(sid) is not None, origin
            assert broadcasts == [], (origin, broadcasts)
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("session_delete", "handle_delete cascades a chat to its sub-agents")
async def t_cascade(ctx: TestContext) -> None:
    from src.gateway.api import sessions as sessions_api

    tmp = ctx.db_path.with_name(f"sd-cascade-{uuid.uuid4().hex[:8]}.db")
    try:
        db = await _open_db(tmp)
        chat = f"chat-{uuid.uuid4().hex[:6]}"
        sub1 = f"{chat}::sub::m::1"
        sub2 = f"{sub1}::sub::n::2"
        dream = f"scheduler:dream-mode:{uuid.uuid4().hex[:6]}"
        ids = [chat, sub1, sub2, dream]

        await db.upsert_session(chat, client_id="alice", title="Chat")
        await db.upsert_session(sub1, client_id="alice", title="Sub",
                                parent_session_id=chat, origin="delegation")
        await db.upsert_session(sub2, client_id="alice", title="Sub-sub",
                                parent_session_id=sub1, origin="delegation")
        await db.upsert_session(dream, client_id="alice", title="Dream",
                                parent_session_id=chat, origin="scheduler")
        for sid in ids:
            await _seed_satellites(db, sid)

        # A bystander chat that must survive untouched.
        bystander = f"chat-{uuid.uuid4().hex[:6]}"
        await db.upsert_session(bystander, client_id="alice", title="Keep")

        broadcasts: list = []
        resp = await sessions_api.handle_delete(
            _delete_request(db, chat, broadcasts)
        )
        assert resp.status == 200, resp.status
        body = json.loads(resp.body.decode())
        assert body["deleted"] is True, body
        assert body["deleted_count"] == 4, body
        assert set(body["deleted_session_ids"]) == set(ids), body

        # Every targeted row + its satellites are gone.
        for sid in ids:
            assert await db.get_session(sid) is None, sid
            assert all(v == 0 for v in (await _satellite_counts(db, sid)).values()), sid
        # Every removal was announced so other clients prune live.
        assert {sid for _, sid in broadcasts} == set(ids), broadcasts
        # The bystander is untouched.
        assert await db.get_session(bystander) is not None
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("session_delete", "cascade matches double-encoded child metadata")
async def t_double_encoded(ctx: TestContext) -> None:
    """The runtime re-persists a child row's metadata DOUBLE-encoded (a JSON
    string of the dict). The cascade's child lookup must use the double
    ``json_extract`` so such rows are still collected — a single extract reads
    NULL and would orphan the sub-agent. Guards against that regression."""
    tmp = ctx.db_path.with_name(f"sd-dbl-{uuid.uuid4().hex[:8]}.db")
    try:
        db = await _open_db(tmp)
        chat = f"chat-{uuid.uuid4().hex[:6]}"
        sub_plain = f"{chat}::sub::m::1"
        sub_dbl = f"{chat}::sub::m::2"
        await db.upsert_session(chat, client_id="alice", title="Chat")
        # One child written the normal (single-encoded) way…
        await db.upsert_session(sub_plain, client_id="alice", title="Sub1",
                                parent_session_id=chat, origin="delegation")
        # …and one written the runtime's double-encoded way.
        await _insert_double_encoded(db, sub_dbl, {
            "parent_session_id": chat, "origin": "delegation", "client_id": "alice",
        })

        got = set(await db.list_descendant_sessions(chat))
        assert got == {sub_plain, sub_dbl}, got  # BOTH must be found
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("session_delete", "handle_delete authorizes by owner handle")
async def t_authz(ctx: TestContext) -> None:
    from src.gateway.api import sessions as sessions_api

    tmp = ctx.db_path.with_name(f"sd-authz-{uuid.uuid4().hex[:8]}.db")
    try:
        db = await _open_db(tmp)

        # A chat owned by alice cannot be deleted by bob → 404 (no existence
        # leak), and the row survives.
        owned = f"chat-{uuid.uuid4().hex[:6]}"
        await db.upsert_session(owned, client_id="alice", title="Alice's chat")
        b1: list = []
        resp = await sessions_api.handle_delete(
            _delete_request(db, owned, b1, caller="bob")
        )
        assert resp.status == 404, resp.status
        assert await db.get_session(owned) is not None
        assert b1 == []

        # The owner alice can delete it.
        b2: list = []
        resp = await sessions_api.handle_delete(
            _delete_request(db, owned, b2, caller="alice")
        )
        assert resp.status == 200, resp.status
        assert await db.get_session(owned) is None

        # A legacy/unowned row has no canonical ACL identity and therefore
        # fails closed instead of being claimable/deletable by a guessed id.
        unowned = f"chat-{uuid.uuid4().hex[:6]}"
        await _insert_double_encoded(db, unowned, {"title": "no owner"})
        b3: list = []
        resp = await sessions_api.handle_delete(
            _delete_request(db, unowned, b3, caller="bob")
        )
        assert resp.status == 404, resp.status
        assert await db.get_session(unowned) is not None
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("session_delete", "cascade cannot delete a foreign-owned descendant")
async def t_cascade_rechecks_each_child_acl(ctx: TestContext) -> None:
    from src.gateway.api import sessions as sessions_api

    tmp = ctx.db_path.with_name(f"sd-child-acl-{uuid.uuid4().hex[:8]}.db")
    try:
        db = await _open_db(tmp)
        parent = f"chat-{uuid.uuid4().hex[:6]}"
        foreign_child = f"{parent}::sub::foreign"
        await db.upsert_session(parent, client_id="alice", title="Alice parent")
        await db.upsert_session(
            foreign_child,
            client_id="bob",
            title="Bob child",
            parent_session_id=parent,
            origin="delegation",
        )
        broadcasts: list = []
        response = await sessions_api.handle_delete(
            _delete_request(db, parent, broadcasts, caller="alice")
        )
        assert response.status == 404
        assert await db.get_session(parent) is not None
        assert await db.get_session(foreign_child) is not None
        assert broadcasts == []
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
