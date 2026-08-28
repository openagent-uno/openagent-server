"""Cross-device session visibility — handle-keyed listing + legacy fallback.

Pre-fix: the gateway filtered ``sessions`` by the WebSocket's
``client_id`` = device pubkey hex. Every device a user owned had a
different pubkey, so device B logged in as the same user could never
see device A's chats. The fix has two halves:

1. ``upsert_session`` now writes the user handle into ``metadata.client_id``
   so the row's primary owner is user-scoped.
2. ``list_all_sessions`` accepts rows whose ``metadata.client_id`` is a
   device pubkey bound to the same handle in ``network_devices`` —
   a soft-fallback for sessions persisted before the fix landed.
"""
from __future__ import annotations

import time as _time
import uuid
from types import SimpleNamespace

from ._framework import TestContext, test


async def _seed_device_binding(db, handle: str, pubkey_hex: str) -> None:
    """Insert a ``network_devices`` row binding a device pubkey to handle."""
    conn = await db._ensure_connected()
    await conn.execute(
        "INSERT OR REPLACE INTO network_users "
        "(handle, pake_record, pake_algo, status, created_at) "
        "VALUES (?, ?, 'srp6a', 'active', ?)",
        (handle, b"", _time.time()),
    )
    await conn.execute(
        "INSERT OR REPLACE INTO network_devices "
        "(device_pubkey, user_handle, label, status, added_at) "
        "VALUES (?, ?, ?, 'active', ?)",
        (bytes.fromhex(pubkey_hex), handle, "test-device", _time.time()),
    )
    await conn.commit()


@test("sessions_cross_device", "list_all_sessions matches by handle")
async def t_match_by_handle(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-handle-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        sid = f"xd-{uuid.uuid4().hex[:8]}"
        await db.upsert_session(sid, client_id="alice", title="Hi", framework="api-based")
        rows = await db.list_all_sessions("alice", limit=50)
        assert any(r["session_id"] == sid for r in rows), rows
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("sessions_cross_device", "list_all_sessions hides delegation children when asked")
async def t_exclude_delegation_children(ctx: TestContext) -> None:
    """The flat history list (``GET /api/sessions``) passes
    ``exclude_child_origins=('delegation',)`` so a delegated sub-agent never
    shows in the sidebar — it's navigable only from its parent's transcript
    card (which uses ``list_child_sessions``, unaffected). Chats and other
    child origins (scheduler / workflow) stay visible. NULL-safe: a legacy
    chat row with no ``origin`` is always kept."""
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-excl-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        await db.upsert_session("chat-A", client_id="alice", title="Chat")  # no origin (legacy)
        await db.upsert_session(
            "chat-A::member::opus::ab12", client_id="alice", title="Sub",
            parent_session_id="chat-A", origin="delegation", kind="opus",
        )
        await db.upsert_session(
            "scheduler:t1:r1", client_id="alice", title="Sched",
            parent_session_id="scheduler:t1", origin="scheduler", kind="t1",
        )
        full = {r["session_id"] for r in await db.list_all_sessions("alice", limit=50)}
        hidden = {
            r["session_id"]
            for r in await db.list_all_sessions(
                "alice", limit=50, exclude_child_origins=("delegation",),
            )
        }
        # Default behaviour unchanged: the child is still inheritable/visible.
        assert "chat-A::member::opus::ab12" in full, full
        # With the exclusion the sub-agent is gone; chat + scheduler remain.
        assert "chat-A::member::opus::ab12" not in hidden, hidden
        assert {"chat-A", "scheduler:t1:r1"} <= hidden, hidden
        # The parent's own card list still sees the child (different query).
        child_ids = {c["session_id"] for c in await db.list_child_sessions("chat-A")}
        assert "chat-A::member::opus::ab12" in child_ids, child_ids
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("sessions_cross_device", "flat list hides every spawned child origin")
async def t_exclude_all_child_origins(ctx: TestContext) -> None:
    """``GET /api/sessions`` now passes ``HIDDEN_CHILD_ORIGINS`` (delegation +
    scheduler + workflow), so a scheduled firing and a workflow node are hidden
    from the sidebar too — each navigable only from its run's execution screen.
    Only chat sessions (and legacy no-origin rows) remain in the flat list."""
    from src.memory.db import MemoryDB
    from src.core.child_session import HIDDEN_CHILD_ORIGINS

    tmp_db = ctx.db_path.with_name(f"xd-excl-all-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        await db.upsert_session("chat-A", client_id="alice", title="Chat")  # legacy, no origin
        await db.upsert_session("chat-B", client_id="alice", title="Chat 2", origin="chat")
        await db.upsert_session(
            "chat-A::member::opus::ab12", client_id="alice", title="Sub",
            parent_session_id="chat-A", origin="delegation", kind="opus",
        )
        await db.upsert_session(
            "scheduler:t1:r1", client_id="alice", title="Sched",
            parent_session_id="scheduler:t1", origin="scheduler", kind="t1",
        )
        await db.upsert_session(
            "workflow:wf1:run1:node1", client_id="alice", title="WF node",
            parent_session_id="workflow:wf1:run1", origin="workflow", kind="wf1:node1",
        )
        visible = {
            r["session_id"]
            for r in await db.list_all_sessions(
                "alice", limit=50, exclude_child_origins=HIDDEN_CHILD_ORIGINS,
            )
        }
        # Only the chats survive the flat-list filter.
        assert visible == {"chat-A", "chat-B"}, visible
        # …but each hidden child is still reachable via its own queries:
        assert {c["session_id"] for c in await db.list_child_sessions("scheduler:t1")} == {"scheduler:t1:r1"}
        assert (await db.list_session_runs("workflow:wf1:run1:node1")) == []  # no runs yet, but row exists
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("sessions_cross_device", "GET /api/sessions marks only genuinely active live sessions")
async def t_rest_sessions_live_flag_uses_gateway_active_state(ctx: TestContext) -> None:
    """The app uses ``_live: false`` to clear stale local processing/reasoning
    flags. The REST list must therefore consult the gateway's real stream-turn
    state, not legacy attached-session RAM."""
    import json
    from aiohttp.test_utils import make_mocked_request
    from src.gateway.api.sessions import handle_list
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-live-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        sid_live = f"live-{uuid.uuid4().hex[:8]}"
        sid_done = f"done-{uuid.uuid4().hex[:8]}"
        await db.upsert_session(sid_live, client_id="alice", title="Live")
        await db.upsert_session(sid_done, client_id="alice", title="Done")

        class _Gateway:
            def __init__(self):
                self.agent = SimpleNamespace(memory_db=db)
                self.calls: list[tuple[str | None, str | None]] = []

            async def active_live_session_ids(self, *, client_id, handle):
                self.calls.append((client_id, handle))
                return {sid_live}

        gateway = _Gateway()
        req = make_mocked_request(
            "GET",
            "/api/sessions",
            app={"gateway": gateway},
        )
        req["client_id"] = "device-pubkey"
        req["user_handle"] = "alice"
        conn = await db._ensure_connected()
        tenant_row = await (
            await conn.execute(
                "SELECT tenant_id FROM sessions_v2 WHERE id=?", (sid_live,)
            )
        ).fetchone()
        tenant = str(tenant_row["tenant_id"])
        req["network_id"] = tenant
        req["device_cert"] = SimpleNamespace(
            network_id=tenant,
            handle="alice",
            device_pubkey_hex="device-pubkey",
            capabilities=[],
        )

        resp = await handle_list(req)
        data = json.loads(resp.text)
        live_by_id = {row["session_id"]: row.get("_live") for row in data["sessions"]}

        assert gateway.calls == [("device-pubkey", "alice")]
        assert live_by_id[sid_live] is True, live_by_id
        assert live_by_id[sid_done] is False, live_by_id
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("sessions_cross_device", "child-session list enforces parent and child ACLs")
async def t_rest_child_sessions_are_acl_scoped(ctx: TestContext) -> None:
    """A legacy ``parent_session_id`` match is discovery only.

    A caller must be able to view the canonical parent, and each returned
    child must independently be owned by or granted to that caller.  Forging
    the legacy ``client_id`` query parameter must not change either decision.
    """
    import json
    from aiohttp.test_utils import make_mocked_request
    from src.gateway.api.sessions import handle_list
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-child-acl-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    try:
        await db.connect()
        parent = f"parent-{uuid.uuid4().hex[:8]}"
        alice_child = f"alice-child-{uuid.uuid4().hex[:8]}"
        bob_child = f"bob-child-{uuid.uuid4().hex[:8]}"
        granted_child = f"granted-child-{uuid.uuid4().hex[:8]}"
        await db.upsert_session(parent, client_id="alice", title="Alice parent")
        await db.upsert_session(
            alice_child,
            client_id="alice",
            title="Alice child",
            parent_session_id=parent,
            origin="delegation",
        )
        await db.upsert_session(
            bob_child,
            client_id="bob",
            title="Bob child",
            parent_session_id=parent,
            origin="delegation",
        )
        await db.upsert_session(
            granted_child,
            client_id="charlie",
            title="Granted child",
            parent_session_id=parent,
            origin="delegation",
        )
        conn = await db._ensure_connected()
        parent_row = await (
            await conn.execute(
                "SELECT tenant_id FROM sessions_v2 WHERE id=?", (parent,)
            )
        ).fetchone()
        tenant = str(parent_row["tenant_id"])
        await conn.execute(
            "INSERT INTO resource_acl "
            "(tenant_id,resource_type,resource_id,principal_type,principal_id,"
            "permission,acl_version,granted_by_principal_id,granted_at_ms) "
            "VALUES(?,?,?,?,?,'view',1,?,?)",
            (
                tenant,
                "session",
                granted_child,
                "user",
                "alice",
                "user:charlie",
                int(_time.time() * 1000),
            ),
        )
        await conn.commit()

        class _Gateway:
            def __init__(self):
                self.agent = SimpleNamespace(memory_db=db)

            async def active_live_session_ids(self, *, client_id, handle):
                return set()

        gateway = _Gateway()

        def _request(handle: str, *, parent_id: str = parent):
            # The forged query value is intentional: canonical certificate
            # identity, never this client-supplied filter, must drive ACLs.
            request = make_mocked_request(
                "GET",
                f"/api/sessions?parent={parent_id}&client_id=alice",
                app={"gateway": gateway},
            )
            device = f"{handle}-device"
            request["network_id"] = tenant
            request["user_handle"] = handle
            request["client_id"] = device
            request["device_cert"] = SimpleNamespace(
                network_id=tenant,
                handle=handle,
                device_pubkey_hex=device,
                capabilities=[],
            )
            return request

        alice_response = await handle_list(_request("alice"))
        assert alice_response.status == 200
        alice_ids = {
            row["session_id"]
            for row in json.loads(alice_response.text)["sessions"]
        }
        assert alice_ids == {alice_child, granted_child}, alice_ids
        assert bob_child not in alice_response.text

        # Bob knows the id and forges Alice's legacy list filter, but cannot
        # view Alice's private parent.  A missing id is indistinguishable.
        guessed = await handle_list(_request("bob"))
        missing = await handle_list(
            _request("bob", parent_id=f"missing-{uuid.uuid4().hex[:8]}")
        )
        assert guessed.status == 404
        assert missing.status == 404
        assert json.loads(guessed.text) == json.loads(missing.text)

        # Once Bob is explicitly allowed to view the parent, the endpoint
        # still filters each child independently: only Bob's own child passes.
        await conn.execute(
            "INSERT INTO resource_acl "
            "(tenant_id,resource_type,resource_id,principal_type,principal_id,"
            "permission,acl_version,granted_by_principal_id,granted_at_ms) "
            "VALUES(?,?,?,?,?,'view',1,?,?)",
            (
                tenant,
                "session",
                parent,
                "user",
                "bob",
                "user:alice",
                int(_time.time() * 1000),
            ),
        )
        await conn.commit()
        bob_response = await handle_list(_request("bob"))
        assert bob_response.status == 200
        bob_ids = {
            row["session_id"]
            for row in json.loads(bob_response.text)["sessions"]
        }
        assert bob_ids == {bob_child}, bob_ids
    finally:
        await db.close()
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("sessions_cross_device", "legacy session REST reads and writes share canonical ACLs")
async def t_legacy_session_endpoints_share_acl_boundary(ctx: TestContext) -> None:
    """Every compatibility payload is gated by the same normalized resource.

    View grants admit reads only; admin grants admit metadata/model mutations;
    missing and invisible ids are indistinguishable; list filters never accept
    a caller-supplied identity.
    """
    import json
    from src.gateway.api import sessions as api
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-rest-acl-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    try:
        await db.connect()
        conn = await db._ensure_connected()
        await conn.execute(
            "INSERT INTO network(singleton, role, network_id, name, created_at) "
            "VALUES(1, 'coordinator', 'rest-acl-network', 'Test', ?) "
            "ON CONFLICT(singleton) DO UPDATE SET network_id=excluded.network_id",
            (_time.time(),),
        )
        await conn.commit()
        session_id = f"alice-rest-{uuid.uuid4().hex[:8]}"
        await db.upsert_session(session_id, client_id="alice", title="Alice title")
        run = {
            "run_id": "rest-acl-run",
            "status": "COMPLETED",
            "created_at": _time.time(),
            "messages": [
                {"id": "rest-acl-user", "role": "user", "content": "private hello"},
                {"id": "rest-acl-agent", "role": "assistant", "content": "private reply"},
            ],
        }
        await conn.execute(
            "UPDATE sessions SET runs=? WHERE session_id=?",
            (json.dumps([run]), session_id),
        )
        await db._project_operational_session(session_id)
        await db.append_session_event(session_id, "user/message", {"text": "private hello"})
        provider_id = await db.upsert_provider(
            name="acl-provider",
            framework="api-based",
            api_key="test-key",
        )
        await db.upsert_model(provider_id=provider_id, model="acl-model")

        async def _forget(_sid):
            return None

        class _Gateway:
            def __init__(self):
                self.agent = SimpleNamespace(
                    memory_db=db,
                    _db=db,
                    model=None,
                    forget_session=_forget,
                )
                self.sessions = SimpleNamespace(_clients={})

            async def active_live_session_ids(self, *, client_id, handle):
                return set()

        gateway = _Gateway()

        class _Request(dict):
            def __init__(
                self,
                method: str,
                handle: str,
                *,
                sid: str = session_id,
                query=None,
                body=None,
            ):
                device = f"{handle}-device"
                cert = SimpleNamespace(
                    network_id="rest-acl-network",
                    handle=handle,
                    device_pubkey_hex=device,
                    capabilities=[],
                )
                super().__init__(
                    device_cert=cert,
                    network_id="rest-acl-network",
                    user_handle=handle,
                    client_id=device,
                )
                self.app = {"gateway": gateway}
                self.match_info = {"session_id": sid}
                self.query = query or {}
                self._body = body

            @property
            def can_read_body(self):
                return self._body is not None

            async def json(self):
                return self._body

        def req(method: str, handle: str, **kwargs):
            return _Request(method, handle, **kwargs)

        runtime_id = "acl-provider:acl-model"
        owner_pin = await api.handle_pin(
            req("PUT", "alice", body={"runtime_id": runtime_id})
        )
        assert owner_pin.status == 200, owner_pin.text

        read_handlers = (
            api.handle_get_context,
            api.handle_get_runs,
            api.handle_get_events,
            api.handle_get,
        )
        missing_id = f"missing-{uuid.uuid4().hex[:8]}"
        for handler in read_handlers:
            owner_response = await handler(req("GET", "alice"))
            assert owner_response.status == 200, (handler.__name__, owner_response.text)
            hidden = await handler(req("GET", "bob"))
            missing = await handler(req("GET", "bob", sid=missing_id))
            assert hidden.status == missing.status == 404, handler.__name__
            assert json.loads(hidden.text) == json.loads(missing.text)

        hidden_patch = await api.handle_patch_metadata(
            req("PATCH", "bob", body={"title": "stolen"})
        )
        hidden_pin = await api.handle_pin(
            req("PUT", "bob", body={"runtime_id": runtime_id})
        )
        hidden_unpin = await api.handle_unpin(req("DELETE", "bob"))
        assert {hidden_patch.status, hidden_pin.status, hidden_unpin.status} == {404}
        assert (await db.get_session(session_id))["title"] == "Alice title"
        assert await db.get_session_pin(session_id) == runtime_id

        # A view grant admits every read but no mutation.
        now_ms = int(_time.time() * 1000)
        await conn.execute(
            "INSERT INTO resource_acl "
            "(tenant_id,resource_type,resource_id,principal_type,principal_id,"
            "permission,acl_version,granted_by_principal_id,granted_at_ms) "
            "VALUES('rest-acl-network','session',?,'user','bob','view',1,'user:alice',?)",
            (session_id, now_ms),
        )
        await conn.commit()
        for handler in read_handlers:
            response = await handler(req("GET", "bob"))
            assert response.status == 200, (handler.__name__, response.text)
        still_read_only = await api.handle_patch_metadata(
            req("PATCH", "bob", body={"title": "still forbidden"})
        )
        assert still_read_only.status == 404

        # Admin is explicit and can mutate model/title without changing owner.
        await conn.execute(
            "INSERT INTO resource_acl "
            "(tenant_id,resource_type,resource_id,principal_type,principal_id,"
            "permission,acl_version,granted_by_principal_id,granted_at_ms) "
            "VALUES('rest-acl-network','session',?,'user','bob','admin',1,'user:alice',?)",
            (session_id, now_ms + 1),
        )
        await conn.commit()
        patched = await api.handle_patch_metadata(
            req("PATCH", "bob", body={"title": "Shared title"})
        )
        assert patched.status == 200, patched.text
        assert (await db.get_session(session_id))["title"] == "Shared title"
        assert (await db.get_session(session_id))["client_id"] == "alice"
        assert (await api.handle_unpin(req("DELETE", "bob"))).status == 200
        assert await db.get_session_pin(session_id) is None
        assert (
            await api.handle_pin(
                req("PUT", "bob", body={"runtime_id": runtime_id})
            )
        ).status == 200

        # A genuinely new id can still be created by first-message PATCH, but
        # ownership comes from the certificate rather than body/query fields.
        bob_new = f"bob-new-{uuid.uuid4().hex[:8]}"
        created = await api.handle_patch_metadata(
            req("PATCH", "bob", sid=bob_new, body={"title": "Bob chat"})
        )
        assert created.status == 200, created.text
        assert (await db.get_session(bob_new))["client_id"] == "bob"

        alice_list = await api.handle_list(
            req("GET", "alice", query={"client_id": "bob", "limit": "200"})
        )
        bob_list = await api.handle_list(
            req("GET", "bob", query={"client_id": "alice", "limit": "200"})
        )
        alice_ids = {row["session_id"] for row in json.loads(alice_list.text)["sessions"]}
        bob_ids = {row["session_id"] for row in json.loads(bob_list.text)["sessions"]}
        assert session_id in alice_ids and bob_new not in alice_ids
        assert bob_new in bob_ids and session_id not in bob_ids
    finally:
        await db.close()
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("sessions_cross_device", "legacy device-pubkey rows resolve via network_devices")
async def t_legacy_pubkey_via_devices(ctx: TestContext) -> None:
    """A pre-fix row carries ``metadata.client_id = <pubkey_hex_B>``. The
    user later pairs device B (so ``network_devices`` binds pubkey_B to
    handle_A). Listing by handle_A must surface that row even though
    its ``client_id`` is still the pubkey form."""
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-legacy-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        # Pubkey bytes printed as 64 hex chars — what the auth layer
        # stamps onto requests.
        pubkey_b = uuid.uuid4().hex + uuid.uuid4().hex
        await _seed_device_binding(db, "handle_A", pubkey_b)
        # Legacy row: client_id is the device pubkey, not the handle.
        sid_legacy = f"legacy-{uuid.uuid4().hex[:8]}"
        await db.upsert_session(
            sid_legacy, client_id=pubkey_b, title="legacy", framework="api-based",
        )
        # Foreign row: a pubkey NOT bound to handle_A. Must NOT leak.
        pubkey_other = uuid.uuid4().hex + uuid.uuid4().hex
        sid_foreign = f"foreign-{uuid.uuid4().hex[:8]}"
        await db.upsert_session(
            sid_foreign, client_id=pubkey_other, title="foreign", framework="api-based",
        )
        rows = await db.list_all_sessions("handle_A", limit=50)
        ids = {r["session_id"] for r in rows}
        assert sid_legacy in ids, ids
        assert sid_foreign not in ids, ids
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("sessions_cross_device", "upsert_session keeps device_id alongside handle owner")
async def t_device_id_preserved(ctx: TestContext) -> None:
    """Per-device routing (sticky retries, WS reconnect) still needs the
    device pubkey. The fix stores the user handle in
    ``metadata.client_id`` AND the originating device in
    ``metadata.device_id`` so both pieces of routing information
    survive a process restart."""
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-devid-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        sid = f"sid-{uuid.uuid4().hex[:8]}"
        pubkey = uuid.uuid4().hex + uuid.uuid4().hex
        await db.upsert_session(
            sid, client_id="alice", device_id=pubkey, framework="api-based",
        )
        s = await db.get_session(sid)
        assert s is not None
        assert s["client_id"] == "alice", s
        # ``get_session`` only surfaces a small projection; read the raw
        # metadata to confirm device_id landed.
        conn = await db._ensure_connected()
        cur = await conn.execute(
            "SELECT metadata FROM sessions WHERE session_id = ?",
            (sid,),
        )
        row = await cur.fetchone()
        meta = db._parse_metadata(row[0])
        assert meta.get("device_id") == pubkey, meta
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
