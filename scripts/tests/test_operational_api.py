"""Contract/security tests for unified operational history and search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from ._framework import TestContext, test


class _Request(dict):
    def __init__(self, gateway, *, tenant: str, handle: str, device: str, query=None, body=None, match=None):
        cert = SimpleNamespace(
            network_id=tenant,
            handle=handle,
            device_pubkey_hex=device,
            capabilities=[],
        )
        super().__init__(
            device_cert=cert,
            network_id=tenant,
            user_handle=handle,
            client_id=device,
        )
        self.app = {"gateway": gateway}
        self.query = query or {}
        self.match_info = match or {}
        self._body = body
        self.content_length = len(json.dumps(body).encode()) if body is not None else None

    async def json(self):
        return self._body


def _payload(response) -> dict:
    return json.loads(response.text)


@test("operational_api", "explicit channel filter supports gateway-only canaries")
async def t_gateway_only_channel_filter(_ctx: TestContext) -> None:
    from src.core.server import _selected_bridge_names

    config = {
        "channels": {
            "telegram": {"token": "canary"},
            "discord": {"token": "canary"},
        }
    }
    assert _selected_bridge_names(config, None) == ["telegram", "discord"]
    assert _selected_bridge_names(config, ["telegram"]) == ["telegram"]
    assert _selected_bridge_names(config, ["gateway"]) == []
    assert _selected_bridge_names(config, []) == []


async def _ready_capabilities(operational, request: _Request) -> dict:
    payload = _payload(await operational.handle_capabilities(request))
    if not payload.get("storage", {}).get("search_ready"):
        await asyncio.wait_for(
            request.app["gateway"]._operational_ready_event.wait(), timeout=10
        )
        payload = _payload(await operational.handle_capabilities(request))
    return payload


async def _seed_complete_fixture(db) -> tuple[str, object]:
    now = time.time()
    await db.upsert_session(
        "orchid-chat", client_id="alice",
        title="Orchid chat token=NEVER_INDEX_CHAT_TITLE",
    )
    runs = [
        {
            "run_id": "orchid-run",
            "status": "COMPLETED",
            "created_at": now,
            "messages": [
                {
                    "id": "orchid-user", "role": "user",
                    "content": "orchid message before Bearer NEVER_INDEX_CHAT_BEARER "
                    "opaque AbCdEf0123456789AbCdEf0123456789AbCdEf01",
                    "redacted_reasoning_content": "NEVER_INDEX_REDACTED_THINKING",
                },
                {"id": "orchid-tool-message", "role": "tool", "tool_call_id": "orchid-call", "content": "orchid tool visible"},
                {"id": "orchid-assistant", "role": "assistant", "content": "orchid message after"},
            ],
            "tools": [
                {
                    "tool_call_id": "orchid-call",
                    "tool_name": "orchid_tool",
                    "tool_args": {"token": "NEVER_INDEX_TOOL_ARG", "nested": {"safe_key": 3}},
                    "result": {"secret": "NEVER_INDEX_TOOL_RESULT"},
                    "status": "completed",
                }
            ],
        }
    ]
    await db._conn.execute(
        "UPDATE sessions SET runs=?, updated_at=? WHERE session_id='orchid-chat'",
        (json.dumps(runs), int(now) + 1),
    )
    await db._project_operational_session("orchid-chat")
    await db.upsert_session(
        "orchid-child", client_id="alice", title="Orchid delegated child",
        parent_session_id="orchid-chat", origin="delegation", kind="subagent",
    )
    await db.upsert_session("forbidden-chat", client_id="bob", title="Orchid forbidden chat")

    c = db._conn
    graph = {
        "version": 1,
        "nodes": [{"id": "node-1", "type": "tool", "data": {"label": "Orchid step", "prompt": "orchid workflow prompt sk-ant-NEVER_INDEX_WORKFLOW_TOKEN123456789"}}],
        "edges": [],
    }
    await c.execute(
        "INSERT INTO workflow_tasks(id,name,description,graph_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
        ("orchid-workflow", "Orchid workflow", "orchid workflow description", json.dumps(graph), now, now),
    )
    trace = [{
        "node_id": "node-1", "type": "tool", "status": "success",
        "tool_name": "orchid_trace_tool", "tool_invocation_ids": ["trace-tool-id"],
        "error": "orchid trace warning token=NEVER_INDEX_TRACE_SECRET",
        "started_at": now, "finished_at": now + 1,
    }]
    await c.execute(
        "INSERT INTO workflow_runs(id,workflow_id,trigger,status,started_at,finished_at,trace_json,error) VALUES(?,?,?,?,?,?,?,?)",
        ("orchid-workflow-run", "orchid-workflow", "manual", "success", now, now + 1, json.dumps(trace), None),
    )
    await c.execute(
        "INSERT INTO scheduled_tasks(id,name,cron_expression,prompt,created_at,updated_at) VALUES(?,?,?,?,?,?)",
        ("orchid-task", "Orchid schedule", "* * * * *", "orchid scheduled prompt", now, now),
    )
    await c.execute(
        "INSERT INTO task_runs(id,task_id,trigger,status,started_at,finished_at,output,error) VALUES(?,?,?,?,?,?,?,?)",
        ("orchid-task-run", "orchid-task", "manual", "success", now, now + 1, "orchid task output eyJhbGciOiJIUzI1NiJ9.NEVERINDEXSCHEDULEDPAYLOAD.NEVERINDEXSCHEDULEDSIGNATURE", None),
    )
    await c.execute(
        "INSERT INTO events(id,name,slug,type,secret_enc,input_schema_json,action_kind,description,prompt_template,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "orchid-event", "Orchid event", "orchid-event", "generic",
            "NEVER_INDEX_EVENT_SECRET", "[]", "prompt", "orchid event description",
            "orchid event prompt", now, now,
        ),
    )
    await c.execute(
        "INSERT INTO event_deliveries(id,event_id,source,status,payload_json,started_at,finished_at,output,error) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (
            "orchid-delivery", "orchid-event", "manual", "success",
            json.dumps({"payload_secret": "NEVER_INDEX_EVENT_PAYLOAD"}),
            now, now + 1, "orchid delivery output https://example.invalid/cb?X-Amz-Signature=NEVER_INDEX_SIGNED_URL", None,
        ),
    )
    # A newly-created private definition owned by Alice must not become an
    # installation-shared legacy row for Bob.
    await c.execute(
        "INSERT INTO workflow_tasks(id,name,description,graph_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
        ("private-workflow", "Private garden", "secretgarden", '{"version":1,"nodes":[],"edges":[]}', now, now),
    )
    from src.memory.operational.automation import claim_resource, project_automation

    tenant_row = await (await c.execute("SELECT tenant_id FROM sessions_v2 LIMIT 1")).fetchone()
    tenant = str(tenant_row[0])
    await claim_resource(
        c, tenant_id=tenant, resource_type="workflow_definition",
        resource_id="private-workflow", owner_principal_id="user:alice",
    )
    await project_automation(c)
    await c.commit()
    gateway = SimpleNamespace(agent=SimpleNamespace(memory_db=db))
    return tenant, gateway


@test("operational_api", "fresh capability warms index and all nine targets are searchable")
async def t_capability_and_all_targets(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB
    from src.memory.operational.search import operational_search_path

    with TemporaryDirectory(prefix="openagent-operational-api-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        try:
            tenant, gateway = await _seed_complete_fixture(db)
            vault = path.with_name("vault_index_canary.db")
            transcript = path.with_name("transcript_index_canary.db")
            vault.write_bytes(b"vault-index-sentinel")
            transcript.write_bytes(b"transcript-index-sentinel")
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (vault, transcript)}

            # No POST /search has happened: capability discovery itself must
            # reach a ready boundary and advertise the real implementation.
            caps = await _ready_capabilities(
                operational,
                _Request(gateway, tenant=tenant, handle="alice", device="alice-device"),
            )
            assert caps["features"]["history"]["version"] == 2
            assert caps["features"]["global_search"]["version"] == 1
            assert caps["storage"]["search_ready"] is True

            search_body = {
                "query": "orchid", "scopes": ["chats", "tools", "workflows", "scheduled", "events"],
                "filters": {}, "sort": "relevance", "grouping": "match", "limit": 100, "cursor": None,
            }
            response = await operational.handle_search(
                _Request(gateway, tenant=tenant, handle="alice", device="alice-device", body=search_body)
            )
            assert response.status == 200, response.text
            page = _payload(response)
            targets = {item["target"]["kind"] for item in page["items"]}
            assert targets == {
                "chat", "chat_message", "chat_tool", "workflow_definition", "workflow_run",
                "scheduled_definition", "scheduled_run", "event_definition", "event_delivery",
            }, targets
            workflow_target = next(item["target"] for item in page["items"] if item["target"]["kind"] == "workflow_run")
            definition_target = next(item["target"] for item in page["items"] if item["target"]["kind"] == "workflow_definition")
            assert "trace_step_id" not in workflow_target
            assert "node_id" not in definition_target and "field" not in definition_target
            assert page["items"] and page["snapshot"]["indexed_seq"] >= 1

            search_path = operational_search_path(path)
            assert search_path not in {vault, transcript}
            raw = search_path.read_bytes()
            for forbidden in (
                b"NEVER_INDEX_TOOL_ARG", b"NEVER_INDEX_TOOL_RESULT",
                b"NEVER_INDEX_EVENT_SECRET", b"NEVER_INDEX_EVENT_PAYLOAD",
                b"NEVER_INDEX_TRACE_SECRET",
                b"NEVER_INDEX_CHAT_TITLE", b"NEVER_INDEX_CHAT_BEARER",
                b"AbCdEf0123456789AbCdEf0123456789AbCdEf01",
                b"NEVER_INDEX_REDACTED_THINKING",
                b"NEVER_INDEX_WORKFLOW_TOKEN", b"NEVERINDEXSCHEDULEDPAYLOAD",
                b"NEVER_INDEX_SIGNED_URL",
            ):
                assert forbidden not in raw, forbidden.decode()
            physical = [
                candidate for candidate in search_path.parent.glob(f"{search_path.name}*")
                if candidate.is_file()
            ]
            for candidate in physical:
                contents = candidate.read_bytes()
                assert b"NEVER_INDEX_" not in contents, candidate.name
                assert b"NEVERINDEX" not in contents, candidate.name
                assert candidate.stat().st_mode & 0o777 == 0o600, candidate.name
            from src.memory.operational.search import _open_index

            pragma_conn = _open_index(search_path)
            try:
                assert int(pragma_conn.execute("PRAGMA synchronous").fetchone()[0]) == 1
            finally:
                pragma_conn.close()
            after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (vault, transcript)}
            assert before == after
        finally:
            if "gateway" in locals():
                await operational.stop_background_maintenance(gateway)
            await db.close()


@test("operational_api", "history/messages/search enforce ACL and stable snapshots")
async def t_acl_history_messages_and_snapshot(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-contract-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            tenant, gateway = await _seed_complete_fixture(db)
            alice = lambda **kwargs: _Request(
                gateway, tenant=tenant, handle="alice", device="alice-device", **kwargs
            )
            bob = lambda **kwargs: _Request(
                gateway, tenant=tenant, handle="bob", device="bob-device", **kwargs
            )
            for index in range(4):
                await db.upsert_session(
                    f"revocation-{index}", client_id="alice",
                    title=f"revocationcanary {index}",
                )
            await _ready_capabilities(operational, alice())

            history = _payload(await operational.handle_history(alice(query={"limit": "100"})))
            assert {item["kind"] for item in history["items"]} >= {
                "chat", "delegated_session", "workflow_run", "scheduled_run", "event_delivery",
            }
            assert all(item["resource_id"] != "forbidden-chat" for item in history["items"])
            assert set(history) == {"items", "next_cursor", "has_more", "revision", "snapshot"}

            anchor = await (
                await db._conn.execute(
                    "SELECT id FROM session_messages WHERE session_id='orchid-chat' "
                    "AND text='orchid tool visible'"
                )
            ).fetchone()
            messages = _payload(await operational.handle_session_messages(alice(
                query={"around": str(anchor[0]), "before": "1", "after": "1"},
                match={"session_id": "orchid-chat"},
            )))
            assert messages["anchor_found"] is True
            assert len(messages["messages"]) == 3
            assert messages["messages"][1]["id"] == anchor[0]
            assert all(message["visible_reasoning"] is None for message in messages["messages"])

            forbidden = await operational.handle_session_messages(bob(
                match={"session_id": "orchid-chat"}
            ))
            assert forbidden.status == 404

            private_search = {
                "query": "secretgarden", "scopes": ["workflows"], "filters": {},
                "sort": "relevance", "grouping": "match", "limit": 10, "cursor": None,
            }
            assert len(_payload(await operational.handle_search(alice(body=private_search)))["items"]) == 1
            assert _payload(await operational.handle_search(bob(body=private_search)))["items"] == []

            revocation_query = {
                "query": "revocationcanary", "scopes": ["chats"], "filters": {},
                "sort": "relevance", "grouping": "match", "limit": 1, "cursor": None,
            }
            revocation_first = _payload(
                await operational.handle_search(alice(body=revocation_query))
            )
            decoded = gateway._operational_cursor_state.decode(
                revocation_first["next_cursor"]
            )
            revocation_snapshot = gateway._operational_cursor_state.search[decoded["s"]]
            revoked_id = str(revocation_snapshot.rows[1]["resource_id"])
            stale_id = str(revocation_snapshot.rows[2]["resource_id"])
            expected_fill_id = str(revocation_snapshot.rows[3]["resource_id"])
            # Mutate canonical authority without consuming an outbox entry.
            # A continuation must not trust the old snapshot/index ACL or
            # source version, and must scan forward to fill the requested page.
            await db._conn.execute(
                "UPDATE sessions_v2 SET owner_principal_id='user:bob', "
                "acl_version=acl_version+1 WHERE id=?",
                (revoked_id,),
            )
            await db._conn.execute(
                "UPDATE sessions_v2 SET source_version=source_version+1 WHERE id=?",
                (stale_id,),
            )
            await db._conn.commit()
            revocation_query["cursor"] = revocation_first["next_cursor"]
            revocation_second = _payload(
                await operational.handle_search(alice(body=revocation_query))
            )
            returned_ids = {item["root"]["id"] for item in revocation_second["items"]}
            assert revoked_id not in returned_ids and stale_id not in returned_ids
            assert returned_ids == {expected_fill_id}

            paged = {
                "query": "orchid", "scopes": ["chats", "tools", "workflows", "scheduled", "events"],
                "filters": {}, "sort": "relevance", "grouping": "match", "limit": 1, "cursor": None,
            }
            first = _payload(await operational.handle_search(alice(body=paged)))
            assert first["has_more"] and first["next_cursor"]
            # Advance canonical/outbox state between pages. The stored redacted
            # highlight must remain usable even if another request later syncs.
            await db.upsert_session("concurrent", client_id="alice", title="Orchid concurrent")
            paged["cursor"] = first["next_cursor"]
            second_response = await operational.handle_search(alice(body=paged))
            assert second_response.status == 200, second_response.text
            second = _payload(second_response)
            assert second["items"]
            assert second["snapshot"]["search_session_id"] == first["snapshot"]["search_session_id"]
        finally:
            await operational.stop_background_maintenance(gateway)
            await db.close()


@test("operational_api", "capability warm-up drains more than one outbox batch")
async def t_capability_multi_batch_warmup(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-warmup-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            tenant = str((await (await db._conn.execute("SELECT tenant_id FROM sessions_v2 LIMIT 1")).fetchone())[0]) if (await (await db._conn.execute("SELECT COUNT(*) FROM sessions_v2")).fetchone())[0] else None
            if tenant is None:
                state = await (await db._conn.execute("SELECT db_instance_id FROM operational_storage_state WHERE singleton_id=1")).fetchone()
                tenant = f"installation:{state[0]}"
            now_ms = int(time.time() * 1000)
            await db._conn.executemany(
                "INSERT INTO search_outbox(tenant_id, source_kind, source_id, operation, source_version, acl_version, committed_at_ms) "
                "VALUES (?, 'synthetic_noop', ?, 'upsert', 1, 1, ?)",
                ((tenant, f"noop-{i}", now_ms) for i in range(10_001)),
            )
            await db._conn.commit()
            gateway = SimpleNamespace(agent=SimpleNamespace(memory_db=db))
            response = await operational.handle_capabilities(
                _Request(gateway, tenant=tenant, handle="alice", device="alice-device")
            )
            payload = _payload(response)
            assert response.status == 200
            # The request is bounded: it may report warming, while a durable
            # background task drains every remaining batch.
            if not payload["storage"]["search_ready"]:
                await asyncio.wait_for(gateway._operational_ready_event.wait(), timeout=10)
                payload = _payload(await operational.handle_capabilities(
                    _Request(gateway, tenant=tenant, handle="alice", device="alice-device")
                ))
            assert payload["storage"]["search_ready"] is True
            assert payload["storage"]["indexed_seq"] >= 10_001
        finally:
            if "gateway" in locals():
                await operational.stop_background_maintenance(gateway)
            await db.close()


@test("operational_api", "persistent worker indexes writes after ready without a search request")
async def t_persistent_search_consumer(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB
    from src.memory.operational.search import operational_search_status

    with TemporaryDirectory(prefix="openagent-operational-consumer-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            tenant, gateway = await _seed_complete_fixture(db)
            request = _Request(
                gateway, tenant=tenant, handle="alice", device="alice-device"
            )
            await _ready_capabilities(operational, request)
            concurrent = await asyncio.gather(
                operational.handle_capabilities(request),
                operational.handle_capabilities(_Request(
                    gateway, tenant=tenant, handle="alice", device="alice-device"
                )),
                db.upsert_session(
                    "concurrent-capability-writer", client_id="alice",
                    title="concurrent capability writer",
                ),
            )
            assert concurrent[0].status == 200 and concurrent[1].status == 200
            before = await operational_search_status(db)
            await db.upsert_session(
                "persistent-consumer-chat", client_id="alice",
                title="persistentconsumerneedle",
            )
            outbox_head = int(
                (
                    await (
                        await db._conn.execute(
                            "SELECT COALESCE(MAX(seq), 0) FROM search_outbox"
                        )
                    ).fetchone()
                )[0]
            )
            deadline = time.monotonic() + 5
            while True:
                status = await operational_search_status(db)
                if status["ready"] and int(status["seq"]) >= outbox_head:
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError("persistent search consumer did not catch up")
                await asyncio.sleep(0.05)
            assert int(status["seq"]) > int(before["seq"])
            search = await operational.handle_search(_Request(
                gateway, tenant=tenant, handle="alice", device="alice-device",
                body={
                    "query": "persistentconsumerneedle", "scopes": ["chats"],
                    "filters": {}, "sort": "relevance", "grouping": "match",
                    "limit": 10, "cursor": None,
                },
            ))
            assert search.status == 200, search.text
            assert _payload(search)["items"][0]["root"]["id"] == "persistent-consumer-chat"
        finally:
            if "gateway" in locals():
                await operational.stop_background_maintenance(gateway)
            await db.close()


@test("operational_api", "large session backfill is background-only and hot history is read-only")
async def t_large_backfill_and_hot_history(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-scale-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            now = int(time.time())
            await db._conn.executemany(
                "INSERT INTO sessions "
                "(session_id, session_type, user_id, metadata, runs, created_at, updated_at) "
                "VALUES (?, 'agent', 'openagent', ?, '[]', ?, ?)",
                (
                    (
                        f"scale-{index}",
                        json.dumps({"client_id": "openagent"}),
                        now + index,
                        now + index,
                    )
                    for index in range(10_001)
                ),
            )
            await db._conn.commit()
            state = await (
                await db._conn.execute(
                    "SELECT db_instance_id FROM operational_storage_state WHERE singleton_id=1"
                )
            ).fetchone()
            tenant = f"installation:{state[0]}"
            gateway = SimpleNamespace(agent=SimpleNamespace(memory_db=db))
            request = _Request(
                gateway, tenant=tenant, handle="openagent", device="scale-device"
            )
            started = time.monotonic()
            initial = _payload(await operational.handle_capabilities(request))
            assert time.monotonic() - started < 1.0
            assert initial["storage"]["history_ready"] is False
            await asyncio.wait_for(gateway._operational_ready_event.wait(), timeout=60)
            projected = int(
                (
                    await (
                        await db._conn.execute(
                            "SELECT COUNT(*) FROM sessions_v2 WHERE deleted_at_ms IS NULL"
                        )
                    ).fetchone()
                )[0]
            )
            assert projected == 10_001
            before = await (
                await db._conn.execute(
                    "SELECT history_revision, (SELECT COUNT(*) FROM search_outbox) "
                    "FROM operational_storage_state WHERE singleton_id=1"
                )
            ).fetchone()
            hot_started = time.monotonic()
            response = await operational.handle_history(_Request(
                gateway, tenant=tenant, handle="openagent", device="scale-device",
                query={"limit": "100"},
            ))
            hot_seconds = time.monotonic() - hot_started
            assert response.status == 200, response.text
            assert len(_payload(response)["items"]) == 100
            assert hot_seconds < 5.0
            after = await (
                await db._conn.execute(
                    "SELECT history_revision, (SELECT COUNT(*) FROM search_outbox) "
                    "FROM operational_storage_state WHERE singleton_id=1"
                )
            ).fetchone()
            assert tuple(after) == tuple(before)
        finally:
            if "gateway" in locals():
                await operational.stop_background_maintenance(gateway)
            await db.close()


@test("operational_api", "malformed search filters return sanitized 400 responses")
async def t_search_validation_is_strict(_ctx: TestContext) -> None:
    import logging
    from src.gateway.api import operational
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-validation-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            tenant, gateway = await _seed_complete_fixture(db)
            base = {
                "query": "PRIVATE_BODY_CANARY", "scopes": ["chats"], "filters": {},
                "sort": "relevance", "grouping": "root", "limit": 10, "cursor": None,
            }
            invalid = [
                {**base, "scopes": [{}]},
                {**base, "filters": {"status": "running"}},
                {**base, "filters": {"root": []}},
                {**base, "filters": {"parent_type": "workflow"}},
                {**base, "filters": {"origin": "x" * 129}},
                {**base, "filters": {"from": "2026-01-02T00:00:00Z", "to": "2026-01-01T00:00:00Z"}},
                {**base, "filters": {"from": 42}},
                {**base, "cursor": "short"},
            ]
            captured: list[logging.LogRecord] = []
            handler = logging.Handler()
            handler.emit = captured.append  # type: ignore[assignment]
            logger = logging.getLogger("src.gateway.api.operational")
            logger.addHandler(handler)
            try:
                for body in invalid:
                    response = await operational.handle_search(
                        _Request(gateway, tenant=tenant, handle="alice", device="alice-device", body=body)
                    )
                    assert response.status == 400, response.text
                    assert "PRIVATE_BODY_CANARY" not in response.text
            finally:
                logger.removeHandler(handler)
            assert all("PRIVATE_BODY_CANARY" not in record.getMessage() for record in captured)
        finally:
            await db.close()


@test("operational_api", "live canary fixture is isolated and explicitly purged")
async def t_live_canary_fixture_cleanup(_ctx: TestContext) -> None:
    from scripts.live_smoke import (
        _purge_operational_canary_metadata,
        _remove_operational_canary_sources,
        _seed_operational_canary,
        _wait_operational_canary_cleanup,
    )
    from src.gateway.api import operational
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-live-canary-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        fixture = None
        sources_removed = False
        gateway = SimpleNamespace(agent=SimpleNamespace(memory_db=db))
        try:
            await db._conn.execute(
                "INSERT INTO network(singleton,role,network_id,name,created_at) "
                "VALUES(1,'coordinator','smoke-network','smoke',0)"
            )
            await db._conn.commit()
            fixture = await _seed_operational_canary(
                path, network_id="smoke-network", handle="smoke-user"
            )
            operational.start_background_maintenance(gateway)
            await asyncio.wait_for(gateway._operational_ready_event.wait(), timeout=10)
            ids = fixture["ids"]
            assert isinstance(ids, dict)
            assert int((await (
                await db._conn.execute(
                    "SELECT COUNT(*) FROM sessions_v2 WHERE id IN (?,?)",
                    (ids["session"], ids["forbidden_session"]),
                )
            ).fetchone())[0]) == 2

            await _remove_operational_canary_sources(path, fixture)
            sources_removed = True
            await _wait_operational_canary_cleanup(path, fixture, timeout_s=10)
            await _purge_operational_canary_metadata(path, fixture)
            fixture = None
            for table in (
                "sessions", "sessions_v2", "session_messages", "tool_invocations",
                "operational_automation_projection",
            ):
                assert int((await (
                    await db._conn.execute(f"SELECT COUNT(*) FROM {table}")
                ).fetchone())[0]) == 0, table
        finally:
            await operational.stop_background_maintenance(gateway)
            if fixture is not None:
                if not sources_removed:
                    await _remove_operational_canary_sources(path, fixture)
                try:
                    await _wait_operational_canary_cleanup(path, fixture, timeout_s=10)
                    await _purge_operational_canary_metadata(path, fixture)
                except Exception:
                    pass
            await db.close()


@test("operational_api", "search snapshot quotas evict the oldest principal cursor")
async def t_search_snapshot_quota(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-quota-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            tenant, gateway = await _seed_complete_fixture(db)
            await _ready_capabilities(
                operational,
                _Request(gateway, tenant=tenant, handle="alice", device="alice-device"),
            )
            body = {
                "query": "orchid", "scopes": ["chats", "tools", "workflows", "scheduled", "events"],
                "filters": {}, "sort": "relevance", "grouping": "match",
                "limit": 1, "cursor": None,
            }
            cursors: list[str] = []
            for _ in range(5):
                page = _payload(await operational.handle_search(_Request(
                    gateway, tenant=tenant, handle="alice", device="alice-device",
                    body=dict(body),
                )))
                cursors.append(page["next_cursor"])
            assert len(gateway._operational_cursor_state.search) == 4
            stale_body = dict(body)
            stale_body["cursor"] = cursors[0]
            stale = await operational.handle_search(_Request(
                gateway, tenant=tenant, handle="alice", device="alice-device",
                body=stale_body,
            ))
            assert stale.status == 409
            assert _payload(stale)["error"]["code"] == "cursor_stale"
        finally:
            if "gateway" in locals():
                await operational.stop_background_maintenance(gateway)
            await db.close()


@test("operational_api", "detail resolvers keep legacy epochs and canonical deep-link ids")
async def t_detail_resolver_contract(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-detail-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            tenant, gateway = await _seed_complete_fixture(db)
            request = _Request(gateway, tenant=tenant, handle="alice", device="alice-device")
            await _ready_capabilities(operational, request)
            workflow = await db.get_workflow_run("orchid-workflow-run")
            workflow_detail = await operational.decorate_workflow_run_detail(request, workflow)
            assert workflow_detail is not None
            assert isinstance(workflow_detail["started_at"], (int, float))
            assert workflow_detail["started_at_iso"].endswith("Z")
            assert workflow_detail["trace"]
            assert workflow_detail["trace_steps"][0]["id"].startswith("trace:")
            assert "trace_step_id" not in workflow_detail["trace_steps"][0]
            assert workflow_detail["trace_steps"][0]["tool_invocation_ids"] == ["trace-tool-id"]

            scheduled = await operational.handle_scheduled_run(_Request(
                gateway, tenant=tenant, handle="alice", device="alice-device",
                match={"run_id": "orchid-task-run"},
            ))
            assert scheduled.status == 200
            scheduled_body = _payload(scheduled)
            assert scheduled_body["started_at"].endswith("Z")
            assert "NEVERINDEXSCHEDULED" not in (scheduled_body["output_summary_safe"] or "")

            event = await db.get_event_delivery("orchid-delivery")
            event_detail = await operational.decorate_event_delivery_detail(request, event)
            assert event_detail is not None
            assert isinstance(event_detail["started_at"], (int, float))
            assert event_detail["occurred_at"].endswith("Z")

            tool_id = str((await (
                await db._conn.execute(
                    "SELECT id FROM tool_invocations WHERE session_id='orchid-chat' LIMIT 1"
                )
            ).fetchone())[0])
            tool = await operational.handle_tool_invocation(_Request(
                gateway, tenant=tenant, handle="alice", device="alice-device",
                match={"tool_id": tool_id},
            ))
            assert tool.status == 200
            tool_text = tool.text
            assert "NEVER_INDEX_TOOL_ARG" not in tool_text
            assert "NEVER_INDEX_TOOL_RESULT" not in tool_text
        finally:
            if "gateway" in locals():
                await operational.stop_background_maintenance(gateway)
            await db.close()
