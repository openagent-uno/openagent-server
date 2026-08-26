"""Contract/security tests for unified operational history and search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
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


@test("operational_api", "user and agent principals with the same handle never alias")
async def t_principal_type_isolation(_ctx: TestContext) -> None:
    from src.memory.operational.access import AccessContext, row_is_visible_without_grant

    gateway = SimpleNamespace()
    user = AccessContext.from_request(
        _Request(gateway, tenant="same-network", handle="same-handle", device="user-device")
    )
    agent_request = _Request(
        gateway, tenant="same-network", handle="same-handle", device="agent-device"
    )
    agent_request["device_cert"].capabilities = ["agent"]
    agent = AccessContext.from_request(agent_request)
    base = {"tenant_id": "same-network", "visibility": "private"}

    assert row_is_visible_without_grant(
        {**base, "owner_principal_id": "user:same-handle"}, user
    )
    assert not row_is_visible_without_grant(
        {**base, "owner_principal_id": "agent:same-handle"}, user
    )
    assert row_is_visible_without_grant(
        {**base, "owner_principal_id": "agent:same-handle"}, agent
    )
    assert not row_is_visible_without_grant(
        {**base, "owner_principal_id": "user:same-handle"}, agent
    )
    # Legacy untyped owner values are ambiguous when both principal classes
    # exist and therefore cannot authorize either caller.
    assert not row_is_visible_without_grant(
        {**base, "owner_principal_id": "same-handle"}, user
    )
    assert not row_is_visible_without_grant(
        {**base, "owner_principal_id": "same-handle"}, agent
    )
    assert ("agent", "same-handle") not in user.grant_identities
    assert ("user", "same-handle") not in agent.grant_identities


@test("operational_api", "reused tool call ids keep message and detail deep-links run-local")
async def t_run_local_tool_deep_links(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-tool-links-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        conn = db._conn
        assert conn is not None
        gateway = SimpleNamespace(
            agent=SimpleNamespace(memory_db=db),
            _operational_history_ready=True,
        )
        try:
            await db.upsert_session(
                "run-local-tools",
                client_id="alice",
                title="Run-local tools",
            )
            runs = []
            for suffix, result in (("a", "first result"), ("b", "second result")):
                runs.append(
                    {
                        "run_id": f"run-{suffix}",
                        "status": "COMPLETED",
                        "created_at": 1_700_000_000 + len(runs),
                        "messages": [
                            {
                                "id": f"tool-message-{suffix}",
                                "role": "tool",
                                "tool_call_id": "reused-call",
                                "content": result,
                            }
                        ],
                        "tools": [
                            {
                                "tool_call_id": "reused-call",
                                "tool_name": "shell_execute",
                                "tool_args": {"run": suffix},
                                "result": result,
                                "status": "completed",
                            }
                        ],
                    }
                )
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps(runs), 1_700_000_100, "run-local-tools"),
            )
            await db._project_operational_session("run-local-tools")
            await conn.commit()
            tenant = str(
                (
                    await (
                        await conn.execute(
                            "SELECT tenant_id FROM sessions_v2 WHERE id=?",
                            ("run-local-tools",),
                        )
                    ).fetchone()
                )[0]
            )
            request = lambda **kwargs: _Request(
                gateway,
                tenant=tenant,
                handle="alice",
                device="alice-device",
                **kwargs,
            )

            messages_response = await operational.handle_session_messages(
                request(
                    match={"session_id": "run-local-tools"},
                    query={"limit": "100"},
                )
            )
            assert messages_response.status == 200, messages_response.text
            messages = {
                item["id"]: item["tool_invocation_id"]
                for item in _payload(messages_response)["messages"]
            }
            tools = await (
                await conn.execute(
                    "SELECT id, session_run_id FROM tool_invocations "
                    "WHERE session_id=? ORDER BY session_run_id",
                    ("run-local-tools",),
                )
            ).fetchall()
            query_plan = await (
                await conn.execute(
                    "EXPLAIN QUERY PLAN SELECT t.id FROM tool_invocations t "
                    "WHERE t.root_kind='session' AND t.session_id=? "
                    "AND t.session_run_id=? AND t.tool_call_id=? LIMIT 1",
                    (
                        "run-local-tools",
                        "run:run-local-tools:run-a",
                        "reused-call",
                    ),
                )
            ).fetchall()
            assert any(
                "uq_tool_invocations_session_run_call_context" in str(row[3])
                for row in query_plan
            ), [str(row[3]) for row in query_plan]
            expected = {
                "msg:run-local-tools:tool-message-a": str(tools[0][0]),
                "msg:run-local-tools:tool-message-b": str(tools[1][0]),
            }
            assert {key: messages[key] for key in expected} == expected

            for index, tool in enumerate(tools):
                detail_response = await operational.handle_tool_invocation(
                    request(match={"tool_id": str(tool[0])})
                )
                assert detail_response.status == 200, detail_response.text
                detail = _payload(detail_response)
                suffix = "a" if index == 0 else "b"
                assert detail["message_id"] == (
                    f"msg:run-local-tools:tool-message-{suffix}"
                )
        finally:
            await operational.stop_background_maintenance(gateway)
            await db.close()


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
                    "opaque AbCdEf0123456789AbCdEf0123456789AbCdEf01\n"
                    '{"api_key":"NEVER_INDEX_JSON_SHORT"}\n'
                    "OPENAGENT_CREDENTIAL=NEVER_INDEX_ENV_CREDENTIAL\n"
                    "private_key=NEVER_INDEX_PRIVATE_KEY\n"
                    '{"headers":{"X-Custom":"NEVER_INDEX_HEADER_CONTAINER",'
                    '"Accept":"text/plain"}}\n'
                    "Cookie: session=public; auth=NEVER_INDEX_COOKIE_SECRET\n"
                    "local path /Users/private-user/NEVER_INDEX_PRIVATE_PATH/file.txt\n"
                    "Authorization: Basic NEVER_INDEX_BASIC_SHORT",
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
        "nodes": [{"id": "node-1", "type": "tool", "config": {"label": "Orchid step", "prompt": "orchid workflowconfigneedle prompt sk-ant-NEVER_INDEX_WORKFLOW_TOKEN123456789"}}],
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
            from src.memory.operational.access import AccessContext
            from src.memory.operational.service import search_rows_visible

            search_snapshot = next(
                iter(gateway._operational_cursor_state.search.values())
            )
            canonical_rows = [
                dict(snapshot_row["_matches"][0])
                for snapshot_row in search_snapshot.rows
            ]
            alice_access = AccessContext.from_request(
                _Request(
                    gateway,
                    tenant=tenant,
                    handle="alice",
                    device="alice-device",
                )
            )
            mixed_statements: list[str] = []
            await db._conn.set_trace_callback(mixed_statements.append)
            try:
                assert all(
                    await search_rows_visible(
                        db._conn,
                        canonical_rows,
                        alice_access,
                    )
                )
            finally:
                await db._conn.set_trace_callback(None)
            assert sum(
                "operational-search-canonical-batch" in statement
                for statement in mixed_statements
            ) == 1
            target_fields = {
                "chat": ("session_id",),
                "chat_message": ("session_id", "message_id"),
                "chat_tool": (
                    "session_id",
                    "message_id",
                    "tool_invocation_id",
                ),
                "workflow_definition": ("workflow_id",),
                "workflow_run": ("workflow_run_id", "workflow_id"),
                "scheduled_definition": ("scheduled_task_id",),
                "scheduled_run": ("scheduled_run_id", "scheduled_task_id"),
                "event_definition": ("event_id",),
                "event_delivery": ("event_delivery_id", "event_id"),
            }
            forged_rows: list[dict] = []
            for canonical_row in canonical_rows:
                for field in target_fields[str(canonical_row["target_kind"])]:
                    forged_rows.append({**canonical_row, field: f"forged-{field}"})
            assert not any(
                await search_rows_visible(db._conn, forged_rows, alice_access)
            )
            hierarchy_forged_rows: list[dict] = []
            for canonical_row in canonical_rows:
                hierarchy_forged_rows.append(
                    {**canonical_row, "root_kind": "forged-root-kind"}
                )
                hierarchy_forged_rows.append(
                    {**canonical_row, "root_id": "forged-root-id"}
                )
                if canonical_row.get("parent_type") is None:
                    hierarchy_forged_rows.append(
                        {
                            **canonical_row,
                            "parent_type": "session",
                            "parent_id": "forged-parent-id",
                        }
                    )
                else:
                    hierarchy_forged_rows.append(
                        {**canonical_row, "parent_id": "forged-parent-id"}
                    )
            assert not any(
                await search_rows_visible(
                    db._conn,
                    hierarchy_forged_rows,
                    alice_access,
                )
            )
            tool_row = next(
                row for row in canonical_rows if row["target_kind"] == "chat_tool"
            )
            alternate_tool_message = await (
                await db._conn.execute(
                    "SELECT m.id FROM session_messages AS m "
                    "JOIN tool_invocations AS t ON t.id=? "
                    "WHERE m.session_id=t.session_id "
                    "AND m.run_id=t.session_run_id AND m.id<>? "
                    "AND m.visibility='user_visible' ORDER BY m.sequence LIMIT 1",
                    (tool_row["resource_id"], tool_row["message_id"]),
                )
            ).fetchone()
            assert alternate_tool_message is not None
            assert not (
                await search_rows_visible(
                    db._conn,
                    [
                        {
                            **tool_row,
                            "message_id": str(alternate_tool_message[0]),
                        }
                    ],
                    alice_access,
                )
            )[0]
            stale_source_rows = [
                {**row, "source_version": int(row["source_version"]) + 1}
                for row in canonical_rows
            ]
            stale_acl_rows = [
                {**row, "acl_version": int(row["acl_version"]) + 1}
                for row in canonical_rows
            ]
            assert not any(
                await search_rows_visible(
                    db._conn,
                    stale_source_rows,
                    alice_access,
                )
            )
            assert not any(
                await search_rows_visible(
                    db._conn,
                    stale_acl_rows,
                    alice_access,
                )
            )
            chat_row = next(
                row for row in canonical_rows if row["target_kind"] == "chat"
            )
            malformed_version_rows = [
                {
                    **chat_row,
                    "acl_version": f"{int(chat_row['acl_version'])}evil",
                },
                {
                    **chat_row,
                    "source_version": f"{int(chat_row['source_version'])}junk",
                },
                {
                    **chat_row,
                    "acl_version": float(chat_row["acl_version"]) + 0.9,
                },
                {
                    **chat_row,
                    "source_version": float(chat_row["source_version"]) + 0.2,
                },
                {**chat_row, "acl_version": True},
            ]
            assert not any(
                await search_rows_visible(
                    db._conn,
                    malformed_version_rows,
                    alice_access,
                )
            )

            # Legacy automation has one installation tenant.  Neither a
            # forged derived tenant nor an attacker-supplied AccessContext can
            # manufacture the previous installation_shared fallback.
            workflow_row = next(
                row
                for row in canonical_rows
                if row["target_kind"] == "workflow_definition"
            )
            forged_tenant_row = {**workflow_row, "tenant_id": "attacker-network"}
            attacker_access = AccessContext.from_request(
                _Request(
                    gateway,
                    tenant="attacker-network",
                    handle="mallory",
                    device="mallory-device",
                )
            )
            assert not (
                await search_rows_visible(
                    db._conn,
                    [forged_tenant_row],
                    attacker_access,
                )
            )[0]
            assert not (
                await search_rows_visible(
                    db._conn,
                    [workflow_row],
                    attacker_access,
                )
            )[0]
            assert not (
                await search_rows_visible(
                    db._conn,
                    [chat_row],
                    attacker_access,
                )
            )[0]
            workflow_target = next(item["target"] for item in page["items"] if item["target"]["kind"] == "workflow_run")
            definition_target = next(item["target"] for item in page["items"] if item["target"]["kind"] == "workflow_definition")
            assert "trace_step_id" not in workflow_target
            assert "node_id" not in definition_target and "field" not in definition_target
            assert page["items"] and page["snapshot"]["indexed_seq"] >= 1

            config_search = await operational.handle_search(
                _Request(
                    gateway,
                    tenant=tenant,
                    handle="alice",
                    device="alice-device",
                    body={
                        "query": "workflowconfigneedle",
                        "scopes": ["workflows"],
                        "filters": {},
                        "sort": "relevance",
                        "grouping": "match",
                        "limit": 10,
                        "cursor": None,
                    },
                )
            )
            assert config_search.status == 200, config_search.text
            assert {
                item["target"]["kind"]
                for item in _payload(config_search)["items"]
            } == {"workflow_definition"}

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
                b"NEVER_INDEX_JSON_SHORT",
                b"NEVER_INDEX_BASIC_SHORT",
                b"NEVER_INDEX_ENV_CREDENTIAL",
                b"NEVER_INDEX_PRIVATE_KEY",
                b"NEVER_INDEX_HEADER_CONTAINER",
                b"NEVER_INDEX_COOKIE_SECRET",
                b"NEVER_INDEX_PRIVATE_PATH",
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

            # Canonical run authority includes the definition it names.  Even
            # if a damaged/imported database has foreign keys disabled, a
            # stale run and search projection may not fall back to
            # installation-shared after its parent has disappeared.
            orphan_candidates = [
                row
                for row in canonical_rows
                if row["target_kind"]
                in {"workflow_run", "scheduled_run", "event_delivery"}
            ]
            await db._conn.commit()
            await db._conn.execute("PRAGMA foreign_keys=OFF")
            try:
                await db._conn.execute(
                    "DELETE FROM workflow_tasks WHERE id='orchid-workflow'"
                )
                await db._conn.execute(
                    "DELETE FROM scheduled_tasks WHERE id='orchid-task'"
                )
                await db._conn.execute(
                    "DELETE FROM events WHERE id='orchid-event'"
                )
                await db._conn.commit()
                surviving_runs = await (
                    await db._conn.execute(
                        "SELECT "
                        "(SELECT COUNT(*) FROM workflow_runs "
                        " WHERE id='orchid-workflow-run') + "
                        "(SELECT COUNT(*) FROM task_runs "
                        " WHERE id='orchid-task-run') + "
                        "(SELECT COUNT(*) FROM event_deliveries "
                        " WHERE id='orchid-delivery')"
                    )
                ).fetchone()
                assert int(surviving_runs[0]) == 3
                assert not any(
                    await search_rows_visible(
                        db._conn,
                        orphan_candidates,
                        alice_access,
                    )
                )
            finally:
                await db._conn.execute("PRAGMA foreign_keys=ON")
        finally:
            if "gateway" in locals():
                await operational.stop_background_maintenance(gateway)
            await db.close()


@test("operational_api", "a ready index from another canonical database is rebuilt")
async def t_foreign_ready_index_is_rebuilt(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB
    from src.memory.operational.search import operational_search_path, sync_operational_search

    with TemporaryDirectory(prefix="openagent-operational-replacement-") as directory:
        path = Path(directory) / "openagent.db"
        old_db = MemoryDB(str(path))
        await old_db.connect()
        await old_db._conn.execute(
            "INSERT INTO network(singleton,role,network_id,name,created_at) "
            "VALUES(1,'coordinator','replacement-network','old',0)"
        )
        await old_db._conn.commit()
        await old_db.upsert_session(
            "same-resource", client_id="alice", title="oldrestoreleakneedle"
        )
        old_status = await sync_operational_search(old_db, limit=1000)
        assert old_status.ready
        search_path = operational_search_path(path)
        await old_db.close()
        os.replace(path, path.with_name("old-canonical.db"))

        new_db = MemoryDB(str(path))
        await new_db.connect()
        gateway = SimpleNamespace(agent=SimpleNamespace(memory_db=new_db))
        try:
            await new_db._conn.execute(
                "INSERT INTO network(singleton,role,network_id,name,created_at) "
                "VALUES(1,'coordinator','replacement-network','new',0)"
            )
            await new_db._conn.commit()
            await new_db.upsert_session(
                "same-resource", client_id="alice", title="newcanonicalneedle"
            )
            request = _Request(
                gateway,
                tenant="replacement-network",
                handle="alice",
                device="alice-device",
            )
            await _ready_capabilities(operational, request)
            old_result = _payload(
                await operational.handle_search(
                    _Request(
                        gateway,
                        tenant="replacement-network",
                        handle="alice",
                        device="alice-device",
                        body={
                            "query": "oldrestoreleakneedle",
                            "scopes": ["chats"],
                            "filters": {},
                            "sort": "relevance",
                            "grouping": "match",
                            "limit": 10,
                            "cursor": None,
                        },
                    )
                )
            )
            assert old_result["items"] == []
            new_result = _payload(
                await operational.handle_search(
                    _Request(
                        gateway,
                        tenant="replacement-network",
                        handle="alice",
                        device="alice-device",
                        body={
                            "query": "newcanonicalneedle",
                            "scopes": ["chats"],
                            "filters": {},
                            "sort": "relevance",
                            "grouping": "match",
                            "limit": 10,
                            "cursor": None,
                        },
                    )
                )
            )
            assert len(new_result["items"]) == 1
            assert search_path.is_file()
            for physical in search_path.parent.glob(f"{search_path.name}*"):
                if physical.is_file():
                    assert b"oldrestoreleakneedle" not in physical.read_bytes(), physical.name
        finally:
            await operational.stop_background_maintenance(gateway)
            await new_db.close()


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

            tool_search = {
                "query": "orchid_tool", "scopes": ["tools"], "filters": {},
                "sort": "relevance", "grouping": "match", "limit": 10, "cursor": None,
            }
            assert _payload(await operational.handle_search(bob(body=tool_search)))["items"] == []
            await db._conn.execute(
                "INSERT INTO resource_acl "
                "(tenant_id,resource_type,resource_id,principal_type,principal_id,"
                "permission,acl_version,granted_by_principal_id,granted_at_ms) "
                "VALUES(?,?,?,?,?,'admin',1,?,?)",
                (
                    tenant,
                    "session",
                    "orchid-chat",
                    "user",
                    "bob",
                    "user:alice",
                    int(time.time() * 1000),
                ),
            )
            await db._conn.commit()
            granted_tools = _payload(
                await operational.handle_search(bob(body=tool_search))
            )["items"]
            assert len(granted_tools) == 1
            assert granted_tools[0]["target"]["kind"] == "chat_tool"

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


@test(
    "operational_api",
    "canonical search authorization is batched across snapshots and cursors",
)
async def t_batched_search_authorization_query_budget(_ctx: TestContext) -> None:
    """A 40-result packaged-style query must not issue SQL per candidate.

    The trace contains synthetic IDs only.  It also proves that current grants,
    ACL revocation, and source-version invalidation are still canonical on a
    cursor continuation rather than inherited from the derived snapshot.
    """

    from src.gateway.api import operational
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-batch-auth-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        statements: list[str] = []
        gateway = None
        try:
            tenant, gateway = await _seed_complete_fixture(db)
            for index in range(192):
                await db.upsert_session(
                    f"batch-auth-{index:03d}",
                    client_id="alice",
                    title=f"batchauthorizationneedle unique{index:03d}",
                )
            alice = lambda **kwargs: _Request(
                gateway,
                tenant=tenant,
                handle="alice",
                device="alice-device",
                **kwargs,
            )
            bob = lambda **kwargs: _Request(
                gateway,
                tenant=tenant,
                handle="bob",
                device="bob-device",
                **kwargs,
            )
            await _ready_capabilities(operational, alice())
            await operational.stop_background_maintenance(gateway)

            # Explicit grants use the same canonical batch decision as owner
            # visibility.  The derived FTS owner metadata alone cannot admit
            # Bob, while this current search grant can admit exactly one row.
            await db._conn.execute(
                "INSERT INTO resource_acl "
                "(tenant_id,resource_type,resource_id,principal_type,principal_id,"
                "permission,acl_version,granted_by_principal_id,granted_at_ms) "
                "VALUES(?,?,?,?,?,'search',1,?,?)",
                (
                    tenant,
                    "session",
                    "batch-auth-000",
                    "user",
                    "bob",
                    "user:alice",
                    int(time.time() * 1000),
                ),
            )
            await db._conn.commit()
            granted = _payload(
                await operational.handle_search(
                    bob(
                        body={
                            "query": "unique000",
                            "scopes": ["chats"],
                            "filters": {},
                            "sort": "relevance",
                            "grouping": "match",
                            "limit": 10,
                            "cursor": None,
                        }
                    )
                )
            )
            assert [item["root"]["id"] for item in granted["items"]] == [
                "batch-auth-000"
            ]
            from src.memory.operational.access import AccessContext
            from src.memory.operational.service import (
                granted_search_resources,
                search_rows_visible,
            )

            bob_snapshot = next(
                snapshot
                for snapshot in gateway._operational_cursor_state.search.values()
                if snapshot.principal_id == "user:bob"
            )
            bob_row = dict(bob_snapshot.rows[0]["_matches"][0])
            stale_prefilter_grants = await granted_search_resources(
                db._conn,
                AccessContext.from_request(bob()),
            )
            assert ("session", "batch-auth-000", 1) in stale_prefilter_grants
            await db._conn.execute(
                "DELETE FROM resource_acl WHERE tenant_id=? "
                "AND resource_type='session' AND resource_id='batch-auth-000' "
                "AND principal_type='user' AND principal_id='bob'",
                (tenant,),
            )
            await db._conn.commit()
            # A projection fetched for FTS prefiltering before the revoke must
            # not authorize the canonical batch that precedes serialization.
            assert not (
                await search_rows_visible(
                    db._conn,
                    [bob_row],
                    AccessContext.from_request(bob()),
                )
            )[0]

            await db._conn.set_trace_callback(statements.append)
            body = {
                "query": "batchauthorizationneedle",
                "scopes": ["chats"],
                "filters": {},
                "sort": "relevance",
                "grouping": "match",
                "limit": 40,
                "cursor": None,
            }
            first = _payload(await operational.handle_search(alice(body=body)))
            assert len(first["items"]) == 40
            assert first["has_more"] and first["next_cursor"]

            canonical_tables = (
                " FROM SESSIONS_V2 ",
                " FROM SESSION_MESSAGES ",
                " FROM TOOL_INVOCATIONS ",
                " FROM OPERATIONAL_RESOURCE_OWNERS ",
                " FROM OPERATIONAL_AUTOMATION_PROJECTION ",
                " FROM RESOURCE_ACL ",
            )

            def canonical_select_count() -> int:
                normalized = [
                    f" {' '.join(statement.upper().split())} "
                    for statement in statements
                    if statement.lstrip().upper().startswith("SELECT")
                ]
                direct = sum(
                    any(table in statement for table in canonical_tables)
                    for statement in normalized
                )
                compound = sum(
                    "operational-search-canonical-batch" in statement
                    for statement in statements
                )
                return direct + compound

            def compound_count() -> int:
                return sum(
                    "operational-search-canonical-batch" in statement
                    for statement in statements
                )

            # One grants read plus one batched session read for snapshot
            # creation and one for page-time reauthorization.  Leave small
            # headroom for harmless metadata reads, but never candidate-scale
            # growth (the old N+1 path is >380 statements for this fixture).
            assert canonical_select_count() <= 8, statements
            assert compound_count() == 2, statements

            decoded = gateway._operational_cursor_state.decode(first["next_cursor"])
            snapshot = gateway._operational_cursor_state.search[decoded["s"]]
            # JSON1 batching has one compound statement through 1,000
            # candidates and exactly one additional statement at the bound.
            repeated = [
                dict(snapshot.rows[index % len(snapshot.rows)])
                for index in range(1_001)
            ]
            statements.clear()
            repeated_visibility = await search_rows_visible(
                db._conn,
                repeated,
                AccessContext.from_request(alice()),
            )
            assert all(repeated_visibility)
            assert compound_count() == 2, statements

            revoked_id = str(snapshot.rows[45]["resource_id"])
            stale_id = str(snapshot.rows[46]["resource_id"])
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
            statements.clear()
            body["cursor"] = first["next_cursor"]
            second = _payload(await operational.handle_search(alice(body=body)))
            returned = {item["root"]["id"] for item in second["items"]}
            assert len(second["items"]) == 40
            assert revoked_id not in returned and stale_id not in returned
            assert second["snapshot"]["search_session_id"] == first["snapshot"][
                "search_session_id"
            ]
            assert canonical_select_count() <= 8, statements
            assert compound_count() == 1, statements

            # A tiny page may materialize the bounded snapshot once, but its
            # pre-snippet recheck must not rehydrate every remaining match.
            # Record only cardinalities; no query or result content leaves
            # this synthetic test process.
            original_batch = operational._search_rows_visible
            batch_sizes: list[int] = []

            async def recorded_batch(conn, rows, access):
                batch_sizes.append(len(rows))
                return await original_batch(conn, rows, access)

            operational._search_rows_visible = recorded_batch
            try:
                one_item_body = {**body, "limit": 1, "cursor": None}
                one_item = _payload(
                    await operational.handle_search(alice(body=one_item_body))
                )
            finally:
                operational._search_rows_visible = original_batch
            assert len(one_item["items"]) == 1
            assert len(batch_sizes) == 2, batch_sizes
            assert batch_sizes[0] > 128
            assert 0 < batch_sizes[1] <= 128 < batch_sizes[0]
        finally:
            if db._conn is not None:
                await db._conn.set_trace_callback(None)
            if gateway is not None:
                await operational.stop_background_maintenance(gateway)
            await db.close()


@test("operational_api", "deleted session immediately revokes tool search and detail")
async def t_deleted_session_revokes_tool(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB
    from src.memory.operational.access import AccessContext
    from src.memory.operational.repository import project_legacy_session_async

    with TemporaryDirectory(prefix="openagent-operational-tool-revoke-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        tenant, gateway = await _seed_complete_fixture(db)
        request = lambda **kwargs: _Request(
            gateway,
            tenant=tenant,
            handle="alice",
            device="alice-device",
            **kwargs,
        )
        try:
            await _ready_capabilities(operational, request())
            await operational.stop_background_maintenance(gateway)
            response = await operational.handle_search(
                request(
                    body={
                        "query": "orchid_tool",
                        "scopes": ["tools"],
                        "filters": {},
                        "sort": "relevance",
                        "grouping": "match",
                        "limit": 10,
                        "cursor": None,
                    }
                )
            )
            assert response.status == 200, response.text
            snapshot = next(iter(gateway._operational_cursor_state.search.values()))
            tool_row = next(row for row in snapshot.rows if row["target_kind"] == "chat_tool")

            activity_time = int(
                (
                    await (
                        await db._conn.execute(
                            "SELECT occurred_at_ms FROM activity_items "
                            "WHERE resource_type='session' AND resource_id='orchid-chat'"
                        )
                    ).fetchone()
                )[0]
            )
            await db._conn.execute("DELETE FROM sessions WHERE session_id='orchid-chat'")
            await project_legacy_session_async(
                db._conn, "orchid-chat", now_ms=activity_time + 1
            )
            await db._conn.commit()

            access = AccessContext.from_request(request())
            assert not await operational._search_row_visible(db._conn, tool_row, access)
            detail = await operational.handle_tool_invocation(
                request(match={"tool_id": str(tool_row["resource_id"])})
            )
            assert detail.status == 404, detail.text
        finally:
            await operational.stop_background_maintenance(gateway)
            await db.close()


@test("operational_api", "history candidate limit is applied after canonical ACL prefilter")
async def t_history_limit_after_acl(_ctx: TestContext) -> None:
    from src.gateway.api import operational
    from src.memory.db import MemoryDB
    from src.memory.operational.access import AccessContext

    with TemporaryDirectory(prefix="openagent-operational-history-acl-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            for index in range(3):
                await db.upsert_session(
                    f"bob-history-{index}", client_id="bob", title="Bob"
                )
            await db.upsert_session("alice-history", client_id="alice", title="Alice")
            tenant = str(
                (await (await db._conn.execute("SELECT tenant_id FROM sessions_v2 LIMIT 1")).fetchone())[0]
            )
            access = AccessContext.from_request(
                _Request(
                    SimpleNamespace(),
                    tenant=tenant,
                    handle="alice",
                    device="alice-device",
                )
            )
            filters = {
                "kinds": [],
                "status": [],
                "origin": None,
                "parent_type": None,
                "parent_id": None,
                "from": None,
                "to": None,
                "include_children": False,
            }
            old_limit = operational._MAX_SNAPSHOT_ITEMS
            operational._MAX_SNAPSHOT_ITEMS = 2
            try:
                rows = await operational._activity_rows(db._conn, access, filters)
            finally:
                operational._MAX_SNAPSHOT_ITEMS = old_limit
            assert [str(row["resource_id"]) for row in rows] == ["alice-history"]
        finally:
            await db.close()


@test("operational_api", "future source timestamps cannot poison session tombstones")
async def t_future_timestamp_tombstone(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.memory.operational.repository import projection_coverage_async

    with TemporaryDirectory(prefix="openagent-operational-future-delete-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            now = time.time()
            await db.upsert_session("future-delete", client_id="alice", title="Future")
            runs = [
                {
                    "run_id": "future-run",
                    "status": "COMPLETED",
                    "created_at": now + 86_400,
                    "messages": [{"id": "future-message", "role": "user", "content": "future"}],
                }
            ]
            await db._conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id='future-delete'",
                (json.dumps(runs), int(now + 86_400)),
            )
            await db._project_operational_session("future-delete")
            await db._conn.commit()
            await db.delete_session("future-delete")

            legacy = await (
                await db._conn.execute(
                    "SELECT 1 FROM sessions WHERE session_id='future-delete'"
                )
            ).fetchone()
            canonical = await (
                await db._conn.execute(
                    "SELECT deleted_at_ms FROM sessions_v2 WHERE id='future-delete'"
                )
            ).fetchone()
            coverage = await projection_coverage_async(db._conn)
            assert legacy is None
            assert canonical is not None and canonical[0] is not None
            assert int(coverage["pending_sessions"]) == 0
        finally:
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
            # Reproduce the commit-to-cache publication window deterministically:
            # the index is current, but the gateway still advertises the previous
            # consumer loop's sequence.  The request must refresh before returning
            # a false warming response.
            gateway._operational_search_status = dict(before)
            refreshed = await operational._current_search_status(
                db, db._conn, gateway=gateway
            )
            assert refreshed["ready"] is True
            assert int(refreshed["seq"]) >= outbox_head
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


@test("operational_api", "consumed outbox keeps only the replayable latest resource rows")
async def t_search_outbox_compaction(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.memory.operational.repository import project_legacy_session_async
    from src.memory.operational.search import sync_operational_search

    with TemporaryDirectory(prefix="openagent-operational-outbox-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        try:
            now = time.time()
            await db.upsert_session("outbox-chat", client_id="alice", title="Outbox")
            for turn in range(1, 6):
                messages = [
                    {"id": f"message-{index}", "role": "user", "content": f"turn {index}"}
                    for index in range(turn)
                ]
                runs = [
                    {
                        "run_id": "outbox-run",
                        "status": "COMPLETED",
                        "created_at": now,
                        "messages": messages,
                    }
                ]
                await db._conn.execute(
                    "UPDATE sessions SET runs=?, updated_at=? WHERE session_id='outbox-chat'",
                    (json.dumps(runs), int(now) + turn),
                )
                await db._project_operational_session("outbox-chat")
                await db._conn.commit()

            before = int(
                (await (await db._conn.execute("SELECT COUNT(*) FROM search_outbox")).fetchone())[0]
            )
            resource_counts = {
                (str(row[0]), str(row[1])): int(row[2])
                for row in await (
                    await db._conn.execute(
                        "SELECT source_kind, source_id, COUNT(*) FROM search_outbox "
                        "GROUP BY source_kind, source_id"
                    )
                ).fetchall()
            }
            # The initial session plus five incremental turns produce one new
            # session revision and one newly appended message per turn.  An
            # unchanged message must not be republished on every projection.
            assert before == 11
            assert resource_counts[("session", "outbox-chat")] == 6
            message_counts = [
                count
                for (kind, _resource_id), count in resource_counts.items()
                if kind == "message"
            ]
            assert len(message_counts) == 5
            assert set(message_counts) == {1}
            status = await sync_operational_search(db, limit=1000)
            assert status.ready
            after = int(
                (await (await db._conn.execute("SELECT COUNT(*) FROM search_outbox")).fetchone())[0]
            )
            # One replayable upsert for the session and each of its five messages.
            assert after == 6

            outbox_head = int(
                (
                    await (
                        await db._conn.execute(
                            "SELECT COALESCE(MAX(seq), 0) FROM search_outbox"
                        )
                    ).fetchone()
                )[0]
            )
            await db._conn.execute(
                "UPDATE sessions SET metadata=?, updated_at=? "
                "WHERE session_id='outbox-chat'",
                (
                    json.dumps({"client_id": "alice", "title": "Renamed outbox"}),
                    int(now) + 10,
                ),
            )
            renamed = await project_legacy_session_async(
                db._conn, "outbox-chat"
            )
            await db._conn.commit()
            assert renamed is not None and renamed.delta.nested_writes == 0
            inherited_replays = {
                str(row[0]): int(row[1])
                for row in await (
                    await db._conn.execute(
                        "SELECT source_kind, COUNT(*) FROM search_outbox "
                        "WHERE seq>? GROUP BY source_kind",
                        (outbox_head,),
                    )
                ).fetchall()
            }
            # Message documents inherit title/owner/visibility/ACL from the
            # root session. Re-publish them even though their canonical rows
            # stayed byte-identical, or derived search authorization goes stale.
            assert inherited_replays == {"message": 5, "session": 1}
        finally:
            await db.close()


@test("operational_api", "extractor version changes purge and replay the derived index")
async def t_search_version_boundary_rebuild(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.memory.operational.search import (
        _open_index,
        operational_search_path,
        operational_search_status,
        sync_operational_search,
    )

    with TemporaryDirectory(prefix="openagent-operational-index-version-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        try:
            await db.upsert_session(
                "version-replay-chat",
                client_id="alice",
                title="currentreplayneedle",
            )
            initial = await sync_operational_search(db, limit=1000)
            assert initial.ready

            search_path = operational_search_path(path)
            index = _open_index(search_path)
            try:
                index.execute(
                    "UPDATE search_fts SET body_safe='staleversionleakneedle'"
                )
                index.execute(
                    "UPDATE search_index_state SET redaction_version='obsolete' "
                    "WHERE singleton_id=1"
                )
                index.commit()
            finally:
                index.close()

            incompatible = await operational_search_status(db)
            assert incompatible["ready"] is False
            rebuilt = await sync_operational_search(db, limit=1000)
            assert rebuilt.ready

            index = _open_index(search_path)
            try:
                assert index.execute(
                    "SELECT COUNT(*) FROM search_fts "
                    "WHERE search_fts MATCH 'currentreplayneedle'"
                ).fetchone()[0] == 1
                assert index.execute(
                    "SELECT COUNT(*) FROM search_fts "
                    "WHERE search_fts MATCH 'staleversionleakneedle'"
                ).fetchone()[0] == 0
            finally:
                index.close()
            for physical in search_path.parent.glob(f"{search_path.name}*"):
                if physical.is_file():
                    assert b"staleversionleakneedle" not in physical.read_bytes(), physical.name
        finally:
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
                {**base, "filters": {"status": ["running"] * 100}},
                {**base, "filters": {"root": []}},
                {**base, "filters": {"parent_type": "workflow"}},
                {**base, "filters": {"parent_type": {"bad": True}, "parent_id": "x"}},
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
                punctuation = await operational.handle_search(
                    _Request(
                        gateway,
                        tenant=tenant,
                        handle="alice",
                        device="alice-device",
                        body={**base, "query": "... --- !!!"},
                    )
                )
                assert punctuation.status == 422, punctuation.text
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


@test("operational_api", "root-grouped matches count against snapshot row quota")
async def t_grouped_snapshot_match_quota(_ctx: TestContext) -> None:
    from src.gateway.api import operational

    state = operational._CursorState()
    old_limit = operational._MAX_SNAPSHOT_ROWS_GLOBAL
    operational._MAX_SNAPSHOT_ROWS_GLOBAL = 3
    try:
        first = operational._SearchSnapshot(
            "first",
            "tenant",
            "user:alice",
            "request",
            "generation",
            1,
            ({"_matches": [{"id": 1}, {"id": 2}]},),
            int(time.time() * 1000) + 10_000,
        )
        second = operational._SearchSnapshot(
            "second",
            "tenant",
            "user:alice",
            "request",
            "generation",
            1,
            ({"_matches": [{"id": 3}, {"id": 4}]},),
            int(time.time() * 1000) + 10_000,
        )
        state.put_search(first)
        state.put_search(second)
        assert set(state.search) == {"second"}
    finally:
        operational._MAX_SNAPSHOT_ROWS_GLOBAL = old_limit


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
