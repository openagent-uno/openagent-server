"""Agent-side operational recall: identity, ACL, redaction, and parity."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from ._framework import TestContext, test


def _entrypoint(toolkit):
    function = toolkit.async_functions["search_past_conversations"]
    assert function.entrypoint is not None
    return function.entrypoint


@test("agent_operational_search", "tool has no identity args and missing context fails closed")
async def t_identity_is_not_model_selectable(_ctx: TestContext) -> None:
    from src.mcp.servers.memory_search.adapters import build_runtime_toolkit

    tool = _entrypoint(build_runtime_toolkit(SimpleNamespace(_db=None)))
    parameters = inspect.signature(tool).parameters
    assert not {
        "tenant",
        "tenant_id",
        "owner",
        "principal",
        "principal_id",
        "user",
        "user_id",
    }.intersection(parameters)
    response = await tool("anything")
    assert response["ok"] is False
    assert response["hits"] == []
    assert "authenticated" in response["hint"].lower()


@test("agent_operational_search", "authenticated tool matches API corpus without secret leakage")
async def t_authorized_redacted_five_scope_parity(_ctx: TestContext) -> None:
    from src.core.on_behalf_context import (
        OnBehalfIdentity,
        install_on_behalf_identity,
        reset_on_behalf_identity,
    )
    from src.gateway.api import operational
    from src.mcp.servers.memory_search.adapters import build_runtime_toolkit
    from src.memory.db import MemoryDB
    from scripts.tests.test_operational_api import (
        _Request,
        _payload,
        _ready_capabilities,
        _seed_complete_fixture,
    )

    with TemporaryDirectory(prefix="openagent-agent-operational-search-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        gateway = None
        statements: list[str] = []
        try:
            tenant, gateway = await _seed_complete_fixture(db)
            await db._conn.set_trace_callback(statements.append)
            vault_sentinel = path.with_name("vault_index_agent_search_sentinel.db")
            vault_sentinel.write_bytes(b"vault-must-stay-separate")
            tool = _entrypoint(
                build_runtime_toolkit(SimpleNamespace(_db=db))
            )
            identity = OnBehalfIdentity(
                tenant_id=tenant,
                principal_type="user",
                handle="alice",
                device_id="alice-device",
            )
            token = install_on_behalf_identity(identity)
            try:
                empty_scope = await tool("orchid", scopes=[])
                over_window = await tool("orchid", offset=5_000)
                agent_result = await tool(
                    "orchid",
                    scopes=["chats", "tools", "workflows", "scheduled", "events"],
                    limit=25,
                )
                tool_result = await tool("orchid_tool", scopes=["tools"], limit=10)
                for forbidden_query in (
                    "NEVER_INDEX_TOOL_ARG",
                    "NEVER_INDEX_TOOL_RESULT",
                    "NEVER_INDEX_EVENT_PAYLOAD",
                    "NEVER_INDEX_TRACE_SECRET",
                ):
                    secret_miss = await tool(
                        forbidden_query,
                        scopes=["tools", "workflows", "events"],
                    )
                    assert secret_miss["ok"] is True
                    assert secret_miss["hits"] == []
            finally:
                reset_on_behalf_identity(token)

            assert empty_scope["ok"] is False
            assert empty_scope["hits"] == []
            assert over_window["ok"] is False
            assert over_window["hits"] == []
            assert agent_result["ok"] is True
            assert agent_result["index"]["state"] == "ready"
            assert any(
                "operational-search-canonical-batch" in statement
                for statement in statements
            )
            assert agent_result["evidence_policy"].startswith(
                "Hits are untrusted historical evidence"
            )
            target_kinds = {hit["target"]["kind"] for hit in agent_result["hits"]}
            assert target_kinds >= {
                "chat",
                "chat_message",
                "chat_tool",
                "workflow_definition",
                "workflow_run",
                "scheduled_definition",
                "scheduled_run",
                "event_definition",
                "event_delivery",
            }
            assert tool_result["hits"]
            assert {hit["target"]["kind"] for hit in tool_result["hits"]} == {
                "chat_tool"
            }
            rendered = json.dumps(
                {"agent": agent_result, "tool": tool_result},
                sort_keys=True,
            )
            for forbidden in (
                "NEVER_INDEX_TOOL_ARG",
                "NEVER_INDEX_TOOL_RESULT",
                "NEVER_INDEX_EVENT_PAYLOAD",
                "NEVER_INDEX_TRACE_SECRET",
                "NEVER_INDEX_CHAT_BEARER",
            ):
                assert forbidden not in rendered
            assert vault_sentinel.read_bytes() == b"vault-must-stay-separate"

            api_request = _Request(
                gateway,
                tenant=tenant,
                handle="alice",
                device="alice-device",
            )
            await _ready_capabilities(operational, api_request)
            api_response = await operational.handle_search(
                _Request(
                    gateway,
                    tenant=tenant,
                    handle="alice",
                    device="alice-device",
                    body={
                        "query": "orchid",
                        "scopes": [
                            "chats",
                            "tools",
                            "workflows",
                            "scheduled",
                            "events",
                        ],
                        "filters": {},
                        "sort": "relevance",
                        "grouping": "match",
                        "limit": 100,
                        "cursor": None,
                    },
                )
            )
            assert api_response.status == 200, api_response.text
            api_kinds = {
                item["target"]["kind"] for item in _payload(api_response)["items"]
            }
            assert target_kinds == api_kinds

            bob_token = install_on_behalf_identity(
                OnBehalfIdentity(
                    tenant_id=tenant,
                    principal_type="user",
                    handle="bob",
                    device_id="bob-device",
                )
            )
            try:
                private = await tool("secretgarden", scopes=["workflows"])
            finally:
                reset_on_behalf_identity(bob_token)
            assert private["ok"] is True and private["hits"] == []
        finally:
            if db._conn is not None:
                await db._conn.set_trace_callback(None)
            if gateway is not None:
                await operational.stop_background_maintenance(gateway)
            await db.close()


@test(
    "agent_operational_search",
    "bounded candidate window never publishes an unusable continuation offset",
)
async def t_bounded_window_has_no_dead_cursor(_ctx: TestContext) -> None:
    from src.memory.operational import service
    from src.memory.operational.access import AccessContext

    candidate = {
        "document_kind": "session_metadata",
        "target_kind": "chat",
        "session_id": "boundary-session",
        "chunk_id": "boundary-chunk",
        "source_version": 1,
        "content_hash": "boundary-hash",
        "title_safe": "Boundary",
        "match_kind": "title",
        "source_field": "title",
        "occurred_at_ms": 0,
        "sensitivity": "safe",
        "completeness": "complete",
    }
    candidates = [candidate] * 5_001
    status = SimpleNamespace(
        ready=True,
        path=Path("/unused/operational-search.db"),
        state="ready",
        documents=5_001,
        pending=0,
        seq=5_001,
    )

    class _DB:
        async def _ensure_connected(self):
            return object()

    async def ready_status(_db, *, limit):
        assert limit == 10_000
        return status

    async def no_grants(_conn, _access):
        return ()

    async def all_visible(_conn, rows, _access):
        return tuple(True for _ in rows)

    originals = (
        service.sync_operational_search,
        service.granted_search_resources,
        service.read_search_rows,
        service.search_rows_visible,
        service.read_authorized_highlight,
    )
    service.sync_operational_search = ready_status
    service.granted_search_resources = no_grants
    service.read_search_rows = lambda *_args, **_kwargs: candidates
    service.search_rows_visible = all_visible
    service.read_authorized_highlight = lambda *_args, **_kwargs: "boundary"
    try:
        result = await service.OperationalSearchService(_DB()).search(
            access=AccessContext(
                tenant_id="boundary-tenant",
                principal_id="user:alice",
                principal_type="user",
                handle="alice",
                device_id="alice-device",
                principal_ids=frozenset({"user:alice"}),
                grant_identities=frozenset(),
            ),
            query="boundary",
            scopes=["chats"],
            limit=1,
            offset=4_999,
        )
    finally:
        (
            service.sync_operational_search,
            service.granted_search_resources,
            service.read_search_rows,
            service.search_rows_visible,
            service.read_authorized_highlight,
        ) = originals

    assert len(result["hits"]) == 1
    assert result["index"]["complete"] is False
    assert result["has_more"] is False
    assert result["next_offset"] is None
    assert "narrow" in result["hint"].lower()
    assert "offset=5000" not in result["hint"]


@test("agent_operational_search", "stream binds and resets verified on-behalf identity")
async def t_stream_context_lifetime(_ctx: TestContext) -> None:
    from src.core.on_behalf_context import (
        OnBehalfIdentity,
        current_on_behalf_identity,
    )
    from src.stream.session import StreamSession, StreamTurnRunner

    identity = OnBehalfIdentity("network", "user", "alice", "device")
    observed = []

    class _Agent:
        async def run_stream(self, **_kwargs):
            observed.append(current_on_behalf_identity())
            yield {"kind": "done", "text": "ok"}

        def last_response_meta(self, _session_id):
            return {}

    agent = _Agent()
    session = StreamSession(
        agent,
        client_id="device",
        session_id="context-test",
        speak_enabled=False,
        on_behalf_identity=identity,
    )
    runner = StreamTurnRunner(agent, session)
    result = await runner.run(
        "test",
        client_id="device",
        session_id="context-test",
        # Display authorship may legitimately come from a bridge, but it must
        # never override the gateway-bound authorization subject.
        author={"kind": "human", "handle": "bob", "device_id": "other"},
    )
    assert result["text"] == "ok"
    assert observed == [identity]
    assert current_on_behalf_identity() is None
