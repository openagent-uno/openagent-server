"""Client-machine capability protocol, routing, and origin isolation."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from types import SimpleNamespace

from ._framework import TestContext, test


class _Ws:
    closed = False

    async def close(self, **_kwargs):
        self.closed = True


def _catalog(marker: str = "client") -> list[dict]:
    return [{
        "name": "filesystem",
        "version": "1.0.0",
        "tools": [{
            "name": "read_file",
            "description": f"read on {marker}",
            "classification": "read_only",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }],
    }]


def _shell_catalog() -> list[dict]:
    return [{
        "name": "shell",
        "version": "1.0.0",
        "tools": [
            {
                "name": "shell_exec",
                "description": "run on client",
                "classification": "mutating",
                "input_schema": {"type": "object"},
            },
            {
                "name": "shell_output",
                "description": "read client process output",
                "classification": "read_only",
                "input_schema": {"type": "object"},
            },
        ],
    }]


def _safety_catalog() -> list[dict]:
    return [{
        "name": "filesystem",
        "version": "1.0.0",
        "tools": [
            {
                "name": "read_file",
                "description": "safe to retry",
                "classification": "read_only",
                "input_schema": {"type": "object"},
            },
            {
                "name": "write_file",
                "description": "may have an effect",
                "classification": "mutating",
                "input_schema": {"type": "object"},
            },
            {
                "name": "ensure_file",
                "description": "declared idempotent",
                "classification": "idempotent",
                "input_schema": {"type": "object"},
            },
        ],
    }]


def _computer_catalog() -> list[dict]:
    return [{
        "name": "computer-control",
        "version": "1.0.0",
        "tools": [{
            "name": "computer",
            "description": "screen and input control",
            "classification": "mutating",
            "classification_by_argument": {
                "action": {
                    "get_screenshot": "read_only",
                    "get_cursor_position": "read_only",
                },
            },
            "input_schema": {"type": "object"},
        }],
    }]


@test("client_capabilities", "registry pins exact instance and preserves MCP result")
async def t_registry_exact_instance(ctx: TestContext) -> None:
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError

    registry = CapabilityRegistry()
    sent_a: list[dict] = []
    sent_b: list[dict] = []

    async def send_a(_ws, payload):
        sent_a.append(payload)
        return True

    async def send_b(_ws, payload):
        sent_b.append(payload)
        return True

    a = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop", generation=1,
        device_label="Mac", ws=_Ws(), send_json=send_a, servers=_catalog("A"),
        network_id="network-1",
    )
    b_ws = _Ws()
    b = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="cli", generation=3,
        device_label="Terminal", ws=b_ws, send_json=send_b, servers=_catalog("B"),
    )
    assert registry.origin_for("dev", "missing") is None
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None and origin.client_instance_id == "desktop"

    call = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "read_file", {"path": "/client"},
        session_id="chat:1", timeout_s=2,
    ))
    await asyncio.sleep(0)
    frame = sent_a[-1]
    assert frame["type"] == "client_tool_call"
    assert frame["generation"] == 1 and frame["session_id"] == "chat:1"
    assert frame["account_id"] == "network-1"
    assert frame["network_id"] == "network-1"
    assert sent_b == [], "an exact desktop call leaked to the CLI instance"
    registry.resolve_result(a, {
        "type": "client_tool_result",
        "call_id": frame["call_id"],
        "generation": 1,
        "result": {
            "content": [{"type": "text", "text": "from client"}],
            "structuredContent": {"path": "/client", "ok": True},
            "isError": False,
            "_meta": {"vendor": "test"},
        },
    })
    result = await call
    assert result["content"][0]["text"] == "from client"
    assert result["structuredContent"]["path"] == "/client"
    assert result["isError"] is False
    assert result["_meta"]["vendor"] == "test"
    assert result["execution_host"]["client_instance_id"] == "desktop"

    # An older generation cannot replace an exact live instance.
    try:
        await registry.register(
            device_id="dev", account_id="network-1", client_instance_id="cli", generation=2,
            device_label="stale", ws=_Ws(), send_json=send_b,
            servers=_catalog("stale"),
        )
        raise AssertionError("stale generation replaced the active host")
    except ClientCapabilityError as exc:
        assert exc.code == "STALE_GENERATION"

    # A same-generation reconnect for the exact instance replaces only that
    # socket; a different instance remains alive.
    dup_old_ws = _Ws()
    dup_old = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="duplicate", generation=1,
        device_label="old", ws=dup_old_ws, send_json=send_b,
        servers=_catalog("old"),
    )
    dup_new = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="duplicate", generation=1,
        device_label="new", ws=_Ws(), send_json=send_b,
        servers=_catalog("new"),
    )
    assert dup_old_ws.closed and registry.connection("dev", "duplicate") is dup_new
    assert registry.connection("dev", "cli") is b

    # A stale result is rejected and cannot complete the current call.
    origin_b = registry.origin_for("dev", "cli")
    assert origin_b is not None
    call_b = asyncio.create_task(registry.call_tool(
        origin_b, "filesystem", "read_file", {}, timeout_s=2,
    ))
    await asyncio.sleep(0)
    frame_b = sent_b[-1]
    try:
        registry.resolve_result(b, {
            "type": "client_tool_result", "call_id": frame_b["call_id"],
            "generation": 2, "result": {"bad": True},
        })
        raise AssertionError("stale result generation was accepted")
    except ClientCapabilityError as exc:
        assert exc.code == "STALE_GENERATION"
    registry.resolve_result(b, {
        "type": "client_tool_result", "call_id": frame_b["call_id"],
        "generation": 3, "result": {"ok": True},
    })
    assert (await call_b)["ok"] is True

    # Taking A offline must not choose still-online B as a fallback.
    await registry.unregister(a)
    try:
        await registry.call_tool(origin, "filesystem", "read_file", {}, timeout_s=0.1)
        raise AssertionError("offline origin unexpectedly dispatched")
    except ClientCapabilityError as exc:
        assert exc.code == "CLIENT_OFFLINE"
    await registry.close_device("dev")
    assert b_ws.closed and registry.connection("dev", "cli") is None


@test("client_capabilities", "artifact chunks are bounded, ordered, and digest-checked")
async def t_artifact_chunks(ctx: TestContext) -> None:
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError

    registry = CapabilityRegistry()
    sent: list[dict] = []

    async def send(_ws, payload):
        sent.append(payload)
        return True

    conn = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop", generation=1,
        device_label="Mac", ws=_Ws(), send_json=send, servers=_catalog(),
    )
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None
    task = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "read_file", {}, timeout_s=2,
    ))
    await asyncio.sleep(0)
    call_id = sent[-1]["call_id"]
    blob = b"\x00client-image\xff"
    registry.receive_artifact_chunk(conn, {
        "type": "client_artifact_chunk", "call_id": call_id,
        "generation": 1, "transfer_id": "img", "seq": 0,
        "data": base64.b64encode(blob[:5]).decode(), "eof": False,
        "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
        "mime_type": "image/png",
    })
    registry.receive_artifact_chunk(conn, {
        "type": "client_artifact_chunk", "call_id": call_id,
        "generation": 1, "transfer_id": "img", "seq": 1,
        "data": base64.b64encode(blob[5:]).decode(), "eof": True,
    })
    registry.resolve_result(conn, {
        "type": "client_tool_result", "call_id": call_id, "generation": 1,
        "result": {"content": [
            {
                "type": "artifact_ref",
                "transfer_id": "img",
                "artifact_template": {"type": "image", "mimeType": "image/png"},
                "artifact_insert_path": ["data"],
            },
            {
                "type": "artifact_ref",
                "transfer_id": "img",
                "artifact_template": {
                    "type": "resource",
                    "resource": {
                        "uri": "client-local:///image.png",
                        "mimeType": "image/png",
                    },
                },
                "artifact_insert_path": ["resource", "blob"],
            },
        ]},
    })
    result = await task
    materialised = result["content"][0]
    assert materialised["type"] == "image"
    assert base64.b64decode(materialised["data"]) == blob
    assert materialised["mimeType"] == "image/png"
    assert materialised["_meta"]["openagent/pathSemantics"] == "client-local"
    embedded = result["content"][1]
    assert embedded["type"] == "resource"
    assert embedded["resource"]["uri"] == "client-local:///image.png"
    assert base64.b64decode(embedded["resource"]["blob"]) == blob
    assert embedded["_meta"]["openagent/location"] == "client"

    # A compact JSON result must not amplify one large transfer into an
    # unbounded in-memory/serialised response by repeating its reference.
    # Use a lightweight stand-in so this regression proves the accounting
    # without allocating a real 64 MiB fixture.
    from src.gateway.capabilities import MAX_ARTIFACT_BYTES_PER_CALL

    class _LargeCompleteArtifact:
        complete = True
        expected_size = MAX_ARTIFACT_BYTES_PER_CALL

        def materialise(self):
            return {
                "type": "blob", "data": "eA==",
                "mimeType": "application/octet-stream",
            }

    repeated = {"type": "artifact_ref", "transfer_id": "large"}
    try:
        registry._materialise_artifact_refs(
            {"content": [repeated, repeated]},
            {"large": _LargeCompleteArtifact()},  # type: ignore[dict-item]
        )
        raise AssertionError("repeated artifact references bypassed expansion quota")
    except ClientCapabilityError as exc:
        assert exc.code == "ARTIFACT_MATERIALISATION_TOO_LARGE"

    # Out-of-order chunks and digest mismatches are rejected while the call is
    # still pinned to the same exact instance/generation.
    bad_order = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "read_file", {}, timeout_s=2,
    ))
    await asyncio.sleep(0)
    bad_order_id = sent[-1]["call_id"]
    try:
        registry.receive_artifact_chunk(conn, {
            "type": "client_artifact_chunk", "call_id": bad_order_id,
            "generation": 1, "transfer_id": "bad-order", "seq": 1,
            "data": base64.b64encode(b"x").decode(), "eof": True,
        })
        raise AssertionError("out-of-order artifact chunk was accepted")
    except ClientCapabilityError as exc:
        assert exc.code == "ARTIFACT_SEQUENCE"
    bad_order.cancel()
    try:
        await bad_order
    except asyncio.CancelledError:
        pass

    bad_digest = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "read_file", {}, timeout_s=2,
    ))
    await asyncio.sleep(0)
    bad_digest_id = sent[-1]["call_id"]
    try:
        registry.receive_artifact_chunk(conn, {
            "type": "client_artifact_chunk", "call_id": bad_digest_id,
            "generation": 1, "transfer_id": "bad-digest", "seq": 0,
            "data": base64.b64encode(b"x").decode(), "eof": True,
            "size": 1, "sha256": "0" * 64,
            "mime_type": "application/octet-stream",
        })
        raise AssertionError("artifact digest mismatch was accepted")
    except ClientCapabilityError as exc:
        assert exc.code == "ARTIFACT_DIGEST_MISMATCH"
    bad_digest.cancel()
    try:
        await bad_digest
    except asyncio.CancelledError:
        pass

    # Integrity metadata is mandatory and the number of simultaneous transfer
    # ids is bounded even if a malicious client tries to allocate only empty
    # buffers (which otherwise bypasses the byte quota).
    missing_metadata = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "read_file", {}, timeout_s=2,
    ))
    await asyncio.sleep(0)
    missing_metadata_id = sent[-1]["call_id"]
    for omitted, frame in (
        ("size", {
            "sha256": hashlib.sha256(b"x").hexdigest(),
            "mime_type": "application/octet-stream",
        }),
        ("sha256", {"size": 1, "mime_type": "application/octet-stream"}),
        ("mime", {"size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}),
    ):
        try:
            registry.receive_artifact_chunk(conn, {
                "type": "client_artifact_chunk", "call_id": missing_metadata_id,
                "generation": 1, "transfer_id": f"missing-{omitted}", "seq": 0,
                "data": base64.b64encode(b"x").decode(), "eof": True,
                **frame,
            })
            raise AssertionError(f"artifact without {omitted} was accepted")
        except ClientCapabilityError as exc:
            assert exc.code == "INVALID_ARTIFACT"

    from src.gateway.capabilities import MAX_ARTIFACT_TRANSFERS_PER_CALL

    digest = hashlib.sha256(b"x").hexdigest()
    for index in range(MAX_ARTIFACT_TRANSFERS_PER_CALL):
        registry.receive_artifact_chunk(conn, {
            "type": "client_artifact_chunk", "call_id": missing_metadata_id,
            "generation": 1, "transfer_id": f"bounded-{index}", "seq": 0,
            "data": base64.b64encode(b"x").decode(), "eof": True,
            "size": 1, "sha256": digest,
            "mime_type": "application/octet-stream",
        })
    try:
        registry.receive_artifact_chunk(conn, {
            "type": "client_artifact_chunk", "call_id": missing_metadata_id,
            "generation": 1, "transfer_id": "bounded-overflow", "seq": 0,
            "data": base64.b64encode(b"x").decode(), "eof": True,
            "size": 1, "sha256": digest,
            "mime_type": "application/octet-stream",
        })
        raise AssertionError("artifact transfer-count limit was bypassed")
    except ClientCapabilityError as exc:
        assert exc.code == "TOO_MANY_ARTIFACTS"
    missing_metadata.cancel()
    try:
        await missing_metadata
    except asyncio.CancelledError:
        pass


@test("client_capabilities", "tool-search exposes canonical scoped catalogs without fallback")
async def t_scoped_tool_search(ctx: TestContext) -> None:
    from src.core.execution_origin import install_execution_origin, reset_execution_origin
    from src.gateway.capabilities import CapabilityRegistry
    from src.mcp.servers.tool_search.adapters import (
        _call_scoped_tool_impl,
        _coerce_to_jsonable,
        _list_scoped_servers_impl,
        _resolve_tool,
    )
    from src.mcp._runtime.function import Function

    hook_hits: list[str] = []
    async def server_read_file(**_kwargs):
        return {"marker": "server"}

    server_fn = Function(
        name="read_file",
        entrypoint=server_read_file,
        description="server reader",
        parameters={"type": "object"},
        pre_hook=lambda: hook_hits.append("pre"),
        post_hook=lambda: hook_hits.append("post"),
    )
    other_fn = SimpleNamespace(
        entrypoint=lambda: "wrong",
        description="wrong server",
        parameters={},
    )

    class Pool:
        def __init__(self):
            self._toolkit_by_name = {
                "filesystem": SimpleNamespace(functions={"read_file": server_fn}),
                "other": SimpleNamespace(functions={"only_other": other_fn}),
            }

        def toolkit_by_name(self, name):
            return self._toolkit_by_name.get(name)

    pool = Pool()
    registry = CapabilityRegistry()
    conn_box: dict[str, object] = {}

    async def send(_ws, payload):
        if payload.get("type") == "client_tool_call":
            asyncio.get_running_loop().call_soon(
                registry.resolve_result,
                conn_box["conn"],
                {
                    "type": "client_tool_result",
                    "call_id": payload["call_id"],
                    "generation": payload["generation"],
                    "result": {"marker": "client"},
                },
            )
        return True

    conn = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop", generation=1,
        device_label="Mac", ws=_Ws(), send_json=send, servers=_catalog(),
    )
    conn_box["conn"] = conn
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None
    token = install_execution_origin(origin)
    try:
        names = [item["name"] for item in _list_scoped_servers_impl(pool)]
        assert "server:filesystem" in names and "client:filesystem" in names
        server_result = await _call_scoped_tool_impl(
            pool, "server:filesystem", "read_file", {},
        )
        client_result = await _call_scoped_tool_impl(
            pool, "client:filesystem", "read_file", {},
        )
        assert server_result["marker"] == "server"
        assert server_result["execution_host"]["kind"] == "server"
        assert hook_hits == ["pre", "post"], "runtime Function hooks were bypassed"
        assert client_result["marker"] == "client"
        assert client_result["execution_host"]["kind"] == "client"
        # Right leaf on the wrong named MCP is an error, never a global search.
        fn, resolved = await _resolve_tool(pool, "filesystem", "only_other")
        assert fn is None and resolved is None
    finally:
        reset_execution_origin(token)

    names = [item["name"] for item in _list_scoped_servers_impl(pool)]
    assert all(not name.startswith("client:") for name in names)

    envelope = SimpleNamespace(
        content=[{"type": "text", "text": "ok"}],
        structuredContent={"count": 1},
        isError=False,
        _meta={"vendor": "x"},
        images=[{"url": "image"}],
        audios=[{"url": "audio"}],
        videos=[{"url": "video"}],
        files=[{"name": "report.pdf"}],
        child_session_id="child:1",
        child_run_id="run:1",
    )
    coerced = _coerce_to_jsonable(envelope)
    for key in (
        "content", "structuredContent", "isError", "_meta", "images",
        "audios", "videos", "files", "child_session_id", "child_run_id",
    ):
        assert key in coerced, f"runtime result field was dropped: {key}"


@test(
    "client_capabilities",
    "tool catalog and dispatcher providers keep server and client isolated",
)
async def t_tool_provider_contracts(ctx: TestContext) -> None:
    from src.core.execution_origin import install_execution_origin, reset_execution_origin
    from src.gateway.capabilities import CapabilityRegistry
    from src.mcp.tool_providers import (
        InteractiveClientMCPProvider,
        ServerMCPProvider,
        ToolCatalogProvider,
        ToolDispatcher,
    )

    async def server_read_file(**_kwargs):
        return {
            "content": [{"type": "text", "text": "server"}],
            "structuredContent": {"marker": "server"},
            "isError": False,
        }

    server_fn = SimpleNamespace(
        entrypoint=server_read_file,
        description="server reader",
        parameters={"type": "object"},
    )

    class Pool:
        _toolkit_by_name = {
            "filesystem": SimpleNamespace(functions={"read_file": server_fn}),
        }

        def toolkit_by_name(self, name):
            return self._toolkit_by_name.get(name)

    pool = Pool()
    server_provider = ServerMCPProvider(pool)
    assert isinstance(server_provider, ToolCatalogProvider)
    assert isinstance(server_provider, ToolDispatcher)
    assert [item["name"] for item in server_provider.list_servers()] == [
        "server:filesystem",
    ]

    registry = CapabilityRegistry()
    conn_box: dict[str, object] = {}

    async def send(_ws, payload):
        if payload.get("type") == "client_tool_call":
            asyncio.get_running_loop().call_soon(
                registry.resolve_result,
                conn_box["conn"],
                {
                    "type": "client_tool_result",
                    "call_id": payload["call_id"],
                    "generation": payload["generation"],
                    "result": {
                        "content": [{"type": "text", "text": "client"}],
                        "structuredContent": {"marker": "client"},
                        "isError": False,
                    },
                },
            )
        return True

    conn = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=9, device_label="Mac", ws=_Ws(), send_json=send,
        servers=_catalog(),
    )
    conn_box["conn"] = conn
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None
    client_provider = InteractiveClientMCPProvider(registry)
    assert isinstance(client_provider, ToolCatalogProvider)
    assert isinstance(client_provider, ToolDispatcher)

    # A client backend is useless outside the trusted turn context; merely
    # retaining the provider cannot turn a future server-owned task into a
    # device-bound execution.
    try:
        client_provider.list_servers()
        raise AssertionError("client provider escaped its interactive turn")
    except PermissionError:
        pass

    token = install_execution_origin(origin)
    try:
        assert [item["name"] for item in client_provider.list_servers()] == [
            "client:filesystem",
        ]
        server_result = await server_provider.call_tool(
            "filesystem", "read_file", {},
        )
        client_result = await client_provider.call_tool(
            "filesystem", "read_file", {}, session_id="chat:provider-test",
        )
    finally:
        reset_execution_origin(token)

    assert server_result["structuredContent"]["marker"] == "server"
    assert server_result["execution_host"]["kind"] == "server"
    assert client_result["structuredContent"]["marker"] == "client"
    assert client_result["execution_host"]["kind"] == "client"
    assert client_result["execution_host"]["generation"] == 9


@test(
    "client_capabilities",
    "workflow mcp-tool reaches client only through turn-scoped tool-search",
)
async def t_workflow_node_client_dispatch(ctx: TestContext) -> None:
    from types import SimpleNamespace

    from src.core.execution_origin import execution_origin_scope
    from src.gateway.capabilities import CapabilityRegistry
    from src.mcp.servers.tool_search.adapters import build_runtime_toolkit
    from src.workflow.executor import _RunCtx, _h_mcp_tool

    registry = CapabilityRegistry()
    conn_box: dict[str, object] = {}

    async def send(_ws, payload):
        if payload.get("type") == "client_tool_call":
            asyncio.get_running_loop().call_soon(
                registry.resolve_result,
                conn_box["conn"],
                {
                    "type": "client_tool_result",
                    "call_id": payload["call_id"],
                    "generation": payload["generation"],
                    "result": {
                        "content": [{"type": "text", "text": "workflow-client"}],
                        "structuredContent": {"marker": "workflow-client"},
                        "isError": False,
                    },
                },
            )
        return True

    conn = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=4, device_label="Mac", ws=_Ws(), send_json=send,
        servers=_catalog(),
    )
    conn_box["conn"] = conn
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None

    class Pool:
        def __init__(self):
            self._toolkit_by_name: dict[str, object] = {}

        def toolkit_by_name(self, name):
            return self._toolkit_by_name.get(name)

    pool = Pool()
    pool._toolkit_by_name["tool-search"] = build_runtime_toolkit(pool=pool)
    executor = SimpleNamespace(agent=SimpleNamespace(_mcp=pool))
    run_ctx = _RunCtx(run_id="run-1", workflow_id="wf-1", inputs={}, vars={})
    config = {
        "mcp_name": "tool-search",
        "tool_name": "tool_search_call_tool",
        "args": {
            "server": "client:filesystem",
            "tool": "read_file",
            "args": {"path": "/client/sentinel"},
        },
    }

    with execution_origin_scope(origin):
        output = await _h_mcp_tool(
            executor, {"id": "local", "type": "mcp-tool"}, config, run_ctx,
        )
    result = output["result"]
    assert result["structuredContent"]["marker"] == "workflow-client"
    assert result["execution_host"]["client_instance_id"] == "desktop"

    # The same durable/automatic node has no implicit target and must fail;
    # it cannot use the live device merely because the workflow references a
    # canonical client MCP id.
    try:
        await _h_mcp_tool(
            executor, {"id": "automatic", "type": "mcp-tool"}, config, run_ctx,
        )
        raise AssertionError("automatic workflow node reached a client host")
    except PermissionError as exc:
        assert "server-owned turn" in str(exc)


@test(
    "client_capabilities",
    "safe calls retry only on the exact same-generation host",
)
async def t_disconnect_result_classification(ctx: TestContext) -> None:
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError

    registry = CapabilityRegistry()
    sent: list[dict] = []

    async def send(_ws, payload):
        sent.append(payload)
        return True

    conn = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=1, device_label="Mac", ws=_Ws(), send_json=send,
        servers=_safety_catalog(),
    )
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None
    read_call = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "read_file", {}, timeout_s=2,
    ))
    idempotent_call = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "ensure_file", {}, timeout_s=2,
    ))
    write_call = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "write_file", {}, timeout_s=2,
    ))
    await asyncio.sleep(0)
    first_frames = [frame for frame in sent if frame["type"] == "client_tool_call"]
    assert len(first_frames) == 3
    first_read = next(frame for frame in first_frames if frame["tool"] == "read_file")
    first_idempotent = next(
        frame for frame in first_frames if frame["tool"] == "ensure_file"
    )
    await registry.unregister(conn)

    try:
        await write_call
        raise AssertionError("disconnected mutation unexpectedly succeeded")
    except ClientCapabilityError as exc:
        assert exc.code == "CLIENT_RESULT_INDETERMINATE"
        assert exc.data == {"classification": "mutating", "retryable": False}

    # A different instance must not receive the frozen safe calls. They remain
    # pending until the exact host comes back.
    wrong_sent: list[dict] = []
    await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="cli",
        generation=1, device_label="CLI", ws=_Ws(),
        send_json=lambda _ws, frame: _record_send(wrong_sent, frame),
        servers=_safety_catalog(),
    )
    await asyncio.sleep(0)
    assert not read_call.done() and not idempotent_call.done()
    assert wrong_sent == []

    retry_sent: list[dict] = []
    retry_conn: dict[str, object] = {}

    async def send_retry(_ws, frame):
        retry_sent.append(frame)
        if frame.get("type") == "client_tool_call":
            asyncio.get_running_loop().call_soon(
                registry.resolve_result,
                retry_conn["conn"],
                {
                    "type": "client_tool_result",
                    "call_id": frame["call_id"],
                    "generation": frame["generation"],
                    "result": {"structuredContent": {"retried": True}},
                },
            )
        return True

    retry_conn["conn"] = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=1, device_label="Mac", ws=_Ws(), send_json=send_retry,
        servers=_safety_catalog(),
    )
    read_result, idempotent_result = await asyncio.gather(
        read_call, idempotent_call,
    )
    retried = [frame for frame in retry_sent if frame["type"] == "client_tool_call"]
    assert len(retried) == 2
    for original in (first_read, first_idempotent):
        retry_frame = next(frame for frame in retried if frame["tool"] == original["tool"])
        for field in ("call_id", "arguments_sha256", "idempotency_key", "deadline_ms"):
            assert retry_frame[field] == original[field]
    assert read_result["structuredContent"] == {"retried": True}
    assert idempotent_result["structuredContent"] == {"retried": True}
    assert read_result["execution_host"]["generation"] == 1

    # A newer generation is a different execution host: it never receives an
    # old turn, and the registry remembers the high-water mark even after that
    # socket disconnects so a delayed generation-1 hello cannot roll it back.
    generation_registry = CapabilityRegistry()
    generation_sent: list[dict] = []
    generation_one = await generation_registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=1, device_label="Old", ws=_Ws(),
        send_json=lambda _ws, frame: _record_send(generation_sent, frame),
        servers=_safety_catalog(),
    )
    generation_origin = generation_registry.origin_for("dev", "desktop")
    assert generation_origin is not None
    old_call = asyncio.create_task(generation_registry.call_tool(
        generation_origin, "filesystem", "read_file", {}, timeout_s=0.1,
    ))
    await asyncio.sleep(0)
    await generation_registry.unregister(generation_one)
    generation_sent.clear()
    generation_two = await generation_registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=2, device_label="New", ws=_Ws(),
        send_json=lambda _ws, frame: _record_send(generation_sent, frame),
        servers=_safety_catalog(),
    )
    try:
        await old_call
        raise AssertionError("old turn moved to a newer generation")
    except ClientCapabilityError as exc:
        assert exc.code == "STALE_GENERATION"
    assert generation_sent == []
    await generation_registry.unregister(generation_two)
    try:
        await generation_registry.register(
            device_id="dev", account_id="network-1", client_instance_id="desktop",
            generation=1, device_label="Rollback", ws=_Ws(),
            send_json=lambda _ws, frame: _record_send(generation_sent, frame),
            servers=_safety_catalog(),
        )
        raise AssertionError("disconnected instance accepted a stale generation")
    except ClientCapabilityError as exc:
        assert exc.code == "STALE_GENERATION"


@test(
    "client_capabilities",
    "argument-specific classification is preserved and used before dispatch",
)
async def t_argument_specific_classification(ctx: TestContext) -> None:
    from src.gateway.capabilities import (
        CapabilityRegistry,
        ClientCapabilityError,
        _classification_for_arguments,
    )

    arguments = {"first": "yes", "second": "yes"}
    forward_rules = {
        "first": {"yes": "mutating"},
        "second": {"yes": "read_only"},
    }
    forward = {
        "classification": "read_only",
        "classification_by_argument": forward_rules,
    }
    reverse = {
        **forward,
        "classification_by_argument": dict(reversed(list(forward_rules.items()))),
    }
    assert _classification_for_arguments(forward, arguments) == "mutating"
    assert _classification_for_arguments(reverse, arguments) == "mutating"
    assert _classification_for_arguments(forward, {}) == "read_only"

    registry = CapabilityRegistry()
    sent: list[dict] = []

    async def send(_ws, payload):
        sent.append(payload)
        return True

    conn = await registry.register(
        device_id="dev",
        account_id="account-shared",
        network_id="network-a",
        client_instance_id="desktop",
        generation=1,
        device_label="Mac",
        ws=_Ws(),
        send_json=send,
        servers=_computer_catalog(),
    )
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None
    listed = registry.list_tools(origin, "computer-control")
    assert listed[0]["classification"] == "mutating"
    assert listed[0]["classification_by_argument"]["action"][
        "get_screenshot"
    ] == "read_only"

    screenshot = asyncio.create_task(registry.call_tool(
        origin,
        "computer-control",
        "computer",
        {"action": "get_screenshot"},
        timeout_s=2,
    ))
    await asyncio.sleep(0)
    screenshot_frame = sent[-1]
    assert screenshot_frame["network_id"] == "network-a"
    assert conn.pending[screenshot_frame["call_id"]].classification == "read_only"
    registry.resolve_result(conn, {
        "type": "client_tool_result",
        "call_id": screenshot_frame["call_id"],
        "generation": 1,
        "result": {"content": []},
    })
    await screenshot

    click = asyncio.create_task(registry.call_tool(
        origin,
        "computer-control",
        "computer",
        {"action": "left_click"},
        timeout_s=2,
    ))
    await asyncio.sleep(0)
    click_frame = sent[-1]
    assert conn.pending[click_frame["call_id"]].classification == "mutating"
    registry.resolve_result(conn, {
        "type": "client_tool_result",
        "call_id": click_frame["call_id"],
        "generation": 1,
        "result": {"content": []},
    })
    await click

    unknown = asyncio.create_task(registry.call_tool(
        origin,
        "computer-control",
        "computer",
        {"action": "future_action"},
        timeout_s=2,
    ))
    await asyncio.sleep(0)
    unknown_frame = sent[-1]
    assert conn.pending[unknown_frame["call_id"]].classification == "mutating"
    registry.resolve_result(conn, {
        "type": "client_tool_result",
        "call_id": unknown_frame["call_id"],
        "generation": 1,
        "result": {"content": []},
    })
    await unknown

    # A transport loss uses the resolved operation class, not the tool's base
    # class: screenshots retry only on the exact host, while input is terminal
    # and indeterminate because the local effect may already have happened.
    disconnected = CapabilityRegistry()
    first_sent: list[dict] = []
    first_conn = await disconnected.register(
        device_id="dynamic-dev",
        account_id="dynamic-network",
        client_instance_id="desktop",
        generation=7,
        device_label="Mac",
        ws=_Ws(),
        send_json=lambda _ws, frame: _record_send(first_sent, frame),
        servers=_computer_catalog(),
    )
    disconnected_origin = disconnected.origin_for("dynamic-dev", "desktop")
    assert disconnected_origin is not None
    retryable_screenshot = asyncio.create_task(disconnected.call_tool(
        disconnected_origin,
        "computer-control",
        "computer",
        {"action": "get_screenshot"},
        timeout_s=2,
    ))
    indeterminate_click = asyncio.create_task(disconnected.call_tool(
        disconnected_origin,
        "computer-control",
        "computer",
        {"action": "left_click"},
        timeout_s=2,
    ))
    await asyncio.sleep(0)
    screenshot_call_id = next(
        frame["call_id"] for frame in first_sent
        if frame.get("args", {}).get("action") == "get_screenshot"
    )
    await disconnected.unregister(first_conn)
    try:
        await indeterminate_click
        raise AssertionError("disconnected click unexpectedly succeeded")
    except ClientCapabilityError as exc:
        assert exc.code == "CLIENT_RESULT_INDETERMINATE"

    retry_conn: dict[str, object] = {}

    async def send_retry(_ws, frame):
        if frame.get("type") == "client_tool_call":
            assert frame["call_id"] == screenshot_call_id
            asyncio.get_running_loop().call_soon(
                disconnected.resolve_result,
                retry_conn["conn"],
                {
                    "type": "client_tool_result",
                    "call_id": frame["call_id"],
                    "generation": frame["generation"],
                    "result": {"structuredContent": {"retried": True}},
                },
            )
        return True

    retry_conn["conn"] = await disconnected.register(
        device_id="dynamic-dev",
        account_id="dynamic-network",
        client_instance_id="desktop",
        generation=7,
        device_label="Mac",
        ws=_Ws(),
        send_json=send_retry,
        servers=_computer_catalog(),
    )
    screenshot_result = await retryable_screenshot
    assert screenshot_result["structuredContent"] == {"retried": True}


async def _record_send(target: list[dict], frame: dict) -> bool:
    target.append(frame)
    return True


@test(
    "client_capabilities",
    "mutating timeout and transport ambiguity are never reported retryable",
)
async def t_indeterminate_timeout_and_transport(ctx: TestContext) -> None:
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError

    registry = CapabilityRegistry()

    async def no_result(_ws, _payload):
        return True

    await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=1, device_label="Mac", ws=_Ws(), send_json=no_result,
        servers=_safety_catalog(),
    )
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None
    try:
        await registry.call_tool(
            origin, "filesystem", "write_file", {}, timeout_s=0.1,
        )
        raise AssertionError("timed-out mutation unexpectedly succeeded")
    except ClientCapabilityError as exc:
        assert exc.code == "CLIENT_RESULT_INDETERMINATE"
        assert exc.data and exc.data["retryable"] is False
    try:
        await registry.call_tool(
            origin, "filesystem", "read_file", {}, timeout_s=0.1,
        )
        raise AssertionError("timed-out read unexpectedly succeeded")
    except ClientCapabilityError as exc:
        assert exc.code == "CLIENT_TIMEOUT"
        assert exc.data and exc.data["retryable"] is True

    transport_registry = CapabilityRegistry()

    async def transport_failed(_ws, _payload):
        return False

    await transport_registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=1, device_label="Mac", ws=_Ws(), send_json=transport_failed,
        servers=_safety_catalog(),
    )
    transport_origin = transport_registry.origin_for("dev", "desktop")
    assert transport_origin is not None
    for tool, expected_code, retryable in (
        ("write_file", "CLIENT_RESULT_INDETERMINATE", False),
        # Safe calls wait only for an exact reconnect; if none arrives before
        # the original deadline, the terminal result is a bounded timeout.
        ("read_file", "CLIENT_TIMEOUT", True),
    ):
        try:
            await transport_registry.call_tool(
                transport_origin, "filesystem", tool, {}, timeout_s=1,
            )
            raise AssertionError(f"failed transport unexpectedly ran {tool}")
        except ClientCapabilityError as exc:
            assert exc.code == expected_code
            assert exc.data and exc.data["retryable"] is retryable

    # A broker-reported terminal error is determinate even for a mutation;
    # only loss/ambiguity after dispatch gets the special indeterminate code.
    result_registry = CapabilityRegistry()
    result_conn: dict[str, object] = {}

    async def explicit_error(_ws, payload):
        asyncio.get_running_loop().call_soon(
            result_registry.resolve_result,
            result_conn["conn"],
            {
                "type": "client_tool_result",
                "call_id": payload["call_id"],
                "generation": payload["generation"],
                "error": {"code": "LOCAL_DENIED", "message": "denied locally"},
            },
        )
        return True

    result_conn["conn"] = await result_registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=1, device_label="Mac", ws=_Ws(), send_json=explicit_error,
        servers=_safety_catalog(),
    )
    result_origin = result_registry.origin_for("dev", "desktop")
    assert result_origin is not None
    try:
        await result_registry.call_tool(
            result_origin, "filesystem", "write_file", {}, timeout_s=1,
        )
        raise AssertionError("explicit local denial unexpectedly succeeded")
    except ClientCapabilityError as exc:
        assert exc.code == "LOCAL_DENIED"


@test(
    "client_capabilities",
    "server cancellation preserves cancellation and audits mutation ambiguity",
)
async def t_mutating_cancellation_audit(ctx: TestContext) -> None:
    import src.gateway.capabilities as capability_module
    from src.gateway.capabilities import CapabilityRegistry

    registry = CapabilityRegistry()
    sent: list[dict] = []
    events: list[tuple[str, dict]] = []

    async def send(_ws, payload):
        sent.append(payload)
        return True

    await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=1, device_label="Mac", ws=_Ws(), send_json=send,
        servers=_safety_catalog(),
    )
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None
    original_elog = capability_module.elog
    capability_module.elog = lambda event, **fields: events.append((event, fields))
    call = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "write_file", {}, timeout_s=2,
    ))
    try:
        await asyncio.sleep(0)
        call.cancel()
        try:
            await call
            raise AssertionError("server cancellation was swallowed")
        except asyncio.CancelledError:
            pass
    finally:
        capability_module.elog = original_elog

    assert [frame["type"] for frame in sent] == [
        "client_tool_call", "client_tool_cancel",
    ]
    cancelled = [
        fields for event, fields in events
        if event == "client_tool_call.cancelled"
    ]
    assert len(cancelled) == 1
    assert cancelled[0]["outcome"] == "indeterminate"
    assert cancelled[0]["error_code"] == "CLIENT_RESULT_INDETERMINATE"
    assert cancelled[0]["classification"] == "mutating"


@test("client_capabilities", "subprocess MCP envelope survives runtime and tool-search")
async def t_runtime_mcp_envelope(ctx: TestContext) -> None:
    from mcp.types import (
        AudioContent,
        CallToolResult,
        ImageContent,
        ResourceLink,
        TextContent,
    )
    from src.core._runner.utils.mcp import get_entrypoint_for_tool
    from src.mcp.servers.tool_search.adapters import _coerce_to_jsonable

    image_bytes = b"image-bytes"
    audio_bytes = b"audio-bytes"
    original = CallToolResult(
        content=[
            TextContent(type="text", text="hello"),
            ImageContent(
                type="image", data=base64.b64encode(image_bytes).decode(),
                mimeType="image/png",
            ),
            AudioContent(
                type="audio", data=base64.b64encode(audio_bytes).decode(),
                mimeType="audio/mpeg",
            ),
            ResourceLink(
                type="resource_link", name="report",
                uri="https://example.test/report.pdf",
                mimeType="application/pdf", size=123,
            ),
        ],
        structuredContent={"rows": [{"id": 1}]},
        isError=False,
        _meta={"vendor": "test"},
    )

    class Session:
        async def send_ping(self):
            return None

        async def call_tool(self, name, args, meta=None):
            assert name == "mixed" and args == {"value": 1}
            return original

    entrypoint = get_entrypoint_for_tool(
        SimpleNamespace(name="mixed"), Session(),
    )
    runtime_result = await entrypoint(value=1)
    assert runtime_result.mcp_result is not None
    assert runtime_result.mcp_result["structuredContent"]["rows"][0]["id"] == 1
    assert runtime_result.mcp_result["isError"] is False
    assert runtime_result.mcp_result["_meta"]["vendor"] == "test"
    assert len(runtime_result.mcp_result["content"]) == 4
    assert runtime_result.images and runtime_result.images[0].content == image_bytes
    assert runtime_result.audios and runtime_result.audios[0].content == audio_bytes

    dispatched = _coerce_to_jsonable(runtime_result)
    assert dispatched["content"][0]["text"] == "hello"
    assert dispatched["structuredContent"] == {"rows": [{"id": 1}]}
    assert dispatched["isError"] is False
    assert dispatched["_meta"] == {"vendor": "test"}
    assert "images" in dispatched and "audios" in dispatched


@test(
    "client_capabilities",
    "deep MCP JSON is lossless and over-limit envelopes fail explicitly",
)
async def t_runtime_mcp_deep_envelope_limits(ctx: TestContext) -> None:
    import src.mcp.servers.tool_search.adapters as adapters

    nested: object = "leaf"
    for index in range(20, 0, -1):
        nested = {f"level_{index}": nested}
    original = {"content": [], "structuredContent": nested, "isError": False}
    stamped = adapters._stamp_execution_host(original, {"kind": "server"})
    assert stamped["structuredContent"] == nested
    assert stamped["structuredContent"] is not nested

    too_deep: object = "leaf"
    for index in range(adapters._MAX_MCP_RESULT_NESTING + 1, 0, -1):
        too_deep = {f"level_{index}": too_deep}
    try:
        adapters._coerce_to_jsonable({"structuredContent": too_deep})
        raise AssertionError("over-deep MCP result was not rejected")
    except adapters.MCPResultEnvelopeLimitError as exc:
        assert "nesting depth" in str(exc)

    original_node_limit = adapters._MAX_MCP_RESULT_NODES
    original_byte_limit = adapters._MAX_MCP_RESULT_JSON_BYTES
    try:
        adapters._MAX_MCP_RESULT_NODES = 5
        try:
            adapters._coerce_to_jsonable({"structuredContent": list(range(10))})
            raise AssertionError("over-complex MCP result was not rejected")
        except adapters.MCPResultEnvelopeLimitError as exc:
            assert "node count" in str(exc)

        adapters._MAX_MCP_RESULT_NODES = original_node_limit
        adapters._MAX_MCP_RESULT_JSON_BYTES = 64
        try:
            adapters._coerce_to_jsonable({"structuredContent": "x" * 128})
            raise AssertionError("oversized MCP result was not rejected")
        except adapters.MCPResultEnvelopeLimitError as exc:
            assert "serialised size" in str(exc)
    finally:
        adapters._MAX_MCP_RESULT_NODES = original_node_limit
        adapters._MAX_MCP_RESULT_JSON_BYTES = original_byte_limit


@test("client_capabilities", "client dispatch audit records routing but no payload")
async def t_client_dispatch_audit(ctx: TestContext) -> None:
    import src.gateway.capabilities as capability_module
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError

    registry = CapabilityRegistry()
    sent: list[dict] = []
    events: list[tuple[str, dict]] = []
    conn_box: dict[str, object] = {}

    async def send(_ws, payload):
        sent.append(payload)
        if payload.get("type") == "client_tool_call" and not payload["args"].get("timeout"):
            asyncio.get_running_loop().call_soon(
                registry.resolve_result,
                conn_box["conn"],
                {
                    "type": "client_tool_result",
                    "call_id": payload["call_id"],
                    "generation": payload["generation"],
                    "result": {
                        "content": [{"type": "text", "text": "secret"}],
                        "isError": bool(payload["args"].get("mcp_error")),
                    },
                },
            )
        return True

    conn = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop", generation=1,
        device_label="Mac", ws=_Ws(), send_json=send, servers=_catalog(),
    )
    conn_box["conn"] = conn
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None

    original_elog = capability_module.elog
    capability_module.elog = lambda event, **fields: events.append((event, fields))
    try:
        await registry.call_tool(
            origin, "filesystem", "read_file", {"path": "/private"},
            timeout_s=1,
        )
        error_envelope = await registry.call_tool(
            origin, "filesystem", "read_file", {"mcp_error": True},
            timeout_s=1,
        )
        assert error_envelope["isError"] is True
        try:
            await registry.call_tool(
                origin, "filesystem", "read_file", {"timeout": True},
                timeout_s=0.1,
            )
            raise AssertionError("client timeout was not surfaced")
        except ClientCapabilityError as exc:
            assert exc.code == "CLIENT_TIMEOUT"
    finally:
        capability_module.elog = original_elog

    names = [name for name, _fields in events]
    assert names.count("client_tool_call.start") == 3
    assert names.count("client_tool_call.result") == 2
    assert "client_tool_call.timeout" in names
    error_results = [
        fields for name, fields in events
        if name == "client_tool_call.result" and fields.get("outcome") == "error"
    ]
    assert len(error_results) == 1
    assert error_results[0]["error_code"] == "MCP_IS_ERROR"
    for _name, fields in events:
        assert fields["device_id"] == "dev"
        assert fields["client_instance_id"] == "desktop"
        assert fields["module"] == "filesystem" and fields["tool"] == "read_file"
        assert len(fields["arguments_sha256"]) == 64
        assert not ({"args", "arguments", "result", "content"} & fields.keys())


@test("client_capabilities", "pending client calls apply bounded backpressure")
async def t_client_call_backpressure(ctx: TestContext) -> None:
    import src.gateway.capabilities as capability_module
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError

    registry = CapabilityRegistry()
    sent: list[dict] = []

    async def send(_ws, payload):
        sent.append(payload)
        return True

    conn = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop", generation=1,
        device_label="Mac", ws=_Ws(), send_json=send, servers=_catalog(),
    )
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None
    original_limit = capability_module.MAX_PENDING_CALLS_PER_CONNECTION
    capability_module.MAX_PENDING_CALLS_PER_CONNECTION = 1
    first = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "read_file", {"path": "first"}, timeout_s=2,
    ))
    try:
        await asyncio.sleep(0)
        assert len(conn.pending) == 1
        try:
            await registry.call_tool(
                origin, "filesystem", "read_file", {"path": "second"}, timeout_s=2,
            )
            raise AssertionError("pending-call limit was not enforced")
        except ClientCapabilityError as exc:
            assert exc.code == "CLIENT_BACKPRESSURE"
        assert len(conn.pending) == 1, "rejected call allocated pending state"
    finally:
        capability_module.MAX_PENDING_CALLS_PER_CONNECTION = original_limit
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass


@test("client_capabilities", "argument hash is stable across JavaScript number normalisation")
async def t_cross_language_argument_hash(ctx: TestContext) -> None:
    from src.gateway.capabilities import _canonical_json_sha256

    integer_form = {"x": 1, "nested": [0, {"v": 2}], "text": "è"}
    javascript_roundtrip_form = {
        "text": "è", "nested": [-0.0, {"v": 2.0}], "x": 1.0,
    }
    digest = _canonical_json_sha256(integer_form)
    assert digest == _canonical_json_sha256(javascript_roundtrip_form)
    # Fixed cross-package vector shared with openagent-host-tools.
    assert digest == "4dc3a30bb9d5c2c92d2135735330fb49b793376890a07e80b55536ad8d8d5009"


@test("client_capabilities", "heartbeat reaper closes stale host and pending calls")
async def t_capability_heartbeat_reaper(ctx: TestContext) -> None:
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError

    registry = CapabilityRegistry()
    sent: list[dict] = []

    async def send(_ws, payload):
        sent.append(payload)
        return True

    stale_ws = _Ws()
    stale = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="stale", generation=1,
        device_label="Old Mac", ws=stale_ws, send_json=send, servers=_safety_catalog(),
    )
    live = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="live", generation=1,
        device_label="New Mac", ws=_Ws(), send_json=send, servers=_catalog(),
    )
    stale.last_seen_at = 10.0
    live.last_seen_at = 19.0
    await registry.update_catalog(
        stale, generation=1, servers=_safety_catalog(),
    )
    assert stale.last_seen_at == 10.0, (
        "catalog spam must not bypass authenticated-heartbeat expiry"
    )
    origin = registry.origin_for("dev", "stale")
    assert origin is not None
    pending = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "write_file", {}, timeout_s=2,
    ))
    await asyncio.sleep(0)

    reaped = await registry.reap_stale(now=20.0, max_idle_s=5.0)
    assert reaped == [stale]
    assert stale_ws.closed
    assert registry.connection("dev", "stale") is None
    assert registry.connection("dev", "live") is live
    try:
        await pending
        raise AssertionError("stale-host pending call was not failed")
    except ClientCapabilityError as exc:
        assert exc.code == "CLIENT_RESULT_INDETERMINATE"


@test(
    "client_capabilities",
    "live revocation closes every chat and capability socket for a device",
)
async def t_live_device_revocation(ctx: TestContext) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.gateway.server import Gateway
    from src.network.auth.middleware import NetworkAuthState

    auth_state = NetworkAuthState(
        coordinator_pubkey=Ed25519PrivateKey.generate().public_key(),
        network_id="network-1",
    )
    gateway = Gateway(
        SimpleNamespace(name="test"),
        SimpleNamespace(auth_state=auth_state),
    )
    device_key = b"\x5a" * 32
    device_id = device_key.hex()
    capability_ws = _Ws()
    await gateway.capabilities.register(
        device_id=device_id,
        account_id="network-1",
        client_instance_id="desktop",
        generation=1,
        device_label="Mac",
        ws=capability_ws,
        send_json=lambda *_args: asyncio.sleep(0, result=True),
        servers=_catalog(),
    )
    chat_a, chat_b = _Ws(), _Ws()
    gateway.clients.update({"a": chat_a, "b": chat_b})
    gateway._chat_client_devices.update({"a": device_id, "b": device_id})

    auth_state.revoke(device_key)
    # Listener scheduling is thread-safe and deliberately asynchronous.
    for _ in range(3):
        await asyncio.sleep(0)
    assert capability_ws.closed and chat_a.closed and chat_b.closed
    assert gateway.capabilities.connection(device_id, "desktop") is None

    fresh_chat = _Ws()
    gateway.clients["fresh"] = fresh_chat
    gateway._chat_client_devices["fresh"] = device_id
    gateway._chat_client_auth_epochs["fresh"] = auth_state.device_epoch(device_key)
    await gateway._close_device_connections(
        device_id, revocation_epoch=auth_state.device_epoch(device_key),
    )
    assert not fresh_chat.closed, (
        "a delayed revocation callback closed a post-reactivation chat socket"
    )


@test(
    "client_capabilities",
    "revocation epoch blocks late hello and aborts safe reconnect waiters",
)
async def t_revocation_epoch_barrier(ctx: TestContext) -> None:
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError

    registry = CapabilityRegistry()
    sent: list[dict] = []
    auth_epoch = 0
    conn = await registry.register(
        device_id="dev",
        account_id="network-1",
        client_instance_id="desktop",
        generation=1,
        device_label="Mac",
        ws=_Ws(),
        send_json=lambda _ws, frame: _record_send(sent, frame),
        servers=_safety_catalog(),
        auth_epoch=auth_epoch,
        auth_epoch_reader=lambda: auth_epoch,
    )
    origin = conn.origin(registry)
    read_call = asyncio.create_task(registry.call_tool(
        origin, "filesystem", "read_file", {}, timeout_s=5,
    ))
    await asyncio.sleep(0)
    await registry.unregister(conn)
    # The safe call is now waiting for this exact host. Revocation must wake it
    # immediately rather than letting it sleep until its five-second deadline.
    await asyncio.sleep(0)
    assert not read_call.done()
    auth_epoch = 1
    await registry.close_device("dev", revocation_epoch=auth_epoch)
    try:
        await asyncio.wait_for(read_call, timeout=0.2)
        raise AssertionError("revoked read-only call unexpectedly succeeded")
    except ClientCapabilityError as exc:
        assert exc.code == "CLIENT_REVOKED"
        assert exc.data == {"retryable": False}

    # A hello authenticated before the epoch advance can never register after
    # the close callback took its snapshot.
    try:
        await registry.register(
            device_id="dev",
            account_id="network-1",
            client_instance_id="late",
            generation=1,
            device_label="Late",
            ws=_Ws(),
            send_json=lambda _ws, frame: _record_send(sent, frame),
            servers=_catalog(),
            auth_epoch=0,
            auth_epoch_reader=lambda: auth_epoch,
        )
        raise AssertionError("pre-revocation capability hello was accepted")
    except ClientCapabilityError as exc:
        assert exc.code == "CLIENT_REVOKED"

    # A fresh request after reversible reactivation carries the current epoch
    # and remains usable; the barrier is not a permanent deny-set.
    fresh = await registry.register(
        device_id="dev",
        account_id="network-1",
        client_instance_id="fresh",
        generation=1,
        device_label="Fresh",
        ws=_Ws(),
        send_json=lambda _ws, frame: _record_send(sent, frame),
        servers=_catalog(),
        auth_epoch=auth_epoch,
        auth_epoch_reader=lambda: auth_epoch,
    )
    assert registry.connection("dev", "fresh") is fresh
    await registry.close_device("dev", revocation_epoch=auth_epoch)
    assert registry.connection("dev", "fresh") is fresh, (
        "a delayed old revocation callback closed a freshly authorized host"
    )

    # Even before a delayed close callback reaches the registry, a safe call
    # from auth epoch 0 cannot jump onto the same instance/generation after it
    # re-authenticated at epoch 1.
    epoch_registry = CapabilityRegistry()
    epoch_value = 0
    old_epoch_conn = await epoch_registry.register(
        device_id="epoch-dev", account_id="n", client_instance_id="desktop",
        generation=1, device_label="Old", ws=_Ws(),
        send_json=lambda _ws, frame: _record_send(sent, frame),
        servers=_safety_catalog(), auth_epoch=0,
        auth_epoch_reader=lambda: epoch_value,
    )
    old_epoch_call = asyncio.create_task(epoch_registry.call_tool(
        old_epoch_conn.origin(epoch_registry),
        "filesystem", "read_file", {}, timeout_s=2,
    ))
    await asyncio.sleep(0)
    epoch_value = 1
    await epoch_registry.register(
        device_id="epoch-dev", account_id="n", client_instance_id="desktop",
        generation=1, device_label="Fresh", ws=_Ws(),
        send_json=lambda _ws, frame: _record_send(sent, frame),
        servers=_safety_catalog(), auth_epoch=1,
        auth_epoch_reader=lambda: epoch_value,
    )
    try:
        await old_epoch_call
        raise AssertionError("older turn crossed an authorization epoch")
    except ClientCapabilityError as exc:
        assert exc.code == "CLIENT_REVOKED"


@test(
    "client_capabilities",
    "capability, generation, catalog, and artifact reservations are bounded",
)
async def t_capability_resource_quotas(ctx: TestContext) -> None:
    import src.gateway.capabilities as capability_module
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError

    originals = {
        name: getattr(capability_module, name)
        for name in (
            "MAX_ACTIVE_CAPABILITY_CONNECTIONS_PER_DEVICE",
            "MAX_KNOWN_CLIENT_INSTANCES_PER_DEVICE",
            "MAX_CAPABILITY_CATALOG_BYTES",
            "MAX_ARTIFACT_BYTES_PER_CONNECTION",
        )
    }
    try:
        capability_module.MAX_ACTIVE_CAPABILITY_CONNECTIONS_PER_DEVICE = 1
        active_registry = CapabilityRegistry()
        first = await active_registry.register(
            device_id="dev", account_id="n", client_instance_id="one",
            generation=1, device_label="One", ws=_Ws(),
            send_json=lambda *_args: asyncio.sleep(0, result=True),
            servers=_catalog(),
        )
        try:
            await active_registry.register(
                device_id="dev", account_id="n", client_instance_id="two",
                generation=1, device_label="Two", ws=_Ws(),
                send_json=lambda *_args: asyncio.sleep(0, result=True),
                servers=_catalog(),
            )
            raise AssertionError("per-device connection quota was bypassed")
        except ClientCapabilityError as exc:
            assert exc.code == "CAPABILITY_CONNECTION_QUOTA"
        await active_registry.unregister(first)

        capability_module.MAX_KNOWN_CLIENT_INSTANCES_PER_DEVICE = 1
        try:
            await active_registry.register(
                device_id="dev", account_id="n", client_instance_id="two",
                generation=1, device_label="Two", ws=_Ws(),
                send_json=lambda *_args: asyncio.sleep(0, result=True),
                servers=_catalog(),
            )
            raise AssertionError("generation-floor quota was bypassed")
        except ClientCapabilityError as exc:
            assert exc.code == "CLIENT_INSTANCE_QUOTA"

        capability_module.MAX_CAPABILITY_CATALOG_BYTES = 64
        try:
            capability_module.normalise_catalog(_catalog())
            raise AssertionError("encoded catalog quota was bypassed")
        except ClientCapabilityError as exc:
            assert exc.code == "CATALOG_QUOTA"

        capability_module.MAX_CAPABILITY_CATALOG_BYTES = originals[
            "MAX_CAPABILITY_CATALOG_BYTES"
        ]
        capability_module.MAX_ACTIVE_CAPABILITY_CONNECTIONS_PER_DEVICE = originals[
            "MAX_ACTIVE_CAPABILITY_CONNECTIONS_PER_DEVICE"
        ]
        artifact_registry = CapabilityRegistry()
        artifact_sent: list[dict] = []
        artifact_conn = await artifact_registry.register(
            device_id="artifact-dev", account_id="n", client_instance_id="one",
            generation=1, device_label="One", ws=_Ws(),
            send_json=lambda _ws, frame: _record_send(artifact_sent, frame),
            servers=_catalog(),
        )
        artifact_call = asyncio.create_task(artifact_registry.call_tool(
            artifact_conn.origin(artifact_registry),
            "filesystem", "read_file", {}, timeout_s=2,
        ))
        await asyncio.sleep(0)
        call_id = artifact_sent[-1]["call_id"]
        capability_module.MAX_ARTIFACT_BYTES_PER_CONNECTION = 5
        artifact_registry.receive_artifact_chunk(artifact_conn, {
            "type": "client_artifact_chunk", "call_id": call_id,
            "generation": 1, "transfer_id": "a", "seq": 0,
            "size": 4, "sha256": "0" * 64,
            "mime_type": "application/octet-stream",
            "data": base64.b64encode(b"x").decode(), "eof": False,
        })
        try:
            artifact_registry.receive_artifact_chunk(artifact_conn, {
                "type": "client_artifact_chunk", "call_id": call_id,
                "generation": 1, "transfer_id": "b", "seq": 0,
                "size": 2, "sha256": "0" * 64,
                "mime_type": "application/octet-stream",
                "data": base64.b64encode(b"x").decode(), "eof": False,
            })
            raise AssertionError("declared artifact reservation quota was bypassed")
        except ClientCapabilityError as exc:
            assert exc.code == "ARTIFACT_QUOTA"
        artifact_call.cancel()
        try:
            await artifact_call
        except asyncio.CancelledError:
            pass
    finally:
        for name, value in originals.items():
            setattr(capability_module, name, value)


@test(
    "client_capabilities",
    "device revocation cancels only that ingress in a shared durable session",
)
async def t_revoke_only_matching_ingress(ctx: TestContext) -> None:
    from src.core.execution_origin import TrustedIngressIdentity
    from src.stream.events import TextFinal
    from src.stream.session import StreamSession

    class Agent:
        db = None

        async def request_cancel(self, _session_id):
            return False

    ingress_a = TrustedIngressIdentity(device_id="A", connection_id="a")
    ingress_b = TrustedIngressIdentity(device_id="B", connection_id="b")
    session = StreamSession(
        Agent(), client_id="shared", session_id="s", coalesce_window_ms=500,
    )
    never = asyncio.Event()
    turn_a = asyncio.create_task(never.wait())
    session._current_turn = turn_a
    session._current_turn_ingress = ingress_a
    msg_a = TextFinal(session_id="s", seq=1, ts_ms=1, text="from A")
    msg_b = TextFinal(session_id="s", seq=2, ts_ms=2, text="from B")
    session._pending_burst = [msg_a, msg_b]
    session._event_ingresses[id(msg_a)] = ingress_a
    session._event_ingresses[id(msg_b)] = ingress_b

    assert await session.revoke_ingress_device("A") is True
    assert turn_a.done(), "A's in-flight turn survived revocation"
    assert session._pending_burst == [msg_b]
    assert session._revoked_ingress_epochs["A"] == 1

    fresh_a = TrustedIngressIdentity(
        device_id="A", connection_id="fresh-a", auth_epoch=1,
    )
    session.allow_ingress_device("A", auth_epoch=1)
    assert not session._ingress_is_revoked(fresh_a)

    # B's active turn in the same durable session must survive another cleanup
    # pass for A. This is the exact regression for shared Desktop/CLI sessions.
    turn_b = asyncio.create_task(never.wait())
    session._current_turn = turn_b
    session._current_turn_ingress = ingress_b
    await session.revoke_ingress_device("A")
    assert not turn_b.done(), "revoking A cancelled B's independent turn"
    turn_b.cancel()
    try:
        await turn_b
    except asyncio.CancelledError:
        pass


@test(
    "client_capabilities",
    "chat frame cannot re-admit revoked ingress after an attach await",
)
async def t_chat_attach_epoch_recheck(ctx: TestContext) -> None:
    from src.gateway.server import Gateway, _StreamHolder
    from src.stream.events import TextFinal
    from src.stream.wire import event_to_wire

    class AuthState:
        epoch = 0

        def device_epoch(self, _key):
            return self.epoch

    auth_state = AuthState()
    gateway = Gateway(
        SimpleNamespace(name="test", model=None),
        SimpleNamespace(auth_state=auth_state),
    )
    gateway.sessions = SimpleNamespace(
        get_or_create_session=lambda *_args, **_kwargs: "s",
    )
    calls = {"allow": 0, "push": 0}

    class Session:
        def has_active_turn(self):
            return False

        def allow_ingress_device(self, _device_id):
            calls["allow"] += 1

        async def push_in(self, *_args, **_kwargs):
            calls["push"] += 1

    class Channel:
        pass

    gateway._stream_sessions[("alice", "s")] = _StreamHolder(
        session=Session(), channel=Channel(),
    )

    async def revoke_during_attach(_key, _holder):
        auth_state.epoch = 1
        await asyncio.sleep(0)
        return False

    gateway._stream_holder_is_stale_for_attach = revoke_during_attach
    device_pubkey = b"\x61" * 32
    await gateway._handle_stream_frame(
        _Ws(),
        device_pubkey.hex(),
        event_to_wire(TextFinal(
            session_id="s", seq=1, ts_ms=1, text="must be dropped",
        )),
        handle="alice",
        connection_id="conn",
        device_pubkey=device_pubkey,
        device_auth_epoch=0,
    )
    assert calls == {"allow": 0, "push": 0}
    assert gateway._live_replays == {}


@test(
    "client_capabilities",
    "detached scoring and compaction clear the interactive execution origin",
)
async def t_detached_tasks_clear_origin(ctx: TestContext) -> None:
    from src.core import compaction, quality_monitor
    from src.core.execution_origin import (
        TurnExecutionOrigin,
        current_execution_origin,
        execution_origin_scope,
    )

    origin = TurnExecutionOrigin(
        device_id="dev", client_instance_id="desktop", generation=1,
        device_label="Mac", registry=object(),
    )
    seen: list[tuple[str, object]] = []
    original_enabled = quality_monitor.enabled
    original_score = quality_monitor.maybe_score_turn
    original_flag = compaction._flag_enabled
    original_compact = compaction._run_background_compaction

    async def fake_score(*_args):
        seen.append(("score", current_execution_origin()))

    async def fake_compact(*_args):
        seen.append(("compact", current_execution_origin()))

    try:
        quality_monitor.enabled = lambda: True
        quality_monitor.maybe_score_turn = fake_score
        compaction._flag_enabled = lambda: True
        compaction._run_background_compaction = fake_compact
        with execution_origin_scope(origin):
            quality_monitor.spawn_scoring(object(), "detached-score", "u", "r")
            compact_task = compaction.compact_after_turn(
                "detached-compact", object(), object(),
            )
            assert current_execution_origin() is origin
        assert compact_task is not None
        await compact_task
        for _ in range(3):
            await asyncio.sleep(0)
        assert sorted(seen) == [("compact", None), ("score", None)]
    finally:
        quality_monitor.enabled = original_enabled
        quality_monitor.maybe_score_turn = original_score
        compaction._flag_enabled = original_flag
        compaction._run_background_compaction = original_compact


@test(
    "client_capabilities",
    "client shell events wake only the exact interactive origin",
)
async def t_client_shell_event_origin(ctx: TestContext) -> None:
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError
    from src.mcp.servers.shell import handlers

    handlers._reset_hub_for_tests()
    hub = handlers.get_hub()
    registry = CapabilityRegistry()
    sent_desktop: list[dict] = []
    sent_cli: list[dict] = []

    async def send_desktop(_ws, payload):
        sent_desktop.append(payload)
        return True

    async def send_cli(_ws, payload):
        sent_cli.append(payload)
        return True

    desktop = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=4, device_label="Mac", ws=_Ws(), send_json=send_desktop,
        servers=_shell_catalog(),
    )
    cli = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="cli",
        generation=2, device_label="CLI", ws=_Ws(), send_json=send_cli,
        servers=_shell_catalog(),
    )
    origin = registry.origin_for("dev", "desktop")
    assert origin is not None
    call = asyncio.create_task(registry.call_tool(
        origin,
        "shell",
        "shell_exec",
        {"command": "build", "run_in_background": True},
        session_id="chat:interactive",
        timeout_s=2,
    ))
    await asyncio.sleep(0)
    frame = sent_desktop[-1]
    registry.resolve_result(desktop, {
        "type": "client_tool_result",
        "call_id": frame["call_id"],
        "generation": 4,
        "result": {
            "content": [{"type": "text", "text": "started"}],
            "structuredContent": {"shell_id": "host-shell-1", "status": "running"},
            "isError": False,
        },
    })
    result = await call
    assert result["structuredContent"]["shell_id"] == "host-shell-1"
    desktop_host = ("dev", "desktop", 4)
    cli_host = ("dev", "cli", 2)
    assert hub.has_running("chat:interactive", client_host=desktop_host)
    assert not hub.has_running("chat:interactive", client_host=cli_host)
    assert not hub.has_running("chat:interactive", client_host=None)

    # A real reconnect unregisters the old WebSocket before the replacement
    # hello arrives. Exact same-generation correlation survives that gap and
    # does not move to another instance.
    await registry.unregister(desktop)
    assert registry.connection("dev", "desktop") is None
    desktop = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=4, device_label="Mac", ws=_Ws(), send_json=send_desktop,
        servers=_shell_catalog(),
    )
    assert "host-shell-1" in desktop.client_shells

    # The other live instance cannot forge completion for Desktop's process.
    try:
        registry.receive_tool_event(cli, {
            "type": "client_tool_event",
            "generation": 2,
            "event": {
                "type": "shell_completed", "server": "shell",
                "shell_id": "host-shell-1", "status": "exited",
            },
        })
        raise AssertionError("another instance completed the client shell")
    except ClientCapabilityError as exc:
        assert exc.code == "UNKNOWN_CLIENT_SHELL"

    try:
        registry.receive_tool_event(desktop, {
            "type": "client_tool_event",
            "generation": 3,
            "event": {
                "type": "shell_completed", "server": "shell",
                "shell_id": "host-shell-1", "status": "exited",
            },
        })
        raise AssertionError("stale shell event generation was accepted")
    except ClientCapabilityError as exc:
        assert exc.code == "STALE_GENERATION"

    ack = registry.receive_tool_event(desktop, {
        "type": "client_tool_event",
        "generation": 4,
        "event": {
            "type": "shell_completed",
            "server": "shell",
            "shell_id": "host-shell-1",
            "status": "exited",
            "exit_code": 0,
            "output_bytes": 19,
            # Untrusted routing fields must be ignored. In particular an event
            # cannot wake a scheduler/automatic session.
            "session_id": "scheduler:forged",
            "account_id": "another-account",
        },
    })
    assert ack == {
        "shell_id": "host-shell-1", "accepted": True, "duplicate": False,
    }
    assert not hub.has_running("chat:interactive", client_host=desktop_host)
    assert hub.drain("chat:interactive") == [], "automatic turn saw client event"
    assert hub.drain("chat:interactive", client_host=cli_host) == []
    events = hub.drain("chat:interactive", client_host=desktop_host)
    assert len(events) == 1
    assert events[0].shell_id == "host-shell-1"
    assert events[0].bytes_stdout == 19
    assert events[0].tool_server == "client:shell"
    assert hub.drain("scheduler:forged", client_host=desktop_host) == []

    # Durable broker replay is acknowledged without producing a second
    # autoloop reminder.
    duplicate = registry.receive_tool_event(desktop, {
        "type": "client_tool_event",
        "generation": 4,
        "event": {
            "type": "shell_completed", "server": "shell",
            "shell_id": "host-shell-1", "status": "exited",
        },
    })
    assert duplicate["duplicate"] is True
    assert hub.drain("chat:interactive", client_host=desktop_host) == []


@test(
    "client_capabilities",
    "client shell reconnect ledger rejects generation changes and revocation",
)
async def t_client_shell_reconnect_boundaries(ctx: TestContext) -> None:
    from src.gateway.capabilities import CapabilityRegistry, ClientCapabilityError
    from src.mcp.servers.shell import handlers

    handlers._reset_hub_for_tests()
    hub = handlers.get_hub()
    registry = CapabilityRegistry()
    sent: list[dict] = []

    async def send(_ws, payload):
        sent.append(payload)
        return True

    async def start_shell(conn, *, generation: int, shell_id: str) -> None:
        origin = registry.origin_for("dev", "desktop")
        assert origin is not None
        task = asyncio.create_task(registry.call_tool(
            origin,
            "shell",
            "shell_exec",
            {"command": "build", "run_in_background": True},
            session_id="chat:reconnect",
            timeout_s=2,
        ))
        await asyncio.sleep(0)
        frame = sent[-1]
        registry.resolve_result(conn, {
            "type": "client_tool_result",
            "call_id": frame["call_id"],
            "generation": generation,
            "result": {
                "content": [{"type": "text", "text": "started"}],
                "structuredContent": {"shell_id": shell_id, "status": "running"},
                "isError": False,
            },
        })
        await task

    first = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=1, device_label="Mac", ws=_Ws(), send_json=send,
        servers=_shell_catalog(),
    )
    await start_shell(first, generation=1, shell_id="old-shell")
    old_host = ("dev", "desktop", 1)
    assert hub.has_running("chat:reconnect", client_host=old_host)
    await registry.unregister(first)

    # A generation is part of execution-host identity. A restarted broker may
    # not inherit a process correlation from the generation it replaced.
    second = await registry.register(
        device_id="dev", account_id="network-1", client_instance_id="desktop",
        generation=2, device_label="Mac", ws=_Ws(), send_json=send,
        servers=_shell_catalog(),
    )
    assert "old-shell" not in second.client_shells
    assert not hub.has_running("chat:reconnect", client_host=old_host)
    try:
        registry.receive_tool_event(second, {
            "type": "client_tool_event",
            "generation": 2,
            "event": {
                "type": "shell_completed", "server": "shell",
                "shell_id": "old-shell", "status": "exited",
            },
        })
        raise AssertionError("new generation inherited an old client shell")
    except ClientCapabilityError as exc:
        assert exc.code == "UNKNOWN_CLIENT_SHELL"

    # Revocation is terminal even when the socket already disconnected and the
    # exact-generation reconnect grace ledger is holding the correlation.
    await start_shell(second, generation=2, shell_id="revoked-shell")
    second_host = ("dev", "desktop", 2)
    await registry.unregister(second)
    assert registry._detached_shells  # noqa: SLF001 - protocol invariant test
    await registry.close_device("dev")
    assert not registry._detached_shells  # noqa: SLF001
    assert not hub.has_running("chat:reconnect", client_host=second_host)
    revoked_events = hub.drain("chat:reconnect", client_host=second_host)
    assert len(revoked_events) == 1
    assert revoked_events[0].shell_id == "revoked-shell"
    assert revoked_events[0].signal == "CLIENT_REVOKED"


@test(
    "client_capabilities",
    "synchronous workflows inherit the turn; durable workflows clear it",
)
async def t_workflow_origin_boundaries(ctx: TestContext) -> None:
    from types import SimpleNamespace

    from src.core.execution_origin import (
        TurnExecutionOrigin,
        current_execution_origin,
        execution_origin_scope,
    )
    from src.core.scheduler import Scheduler
    from src.mcp.servers.tool_search.adapters import _call_tool_impl

    seen: list[object] = []
    fallback_calls: list[dict] = []

    async def queued_run_workflow(**kwargs):
        fallback_calls.append(kwargs)
        return {"status": "queued"}

    class Pool:
        def __init__(self):
            self._interactive_workflow_runner = None
            self._toolkit_by_name = {
                "workflow-manager": SimpleNamespace(functions={
                    "workflow_manager_run_workflow": queued_run_workflow,
                }),
            }

        def bind_interactive_workflow_runner(self, runner):
            self._interactive_workflow_runner = runner

        def toolkit_by_name(self, name):
            return self._toolkit_by_name.get(name)

    class DB:
        async def get_workflow(self, id_or_name):
            return {
                "id": "wf-1", "name": id_or_name,
                "graph": {"version": 1, "nodes": [], "edges": [], "variables": {}},
            }

        async def update_workflow_run(self, *_args, **_kwargs):
            return None

    class Executor:
        async def run(self, workflow, **kwargs):
            seen.append(current_execution_origin())
            return {
                "id": kwargs["run_id"], "workflow_id": workflow["id"],
                "status": "success", "trace": [],
            }

    pool = Pool()

    class Agent:
        _mcp = pool

        async def refresh_registries(self):
            return None

    scheduler = Scheduler(DB(), Agent())
    scheduler._workflow_executor = Executor()
    origin = TurnExecutionOrigin(
        device_id="dev", client_instance_id="desktop", generation=7,
        device_label="Mac", registry=object(),
    )

    with execution_origin_scope(origin):
        result = await _call_tool_impl(
            pool,
            "workflow-manager",
            "workflow_manager_run_workflow",
            {"id_or_name": "interactive", "wait": True},
        )
        assert result["status"] == "success"
        assert seen == [origin]

        # Explicitly asynchronous work never captures the origin; it keeps the
        # existing workflow-manager queue path.
        queued = await _call_tool_impl(
            pool,
            "workflow-manager",
            "run_workflow",
            {"id_or_name": "later", "wait": False},
        )
        assert queued["status"] == "queued"
        assert fallback_calls[-1]["id_or_name"] == "later"

        # The scheduler's durable entry point clears even an accidentally
        # inherited context before the executor sees it.
        await scheduler._run_workflow(
            await scheduler.db.get_workflow("automatic"),
            trigger="schedule",
        )
        assert seen[-1] is None

    # With no trusted client origin, wait=True also stays on the durable path.
    automatic = await _call_tool_impl(
        pool,
        "workflow-manager",
        "run_workflow",
        {"id_or_name": "server-owned", "wait": True},
    )
    assert automatic["status"] == "queued"


@test("client_capabilities", "delegation inherits origin; automation explicitly clears it")
async def t_automation_origin_clearing(ctx: TestContext) -> None:
    import src.core.scheduler as scheduler_module
    from src.core.child_session import run_child_session
    from src.core.execution_origin import (
        TurnExecutionOrigin,
        current_execution_origin,
        install_execution_origin,
        reset_execution_origin,
    )

    seen: list[object] = []

    class Agent:
        name = "test"
        model = None

        async def run(self, **_kwargs):
            seen.append(current_execution_origin())
            return "ok"

        async def release_session(self, *_args, **_kwargs):
            return None

    origin = TurnExecutionOrigin(
        device_id="dev", client_instance_id="desktop", generation=1,
        device_label="Mac", registry=object(),
    )
    token = install_execution_origin(origin)
    try:
        await run_child_session(
            agent=Agent(), db=None, parent_session_id="chat:1",
            origin="delegation", title="delegate", prompt="work",
        )
        await run_child_session(
            agent=Agent(), db=None, parent_session_id="scheduler:t1",
            origin="scheduler", title="scheduled", prompt="work",
        )
        await run_child_session(
            agent=Agent(), db=None, parent_session_id="workflow:w1",
            origin="workflow", title="workflow", prompt="work",
        )
        await run_child_session(
            agent=Agent(), db=None, parent_session_id="chat:1",
            origin="workflow", title="inline workflow", prompt="work",
            inherit_execution_origin=True,
        )
        await run_child_session(
            agent=Agent(), db=None, parent_session_id="event:e1",
            origin="event", title="event", prompt="work",
        )
    finally:
        reset_execution_origin(token)
    assert seen[0] is origin
    assert seen[1:] == [None, None, origin, None]

    # Cover the two detached scheduler paths that do not necessarily enter the
    # child-session primitive: legacy scheduled Agent.run and the workflow DAG
    # executor (whose mcp-tool blocks can invoke tools directly).
    automatic_seen: list[object] = []

    class AutomaticAgent:
        name = "automatic"
        model = None

        async def refresh_registries(self):
            return None

        async def run(self, **_kwargs):
            automatic_seen.append(current_execution_origin())
            return "ok"

        async def forget_session(self, *_args, **_kwargs):
            return None

    class WorkflowExecutor:
        async def run(self, *_args, **kwargs):
            automatic_seen.append(current_execution_origin())
            return {"id": kwargs["run_id"], "status": "success"}

    scheduler = scheduler_module.Scheduler(None, AutomaticAgent())
    scheduler._workflow_executor = WorkflowExecutor()
    original_durable = scheduler_module._durable_child_sessions
    scheduler_module._durable_child_sessions = lambda: False
    token = install_execution_origin(origin)
    try:
        await scheduler.run_task({"id": "t1", "name": "task", "prompt": "work"})
        await scheduler._run_workflow(
            {"id": "w1", "name": "workflow"}, trigger="api", run_id="run1",
        )
        assert current_execution_origin() is origin
    finally:
        reset_execution_origin(token)
        scheduler_module._durable_child_sessions = original_durable
    assert automatic_seen == [None, None]


@test("client_capabilities", "stream turn freezes and resets its trusted origin")
async def t_stream_turn_origin(ctx: TestContext) -> None:
    from src.core.execution_origin import TurnExecutionOrigin, current_execution_origin
    from src.stream.session import StreamSession, StreamTurnRunner

    seen: list[object] = []

    class Agent:
        db = None

        async def run_stream(self, **_kwargs):
            seen.append(current_execution_origin())
            yield {"kind": "done", "text": "ok"}

        def last_response_meta(self, _session_id):
            return {}

    origin = TurnExecutionOrigin(
        device_id="dev", client_instance_id="desktop", generation=1,
        device_label="Mac", registry=object(),
    )
    session = StreamSession(
        Agent(), client_id="dev", session_id="s", coalesce_window_ms=0,
    )
    runner = StreamTurnRunner(Agent(), session)
    await runner.run(
        "hello", client_id="dev", session_id="s", execution_origin=origin,
    )
    assert seen == [origin]
    assert current_execution_origin() is None


@test("client_capabilities", "session-open carries instance id and prompt tail is uncached")
async def t_wire_and_prompt_tail(ctx: TestContext) -> None:
    from src.gateway import protocol as P
    from src.gateway.capabilities import CAPABILITY_PROTOCOL
    from src.stream.events import SessionOpen
    from src.stream.wire import event_to_wire, wire_to_event
    from src.models.providers.anthropic.claude import _split_session_id_tag

    evt = SessionOpen(session_id="s", client_instance_id="desktop")
    wire = event_to_wire(evt)
    assert wire["client_instance_id"] == "desktop"
    parsed = wire_to_event(wire)
    assert isinstance(parsed, SessionOpen)
    assert parsed.client_instance_id == "desktop"
    assert CAPABILITY_PROTOCOL == "client-capabilities/1"
    assert (
        P.CAPABILITY_HELLO,
        P.CAPABILITY_HELLO_ACK,
        P.CAPABILITY_CATALOG_UPDATE,
        P.CLIENT_TOOL_CALL,
        P.CLIENT_TOOL_RESULT,
        P.CLIENT_TOOL_CANCEL,
        P.CLIENT_TOOL_EVENT,
        P.CLIENT_TOOL_EVENT_ACK,
        P.CLIENT_ARTIFACT_CHUNK,
    ) == (
        "capability_hello",
        "capability_hello_ack",
        "capability_catalog_update",
        "client_tool_call",
        "client_tool_result",
        "client_tool_cancel",
        "client_tool_event",
        "client_tool_event_ack",
        "client_artifact_chunk",
    )

    prompt = (
        "shared framework\n\n"
        '<execution-host>{"kind":"client"}</execution-host>\n\n'
        "<session-id>s</session-id>"
    )
    body, tail = _split_session_id_tag(prompt)
    assert body == "shared framework"
    assert "<execution-host>" in tail and tail.endswith("<session-id>s</session-id>")
