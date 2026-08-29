"""Real-wire client capability vertical slice.

This test deliberately does not use ``InProcConnection`` or the gateway's
optional TCP listener. It boots two real Iroh nodes, enrolls the client via
the real coordinator PAKE service, opens both gateway WebSockets through the
real loopback-to-Iroh adapter, and drives the real :class:`Agent` plus its
runtime tool loop through a deterministic OpenAI-compatible model endpoint.
That agent writes and reads through the actual single-instance local broker.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from contextlib import suppress
from pathlib import Path

from ._framework import TestContext, test


async def _start_deterministic_model_endpoint(target: Path):
    """Serve deterministic OpenAI chat completions that require two tools."""

    from aiohttp import web

    calls: list[dict] = []

    async def chat(request):
        payload = await request.json()
        calls.append(payload)
        tools = payload.get("tools") or []
        names = [
            str((item.get("function") or {}).get("name") or "")
            for item in tools
            if isinstance(item, dict)
        ]
        tool_name = next(
            (name for name in names if name.endswith("tool_search_call_tool")),
            None,
        )
        messages = payload.get("messages") or []
        tool_results = [
            message for message in messages
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        message: dict
        finish_reason: str
        if tool_name and len(tool_results) == 0:
            arguments = {
                "server": "client:filesystem",
                "tool": "write_file",
                "args": {
                    "path": str(target),
                    "content": "real-iroh-client",
                },
            }
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-client-write",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                    },
                }],
            }
            finish_reason = "tool_calls"
        elif tool_name and len(tool_results) == 1:
            arguments = {
                "server": "client:filesystem",
                "tool": "read_text_file",
                "args": {"path": str(target)},
            }
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-client-read",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                    },
                }],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": "client file written"}
            finish_reason = "stop"
        completion = {
            "id": f"chatcmpl-client-e2e-{len(calls)}",
            "object": "chat.completion",
            "created": 1,
            "model": "deterministic-client-e2e",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        if not payload.get("stream"):
            return web.json_response(completion)

        chunk_id = completion["id"]
        chunks: list[dict] = []
        if message.get("tool_calls"):
            chunks.append({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": completion["model"],
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "index": 0,
                            **message["tool_calls"][0],
                        }],
                    },
                    "finish_reason": None,
                }],
            })
        else:
            chunks.append({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": completion["model"],
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": message.get("content") or "",
                    },
                    "finish_reason": None,
                }],
            })
        chunks.append({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": completion["model"],
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        })
        chunks.append({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": completion["model"],
            "choices": [],
            "usage": completion["usage"],
        })
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        })
        await response.prepare(request)
        for chunk in chunks:
            await response.write(
                f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()
            )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # noqa: SLF001 - bound test fixture port
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1", calls


async def _wait_for_direct_addresses(node, *, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    relay = None
    addresses: tuple[str, ...] = ()
    while asyncio.get_running_loop().time() < deadline:
        relay, addresses = await node.local_node_addr()
        if relay or addresses:
            break
        await asyncio.sleep(0.05)
    return relay, list(addresses)


async def _wait_for_owned_broker(server, task: asyncio.Task, *, timeout: float = 8.0) -> None:
    """Wait until the test-owned broker is accepting local clients.

    Calling ``LocalCapabilityClient.start`` before this point would exercise
    its production auto-spawn fallback and leave an intentionally persistent
    detached broker behind after the test process exits.  The server exposes
    the actual bound socket/listener, so readiness does not require a probe
    client or a platform-specific sleep.
    """

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if task.done():
            # Propagate the real startup failure instead of timing out with an
            # unrelated client connection error.
            await task
        ready = (
            server._windows_listener is not None
            if os.name == "nt"
            else server.unix_socket_path.exists()
        )
        if ready:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("test-owned local capability broker did not become ready")


@test(
    "real_iroh_client_e2e",
    "real coordinator + Gateway + Iroh + Agent write and read through the client host",
)
async def t_real_iroh_client_tool_turn(ctx: TestContext) -> None:
    import aiohttp
    from openagent_host_tools import (
        CapabilityBridge,
        HostPaths,
        LocalCapabilityClient,
    )
    from openagent_host_tools.local_broker import LocalBrokerServer

    from src.core.agent import Agent
    from src.core import child_session as child_session_hooks
    from src.gateway.server import Gateway
    from src.memory.db import MemoryDB
    from src.mcp.servers.agent_federation import handlers as federation_handlers
    from src.network.client.login import register
    from src.network.client.session import LoopbackProxy, NetworkBinding, SessionDialer
    from src.network.coordinator.store import CoordinatorStore
    from src.network.identity import Identity, load_or_create_identity
    from src.network.iroh_node import IrohNode
    from src.network.state import NetworkState
    from src.mcp.pool import MCPPool
    from src.models.native_provider import NativeProvider
    from src.stream import child_stream as child_stream_hooks
    from src.stream import resource_events as resource_event_hooks

    root = ctx.db_path.with_name(f"real-iroh-{uuid.uuid4().hex[:8]}")
    root.mkdir(parents=True, exist_ok=True)
    db = MemoryDB(str(root / "gateway.db"))
    await db.connect()
    store = CoordinatorStore(db)
    network_id = "net-" + uuid.uuid4().hex[:12]
    network_name = "real-iroh-e2e"
    identity_path = root / "coordinator.key"
    coordinator_identity = load_or_create_identity(identity_path)
    await store.set_network_role(
        role="coordinator",
        network_id=network_id,
        name=network_name,
        coordinator_node_id=coordinator_identity.public_hex,
        coordinator_pubkey=coordinator_identity.public_bytes,
    )
    await store.register_agent(
        handle="coordinator",
        node_id=coordinator_identity.public_hex,
        owner_handle="system",
        label="coordinator",
    )

    target = root / "client-machine" / "sentinel.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    model_runner, model_base_url, model_calls = (
        await _start_deterministic_model_endpoint(target)
    )
    pool = MCPPool.from_config(
        mcp_config=[{"builtin": "tool-search"}],
        include_defaults=False,
        db_path=str(root / "runtime.db"),
    )
    model = NativeProvider(
        model="local:deterministic-client-e2e",
        api_key="local",
        base_url=model_base_url,
        providers_config=[{
            "name": "local",
            "framework": "api-based",
            "api_key": "local",
            "base_url": model_base_url,
        }],
        db_path=str(root / "runtime.db"),
    )

    agent = Agent(
        name="deterministic-client-e2e",
        model=model,
        system_prompt=(
            "You are the deterministic real-wire capability acceptance agent. "
            "Client paths are never server paths."
        ),
        mcp_pool=pool,
        memory=None,
    )
    state = await NetworkState.from_db(db=db, identity_path=identity_path)
    gateway = Gateway(agent=agent, network_state=state)

    # Gateway wiring uses process-level hooks because production has one
    # Gateway.  The full-suite runner may already have its long-lived Gateway
    # online, so this temporary vertical slice must restore those bindings.
    previous_process_hooks = (
        child_session_hooks._listener,
        child_stream_hooks._broadcast_sink,
        resource_event_hooks._sink,
        federation_handlers._iroh_node,
        federation_handlers._db,
    )

    client_identity = Identity.generate()
    client_node = IrohNode(client_identity)
    proxy = None
    dialer = None
    host = None
    bridge = None
    receive_task = None
    session = None
    capability_ws = None
    chat_ws = None
    broker = None
    broker_task = None
    try:
        gateway._prepare_iroh_site()
        await state.start()
        await gateway.start()
        await client_node.start()

        coordinator_node_id = await state.node_id()
        relay, addresses = await _wait_for_direct_addresses(state.iroh_node)
        invitation = await store.create_invitation(
            role="user",
            created_by="system",
            ttl_seconds=3600,
            uses=1,
            bind_to_handle="alice",
        )
        cert_wire = await register(
            node=client_node,
            coordinator_node_id=coordinator_node_id,
            coordinator_pubkey_bytes=coordinator_identity.public_bytes,
            handle="alice",
            password="real-iroh-password",
            invite_code=invitation.code,
            device_identity=client_identity,
            network_id=network_id,
            label="Real Iroh Client",
            relay_url=relay,
            addresses=addresses,
        )

        dialer = SessionDialer(
            node=client_node,
            binding=NetworkBinding(
                network_id=network_id,
                network_name=network_name,
                coordinator_node_id=coordinator_node_id,
                coordinator_pubkey_bytes=coordinator_identity.public_bytes,
                our_handle="alice",
            ),
            cert_wire=cert_wire,
        )
        proxy = LoopbackProxy(dialer=dialer, target_node_id=coordinator_node_id)
        await proxy.start()

        session = aiohttp.ClientSession()

        # The client may state the network it expects, but the Gateway must
        # derive the binding from the coordinator certificate. A conflicting
        # claim is rejected on the real Iroh stream before any catalog is
        # registered; omitting the v1 extension remains compatible with an
        # older client and the ACK still carries the certified network.
        wrong_network_ws = await session.ws_connect(
            f"{proxy.base_url}/ws/capabilities",
        )
        await wrong_network_ws.send_json({
            "type": "capability_hello",
            "protocol": "client-capabilities/1",
            "client_instance_id": "wrong-network",
            "generation": 1,
            "device_label": "Wrong Network",
            "network_id": "forged-network",
            "servers": [],
        })
        wrong_close = await asyncio.wait_for(wrong_network_ws.receive(), timeout=8)
        assert wrong_close.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
        }, wrong_close
        assert wrong_network_ws.close_code == 4003 or wrong_close.data == 4003

        legacy_ws = await session.ws_connect(f"{proxy.base_url}/ws/capabilities")
        await legacy_ws.send_json({
            "type": "capability_hello",
            "protocol": "client-capabilities/1",
            "client_instance_id": "legacy-no-network-field",
            "generation": 1,
            "device_label": "Legacy Client",
            "servers": [],
        })
        legacy_ack_message = await asyncio.wait_for(legacy_ws.receive(), timeout=8)
        assert legacy_ack_message.type == aiohttp.WSMsgType.TEXT, legacy_ack_message
        legacy_ack = json.loads(legacy_ack_message.data)
        assert legacy_ack["type"] == "capability_hello_ack", legacy_ack
        assert legacy_ack["network_id"] == network_id, legacy_ack
        await legacy_ws.close()

        # The bridge talks to the real single-instance broker over its local
        # user socket/named pipe. The broker is explicitly test-owned: the
        # production client's auto-spawned broker is intentionally persistent,
        # which is the wrong lifecycle for an E2E fixture.
        host_paths = HostPaths.discover(root / "client-user")
        broker = LocalBrokerServer(host_paths)
        broker_task = asyncio.create_task(
            broker.run(), name="real-iroh-local-capability-broker",
        )
        await _wait_for_owned_broker(broker, broker_task)
        host = LocalCapabilityClient(
            paths=host_paths,
        )
        await host.start()
        await host.set_consent(True)
        instance_id = "desktop-real-iroh"
        generation = 41

        capability_ws = await session.ws_connect(
            f"{proxy.base_url}/ws/capabilities",
            autoping=True,
            heartbeat=10,
        )
        bridge = CapabilityBridge(
            host,
            client_instance_id=instance_id,
            generation=generation,
            device_label="Real Iroh Client",
            trusted_account_id=network_id,
            trusted_network_id=network_id,
            trusted_device_id=client_identity.public_bytes.hex(),
            send_json=capability_ws.send_json,
        )
        await bridge.hello()
        hello_ack_msg = await asyncio.wait_for(capability_ws.receive(), timeout=8)
        hello_ack = json.loads(hello_ack_msg.data)
        assert hello_ack["type"] == "capability_hello_ack", hello_ack
        await bridge.handle(hello_ack)
        bridge.activate_events()

        async def receive_capability_frames() -> None:
            async for message in capability_ws:
                if message.type == aiohttp.WSMsgType.TEXT:
                    await bridge.handle(json.loads(message.data))
                elif message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    break

        receive_task = asyncio.create_task(
            receive_capability_frames(), name="real-iroh-capability-receive",
        )

        chat_ws = await session.ws_connect(f"{proxy.base_url}/ws")
        auth_ok = json.loads((await asyncio.wait_for(chat_ws.receive(), timeout=8)).data)
        assert auth_ok["type"] == "auth_ok", auth_ok
        session_id = "real-iroh-session"
        await chat_ws.send_json({
            "type": "session_open",
            "session_id": session_id,
            "client_instance_id": instance_id,
            "coalesce_window_ms": 0,
            "speak": False,
        })
        await chat_ws.send_json({
            "type": "text_final",
            "session_id": session_id,
            "seq": 1,
            "ts_ms": 1,
            "text": "write the client sentinel",
        })

        frames: list[dict] = []
        while True:
            message = await asyncio.wait_for(chat_ws.receive(), timeout=15)
            assert message.type == aiohttp.WSMsgType.TEXT, message
            frame = json.loads(message.data)
            frames.append(frame)
            if frame.get("type") == "turn_complete":
                break

        assert target.read_text() == "real-iroh-client"
        assert len(model_calls) >= 3, model_calls
        assert all(call.get("stream") is True for call in model_calls[:3]), model_calls
        model_transcript = json.dumps(model_calls, sort_keys=True)
        assert "real-iroh-client" in model_transcript, model_transcript
        assert "execution_host" in model_transcript, model_transcript
        assert "'kind': 'client'" in model_transcript, model_transcript
        assert client_identity.public_bytes.hex() in model_transcript, model_transcript
        offered_names = {
            str((item.get("function") or {}).get("name") or "")
            for item in (model_calls[0].get("tools") or [])
            if isinstance(item, dict)
        }
        assert any(
            name.endswith("tool_search_call_tool") for name in offered_names
        ), offered_names
        assert gateway.capabilities.origin_for(
            client_identity.public_bytes.hex(), instance_id,
        ) is not None
        with sqlite3.connect(host_paths.audit_db) as audit_db:
            audit_rows = audit_db.execute(
                "SELECT server, tool, outcome FROM audit ORDER BY seq"
            ).fetchall()
        successful_filesystem_tools = [
            tool
            for server, tool, outcome in audit_rows
            if server == "filesystem" and outcome == "success"
        ]
        assert successful_filesystem_tools[-2:] == [
            "write_file", "read_text_file",
        ], audit_rows
        meta = agent.last_response_meta(session_id)
        assert meta.get("model"), meta
        assert "deterministic-client-e2e" in str(meta["model"]), meta
        assert any(
            frame.get("type") == "delta"
            and "client file written" in str(frame.get("text"))
            for frame in frames
        ), frames
    finally:
        try:
            if chat_ws is not None:
                with suppress(Exception):
                    await chat_ws.close()
            if capability_ws is not None:
                with suppress(Exception):
                    await capability_ws.close()
            if receive_task is not None:
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
            if bridge is not None:
                with suppress(Exception):
                    await bridge.close()
            if session is not None:
                with suppress(Exception):
                    await session.close()
            if host is not None:
                with suppress(Exception):
                    await host.close()
            if broker_task is not None:
                broker_task.cancel()
                await asyncio.gather(broker_task, return_exceptions=True)
            if proxy is not None:
                with suppress(Exception):
                    await proxy.stop()
            if dialer is not None:
                with suppress(Exception):
                    await dialer.close()
            with suppress(Exception):
                await client_node.stop()
            with suppress(Exception):
                await gateway.stop()
            with suppress(Exception):
                await agent.shutdown()
            with suppress(Exception):
                await state.stop()
            with suppress(Exception):
                await model_runner.cleanup()
            with suppress(Exception):
                await db.close()
            with suppress(Exception):
                for path in sorted(root.rglob("*"), reverse=True):
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                    else:
                        path.rmdir()
                root.rmdir()
            if broker is not None:
                assert broker._lock_file is None, "test-owned broker retained its lock"
                if os.name != "nt":
                    assert not broker.unix_socket_path.exists(), (
                        "test-owned broker retained its Unix socket"
                    )
        finally:
            (
                previous_child_listener,
                previous_child_broadcast,
                previous_resource_sink,
                previous_federation_node,
                previous_federation_db,
            ) = previous_process_hooks
            child_session_hooks.set_child_session_listener(previous_child_listener)
            child_stream_hooks.set_child_broadcast_sink(previous_child_broadcast)
            resource_event_hooks.set_resource_event_sink(previous_resource_sink)
            federation_handlers.set_agent_runtime(
                previous_federation_node, previous_federation_db,
            )
