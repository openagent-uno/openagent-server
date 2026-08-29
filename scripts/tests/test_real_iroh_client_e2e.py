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


async def _start_deterministic_model_endpoint(targets: dict[str, Path]):
    """Serve deterministic completions that exercise the selected client host.

    The latest user message selects either the Desktop or CLI sentinel.  Each
    normal turn writes and then reads its host-specific file; the dedicated
    ``desktop-offline`` turn only attempts a read after Desktop disconnects.
    """

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
        latest_user_index = max(
            (
                index for index, item in enumerate(messages)
                if isinstance(item, dict) and item.get("role") == "user"
            ),
            default=-1,
        )
        latest_user = (
            json.dumps(messages[latest_user_index].get("content"), sort_keys=True)
            if latest_user_index >= 0
            else ""
        ).lower()
        if "desktop-offline" in latest_user:
            client_kind = "desktop-offline"
            target = targets["desktop"]
        elif "cli" in latest_user:
            client_kind = "cli"
            target = targets["cli"]
        else:
            client_kind = "desktop"
            target = targets["desktop"]
        tool_results = [
            message for message in messages[latest_user_index + 1:]
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        latest_tool_result = (
            json.dumps(tool_results[-1].get("content"), sort_keys=True)
            if tool_results
            else ""
        )
        latest_tool_failed = (
            '"iserror": true' in latest_tool_result.lower()
            or '"iserror":true' in latest_tool_result.lower()
        )
        message: dict
        finish_reason: str
        if tool_name and len(tool_results) == 0 and client_kind == "desktop-offline":
            arguments = {
                "server": "client:filesystem",
                "tool": "read_text_file",
                "args": {"path": str(target)},
            }
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-client-desktop-offline-read",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                    },
                }],
            }
            finish_reason = "tool_calls"
        elif tool_name and len(tool_results) == 0:
            arguments = {
                "server": "client:filesystem",
                "tool": "write_file",
                "args": {
                    "path": str(target),
                    "content": f"{client_kind}-sentinel",
                },
            }
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call-client-{client_kind}-write",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                    },
                }],
            }
            finish_reason = "tool_calls"
        elif tool_name and len(tool_results) == 1 and not latest_tool_failed:
            arguments = {
                "server": "client:filesystem",
                "tool": "read_text_file",
                "args": {"path": str(target)},
            }
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call-client-{client_kind}-read",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                    },
                }],
            }
            finish_reason = "tool_calls"
        else:
            expected_read_content = f"{client_kind}-sentinel"
            read_succeeded = (
                client_kind != "desktop-offline"
                and len(tool_results) >= 2
                and not latest_tool_failed
                and expected_read_content in latest_tool_result
            )
            message = {
                "role": "assistant",
                "content": (
                    "desktop client unavailable"
                    if client_kind == "desktop-offline"
                    else f"{client_kind} file written"
                    if read_succeeded
                    else f"{client_kind} file read failed"
                ),
            }
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
    "real Iroh routes one session exactly across simultaneous Desktop and CLI hosts",
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

    desktop_target = root / "desktop-machine" / "sentinel.txt"
    cli_target = root / "cli-machine" / "sentinel.txt"
    desktop_target.parent.mkdir(parents=True, exist_ok=True)
    cli_target.parent.mkdir(parents=True, exist_ok=True)
    model_runner, model_base_url, model_calls = (
        await _start_deterministic_model_endpoint({
            "desktop": desktop_target,
            "cli": cli_target,
        })
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
    desktop_host = None
    cli_host = None
    desktop_bridge = None
    cli_bridge = None
    desktop_receive_task = None
    cli_receive_task = None
    session = None
    desktop_capability_ws = None
    cli_capability_ws = None
    desktop_chat_ws = None
    cli_chat_ws = None
    desktop_broker = None
    cli_broker = None
    desktop_broker_task = None
    cli_broker_task = None
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

        # Each simulated client owns a real single-instance local broker. They
        # intentionally use separate roots so both dispatch and local audit
        # attribution remain observable even though this test runs on one OS.
        desktop_paths = HostPaths.discover(root / "desktop-user")
        cli_paths = HostPaths.discover(root / "cli-user")
        desktop_broker = LocalBrokerServer(desktop_paths)
        cli_broker = LocalBrokerServer(cli_paths)
        desktop_broker_task = asyncio.create_task(
            desktop_broker.run(), name="real-iroh-desktop-capability-broker",
        )
        cli_broker_task = asyncio.create_task(
            cli_broker.run(), name="real-iroh-cli-capability-broker",
        )
        await asyncio.gather(
            _wait_for_owned_broker(desktop_broker, desktop_broker_task),
            _wait_for_owned_broker(cli_broker, cli_broker_task),
        )
        desktop_host = LocalCapabilityClient(paths=desktop_paths)
        cli_host = LocalCapabilityClient(paths=cli_paths)
        await asyncio.gather(desktop_host.start(), cli_host.start())
        await asyncio.gather(
            desktop_host.set_consent(True), cli_host.set_consent(True),
        )

        device_id = client_identity.public_bytes.hex()
        desktop_instance_id = "desktop-real-iroh"
        cli_instance_id = "cli-real-iroh"

        async def connect_capability(
            host: LocalCapabilityClient,
            *,
            instance_id: str,
            generation: int,
            device_label: str,
        ):
            ws = await session.ws_connect(
                f"{proxy.base_url}/ws/capabilities",
                autoping=True,
                heartbeat=10,
            )
            local_bridge = CapabilityBridge(
                host,
                client_instance_id=instance_id,
                generation=generation,
                device_label=device_label,
                trusted_account_id=network_id,
                trusted_network_id=network_id,
                trusted_device_id=device_id,
                send_json=ws.send_json,
            )
            await local_bridge.hello()
            hello_ack_msg = await asyncio.wait_for(ws.receive(), timeout=8)
            assert hello_ack_msg.type == aiohttp.WSMsgType.TEXT, hello_ack_msg
            hello_ack = json.loads(hello_ack_msg.data)
            assert hello_ack["type"] == "capability_hello_ack", hello_ack
            assert hello_ack["client_instance_id"] == instance_id, hello_ack
            assert hello_ack["device_id"] == device_id, hello_ack
            await local_bridge.handle(hello_ack)
            local_bridge.activate_events()

            async def receive_capability_frames() -> None:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        await local_bridge.handle(json.loads(message.data))
                    elif message.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        break

            task = asyncio.create_task(
                receive_capability_frames(),
                name=f"real-iroh-{instance_id}-capability-receive",
            )
            return ws, local_bridge, task

        (
            desktop_capability_ws,
            desktop_bridge,
            desktop_receive_task,
        ) = await connect_capability(
            desktop_host,
            instance_id=desktop_instance_id,
            generation=41,
            device_label="Real Iroh Desktop",
        )
        (
            cli_capability_ws,
            cli_bridge,
            cli_receive_task,
        ) = await connect_capability(
            cli_host,
            instance_id=cli_instance_id,
            generation=73,
            device_label="Real Iroh CLI",
        )

        # A second connection under the same certified device identity must
        # coexist, not replace or expel the first.
        desktop_origin = gateway.capabilities.origin_for(
            device_id, desktop_instance_id,
        )
        cli_origin = gateway.capabilities.origin_for(device_id, cli_instance_id)
        assert desktop_origin is not None
        assert cli_origin is not None
        assert desktop_origin.client_instance_id == desktop_instance_id
        assert cli_origin.client_instance_id == cli_instance_id
        assert desktop_origin.generation == 41
        assert cli_origin.generation == 73
        assert not desktop_capability_ws.closed
        assert not cli_capability_ws.closed

        session_id = "real-iroh-shared-session"

        async def connect_chat(*, client_kind: str, instance_id: str):
            ws = await session.ws_connect(f"{proxy.base_url}/ws")
            auth_message = await asyncio.wait_for(ws.receive(), timeout=8)
            assert auth_message.type == aiohttp.WSMsgType.TEXT, auth_message
            auth_ok = json.loads(auth_message.data)
            assert auth_ok["type"] == "auth_ok", auth_ok
            await ws.send_json({
                "type": "session_open",
                "session_id": session_id,
                "client_kind": client_kind,
                "client_instance_id": instance_id,
                "coalesce_window_ms": 0,
                "speak": False,
            })
            return ws

        desktop_chat_ws = await connect_chat(
            client_kind="desktop", instance_id=desktop_instance_id,
        )
        # Avoid racing two SessionOpen frames while the shared holder is first
        # created. The traffic itself still uses only the real Iroh sockets.
        for _ in range(200):
            if any(key[1] == session_id for key in gateway._stream_sessions):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("shared real-Iroh stream session was not created")
        cli_chat_ws = await connect_chat(
            client_kind="cli", instance_id=cli_instance_id,
        )

        async def run_turn(ws, *, seq: int, text: str):
            model_start = len(model_calls)
            await ws.send_json({
                "type": "text_final",
                "session_id": session_id,
                "seq": seq,
                "ts_ms": seq,
                "text": text,
            })
            frames: list[dict] = []
            while True:
                message = await asyncio.wait_for(ws.receive(), timeout=20)
                assert message.type == aiohttp.WSMsgType.TEXT, message
                frame = json.loads(message.data)
                frames.append(frame)
                if frame.get("type") == "turn_complete":
                    break
            return frames, model_calls[model_start:]

        def successful_filesystem_tools(paths: HostPaths) -> list[str]:
            with sqlite3.connect(paths.audit_db) as audit_db:
                rows = audit_db.execute(
                    "SELECT server, tool, outcome FROM audit ORDER BY seq"
                ).fetchall()
            return [
                tool for server, tool, outcome in rows
                if server == "filesystem" and outcome == "success"
            ]

        desktop_frames, desktop_model_calls = await run_turn(
            desktop_chat_ws,
            seq=1,
            text="desktop: write and read the desktop sentinel",
        )
        assert desktop_target.read_text() == "desktop-sentinel"
        assert not cli_target.exists()
        assert successful_filesystem_tools(desktop_paths) == [
            "write_file", "read_text_file",
        ]
        assert successful_filesystem_tools(cli_paths) == []
        assert len(desktop_model_calls) >= 3, desktop_model_calls
        assert all(
            call.get("stream") is True for call in desktop_model_calls[:3]
        ), desktop_model_calls
        desktop_transcript = json.dumps(desktop_model_calls, sort_keys=True)
        assert "desktop-sentinel" in desktop_transcript, desktop_transcript
        assert "execution_host" in desktop_transcript, desktop_transcript
        assert "'kind': 'client'" in desktop_transcript, desktop_transcript
        assert desktop_instance_id in desktop_transcript, desktop_transcript
        assert device_id in desktop_transcript, desktop_transcript

        cli_frames, cli_model_calls = await run_turn(
            cli_chat_ws,
            seq=2,
            text="cli: write and read the cli sentinel",
        )
        assert desktop_target.read_text() == "desktop-sentinel"
        assert cli_target.read_text() == "cli-sentinel"
        assert successful_filesystem_tools(desktop_paths) == [
            "write_file", "read_text_file",
        ]
        assert successful_filesystem_tools(cli_paths) == [
            "write_file", "read_text_file",
        ]
        assert len(cli_model_calls) >= 3, cli_model_calls
        cli_transcript = json.dumps(cli_model_calls, sort_keys=True)
        assert "cli-sentinel" in cli_transcript, cli_transcript
        assert cli_instance_id in cli_transcript, cli_transcript

        offered_names = {
            str((item.get("function") or {}).get("name") or "")
            for item in (desktop_model_calls[0].get("tools") or [])
            if isinstance(item, dict)
        }
        assert any(
            name.endswith("tool_search_call_tool") for name in offered_names
        ), offered_names
        meta = agent.last_response_meta(session_id)
        assert meta.get("model"), meta
        assert "deterministic-client-e2e" in str(meta["model"]), meta
        assert any(
            frame.get("type") == "delta"
            and "desktop file written" in str(frame.get("text"))
            for frame in desktop_frames
        ), desktop_frames
        assert any(
            frame.get("type") == "delta"
            and "cli file written" in str(frame.get("text"))
            for frame in cli_frames
        ), cli_frames

        # Disconnect Desktop while CLI remains live. A new turn submitted on
        # Desktop's still-authenticated chat socket must become server-only;
        # its explicit client:* call fails and is never redirected to CLI.
        desktop_audit_before = successful_filesystem_tools(desktop_paths)
        cli_audit_before = successful_filesystem_tools(cli_paths)
        await desktop_capability_ws.close()
        await asyncio.wait_for(desktop_receive_task, timeout=8)
        for _ in range(200):
            if gateway.capabilities.origin_for(
                device_id, desktop_instance_id,
            ) is None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("disconnected Desktop capability stayed registered")
        assert gateway.capabilities.origin_for(
            device_id, cli_instance_id,
        ) is not None
        assert not cli_capability_ws.closed

        offline_frames, offline_model_calls = await run_turn(
            desktop_chat_ws,
            seq=3,
            text="desktop-offline: read the disconnected desktop sentinel",
        )
        assert successful_filesystem_tools(desktop_paths) == desktop_audit_before
        assert successful_filesystem_tools(cli_paths) == cli_audit_before
        offline_transcript = json.dumps(offline_model_calls, sort_keys=True)
        assert "Client MCPs are unavailable" in offline_transcript, offline_transcript
        assert any(
            frame.get("type") == "delta"
            and "desktop client unavailable" in str(frame.get("text"))
            for frame in offline_frames
        ), offline_frames
    finally:
        try:
            if desktop_chat_ws is not None:
                with suppress(Exception):
                    await desktop_chat_ws.close()
            if cli_chat_ws is not None:
                with suppress(Exception):
                    await cli_chat_ws.close()
            if desktop_capability_ws is not None:
                with suppress(Exception):
                    await desktop_capability_ws.close()
            if cli_capability_ws is not None:
                with suppress(Exception):
                    await cli_capability_ws.close()
            for receive_task in (desktop_receive_task, cli_receive_task):
                if receive_task is not None:
                    receive_task.cancel()
                    await asyncio.gather(receive_task, return_exceptions=True)
            for bridge in (desktop_bridge, cli_bridge):
                if bridge is not None:
                    with suppress(Exception):
                        await bridge.close()
            if session is not None:
                with suppress(Exception):
                    await session.close()
            for host in (desktop_host, cli_host):
                if host is not None:
                    with suppress(Exception):
                        await host.close()
            for broker_task in (desktop_broker_task, cli_broker_task):
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
            for broker in (desktop_broker, cli_broker):
                if broker is not None:
                    assert broker._lock_file is None, (
                        "test-owned broker retained its lock"
                    )
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
