"""Real-wire client capability vertical slice.

This test deliberately does not use ``InProcConnection`` or the gateway's
optional TCP listener. It boots two real Iroh nodes, enrolls the client via
the real coordinator PAKE service, opens both gateway WebSockets through the
real loopback-to-Iroh adapter, and lets a deterministic agent turn dispatch a
filesystem operation through the actual single-instance local broker.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import suppress
from pathlib import Path

from ._framework import TestContext, test


class _DeterministicRuntime:
    history_mode = None

    def set_session_handle(self, _session_id: str, _handle: str) -> None:
        return None


class _DeterministicClientToolAgent:
    """Minimal model-shaped agent that performs one trusted client call."""

    name = "deterministic-client-e2e"
    memory_db = None
    model = _DeterministicRuntime()

    def __init__(self, target: Path) -> None:
        self.target = target
        self.seen_origin = None
        self.tool_result: dict | None = None

    async def refresh_registries(self):
        return False, 1

    async def run_stream(
        self,
        *,
        message,
        user_id,
        session_id,
        attachments=None,
        on_status=None,
        author=None,
    ):
        del message, user_id, attachments, author
        from src.core.execution_origin import current_execution_origin
        from src.mcp.tool_providers import InteractiveClientMCPProvider

        origin = current_execution_origin()
        assert origin is not None, "interactive Iroh turn lost its client origin"
        self.seen_origin = origin
        if on_status is not None:
            await on_status("Using client:filesystem.write_file")
        provider = InteractiveClientMCPProvider(origin.registry)
        self.tool_result = await provider.call_tool(
            "filesystem",
            "write_file",
            {"path": str(self.target), "content": "real-iroh-client"},
            session_id=session_id,
        )
        yield {"kind": "delta", "text": "client file written"}
        yield {"kind": "done", "text": "client file written"}

    def last_response_meta(self, _session_id: str) -> dict:
        return {"model": "deterministic-e2e"}


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
    "real coordinator + Gateway + Iroh + capability host execute one client tool turn",
)
async def t_real_iroh_client_tool_turn(ctx: TestContext) -> None:
    import aiohttp
    from openagent_host_tools import (
        CapabilityBridge,
        HostPaths,
        LocalCapabilityClient,
    )
    from openagent_host_tools.local_broker import LocalBrokerServer

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

    state = await NetworkState.from_db(db=db, identity_path=identity_path)
    target = root / "client-machine" / "sentinel.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    agent = _DeterministicClientToolAgent(target)
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

        session = aiohttp.ClientSession()
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
        assert agent.seen_origin is not None
        assert agent.seen_origin.device_id == client_identity.public_bytes.hex()
        assert agent.seen_origin.client_instance_id == instance_id
        assert agent.tool_result is not None
        execution_host = agent.tool_result.get("execution_host") or {}
        assert execution_host.get("kind") == "client", agent.tool_result
        assert execution_host.get("client_instance_id") == instance_id
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
                await state.stop()
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
