"""Device certificates are bound to the Iroh peer that presents them."""
from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ._framework import TestContext, test


@test("device_cert_pop", "concurrent first launch returns one persisted device identity")
async def t_concurrent_identity_first_creation(ctx: TestContext) -> None:
    del ctx
    repo_root = Path(__file__).resolve().parents[2]
    worker_count = 8
    worker = r"""
import os
import sys
import time
from pathlib import Path

from src.network import identity as identity_module

identity_path = Path(sys.argv[1])
barrier_dir = Path(sys.argv[2])
expected = int(sys.argv[3])
original_generate = identity_module.Identity.generate

@classmethod
def coordinated_generate(cls):
    # Force every process past the initial ENOENT before any candidate is
    # published. This deterministically reproduces the former overwrite race.
    (barrier_dir / str(os.getpid())).touch(exist_ok=False)
    deadline = time.monotonic() + 15
    while len(tuple(barrier_dir.iterdir())) < expected:
        if time.monotonic() >= deadline:
            raise TimeoutError("identity race barrier timed out")
        time.sleep(0.005)
    return original_generate()

identity_module.Identity.generate = coordinated_generate
identity = identity_module.load_or_create_identity(identity_path)
print(identity.public_hex, flush=True)
"""

    with tempfile.TemporaryDirectory(prefix="openagent-identity-race-") as tmp:
        temp_root = Path(tmp)
        identity_path = temp_root / "identity.key"
        barrier_dir = temp_root / "barrier"
        barrier_dir.mkdir()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        workers = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    worker,
                    str(identity_path),
                    str(barrier_dir),
                    str(worker_count),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            for _ in range(worker_count)
        ]
        results: list[str] = []
        try:
            for process in workers:
                stdout, stderr = await asyncio.to_thread(process.communicate, None, 20)
                assert process.returncode == 0, stderr
                results.append(stdout.strip())
        finally:
            for process in workers:
                if process.poll() is None:
                    process.kill()
                    process.wait()

        assert len(set(results)) == 1, results
        assert len(identity_path.read_bytes()) == 32
        assert not tuple(temp_root.glob(".identity-*"))
        if os.name == "posix":
            assert stat.S_IMODE(identity_path.stat().st_mode) == 0o600


async def _dial(*, cert_wire: bytes, peer_node_id: str, state):
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    from src.network.auth.middleware import make_auth_middleware
    from src.network.transport.aiohttp_iroh_site import (
        _current_cert_wire,
        _current_peer_node_id,
    )

    middleware = make_auth_middleware(state)
    ran = {"value": False}
    observed = {"auth_kind": None}

    async def handler(request):
        ran["value"] = True
        observed["auth_kind"] = request.get("auth_kind")
        return web.Response(status=200)

    request = make_mocked_request("GET", "/api/agent-info")
    cert_token = _current_cert_wire.set(cert_wire)
    peer_token = _current_peer_node_id.set(peer_node_id)
    try:
        response = await middleware(request, handler)
    finally:
        _current_peer_node_id.reset(peer_token)
        _current_cert_wire.reset(cert_token)
    return response, ran["value"], observed["auth_kind"]


@test("device_cert_pop", "gateway accepts cert only from its Iroh key")
async def t_gateway_cert_is_peer_bound(ctx: TestContext) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.device_cert import issue_cert
    from src.network.auth.middleware import NetworkAuthState
    from src.network.identity import Identity

    coordinator_key = Ed25519PrivateKey.generate()
    device = Identity.generate()
    impostor = Identity.generate()
    cert_wire = issue_cert(
        coordinator_key=coordinator_key,
        handle="alice",
        device_pubkey=device.public_bytes,
        network_id="net-pop",
    )
    state = NetworkAuthState(
        coordinator_pubkey=coordinator_key.public_key(),
        network_id="net-pop",
    )

    accepted, ran, auth_kind = await _dial(
        cert_wire=cert_wire, peer_node_id=device.public_hex, state=state,
    )
    assert accepted.status == 200 and ran
    assert auth_kind == "device_cert"

    rejected, ran, _ = await _dial(
        cert_wire=cert_wire, peer_node_id=impostor.public_hex, state=state,
    )
    assert rejected.status == 401 and not ran
    assert "does not belong" in rejected.text


@test("device_cert_pop", "HTTP token can never register client capabilities")
async def t_capability_ws_rejects_synthetic_http_token(ctx: TestContext) -> None:
    from aiohttp import WSServerHandshakeError, web
    from aiohttp.test_utils import TestClient, TestServer
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.gateway.server import Gateway
    from src.network.auth.middleware import NetworkAuthState, make_auth_middleware
    import src.network.transport.aiohttp_iroh_site as transport

    previous = os.environ.get("OPENAGENT_HTTP_TOKEN")
    os.environ["OPENAGENT_HTTP_TOKEN"] = "synthetic-token-must-not-register"
    original_wire = transport.current_device_cert_wire
    original_agent = transport.current_is_authenticated_agent
    client = None
    try:
        state = NetworkAuthState(
            coordinator_pubkey=Ed25519PrivateKey.generate().public_key(),
            network_id="net-token-rejection",
        )
        middleware = make_auth_middleware(state)
        # Reproduce the hostile context exactly: a token-authenticated request
        # plus arbitrary non-empty bytes where the handler historically only
        # checked truthiness. The trusted middleware provenance must win.
        transport.current_device_cert_wire = lambda: b"not-a-verified-cert"
        transport.current_is_authenticated_agent = lambda: False
        gateway = object.__new__(Gateway)
        app = web.Application(middlewares=[middleware])
        app.router.add_get("/ws/capabilities", gateway._handle_capabilities_ws)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            await client.ws_connect(
                "/ws/capabilities",
                headers={"X-OpenAgent-Token": "synthetic-token-must-not-register"},
            )
        except WSServerHandshakeError as error:
            assert error.status == 403
        else:
            raise AssertionError("synthetic HTTP token registered client capabilities")
    finally:
        if client is not None:
            await client.close()
        transport.current_device_cert_wire = original_wire
        transport.current_is_authenticated_agent = original_agent
        if previous is None:
            os.environ.pop("OPENAGENT_HTTP_TOKEN", None)
        else:
            os.environ["OPENAGENT_HTTP_TOKEN"] = previous


@test("device_cert_pop", "coordinator live roster rejects deleted device")
async def t_gateway_checks_live_device_roster(ctx: TestContext) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.device_cert import issue_cert
    from src.network.auth.middleware import NetworkAuthState
    from src.network.identity import Identity

    coordinator_key = Ed25519PrivateKey.generate()
    device = Identity.generate()
    rows = {device.public_bytes: SimpleNamespace(status="active")}

    async def lookup(key: bytes):
        return rows.get(key)

    state = NetworkAuthState(
        coordinator_pubkey=coordinator_key.public_key(),
        network_id="net-live",
        device_lookup=lookup,
    )
    cert_wire = issue_cert(
        coordinator_key=coordinator_key,
        handle="alice",
        device_pubkey=device.public_bytes,
        network_id="net-live",
    )

    accepted, ran, auth_kind = await _dial(
        cert_wire=cert_wire, peer_node_id=device.public_hex, state=state,
    )
    assert accepted.status == 200 and ran
    assert auth_kind == "device_cert"

    rows.clear()
    rejected, ran, _ = await _dial(
        cert_wire=cert_wire, peer_node_id=device.public_hex, state=state,
    )
    assert rejected.status == 401 and not ran
    assert device.public_bytes not in state.revoked_pubkeys


@test(
    "device_cert_pop",
    "revocation during live-roster authentication rejects the stale request",
)
async def t_auth_epoch_closes_roster_lookup_race(ctx: TestContext) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.device_cert import issue_cert
    from src.network.auth.middleware import NetworkAuthState
    from src.network.identity import Identity

    coordinator_key = Ed25519PrivateKey.generate()
    device = Identity.generate()
    state: NetworkAuthState

    async def lookup(_key: bytes):
        # Reproduce the TOCTOU window: the roster answer was active, but a
        # revocation/suspension callback lands before middleware can publish
        # the authenticated request to a WebSocket handler.
        state.disconnect(device.public_bytes)
        return SimpleNamespace(status="active")

    state = NetworkAuthState(
        coordinator_pubkey=coordinator_key.public_key(),
        network_id="net-auth-race",
        device_lookup=lookup,
    )
    cert_wire = issue_cert(
        coordinator_key=coordinator_key,
        handle="alice",
        device_pubkey=device.public_bytes,
        network_id="net-auth-race",
    )
    rejected, ran, _ = await _dial(
        cert_wire=cert_wire, peer_node_id=device.public_hex, state=state,
    )
    assert rejected.status == 401 and not ran
    assert rejected.text == "device authorization changed"
    assert state.device_epoch(device.public_bytes) == 1


@test("device_cert_pop", "member roster poll disconnects open streams without sticky revoke")
async def t_member_live_roster_poll(ctx: TestContext) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.middleware import NetworkAuthState

    coordinator = Ed25519PrivateKey.generate()
    device = b"\x33" * 32
    active = {device}

    async def remote_lookup(key: bytes):
        return SimpleNamespace(status="active") if key in active else None

    state = NetworkAuthState(
        coordinator_pubkey=coordinator.public_key(),
        network_id="network",
        device_lookup=remote_lookup,
    )
    seen: list[bytes] = []
    state.add_revocation_listener(seen.append)

    assert await state.device_is_active(device)
    active.clear()
    assert await state.revalidate_observed_devices() == [device]
    assert seen == [device]
    assert device not in state.revoked_pubkeys

    # Reactivation remains possible because the member did not manufacture a
    # permanent deny entry from a remote suspension/lookup result.
    active.add(device)
    assert await state.device_is_active(device)


@test("device_cert_pop", "device-status RPC is restricted to enrolled agents")
async def t_device_status_rpc_auth(ctx: TestContext) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.coordinator.service import CoordinatorService

    device = b"\x71" * 32

    class Store:
        async def agent_is_registered(self, node_id: str) -> bool:
            return node_id == "member-node"

        async def get_active_device(self, key: bytes):
            return SimpleNamespace(status="active") if key == device else None

    service = CoordinatorService(
        store=Store(),
        coordinator_key=Ed25519PrivateKey.generate(),
        network_id="network",
        network_name="test",
    )
    assert await service._m_device_status(
        {"device_pubkey": device}, peer_node_id="member-node",
    ) == {"active": True}
    assert await service._m_device_status(
        {"device_pubkey": b"\x72" * 32}, peer_node_id="member-node",
    ) == {"active": False}
    try:
        await service._m_device_status(
            {"device_pubkey": device}, peer_node_id="unregistered-node",
        )
        raise AssertionError("unregistered Iroh peer queried the device roster")
    except Exception as exc:
        assert "enrolled agents" in str(exc)


@test("device_cert_pop", "temporary disconnect closes listeners without revoking pairing")
async def t_temporary_disconnect(ctx: TestContext) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.middleware import NetworkAuthState

    coordinator = Ed25519PrivateKey.generate()
    state = NetworkAuthState(
        coordinator_pubkey=coordinator.public_key(),
        network_id="network",
    )
    device = b"\x44" * 32
    seen: list[bytes] = []
    state.add_revocation_listener(seen.append)

    state.disconnect(device)
    assert seen == [device]
    assert device not in state.revoked_pubkeys

    state.revoke(device)
    assert seen == [device, device]
    assert device in state.revoked_pubkeys


@test("device_cert_pop", "login_finish refuses a claimed key owned by another peer")
async def t_login_finish_requires_peer_key(ctx: TestContext) -> None:
    import uuid

    from .test_coordinator_login_resilience import (
        _FakeStore,
        _make_login_state,
        _make_service,
    )
    from src.network.identity import Identity

    device = Identity.generate()
    impostor = Identity.generate()
    store = _FakeStore(existing_device=None, user_has_other_devices=False)
    service, _ = _make_service(store)
    state_id = "state-" + uuid.uuid4().hex
    service._logins[state_id] = _make_login_state("alice")

    try:
        await service._m_login_finish(
            {
                "state_id": state_id,
                "ke3": b"\x00" * 64,
                "device_pubkey": device.public_bytes,
            },
            peer_node_id=impostor.public_hex,
        )
    except Exception as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("coordinator issued a cert to an unrelated Iroh peer")
