"""BridgeSession + InProcConnection — in-process iroh-shaped transport.

Covers the path that wires telegram/discord/whatsapp bridges to the
gateway without going through real iroh. Specifically:

- ``InProcConnection`` round-trips bi-stream bytes correctly.
- ``InProcDialer`` writes the cert prefix the gateway expects.
- ``BridgeSession`` rejects member-mode and missing-coordinator-key
  scenarios with ``BridgeSessionUnavailable``.
- ``BridgeSession`` happy path: produces a working LoopbackProxy whose
  ws_url accepts bridge connections, the synthetic IrohSite handler
  receives the cert, and the auth middleware would accept it (we
  verify the cert signature directly here since we don't stand up a
  full gateway).
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from ._framework import TestContext, test


@test("bridge_session", "InProcConnection round-trips bytes through bi-stream")
async def t_inproc_roundtrip(ctx: TestContext) -> None:
    from openagent.network.transport.inproc import (
        InProcConnection,
        InProcDialer,
    )

    conn = InProcConnection()

    async def server() -> bytes:
        bi = await conn.accept_bi()
        recv = bi.recv()
        ln = int.from_bytes(await recv.read(4), "big")
        cert = await recv.read(ln)
        body = await recv.read(64)
        await bi.send().write_all(b"echo:" + body)
        await bi.send().finish()
        return cert

    server_task = asyncio.create_task(server())
    dialer = InProcDialer(connection=conn, cert_wire=b"FAKECERT")
    stream = await dialer.open_gateway_stream("inproc")
    await stream.send.write_all(b"hello")
    await stream.send.finish()
    reply = await asyncio.wait_for(stream.recv.read(64), timeout=2)
    cert_received = await asyncio.wait_for(server_task, timeout=2)

    assert reply == b"echo:hello", f"echo body wrong: {reply!r}"
    assert cert_received == b"FAKECERT", f"cert wrong: {cert_received!r}"


@test("bridge_session", "InProcConnection.close wakes pending accept_bi")
async def t_inproc_close_wakes_accept(ctx: TestContext) -> None:
    from openagent.network.transport.inproc import InProcConnection

    conn = InProcConnection()
    accepted = asyncio.create_task(conn.accept_bi())
    await asyncio.sleep(0.05)
    conn.close()
    result = await asyncio.wait_for(accepted, timeout=1)
    assert result is None, f"close should yield None, got {result!r}"


@test("bridge_session", "BridgeSession rejects member-mode")
async def t_bridge_rejects_member(ctx: TestContext) -> None:
    from openagent.network.bridge_session import (
        BridgeSession,
        BridgeSessionUnavailable,
    )

    network_state = MagicMock()
    network_state.role = "member"
    network_state.coordinator_key = None
    site = MagicMock()
    session = BridgeSession(bridge_name="telegram")
    raised = False
    try:
        with tempfile.TemporaryDirectory() as d:
            await session.start(
                network_state=network_state,
                gateway_site=site,
                agent_dir=Path(d),
            )
    except BridgeSessionUnavailable as e:
        raised = True
        assert "coordinator-mode" in str(e), f"unexpected message: {e}"
    assert raised, "expected BridgeSessionUnavailable for member-mode"


@test("bridge_session", "BridgeSession rejects missing coordinator key")
async def t_bridge_rejects_no_coord_key(ctx: TestContext) -> None:
    from openagent.network.bridge_session import (
        BridgeSession,
        BridgeSessionUnavailable,
    )

    network_state = MagicMock()
    network_state.role = "coordinator"
    network_state.coordinator_key = None  # bad
    site = MagicMock()
    session = BridgeSession(bridge_name="telegram")
    raised = False
    try:
        with tempfile.TemporaryDirectory() as d:
            await session.start(
                network_state=network_state,
                gateway_site=site,
                agent_dir=Path(d),
            )
    except BridgeSessionUnavailable as e:
        raised = True
        assert "coordinator key" in str(e), f"unexpected message: {e}"
    assert raised, "expected BridgeSessionUnavailable for missing key"


def _make_network_state_for_test(coord_key, network_id="test-network-uuid"):
    network_state = MagicMock()
    network_state.role = "coordinator"
    network_state.coordinator_key = coord_key
    network_state.network_id = network_id
    network_state.network_name = "test-net"
    network_state.identity = MagicMock(
        public_bytes=coord_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ),
    )
    return network_state


def _make_fake_site(streams_received):
    async def fake_handle_stream(connection) -> None:
        while True:
            bi = await connection.accept_bi()
            if bi is None:
                return
            recv = bi.recv()
            ln = int.from_bytes(await recv.read(4), "big")
            cert_wire = await recv.read(ln)
            streams_received.append(cert_wire)
            try:
                while True:
                    chunk = await recv.read(4096)
                    if not chunk:
                        break
            except Exception:
                pass

    fake_site = MagicMock()
    fake_site._handle_stream = fake_handle_stream
    return fake_site


@test("bridge_session", "BridgeSession happy path mints a verifiable cert + working LoopbackProxy")
async def t_bridge_happy_path(ctx: TestContext) -> None:
    from openagent.network.bridge_session import (
        BridgeSession,
        bridge_handle_for,
    )
    from openagent.network.auth.device_cert import verify_cert

    coord_key = Ed25519PrivateKey.generate()
    network_id = "test-network-uuid"
    network_state = _make_network_state_for_test(coord_key, network_id)
    streams_received: list = []
    fake_site = _make_fake_site(streams_received)

    session = BridgeSession(bridge_name="telegram")
    with tempfile.TemporaryDirectory() as d:
        await session.start(
            network_state=network_state,
            gateway_site=fake_site,
            agent_dir=Path(d),
        )
        ws = session.ws_url
        assert ws.startswith("ws://127.0.0.1:"), f"unexpected ws_url: {ws}"

        # The cert MUST verify against the coordinator pubkey we used
        # to mint it, with the bridge handle and the right network.
        cert = verify_cert(
            session.cert_wire,
            coordinator_pubkey=coord_key.public_key(),
            expected_network_id=network_id,
        )
        assert cert.handle == bridge_handle_for("telegram"), (
            f"cert handle should be __bridge_telegram, got {cert.handle!r}"
        )
        assert cert.handle == "__bridge_telegram"
        assert "bridge" in cert.capabilities

        # Open a TCP connection to the loopback's HTTP port and write
        # an arbitrary HTTP-like prefix. The LoopbackProxy should
        # forward those bytes to a fresh InProcConnection bi-stream;
        # the fake site handler should record one cert wire.
        host = ws.replace("ws://", "").split("/")[0]
        host, port_s = host.split(":")
        reader, writer = await asyncio.open_connection(host, int(port_s))
        writer.write(b"GET /api/health HTTP/1.1\r\nHost: x\r\n\r\n")
        await writer.drain()
        await asyncio.sleep(0.1)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        assert len(streams_received) == 1, (
            f"expected 1 stream cert recorded, got {len(streams_received)}"
        )
        assert streams_received[0] == session.cert_wire, (
            "cert observed by site handler differs from BridgeSession.cert_wire"
        )

        await session.stop()


@test("bridge_session", "two BridgeSessions in the same agent_dir get distinct keys, certs, ws_urls")
async def t_bridge_distinct_per_bridge(ctx: TestContext) -> None:
    """Regression test for v0.12.49 friday outage.

    Pre-fix: telegram + whatsapp on one agent shared a single
    BridgeSession → single device_pubkey → single client_id on the
    gateway. The second bridge's WS reconnect kicked the first off
    via gateway.client_replaced, breaking the first bridge's send
    half. Post-fix: each bridge has its own BridgeSession with a
    distinct device key, distinct cert handle, distinct LoopbackProxy
    bound port — so the gateway sees them as independent clients.
    """
    from openagent.network.bridge_session import (
        BridgeSession,
        bridge_handle_for,
        bridge_device_key_filename,
    )
    from openagent.network.auth.device_cert import verify_cert

    coord_key = Ed25519PrivateKey.generate()
    network_state = _make_network_state_for_test(coord_key)
    streams: list = []
    site = _make_fake_site(streams)

    with tempfile.TemporaryDirectory() as d:
        s_tg = BridgeSession(bridge_name="telegram")
        s_wa = BridgeSession(bridge_name="whatsapp")
        await s_tg.start(
            network_state=network_state, gateway_site=site, agent_dir=Path(d),
        )
        await s_wa.start(
            network_state=network_state, gateway_site=site, agent_dir=Path(d),
        )
        try:
            tg_cert = verify_cert(
                s_tg.cert_wire, coordinator_pubkey=coord_key.public_key(),
            )
            wa_cert = verify_cert(
                s_wa.cert_wire, coordinator_pubkey=coord_key.public_key(),
            )

            # 1. Distinct cert handles — proves the gateway will see
            #    them as different client_ids (gateway derives client_id
            #    from device_pubkey, but the handles differ + the
            #    pubkeys differ, so client_ids differ).
            assert tg_cert.handle == "__bridge_telegram", tg_cert.handle
            assert wa_cert.handle == "__bridge_whatsapp", wa_cert.handle
            assert tg_cert.handle == bridge_handle_for("telegram")
            assert wa_cert.handle == bridge_handle_for("whatsapp")

            # 2. Distinct device pubkeys — the actual collision driver
            #    on the gateway side. Same pubkey across bridges =
            #    same client_id = client_replaced cascade.
            assert tg_cert.device_pubkey != wa_cert.device_pubkey, (
                "device_pubkey must differ per bridge — same pubkey "
                "would re-introduce the gateway client_id collision"
            )

            # 3. Distinct LoopbackProxy bound ports — each bridge has
            #    its own localhost endpoint to send WS to.
            assert s_tg.ws_url != s_wa.ws_url, (
                f"ws_urls collided: tg={s_tg.ws_url} wa={s_wa.ws_url}"
            )

            # 4. Distinct on-disk device key files (so a restart keeps
            #    each bridge's identity stable independently).
            tg_key = Path(d) / bridge_device_key_filename("telegram")
            wa_key = Path(d) / bridge_device_key_filename("whatsapp")
            assert tg_key.exists() and wa_key.exists()
            assert tg_key.read_bytes() != wa_key.read_bytes()
            # Both must be 32 bytes (raw ed25519 secret).
            assert len(tg_key.read_bytes()) == 32
            assert len(wa_key.read_bytes()) == 32
        finally:
            await s_tg.stop()
            await s_wa.stop()


@test("bridge_session", "BridgeSession persists device key across restarts (per-bridge)")
async def t_bridge_persists_device_key(ctx: TestContext) -> None:
    from openagent.network.bridge_session import (
        bridge_device_key_filename,
        _load_or_create_bridge_device_key,
    )

    with tempfile.TemporaryDirectory() as d:
        agent = Path(d)
        k1_tg = _load_or_create_bridge_device_key(agent, "telegram")
        k2_tg = _load_or_create_bridge_device_key(agent, "telegram")
        b1 = k1_tg.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        b2 = k2_tg.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        assert b1 == b2, "device key was not persisted across calls"
        assert (agent / bridge_device_key_filename("telegram")).exists()
        # A different bridge_name should produce a different key.
        k_wa = _load_or_create_bridge_device_key(agent, "whatsapp")
        b_wa = k_wa.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        assert b_wa != b1, "whatsapp key should differ from telegram key"
        assert (agent / bridge_device_key_filename("whatsapp")).exists()


@test("bridge_session", "BridgeSession rejects invalid bridge_name")
async def t_bridge_rejects_bad_name(ctx: TestContext) -> None:
    from openagent.network.bridge_session import BridgeSession

    for bad in ("", "has space", "../escape", "name/with/slash"):
        try:
            BridgeSession(bridge_name=bad)
        except ValueError:
            continue
        raise AssertionError(f"BridgeSession({bad!r}) should have raised ValueError")
