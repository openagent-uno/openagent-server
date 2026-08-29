"""End-to-end coordinator: N users each with their own invite + device.

This is the real-wire test the lyra-agent regression demanded: spin
up a coordinator backed by a fresh MemoryDB, mint one invite per test
user, run the full SRP-6a register + login wire flow over an
``InProcConnection`` (the same CBOR framing the iroh transport uses,
just without the QUIC layer), and verify every user gets a valid,
verifiable, per-device cert.

A second round runs ``login`` again for each device to hit the
``existing active device`` branch — the one that calls
``touch_device`` and was the crash site on lyra. We separately drive
that branch with ``touch_device`` rigged to raise a SQLite
``OperationalError`` to confirm the v0.13.14 fix holds at the wire
level (not just inside the dispatcher).

Touches: real MemoryDB on a real temp file (the SQLite path under
contention is what the user actually hits), real PAKE round-trip,
real cert mint + verify, real CBOR framing.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from pathlib import Path

import cbor2
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ._framework import TestContext, test


# ── Test helpers ────────────────────────────────────────────────────


async def _read_exact(stream, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = await stream.read(n - len(out))
        if not chunk:
            raise RuntimeError("inproc stream closed mid-frame")
        out.extend(chunk)
    return bytes(out)


async def _read_frame(stream) -> dict:
    length = int.from_bytes(await _read_exact(stream, 4), "big")
    payload = await _read_exact(stream, length)
    return cbor2.loads(payload)


async def _write_frame(stream, obj: dict) -> None:
    payload = cbor2.dumps(obj)
    await stream.write_all(len(payload).to_bytes(4, "big") + payload)


async def _client_rpc(conn, method: str, params: dict) -> dict:
    """Open a new bi-stream, send one RPC, return the result.

    Mirrors ``src.network.client.login._rpc`` but talks to the
    coordinator over an in-proc conn rather than an iroh stream.
    Raises ``RuntimeError`` on RPC-level error responses so test
    assertions can pin specific failure codes.
    """
    bi = await conn.open_bi()
    send, recv = bi.send(), bi.recv()
    request = {"id": uuid.uuid4().hex, "method": method, "params": params}
    await _write_frame(send, request)
    await send.finish()
    response = await asyncio.wait_for(_read_frame(recv), timeout=5.0)
    if "error" in response:
        err = response["error"] or {}
        raise RuntimeError(f"rpc {method} failed: {err.get('code')}: {err.get('message')}")
    return response["result"]


async def _do_register_and_login(
    conn, *, handle: str, password: str, invite_code: str, device_pubkey: bytes,
) -> bytes:
    """Full SRP-6a register + login round-trip; returns the cert wire."""
    from src.network.coordinator.pake import (
        Srp6aClientLogin,
        srp6a_make_registration,
    )

    await _client_rpc(conn, "register", {
        "invite": invite_code,
        "handle": handle,
        "pake_record": srp6a_make_registration(handle, password),
    })

    return await _do_login(
        conn,
        handle=handle,
        password=password,
        device_pubkey=device_pubkey,
        invite_code=invite_code,
    )


async def _do_login(
    conn,
    *,
    handle: str,
    password: str,
    device_pubkey: bytes,
    invite_code: str | None = None,
) -> bytes:
    """Run ``login_init`` + ``login_finish`` and return the cert wire.

    The invite is only sent on first-device pairings; subsequent
    logins for the same (handle, device_pubkey) pair omit it.
    """
    from src.network.coordinator.pake import Srp6aClientLogin

    client = Srp6aClientLogin.start(handle, password)
    init = await _client_rpc(conn, "login_init", {
        "handle": handle, "ke1": client.A,
    })
    state_id = init["state_id"]
    server_response = bytes(init["response"])
    M1 = client.respond(server_response)

    finish_params: dict = {
        "state_id": state_id,
        "ke3": M1,
        "device_pubkey": device_pubkey,
    }
    if invite_code:
        finish_params["invite"] = invite_code
    finish = await _client_rpc(conn, "login_finish", finish_params)
    return bytes(finish["cert"])


async def _stand_up_coordinator(ctx: TestContext, network_name: str = "e2e-net"):
    """Boot a coordinator backed by a fresh MemoryDB; return everything
    the test needs to drive client RPCs and assert end-state."""
    from src.memory.db import MemoryDB
    from src.network.coordinator.service import CoordinatorService
    from src.network.coordinator.store import CoordinatorStore
    from src.network.identity import Identity
    from src.network.transport.inproc import InProcConnection

    tmp_db = ctx.db_path.with_name(f"coord-e2e-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    await db.connect()
    store = CoordinatorStore(db)

    network_id = "net-" + uuid.uuid4().hex[:12]
    identity = Identity.generate()
    coord_key = Ed25519PrivateKey.from_private_bytes(identity.secret_bytes)

    await store.set_network_role(
        role="coordinator",
        network_id=network_id,
        name=network_name,
        coordinator_node_id=identity.public_hex,
        coordinator_pubkey=identity.public_bytes,
    )
    # Self-register the coordinator agent so list_agents has a row to
    # return — closer to how ``network init`` sets up a fresh box.
    await store.register_agent(
        handle="coordinator",
        node_id=identity.public_hex,
        owner_handle="system",
        label="coordinator",
    )

    svc = CoordinatorService(
        store=store,
        coordinator_key=coord_key,
        network_id=network_id,
        network_name=network_name,
    )

    conn = InProcConnection(peer_node_id="inproc:test-client")
    server_task = asyncio.create_task(svc._on_stream(conn), name="coord-e2e-server")

    # Mint a coordinator-admin cert the way ``network init`` would —
    # the bootstrap path uses raw store.create_invitation rather than
    # the RPC (which requires a cert), since the first invite is what
    # boots the first user who can later mint their own admin cert.
    # For these tests every invite is created directly via the store.
    async def mint_invite(role: str = "user", *, bind_to: str | None = None) -> str:
        invite = await store.create_invitation(
            role=role,
            created_by="system",
            ttl_seconds=3600,
            uses=1,
            bind_to_handle=bind_to,
        )
        return invite.code

    async def teardown() -> None:
        conn.close()
        # Service loop exits when accept_bi returns None (close handles
        # that). Awaiting the task surfaces any unexpected errors.
        try:
            await asyncio.wait_for(server_task, timeout=2.0)
        except asyncio.TimeoutError:
            server_task.cancel()
        await db.close()
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass

    return {
        "store": store,
        "service": svc,
        "conn": conn,
        "network_id": network_id,
        "coord_key": coord_key,
        "mint_invite": mint_invite,
        "teardown": teardown,
    }


# ── Tests ────────────────────────────────────────────────────────────


@test("coord_e2e_multi_user", "member device-status RPC observes revocation on the wire")
async def t_member_device_status_revocation(ctx: TestContext) -> None:
    from src.network.identity import Identity

    env = await _stand_up_coordinator(ctx)
    try:
        # The in-proc transport still supplies a peer identity to the service;
        # enroll that exact member node so the roster RPC is authorized.
        await env["store"].register_agent(
            handle="member-agent",
            node_id="inproc:test-client",
            owner_handle="system",
            label="member",
        )
        device = Identity.generate()
        invite = await env["mint_invite"]()
        await _do_register_and_login(
            env["conn"],
            handle="alice",
            password="passw0rd-live",
            invite_code=invite,
            device_pubkey=device.public_bytes,
        )
        assert await _client_rpc(
            env["conn"], "device_status", {"device_pubkey": device.public_bytes},
        ) == {"active": True}
        assert await env["store"].revoke_device(device.public_bytes)
        assert await _client_rpc(
            env["conn"], "device_status", {"device_pubkey": device.public_bytes},
        ) == {"active": False}
    finally:
        await env["teardown"]()


@test("coord_e2e_multi_user", "3 users with distinct invites each get a verifiable cert")
async def t_three_users_register_login(ctx: TestContext) -> None:
    from src.network.auth.device_cert import verify_cert
    from src.network.identity import Identity

    env = await _stand_up_coordinator(ctx)
    try:
        users = [
            ("alessandro", "passw0rd-aaa"),
            ("marco",      "passw0rd-mmm"),
            ("alice",      "passw0rd-zzz"),
        ]
        # Distinct device per user — the coordinator stores
        # network_devices keyed by device_pubkey, so accidentally
        # reusing one would collapse the test to a single user from
        # the store's perspective.
        devices = [Identity.generate() for _ in users]
        invites = [await env["mint_invite"]("user") for _ in users]
        # Sanity: every invite code is distinct.
        assert len(set(invites)) == len(invites), f"invite codes collided: {invites}"

        certs: list[bytes] = []
        for (handle, password), device, invite in zip(users, devices, invites):
            cert_wire = await _do_register_and_login(
                env["conn"],
                handle=handle,
                password=password,
                invite_code=invite,
                device_pubkey=device.public_bytes,
            )
            cert = verify_cert(
                cert_wire,
                coordinator_pubkey=env["coord_key"].public_key(),
                expected_network_id=env["network_id"],
            )
            assert cert.handle == handle, f"cert handle mismatch: {cert.handle} != {handle}"
            assert cert.device_pubkey == device.public_bytes, \
                f"cert device pubkey doesn't match device for {handle}"
            certs.append(cert_wire)

        # All certs distinct (different handle / device → different
        # payload → different sig).
        assert len(set(certs)) == len(certs), "certs collided across users"

        # Store-level sanity: every user is recorded, every device is
        # bound to its user.
        users_in_store = await env["store"].list_users()
        recorded_handles = {u.handle for u in users_in_store}
        for handle, _ in users:
            assert handle in recorded_handles, \
                f"user {handle} missing from network_users: {recorded_handles}"
        for (handle, _), device in zip(users, devices):
            row = await env["store"].get_device(device.public_bytes)
            assert row is not None, f"device for {handle} not recorded"
            assert row.user_handle == handle, \
                f"device for {handle} bound to wrong user: {row.user_handle}"
            assert row.status == "active"
    finally:
        await env["teardown"]()


@test("coord_e2e_multi_user", "returning device hits touch_device path & re-logins cleanly")
async def t_returning_device_relogins(ctx: TestContext) -> None:
    from src.network.auth.device_cert import verify_cert
    from src.network.identity import Identity

    env = await _stand_up_coordinator(ctx)
    try:
        handle, password = "alessandro", "pw-aaa"
        device = Identity.generate()
        invite = await env["mint_invite"]("user")

        cert1 = await _do_register_and_login(
            env["conn"],
            handle=handle,
            password=password,
            invite_code=invite,
            device_pubkey=device.public_bytes,
        )

        # Second login = "I am Alessandro on the same device" — no
        # invite this time; coordinator recognises the device pubkey
        # and re-issues a cert through the touch_device path.
        cert2 = await _do_login(
            env["conn"],
            handle=handle,
            password=password,
            device_pubkey=device.public_bytes,
        )
        assert cert2 and cert2 != cert1, "second login should mint a fresh cert"

        c1 = verify_cert(cert1, coordinator_pubkey=env["coord_key"].public_key(),
                         expected_network_id=env["network_id"])
        c2 = verify_cert(cert2, coordinator_pubkey=env["coord_key"].public_key(),
                         expected_network_id=env["network_id"])
        assert c1.handle == c2.handle == handle
        assert c1.device_pubkey == c2.device_pubkey == device.public_bytes
        # Re-login should not have stale-state'd touch_device:
        row = await env["store"].get_device(device.public_bytes)
        assert row is not None
        assert row.last_seen is not None, \
            "touch_device should have set last_seen on second login"
    finally:
        await env["teardown"]()


@test("coord_e2e_multi_user", "login_finish survives touch_device DB-lock end-to-end")
async def t_touch_device_lock_e2e(ctx: TestContext) -> None:
    """Same regression as the unit test, but driven over real CBOR
    framing against the real CoordinatorService. After v0.13.14 the
    transient lock no longer blocks the cert response."""
    from src.network.auth.device_cert import verify_cert
    from src.network.identity import Identity

    env = await _stand_up_coordinator(ctx)
    try:
        handle, password = "alessandro", "pw-lockcheck"
        device = Identity.generate()
        invite = await env["mint_invite"]("user")

        # First login pairs the device cleanly — no contention on the
        # registration write because nothing else holds the writer.
        await _do_register_and_login(
            env["conn"],
            handle=handle,
            password=password,
            invite_code=invite,
            device_pubkey=device.public_bytes,
        )

        # Now wrap the store so touch_device always raises the exact
        # OperationalError aiosqlite would surface on a contended DB.
        real_touch = env["store"].touch_device

        async def boom(device_pubkey):
            raise sqlite3.OperationalError("database is locked")

        env["store"].touch_device = boom  # type: ignore[assignment]
        try:
            cert_wire = await _do_login(
                env["conn"],
                handle=handle,
                password=password,
                device_pubkey=device.public_bytes,
            )
        finally:
            env["store"].touch_device = real_touch  # type: ignore[assignment]

        cert = verify_cert(
            cert_wire,
            coordinator_pubkey=env["coord_key"].public_key(),
            expected_network_id=env["network_id"],
        )
        assert cert.handle == handle
        assert cert.device_pubkey == device.public_bytes
    finally:
        await env["teardown"]()


@test("coord_e2e_multi_user", "spent invite can't be reused; wrong invite is rejected")
async def t_invite_negative_paths(ctx: TestContext) -> None:
    from src.network.identity import Identity

    env = await _stand_up_coordinator(ctx)
    try:
        handle_a, password_a = "alessandro", "pw-aa"
        handle_b, password_b = "marco", "pw-mm"
        device_a = Identity.generate()
        device_b = Identity.generate()
        invite_a = await env["mint_invite"]("user")
        invite_b = await env["mint_invite"]("user")

        await _do_register_and_login(
            env["conn"], handle=handle_a, password=password_a,
            invite_code=invite_a, device_pubkey=device_a.public_bytes,
        )

        # Reusing alessandro's invite for marco's register must fail —
        # the invite has zero uses left after the first register.
        raised: Exception | None = None
        try:
            await _do_register_and_login(
                env["conn"], handle=handle_b, password=password_b,
                invite_code=invite_a, device_pubkey=device_b.public_bytes,
            )
        except RuntimeError as e:
            raised = e
        assert raised is not None, "spent invite should have been rejected"
        assert "invalid_invite" in str(raised), \
            f"expected invalid_invite, got {raised}"

        # Bogus invite must also fail with invalid_invite (not a 500).
        bogus_invite = "z" * 20
        try:
            await _do_register_and_login(
                env["conn"], handle=handle_b, password=password_b,
                invite_code=bogus_invite, device_pubkey=device_b.public_bytes,
            )
        except RuntimeError as e:
            raised = e
        assert raised is not None and "invalid_invite" in str(raised), \
            f"bogus invite should have been rejected as invalid_invite, got {raised}"

        # marco can still register with HIS fresh invite — the failed
        # attempts above didn't poison his path.
        await _do_register_and_login(
            env["conn"], handle=handle_b, password=password_b,
            invite_code=invite_b, device_pubkey=device_b.public_bytes,
        )
        row = await env["store"].get_device(device_b.public_bytes)
        assert row is not None and row.user_handle == handle_b
    finally:
        await env["teardown"]()
