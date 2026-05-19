#!/usr/bin/env python3
"""Live E2E smoke test against a deployed OpenAgent coordinator.

Two modes:

    # 1. Register a fresh smoke user (consumes a USER invite)
    python scripts/live_smoke.py register \\
        --ticket <oa1...> \\
        --password <pw>

       Prints the handle on success. Use the handle + password for
       the pair step below.

    # 2. Pair a NEW device for an existing user (consumes a DEVICE invite)
    python scripts/live_smoke.py pair \\
        --ticket <oa1...> \\
        --handle <handle> \\
        --password <pw>

Exit code: 0 = success, 1 = failure.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# Repo root → import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


async def _gateway_ws_handshake(
    node, dialer, target_node_id: str, timeout: float = 15.0,
) -> str:
    """Open a cert-prefixed iroh gateway stream + WS upgrade + read AUTH_OK.

    Returns the JSON-decoded AUTH_OK frame on success. Raises on any
    failure, with a message indicating which step broke. This is the
    EXACT path the Electron app takes after a successful SRP login.
    """
    import aiohttp

    proxy = __import__("src.network.client.session", fromlist=["LoopbackProxy"]).LoopbackProxy(
        dialer=dialer, target_node_id=target_node_id,
    )
    host, port = await proxy.start()
    try:
        ws_url = f"ws://{host}:{port}/ws"
        async with aiohttp.ClientSession() as session:
            ws = await asyncio.wait_for(
                session.ws_connect(ws_url, heartbeat=None),
                timeout=timeout,
            )
            try:
                first = await asyncio.wait_for(ws.receive(), timeout=timeout)
                if first.type != aiohttp.WSMsgType.TEXT:
                    raise RuntimeError(
                        f"expected TEXT frame for AUTH_OK, got {first.type.name} "
                        f"(data={first.data!r})",
                    )
                return first.data
            finally:
                await ws.close()
    finally:
        await proxy.stop()


async def _do_register(ticket_str: str, password: str) -> int:
    from src.network.auth.device_cert import verify_cert
    from src.network.client.login import LoginError, login, register
    from src.network.client.session import NetworkBinding, SessionDialer
    from src.network.identity import Identity
    from src.network.iroh_node import IrohNode
    from src.network.client.login import list_agents as coord_list_agents
    from src.network.peers import coordinator_node_id_to_pubkey_bytes
    from src.network.ticket import InviteTicket

    ut = InviteTicket.decode(ticket_str)
    if ut.role != "user":
        print(f"  ✗ ticket role is {ut.role!r}, expected 'user'", flush=True)
        return 1

    coord = ut.coordinator_node_id
    coord_pub = coordinator_node_id_to_pubkey_bytes(coord)
    handle = f"e2esmoke-{uuid.uuid4().hex[:6]}"

    print(f"  coordinator : {coord}", flush=True)
    print(f"  network     : {ut.network_name} / {ut.network_id}", flush=True)
    print(f"  invite code : {ut.code}", flush=True)
    print(f"  handle      : {handle}", flush=True)
    print(f"  password    : {password}", flush=True)

    dev = Identity.generate()
    node = IrohNode(dev)
    await node.start()
    try:
        print("\n[1/2] register + first login", flush=True)
        try:
            cert_wire = await register(
                node=node,
                coordinator_node_id=coord,
                coordinator_pubkey_bytes=coord_pub,
                handle=handle, password=password,
                invite_code=ut.code,
                device_identity=dev,
                network_id=ut.network_id,
                label=f"smoke-register",
            )
        except LoginError as e:
            print(f"  ✗ register failed: {e}", flush=True)
            return 1
        cert = verify_cert(
            cert_wire,
            coordinator_pubkey=Ed25519PublicKey.from_public_bytes(coord_pub),
        )
        print(f"  ✓ registered + cert minted ({cert.handle}@{cert.network_id[:8]}…)",
              flush=True)

        print("\n[2/3] returning-device login (touch_device path)", flush=True)
        try:
            cert_wire2 = await login(
                node=node,
                coordinator_node_id=coord,
                coordinator_pubkey_bytes=coord_pub,
                handle=handle, password=password,
                device_identity=dev,
                network_id=ut.network_id,
            )
        except LoginError as e:
            print(f"  ✗ returning-device login failed: {e}", flush=True)
            return 1
        verify_cert(
            cert_wire2,
            coordinator_pubkey=Ed25519PublicKey.from_public_bytes(coord_pub),
        )
        print("  ✓ cert renewed", flush=True)

        # ── 3. Gateway WS handshake (exact path the Electron app takes) ──
        print("\n[3/3] gateway WS auth (the Electron-app failure path)",
              flush=True)
        try:
            agents = await coord_list_agents(
                node=node, coordinator_node_id=coord,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ list_agents failed: {e}", flush=True)
            return 1
        if not agents:
            print("  ✗ no agents registered in network", flush=True)
            return 1
        agent_node_id = agents[0]["node_id"]
        print(f"  agent: {agents[0].get('handle','?')} @ {agent_node_id[:24]}…",
              flush=True)

        binding = NetworkBinding(
            network_id=ut.network_id,
            network_name=ut.network_name,
            coordinator_node_id=coord,
            coordinator_pubkey_bytes=coord_pub,
            our_handle=handle,
        )
        dialer = SessionDialer(node=node, binding=binding, cert_wire=cert_wire2)
        try:
            auth_ok_payload = await _gateway_ws_handshake(node, dialer, agent_node_id)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ gateway WS handshake failed: {type(e).__name__}: {e}",
                  flush=True)
            await dialer.close()
            return 1
        await dialer.close()
        print(f"  ✓ AUTH_OK received: {auth_ok_payload[:140]}…", flush=True)

    finally:
        await node.stop()

    print(f"\nREGISTER + WS PASSED. Handle for pair step: {handle}", flush=True)
    return 0


async def _do_pair(ticket_str: str, handle: str, password: str) -> int:
    from src.network.auth.device_cert import verify_cert
    from src.network.client.login import LoginError, login
    from src.network.identity import Identity
    from src.network.iroh_node import IrohNode
    from src.network.peers import coordinator_node_id_to_pubkey_bytes
    from src.network.ticket import InviteTicket

    dt = InviteTicket.decode(ticket_str)
    if dt.role != "device":
        print(f"  ✗ ticket role is {dt.role!r}, expected 'device'", flush=True)
        return 1
    if dt.bind_to and dt.bind_to != handle:
        print(f"  ✗ ticket bind_to={dt.bind_to!r}, but --handle={handle!r}",
              flush=True)
        return 1

    coord = dt.coordinator_node_id
    coord_pub = coordinator_node_id_to_pubkey_bytes(coord)
    print(f"  coordinator : {coord}", flush=True)
    print(f"  network     : {dt.network_name} / {dt.network_id}", flush=True)
    print(f"  invite code : {dt.code} (bind_to={dt.bind_to!r})", flush=True)
    print(f"  handle      : {handle}", flush=True)

    new_dev = Identity.generate()  # fresh device pubkey
    node = IrohNode(new_dev)
    await node.start()
    try:
        print("\n[1/1] login with device invite (new device pairing)", flush=True)
        try:
            cert_wire = await login(
                node=node,
                coordinator_node_id=coord,
                coordinator_pubkey_bytes=coord_pub,
                handle=handle, password=password,
                device_identity=new_dev,
                network_id=dt.network_id,
                invite_code=dt.code,
                label="smoke-pair",
            )
        except LoginError as e:
            print(f"  ✗ device pair failed: {e}", flush=True)
            return 1
        verify_cert(
            cert_wire,
            coordinator_pubkey=Ed25519PublicKey.from_public_bytes(coord_pub),
        )
        print("  ✓ new device paired + cert minted", flush=True)
    finally:
        await node.stop()

    print("\nPAIR PHASE PASSED.", flush=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("register", help="Register a smoke user (consumes user invite)")
    pr.add_argument("--ticket", required=True)
    pr.add_argument("--password", required=True)

    pp = sub.add_parser("pair", help="Pair a new device for an existing user (consumes device invite)")
    pp.add_argument("--ticket", required=True)
    pp.add_argument("--handle", required=True)
    pp.add_argument("--password", required=True)

    args = p.parse_args()
    if args.mode == "register":
        rc = asyncio.run(_do_register(args.ticket, args.password))
    else:
        rc = asyncio.run(_do_pair(args.ticket, args.handle, args.password))
    sys.exit(rc)


if __name__ == "__main__":
    main()
