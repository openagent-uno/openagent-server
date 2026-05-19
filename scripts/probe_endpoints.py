#!/usr/bin/env python3
"""Quick endpoint probe — hit each /api/network/* verb against a live
gateway and print status + payload snippet for each, so the user's
"button does nothing" reports can be localized to client vs. server.

Usage:
    python scripts/probe_endpoints.py --ticket <oa1...> --password <pw>
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def _run(ticket_str: str, password: str) -> int:
    import aiohttp
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from src.network.auth.device_cert import verify_cert
    from src.network.client.login import (
        LoginError, login, register, list_agents as coord_list_agents,
    )
    from src.network.client.session import (
        LoopbackProxy, NetworkBinding, SessionDialer,
    )
    from src.network.identity import Identity
    from src.network.iroh_node import IrohNode
    from src.network.peers import coordinator_node_id_to_pubkey_bytes
    from src.network.ticket import InviteTicket

    t = InviteTicket.decode(ticket_str)
    coord = t.coordinator_node_id
    coord_pub = coordinator_node_id_to_pubkey_bytes(coord)
    handle = f"probe-{uuid.uuid4().hex[:6]}"

    dev = Identity.generate()
    node = IrohNode(dev)
    await node.start()

    try:
        # Register + login to mint a usable cert.
        cert_wire = await register(
            node=node, coordinator_node_id=coord, coordinator_pubkey_bytes=coord_pub,
            handle=handle, password=password, invite_code=t.code,
            device_identity=dev, network_id=t.network_id, label="probe",
        )
        verify_cert(cert_wire,
                    coordinator_pubkey=Ed25519PublicKey.from_public_bytes(coord_pub))

        agents = await coord_list_agents(node=node, coordinator_node_id=coord)
        target_node_id = agents[0]["node_id"]

        binding = NetworkBinding(
            network_id=t.network_id, network_name=t.network_name,
            coordinator_node_id=coord, coordinator_pubkey_bytes=coord_pub,
            our_handle=handle,
        )
        dialer = SessionDialer(node=node, binding=binding, cert_wire=cert_wire)
        proxy = LoopbackProxy(dialer=dialer, target_node_id=target_node_id)
        host, port = await proxy.start()
        base = f"http://{host}:{port}"

        async with aiohttp.ClientSession() as s:
            async def probe(method: str, path: str, body: dict | None = None):
                kw = {"json": body} if body is not None else {}
                try:
                    async with s.request(method, f"{base}{path}", **kw) as r:
                        text = await r.text()
                        print(f"  {method:6} {path:50} → {r.status}", flush=True)
                        if r.status >= 400:
                            print(f"         body: {text[:120]}", flush=True)
                        return r.status, text
                except Exception as e:
                    print(f"  {method:6} {path:50} → EXCEPTION: {type(e).__name__}: {e}",
                          flush=True)
                    return None, str(e)

            print("\n── Reads ──", flush=True)
            await probe("GET",  "/api/network/users")
            await probe("GET",  "/api/network/agents")
            await probe("GET",  "/api/network/invitations")

            print("\n── Mint variants ──", flush=True)
            await probe("POST", "/api/network/invitations", {})
            await probe("POST", "/api/network/invitations", {"handle": "probe-onboard"})
            agent_status, agent_body = await probe(
                "POST", "/api/network/invitations", {"role": "agent"},
            )

            print("\n── Edit/remove ──", flush=True)
            await probe("PATCH",  f"/api/network/users/{handle}",
                        {"status": "suspended"})
            await probe("PATCH",  f"/api/network/users/{handle}",
                        {"status": "active"})
            # Try patching the coord's own agent (cosmetic label change).
            coord_handle = agents[0]["handle"]
            await probe("PATCH",  f"/api/network/agents/{coord_handle}",
                        {"label": "probe relabel"})
            # Try DELETE on coord agent — should 409.
            await probe("DELETE", f"/api/network/agents/{coord_handle}")
            # Try DELETE on the probe user (different from caller? no — caller
            # IS the probe user; expect 409 from self-delete guard).
            await probe("DELETE", f"/api/network/users/{handle}")

        await proxy.stop()
        await dialer.close()

    finally:
        await node.stop()

    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticket", required=True)
    p.add_argument("--password", required=True)
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args.ticket, args.password)))


if __name__ == "__main__":
    main()
