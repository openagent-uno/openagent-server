"""Peer-network management: an agent acting as a CLIENT of other networks.

The local agent's *home* network sits in the singleton ``network`` row.
``peer_networks`` is the registry of OTHER networks this agent joins
to talk to peer agents (federation). Each row pairs a network_id with
the coordinator's pinned NodeId/pubkey and the handle this agent uses
to authenticate there.

REST surface (``/api/peers``):

    GET    /api/peers                       -> list rows + cached cert status
    POST   /api/peers/join                  -> join via agent invite ticket (no password)
    POST   /api/peers                       -> join via SRP user login (handle + password)
    DELETE /api/peers/{network_id}          -> drop a peer membership
    POST   /api/peers/{network_id}/refresh  -> force a cert refresh
    POST   /api/peers/{network_id}/chat     -> send a message to a peer agent
    GET    /api/peers/{network_id}/agents   -> list agents in peer network
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import aiosqlite
from aiohttp import web

from src.core.logging import elog
from src.memory.db import MemoryDB
from src.network.auth.device_cert import (
    CertVerificationError,
    verify_cert,
)
from src.network.client.login import (
    LoginError,
    fetch_network_info,
    list_agents as coord_list_agents,
    login as coord_login,
    refresh_cert as coord_refresh_cert,
)
from src.network.client.session import NetworkBinding, SessionDialer
from src.network.identity import Identity, load_or_create_identity
from src.network.iroh_node import IrohNode

logger = logging.getLogger(__name__)


@dataclass
class PeerNetworkRow:
    network_id: str
    name: str
    coordinator_node_id: str
    coordinator_pubkey: bytes
    our_handle: str
    status: str
    added_at: float
    last_seen: float | None
    join_type: str = "user"  # "user" (SRP) | "agent" (Iroh key-based)


class PeerStore:
    """Async helpers around the ``peer_networks`` + ``device_certs`` tables."""

    def __init__(self, db: MemoryDB) -> None:
        self._db = db

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db._conn is None:
            raise RuntimeError("MemoryDB.connect() must be called before PeerStore use")
        return self._db._conn

    async def list_peers(self) -> list[PeerNetworkRow]:
        cur = await self._conn.execute(
            "SELECT network_id, name, coordinator_node_id, coordinator_pubkey, our_handle, "
            "status, added_at, last_seen, "
            "COALESCE(join_type, 'user') AS join_type "
            "FROM peer_networks ORDER BY added_at",
        )
        return [PeerNetworkRow(**dict(row)) for row in await cur.fetchall()]

    async def get_peer(self, network_id: str) -> PeerNetworkRow | None:
        cur = await self._conn.execute(
            "SELECT network_id, name, coordinator_node_id, coordinator_pubkey, our_handle, "
            "status, added_at, last_seen, "
            "COALESCE(join_type, 'user') AS join_type "
            "FROM peer_networks WHERE network_id=?",
            (network_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return PeerNetworkRow(**dict(row))

    async def add_peer(
        self,
        *,
        network_id: str,
        name: str,
        coordinator_node_id: str,
        coordinator_pubkey: bytes,
        our_handle: str,
        join_type: str = "user",
    ) -> None:
        await self._conn.execute(
            "INSERT INTO peer_networks (network_id, name, coordinator_node_id, "
            "coordinator_pubkey, our_handle, status, added_at, join_type) "
            "VALUES (?, ?, ?, ?, ?, 'active', ?, ?) "
            "ON CONFLICT(network_id) DO UPDATE SET name=excluded.name, "
            "coordinator_node_id=excluded.coordinator_node_id, "
            "coordinator_pubkey=excluded.coordinator_pubkey, "
            "our_handle=excluded.our_handle, status='active', "
            "join_type=excluded.join_type",
            (network_id, name, coordinator_node_id, coordinator_pubkey,
             our_handle, time.time(), join_type),
        )
        await self._conn.commit()

    async def remove_peer(self, network_id: str) -> bool:
        cur = await self._conn.execute(
            "DELETE FROM peer_networks WHERE network_id=?",
            (network_id,),
        )
        await self._conn.execute(
            "DELETE FROM device_certs WHERE network_id=?",
            (network_id,),
        )
        await self._conn.commit()
        return cur.rowcount > 0

    async def store_cert(self, *, network_id: str, handle: str, cert_wire: bytes, expires_at: float) -> None:
        await self._conn.execute(
            "INSERT INTO device_certs (network_id, handle, cert, expires_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(network_id, handle) DO UPDATE SET cert=excluded.cert, "
            "expires_at=excluded.expires_at",
            (network_id, handle, cert_wire, expires_at),
        )
        await self._conn.commit()

    async def get_cert(self, *, network_id: str, handle: str) -> tuple[bytes, float] | None:
        cur = await self._conn.execute(
            "SELECT cert, expires_at FROM device_certs WHERE network_id=? AND handle=?",
            (network_id, handle),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return bytes(row[0]), float(row[1])


# ── Builders ────────────────────────────────────────────────────────────


async def make_dialer_for_peer(
    *,
    db: MemoryDB,
    peer: PeerNetworkRow,
    node: IrohNode,
    refresh_password: str | None = None,
    device_identity: Identity | None = None,
) -> SessionDialer:
    """Build a SessionDialer for an existing peer-network membership.

    Two refresh paths depending on ``peer.join_type``:
    - ``"agent"``: re-runs ``agent_login`` using the local Iroh identity
      (no password needed — the QUIC handshake is the proof of ownership).
    - ``"user"``: re-runs SRP-6a login; requires ``refresh_password`` and
      ``device_identity``.
    """
    store = PeerStore(db)
    cached = await store.get_cert(network_id=peer.network_id, handle=peer.our_handle)
    cert_wire: bytes | None = cached[0] if cached else None
    expires_at = cached[1] if cached else 0.0

    if cert_wire is None or expires_at <= time.time():
        if peer.join_type == "agent":
            # Agent refresh: no password — use Iroh keypair as proof.
            from src.network.client.login import agent_login
            node_id = await node.node_id()
            cert_wire = await agent_login(
                node=node,
                coordinator_node_id=peer.coordinator_node_id,
                coordinator_pubkey_bytes=peer.coordinator_pubkey,
                handle=peer.our_handle,
                node_id=node_id,
                invite_code="",   # empty — refresh uses existing agent record
                network_id=peer.network_id,
            )
        else:
            if refresh_password is None or device_identity is None:
                raise LoginError(
                    f"no valid cert for {peer.our_handle}@{peer.name} "
                    "and no credentials to refresh",
                )
            cert_wire = await coord_refresh_cert(
                node=node,
                coordinator_node_id=peer.coordinator_node_id,
                coordinator_pubkey_bytes=peer.coordinator_pubkey,
                handle=peer.our_handle,
                password=refresh_password,
                device_identity=device_identity,
                network_id=peer.network_id,
            )
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            pubkey = Ed25519PublicKey.from_public_bytes(peer.coordinator_pubkey)
            cert = verify_cert(
                cert_wire,
                coordinator_pubkey=pubkey,
                expected_network_id=peer.network_id,
            )
        except CertVerificationError as e:
            raise LoginError(f"refreshed cert failed verification: {e}") from e
        await store.store_cert(
            network_id=peer.network_id,
            handle=peer.our_handle,
            cert_wire=cert_wire,
            expires_at=cert.expires_at,
        )

    binding = NetworkBinding(
        network_id=peer.network_id,
        network_name=peer.name,
        coordinator_node_id=peer.coordinator_node_id,
        coordinator_pubkey_bytes=peer.coordinator_pubkey,
        our_handle=peer.our_handle,
    )
    return SessionDialer(node=node, binding=binding, cert_wire=cert_wire)


# ── REST handlers ──────────────────────────────────────────────────────


async def handle_list(request: web.Request) -> web.Response:
    """GET /api/peers — list peer networks this agent has joined."""
    gw = request.app["gateway"]
    db = gw.agent._db
    if db is None:
        return web.json_response({"error": "no DB attached"}, status=503)
    store = PeerStore(db)
    rows = await store.list_peers()
    out = []
    for r in rows:
        cert = await store.get_cert(network_id=r.network_id, handle=r.our_handle)
        out.append({
            "network_id": r.network_id,
            "name": r.name,
            "coordinator_node_id": r.coordinator_node_id,
            "our_handle": r.our_handle,
            "status": r.status,
            "added_at": r.added_at,
            "last_seen": r.last_seen,
            "cert_expires_at": cert[1] if cert else None,
        })
    return web.json_response({"peers": out})


async def handle_create(request: web.Request) -> web.Response:
    """POST /api/peers — add a new peer-network membership.

    Body: ``{coordinator_node_id, handle, password, invite?, label?}``.
    Performs a fresh login, pins the coordinator pubkey, persists the
    membership + cert. The handler is synchronous-ish (login is one
    Iroh round-trip) so the UI gets a single OK/fail response.
    """
    gw = request.app["gateway"]
    db = gw.agent._db
    if db is None:
        return web.json_response({"error": "no DB attached"}, status=503)
    body = await request.json()
    coordinator_node_id = body.get("coordinator_node_id")
    handle = (body.get("handle") or "").strip().lower()
    password = body.get("password") or ""
    invite = body.get("invite")
    if not (coordinator_node_id and handle and password):
        return web.json_response(
            {"error": "coordinator_node_id, handle, password are required"},
            status=400,
        )

    state = getattr(gw, "_network_state", None)
    if state is None:
        return web.json_response({"error": "gateway has no network state"}, status=500)

    info = await fetch_network_info(node=state.iroh_node, coordinator_node_id=coordinator_node_id)
    network_id = info["network_id"]
    network_name = info.get("name") or network_id
    coord_pubkey_bytes = bytes(coordinator_node_id_to_pubkey_bytes(coordinator_node_id))

    user_identity_path_value = body.get("device_identity_path") or None
    if user_identity_path_value:
        device_identity = load_or_create_identity(user_identity_path_value)
    else:
        # Federation: peer agent uses its own agent identity for inbound
        # auth at the peer network. Fine because the agent IS a "device"
        # in that network's eyes.
        device_identity = state.identity

    try:
        cert_wire = await coord_login(
            node=state.iroh_node,
            coordinator_node_id=coordinator_node_id,
            coordinator_pubkey_bytes=coord_pubkey_bytes,
            handle=handle,
            password=password,
            device_identity=device_identity,
            network_id=network_id,
            invite_code=invite,
            label=body.get("label"),
        )
    except LoginError as e:
        elog("peers.add_failed", level="warning", error=str(e), handle=handle, network=network_name)
        return web.json_response({"error": str(e)}, status=400)

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pubkey = Ed25519PublicKey.from_public_bytes(coord_pubkey_bytes)
        cert = verify_cert(
            cert_wire, coordinator_pubkey=pubkey, expected_network_id=network_id,
        )
    except CertVerificationError as e:
        return web.json_response({"error": f"cert verification failed: {e}"}, status=502)

    store = PeerStore(db)
    await store.add_peer(
        network_id=network_id,
        name=network_name,
        coordinator_node_id=coordinator_node_id,
        coordinator_pubkey=coord_pubkey_bytes,
        our_handle=handle,
    )
    await store.store_cert(
        network_id=network_id, handle=handle,
        cert_wire=cert_wire, expires_at=cert.expires_at,
    )
    elog("peers.added", network=network_name, handle=handle)
    return web.json_response({
        "ok": True, "network_id": network_id, "name": network_name,
        "handle": handle, "expires_at": cert.expires_at,
    })


async def handle_delete(request: web.Request) -> web.Response:
    """DELETE /api/peers/{network_id} — drop a peer-network membership."""
    gw = request.app["gateway"]
    db = gw.agent._db
    if db is None:
        return web.json_response({"error": "no DB attached"}, status=503)
    network_id = request.match_info["network_id"]
    store = PeerStore(db)
    ok = await store.remove_peer(network_id)
    return web.json_response({"ok": ok})


async def handle_list_agents(request: web.Request) -> web.Response:
    """GET /api/peers/{network_id}/agents — list agents in a peer network."""
    gw = request.app["gateway"]
    state = getattr(gw, "_network_state", None)
    db = gw.agent._db
    if db is None or state is None:
        return web.json_response({"error": "gateway misconfigured"}, status=503)
    store = PeerStore(db)
    network_id = request.match_info["network_id"]
    peer = await store.get_peer(network_id)
    if peer is None:
        return web.json_response({"error": "unknown peer"}, status=404)
    agents = await coord_list_agents(
        node=state.iroh_node,
        coordinator_node_id=peer.coordinator_node_id,
    )
    return web.json_response({"agents": agents})


async def handle_join(request: web.Request) -> web.Response:
    """POST /api/peers/join — join a peer network using an agent invite ticket.

    Body: ``{ticket: "oa1…", handle?: "my-handle", label?: "friendly name"}``.

    This is the password-free path for agent-to-agent federation.  The local
    agent proves its identity via the Iroh QUIC handshake (``peer_node_id``
    in the coordinator RPC equals our actual node_id — no SRP/password needed).

    On success the peer is persisted in ``peer_networks`` with
    ``join_type='agent'`` and the cert is cached in ``device_certs``.
    """
    from src.network.client.login import agent_login, fetch_network_info
    from src.network.ticket import InviteTicket, TicketError

    gw = request.app["gateway"]
    db = gw.agent._db
    state = getattr(gw, "_network_state", None)
    if db is None or state is None:
        return web.json_response({"error": "gateway misconfigured"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    raw_ticket = (body.get("ticket") or "").strip()
    if not raw_ticket:
        return web.json_response({"error": "ticket is required"}, status=400)

    try:
        ticket = InviteTicket.decode(raw_ticket)
    except TicketError as e:
        return web.json_response({"error": f"invalid ticket: {e}"}, status=400)

    if ticket.role != "agent":
        return web.json_response(
            {"error": f"ticket role is {ticket.role!r}; /api/peers/join requires role=agent"},
            status=400,
        )

    # Determine the handle to use in the remote network.
    # Priority: body override → config agent name → first 8 chars of node_id.
    handle = (body.get("handle") or "").strip().lower()
    if not handle:
        handle = (gw.agent.name or "agent").lower()
    label = body.get("label") or f"{gw.agent.name or 'agent'} (federated)"

    coord_node_id = ticket.coordinator_node_id
    coord_pubkey_bytes = bytes(coordinator_node_id_to_pubkey_bytes(coord_node_id))

    try:
        node_id = await state.iroh_node.node_id()
    except Exception as e:
        return web.json_response({"error": f"could not get local node_id: {e}"}, status=500)

    try:
        cert_wire = await agent_login(
            node=state.iroh_node,
            coordinator_node_id=coord_node_id,
            coordinator_pubkey_bytes=coord_pubkey_bytes,
            handle=handle,
            node_id=node_id,
            invite_code=ticket.code,
            network_id=ticket.network_id,
            label=label,
        )
    except LoginError as e:
        elog("peers.join_failed", level="warning", error=str(e), network=ticket.network_name)
        return web.json_response({"error": str(e)}, status=400)

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pubkey = Ed25519PublicKey.from_public_bytes(coord_pubkey_bytes)
        cert = verify_cert(cert_wire, coordinator_pubkey=pubkey,
                           expected_network_id=ticket.network_id)
    except CertVerificationError as e:
        return web.json_response({"error": f"cert verification failed: {e}"}, status=502)

    store = PeerStore(db)
    await store.add_peer(
        network_id=ticket.network_id,
        name=ticket.network_name,
        coordinator_node_id=coord_node_id,
        coordinator_pubkey=coord_pubkey_bytes,
        our_handle=handle,
        join_type="agent",
    )
    await store.store_cert(
        network_id=ticket.network_id,
        handle=handle,
        cert_wire=cert_wire,
        expires_at=cert.expires_at,
    )
    elog("peers.joined", network=ticket.network_name, handle=handle, join_type="agent")
    return web.json_response({
        "ok": True,
        "network_id": ticket.network_id,
        "name": ticket.network_name,
        "handle": handle,
        "expires_at": cert.expires_at,
        "join_type": "agent",
    })


async def handle_peer_chat(request: web.Request) -> web.Response:
    """POST /api/peers/{network_id}/chat — send a message to a peer agent.

    Body: ``{message: "…", session_id?: "…", agent_handle?: "…"}``.

    The server acts as a relay: it opens an authenticated Iroh connection to
    the peer agent's gateway using the cached cert (refreshing if needed via
    agent_login), POSTs to the peer's ``/api/chat``, and returns the reply.

    ``agent_handle`` lets the caller target a specific agent in the peer
    network (default: the first non-self agent in ``list_agents``).
    """
    import aiohttp as _aiohttp

    gw = request.app["gateway"]
    db = gw.agent._db
    state = getattr(gw, "_network_state", None)
    if db is None or state is None:
        return web.json_response({"error": "gateway misconfigured"}, status=503)

    network_id = request.match_info["network_id"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    message = (body.get("message") or "").strip()
    if not message:
        return web.json_response({"error": "message is required"}, status=400)

    session_id = (body.get("session_id") or "peer-default").strip()
    target_agent_handle = body.get("agent_handle")

    store = PeerStore(db)
    peer = await store.get_peer(network_id)
    if peer is None:
        return web.json_response({"error": "unknown peer network"}, status=404)

    # Discover the target agent's node_id.
    try:
        agents = await coord_list_agents(
            node=state.iroh_node,
            coordinator_node_id=peer.coordinator_node_id,
        )
    except Exception as e:
        return web.json_response({"error": f"could not list peer agents: {e}"}, status=502)

    our_node_id: str | None = None
    try:
        our_node_id = await state.iroh_node.node_id()
    except Exception:
        pass

    # Pick target: explicit handle > first agent that isn't us.
    target_agent = None
    if target_agent_handle:
        target_agent = next((a for a in agents if a.get("handle") == target_agent_handle), None)
    if target_agent is None:
        target_agent = next(
            (a for a in agents if a.get("node_id") != our_node_id),
            agents[0] if agents else None,
        )
    if target_agent is None:
        return web.json_response({"error": "no agents found in peer network"}, status=404)

    target_node_id: str = target_agent["node_id"]

    # Build the loopback proxy — strategy depends on how we joined.
    # Agent-type peers use the AGENT ALPN (no cert needed; Iroh QUIC
    # handshake proves node_id ownership). User-type peers use the
    # GATEWAY ALPN with a coordinator-signed cert.
    from src.network.client.session import AgentDialer, LoopbackProxy

    if peer.join_type == "agent":
        agent_dialer = AgentDialer(node=state.iroh_node, target_node_id=target_node_id)
        proxy = LoopbackProxy(stream_factory=agent_dialer.open_agent_stream)
    else:
        try:
            dialer = await make_dialer_for_peer(db=db, peer=peer, node=state.iroh_node)
        except LoginError as e:
            return web.json_response(
                {"error": f"could not authenticate to peer: {e}"}, status=401,
            )
        proxy = LoopbackProxy(dialer=dialer, target_node_id=target_node_id)

    try:
        await proxy.start()
        async with _aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{proxy.base_url}/api/chat",
                    json={"message": message, "session_id": session_id},
                    timeout=_aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.content_type == "application/json":
                        data = await resp.json()
                    else:
                        text = await resp.text()
                        data = {"error": text, "status": resp.status}
            except _aiohttp.ClientError as e:
                return web.json_response(
                    {"error": f"network error contacting peer: {e}"}, status=502,
                )
    finally:
        await proxy.stop()

    # Surface any error from the remote agent.
    if "error" in data and resp.status >= 400:
        return web.json_response({"error": data.get("error"), "peer": peer.name}, status=502)

    return web.json_response({
        "response": data.get("response", ""),
        "model": data.get("model", ""),
        "errored": data.get("errored", False),
        "peer_name": peer.name,
        "peer_handle": peer.our_handle,
        "agent_handle": target_agent.get("handle"),
    })


async def handle_refresh(request: web.Request) -> web.Response:
    """POST /api/peers/{network_id}/refresh — force a cert refresh for a peer."""
    gw = request.app["gateway"]
    db = gw.agent._db
    state = getattr(gw, "_network_state", None)
    if db is None or state is None:
        return web.json_response({"error": "gateway misconfigured"}, status=503)

    network_id = request.match_info["network_id"]
    store = PeerStore(db)
    peer = await store.get_peer(network_id)
    if peer is None:
        return web.json_response({"error": "unknown peer"}, status=404)

    # Agent-type peers use the AGENT ALPN and don't need a coordinator cert.
    if peer.join_type == "agent":
        return web.json_response({
            "ok": True,
            "message": "agent peers authenticate via Iroh QUIC handshake — no cert required",
            "join_type": "agent",
        })

    # Force refresh by clearing the cached cert.
    await store._conn.execute(
        "DELETE FROM device_certs WHERE network_id=? AND handle=?",
        (peer.network_id, peer.our_handle),
    )
    await store._conn.commit()

    try:
        dialer = await make_dialer_for_peer(db=db, peer=peer, node=state.iroh_node)
    except LoginError as e:
        return web.json_response({"error": f"refresh failed: {e}"}, status=400)

    cert = dialer.parsed_cert()
    return web.json_response({"ok": True, "expires_at": cert.expires_at})


# ── Helpers ─────────────────────────────────────────────────────────────


def coordinator_node_id_to_pubkey_bytes(node_id: str) -> bytes:
    """Decode an Iroh NodeId string into raw 32 bytes.

    Iroh NodeIds are an encoded form of the Ed25519 public key bytes.
    iroh-py 0.35 exposes ``PublicKey.from_string(s).as_bytes()``; we
    defer the import so this module doesn't pull iroh into agents that
    never join a peer network.
    """
    import iroh  # noqa: WPS433

    pk = iroh.PublicKey.from_string(node_id)
    raw = pk.to_bytes()
    if len(raw) != 32:
        raise ValueError(f"NodeId pubkey is not 32 bytes: {len(raw)}")
    return bytes(raw)
