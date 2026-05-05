"""In-process bridge → gateway session over a synthetic iroh stream.

Bridges (telegram, discord, whatsapp) run in the same process as the
agent gateway but are clients of the gateway over the wire. The iroh
transport replaces the legacy ``ws://localhost:8765/ws + token`` path,
so we mint a coordinator-signed device cert for the synthetic
``__bridge`` user and feed bytes through an in-process pipe that
mimics an iroh ``Connection``.

Why in-process and not real iroh self-dial: iroh-py 0.35 doesn't
support a node connecting to its own NodeId (the local QUIC stack
refuses), so the bridge can't reach the gateway through the real
transport even though they're co-located. ``transport.inproc`` provides
``InProcConnection`` + ``InProcDialer`` that look like iroh objects to
the IrohSite handler — the auth / framing path stays identical, only
the byte pump is swapped. See ``openagent/network/transport/inproc.py``.

Only coordinator-mode agents can run bridges this way: minting a cert
requires the coordinator's signing key, which only the coordinator has.
Member-mode agents would need a coordinator-issued bridge cert at join
time — that flow is not implemented yet, so member-mode raises
``BridgeSessionUnavailable`` and the gateway brings up without bridges.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openagent.network.auth.device_cert import issue_cert
from openagent.network.client.session import LoopbackProxy
from openagent.network.state import NetworkState
from openagent.network.transport.inproc import InProcConnection, InProcDialer

logger = logging.getLogger(__name__)


BRIDGE_HANDLE = "__bridge"
BRIDGE_DEVICE_KEY_FILENAME = ".bridge-device.key"


class BridgeSessionUnavailable(Exception):
    """Raised when bridges cannot be wired (member-mode, missing key, …)."""


def _load_or_create_bridge_device_key(agent_dir: Path) -> Ed25519PrivateKey:
    p = agent_dir / BRIDGE_DEVICE_KEY_FILENAME
    if p.exists():
        return Ed25519PrivateKey.from_private_bytes(p.read_bytes())
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(raw)
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    return key


class BridgeSession:
    """In-process bridge plumbing: cert + InProcConnection + LoopbackProxy.

    Lifecycle: ``await session.start(network_state, gateway_site, agent_dir)`` →
    ``session.ws_url`` → ``await session.stop()``.
    """

    def __init__(self) -> None:
        self._connection: InProcConnection | None = None
        self._dialer: InProcDialer | None = None
        self._proxy: LoopbackProxy | None = None
        self._site_handler_task: asyncio.Task | None = None
        self._cert_wire: bytes | None = None

    @property
    def ws_url(self) -> str:
        if self._proxy is None:
            raise RuntimeError("BridgeSession.start() not awaited")
        return self._proxy.ws_url

    @property
    def base_url(self) -> str:
        if self._proxy is None:
            raise RuntimeError("BridgeSession.start() not awaited")
        return self._proxy.base_url

    @property
    def cert_wire(self) -> bytes:
        if self._cert_wire is None:
            raise RuntimeError("BridgeSession.start() not awaited")
        return self._cert_wire

    async def start(
        self,
        *,
        network_state: NetworkState,
        gateway_site,  # openagent.network.transport.aiohttp_iroh_site.IrohSite
        agent_dir: Path,
    ) -> None:
        if network_state.role != "coordinator":
            raise BridgeSessionUnavailable(
                "in-process bridges require coordinator-mode (member-mode "
                "agents need a coordinator-issued bridge cert; not "
                "implemented yet)",
            )
        if network_state.coordinator_key is None:
            raise BridgeSessionUnavailable(
                "coordinator key missing on the network state — cannot "
                "mint a bridge cert",
            )
        if gateway_site is None:
            raise BridgeSessionUnavailable(
                "gateway site is not initialized — bridges run after the "
                "gateway, not before",
            )

        device_key = _load_or_create_bridge_device_key(Path(agent_dir))
        device_pubkey = device_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        # Mint a fresh cert at every start. The TTL (a week or so) is
        # plenty for a single agent uptime; we don't have an in-process
        # refresh path because bridges reconnect from scratch on every
        # process restart anyway.
        cert_wire = issue_cert(
            coordinator_key=network_state.coordinator_key,
            handle=BRIDGE_HANDLE,
            device_pubkey=device_pubkey,
            network_id=network_state.network_id,
            capabilities=["bridge"],
        )

        # Synthetic connection: the gateway's IrohSite handler reads
        # bi-streams from the connection just like it would for a
        # remote iroh peer. The cert prefix is written by InProcDialer
        # exactly the same way a real SessionDialer would.
        self._cert_wire = cert_wire
        self._connection = InProcConnection(peer_node_id="inproc:bridge")
        self._dialer = InProcDialer(
            connection=self._connection,
            cert_wire=cert_wire,
        )

        # Hand the connection to IrohSite. ``_handle_stream`` loops on
        # ``accept_bi`` and spawns a task per stream — same code path
        # as a remote inbound connection, with a different byte source.
        self._site_handler_task = asyncio.create_task(
            gateway_site._handle_stream(self._connection),
            name="bridge-inproc-site-handler",
        )

        self._proxy = LoopbackProxy(
            dialer=self._dialer,
            target_node_id="inproc",
        )
        await self._proxy.start()
        logger.info(
            "bridge session ready: ws=%s handle=%s capability=bridge",
            self._proxy.ws_url, BRIDGE_HANDLE,
        )

    async def stop(self) -> None:
        # Order matters: ``LoopbackProxy.stop`` awaits ``server.wait_closed``
        # which blocks on active TCP connections, and those connections'
        # byte-pumps are waiting for EOF on the InProc bi-stream queues.
        # Closing the connection first drains the queues so the pumps
        # (and therefore wait_closed) can complete.
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception as e:
                logger.debug("bridge_session: connection close error: %s", e)
            self._connection = None
        if self._proxy is not None:
            try:
                await self._proxy.stop()
            except Exception as e:
                logger.debug("bridge_session: proxy stop error: %s", e)
            self._proxy = None
        if self._site_handler_task is not None:
            self._site_handler_task.cancel()
            try:
                await self._site_handler_task
            except (asyncio.CancelledError, Exception):
                pass
            self._site_handler_task = None
        self._dialer = None
        self._cert_wire = None
