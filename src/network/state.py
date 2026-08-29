"""``NetworkState`` — the bag of state the gateway needs to come up.

The AgentServer builds one of these from config + DB and hands it to
the Gateway. The Gateway plugs the IrohNode into ``IrohSite`` and the
NetworkAuthState into the auth middleware. Coordinator-mode agents
also get a ``CoordinatorService`` attached.

Standalone agents (no network configured) raise ``StandaloneAgentError``
when ``NetworkState.from_db`` is called — the AgentServer surfaces a
friendly "run `openagent network init`" message and skips the gateway.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from src.memory.db import MemoryDB
from src.network.auth.middleware import NetworkAuthState
from src.network.coordinator import CoordinatorService, CoordinatorStore
from src.network.identity import Identity, load_or_create_identity
from src.network.iroh_node import IrohNode

logger = logging.getLogger(__name__)


class StandaloneAgentError(Exception):
    """Raised when an agent has no network configured.

    Caller is expected to surface a "run ``openagent network init``"
    message and skip the gateway. The agent itself can still run
    headless (scheduler, dream mode, etc.) — only the public
    interface is gated.
    """


@dataclass
class NetworkState:
    """All the network-layer objects a Gateway needs at construction."""

    role: str  # standalone | coordinator | member
    network_id: str
    network_name: str
    identity: Identity
    iroh_node: IrohNode
    auth_state: NetworkAuthState
    coordinator_service: CoordinatorService | None = None
    coordinator_key: Ed25519PrivateKey | None = None  # only set on coordinator-mode
    _device_revocation_task: asyncio.Task | None = field(
        default=None, init=False, repr=False,
    )

    @classmethod
    async def from_db(
        cls,
        *,
        db: MemoryDB,
        identity_path: Path | str,
        derp_url: str | None = None,
    ) -> "NetworkState":
        """Build a ``NetworkState`` by reading the singleton ``network`` row.

        - ``identity_path`` is where this agent's Iroh secret key lives —
          in coordinator mode it doubles as the coordinator's signing
          key. (Two keys would let agent NodeId rotate independently
          from coordinator pubkey, but no rotation logic is implemented;
          the second key only created mismatches between the running
          iroh node and what tickets advertised.)
        - ``derp_url`` overrides the default Iroh DERP relay; pass None
          to use Iroh's public network.
        """
        store = CoordinatorStore(db)
        row = await store.get_network_role()
        if row is None or row["role"] == "standalone":
            raise StandaloneAgentError(
                "agent is not part of a network — run `openagent network init`",
            )

        identity = load_or_create_identity(Path(identity_path))
        node = IrohNode(identity, derp_url=derp_url)

        coordinator_service: CoordinatorService | None = None
        coordinator_key: Ed25519PrivateKey | None = None

        if row["role"] == "coordinator":
            coordinator_key = Ed25519PrivateKey.from_private_bytes(
                identity.secret_bytes,
            )
            coord_pubkey_bytes = identity.public_bytes
        else:
            stored = row["coordinator_pubkey"]
            if not stored:
                raise ValueError(
                    "member-mode agent missing coordinator_pubkey — "
                    "the network row is corrupted",
                )
            coord_pubkey_bytes = bytes(stored)

        coordinator_pubkey = Ed25519PublicKey.from_public_bytes(coord_pubkey_bytes)

        # Coordinator gateways read the local roster. Member gateways use a
        # peer-authenticated coordinator RPC for the same per-stream check.
        revoked = await store.list_revoked_pubkeys() if row["role"] == "coordinator" else set()
        member_device_lookup = None
        if row["role"] == "member":
            coordinator_node_id = (row.get("coordinator_node_id") or "").strip()
            if not coordinator_node_id:
                raise ValueError(
                    "member-mode agent missing coordinator_node_id — "
                    "live device revocation cannot be enforced",
                )

            async def member_device_lookup(device_pubkey: bytes):
                from types import SimpleNamespace

                from src.network.client.login import device_is_active

                active = await device_is_active(
                    node=node,
                    coordinator_node_id=coordinator_node_id,
                    device_pubkey=device_pubkey,
                )
                return SimpleNamespace(status="active") if active else None

        auth_state = NetworkAuthState(
            coordinator_pubkey=coordinator_pubkey,
            network_id=row["network_id"],
            revoked_pubkeys=revoked,
            device_lookup=(
                store.get_active_device
                if row["role"] == "coordinator"
                else member_device_lookup
            ),
        )

        if coordinator_key is not None:
            coordinator_service = CoordinatorService(
                store=store,
                coordinator_key=coordinator_key,
                network_id=row["network_id"],
                network_name=row["name"],
                on_device_revoked=auth_state.revoke,
            )
            coordinator_service.attach(node)

        return cls(
            role=row["role"],
            network_id=row["network_id"],
            network_name=row["name"],
            identity=identity,
            iroh_node=node,
            auth_state=auth_state,
            coordinator_service=coordinator_service,
            coordinator_key=coordinator_key,
        )

    async def start(self) -> None:
        """Start the underlying Iroh endpoint and the coordinator GC."""
        await self.iroh_node.start()
        if self.coordinator_service is not None:
            await self.coordinator_service.start_gc()
        # Revalidate open transports in both roles. Member agents learn live
        # status over coordinator RPC; the coordinator catches out-of-process
        # DB mutations such as ``openagent network revoke`` which cannot call
        # the in-process listener directly.
        if self._device_revocation_task is None:
            self._device_revocation_task = asyncio.create_task(
                self._device_revocation_loop(),
                name="device-revocation-watcher",
            )

    async def stop(self) -> None:
        if self._device_revocation_task is not None:
            self._device_revocation_task.cancel()
            try:
                await self._device_revocation_task
            except asyncio.CancelledError:
                pass
            self._device_revocation_task = None
        if self.coordinator_service is not None:
            await self.coordinator_service.stop()
        await self.iroh_node.stop()

    async def _device_revocation_loop(self) -> None:
        """Poll the authoritative roster and close revoked live streams."""
        while True:
            try:
                await asyncio.sleep(1.0)
                await self.auth_state.revalidate_observed_devices()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the security loop live
                logger.warning("device revocation poll failed: %s", exc)

    async def node_id(self) -> str:
        return await self.iroh_node.node_id()
