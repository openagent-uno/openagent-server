"""PeerNetworksBackend — the embedded backend for agent-federation.

Implements ``openagent_mcp.backends.base.Backend`` by reading this agent's
``peer_networks`` rows and BORROWING the server's already-running
``IrohNode`` (no second node, no cert, no ``/api/peers`` relay). Adding a peer
= joining its Iroh network; nothing else to configure.
"""

from __future__ import annotations

import logging

from oa_agent_client import AgentClient
from openagent_mcp.backends.base import Target

from src.network.peers import PeerStore

from . import handlers

logger = logging.getLogger(__name__)


class PeerNetworksBackend:
    # Generous: long peer jobs run to completion synchronously; anything
    # longer persists to the (first-class) session and is resumable. Must
    # match the server's OPENAGENT_CHAT_TURN_TIMEOUT default so the caller
    # doesn't give up while the target is still working.
    default_timeout = 900.0

    async def list_targets(self) -> list[Target]:
        store = PeerStore(handlers.get_db())
        targets: list[Target] = []
        for p in await store.list_peers():
            # Only agent-ALPN (key-based) peers are reachable this way; a
            # user/SRP peer would need the coordinator-signed GATEWAY cert
            # path, which this backend doesn't implement.
            if p.join_type != "agent":
                logger.debug("agent-federation: skipping non-agent peer %r", p.name)
                continue
            if p.status and p.status != "active":
                continue
            targets.append(Target(
                name=p.name,
                # These are single-agent personal networks, so the peer's
                # coordinator node IS its agent node — dial it directly on the
                # AGENT ALPN (proven). Multi-agent networks would instead need
                # coord list_agents discovery.
                node_id=p.coordinator_node_id,
                description=f"Federated OpenAgent peer '{p.name}'.",
            ))
        return targets

    async def dial(self, target: Target, message: str, session_id, timeout) -> dict:
        # Borrow the server's running node (no ownership, no start()).
        client = AgentClient(node=handlers.get_iroh_node())
        return await client.chat(
            node_id=target.node_id,
            message=message,
            session_id=session_id,
            timeout=timeout,
        )
