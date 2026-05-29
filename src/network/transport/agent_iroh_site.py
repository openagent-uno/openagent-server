"""AgentSite — IrohNode handler for the ``openagent/agent/1`` ALPN.

Unlike IrohSite (which requires a coordinator-signed device cert),
AgentSite authenticates by Iroh's QUIC handshake alone: the caller's
node_id is cryptographically bound to the TLS certificate exchanged
during the QUIC handshake, so a successful connection proves node_id
ownership without any coordinator involvement.

Wire format: raw HTTP/1.1 — no cert prefix. The peer_node_id is placed
in ``_current_peer_node_id`` and ``_is_authenticated_agent`` is set to
``True`` before the aiohttp RequestHandler is created, so the auth
middleware (which runs inside the request handler task) sees the agent
context and synthesises a DeviceCert rather than looking for cert wire.

This ALPN is used for agent-to-agent federation: two OpenAgent instances
can exchange messages without the remote having issued us a coordinator
cert. The remote just needs to trust that the QUIC handshake is proof of
identity (Iroh enforces this at the protocol level).
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web
from aiohttp.web_protocol import RequestHandler

from src.network.iroh_node import IrohNode, NetworkAlpn
from src.network.transport.aiohttp_iroh_site import (
    _current_peer_node_id,
    _is_authenticated_agent,
    _ProtocolTransport,
)
from src.network.transport.asyncio_bridge import IrohStreamWriter

logger = logging.getLogger(__name__)


class AgentSite(web.BaseSite):
    """BaseSite variant for the ``openagent/agent/1`` ALPN.

    Plugs into the same ``web.AppRunner`` as ``IrohSite``. Like
    ``IrohSite``, the ALPN handler must be registered on the IrohNode
    BEFORE ``IrohNode.start()`` — the Gateway does this in
    ``_prepare_iroh_site()``, which runs before the node binds.

    Inbound connections on this ALPN skip the cert-prefix framing and
    authenticate by Iroh node_id alone. Every route registered on the
    shared aiohttp app is reachable, subject to the auth middleware
    synthesising a valid DeviceCert for the peer.
    """

    __slots__ = ("_node", "_started")

    def __init__(self, runner: web.AppRunner, node: IrohNode) -> None:
        super().__init__(runner, shutdown_timeout=60.0)
        self._node = node
        self._started = False

    @property
    def name(self) -> str:
        return f"iroh-agent://{self._node.identity.public_hex[:8]}"

    async def start(self) -> None:
        if self._started:
            return
        await super().start()
        self._started = True

    async def _handle_stream(self, connection: object) -> None:
        """Handle one accepted Iroh connection on the AGENT ALPN.

        Drains bi-streams off the connection and spawns per-stream tasks,
        exactly like ``IrohSite._handle_stream`` does for GATEWAY streams.
        """
        peer_node_id = "unknown"
        try:
            raw = connection.remote_node_id()
            peer_node_id = str(raw) if raw is not None else "unknown"
        except Exception:
            pass

        logger.debug("iroh-agent: inbound connection from %s", peer_node_id[:16])
        try:
            while True:
                try:
                    bi = await connection.accept_bi()
                except Exception as e:  # noqa: BLE001
                    logger.debug("iroh-agent: connection ended from %s: %s", peer_node_id[:16], e)
                    return
                if bi is None:
                    return
                send_stream = bi.send()
                recv_stream = bi.recv()
                asyncio.create_task(
                    self._handle_one_stream(peer_node_id, send_stream, recv_stream),
                    name=f"iroh-agent-stream-{peer_node_id[:8]}",
                )
        except asyncio.CancelledError:
            raise

    async def _handle_one_stream(
        self,
        peer_node_id: str,
        send_stream: object,
        recv_stream: object,
    ) -> None:
        """Handle one bi-stream: set auth context → HTTP dispatch → response.

        No cert-prefix is read — the stream starts immediately with HTTP/1.1
        bytes. The auth middleware detects ``_is_authenticated_agent=True``
        and synthesises a DeviceCert for the peer node.
        """
        loop = asyncio.get_event_loop()
        writer = IrohStreamWriter(
            send_stream,
            peer_node_id=peer_node_id,
            alpn=NetworkAlpn.AGENT,
        )

        # Set contextvars BEFORE ``connection_made`` — the request handler
        # task captures them at creation time (same constraint as IrohSite).
        node_token = _current_peer_node_id.set(peer_node_id)
        agent_token = _is_authenticated_agent.set(True)

        protocol = RequestHandler(
            self._runner.server,
            loop=loop,
            access_log=None,
        )
        protocol.connection_made(_ProtocolTransport(writer))

        async def _pump_to_protocol() -> None:
            try:
                while True:
                    chunk = await recv_stream.read(64 * 1024)
                    if not chunk:
                        break
                    protocol.data_received(chunk)
            except Exception as e:  # noqa: BLE001
                logger.debug("iroh-agent: read pump ended: %s", e)
            finally:
                try:
                    protocol.connection_lost(None)
                except Exception:
                    pass

        pump_task = asyncio.create_task(
            _pump_to_protocol(),
            name=f"iroh-agent-pump-{peer_node_id[:8]}",
        )

        try:
            task_handler = getattr(protocol, "_task_handler", None)
            try:
                if task_handler is None:
                    while not writer.is_closing():
                        await asyncio.sleep(0.05)
                else:
                    await task_handler
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug("aiohttp task_handler ended (agent stream): %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("aiohttp protocol crashed on agent stream: %s", e)
        finally:
            _current_peer_node_id.reset(node_token)
            _is_authenticated_agent.reset(agent_token)
            if not pump_task.done():
                pump_task.cancel()
                try:
                    await pump_task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                writer.close()
            except Exception:
                pass
