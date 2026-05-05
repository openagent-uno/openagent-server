"""Singleton wrapper around an ``iroh`` node + custom-ALPN protocols.

This is the only module that imports ``iroh`` directly so all version-skew
and FFI quirks land in one place. Higher layers (transport, coordinator,
client) talk to ``IrohNode`` via this small surface:

  await node.start()                         # build node, wire handlers
  await node.dial(node_id, alpn) -> Conn     # outbound connection
  node.register_handler(alpn, handler)       # before start: per-ALPN cb
  await node.stop()

iroh-py 0.35 surface notes (the FFI is *not* stable across minors):
- ``iroh.Iroh.memory_with_options(NodeOptions(...))`` builds a node.
- ``NodeOptions.protocols`` is a ``dict[bytes, ProtocolCreator]`` —
  the ProtocolCreator's ``create(endpoint)`` returns a
  ``ProtocolHandler`` whose ``accept(connection)`` runs every time a
  peer connects with that ALPN.
- ``Endpoint`` only exposes ``connect(node_addr, alpn) -> Connection``
  and ``node_id() -> str``. There is no manual accept loop — Iroh runs
  one internally and dispatches to our handler.

A pinned IROH_VERSION constant tracks what we tested against. Bump it
intentionally when the FFI changes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import iroh

from openagent.network.identity import Identity

logger = logging.getLogger(__name__)

IROH_VERSION = "0.35"


class NetworkAlpn:
    """ALPN protocol identifiers used on the OpenAgent overlay."""

    GATEWAY = b"openagent/gateway/1"
    COORDINATOR = b"openagent/coordinator/1"


# Per-connection handler signature. The transport layer wraps the
# ``Connection`` to drain its bi-streams; the coordinator handler does
# the same for its JSON-RPC dispatch. We accept the connection object
# typed as ``object`` here so this module stays the only one that
# imports ``iroh`` types directly.
StreamHandler = Callable[[object], Awaitable[None]]


class _PythonProtocolHandler(iroh.ProtocolHandler):
    """Adapter turning an ``async (connection)`` callable into iroh's handler."""

    def __init__(self, name: str, handler: StreamHandler) -> None:
        super().__init__()
        self._name = name
        self._handler = handler
        self._tasks: set[asyncio.Task] = set()
        self._stopped = False

    async def accept(self, connection: "iroh.Connection") -> None:
        # iroh calls this in its own event loop. We just hand off to the
        # registered handler — but track the task so ``shutdown`` can
        # tear pending in-flight handlers down. ``Connection.close`` is
        # sync in iroh-py 0.35 — no await.
        if self._stopped:
            try:
                connection.close(0, b"shutting down")
            except Exception:
                pass
            return
        task = asyncio.create_task(
            self._handler(connection),
            name=f"iroh-{self._name}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def shutdown(self) -> None:
        self._stopped = True
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()


class _PythonProtocolCreator(iroh.ProtocolCreator):
    """``ProtocolCreator.create(endpoint)`` factory for ``_PythonProtocolHandler``."""

    def __init__(self, name: str, handler: StreamHandler) -> None:
        super().__init__()
        self._name = name
        self._handler = handler

    def create(self, endpoint: "iroh.Endpoint") -> iroh.ProtocolHandler:
        # iroh's UniFFI binding calls this synchronously and expects a
        # ProtocolHandler back, not a coroutine — declaring async here
        # silently returned a coroutine that Rust then tried to invoke
        # ``.accept`` on, with predictable results.
        return _PythonProtocolHandler(self._name, self._handler)


class IrohNode:
    """Holds the live iroh node + the per-ALPN dispatch table.

    One ``IrohNode`` per process. Agent processes create theirs from
    the agent identity; CLI/app processes from the user-device identity.
    The same node is used for inbound (gateway) and outbound (peer
    dials) — Iroh multiplexes both over one endpoint.
    """

    def __init__(self, identity: Identity, *, derp_url: str | None = None) -> None:
        self.identity = identity
        self.derp_url = derp_url  # currently advisory; iroh-py 0.35's NodeOptions doesn't expose relay overrides
        self._handlers: dict[bytes, StreamHandler] = {}
        self._node: "iroh.Iroh | None" = None
        self._endpoint: "iroh.Endpoint | None" = None
        self._cached_node_id: str | None = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._node is not None:
            return

        # Register the running asyncio loop with iroh's UniFFI
        # bindings. iroh's Rust side spawns its own threads to deliver
        # incoming connections to ``ProtocolHandler.accept``; without
        # this, the callback fires on a thread with no Python loop and
        # fails with ``RuntimeError: no running event loop``. The
        # binding caches a single global loop, so calling start() from
        # a different loop later would route callbacks to the wrong
        # one — but we only ever start the IrohNode once per process.
        # ``uniffi_set_event_loop`` is exposed on the FFI submodule, not
        # the package root.
        from iroh.iroh_ffi import uniffi_set_event_loop
        uniffi_set_event_loop(asyncio.get_running_loop())

        # Wrap each registered handler in a ProtocolCreator. iroh's
        # NodeOptions.protocols is the canonical way to register a
        # custom ALPN — there is no post-construction "add_protocol".
        creators: dict[bytes, iroh.ProtocolCreator] = {
            alpn: _PythonProtocolCreator(alpn.decode(errors="replace"), handler)
            for alpn, handler in self._handlers.items()
        }
        opts = iroh.NodeOptions(
            secret_key=self.identity.secret_bytes,
            protocols=creators or None,
            node_discovery=iroh.NodeDiscoveryConfig.DEFAULT,
            enable_docs=False,
        )
        self._node = await iroh.Iroh.memory_with_options(opts)
        # ``node().endpoint()`` and ``endpoint.node_id()`` are SYNC in
        # iroh-py 0.35 — they return the value directly. ``Iroh.memory``
        # and ``node.shutdown`` are async; the others on the node + endpoint
        # path are not.
        self._endpoint = self._node.node().endpoint()
        self._cached_node_id = self._endpoint.node_id()
        logger.info("iroh node started: node_id=%s", self._cached_node_id)

    async def stop(self) -> None:
        if self._node is None:
            return
        try:
            # ``Node.shutdown`` is async but takes no kwargs in 0.35.
            await self._node.node().shutdown()
        except Exception as e:  # noqa: BLE001
            logger.debug("iroh shutdown failed: %s", e)
        self._node = None
        self._endpoint = None

    # ── Identity ─────────────────────────────────────────────────────

    async def node_id(self) -> str:
        """Return the public NodeId string for this endpoint.

        Pre-start: derive synchronously from the secret key so callers
        like ``openagent network init`` (which print the NodeId without
        starting a runtime) work.
        """
        if self._cached_node_id:
            return self._cached_node_id
        if self._endpoint is not None:
            self._cached_node_id = self._endpoint.node_id()
            return self._cached_node_id
        # Pre-start derivation: PublicKey is FFI-cheap and deterministic.
        # This bypasses the node entirely.
        return _node_id_from_secret(self.identity.secret_bytes)

    # ── Handlers ─────────────────────────────────────────────────────

    def register_handler(self, alpn: bytes, handler: StreamHandler) -> None:
        """Register the inbound handler for *alpn*. Must be called before ``start``."""
        if self._node is not None:
            raise RuntimeError(
                "register_handler called after start — re-create IrohNode if you "
                "need a different ALPN set",
            )
        self._handlers[alpn] = handler

    # ── Outbound ─────────────────────────────────────────────────────

    async def dial(self, node_id: str, alpn: bytes) -> "iroh.Connection":
        """Open an outbound connection to *node_id* using *alpn*.

        ``node_id`` is the canonical hex (or base32) string a peer
        printed; iroh-py 0.35's ``NodeAddr`` constructor wants a
        ``PublicKey`` object, not a string, so we wrap it first.

        Iroh wraps every transport failure in an opaque ``IrohError``
        whose ``str()`` is empty — so a "coordinator not running" looks
        identical to a "coordinator on the other side of a black-hole
        firewall". We translate it into a ``DialError`` carrying the
        target NodeId + ALPN so callers can surface something the user
        can actually act on.
        """
        if self._endpoint is None:
            raise RuntimeError("IrohNode.dial() called before start")
        pubkey = iroh.PublicKey.from_string(node_id)
        addr = iroh.NodeAddr(pubkey, None, [])
        try:
            return await self._endpoint.connect(addr, alpn)
        except Exception as e:
            raise DialError(node_id=node_id, alpn=alpn, cause=e) from e


class DialError(RuntimeError):
    """Wraps an iroh connect failure with the target + ALPN it was for."""

    def __init__(self, *, node_id: str, alpn: bytes, cause: BaseException):
        self.node_id = node_id
        self.alpn = alpn
        self.cause = cause
        try:
            alpn_label = alpn.decode("ascii")
        except UnicodeDecodeError:
            alpn_label = alpn.hex()
        cause_str = str(cause).strip() or type(cause).__name__
        super().__init__(
            f"could not reach {node_id[:16]}… on alpn {alpn_label!r}: {cause_str} "
            f"(is the OpenAgent server running on the host that minted this ticket? "
            f"check that ``openagent serve <agent-dir>`` is up and that the host has "
            f"network connectivity to iroh's relay network)"
        )


def _node_id_from_secret(secret_bytes: bytes) -> str:
    """Derive the NodeId string from a 32-byte Ed25519 secret without starting a node.

    Done via the iroh PublicKey class so the encoding matches what a
    running node would print. This is the same path ``openagent network
    init`` uses to dump identifiers into the on-disk ``network.toml``
    before any node has bound.
    """
    # Iroh's NodeId encoding == base32 of the Ed25519 public key. We
    # derive the public key with cryptography (which we already depend
    # on) and feed it to iroh.PublicKey for canonical formatting.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    priv = Ed25519PrivateKey.from_private_bytes(secret_bytes)
    pub_raw = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    # ``str()`` on an iroh.PublicKey returns the full canonical hex
    # form. ``fmt_short()`` is the 10-char abbreviated form intended
    # for logs and IS NOT round-trippable through ``from_string``.
    return str(iroh.PublicKey.from_bytes(pub_raw))
