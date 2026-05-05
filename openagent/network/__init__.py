"""Network layer — Iroh transport, identity, coordinator, device certs.

Replaces the legacy ``host:port + token`` connection model. Users log
in as ``handle@network`` with a password (PAKE); the coordinator
issues short-lived signed device certificates that gate every inbound
gateway request. See ``docs/network.md`` (or the plan file) for the
architecture overview.

Public re-exports kept intentionally small so callers don't reach into
sub-packages for things that should be stable across the network
layer's refactors.
"""

from openagent.network.identity import Identity, load_or_create_identity
from openagent.network.iroh_node import IrohNode, NetworkAlpn

__all__ = [
    "Identity",
    "IrohNode",
    "NetworkAlpn",
    "load_or_create_identity",
]
