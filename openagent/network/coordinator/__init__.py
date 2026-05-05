"""Embedded coordinator service: PAKE login, device-cert issuance, agent registry.

Activated when ``network.coordinator.enabled: true`` in the agent config
(or when ``openagent network init`` has flipped the singleton ``network``
row to ``role='coordinator'``). Listens on its own Iroh ALPN
(``openagent/coordinator/1``) so it can run alongside the gateway on
the same Iroh endpoint.
"""

from openagent.network.coordinator.pake import (
    PakeBackend,
    Srp6aBackend,
    LoginInProgress,
    PakeError,
)
from openagent.network.coordinator.service import CoordinatorService
from openagent.network.coordinator.store import CoordinatorStore

__all__ = [
    "CoordinatorService",
    "CoordinatorStore",
    "PakeBackend",
    "Srp6aBackend",
    "LoginInProgress",
    "PakeError",
]
