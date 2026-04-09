"""Auxiliary services managed by AgentServer (Syncthing, future additions)."""

from openagent.services.base import AuxService
from openagent.services.manager import ServiceManager
from openagent.services.syncthing import SyncthingService

__all__ = ["AuxService", "ServiceManager", "SyncthingService"]
