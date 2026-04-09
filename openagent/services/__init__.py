"""Auxiliary services for OpenAgent (Obsidian web UI, future: VNC, Caddy, etc.)."""

from openagent.services.base import AuxService
from openagent.services.manager import ServiceManager
from openagent.services.obsidian import ObsidianWebService

__all__ = ["AuxService", "ServiceManager", "ObsidianWebService"]
