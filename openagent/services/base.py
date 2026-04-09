"""Base class for auxiliary services managed alongside the agent."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AuxService(ABC):
    """An auxiliary service managed by OpenAgent's lifecycle.

    Examples: a Docker container running Obsidian web UI, a VNC server for
    computer-use, a Caddy reverse proxy. Services are started when the agent
    starts and stopped when it shuts down.

    Subclasses implement `start()`, `stop()`, and `status()`.
    """

    name: str = "aux-service"

    @abstractmethod
    async def start(self) -> None:
        """Start the service. Idempotent: calling twice is a no-op."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the service. Idempotent: calling twice is a no-op."""
        ...

    @abstractmethod
    async def status(self) -> str:
        """Return a human-readable status string (e.g. 'running', 'stopped')."""
        ...
