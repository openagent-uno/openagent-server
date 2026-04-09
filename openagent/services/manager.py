"""ServiceManager: starts/stops a set of auxiliary services as a group."""

from __future__ import annotations

import asyncio
import logging

from openagent.services.base import AuxService

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages the lifecycle of a collection of auxiliary services.

    Services are started in registration order and stopped in reverse.
    Failures in one service don't prevent the others from starting/stopping.
    """

    def __init__(self) -> None:
        self._services: list[AuxService] = []

    def add(self, service: AuxService) -> None:
        self._services.append(service)

    def __len__(self) -> int:
        return len(self._services)

    def __iter__(self):
        return iter(self._services)

    async def start_all(self) -> None:
        for svc in self._services:
            try:
                logger.info(f"Starting aux service: {svc.name}")
                await svc.start()
            except Exception as e:
                logger.error(f"Failed to start service '{svc.name}': {e}")

    async def stop_all(self) -> None:
        for svc in reversed(self._services):
            try:
                logger.info(f"Stopping aux service: {svc.name}")
                await svc.stop()
            except Exception as e:
                logger.error(f"Failed to stop service '{svc.name}': {e}")

    async def status_all(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for svc in self._services:
            try:
                out[svc.name] = await svc.status()
            except Exception as e:
                out[svc.name] = f"error: {e}"
        return out
