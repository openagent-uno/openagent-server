"""Base channel interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openagent.agent import Agent


class BaseChannel(ABC):
    """Abstract base for messaging channels (Telegram, Discord, WhatsApp, etc.).

    Each channel manages per-user sessions via the agent's memory system.
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel."""
        ...

    def _user_session_id(self, platform: str, user_id: str) -> str:
        """Generate a consistent session ID from platform + user ID."""
        return f"{platform}:{self.agent.name}:{user_id}"
