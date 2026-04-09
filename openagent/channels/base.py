"""Base channel interface and shared utilities."""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openagent.agent import Agent

logger = logging.getLogger(__name__)

# Retry cooldown between channel crashes
CHANNEL_RETRY_SECONDS = 45


@dataclass
class Attachment:
    """A file/image/voice attachment from or to a channel."""
    type: str          # "image", "file", "voice", "video"
    path: str          # local file path
    filename: str
    mime_type: str | None = None
    caption: str | None = None


# Pattern for response markers: [IMAGE:/path/to/file.png] [FILE:/path] [VOICE:/path]
_MARKER_PATTERN = re.compile(r'\[(IMAGE|FILE|VOICE|VIDEO):([^\]]+)\]')

_MARKER_TYPE_MAP = {
    "IMAGE": "image",
    "FILE": "file",
    "VOICE": "voice",
    "VIDEO": "video",
}


def parse_response_markers(text: str) -> tuple[str, list[Attachment]]:
    """Extract file markers from agent response text.

    Returns (clean_text, attachments).
    Markers like [IMAGE:/path/to/chart.png] are removed from text
    and returned as Attachment objects.
    """
    attachments: list[Attachment] = []
    for match in _MARKER_PATTERN.finditer(text):
        marker_type = match.group(1)
        file_path = match.group(2).strip()
        att_type = _MARKER_TYPE_MAP.get(marker_type, "file")
        filename = Path(file_path).name
        attachments.append(Attachment(
            type=att_type,
            path=file_path,
            filename=filename,
        ))

    clean_text = _MARKER_PATTERN.sub("", text).strip()
    return clean_text, attachments


def format_attachments_for_prompt(attachments: list[Attachment], caption: str = "") -> str:
    """Build a prompt string describing attachments for the agent."""
    parts = []
    for att in attachments:
        parts.append(f"[Attached {att.type}: {att.filename}]")
    prefix = " ".join(parts)
    if caption:
        return f"{prefix}\n{caption}"
    return prefix


class BaseChannel(ABC):
    """Abstract base for messaging channels (Telegram, Discord, WhatsApp, etc.).

    Subclasses implement `_run()` (the actual listen loop) and `_shutdown()`
    (cleanup). `start()` handles the supervised retry loop automatically:
    on crash it waits CHANNEL_RETRY_SECONDS and restarts, until `stop()` is
    called.
    """

    name: str = "channel"  # override in subclass for logging

    def __init__(self, agent: Agent):
        self.agent = agent
        self._should_stop = False
        self._stop_event: asyncio.Event | None = None

    async def start(self) -> None:
        """Supervised start. Retries on crash until stop() is called."""
        self._should_stop = False
        self._stop_event = asyncio.Event()
        while not self._should_stop:
            try:
                await self._run()
                if self._should_stop:
                    break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._should_stop:
                    break
                logger.error(
                    f"{self.name} channel crashed: {e}, "
                    f"restarting in {CHANNEL_RETRY_SECONDS}s..."
                )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=CHANNEL_RETRY_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass

    async def stop(self) -> None:
        """Request stop. Calls subclass shutdown then signals the retry loop."""
        self._should_stop = True
        try:
            await self._shutdown()
        except Exception as e:
            logger.warning(f"{self.name} shutdown error: {e}")
        if self._stop_event is not None:
            self._stop_event.set()

    @abstractmethod
    async def _run(self) -> None:
        """Run the channel listener. Subclasses implement this.

        Should block until the channel stops listening or an error occurs.
        The base class handles retry on exceptions.
        """
        ...

    @abstractmethod
    async def _shutdown(self) -> None:
        """Clean up channel resources (close client, stop polling, etc.)."""
        ...

    def _user_session_id(self, platform: str, user_id: str) -> str:
        """Generate a consistent session ID from platform + user ID."""
        return f"{platform}:{self.agent.name}:{user_id}"
