"""Base channel interface and shared utilities."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openagent.agent import Agent


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
