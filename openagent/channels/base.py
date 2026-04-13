"""Shared utilities for message formatting, attachment parsing, and text splitting.

Used by both the Gateway and platform bridges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Attachment:
    """A file/image/voice attachment."""
    type: str          # "image", "file", "voice", "video"
    path: str
    filename: str
    mime_type: str | None = None
    caption: str | None = None


# Response markers: [IMAGE:/path] [FILE:/path] [VOICE:/path] [VIDEO:/path]
_MARKER_RE = re.compile(r'\[(IMAGE|FILE|VOICE|VIDEO):([^\]]+)\]')
_MARKER_MAP = {"IMAGE": "image", "FILE": "file", "VOICE": "voice", "VIDEO": "video"}


def parse_response_markers(text: str) -> tuple[str, list[Attachment]]:
    """Extract file markers from agent response, return (clean_text, attachments)."""
    attachments = [
        Attachment(type=_MARKER_MAP.get(m.group(1), "file"), path=m.group(2).strip(), filename=Path(m.group(2).strip()).name)
        for m in _MARKER_RE.finditer(text)
    ]
    return _MARKER_RE.sub("", text).strip(), attachments


BLOCKED_EXTENSIONS = frozenset({
    ".exe", ".bat", ".cmd", ".com", ".msi", ".scr", ".pif",
    ".vbs", ".vbe", ".jse", ".ws", ".wsf", ".wsh", ".ps1", ".hta",
    ".cpl", ".lnk", ".reg", ".jar",
})

ATTACHMENT_READ_HINT = "Use the Read tool with the local path to inspect each file."


def is_blocked_attachment(filename: str | None) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in BLOCKED_EXTENSIONS


def build_attachment_context(
    files_info: list[str],
    *,
    read_hint: str = ATTACHMENT_READ_HINT,
) -> str:
    """Build a consistent context block for local files attached by the user."""
    lines = ["The user attached files:", *files_info]
    if read_hint:
        lines.append(read_hint)
    return "\n".join(lines)


def prepend_context_block(text: str, context: str) -> str:
    """Prepend a context block to user text with a blank-line separator."""
    return f"{context}\n\n{text}" if text else context


def split_preserving_code_blocks(text: str, max_len: int) -> list[str]:
    """Split text into chunks ≤ max_len, keeping ``` fences balanced."""
    if max_len <= 0:
        return [text] if text else []
    if len(text) <= max_len:
        return [text] if text.strip() else []

    chunks: list[str] = []
    carry = ""
    i, n = 0, len(text)

    while i < n:
        budget = max(max_len - len(carry), max_len // 2)
        end = min(i + budget, n)
        if end < n:
            nl = text.rfind("\n", i, end)
            if nl > i + budget // 4:
                end = nl
            else:
                sp = text.rfind(" ", i, end)
                if sp > i + budget // 4:
                    end = sp

        chunk = carry + text[i:end]
        if chunk.count("```") % 2 == 1:
            chunk += "\n```"
            carry = "```\n"
        else:
            carry = ""

        if chunk.strip():
            chunks.append(chunk)
        i = end
        while i < n and text[i] in ("\n", " "):
            i += 1

    return chunks
