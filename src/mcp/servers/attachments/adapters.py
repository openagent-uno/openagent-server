"""Provider adapters for the in-process attachments MCP.

Exposes ``send_file_to_user(path)`` to the agent. Calling it validates
the path and returns a marker string — ``[IMAGE:/path]``,
``[VIDEO:/path]``, ``[VOICE:/path]``, or ``[FILE:/path]`` depending on
the file's extension — that the agent must include verbatim in its
final reply. ``parse_response_markers`` in ``src/channels/base.py``
then extracts the marker and the channel adapter delivers the file as
a real attachment (inline image in the desktop app, ``reply_photo`` /
``reply_document`` on Telegram, ``discord.File`` on Discord, native
upload on WhatsApp/Slack).

This is the ONLY way for the agent to ship a file back through the
current chat. Reading a file with ``Read`` and quoting the path in
prose does not attach anything — only this tool does.

Both Claude Agent SDK and the native runtime consume in-process tools,
so we wrap the same Python implementation twice: once as an SDK MCP
server, once as a runtime ``Toolkit``. Pattern matches ``shell/adapters.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".tiff", ".svg",
})
_VIDEO_EXTS = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
})
_VOICE_EXTS = frozenset({
    ".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac",
})


def _marker_kind(path: Path) -> str:
    """Pick IMAGE / VIDEO / VOICE / FILE from the file's extension.

    Image wins over video wins over voice on ties (``.webm`` is treated
    as video, not audio — matches how Telegram/Discord auto-preview it).
    """
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "IMAGE"
    if ext in _VIDEO_EXTS:
        return "VIDEO"
    if ext in _VOICE_EXTS:
        return "VOICE"
    return "FILE"


def _send_file_impl(path: str) -> dict:
    """Validate the path and build the marker payload."""
    if not path or not isinstance(path, str):
        return {"ok": False, "error": "path required (absolute filesystem path)"}
    try:
        real = Path(path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError):
        return {"ok": False, "error": f"path not found: {path}"}
    if not real.is_file():
        return {"ok": False, "error": f"path is not a regular file: {real}"}
    kind = _marker_kind(real)
    marker = f"[{kind}:{real}]"
    return {
        "ok": True,
        "marker": marker,
        "kind": kind.lower(),
        "path": str(real),
        "filename": real.name,
        "instruction": (
            f"Include {marker!r} VERBATIM in your reply to the user. "
            "The marker is stripped from the displayed text and the "
            "file is delivered as a proper chat attachment. Do NOT "
            "describe the path in prose instead — without the marker, "
            "nothing is attached."
        ),
    }


# ── Native runtime ──────────────────────────────────────────────────

def build_runtime_toolkit() -> Any:
    """Return a ``Toolkit`` exposing ``send_file_to_user``.

    Follows the runtime Toolkit pattern: plain async callables with type
    hints + docstrings; the runtime introspects the signature to build
    the tool schema and the docstring becomes the description shown to
    the model.
    """
    from src.mcp._runtime import Toolkit

    async def send_file_to_user(path: str) -> dict:
        """Send a file from the agent's filesystem to the user as a chat attachment.

        Auto-detects whether to deliver the file as an image, video,
        voice note, or generic document from its extension. Images
        render inline in the desktop app; on Telegram/WhatsApp/Discord
        the platform's native media type is used.

        Returns a payload with a ``marker`` field (e.g.
        ``[FILE:/tmp/report.pdf]``) that you MUST include verbatim in
        your reply to the user — otherwise no attachment is delivered.
        Quoting the path in prose without the marker attaches nothing.

        Args:
            path: Absolute path to the file on the agent's filesystem.
        """
        return _send_file_impl(path)

    return Toolkit(
        name="attachments",
        tools=[send_file_to_user],
    )
