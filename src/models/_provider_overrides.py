"""Runtime provider subclasses with OpenAgent-specific behavior.

DeepSeek's chat completions API only accepts plain ``text`` content
parts — it rejects ``{type:"file"}`` with
``Failed to deserialize the JSON body into the target type: messages[N]:
unknown variant 'file', expected 'text'`` and rejects ``image_url`` /
``input_audio`` parts the same way. The runtime's default ``DeepSeek``
class still emits them when ``message.files`` / ``message.images`` /
``message.audio`` / ``message.videos`` is populated, so any session that ever stored a
multimodal artifact (user upload, MCP-tool screenshot, prior turn from
a multimodal model) crashes the turn — and every subsequent turn until
the artifact scrolls out of history.

:class:`DeepSeekTextOnly` intercepts ``_format_message`` and rewrites
those attachments to text BEFORE the parent serializer builds the
request:

* Text-readable files (``application/json``, ``text/*``,
  ``application/x-python``, …) get their full UTF-8 content inlined
  in an ``<attachment>`` block.
* Binary files (``application/pdf``, office docs, …) and all images /
  audio/video get a placeholder line that names what was omitted and tells
  the user to switch to a multimodal model.

The mutation is local to the serialization call — the original
runtime ``Message`` is restored in the ``finally`` block so other
components reading the same message later don't see ``files=None`` /
``images=None`` unexpectedly (the runtime's team mode can serialize the
same in-memory Message multiple times).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.providers.deepseek import DeepSeek
from src.models.providers.message import Message

# ── Files ────────────────────────────────────────────────────────────

# MIME types DeepSeek can read once inlined as UTF-8 text. Everything
# else gets a placeholder. Kept narrow to avoid emitting garbled
# base64 into the prompt for binary formats.
_TEXT_READABLE_MIMES: frozenset[str] = frozenset({
    "application/json",
    "application/x-javascript",
    "application/x-python",
    "text/javascript",
    "text/x-python",
    "text/plain",
    "text/html",
    "text/css",
    "text/markdown",
    "text/csv",
    "text/xml",
})

_BINARY_PLACEHOLDER_HINT = (
    "DeepSeek can't read binary attachments. Switch to a multimodal "
    "model (e.g. anthropic / openai) to analyze this file."
)

# Per-file size cap (bytes). DeepSeek's context can absorb the small
# end of attachments; multi-MB JSON/CSV files would blow the prompt
# budget and slow down every subsequent turn for the same session.
# Truncate with a trailer so the model still sees the file shape but
# the rest gets dropped.
_MAX_INLINE_FILE_BYTES = 100_000


def _is_text_readable(mime: str | None, filename: str | None) -> bool:
    if mime and (mime in _TEXT_READABLE_MIMES or mime.startswith("text/")):
        return True
    # Last-ditch by extension when mime is missing — DeepSeek can read
    # the bytes either way once they're UTF-8 decoded.
    if filename:
        lower = filename.lower()
        for ext in (".json", ".txt", ".md", ".csv", ".xml", ".html", ".css",
                    ".py", ".js", ".ts", ".tsx", ".jsx", ".yml", ".yaml",
                    ".toml", ".ini", ".cfg", ".log", ".sh"):
            if lower.endswith(ext):
                return True
    return False


def _file_to_inline_block(file: Any) -> str:
    """Render one runtime ``File`` as an ``<attachment>`` text block.

    Reads via ``file.get_content_bytes()`` (sync — DeepSeek's
    ``_format_message`` is sync). On any read error we still emit a
    placeholder so the model is told something was attached.
    """
    filename = (
        getattr(file, "filename", None)
        or getattr(file, "name", None)
        or "attachment"
    )
    mime = getattr(file, "mime_type", None) or ""

    if not _is_text_readable(mime, filename):
        return (
            f'<attachment filename="{filename}" mime="{mime}" omitted="true">\n'
            f"{_BINARY_PLACEHOLDER_HINT}\n"
            f"</attachment>"
        )

    try:
        content_bytes = file.get_content_bytes()
    except Exception as exc:  # noqa: BLE001
        return (
            f'<attachment filename="{filename}" mime="{mime}" omitted="true">\n'
            f"Failed to read file: {exc}\n"
            f"</attachment>"
        )

    if not content_bytes:
        return (
            f'<attachment filename="{filename}" mime="{mime}" omitted="true">\n'
            f"File is empty or unreadable.\n"
            f"</attachment>"
        )

    original_size = len(content_bytes)
    truncated = False
    if original_size > _MAX_INLINE_FILE_BYTES:
        content_bytes = content_bytes[:_MAX_INLINE_FILE_BYTES]
        truncated = True

    try:
        text = content_bytes.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return (
            f'<attachment filename="{filename}" mime="{mime}" omitted="true">\n'
            f"Failed to decode file as UTF-8: {exc}\n"
            f"</attachment>"
        )

    trailer = ""
    if truncated:
        trailer = (
            f"\n... (truncated: original size {original_size} bytes, "
            f"showing first {_MAX_INLINE_FILE_BYTES})"
        )

    return (
        f'<attachment filename="{filename}" mime="{mime}">\n'
        f"{text}{trailer}\n"
        f"</attachment>"
    )


def _files_to_inline_text(files: list[Any]) -> str:
    blocks = [_file_to_inline_block(f) for f in files if f is not None]
    return "\n\n".join(b for b in blocks if b)


# ── Images / audio ──────────────────────────────────────────────────

# DeepSeek can't read binary modalities at all. Replace each one with
# a single labelled placeholder line so the model still sees that an
# artifact existed (useful for "the user attached an image, ask them
# to switch model" hints) without ever emitting an image_url /
# input_audio content part that the API would reject.


def _image_label(image: Any) -> str:
    mime = getattr(image, "mime_type", None) or "image/*"
    name = (
        getattr(image, "filename", None)
        or getattr(image, "id", None)
        or ""
    )
    label = f"image ({mime})"
    if name:
        label = f"{label}: {name}"
    return label


def _audio_label(audio: Any) -> str:
    fmt = getattr(audio, "format", None) or getattr(audio, "mime_type", None) or "audio/*"
    name = (
        getattr(audio, "filename", None)
        or getattr(audio, "id", None)
        or ""
    )
    label = f"audio ({fmt})"
    if name:
        label = f"{label}: {name}"
    return label


def _media_placeholder(
    images: list[Any] | None,
    audios: list[Any] | None,
    videos: list[Any] | None = None,
) -> str:
    """Render binary modalities as a single ``<media-omitted>`` block.

    Returns an empty string when nothing was attached. Kept compact —
    DeepSeek doesn't need a per-item block, just a note that the user
    (or a tool result) had multimodal content the model can't see.
    """
    lines: list[str] = []
    for img in images or []:
        lines.append(f"- {_image_label(img)}")
    for au in audios or []:
        lines.append(f"- {_audio_label(au)}")
    for video in videos or []:
        mime = getattr(video, "mime_type", None) or "video/*"
        name = getattr(video, "filename", None) or getattr(video, "id", None) or ""
        lines.append(f"- video ({mime})" + (f": {name}" if name else ""))
    if not lines:
        return ""
    return (
        "<media-omitted>\n"
        "DeepSeek can't read images, audio, or video. The following attachments "
        "were stripped from this turn — switch to a multimodal model "
        "(e.g. anthropic / openai / google) to analyze them:\n"
        + "\n".join(lines)
        + "\n</media-omitted>"
    )


# ── Model subclass ──────────────────────────────────────────────────


@dataclass
class DeepSeekTextOnly(DeepSeek):
    """DeepSeek subclass that strips files / images / audio / video to text.

    Overrides ``_format_message`` to translate ``message.files`` into
    inline ``<attachment>`` blocks and replace ``message.images`` /
    ``message.audio`` with a ``<media-omitted>`` placeholder, then
    delegates to the parent serializer with those fields cleared so
    DeepSeek's API never sees ``{type:"file"}`` / ``image_url`` /
    ``input_audio`` content parts.
    """

    name: str = "DeepSeek"

    def _format_message(
        self,
        message: Message,
        compress_tool_results: bool = False,
    ) -> dict[str, Any]:
        has_files = bool(getattr(message, "files", None))
        has_images = bool(getattr(message, "images", None))
        has_audio = bool(getattr(message, "audio", None))
        has_videos = bool(getattr(message, "videos", None))
        if not (has_files or has_images or has_audio or has_videos):
            return super()._format_message(message, compress_tool_results)

        inlined_files = _files_to_inline_text(message.files) if has_files else ""
        media_block = (
            _media_placeholder(
                message.images if has_images else None,
                message.audio if has_audio else None,
                message.videos if has_videos else None,
            )
            if (has_images or has_audio or has_videos) else ""
        )

        prefix_parts = [p for p in (inlined_files, media_block) if p]
        prefix = "\n\n".join(prefix_parts)

        original_files = message.files
        original_images = message.images
        original_audio = message.audio
        original_videos = message.videos
        original_content = message.content
        try:
            message.files = None
            message.images = None
            message.audio = None
            message.videos = None
            if prefix:
                if isinstance(message.content, str) and message.content:
                    message.content = (prefix + "\n\n" + message.content).strip()
                elif message.content is None or message.content == "":
                    message.content = prefix
            return super()._format_message(message, compress_tool_results)
        finally:
            message.files = original_files
            message.images = original_images
            message.audio = original_audio
            message.videos = original_videos
            message.content = original_content


# Back-compat alias for the initial files-only release.
DeepSeekInlineFiles = DeepSeekTextOnly


__all__ = ["DeepSeekTextOnly", "DeepSeekInlineFiles"]
