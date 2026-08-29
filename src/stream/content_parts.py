"""Ordered assistant content parts and streaming-safe OA-UI marker filtering.

Markers are an agent-to-gateway carrier only.  They must never be rendered or
spoken verbatim: rich clients receive a structured ``ui_view`` part, while
text-only clients keep the surrounding fallback text.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


UI_MARKER_PREFIX = "[OPENAGENT_UI:"
CONTENT_MARKER_PREFIXES = (
    UI_MARKER_PREFIX,
    "[FILE:",
    "[IMAGE:",
    "[VOICE:",
    "[VIDEO:",
)
_UI_ID = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
UI_MARKER_RE = re.compile(
    rf"\[OPENAGENT_UI:(?P<view_id>{_UI_ID})@(?P<revision>[1-9][0-9]{{0,9}})\]"
)
_CONTENT_MARKER_RE = re.compile(
    rf"\[(?P<attachment_kind>IMAGE|FILE|VOICE|VIDEO):(?P<path>[^\]]+)\]"
    rf"|\[OPENAGENT_UI:(?P<view_id>{_UI_ID})@(?P<revision>[1-9][0-9]{{0,9}})\]"
)
_ATTACHMENT_KIND = {
    "IMAGE": "image",
    "FILE": "file",
    "VOICE": "voice",
    "VIDEO": "video",
}


@dataclass(frozen=True)
class ParsedContent:
    text: str
    parts: tuple[dict[str, Any], ...] = ()
    attachments: tuple[dict[str, Any], ...] = ()
    ui_refs: tuple[dict[str, Any], ...] = ()


def _append_text(parts: list[dict[str, Any]], value: str) -> None:
    if not value:
        return
    if parts and parts[-1].get("kind") == "text":
        parts[-1]["text"] = str(parts[-1].get("text") or "") + value
    else:
        parts.append({"kind": "text", "text": value})


def _trim_text_edges(parts: list[dict[str, Any]]) -> None:
    text_indexes = [i for i, part in enumerate(parts) if part.get("kind") == "text"]
    if not text_indexes:
        return
    first, last = text_indexes[0], text_indexes[-1]
    parts[first]["text"] = str(parts[first].get("text") or "").lstrip()
    parts[last]["text"] = str(parts[last].get("text") or "").rstrip()
    parts[:] = [
        part for part in parts
        if part.get("kind") != "text" or bool(str(part.get("text") or ""))
    ]


def parse_response_content(text: str, *, allow_inline_ui: bool) -> ParsedContent:
    """Parse attachment and OA-UI markers while retaining their order.

    OA-UI-looking markers are always stripped from visible text.  A valid
    ``ui_view`` part is emitted only for a client that advertised inline UI;
    this is the defence-in-depth channel/CLI policy.
    """

    raw = text or ""
    parts: list[dict[str, Any]] = []
    attachments: list[dict[str, Any]] = []
    ui_refs: list[dict[str, Any]] = []
    cursor = 0

    for match in _CONTENT_MARKER_RE.finditer(raw):
        _append_text(parts, raw[cursor:match.start()])
        cursor = match.end()
        marker_kind = match.group("attachment_kind")
        if marker_kind:
            path = (match.group("path") or "").strip()
            if not path:
                continue
            attachment = {
                "type": _ATTACHMENT_KIND.get(marker_kind, "file"),
                "path": path,
                "filename": Path(path).name,
            }
            attachments.append(attachment)
            parts.append({"kind": "attachment", "attachment": attachment})
            continue

        # This branch can only be reached by a syntactically valid marker.
        ref = {
            "kind": "ui_view",
            "view_id": str(match.group("view_id")),
            "revision": int(match.group("revision")),
        }
        ui_refs.append(ref)
        if allow_inline_ui:
            parts.append(ref)

    _append_text(parts, raw[cursor:])

    # Strip malformed carrier fragments too. They are neither useful fallback
    # text nor safe content for a rendered/spoken surface.
    def _strip_malformed(value: str) -> str:
        while True:
            starts = [
                (start, prefix)
                for prefix in CONTENT_MARKER_PREFIXES
                if (start := value.find(prefix)) >= 0
            ]
            if not starts:
                break
            start, prefix = min(starts, key=lambda item: item[0])
            end = value.find("]", start + len(prefix))
            value = value[:start] + (value[end + 1:] if end >= 0 else "")
        return value

    for part in parts:
        if part.get("kind") == "text":
            part["text"] = _strip_malformed(str(part.get("text") or ""))
    _trim_text_edges(parts)

    clean_text = _CONTENT_MARKER_RE.sub("", raw)
    clean_text = _strip_malformed(clean_text).strip()
    return ParsedContent(
        text=clean_text,
        parts=tuple(parts),
        attachments=tuple(attachments),
        ui_refs=tuple(ui_refs),
    )


def _carrier_free_content(value: Any) -> Any:
    """Copy text-bearing content while removing transport-only carriers."""

    if isinstance(value, str):
        return parse_response_content(value, allow_inline_ui=False).text
    if isinstance(value, list):
        cleaned: list[Any] = []
        for item in value:
            if isinstance(item, str):
                cleaned.append(_carrier_free_content(item))
            elif isinstance(item, dict):
                cloned = dict(item)
                for key in ("text", "content"):
                    if isinstance(cloned.get(key), str):
                        cloned[key] = _carrier_free_content(cloned[key])
                cleaned.append(cloned)
            else:
                cleaned.append(item)
        return cleaned
    return value


def scrub_run_output_carriers_for_storage(run_output: Any) -> None:
    """Remove UI/file carrier markers from a storage-only run copy.

    The live ``RunOutput`` remains untouched so the gateway can still turn a
    marker into ordered structured content.  Legacy session JSON and the v2
    projection receive only the human-readable fallback text; durable UI and
    attachment identity belongs in normalized tables.
    """

    content = getattr(run_output, "content", None)
    cleaned_content = _carrier_free_content(content)
    if cleaned_content is not content:
        setattr(run_output, "content", cleaned_content)

    messages = getattr(run_output, "messages", None)
    if not isinstance(messages, list):
        return
    cleaned_messages: list[Any] = []
    for message in messages:
        raw_role = getattr(message, "role", "")
        role = str(getattr(raw_role, "value", raw_role) or "").lower()
        if role != "assistant":
            cleaned_messages.append(message)
            continue
        cloned = copy.copy(message)
        message_content = getattr(message, "content", None)
        setattr(cloned, "content", _carrier_free_content(message_content))
        cleaned_messages.append(cloned)
    setattr(run_output, "messages", cleaned_messages)


@dataclass
class ContentMarkerStreamFilter:
    """Hide UI and legacy attachment carriers across arbitrary deltas.

    The final-response parser still turns complete carriers into structured
    message parts.  This filter only protects the live text/TTS path, where a
    marker can be split at any byte boundary and must never briefly render or
    be spoken.
    """

    pending: str = ""
    refs: list[dict[str, Any]] = field(default_factory=list)
    max_marker_chars: int = 2048
    _discarding_marker: bool = False

    @staticmethod
    def _partial_prefix_len(value: str) -> int:
        limit = min(
            len(value),
            max(len(prefix) for prefix in CONTENT_MARKER_PREFIXES) - 1,
        )
        for size in range(limit, 0, -1):
            if any(
                prefix.startswith(value[-size:])
                for prefix in CONTENT_MARKER_PREFIXES
            ):
                return size
        return 0

    @staticmethod
    def _next_marker(value: str) -> tuple[int, str] | None:
        matches = [
            (index, prefix)
            for prefix in CONTENT_MARKER_PREFIXES
            if (index := value.find(prefix)) >= 0
        ]
        return min(matches, key=lambda item: item[0]) if matches else None

    def feed(self, chunk: str) -> str:
        data = self.pending + (chunk or "")
        self.pending = ""
        visible: list[str] = []

        if self._discarding_marker:
            end = data.find("]")
            if end < 0:
                return ""
            self._discarding_marker = False
            data = data[end + 1:]

        while data:
            found = self._next_marker(data)
            if found is None:
                keep = self._partial_prefix_len(data)
                if keep:
                    visible.append(data[:-keep])
                    self.pending = data[-keep:]
                else:
                    visible.append(data)
                break

            start, prefix = found
            visible.append(data[:start])
            candidate = data[start:]
            end = candidate.find("]", len(prefix))
            if end < 0:
                if len(candidate) <= self.max_marker_chars:
                    self.pending = candidate
                else:
                    # Keep suppressing future chunks until the carrier closes.
                    self._discarding_marker = True
                break

            marker = candidate[:end + 1]
            if prefix == UI_MARKER_PREFIX:
                match = UI_MARKER_RE.fullmatch(marker)
                if match:
                    self.refs.append({
                        "kind": "ui_view",
                        "view_id": match.group("view_id"),
                        "revision": int(match.group("revision")),
                    })
            # Valid or malformed, the carrier itself is never visible.
            data = candidate[end + 1:]

        return "".join(visible)

    def finish(self) -> str:
        pending, self.pending = self.pending, ""
        # A complete marker prefix is a truncated carrier and stays hidden;
        # a shorter partial prefix is ordinary prose and may be released.
        if self._discarding_marker:
            self._discarding_marker = False
            return ""
        return "" if any(
            pending.startswith(prefix) for prefix in CONTENT_MARKER_PREFIXES
        ) else pending


# Compatibility for imports introduced with the first OA-UI implementation.
# The class now filters every agent-to-server content carrier, not UI alone.
UiMarkerStreamFilter = ContentMarkerStreamFilter


__all__ = [
    "CONTENT_MARKER_PREFIXES",
    "UI_MARKER_PREFIX",
    "UI_MARKER_RE",
    "ContentMarkerStreamFilter",
    "ParsedContent",
    "UiMarkerStreamFilter",
    "parse_response_content",
    "scrub_run_output_carriers_for_storage",
]
