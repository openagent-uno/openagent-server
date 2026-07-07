"""Shared utilities for message formatting, attachment parsing, and text splitting.

Used by both the Gateway and platform bridges.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ToolStatusEvent:
    """One parsed tool-status event.

    The agent fires status events in two shapes (the chat store and
    every bridge / turn runner has historically had to detect both):

      * JSON (API-native ``ToolExecution.to_dict()``):
        ``{"tool_name": "ReadFile", "tool_call_error": false, "result": ..., ...}``
      * Plain: ``"Using ReadFile..."``

    Phase is derived from the runtime fields: ``tool_call_error`` → error,
    ``result`` present → done, otherwise → running. This dataclass
    normalises both shapes so callers can branch on ``status`` without
    re-parsing.

    ``tool`` is the REAL tool: when the agent reaches a tool through the
    deferred-tool dispatcher (``tool_search_call_tool({server, tool, args})``
    — see :mod:`src.mcp.servers.tool_search`), ``parse_status_event`` unwraps
    it so ``tool`` is the inner tool, ``server`` its MCP, and ``tool_args``
    the inner args. This mirrors the app's ``effectiveTool`` so the friendly
    channel labels (:mod:`src.channels.tool_labels`) match the app's chips.
    """
    tool: str
    status: str  # "running" / "done" / "error"
    error: str | None = None
    tool_args: dict | None = None
    server: str | None = None  # MCP server when reached via the dispatcher


_TOOL_SEARCH_DISPATCHER = "tool_search_call_tool"


def parse_status_event(raw: str) -> Optional[ToolStatusEvent]:
    """Parse a status string from ``Agent._status``. Returns ``None``
    for non-tool statuses (``"Thinking..."``, plain text, empty input).

    Same try-JSON-then-fallback shape every consumer used to inline.
    """
    text = (raw or "").strip()
    if not text:
        return None

    # JSON shape first — the runtime's native ``ToolExecution.to_dict()``.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("tool_name"):
            is_err = bool(parsed.get("tool_call_error"))
            has_result = parsed.get("result") is not None
            status = "error" if is_err else ("done" if has_result else "running")
            err_text: str | None = None
            if is_err and parsed.get("result") is not None:
                err_text = str(parsed["result"])
            tool_name = str(parsed["tool_name"])
            raw_args = parsed.get("tool_args")
            tool_args = raw_args if isinstance(raw_args, dict) else None
            server: str | None = None
            # Unwrap the deferred-tool dispatcher so the real tool surfaces —
            # exactly like the app's ``effectiveTool``.
            if tool_name == _TOOL_SEARCH_DISPATCHER and tool_args:
                inner = tool_args.get("tool")
                if isinstance(inner, str) and inner:
                    srv = tool_args.get("server")
                    server = srv if isinstance(srv, str) else None
                    inner_args = tool_args.get("args")
                    tool_args = inner_args if isinstance(inner_args, dict) else {}
                    tool_name = inner
            return ToolStatusEvent(
                tool=tool_name,
                status=status,
                error=err_text,
                tool_args=tool_args,
                server=server,
            )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Plain "Using X..." text from the legacy code path.
    if text.startswith("Using "):
        tool = text[len("Using "):].rstrip(".").strip()
        if tool:
            return ToolStatusEvent(tool=tool, status="running")

    return None


def is_reasoning_status(raw: str) -> bool:
    """True for a plain "agent is thinking" UI string (``"Thinking..."``,
    ``"Loading context..."``), False for anything that is structured data.

    Used by the turn runner to split the ``on_status`` stream: a plain
    thinking string becomes the typed ``OutReasoning(active=True)`` flag
    (the client renders its own indicator), while tool-execution JSON
    keeps flowing as ``OutToolStatus`` because it carries real data the
    client renders. The ``{"kind":"session.compacted"}`` envelope is
    lifted out even earlier — the turn runner checks
    :func:`parse_compaction_status` first and emits a typed
    ``SessionCompacted`` frame — so it never reaches this classifier;
    were it to, the JSON-object guard below would (harmlessly) treat it
    as non-reasoning too.

    The test is deliberately structural rather than a hardcoded string set,
    so a new plain status added upstream is classified correctly without a
    second edit here: it's reasoning iff it is non-empty, is not a parsed
    tool event, and is not a JSON object/array.
    """
    text = (raw or "").strip()
    if not text:
        return False
    if parse_status_event(text) is not None:
        return False  # tool execution event
    if text[:1] in "{[":
        try:
            json.loads(text)
            return False  # structured JSON envelope (e.g. session.compacted)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return True


def parse_compaction_status(raw: str) -> dict | None:
    """Parse a session-compaction status envelope, or return ``None``.

    ``src.core.compaction.compact`` fires its progress hints through the
    same ``on_status`` channel every tool uses, as a JSON envelope tagged
    ``{"kind": "session.compacted", "phase": ...}``. The turn runner and
    the bridges call this to lift that envelope out of the tool-status
    stream and render it as a first-class compaction affordance (vision
    §2) instead of leaking the raw JSON into a "Using ..." line.

    Returns the payload dict with a normalised ``phase`` in
    ``{"running", "done", "error"}`` (legacy envelopes without a phase read
    as ``"done"`` — the pre-phase behaviour was a single fire-after-fold),
    the integer stat fields coerced to ``int``, or ``None`` when *raw* is
    not a compaction envelope.
    """
    text = (raw or "").strip()
    if not text or text[:1] != "{":
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or parsed.get("kind") != "session.compacted":
        return None
    phase = parsed.get("phase")
    if phase not in ("running", "done", "error"):
        phase = "done"
    out: dict = {"phase": phase}
    for key in (
        "folded_runs", "kept_runs_count", "summary_chars",
        "tokens_before", "tokens_after",
    ):
        try:
            out[key] = int(parsed.get(key) or 0)
        except (TypeError, ValueError):
            out[key] = 0
    return out


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
