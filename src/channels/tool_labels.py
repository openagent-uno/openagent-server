"""Friendly, user-facing tool labels for channel messages.

Channels (Telegram, WhatsApp, Discord, Slack) narrate the agent's tool use
as chat messages and typing-status lines (see ``BaseBridge.dispatch_turn``).
Raw tool names — ``vault_write_note``, ``shell_shell_exec``, or the
dispatcher-unwrapped ``read_note`` — are an implementation detail. This module
turns them into human verbs, with first-class copy for memory-vault operations
("Recalling", "Memorizing", "Forgetting", …).

It is the Python twin of the universal app's ``toolDisplay`` (the app's
``common/types.ts``): a tool call reads the same on a phone channel as it does
on the app's tool chip. Keep the two in sync — same verbs, same op coverage.

The input is a :class:`src.channels.base.ToolStatusEvent`, which has already
unwrapped the deferred-tool dispatcher (so ``evt.tool`` is the real tool and
``evt.server`` its MCP).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from src.channels.base import ToolStatusEvent

# Memory op → (friendly verb, emoji). Keyed by the bare op (see ``_memory_op``);
# a few ops carry aliases (the obsidian-MCP raw name vs the vault-gate name).
_MEMORY_VERBS: dict[str, tuple[str, str]] = {
    "read_note": ("Recalling", "📖"),
    "read_multiple_notes": ("Recalling", "📖"),
    "get_frontmatter": ("Reading memory details", "📖"),
    "write_note": ("Memorizing", "🧠"),
    "patch_note": ("Updating memory", "📝"),
    "update_frontmatter": ("Updating memory", "📝"),
    "update_user_memory": ("Updating memory", "🧠"),
    "validate_note": ("Checking memory", "✅"),
    "delete_note": ("Forgetting", "🗑️"),
    "move_note": ("Reorganizing memory", "🔀"),
    "rename_note": ("Renaming memory", "🔀"),
    "search_notes": ("Searching memory", "🔍"),
    "search": ("Searching memory", "🔍"),
    "list_notes": ("Browsing memory", "🗂️"),
    "list_all_tags": ("Memory tags", "🏷️"),
    "manage_tags": ("Tagging memory", "🏷️"),
    "get_backlinks": ("Tracing memory links", "🔗"),
    "backlinks": ("Tracing memory links", "🔗"),
    "get_vault_stats": ("Memory stats", "📊"),
    "stats": ("Memory stats", "📊"),
    "gate": ("Auditing memory", "🛡️"),
    "doctor": ("Healing memory", "🩺"),
    "dream": ("Memory maintenance", "🌙"),
    "init": ("Setting up memory", "🗄️"),
    "regenerate_derived": ("Rebuilding memory index", "🔄"),
}

# General (non-memory) tools: first matching pattern wins, else a Title-Case
# fallback. Mirrors the app's GENERAL_VERBS.
_GENERAL_VERBS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"(^|_)(bash|shell|exec|run_command|terminal)"), "Running command", "🔧"),
    (re.compile(r"(web[_-]?search|search_web)"), "Searching the web", "🌐"),
    (re.compile(r"(fetch|http|browse|navigate|open_url|web)"), "Browsing the web", "🌐"),
    (re.compile(r"(read_file|read_multiple_files|^read$|cat_file|view_file)"), "Reading file", "📄"),
    (re.compile(r"(write_file|^write$|str_replace|edit_file|^edit$|apply_patch|create_file)"), "Editing file", "✏️"),
    (re.compile(r"(list_dir|list_files|^ls$|glob|find_files)"), "Listing files", "📁"),
    (re.compile(r"(grep|ripgrep|search_code|search_files)"), "Searching files", "🔍"),
    (re.compile(r"(send_file|send_message|messaging|notify|email)"), "Sending message", "✉️"),
    (re.compile(r"(image|media|generate|render|draw)"), "Generating media", "🖼️"),
]

_NOTE_PATH_KEYS = ("new_path", "path", "note_path", "filepath", "filename")
_QUERY_KEYS = ("query", "q", "pattern", "search", "text")
_MEMORY_NAME_RE = re.compile(r"(^|[_-])(vault|note)(s)?([_-]|$)")


def _memory_op(name: str) -> str:
    """Reduce a tool name to its bare op by stripping the MCP server prefix,
    so prefixed ``vault_read_note`` and unwrapped ``read_note`` collapse to
    the same op."""
    n = name.lower()
    n = re.sub(r"^vault[_-]gate[_-]", "", n)
    n = re.sub(r"^vaultgate[_-]", "", n)
    n = re.sub(r"^vault[_-]", "", n)
    n = re.sub(r"^memory[_-]search[_-]", "", n)
    return n


def _is_memory_server(server: Optional[str]) -> bool:
    if not server:
        return False
    x = server.lower()
    return x == "vault" or x.startswith("vault") or "memory" in x


def is_memory_tool(evt: ToolStatusEvent) -> bool:
    """Whether a tool call is a memory-vault operation (by server or name)."""
    name = evt.tool.lower()
    return (
        _is_memory_server(evt.server)
        or name == "update_user_memory"
        or _MEMORY_NAME_RE.search(name) is not None
        or _memory_op(evt.tool) in _MEMORY_VERBS
    )


def _note_title(path: str) -> str:
    base = path.split("/")[-1] or path
    return re.sub(r"\.md$", "", base, flags=re.IGNORECASE)


def _truncate(text: str, limit: int = 48) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _detail(op: str, args: Optional[dict[str, Any]]) -> Optional[str]:
    """The most relevant arg to show after the verb — the note, query, or
    memory task. Mirrors the app's ``detailFromArgs``."""
    if not isinstance(args, dict):
        return None
    for k in _NOTE_PATH_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return _note_title(v)
    for k in _QUERY_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return f"“{_truncate(v)}”"
    if op == "update_user_memory":
        v = args.get("task") or args.get("instruction")
        if isinstance(v, str) and v.strip():
            return _truncate(v)
    return None


def _title_case(name: str) -> str:
    words = re.sub(r"[_-]+", " ", name).strip()
    if not words:
        return "Tool"
    return words.title()


def tool_label(evt: ToolStatusEvent) -> tuple[str, Optional[str], str]:
    """Return ``(title, detail, emoji)`` for a tool-status event.

    ``title`` is the user-facing verb ("Recalling", "Running command"),
    ``detail`` the optional note / query it acts on, ``emoji`` the icon.
    """
    if is_memory_tool(evt):
        op = _memory_op(evt.tool)
        title, emoji = _MEMORY_VERBS.get(op, ("Memory", "🧠"))
        return title, _detail(op, evt.tool_args), emoji
    lower = evt.tool.lower()
    for rx, title, emoji in _GENERAL_VERBS:
        if rx.search(lower):
            return title, _detail("", evt.tool_args), emoji
    base = _title_case(evt.tool)
    if evt.server:
        base = f"{base} · {_title_case(evt.server)}"
    return base, None, "🔧"


def status_line(evt: ToolStatusEvent) -> str:
    """Friendly text for the editable "is typing" status line.

    running → the bare verb ("Recalling Brain"); done → "✓ …"; error → "✗ …".
    """
    title, detail, _emoji = tool_label(evt)
    label = f"{title} {detail}" if detail else title
    if evt.status == "running":
        return label
    if evt.status == "error":
        return f"✗ {label} failed: {evt.error or 'unknown error'}"
    return f"✓ {label}"


def message_line(evt: ToolStatusEvent) -> str:
    """Friendly text for a standalone live chat message (live mode).

    running → "🧠 Memorizing — Brain"; error → "⚠️ … failed: …".
    """
    title, detail, emoji = tool_label(evt)
    label = f"{title} — {detail}" if detail else title
    if evt.status == "error":
        return f"⚠️ {label} failed: {evt.error or 'unknown error'}"
    return f"{emoji} {label}"


__all__ = ["tool_label", "status_line", "message_line", "is_memory_tool"]
