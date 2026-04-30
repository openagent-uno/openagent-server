"""Gateway WebSocket protocol — message types and constants.

This is the single source of truth for the JSON protocol used between
the Gateway server and all clients (app, CLI, bridges).

Client → Server::

    {"type": "auth",    "token": "...", "client_id": "..."}
    {"type": "message", "text": "...", "session_id": "..."}
    {"type": "command", "name": "<gateway command>", "session_id": "..."}
    {"type": "ping"}

``session_id`` on a ``command`` is optional but strongly recommended for any
client that multiplexes multiple independent conversations onto a single
``client_id`` — telegram/discord/whatsapp bridges (many users on one bot)
AND the desktop app (multiple chat tabs per user). When present, the
scope-sensitive commands ``stop``, ``clear``, ``new``, ``reset`` act only
on that conversation; other users/tabs on the same ``client_id`` are left
untouched. When omitted, those commands fall back to the legacy
client-wide behaviour — useful for single-user direct ws clients and
administrative shutdowns.

Server → Client::

    {"type": "auth_ok",        "agent_name": "...", "version": "..."}
    {"type": "auth_error",     "reason": "..."}
    {"type": "status",         "text": "...",  "session_id": "..."}
    {"type": "delta",          "text": "...",  "session_id": "..."}
    {"type": "response",       "text": "...",  "session_id": "...", "attachments": [...], "model": "..."}
    {"type": "error",          "text": "..."}
    {"type": "queued",         "position": N}
    {"type": "command_result", "text": "..."}
    {"type": "pong"}
    {"type": "resource_event", "resource": "...", "action": "...", "id": "..."}
    {"type": "system_snapshot", "snapshot": {host, cpu, memory, swap, disks, network, processes, timestamp}}
    {"type": "audio_start",    "session_id": "...", "format": "mp3", "voice_id": "...", "mime": "audio/mpeg"}
    {"type": "audio_chunk",    "session_id": "...", "seq": N, "data": "<base64>"}
    {"type": "audio_end",      "session_id": "...", "total_chunks": N}

The ``audio_*`` events stream a TTS reply alongside the regular text
response: when the client sets ``input_was_voice=true`` on its inbound
``message`` (because the user just spoke), the server pipes the
LLM stream through TTS and emits ``audio_start`` → ``audio_chunk`` ×N →
``audio_end``, then a final ``response`` carrying the full text +
attachments. Clients that don't render audio can ignore ``audio_*`` —
the trailing ``response`` still has everything they need.

``delta`` frames stream tokens as they arrive from the LLM during a
text-mode turn (``input_was_voice=false``). Clients should accumulate
each delta into the in-progress assistant bubble; the trailing
``response`` is the canonical record (full text + attachments + model
meta) and replaces the streaming buffer with the final clean text.
Older clients that don't recognize ``delta`` simply ignore it and
render the final ``response`` like before — backward-compatible. The
voice-mode path emits ``audio_*`` and skips ``delta`` (TTS is the
streaming surface for voice); ``delta`` is exclusive to text-mode.

A ``resource_event`` tells subscribed clients (the desktop app's MCPs /
Tasks / Workflows / Memory screens) that a server-side resource list
changed and they should refetch. ``resource`` is one of ``"mcp"``,
``"scheduled_task"``, ``"workflow"``, ``"vault"`` or ``"config"``;
``action`` is one of ``"created"``, ``"updated"``, ``"deleted"``, or
``"changed"`` (the coarse hint used when we know *something* in that
namespace moved but not exactly what — e.g. an MCP-tool driven write
from a chat turn). ``id`` is optional.
"""

# Message type constants
AUTH = "auth"
AUTH_OK = "auth_ok"
AUTH_ERROR = "auth_error"
MESSAGE = "message"
COMMAND = "command"
COMMAND_RESULT = "command_result"
STATUS = "status"
# Streaming token frame for the text-mode chat path. Emitted between
# MESSAGE (in) and RESPONSE (out). See module docstring.
DELTA = "delta"
RESPONSE = "response"
ERROR = "error"
QUEUED = "queued"
PING = "ping"
PONG = "pong"
RESOURCE_EVENT = "resource_event"
# Periodic host telemetry push (CPU/RAM/disk/network/processes). One
# emission every ~2s when at least one client is connected. See
# ``api/system.py``.
SYSTEM_SNAPSHOT = "system_snapshot"
# Streaming TTS events for voice-mode replies. See module docstring.
AUDIO_START = "audio_start"
AUDIO_CHUNK = "audio_chunk"
AUDIO_END = "audio_end"

from openagent.gateway.commands import COMMANDS
