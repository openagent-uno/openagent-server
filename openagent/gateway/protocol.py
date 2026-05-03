"""Gateway WebSocket protocol — message types and constants.

This is the single source of truth for the JSON protocol used between
the Gateway server and all clients (app, CLI, bridges). Every text /
voice / video / attachment message flows through the typed *stream*
protocol; the legacy one-shot ``message`` frame was retired once
bridges, the universal app, and the CLI all migrated.

Client → Server::

    {"type": "auth",         "token": "...", "client_id": "..."}
    {"type": "command",      "name": "<gateway command>", "session_id": "..."}
    {"type": "ping"}

    # Stream protocol (one long-lived StreamSession per session_id):
    {"type": "session_open", "session_id": "...", "profile": "batched|realtime",
                              "language": "...", "speak": true|false,
                              "coalesce_window_ms": N, "client_kind": "..."}
    {"type": "session_close","session_id": "..."}
    {"type": "text_final",   "session_id": "...", "text": "...", "source": "user_typed|stt|system"}
    {"type": "audio_chunk_in","session_id": "...", "data": "<base64>",
                              "encoding": "pcm16|webm|...", "sample_rate": N,
                              "end_of_speech": false}
    {"type": "audio_end_in", "session_id": "..."}
    {"type": "video_frame",  "session_id": "...", "stream": "webcam|screen|...", "data": "<base64>"}
    {"type": "attachment",   "session_id": "...", "kind": "image|file|voice|video", "path": "..."}
    {"type": "interrupt",    "session_id": "...", "reason": "..."}

``session_id`` on a ``command`` is optional but strongly recommended for any
client that multiplexes multiple independent conversations onto a single
``client_id`` — telegram/discord/whatsapp bridges (many users on one bot)
AND the desktop app (multiple chat tabs per user). When present, the
scope-sensitive commands ``stop``, ``clear``, ``new``, ``reset`` act only
on that conversation; other users/tabs on the same ``client_id`` are left
untouched.

Server → Client::

    {"type": "auth_ok",        "agent_name": "...", "version": "..."}
    {"type": "auth_error",     "reason": "..."}
    {"type": "status",         "text": "...",  "session_id": "..."}
    {"type": "delta",          "text": "...",  "session_id": "..."}
    {"type": "response",       "text": "...",  "session_id": "...", "attachments": [...], "model": "..."}
    {"type": "audio_start",    "session_id": "...", "format": "mp3", "voice_id": "...", "mime": "audio/mpeg"}
    {"type": "audio_chunk",    "session_id": "...", "seq": N, "data": "<base64>"}
    {"type": "audio_end",      "session_id": "...", "total_chunks": N}
    {"type": "turn_complete",  "session_id": "..."}
    {"type": "error",          "text": "..."}
    {"type": "command_result", "text": "..."}
    {"type": "pong"}
    {"type": "resource_event", "resource": "...", "action": "...", "id": "..."}
    {"type": "system_snapshot", "snapshot": {host, cpu, memory, swap, disks, network, processes, timestamp}}

A turn lifecycle: ``status`` (any number, "Using bash...") + ``delta``
(any number, streamed tokens) + ``response`` (one canonical text +
model + attachments) + optionally ``audio_start`` / ``audio_chunk`` ×N /
``audio_end`` (when the session has ``speak=true`` and a TTS provider
is configured) + ``turn_complete`` (terminator — clients waiting for a
single reply resolve here).

The mirror-modality rule on the server side: ``text_final`` with
``source="stt"`` always speaks the reply when TTS is configured, even
when the session was opened with ``speak=false``. That way chat-tab
typed messages stay silent but voice notes get spoken back.

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
# Legacy one-shot client→server text frame. Retired in favour of the
# stream protocol (``session_open`` + ``text_final``); the wire codec
# still maps an inbound ``message`` to a ``TextFinal`` for graceful
# degradation, but nothing in-tree emits this anymore.
MESSAGE = "message"
COMMAND = "command"
COMMAND_RESULT = "command_result"
STATUS = "status"
# Streaming token frame for text-mode replies (server→client). Emitted
# by ``StreamSession`` while the LLM streams; the trailing ``response``
# is the canonical record. See module docstring.
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

# Stream protocol — typed event vocabulary that complements the legacy
# frames above. Older clients never see these (the gateway only emits
# them when a session opens with profile="realtime"). See
# :mod:`openagent.stream.wire` for the codec.
TEXT_DELTA_IN = "text_delta"
TEXT_FINAL_IN = "text_final"
AUDIO_CHUNK_IN = "audio_chunk_in"
AUDIO_END_IN = "audio_end_in"
VIDEO_FRAME_IN = "video_frame"
ATTACHMENT_IN = "attachment"
INTERRUPT = "interrupt"
SESSION_OPEN = "session_open"
SESSION_CLOSE = "session_close"
VIDEO_FRAME_OUT = "video_frame_out"
TURN_COMPLETE = "turn_complete"

from openagent.gateway.commands import COMMANDS
