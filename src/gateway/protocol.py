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
``"scheduled_task"``, ``"workflow"``, ``"vault"``, ``"config"`` or
``"session"``; ``action`` is one of ``"created"``, ``"updated"``,
``"deleted"``, or ``"changed"`` (the coarse hint used when we know
*something* in that namespace moved but not exactly what — e.g. an
MCP-tool driven write from a chat turn). ``id`` is optional.

``resource="session"`` fires when a child session is spawned (a delegated
sub-agent, a scheduled-task firing, or a workflow AI-prompt node) so the
client adds it to the flat session list and a parent transcript's
delegation cards refresh live. ``id`` carries the new child ``session_id``;
the client refetches ``GET /api/sessions`` to pick up its metadata.

Per-message authorship: messages returned by ``GET /api/sessions/{id}/runs``
may carry an optional ``author`` object — ``{"kind": "human"|"agent",
"handle"?, "display"?}`` — distinguishing which human sent a message (so
multi-user / bridge sessions attribute correctly) from an agent-self seed
(a delegated task / scheduled mission / workflow node prompt). Inbound
``text_final`` / legacy ``message`` frames may also carry ``author`` so a
bridge can attribute each multiplexed user.
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
# Agent reasoning/thinking state (server→client), a boolean flag — NOT a
# UI string. ``{"type":"reasoning","active":true|false,"session_id":...}``.
# Clients render their own affordance (native typing / spinner / animated
# component); the server never dictates the "Thinking..." copy. The actual
# wire (de)serialisation lives in ``src/stream/wire.py``.
REASONING = "reasoning"
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

# Interactive terminals — a PTY on the host the gateway runs on, driven
# live by a client (desktop app System tab, CLI ``terminal`` command).
# Separate from the agent-facing ``shell`` MCP: this is the human "SSH
# terminal" surface. See :mod:`src.gateway.terminals`. ``data`` fields
# are base64-encoded raw bytes so binary-safe streams (UTF-8, control
# sequences, even non-text) survive the JSON transport intact.
#
# Client → Server::
#   {"type": "terminal_open",   "terminal_id": "...", "cols": N, "rows": N, "cwd": "...", "shell": "..."}
#   {"type": "terminal_input",  "terminal_id": "...", "data": "<base64>"}
#   {"type": "terminal_resize", "terminal_id": "...", "cols": N, "rows": N}
#   {"type": "terminal_signal", "terminal_id": "...", "signal": "INT|TERM|HUP|QUIT|KILL"}
#   {"type": "terminal_close",  "terminal_id": "..."}
#
# Server → Client::
#   {"type": "terminal_ready",  "terminal_id": "...", "pid": N, "shell": "...", "cols": N, "rows": N}
#   {"type": "terminal_output", "terminal_id": "...", "data": "<base64>"}
#   {"type": "terminal_exit",   "terminal_id": "...", "exit_code": N|null, "signal": "..."|null}
#   {"type": "terminal_error",  "terminal_id": "...", "error": "..."}
TERMINAL_OPEN = "terminal_open"
TERMINAL_INPUT = "terminal_input"
TERMINAL_RESIZE = "terminal_resize"
TERMINAL_SIGNAL = "terminal_signal"
TERMINAL_CLOSE = "terminal_close"
TERMINAL_READY = "terminal_ready"
TERMINAL_OUTPUT = "terminal_output"
TERMINAL_EXIT = "terminal_exit"
TERMINAL_ERROR = "terminal_error"

from src.gateway.commands import COMMANDS
