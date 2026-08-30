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
                              "coalesce_window_ms": N, "client_kind": "...",
                              "client_capabilities": {"attachments": true,
                                "ordered_parts": true, "inline_ui": true}}
    {"type": "session_close","session_id": "..."}
    {"type": "text_final",   "session_id": "...", "text": "...", "source": "user_typed|stt|system"}
    {"type": "audio_chunk_in","session_id": "...", "data": "<base64>",
                              "encoding": "pcm16|webm|...", "sample_rate": N,
                              "end_of_speech": false}
    {"type": "audio_end_in", "session_id": "..."}
    {"type": "video_frame",  "session_id": "...", "stream": "webcam|screen|...", "data": "<base64>"}
    {"type": "attachment",   "session_id": "...", "kind": "image|file|voice|video", "path": "..."}
    {"type": "interrupt",    "session_id": "...", "reason": "..."}

The privileged client-machine tool plane uses a separate authenticated
``/ws/capabilities`` socket and protocol ``client-capabilities/1``.  It never
shares chat frames and cannot be opened by HTTP-token or federated-agent auth::

    {"type":"capability_hello", "protocol":"client-capabilities/1",
     "client_instance_id":"...", "generation":1, "device_label":"...",
     "servers":[...]}
    {"type":"capability_catalog_update", "generation":1, "servers":[...]}
    {"type":"client_tool_result", "call_id":"...", "generation":1,
     "result":{"content":[...], "structuredContent":{...}, "isError":false}}
    {"type":"client_tool_event", "generation":1,
     "event":{"type":"shell_completed", "server":"shell",
              "shell_id":"...", "status":"exited", "exit_code":0}}
    {"type":"client_artifact_chunk", "call_id":"...", "generation":1,
     "transfer_id":"...", "seq":0, "data":"<base64>", "eof":true,
     "size":123, "sha256":"...", "mime_type":"image/png"}
    {"type":"capability_heartbeat", "generation":1, "ts_ms":123}

Server → capability host::

    {"type":"capability_hello_ack", "protocol":"client-capabilities/1", ...}
    {"type":"client_tool_call", "call_id":"...", "generation":1,
     "server":"filesystem", "tool":"read_file", "args":{},
     "account_id":"<trusted cert network>", "arguments_sha256":"...", ...}
    {"type":"client_tool_cancel", "call_id":"...", "generation":1, ...}
    {"type":"client_tool_event_ack", "generation":1,
     "shell_id":"...", "accepted":true, "duplicate":false}
    {"type":"capability_heartbeat_ack", "generation":1, "ts_ms":123}

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
    {"type": "response",       "text": "...",  "session_id": "...", "attachments": [...],
                                  "parts": [...], "model": "..."}
    {"type": "live_state",     "session_id": "...", "active": true, "frames": [...]}
    {"type": "audio_start",    "session_id": "...", "format": "mp3", "voice_id": "...", "mime": "audio/mpeg"}
    {"type": "audio_chunk",    "session_id": "...", "seq": N, "data": "<base64>"}
    {"type": "audio_end",      "session_id": "...", "total_chunks": N}
    {"type": "turn_complete",  "session_id": "..."}
    {"type": "error",          "text": "..."}
    {"type": "command_result", "text": "...", "picker"?: {...}, "context"?: {...}}
    {"type": "context_report", "session_id": "...", "report": {...}}
    {"type": "pong"}
    {"type": "resource_event", "resource": "...", "action": "...", "id": "..."}
    {"type": "agent_identity_changed", "name": "...", "revision": "..."}
    {"type": "system_snapshot", "snapshot": {host, cpu, memory, swap, disks, network, processes, timestamp}}

The gateway also uses one private WebSocket close code.  When a freshly
authenticated transport takes over the same device identity, the superseded
socket is closed with ``WS_CLOSE_CONNECTION_REPLACED_CODE`` and
``WS_CLOSE_CONNECTION_REPLACED_REASON``.  Clients must not reconnect that
specific socket: doing so would replace the fresh transport in turn and create
an endless reconnect/replacement loop.

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

When a client reattaches while a turn is still running, the gateway may
send ``live_state`` snapshots. ``frames`` is an ordered replay of ordinary
stream frames (``text_final`` / ``seed`` / ``status`` / ``delta`` /
``response`` / …) for the not-yet-persisted transcript tail. Clients should
replace their live tail from this snapshot, then continue processing normal
frames. Completed turns are still hydrated from ``GET /api/sessions/{id}/runs``.

A ``command_result`` may carry an optional ``picker`` when the command
offers a list to choose from (today only ``/model`` with no argument). The
shape is::

    {"command": "model",
     "prompt": "Pick a model for this conversation:",
     "options": [{"label": "...", "value": "<runtime_id>",
                  "subtitle"?: "...", "active"?: true|false}, ...]}

``text`` always fully describes the result on its own, so clients that
don't understand ``picker`` degrade gracefully. Thin bridges render the
options as native buttons / select menus; on selection they re-issue the
command with ``arg=<value>`` (a ``value`` of ``"default"`` clears the pin).
Rich clients (desktop app, CLI) ignore this and build their own picker
from ``GET /api/models`` + ``GET /api/commands`` (``arg_source``).

A ``command_result`` for ``/context`` additionally carries an optional
``context`` object — the Claude-Code-style context-window composition for
the conversation (``src.core.context_report.build_context_report``): model,
``context_window``, per-section token counts/percentages, and cumulative
cost. Like ``picker`` it is purely additive: ``text`` already renders the
same breakdown as a fenced monospace block, so text-only bridges ignore
``context`` and degrade gracefully, while rich clients draw the panel from
it. The same payload is pushed unsolicited after each turn as a standalone
``context_report`` frame (so an always-visible panel updates in realtime)
and is served by ``GET /api/sessions/{id}/context`` for the initial paint.

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
# Private WebSocket close contract for two transports presenting the same
# device certificate.  4000-4999 are application-defined by RFC 6455.  Keep
# both values stable: first-party clients use the code as the authoritative
# discriminator and retain the reason for diagnostics.
WS_CLOSE_CONNECTION_REPLACED_CODE = 4009
WS_CLOSE_CONNECTION_REPLACED_REASON = "connection_replaced"

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
# Owner-controlled display identity changed.  The persona itself is never put
# on the wire: clients only need the public display name and a revision hint.
AGENT_IDENTITY_CHANGED = "agent_identity_changed"
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

# Client-machine capability plane (GET /ws/capabilities).
CAPABILITY_PROTOCOL = "client-capabilities/1"
CAPABILITY_HELLO = "capability_hello"
CAPABILITY_HELLO_ACK = "capability_hello_ack"
CAPABILITY_CATALOG_UPDATE = "capability_catalog_update"
CAPABILITY_HEARTBEAT = "capability_heartbeat"
CAPABILITY_HEARTBEAT_ACK = "capability_heartbeat_ack"
CLIENT_TOOL_CALL = "client_tool_call"
CLIENT_TOOL_RESULT = "client_tool_result"
CLIENT_TOOL_CANCEL = "client_tool_cancel"
CLIENT_TOOL_EVENT = "client_tool_event"
CLIENT_TOOL_EVENT_ACK = "client_tool_event_ack"
CLIENT_ARTIFACT_CHUNK = "client_artifact_chunk"

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
