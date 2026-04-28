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
    {"type": "response",       "text": "...",  "session_id": "...", "attachments": [...], "model": "..."}
    {"type": "error",          "text": "..."}
    {"type": "queued",         "position": N}
    {"type": "command_result", "text": "..."}
    {"type": "pong"}
    {"type": "resource_event", "resource": "...", "action": "...", "id": "..."}

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
RESPONSE = "response"
ERROR = "error"
QUEUED = "queued"
PING = "ping"
PONG = "pong"
RESOURCE_EVENT = "resource_event"

from openagent.gateway.commands import COMMANDS
