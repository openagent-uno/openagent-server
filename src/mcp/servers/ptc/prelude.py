"""The code injected next to a PTC script so it can reach the agent's tools.

Two artefacts land in the per-run tmpdir:

  * ``openagent_tools.py`` — a stdlib-only bridge MODULE (:data:`_BRIDGE_MODULE`)
    exposing ``call_tool(server, tool, args)``. It reads ``OPENAGENT_PTC_SOCKET``
    and ``OPENAGENT_PTC_TOKEN`` from the environment and does one locked,
    blocking newline-delimited-JSON round-trip over the Unix socket for each
    call. The gateway process answers on the other end (see ``handlers``).

  * the SCRIPT itself — the model's ``code`` with a one-line header prepended
    (``from openagent_tools import call_tool``) so ``call_tool`` is in scope
    WITHOUT the model having to import anything. The tmpdir is on ``PYTHONPATH``
    (and is the script's own directory), so the import resolves.

Everything here is pure stdlib — the child process is spawned with a scrubbed
environment and reaches tools ONLY through the socket, never by importing
OpenAgent or touching its credentials.
"""
from __future__ import annotations

import os

# The bridge module written as ``openagent_tools.py``. Kept stdlib-only and
# free of OpenAgent imports: it runs inside the (potentially sandboxed) child,
# which must not be able to reach anything but the RPC socket.
_BRIDGE_MODULE = r'''"""Auto-generated OpenAgent PTC bridge — do not edit.

Reaches the agent's own tools over a local RPC socket. Import is injected for
you, so a PTC script can just call ``call_tool(server, tool, args)``.
"""
import json
import os
import socket
import threading

_SOCK = os.environ.get("OPENAGENT_PTC_SOCKET")
_TOKEN = os.environ.get("OPENAGENT_PTC_TOKEN")
_LOCK = threading.Lock()


class PtcError(RuntimeError):
    """Raised when a call_tool round-trip fails (transport or tool error)."""


def call_tool(server, tool, args=None):
    """Invoke one of the agent's tools and return its JSON-coerced result.

    ``server`` / ``tool`` are the same names you would pass to
    ``tool_search_call_tool``; ``args`` is a dict of tool arguments. Raises
    ``PtcError`` on an unknown tool, a rejected call, or a transport failure.
    """
    if not _SOCK or not _TOKEN:
        raise PtcError("PTC bridge is not configured (no socket/token in env)")
    payload = json.dumps(
        {"token": _TOKEN, "server": server, "tool": tool, "args": args or {}}
    ) + "\n"
    with _LOCK:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(_SOCK)
            sock.sendall(payload.encode("utf-8"))
            buf = bytearray()
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf.extend(chunk)
        finally:
            sock.close()
    line = bytes(buf).split(b"\n", 1)[0]
    if not line:
        raise PtcError("empty response from PTC bridge")
    resp = json.loads(line.decode("utf-8"))
    if not resp.get("ok"):
        raise PtcError(str(resp.get("error") or "unknown PTC error"))
    return resp.get("result")
'''

# Prepended to the model's ``code`` so ``call_tool`` is a bare global. A blank
# line follows so a leading ``"""docstring"""`` in the user's code still parses.
_SCRIPT_HEADER = "from openagent_tools import call_tool  # injected by OpenAgent PTC\n\n"

_BRIDGE_FILENAME = "openagent_tools.py"
_SCRIPT_FILENAME = "ptc_script.py"


def render_script(user_code: str) -> str:
    """Return the full script text: the injected import header + the user code."""
    return _SCRIPT_HEADER + user_code


def write_prelude(tmpdir: str, user_code: str) -> str:
    """Write the bridge module and the script into ``tmpdir``.

    Returns the absolute path of the script to execute. The bridge module sits
    next to it (both on ``PYTHONPATH``), so ``from openagent_tools import
    call_tool`` in the header resolves.
    """
    bridge_path = os.path.join(tmpdir, _BRIDGE_FILENAME)
    with open(bridge_path, "w", encoding="utf-8") as fh:
        fh.write(_BRIDGE_MODULE)
    script_path = os.path.join(tmpdir, _SCRIPT_FILENAME)
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(render_script(user_code))
    return script_path
