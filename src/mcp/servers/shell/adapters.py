"""Provider adapters for the in-process shell MCP.

Both Claude Agent SDK and the native runtime accept in-process tool
registration, so the shell tools live as plain async functions in
``handlers.py`` and we wrap them once per provider with the native
decorator here.
"""
from __future__ import annotations

import contextvars
import json
from typing import Any

from openagent_host_tools.builtins._util import json_result
from openagent_host_tools.types import HostError, tool_error_result
from src.mcp.servers.shell import handlers

_session_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "openagent_shell_session_id", default=None,
)


def set_session_context(session_id: str | None):
    """Install ``session_id`` into the contextvar and return the token."""
    return _session_context.set(session_id)


def reset_session_context(token) -> None:
    _session_context.reset(token)


def current_session_id() -> str | None:
    return _session_context.get()


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


async def _wire(awaitable) -> dict[str, Any]:
    """Normalize server shell handlers to the shared MCP result boundary."""

    try:
        value = await awaitable
    except HostError as exc:
        return tool_error_result(exc).to_wire()
    except ValueError as exc:
        return tool_error_result(HostError("invalid_arguments", str(exc))).to_wire()
    except RuntimeError as exc:
        return tool_error_result(HostError("shell_error", str(exc))).to_wire()
    return json_result(value).to_wire()


# ── Native runtime ──────────────────────────────────────────────────

def build_runtime_toolkit() -> Any:
    """Return a ``Toolkit`` wrapping the six shell tools.

    The Toolkit pattern expects plain async callables; the runtime
    introspects signatures to build the tool schema. We re-export the
    handlers directly (same names — match existing prompt conventions).
    """
    from openagent_host_tools.builtins import ShellServer
    from src.mcp._runtime import Toolkit

    async def shell_exec(
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        run_in_background: bool = False,
        stdin: str | None = None,
        description: str | None = None,
    ) -> dict:
        """Execute a shell command. Returns foreground output, or a shell_id when run_in_background=True."""
        return await _wire(
            handlers.shell_exec(
                command=command, cwd=cwd, env=env, timeout=timeout,
                run_in_background=run_in_background, stdin=stdin, description=description,
                session_id=None,  # runtime tools don't receive session_id directly; see adapter wiring in pool.
            )
        )

    async def shell_output(
        shell_id: str, filter: str | None = None, since_last: bool = True,
    ) -> dict:
        """Read new output from a background shell since the last call."""
        return await _wire(
            handlers.shell_output(shell_id=shell_id, filter=filter, since_last=since_last)
        )

    async def shell_input(
        shell_id: str, text: str, press_enter: bool = True,
    ) -> dict:
        """Write text to a running background shell's stdin."""
        return await _wire(
            handlers.shell_input(shell_id=shell_id, text=text, press_enter=press_enter)
        )

    async def shell_kill(shell_id: str, signal: str = "TERM") -> dict:
        """Kill a background shell by id."""
        return await _wire(handlers.shell_kill(shell_id=shell_id, signal=signal))

    async def shell_list(session_id: str | None = None) -> dict:
        """List active and recently-completed background shells."""
        async def shaped() -> dict[str, Any]:
            return {"shells": await handlers.shell_list(session_id=session_id)}

        return await _wire(shaped())

    async def shell_which(command: str) -> dict:
        """Check whether a command is available on PATH."""
        return await _wire(handlers.shell_which(command=command))

    entrypoints = {
        function.__name__: function
        for function in (
            shell_exec,
            shell_output,
            shell_input,
            shell_kill,
            shell_list,
            shell_which,
        )
    }
    toolkit = Toolkit(
        name="shell",
        tools=list(entrypoints.values()),
        instructions=ShellServer.manifest.instructions,
    )
    for manifest in ShellServer.manifest.tools:
        runtime = toolkit.async_functions[manifest.name]
        runtime.description = manifest.description
        runtime.parameters = manifest.input_schema
        runtime.classification = manifest.classification.value
    return toolkit
