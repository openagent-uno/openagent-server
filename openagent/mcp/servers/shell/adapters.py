"""Provider adapters for the in-process shell MCP.

Both Claude Agent SDK and Agno accept in-process tool registration, so
the shell tools live as plain async functions in ``handlers.py`` and
we wrap them once per provider with the native decorator here.
"""
from __future__ import annotations

import logging
from typing import Any

from openagent.mcp.servers.shell import handlers

logger = logging.getLogger(__name__)


# ── Claude Agent SDK ────────────────────────────────────────────────

def build_sdk_server() -> Any:
    """Return a ``McpSdkServerConfig`` wrapping the six shell tools."""
    from claude_agent_sdk import create_sdk_mcp_server, tool as sdk_tool

    @sdk_tool(
        "shell_exec",
        "Execute a shell command. Returns foreground output, or if "
        "run_in_background=true returns a shell_id.",
        {
            "command": str,
            "cwd": (str, None),
            "env": (dict, None),
            "timeout": (int, None),
            "run_in_background": (bool, None),
            "stdin": (str, None),
            "description": (str, None),
        },
    )
    async def _shell_exec(args: dict) -> dict:
        return {"content": [{"type": "text", "text": _json_dump(await handlers.shell_exec(
            command=args["command"],
            cwd=args.get("cwd"),
            env=args.get("env"),
            timeout=args.get("timeout"),
            run_in_background=args.get("run_in_background", False),
            stdin=args.get("stdin"),
            description=args.get("description"),
            session_id=args.get("_session_id"),
        ))}]}

    @sdk_tool(
        "shell_output",
        "Read new output from a background shell since the last call.",
        {"shell_id": str, "filter": (str, None), "since_last": (bool, None)},
    )
    async def _shell_output(args: dict) -> dict:
        return {"content": [{"type": "text", "text": _json_dump(await handlers.shell_output(
            shell_id=args["shell_id"],
            filter=args.get("filter"),
            since_last=args.get("since_last", True),
        ))}]}

    @sdk_tool(
        "shell_input",
        "Write text to a running background shell's stdin.",
        {"shell_id": str, "text": str, "press_enter": (bool, None)},
    )
    async def _shell_input(args: dict) -> dict:
        return {"content": [{"type": "text", "text": _json_dump(await handlers.shell_input(
            shell_id=args["shell_id"],
            text=args["text"],
            press_enter=args.get("press_enter", True),
        ))}]}

    @sdk_tool(
        "shell_kill",
        "Kill a background shell by id (TERM, INT, or KILL).",
        {"shell_id": str, "signal": (str, None)},
    )
    async def _shell_kill(args: dict) -> dict:
        return {"content": [{"type": "text", "text": _json_dump(await handlers.shell_kill(
            shell_id=args["shell_id"],
            signal=args.get("signal", "TERM"),
        ))}]}

    @sdk_tool(
        "shell_list",
        "List active and recently-completed background shells.",
        {"session_id": (str, None)},
    )
    async def _shell_list(args: dict) -> dict:
        return {"content": [{"type": "text", "text": _json_dump(await handlers.shell_list(
            session_id=args.get("session_id") or args.get("_session_id"),
        ))}]}

    @sdk_tool(
        "shell_which",
        "Check whether a command is available on PATH.",
        {"command": str},
    )
    async def _shell_which(args: dict) -> dict:
        return {"content": [{"type": "text", "text": _json_dump(await handlers.shell_which(
            command=args["command"],
        ))}]}

    return create_sdk_mcp_server(
        "shell",
        tools=[_shell_exec, _shell_output, _shell_input, _shell_kill, _shell_list, _shell_which],
    )


def _json_dump(value: Any) -> str:
    import json
    return json.dumps(value, indent=2, default=str)


# ── Agno ────────────────────────────────────────────────────────────

def build_agno_toolkit() -> Any:
    """Return an Agno ``Toolkit`` wrapping the six shell tools.

    The Toolkit pattern expects plain async callables; Agno introspects
    signatures to build the tool schema. We re-export the handlers
    directly (same names — match existing prompt conventions).
    """
    from agno.tools import Toolkit

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
        return await handlers.shell_exec(
            command=command, cwd=cwd, env=env, timeout=timeout,
            run_in_background=run_in_background, stdin=stdin, description=description,
            session_id=None,  # Agno tools don't receive session_id directly; see adapter wiring in pool.
        )

    async def shell_output(
        shell_id: str, filter: str | None = None, since_last: bool = True,
    ) -> dict:
        """Read new output from a background shell since the last call."""
        return await handlers.shell_output(
            shell_id=shell_id, filter=filter, since_last=since_last,
        )

    async def shell_input(
        shell_id: str, text: str, press_enter: bool = True,
    ) -> dict:
        """Write text to a running background shell's stdin."""
        return await handlers.shell_input(
            shell_id=shell_id, text=text, press_enter=press_enter,
        )

    async def shell_kill(shell_id: str, signal: str = "TERM") -> dict:
        """Kill a background shell by id."""
        return await handlers.shell_kill(shell_id=shell_id, signal=signal)

    async def shell_list(session_id: str | None = None) -> list:
        """List active and recently-completed background shells."""
        return await handlers.shell_list(session_id=session_id)

    async def shell_which(command: str) -> dict:
        """Check whether a command is available on PATH."""
        return await handlers.shell_which(command=command)

    return Toolkit(
        name="shell",
        tools=[shell_exec, shell_output, shell_input, shell_kill, shell_list, shell_which],
    )
