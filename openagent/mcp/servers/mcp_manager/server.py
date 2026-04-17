"""MCP-manager MCP server.

Exposes the ``mcps`` table over MCP so the agent can inspect, add, update,
enable/disable, and remove its own MCP servers at runtime. Writes land
directly in SQLite (same pattern as the scheduler MCP) and the gateway
polls ``MAX(updated_at)`` per message to trigger ``MCPPool.reload()`` —
so changes take effect on the very next turn without a process restart.

Transport: stdio, launched as a subprocess by MCPPool.
Storage: the shared OpenAgent SQLite DB. Path from ``OPENAGENT_DB_PATH``
(injected by MCPPool), falling back to ``./openagent.db`` so the server
still works when run standalone.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import aiosqlite
from mcp.server.fastmcp import FastMCP
from openagent.memory.db import MemoryDB
from openagent.mcp.servers._common import SharedConnection, ensure_row_exists, run_stdio

logger = logging.getLogger(__name__)

_shared = SharedConnection("mcp-manager")


async def _get_conn() -> aiosqlite.Connection:
    return await _shared.get()


# Row decoding is shared with MemoryDB so there's one source of truth for
# the JSON column shapes.
_row_to_dict = MemoryDB._row_to_mcp


async def _touch_name(conn: aiosqlite.Connection, name: str) -> str:
    return await ensure_row_exists(conn, "mcps", "name", name)


mcp = FastMCP("mcp-manager")


@mcp.tool()
async def list_mcps(enabled_only: bool = False) -> list[dict[str, Any]]:
    """List every MCP server currently configured.

    Each row has ``name``, ``kind`` (``default``/``builtin``/``custom``),
    optional ``builtin_name``, ``command`` (argv list), ``args``,
    ``url``, ``env``, ``enabled``, ``source``, and timestamps.
    """
    conn = await _get_conn()
    if enabled_only:
        cursor = await conn.execute(
            "SELECT * FROM mcps WHERE enabled = 1 ORDER BY name ASC"
        )
    else:
        cursor = await conn.execute("SELECT * FROM mcps ORDER BY name ASC")
    rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]


@mcp.tool()
async def get_mcp(name: str) -> dict[str, Any]:
    """Fetch one MCP row by name."""
    conn = await _get_conn()
    cursor = await conn.execute("SELECT * FROM mcps WHERE name = ?", (name,))
    row = await cursor.fetchone()
    if not row:
        raise ValueError(f"No MCP server named {name!r}")
    return _row_to_dict(row)


@mcp.tool()
async def add_custom_mcp(
    name: str,
    command: list[str] | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    env: dict[str, str] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Register a custom MCP server (stdio or HTTP).

    Pass either ``command`` (argv list for stdio) OR ``url`` (for
    HTTP/SSE servers). ``args`` are appended to the command verbatim.
    Takes effect on the next message (pool reload is automatic).
    """
    if not name or not name.strip():
        raise ValueError("name is required")
    if not command and not url:
        raise ValueError("either command (argv list) or url is required")
    conn = await _get_conn()
    now = time.time()
    cmd_text = json.dumps(list(command)) if command else None
    await conn.execute(
        "INSERT INTO mcps (name, kind, command, args_json, url, env_json, "
        "enabled, source, created_at, updated_at) "
        "VALUES (?, 'custom', ?, ?, ?, ?, ?, 'mcp-manager', ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        "kind = 'custom', command = excluded.command, args_json = excluded.args_json, "
        "url = excluded.url, env_json = excluded.env_json, "
        "enabled = excluded.enabled, updated_at = excluded.updated_at",
        (
            name,
            cmd_text,
            json.dumps(list(args or [])),
            url,
            json.dumps(dict(env or {})),
            1 if enabled else 0,
            now,
            now,
        ),
    )
    await conn.commit()
    return await get_mcp(name)


@mcp.tool()
async def add_builtin_mcp(
    builtin_name: str,
    env: dict[str, str] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Enable one of OpenAgent's built-in MCP servers.

    ``builtin_name`` must be one of the keys in ``BUILTIN_MCP_SPECS``
    (vault, filesystem, editor, web-search, shell, computer-control,
    chrome-devtools, messaging, scheduler, mcp-manager, model-manager).
    The row is stored under the same ``name`` as ``builtin_name``.
    """
    # Local import: BUILTIN_MCP_SPECS lives in openagent.mcp.builtins which
    # pulls in optional deps. Importing inside the tool body keeps server
    # startup lean and avoids circular-import pitfalls.
    from openagent.mcp.builtins import BUILTIN_MCP_SPECS

    if builtin_name not in BUILTIN_MCP_SPECS:
        available = ", ".join(sorted(BUILTIN_MCP_SPECS.keys()))
        raise ValueError(f"Unknown builtin {builtin_name!r}. Available: {available}")

    conn = await _get_conn()
    now = time.time()
    await conn.execute(
        "INSERT INTO mcps (name, kind, builtin_name, env_json, enabled, "
        "source, created_at, updated_at) "
        "VALUES (?, 'builtin', ?, ?, ?, 'mcp-manager', ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        "kind = 'builtin', builtin_name = excluded.builtin_name, "
        "env_json = excluded.env_json, enabled = excluded.enabled, "
        "updated_at = excluded.updated_at",
        (
            builtin_name,
            builtin_name,
            json.dumps(dict(env or {})),
            1 if enabled else 0,
            now,
            now,
        ),
    )
    await conn.commit()
    return await get_mcp(builtin_name)


@mcp.tool()
async def update_mcp(
    name: str,
    command: list[str] | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    env: dict[str, str] | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Partially update a custom MCP row.

    Only the fields you pass are changed. For ``builtin``/``default``
    rows, ``env`` and ``enabled`` can be patched but ``command``/``url``
    should not be (they come from the built-in spec); passing them
    converts the row to ``kind='custom'`` so the change sticks.
    """
    conn = await _get_conn()
    await _touch_name(conn, name)

    updates: dict[str, Any] = {}
    convert_to_custom = False
    if command is not None:
        updates["command"] = json.dumps(list(command)) if command else None
        convert_to_custom = True
    if args is not None:
        updates["args_json"] = json.dumps(list(args))
    if url is not None:
        updates["url"] = url or None
        convert_to_custom = True
    if env is not None:
        updates["env_json"] = json.dumps(dict(env))
    if enabled is not None:
        updates["enabled"] = 1 if enabled else 0
    if not updates:
        raise ValueError("No fields to update")

    if convert_to_custom:
        updates["kind"] = "custom"
    updates["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [name]
    await conn.execute(f"UPDATE mcps SET {set_clause} WHERE name = ?", values)
    await conn.commit()
    return await get_mcp(name)


@mcp.tool()
async def enable_mcp(name: str) -> dict[str, Any]:
    """Enable one MCP server (takes effect on next message)."""
    conn = await _get_conn()
    await _touch_name(conn, name)
    await conn.execute(
        "UPDATE mcps SET enabled = 1, updated_at = ? WHERE name = ?",
        (time.time(), name),
    )
    await conn.commit()
    return await get_mcp(name)


@mcp.tool()
async def disable_mcp(name: str) -> dict[str, Any]:
    """Disable one MCP server (row preserved for re-enable later)."""
    conn = await _get_conn()
    await _touch_name(conn, name)
    await conn.execute(
        "UPDATE mcps SET enabled = 0, updated_at = ? WHERE name = ?",
        (time.time(), name),
    )
    await conn.commit()
    return await get_mcp(name)


@mcp.tool()
async def remove_mcp(name: str) -> dict[str, Any]:
    """Remove an MCP server permanently.

    Prefer ``disable_mcp`` if you might want it back later. This tool
    cannot remove ``mcp-manager`` itself — that would leave the agent
    with no way to add anything back.
    """
    if name == "mcp-manager":
        raise ValueError(
            "Refusing to remove mcp-manager — disable_mcp instead "
            "if you really want to turn it off."
        )
    conn = await _get_conn()
    await _touch_name(conn, name)
    await conn.execute("DELETE FROM mcps WHERE name = ?", (name,))
    await conn.commit()
    return {"removed": True, "name": name}


@mcp.tool()
async def list_builtin_mcps() -> list[dict[str, Any]]:
    """List every built-in MCP server OpenAgent ships with.

    Useful when adding one via ``add_builtin_mcp``. Returns the spec key
    plus whether the name is currently registered in the ``mcps`` table.
    """
    from openagent.mcp.builtins import BUILTIN_MCP_SPECS

    conn = await _get_conn()
    cursor = await conn.execute("SELECT name FROM mcps")
    configured = {row[0] for row in await cursor.fetchall()}
    return [
        {
            "builtin_name": name,
            "configured": name in configured,
            "native": bool(spec.get("native")),
            "in_process": bool(spec.get("in_process")),
            "python": bool(spec.get("python")),
        }
        for name, spec in sorted(BUILTIN_MCP_SPECS.items())
    ]


def main() -> None:
    run_stdio(mcp, loglevel_env="OPENAGENT_MCP_MANAGER_LOGLEVEL")


if __name__ == "__main__":
    main()
