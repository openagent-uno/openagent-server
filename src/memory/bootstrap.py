"""Backfill missing default MCP rows on boot.

The ``mcps`` SQLite table is the sole source of truth for configured
MCP servers at runtime (``MCPPool.from_db`` reads only from the DB).
This module exposes a single helper — ``ensure_builtin_mcps`` — that
runs every boot and adds any ``DEFAULT_MCPS`` entry that doesn't have
a row yet. Users who want a default off keep the row and flip
``enabled=0``; we only *add* missing rows, we never touch existing ones.
"""

from __future__ import annotations

import logging

from src.memory.db import MemoryDB

logger = logging.getLogger(__name__)


async def ensure_builtin_mcps(db: MemoryDB, config: dict | None = None) -> int:
    """Make sure every ``DEFAULT_MCPS`` entry has a row in the ``mcps`` table.

    ``config`` is the agent's ``openagent.yaml`` dict. It is consulted ONLY
    for opt-in, off-by-default builtins (``config_gated_mcp_entries`` — today
    just the native Skills subsystem behind ``skills.enabled``). With no
    config, or the flags unset, nothing beyond ``DEFAULT_MCPS`` is seeded, so
    the ``mcps`` table is byte-identical to before this feature existed.

    Runs every boot. Two shapes coexist in ``DEFAULT_MCPS``:

      * ``{"builtin": <name>, ...}`` — resolves at runtime to one of the
        in-process / ``python -m`` Python servers in ``BUILTIN_MCP_SPECS``.
        Stored with ``builtin_name`` filled.
      * ``{"name": <name>, "command": [...], "args": [...]}`` — bare
        subprocess MCPs (currently ``vault`` and ``filesystem``, both
        ``npx``-launched). Stored with ``command`` / ``args`` filled and
        ``builtin_name`` left null. Without this branch, fresh
        ``--agent-dir`` installs come up without the vault MCP and the
        agent has no way to write memory through ``vault_*`` tools — it
        falls back to direct filesystem writes that bypass the OpenAgent
        vault entirely.

    Existing rows — including disabled ones — are untouched (forward-
    compat for future defaults + safety net for manual deletions).
    Returns the number of rows added this boot (zero is steady state).
    """
    from src.mcp.builtins import DEFAULT_MCPS, config_gated_mcp_entries

    rows = await db.list_mcps()

    # Migration: early agents seeded the vault MCP as the upstream npx package
    # (``@bitbonsai/mcpvault``), which has no write-time quality validation.
    # It now ships as a vendored, validated built-in (src/mcp/servers/vault).
    # Convert that one specific legacy row in place — matched by its npx
    # command so a user's own custom vault MCP is never touched — preserving
    # whether it was enabled. Built-in resolution then injects the validation
    # env (see resolve_default_entry).
    migrated = 0
    for row in rows:
        if row["name"] != "vault":
            continue
        builtin_name = row["builtin_name"] if "builtin_name" in row.keys() else None
        command = row["command"] if "command" in row.keys() else None
        if not builtin_name and command and "mcpvault" in str(command):
            enabled = bool(row["enabled"]) if "enabled" in row.keys() else True
            await db.upsert_mcp(
                "vault",
                kind="default",
                builtin_name="vault",
                enabled=enabled,
                source="migrate-vendored-vault",
            )
            migrated += 1
            logger.info("bootstrap: migrated the vault MCP to the vendored "
                        "validated built-in (was npx @bitbonsai/mcpvault)")
        break

    # Migration: the subprocess ``agent-bridge`` shim is replaced by the
    # in-process ``agent-federation`` builtin (same tools, no /api/peers
    # relay). Rename any persisted agent-bridge row to agent-federation,
    # preserving its enabled flag, and drop the stale row.
    for row in rows:
        if row["name"] == "agent-bridge":
            enabled = bool(row["enabled"]) if "enabled" in row.keys() else True
            await db.upsert_mcp(
                "agent-federation", kind="default",
                builtin_name="agent-federation", enabled=enabled,
                source="migrate-agent-federation",
            )
            await db.delete_mcp("agent-bridge")
            migrated += 1
            logger.info("bootstrap: migrated agent-bridge -> agent-federation MCP row")
            break

    existing = {row["name"] for row in (rows if not migrated else await db.list_mcps())}
    added = 0
    for entry in [*DEFAULT_MCPS, *config_gated_mcp_entries(config)]:
        if "builtin" in entry:
            name = entry["builtin"]
            if name in existing:
                continue
            await db.upsert_mcp(
                name,
                kind="default",
                builtin_name=name,
                enabled=True,
                source="ensure-builtin",
            )
        else:
            name = (entry.get("name") or "").strip()
            if not name or name in existing:
                continue
            await db.upsert_mcp(
                name,
                kind="default",
                command=entry.get("command"),
                args=entry.get("args") or [],
                env=entry.get("env"),
                enabled=True,
                source="ensure-builtin",
            )
        added += 1
    if added:
        logger.info("bootstrap: auto-seeded %d missing default MCP row(s)", added)
    return added
