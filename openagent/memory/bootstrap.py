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

from openagent.memory.db import MemoryDB

logger = logging.getLogger(__name__)


async def ensure_builtin_mcps(db: MemoryDB) -> int:
    """Make sure every ``DEFAULT_MCPS`` entry has a row in the ``mcps`` table.

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
    from openagent.mcp.builtins import DEFAULT_MCPS

    existing = {row["name"] for row in await db.list_mcps()}
    added = 0
    for entry in DEFAULT_MCPS:
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
