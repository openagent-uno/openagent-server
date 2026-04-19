"""Backfill missing builtin MCP rows on boot.

The ``mcps`` SQLite table is the sole source of truth for configured
MCP servers. This module exposes a single helper — ``ensure_builtin_mcps``
— that runs every boot and adds any ``BUILTIN_MCP_SPECS`` entry that
doesn't have a row yet. Users who want a builtin off keep the row and
flip ``enabled=0``; we only *add* missing rows, we never touch existing
ones.
"""

from __future__ import annotations

import logging

from openagent.memory.db import MemoryDB

logger = logging.getLogger(__name__)


async def ensure_builtin_mcps(db: MemoryDB) -> int:
    """Make sure every ``BUILTIN_MCP_SPECS`` entry has a row.

    Runs every boot. Purposes:

      1. **Forward compat**: when a new builtin lands in a future
         release, existing installs pick it up on the next boot without
         needing a yaml edit or manual DB touch.
      2. **Safety net**: if someone manually deletes a builtin row
         (bypassing the API guards), it's reinstated here with
         ``enabled=1``. Existing rows — including disabled ones — are
         untouched.

    Returns the number of rows added this boot (zero is the steady
    state).
    """
    from openagent.mcp.builtins import BUILTIN_MCP_SPECS

    existing = {row["name"] for row in await db.list_mcps()}
    added = 0
    for builtin_name in BUILTIN_MCP_SPECS:
        if builtin_name in existing:
            continue
        await db.upsert_mcp(
            builtin_name,
            kind="default",
            builtin_name=builtin_name,
            enabled=True,
            source="ensure-builtin",
        )
        added += 1
    if added:
        logger.info("bootstrap: auto-seeded %d missing builtin MCP row(s)", added)
    return added
