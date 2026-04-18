"""One-shot import of yaml-configured MCPs into the DB.

The MCP server list used to live in ``openagent.yaml`` and was read on
every boot. Moving it to the DB lets the agent itself edit it at
runtime (via the mcp-manager MCP) and hot-reload without a restart. To
preserve existing user configs without a manual migration, on first
boot we copy the yaml entries into the DB and set a ``config_state``
flag so we never run the import twice against the same database.

Idempotency rests on two layers:

  1. The ``mcps_imported`` flag in ``config_state`` short-circuits the
     whole function after the first successful run.
  2. Every row-write uses ``upsert_*``'s ``ON CONFLICT DO UPDATE`` — so
     even if the flag is missing (someone deleted the row manually) the
     re-import just refreshes the existing rows instead of duplicating.

Subsequent yaml edits to ``mcp:`` are NOT reflected in the DB — we log
a one-line warning the first time we detect the flag is set but the
yaml still has entries. Users who want to change MCPs after first boot
go through the manager MCPs or the REST endpoints.
"""

from __future__ import annotations

import logging

from openagent.memory.db import MemoryDB

logger = logging.getLogger(__name__)


async def import_yaml_mcps_once(
    db: MemoryDB,
    mcp_config: list[dict] | None,
    include_defaults: bool,
    disable: list[str] | None,
) -> bool:
    """Seed the ``mcps`` table from yaml on first boot.

    Returns True iff rows were written. No-op (returns False) on
    subsequent boots because the ``mcps_imported`` state flag is set.

    Layout of rows written:
      - Every entry in ``DEFAULT_MCPS`` (if ``include_defaults``) becomes
        a ``kind='default'`` row. Disabled entries stay in the table with
        ``enabled=0`` so the UI can show them for re-enabling.
      - User-supplied ``mcp:`` entries become ``kind='builtin'`` when they
        use the ``{builtin: ...}`` shape, else ``kind='custom'``.

    The ``command``/``args``/``env`` fields are copied verbatim — spec
    resolution (relative→absolute paths, token injection) still happens
    in ``MCPPool._normalise_spec`` on load.
    """
    if await db.get_state("mcps_imported") == "1":
        if mcp_config or include_defaults:
            logger.debug("mcps already imported; ignoring yaml mcp: entries")
        return False

    # Deferred import: avoids a module-level cycle with openagent.mcp.
    from openagent.mcp.builtins import DEFAULT_MCPS

    disabled = set(disable or [])
    user_names = {
        str(entry.get("name") or entry.get("builtin") or "").strip()
        for entry in (mcp_config or [])
        if entry.get("name") or entry.get("builtin")
    }

    written = 0
    if include_defaults:
        for entry in DEFAULT_MCPS:
            name = str(entry.get("name") or entry.get("builtin") or "").strip()
            if not name:
                continue
            if name in user_names:
                # User override is written by the next loop; skip to avoid duplicates.
                continue
            enabled = name not in disabled
            if "builtin" in entry:
                await db.upsert_mcp(
                    name,
                    kind="default",
                    builtin_name=str(entry["builtin"]),
                    enabled=enabled,
                    source="yaml-default",
                )
            else:
                await db.upsert_mcp(
                    name,
                    kind="default",
                    command=list(entry.get("command") or []) or None,
                    args=list(entry.get("args") or []),
                    url=entry.get("url"),
                    env=dict(entry.get("env") or {}),
                    enabled=enabled,
                    source="yaml-default",
                )
            written += 1

    for entry in mcp_config or []:
        name = str(entry.get("name") or entry.get("builtin") or "").strip()
        if not name:
            continue
        enabled = name not in disabled
        if "builtin" in entry:
            await db.upsert_mcp(
                name,
                kind="builtin",
                builtin_name=str(entry["builtin"]),
                env=dict(entry.get("env") or {}),
                enabled=enabled,
                source="yaml",
            )
        else:
            await db.upsert_mcp(
                name,
                kind="custom",
                command=list(entry.get("command") or []) or None,
                args=list(entry.get("args") or []),
                url=entry.get("url"),
                env=dict(entry.get("env") or {}),
                headers=dict(entry.get("headers") or {}),
                oauth=bool(entry.get("oauth")),
                enabled=enabled,
                source="yaml",
            )
        written += 1

    await db.set_state("mcps_imported", "1")
    logger.info("bootstrap: imported %d MCP rows from yaml", written)
    return True


async def ensure_builtin_mcps(db: MemoryDB) -> int:
    """Make sure every ``BUILTIN_MCP_SPECS`` entry has a row.

    Unlike ``import_yaml_mcps_once``, this runs every boot (it is NOT
    guarded by ``config_state``). Purposes:

      1. **Forward compat**: when a new builtin lands in a future
         release, existing installs pick it up on the next boot without
         needing a yaml edit or manual DB touch.
      2. **Safety net**: if someone manually deletes a builtin row
         (bypassing the API guards), it's reinstated here with
         ``enabled=1``. Users who want a builtin off keep the row and
         flip ``enabled=0`` — that's preserved; we only *add* missing
         rows, we never touch existing ones.

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
