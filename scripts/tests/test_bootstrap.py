"""Bootstrap — DEFAULT_MCPS seeding into the ``mcps`` table.

Regression test for the bug where ``ensure_builtin_mcps`` only iterated
``BUILTIN_MCP_SPECS`` and never seeded the ``npx``-based defaults
(``vault``, ``filesystem``). Fresh ``--agent-dir`` installs came up
without the vault MCP, the agent reported "il vault MCP non è
disponibile in questa installazione", and memory writes leaked to the
filesystem instead of the OpenAgent vault.
"""
from __future__ import annotations

from unittest.mock import patch

from ._framework import TestContext, test


@test("bootstrap", "ensure_builtin_mcps seeds vault + filesystem (npx defaults)")
async def t_seeds_npx_defaults(ctx: TestContext) -> None:
    from openagent.memory.bootstrap import ensure_builtin_mcps
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        added = await ensure_builtin_mcps(db)
        assert added > 0, "fresh DB should have rows seeded"

        rows = await db.list_mcps()
        names = {row["name"] for row in rows}

        # The two npx-based defaults that were missing before the fix.
        assert "vault" in names, (
            "vault MCP must be seeded so memory writes route through "
            "vault_* tools (not raw filesystem)"
        )
        assert "filesystem" in names, "filesystem MCP must be seeded"

        # Vault row shape: command + args, no builtin_name.
        vault = next(r for r in rows if r["name"] == "vault")
        assert vault["command"], "vault row must store the npx argv"
        assert vault["command"][0] == "npx"
        assert not vault.get("builtin_name"), "vault is not a python builtin"
        assert vault["enabled"] is True
        assert vault["kind"] == "default"
    finally:
        await db.close()


@test("bootstrap", "ensure_builtin_mcps seeds every DEFAULT_MCPS entry on a fresh DB")
async def t_seeds_every_default(ctx: TestContext) -> None:
    from openagent.mcp.builtins import DEFAULT_MCPS
    from openagent.memory.bootstrap import ensure_builtin_mcps
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await ensure_builtin_mcps(db)
        rows = await db.list_mcps()
        names = {row["name"] for row in rows}

        for entry in DEFAULT_MCPS:
            expected = entry.get("name") or entry.get("builtin")
            assert expected in names, (
                f"DEFAULT_MCPS entry {expected!r} not seeded — fresh "
                f"installs would silently miss this MCP"
            )
    finally:
        await db.close()


@test("bootstrap", "ensure_builtin_mcps is idempotent (second boot adds zero)")
async def t_idempotent(ctx: TestContext) -> None:
    from openagent.memory.bootstrap import ensure_builtin_mcps
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await ensure_builtin_mcps(db)
        added_again = await ensure_builtin_mcps(db)
        assert added_again == 0, (
            "subsequent boots must not re-add or touch existing rows "
            "(would clobber user-flipped enabled=0 etc.)"
        )
    finally:
        await db.close()


@test("bootstrap", "seeded vault row flows through MCPPool.from_db to a real spec")
async def t_pool_loads_vault(ctx: TestContext) -> None:
    """End-to-end: after bootstrap, the live pool must expose a vault spec
    with an executable command. Without this guard, the bootstrap could
    write a row that ``_specs_from_db`` then rejects (e.g. missing command
    after column-shape changes), and the agent would still come up
    without the vault MCP — exactly the bug we're fixing."""
    from openagent.mcp.pool import MCPPool
    from openagent.memory.bootstrap import ensure_builtin_mcps
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await ensure_builtin_mcps(db)
        pool = await MCPPool.from_db(db, db_path=str(ctx.db_path))
        names = {spec.name for spec in pool.specs}
        assert "vault" in names, (
            "vault must surface as a real MCPPool spec after "
            "ensure_builtin_mcps + from_db; if missing, the agent "
            "still won't have vault_* tools at runtime"
        )
        vault = next(s for s in pool.specs if s.name == "vault")
        assert vault.is_stdio, "vault is a stdio MCP, not http/sse"
        assert vault.command, "vault spec must carry an argv"
        # ``_normalise_spec`` resolves the head to absolute. ``npx`` should
        # have been turned into something like /…/.nvm/…/npx.
        assert vault.command[0].endswith("npx") or vault.command[0] == "npx"
    finally:
        await db.close()


@test("bootstrap", "_resolve_subprocess_python falls back to system python when frozen")
async def t_resolve_subprocess_python_frozen(ctx: TestContext) -> None:
    """In a frozen bundle ``sys.executable`` is the openagent Click CLI,
    not a Python interpreter — calling it with a ``.py`` path makes Click
    emit a misleading ``No such command`` error. ``_resolve_subprocess_python``
    must look up a system python instead so the Chromium auto-install
    subprocess actually runs Python."""
    import openagent.mcp.builtins as builtins_mod

    with patch.object(builtins_mod, "is_frozen", return_value=False):
        # Non-frozen: must hand back the running interpreter unchanged.
        import sys as _sys
        assert builtins_mod._resolve_subprocess_python() == _sys.executable

    with patch.object(builtins_mod, "is_frozen", return_value=True):
        # Frozen + python on PATH: must hand back the system python,
        # never ``sys.executable`` (which would be the openagent binary).
        with patch.object(
            builtins_mod.shutil, "which",
            side_effect=lambda name: "/usr/bin/python3" if name == "python3" else None,
        ):
            resolved = builtins_mod._resolve_subprocess_python()
            assert resolved == "/usr/bin/python3", (
                f"frozen mode must use system python3, got {resolved!r}"
            )

        # Frozen + no system python: returns None so caller can skip cleanly
        # instead of feeding the .py path to Click as a subcommand.
        with patch.object(builtins_mod.shutil, "which", return_value=None):
            resolved = builtins_mod._resolve_subprocess_python()
            assert resolved is None, (
                f"frozen mode with no system python must return None, got {resolved!r}"
            )


@test("bootstrap", "ensure_builtin_mcps does not reset disabled rows")
async def t_preserves_disabled(ctx: TestContext) -> None:
    from openagent.memory.bootstrap import ensure_builtin_mcps
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        # First boot seeds everything as enabled.
        await ensure_builtin_mcps(db)
        # User disables vault.
        await db.set_mcp_enabled("vault", False)
        # Next boot must NOT re-enable it.
        await ensure_builtin_mcps(db)
        vault = await db.get_mcp("vault")
        assert vault is not None
        assert vault["enabled"] is False, (
            "ensure_builtin_mcps must not flip user-disabled rows back on"
        )
    finally:
        await db.close()
