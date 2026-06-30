"""Bootstrap — DEFAULT_MCPS seeding into the ``mcps`` table.

Regression test for the bug where ``ensure_builtin_mcps`` failed to seed
the memory defaults (``vault``, ``filesystem``). Fresh ``--agent-dir``
installs came up without the vault MCP, the agent reported "il vault MCP
non è disponibile in questa installazione", and memory writes leaked to
the filesystem instead of the OpenAgent vault.

``vault`` is a vendored node builtin (``src/mcp/servers/vault``, run as
``node dist/server.js`` and seeded by ``builtin_name``); ``filesystem``
is an ``npx`` default. Both must land in the ``mcps`` table on a fresh DB
and surface as real, executable specs through ``MCPPool.from_db``.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

from ._framework import TestContext, test


def _fresh_db_path(ctx: TestContext) -> str:
    """A unique, never-seeded DB path so a ``fresh DB`` assertion holds
    regardless of suite ordering — ``ctx.db_path`` is shared and another
    test may already have bootstrapped it (``added`` would then be 0)."""
    return str(ctx.db_path.with_name(f"seed-{uuid.uuid4().hex[:8]}.db"))


@test("bootstrap", "ensure_builtin_mcps seeds vault (node builtin) + filesystem (npx)")
async def t_seeds_memory_defaults(ctx: TestContext) -> None:
    from src.memory.bootstrap import ensure_builtin_mcps
    from src.memory.db import MemoryDB

    db = MemoryDB(_fresh_db_path(ctx))
    await db.connect()
    try:
        added = await ensure_builtin_mcps(db)
        assert added > 0, "fresh DB should have rows seeded"

        rows = await db.list_mcps()
        names = {row["name"] for row in rows}

        # The two memory defaults that were missing before the fix.
        assert "vault" in names, (
            "vault MCP must be seeded so memory writes route through "
            "vault_* tools (not raw filesystem)"
        )
        assert "filesystem" in names, "filesystem MCP must be seeded"

        # Vault is the vendored node builtin: seeded by ``builtin_name`` and
        # resolved to ``node dist/server.js`` at runtime (BUILTIN_MCP_SPECS),
        # so the row stores no command of its own.
        vault = next(r for r in rows if r["name"] == "vault")
        assert vault.get("builtin_name") == "vault", "vault seeds as a builtin"
        assert not vault["command"], "builtin vault resolves its command at runtime"
        assert vault["enabled"] is True
        assert vault["kind"] == "default"

        # Filesystem stays an npx default: a concrete argv on the row.
        filesystem = next(r for r in rows if r["name"] == "filesystem")
        assert filesystem["command"], "filesystem row must store the npx argv"
        assert filesystem["command"][0] == "npx"
        assert not filesystem.get("builtin_name"), "filesystem is not a builtin"
    finally:
        await db.close()


@test("bootstrap", "ensure_builtin_mcps seeds every DEFAULT_MCPS entry on a fresh DB")
async def t_seeds_every_default(ctx: TestContext) -> None:
    from src.mcp.builtins import DEFAULT_MCPS
    from src.memory.bootstrap import ensure_builtin_mcps
    from src.memory.db import MemoryDB

    db = MemoryDB(_fresh_db_path(ctx))
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
    from src.memory.bootstrap import ensure_builtin_mcps
    from src.memory.db import MemoryDB

    db = MemoryDB(_fresh_db_path(ctx))
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
    from src.mcp.pool import MCPPool
    from src.memory.bootstrap import ensure_builtin_mcps
    from src.memory.db import MemoryDB

    db_path = _fresh_db_path(ctx)
    db = MemoryDB(db_path)
    await db.connect()
    try:
        await ensure_builtin_mcps(db)
        pool = await MCPPool.from_db(db, db_path=db_path)
        names = {spec.name for spec in pool.specs}
        assert "vault" in names, (
            "vault must surface as a real MCPPool spec after "
            "ensure_builtin_mcps + from_db; if missing, the agent "
            "still won't have vault_* tools at runtime"
        )
        vault = next(s for s in pool.specs if s.name == "vault")
        assert vault.is_stdio, "vault is a stdio MCP, not http/sse"
        assert vault.command, "vault spec must carry an argv"
        # The vendored node builtin resolves to ``node dist/server.js``.
        # ``_normalise_spec`` resolves the head to an absolute node path and
        # the script to the vendored server under src/mcp/servers/vault.
        assert vault.command[0].endswith("node") or vault.command[0] == "node"
        assert vault.command[-1].endswith("vault/dist/server.js")
    finally:
        await db.close()


@test("bootstrap", "_resolve_subprocess_python falls back to system python when frozen")
async def t_resolve_subprocess_python_frozen(ctx: TestContext) -> None:
    """In a frozen bundle ``sys.executable`` is the openagent Click CLI,
    not a Python interpreter — calling it with a ``.py`` path makes Click
    emit a misleading ``No such command`` error. ``_resolve_subprocess_python``
    must look up a system python instead so the Chromium auto-install
    subprocess actually runs Python."""
    import src.mcp.builtins as builtins_mod

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
    from src.memory.bootstrap import ensure_builtin_mcps
    from src.memory.db import MemoryDB

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


@test("bootstrap", "_migrate_legacy_agno_sessions_to_sessions renames the table in place")
async def t_migrate_agno_sessions_renamed(ctx: TestContext) -> None:
    """Regression: existing user DBs ship with the legacy ``agno_sessions``
    table. On boot, ``MemoryDB.connect()`` must rename it to ``sessions``
    so the rest of the codebase (which now queries ``sessions``) keeps
    seeing the same rows. The legacy table name must be gone afterwards.
    Idempotent — running connect() twice is a no-op for the rename.
    """
    import sqlite3
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"rename-{uuid.uuid4().hex[:8]}.db")
    try:
        # Seed a DB with the legacy schema + one row, BEFORE connect()
        # has a chance to run SCHEMA_SQL or any migration.
        conn = sqlite3.connect(str(tmp_db))
        try:
            conn.execute(
                """
                CREATE TABLE agno_sessions (
                    session_id   TEXT PRIMARY KEY,
                    session_type TEXT,
                    agent_id     TEXT,
                    team_id      TEXT,
                    workflow_id  TEXT,
                    user_id      TEXT,
                    session_data TEXT,
                    agent_data   TEXT,
                    team_data    TEXT,
                    workflow_data TEXT,
                    metadata     TEXT,
                    runs         TEXT,
                    summary      TEXT,
                    created_at   INTEGER NOT NULL,
                    updated_at   INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX idx_agno_sessions_updated "
                "ON agno_sessions(updated_at)"
            )
            conn.execute(
                "INSERT INTO agno_sessions "
                "(session_id, session_type, user_id, metadata, "
                " created_at, updated_at) "
                "VALUES (?, 'agent', 'openagent', '{\"title\":\"legacy row\"}', 100, 200)",
                ("legacy-sid",),
            )
            conn.commit()
        finally:
            conn.close()

        # Boot — should detect agno_sessions, rename to sessions, drop
        # the legacy index, then let SCHEMA_SQL recreate the new index.
        db = MemoryDB(str(tmp_db))
        await db.connect()
        try:
            conn = await db._ensure_connected()
            # Legacy table gone.
            cur = await conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='agno_sessions'"
            )
            assert await cur.fetchone() is None, (
                "agno_sessions must be gone after the rename migration"
            )
            # New table present.
            cur = await conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='sessions'"
            )
            assert await cur.fetchone() is not None, (
                "sessions table must exist after the rename migration"
            )
            # Row preserved.
            cur = await conn.execute(
                "SELECT session_id, metadata FROM sessions "
                "WHERE session_id = ?",
                ("legacy-sid",),
            )
            row = await cur.fetchone()
            assert row is not None, "legacy row was dropped during rename"
            assert row[0] == "legacy-sid"
            assert "legacy row" in (row[1] or ""), row[1]
            # Index re-created under the new name (idempotent).
            cur = await conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name='idx_sessions_updated'"
            )
            assert await cur.fetchone() is not None, (
                "idx_sessions_updated must be recreated by SCHEMA_SQL "
                "after the rename"
            )
        finally:
            await db.close()

        # Second connect() must be a no-op: agno_sessions doesn't exist
        # anymore, so the migration short-circuits and the row stays.
        db2 = MemoryDB(str(tmp_db))
        await db2.connect()
        try:
            conn = await db2._ensure_connected()
            cur = await conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
                ("legacy-sid",),
            )
            count = (await cur.fetchone())[0]
            assert count == 1, (
                f"row count after second connect must be 1, got {count}"
            )
        finally:
            await db2.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("bootstrap", "_migrate_legacy_agno_sessions_to_sessions skips when both tables exist")
async def t_migrate_agno_sessions_both_present(ctx: TestContext) -> None:
    """Defensive case: if both ``sessions`` and ``sessions`` exist
    (shouldn't happen in practice, but operator-staged DBs can land in
    that shape), the migration must leave both alone — no merge, no
    drop. We only assert that connect() doesn't blow up and both tables
    survive."""
    import sqlite3
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"both-{uuid.uuid4().hex[:8]}.db")
    try:
        conn = sqlite3.connect(str(tmp_db))
        try:
            # Create both tables with one distinguishable row each.
            for name in ("agno_sessions", "sessions"):
                conn.execute(
                    f"""
                    CREATE TABLE {name} (
                        session_id   TEXT PRIMARY KEY,
                        session_type TEXT,
                        agent_id     TEXT,
                        team_id      TEXT,
                        workflow_id  TEXT,
                        user_id      TEXT,
                        session_data TEXT,
                        agent_data   TEXT,
                        team_data    TEXT,
                        workflow_data TEXT,
                        metadata     TEXT,
                        runs         TEXT,
                        summary      TEXT,
                        created_at   INTEGER NOT NULL,
                        updated_at   INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    f"INSERT INTO {name} "
                    f"(session_id, session_type, created_at, updated_at) "
                    f"VALUES (?, 'agent', 100, 200)",
                    (f"row-from-{name}",),
                )
            conn.commit()
        finally:
            conn.close()

        db = MemoryDB(str(tmp_db))
        await db.connect()  # Must not raise.
        try:
            conn = await db._ensure_connected()
            # Both tables still exist.
            cur = await conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name IN ('agno_sessions','sessions') "
                "ORDER BY name"
            )
            names = [r[0] for r in await cur.fetchall()]
            assert names == ["agno_sessions", "sessions"], names
            # Both rows preserved.
            cur = await conn.execute(
                "SELECT COUNT(*) FROM agno_sessions WHERE session_id = ?",
                ("row-from-agno_sessions",),
            )
            assert (await cur.fetchone())[0] == 1
            cur = await conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
                ("row-from-sessions",),
            )
            assert (await cur.fetchone())[0] == 1
        finally:
            await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
