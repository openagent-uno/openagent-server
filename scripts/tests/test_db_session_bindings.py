"""Session runtime bindings — CRUD on ``session_bindings`` + fallthrough
to ``sdk_sessions`` for claude-cli sessions.

Under v0.12 the ``session_bindings.provider`` column was renamed to
``framework`` (it always carried ``"agno"`` / ``"claude-cli"``). Tests
talk to the ``framework=`` kwarg on ``set_session_binding``.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


@test("db_session_bindings", "set + get roundtrip for agno")
async def t_agno_binding(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"sb-agno-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        assert await db.get_session_binding("sess-a") is None
        await db.set_session_binding("sess-a", "agno")
        assert await db.get_session_binding("sess-a") == "agno"
        # Upsert must overwrite
        await db.set_session_binding("sess-a", "agno")
        await db.delete_session_binding("sess-a")
        assert await db.get_session_binding("sess-a") is None
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("db_session_bindings", "claude-cli binding resolves via sdk_sessions")
async def t_claude_cli_via_sdk_sessions(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"sb-cli-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        # Write only to sdk_sessions — mimics ClaudeCLI._persist_sdk_session.
        await db.set_sdk_session("sess-b", "sdk-uuid-xyz", provider="claude-cli")
        side = await db.get_session_binding("sess-b")
        assert side == "claude-cli", side
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("db_session_bindings", "invalid framework is rejected at the CHECK constraint")
async def t_invalid_framework(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"sb-invalid-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        raised = False
        try:
            await db.set_session_binding("sess-x", "madeup-framework")
        except ValueError:
            raised = True
        assert raised, "set_session_binding must reject invalid framework values"
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
