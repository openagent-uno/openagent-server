"""mcp-manager.remove_mcp guardrails — builtins can be disabled, not deleted."""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


@test("mcp_manager_guards", "remove_mcp refuses kind='default' and kind='builtin'")
async def t_remove_rejects_builtins(ctx: TestContext) -> None:
    import openagent.mcp.servers.mcp_manager.server as mgr
    from openagent.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"mgrguard-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        # Seed one row of each kind.
        await db.upsert_mcp("shell", kind="default", builtin_name="shell",
                            enabled=True, source="yaml-default")
        await db.upsert_mcp("vault", kind="builtin", builtin_name="vault",
                            enabled=True, source="user")
        await db.upsert_mcp("custom-one", kind="custom",
                            command=["/bin/true"], enabled=True)
        await db.close()

        # The manager MCP server uses a module-level ``_shared`` connection;
        # point it at our tmp DB.
        import os
        prev = os.environ.get("OPENAGENT_DB_PATH")
        os.environ["OPENAGENT_DB_PATH"] = str(tmp)
        # Reset the singleton so it re-opens against the new path.
        mgr._shared._conn = None  # type: ignore[attr-defined]
        try:
            async def _expect_refusal(name: str) -> None:
                raised = False
                try:
                    await mgr.remove_mcp(name)  # type: ignore[attr-defined]
                except ValueError:
                    raised = True
                except Exception:  # noqa: BLE001 — FastMCP wraps some errors
                    raised = True
                assert raised, f"remove_mcp {name!r} should have raised"

            await _expect_refusal("shell")   # kind=default
            await _expect_refusal("vault")   # kind=builtin

            # kind=custom goes through
            result = await mgr.remove_mcp("custom-one")
            assert result.get("removed") is True

            # Rows still present for both builtins
            db2 = MemoryDB(str(tmp))
            await db2.connect()
            assert await db2.get_mcp("shell") is not None
            assert await db2.get_mcp("vault") is not None
            assert await db2.get_mcp("custom-one") is None
            await db2.close()
        finally:
            mgr._shared._conn = None  # type: ignore[attr-defined]
            if prev is None:
                os.environ.pop("OPENAGENT_DB_PATH", None)
            else:
                os.environ["OPENAGENT_DB_PATH"] = prev
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
