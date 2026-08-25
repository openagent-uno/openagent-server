"""Every in-tree MCP subprocess must land on the agent's real database.

The bug this pins, found on a live agent 2026-08-25: ``resolve_builtin_entry``
injected ``OPENAGENT_DB_PATH`` only into a hand-kept list of builtins known to
touch the shared SQLite DB. ``media-gen`` — added later, and which resolves its
image backend by reading the model catalogue — was not on the list. It fell
back to ``"openagent.db"`` relative to the subprocess CWD, which under
PyInstaller is a scratch directory, so it CREATED an empty database there and
answered "no image backend configured" on a fleet whose catalogue declared one.

Two guards, because either one alone would have let it through: the injection
covers every builtin, and the fallback is never the CWD.
"""
from __future__ import annotations

import os

from ._framework import TestContext, test


@test("mcp_db_path", "every builtin is told where the database is")
async def t_all_builtins_get_db_path(ctx: TestContext) -> None:
    from src.mcp.builtins import BUILTIN_MCP_SPECS, resolve_default_entry

    db = "/tmp/some-agent/openagent.db"
    checked, missing = 0, []
    for name, spec in BUILTIN_MCP_SPECS.items():
        # In-process builtins live inside the main process and share its
        # MemoryDB connection: there is no subprocess env to inject into, and
        # no CWD to get lost in. Only the spawned ones are at risk.
        if spec.get("in_process"):
            continue
        resolved = resolve_default_entry({"builtin": name}, db_path=db)
        if resolved is None:
            continue  # not installable here (no node, etc.) — nothing to assert
        checked += 1
        env = resolved.get("env") or {}
        if env.get("OPENAGENT_DB_PATH") != os.path.abspath(db):
            missing.append(name)
    assert checked, "nessun builtin risolvibile: il test non sta guardando niente"
    assert not missing, f"builtin senza OPENAGENT_DB_PATH: {missing}"
    # The one that actually broke, named so a regression is unmistakable.
    assert not BUILTIN_MCP_SPECS["media-gen"].get("in_process")
    media = resolve_default_entry({"builtin": "media-gen"}, db_path=db)
    assert (media.get("env") or {}).get("OPENAGENT_DB_PATH") == os.path.abspath(db)


@test("mcp_db_path", "the fallback is the agent's DB, never the CWD")
async def t_fallback_is_not_cwd(ctx: TestContext) -> None:
    from src.mcp.servers import _common

    saved = os.environ.get("OPENAGENT_DB_PATH")
    try:
        os.environ["OPENAGENT_DB_PATH"] = "/tmp/injected/openagent.db"
        assert _common.db_path() == "/tmp/injected/openagent.db"

        os.environ.pop("OPENAGENT_DB_PATH", None)
        resolved = _common.db_path()
        # The point of the fix: a subprocess whose env was not injected must
        # not quietly open (and create) a database in whatever directory it
        # happens to be running from.
        assert resolved != "openagent.db", resolved
        assert os.path.isabs(resolved), resolved
    finally:
        if saved is None:
            os.environ.pop("OPENAGENT_DB_PATH", None)
        else:
            os.environ["OPENAGENT_DB_PATH"] = saved


@test("mcp_db_path", "the DB-touching servers all use the shared resolver")
async def t_servers_share_the_resolver(ctx: TestContext) -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    for rel in (
        "src/mcp/servers/workflow_manager/server.py",
        "src/mcp/servers/scheduler/server.py",
        "src/mcp/servers/events_manager/server.py",
    ):
        text = (root / rel).read_text()
        assert 'or "openagent.db"' not in text, f"{rel}: fallback sulla CWD tornato"
