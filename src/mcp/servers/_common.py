"""Shared helpers for in-tree Python MCP servers (scheduler, mcp-manager,
model-manager).

Each of those runs as its own subprocess spawned by ``MCPPool`` and talks
to the same SQLite DB (``OPENAGENT_DB_PATH``) that the main OpenAgent
process uses. They all need the same tiny bit of plumbing: resolve the
DB path, open a single aiosqlite connection with WAL + the shared
schema, and keep it around for the life of the subprocess.

The servers are in separate processes from the main MemoryDB connection,
so ``aiosqlite`` concurrency at the Python level isn't shared — each
process needs its own connection. Using ``MemoryDB`` directly from here
would work but pull in the whole memory layer; the goal was to keep
these subprocesses lean and dependency-free beyond ``SCHEMA_SQL``.
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiosqlite
from src.memory.db import SCHEMA_SQL, sqlite_busy_timeout_ms, sqlite_busy_timeout_s

logger = logging.getLogger(__name__)


def db_path() -> str:
    """Resolve the SQLite path for an in-tree MCP subprocess.

    Precedence: ``OPENAGENT_DB_PATH`` env var (injected by MCPPool for
    the scheduler / mcp-manager / model-manager), else ``openagent.db``
    in the current working directory so the server still works when
    invoked directly with ``python -m openagent.mcp.servers.X.server``.
    """
    injected = (os.environ.get("OPENAGENT_DB_PATH") or "").strip()
    if injected:
        return injected
    # NOT ``"openagent.db"``. Relative to the CWD means, under PyInstaller, a
    # scratch directory that gets a brand-new empty schema written into it —
    # so a subprocess whose env was not injected does not fail, it silently
    # answers every question from an empty database. Fall back to the same
    # file the main process uses.
    try:
        from src.core.paths import default_db_path

        resolved = str(default_db_path())
        logger.warning(
            "OPENAGENT_DB_PATH not injected; falling back to %s", resolved,
        )
        return resolved
    except Exception:  # noqa: BLE001
        return "openagent.db"


class SharedConnection:
    """Lazy singleton aiosqlite connection for the subprocess.

    Use one per server module:

        _conn = SharedConnection("mcp-manager")
        async def handler(...):
            conn = await _conn.get()
            ...

    Multiple subprocesses can safely share the DB file thanks to WAL.
    """

    def __init__(self, server_name: str):
        self._server_name = server_name
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> aiosqlite.Connection:
        async with self._lock:
            if self._conn is None:
                path = db_path()
                conn = await aiosqlite.connect(
                    path, timeout=sqlite_busy_timeout_s(),
                )
                conn.row_factory = aiosqlite.Row
                await conn.execute(
                    f"PRAGMA busy_timeout = {sqlite_busy_timeout_ms()}"
                )
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.executescript(SCHEMA_SQL)
                await conn.commit()
                self._conn = conn
                logger.info("%s MCP connected to %s", self._server_name, path)
            return self._conn


def run_stdio(mcp, *, loglevel_env: str) -> None:
    """Common entrypoint: configure logging and run FastMCP over stdio."""
    logging.basicConfig(
        level=os.environ.get(loglevel_env, "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run()


async def ensure_row_exists(
    conn: aiosqlite.Connection, table: str, id_col: str, id_val: str
) -> str:
    """Assert that ``table`` has a row with ``id_col = id_val`` or raise.

    Returns ``id_val`` so callers can chain. Used by the manager MCP
    CRUD tools so update/enable/disable/remove get a clear error when
    the id doesn't resolve, instead of silently no-op'ing the UPDATE.
    """
    cursor = await conn.execute(
        f"SELECT 1 FROM {table} WHERE {id_col} = ?", (id_val,)
    )
    if not await cursor.fetchone():
        raise ValueError(f"No {table} row with {id_col}={id_val!r}")
    return id_val
