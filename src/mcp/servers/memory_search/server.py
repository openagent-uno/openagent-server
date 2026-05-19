"""Memory-search MCP server.

Exposes the ``search_past_conversations`` tool so the agent can answer
"remember when we discussed X?" style questions across past sessions.
The actual cosine-similarity + embedding logic lives in
:mod:`src.learning.semantic_search`; this server is a thin MCP wrapper
that connects to the agent's SQLite DB via ``OPENAGENT_DB_PATH``.

Off by default. Enable via ``memory.semantic_search.enabled: true`` in
``openagent.yaml`` AND make sure ``OPENAI_API_KEY`` is set (we use
OpenAI's ``text-embedding-3-small`` for vectorisation). Without the
key, the tool returns an empty list with a hint.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import aiosqlite
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("memory-search")


class _ConnShim:
    """Adapter — the ``semantic_search`` helpers expect ``db._conn`` to
    be an aiosqlite Connection. Mirrors the same shim
    :mod:`src.bridges.base` uses for the skill matcher; defined locally
    so a frozen-binary subprocess doesn't have to import the bridge
    module just for one dataclass."""

    __slots__ = ("_conn",)

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn


def _db_path() -> str:
    """Resolve the agent DB path the same way every other Python MCP
    server does — env var first (injected by the agent at MCP spawn),
    then a relative fallback for tests."""
    return os.environ.get("OPENAGENT_DB_PATH", "./openagent.db")


@mcp.tool()
async def search_past_conversations(
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Semantic search over past user/assistant turns.

    Args:
        query: Natural-language description of what you're looking for.
            "the deadline we set for the React migration", "user's
            preference for jq indent", etc.
        top_k: How many hits to return (default 5, hard cap 25).
        session_id: Restrict to one conversation. ``None`` searches
            across every session on this agent — useful for "have I
            ever discussed X?" type queries.

    Returns ``{ok, hits, model, hint?}``. Each hit carries the source
    turn's role / text / timestamp / similarity score so the agent can
    quote it back to the user verbatim. ``ok=False`` with a ``hint``
    when the feature is disabled or the OpenAI key isn't set.
    """
    if not query or not query.strip():
        return {"ok": False, "hits": [], "hint": "empty query"}

    if (
        os.environ.get("OPENAGENT_SEMANTIC_SEARCH", "0").strip().lower()
        not in ("1", "true", "yes", "on")
    ):
        return {
            "ok": False,
            "hits": [],
            "hint": "Feature off — set memory.semantic_search.enabled: true in openagent.yaml.",
        }
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        return {
            "ok": False,
            "hits": [],
            "hint": "OPENAI_API_KEY not set — embeddings need OpenAI access.",
        }

    capped = max(1, min(int(top_k), 25))
    try:
        conn = await aiosqlite.connect(_db_path(), timeout=5.0)
        conn.row_factory = aiosqlite.Row
        try:
            # Imported here so a frozen build doesn't pay the
            # ``learning.semantic_search`` import cost when the MCP
            # boots — only when the tool is actually called.
            from src.learning.semantic_search import search as _search

            hits = await _search(_ConnShim(conn), query, top_k=capped, session_id=session_id)
        finally:
            await conn.close()
    except Exception as e:
        logger.exception("search_past_conversations failed")
        return {"ok": False, "hits": [], "hint": f"error: {e}"}

    return {"ok": True, "hits": hits, "model": "text-embedding-3-small"}


def main() -> None:
    """Entry point matched by ``builtins.py`` python_module pattern."""
    mcp.run()


if __name__ == "__main__":
    main()
