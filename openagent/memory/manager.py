"""Memory manager: handles session history.

Long-term memory and knowledge base are handled by the Obsidian vault
via MCPVault MCP — the agent searches and writes to .md files directly.

This manager only handles:
- Session history (conversation messages in SQLite)
"""

from __future__ import annotations

import logging
from typing import Any

from openagent.memory.db import MemoryDB

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages session history (SQLite).

    Knowledge base is handled by MCPVault MCP (Obsidian-compatible .md files).
    The agent uses MCPVault tools to search/read/write memories autonomously.
    """

    def __init__(self, db: MemoryDB, history_limit: int = 50):
        self.db = db
        self.history_limit = history_limit

    async def ensure_session(self, agent_id: str, user_id: str = "", session_id: str | None = None) -> str:
        return await self.db.get_or_create_session(agent_id, user_id, session_id)

    async def store_message(
        self,
        session_id: str,
        role: str,
        content: str = "",
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        tool_result: str | None = None,
    ) -> None:
        """Store a message immediately after each turn."""
        await self.db.add_message(
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            tool_result=tool_result,
        )

    async def get_history(self, session_id: str) -> list[dict]:
        """Load recent conversation history for context."""
        return await self.db.get_recent_messages(session_id, limit=self.history_limit)
