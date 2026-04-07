"""Memory manager: handles session history, long-term memory, and knowledge base.

Two memory systems:
1. SQL memories (quick facts, user preferences) — stored in SQLite rows
2. Knowledge base (detailed knowledge, docs, procedures) — stored as .md files indexed by FTS5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openagent.models.base import BaseModel

from openagent.memory.db import MemoryDB
from openagent.memory.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

MEMORY_EXTRACTION_PROMPT = """Analyze this conversation and extract key facts worth remembering.

Classify each fact as either:
- "fact": short user preference or piece of info (stored in quick-access DB)
- "knowledge": detailed procedure, architecture, or reference info (stored as a document)

Return a JSON array of objects:
[
  {"type": "fact", "content": "User prefers Python", "topic": "preferences"},
  {"type": "knowledge", "title": "Deploy Wardrobe Service", "content": "Full procedure...", "topic": "deploy", "tags": ["k8s", "ovh"]}
]

Only include genuinely useful, non-obvious information. Return [] if nothing worth remembering.

Conversation:
{conversation}

Existing facts (avoid duplicates):
{existing}

JSON array:"""


class MemoryManager:
    """Manages session history, quick-access memories, and knowledge base.

    - SQL memories: user preferences, short facts (fast lookup by user/topic)
    - Knowledge base: detailed docs, procedures, architecture notes (FTS5 search, .md files)
    """

    def __init__(
        self,
        db: MemoryDB,
        auto_extract: bool = True,
        history_limit: int = 50,
        knowledge_dir: str | Path = "./memories",
    ):
        self.db = db
        self.auto_extract = auto_extract
        self.history_limit = history_limit
        self._kb: KnowledgeBase | None = None
        self._knowledge_dir = knowledge_dir

    async def initialize_knowledge(self) -> None:
        """Initialize the knowledge base (call after db.connect())."""
        if self._kb is None and self.db._conn is not None:
            self._kb = KnowledgeBase(self.db._conn, self._knowledge_dir)
            await self._kb.initialize()

    @property
    def knowledge(self) -> KnowledgeBase | None:
        return self._kb

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

    async def get_memories_for_context(self, agent_id: str, user_id: str = "") -> list[dict]:
        """Retrieve all relevant memories to inject into system prompt."""
        return await self.db.get_memories(agent_id, user_id, limit=20)

    async def build_memory_context(self, agent_id: str, user_id: str = "", query: str = "") -> str:
        """Build a context string from SQL memories + knowledge base."""
        parts = []

        # SQL quick-access memories
        memories = await self.get_memories_for_context(agent_id, user_id)
        if memories:
            lines = ["## Things I remember about you:"]
            for mem in memories:
                topic = mem.get("topic", "")
                prefix = f"[{topic}] " if topic else ""
                lines.append(f"- {prefix}{mem['content']}")
            parts.append("\n".join(lines))

        # Knowledge base context (if query provided and KB initialized)
        if query and self._kb:
            try:
                kb_context = await self._kb.build_context(query)
                if kb_context:
                    parts.append(kb_context)
            except Exception as e:
                logger.warning(f"Knowledge base search failed: {e}")

        return "\n\n".join(parts)

    async def extract_and_store_memories(
        self,
        model: BaseModel,
        agent_id: str,
        user_id: str,
        messages: list[dict[str, Any]],
    ) -> list[dict]:
        """Use the model to extract key facts and knowledge from the conversation."""
        if not messages:
            return []

        conv_parts = []
        for msg in messages[-10:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                conv_parts.append(f"{role}: {content}")
        conversation = "\n".join(conv_parts)

        existing = await self.db.get_memories(agent_id, user_id, limit=50)
        existing_text = "\n".join(m["content"] for m in existing) if existing else "None"

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            conversation=conversation,
            existing=existing_text,
        )

        response = await model.generate([{"role": "user", "content": prompt}])

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            facts = json.loads(content)
        except (json.JSONDecodeError, IndexError):
            return []

        stored = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue

            fact_type = fact.get("type", "fact")

            if fact_type == "knowledge" and self._kb:
                # Store as .md file in knowledge base
                title = fact.get("title", fact.get("content", "")[:50])
                kb_content = fact.get("content", "")
                topic = fact.get("topic", "")
                tags = fact.get("tags", [])
                try:
                    await self._kb.add(title, kb_content, topic=topic, tags=tags)
                    stored.append(fact)
                except Exception as e:
                    logger.warning(f"Failed to store knowledge: {e}")

            elif fact_type == "fact" and "content" in fact:
                mem_content = fact["content"]
                topic = fact.get("topic", "")
                # Dedup
                if not any(
                    mem_content.lower() in m["content"].lower()
                    or m["content"].lower() in mem_content.lower()
                    for m in existing
                ):
                    await self.db.add_memory(agent_id, user_id, mem_content, topic)
                    stored.append(fact)

        return stored

    async def remember(self, agent_id: str, user_id: str, content: str, topic: str = "") -> str:
        """Explicitly store a quick-access memory."""
        return await self.db.add_memory(agent_id, user_id, content, topic)

    async def remember_knowledge(self, title: str, content: str, topic: str = "", tags: list[str] | None = None) -> str:
        """Explicitly store a knowledge base entry as .md file."""
        if not self._kb:
            raise RuntimeError("Knowledge base not initialized")
        return await self._kb.add(title, content, topic=topic, tags=tags)

    async def forget(self, memory_id: str) -> None:
        """Delete a quick-access memory."""
        await self.db.delete_memory(memory_id)

    async def search_knowledge(self, query: str, topic: str | None = None, limit: int = 10) -> list[dict]:
        """Search the knowledge base."""
        if not self._kb:
            return []
        return await self._kb.search(query, topic=topic, limit=limit)
