"""Memory manager: handles session history and long-term memory extraction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openagent.models.base import BaseModel

from openagent.memory.db import MemoryDB


MEMORY_EXTRACTION_PROMPT = """Analyze this conversation and extract key facts worth remembering about the user.
Return a JSON array of objects, each with "content" (the fact) and "topic" (a short category).
Only include genuinely useful, non-obvious facts. Return [] if nothing worth remembering.
Example: [{"content": "User prefers Python over JavaScript", "topic": "preferences"}]

Conversation:
{conversation}

Existing memories (avoid duplicates):
{existing}

JSON array:"""


class MemoryManager:
    """Manages session history and long-term memory extraction.

    Modes:
    - auto_extract=True: after each run, uses the model to extract key facts
    - auto_extract=False: memories only stored via explicit remember() calls
    """

    def __init__(self, db: MemoryDB, auto_extract: bool = True, history_limit: int = 50):
        self.db = db
        self.auto_extract = auto_extract
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

    async def get_memories_for_context(self, agent_id: str, user_id: str = "") -> list[dict]:
        """Retrieve all relevant memories to inject into system prompt."""
        return await self.db.get_memories(agent_id, user_id, limit=20)

    async def build_memory_context(self, agent_id: str, user_id: str = "") -> str:
        """Build a text block of memories to prepend to system prompt."""
        memories = await self.get_memories_for_context(agent_id, user_id)
        if not memories:
            return ""
        lines = ["## Things I remember about you:"]
        for mem in memories:
            topic = mem.get("topic", "")
            prefix = f"[{topic}] " if topic else ""
            lines.append(f"- {prefix}{mem['content']}")
        return "\n".join(lines)

    async def extract_and_store_memories(
        self,
        model: BaseModel,
        agent_id: str,
        user_id: str,
        messages: list[dict[str, Any]],
    ) -> list[dict]:
        """Use the model to extract key facts from the conversation and store them."""
        if not messages:
            return []

        # Build conversation text
        conv_parts = []
        for msg in messages[-10:]:  # Last 10 messages max
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                conv_parts.append(f"{role}: {content}")
        conversation = "\n".join(conv_parts)

        # Get existing memories for dedup
        existing = await self.db.get_memories(agent_id, user_id, limit=50)
        existing_text = "\n".join(m["content"] for m in existing) if existing else "None"

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            conversation=conversation,
            existing=existing_text,
        )

        response = await model.generate([{"role": "user", "content": prompt}])

        # Parse the JSON response
        try:
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            facts = json.loads(content)
        except (json.JSONDecodeError, IndexError):
            return []

        stored = []
        for fact in facts:
            if isinstance(fact, dict) and "content" in fact:
                mem_content = fact["content"]
                topic = fact.get("topic", "")
                # Simple dedup: skip if very similar content exists
                if not any(mem_content.lower() in m["content"].lower() or m["content"].lower() in mem_content.lower() for m in existing):
                    await self.db.add_memory(agent_id, user_id, mem_content, topic)
                    stored.append(fact)

        return stored

    async def remember(self, agent_id: str, user_id: str, content: str, topic: str = "") -> str:
        """Explicitly store a memory (for manual mode or agent-invoked)."""
        return await self.db.add_memory(agent_id, user_id, content, topic)

    async def forget(self, memory_id: str) -> None:
        """Delete a specific memory."""
        await self.db.delete_memory(memory_id)
