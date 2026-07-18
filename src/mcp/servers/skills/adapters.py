"""Provider adapter for the in-process ``skills`` MCP.

Follows the runtime ``Toolkit`` pattern (see ``vault_gate/adapters.py`` and
``shell/adapters.py``): plain async callables with type hints + docstrings,
wrapped once. The runtime turns the docstrings/signatures into the tool
schema the model sees — no JSON-schema boilerplate here.

``build_runtime_toolkit()`` takes NO args: each handler resolves the skills
directory itself via ``paths.default_skills_path()``.
"""
from __future__ import annotations

from typing import Any

from src.mcp.servers.skills import handlers


def build_runtime_toolkit() -> Any:
    from src.mcp._runtime import Toolkit

    async def skill_view(name: str) -> dict:
        """Load the FULL body of one skill by name. The system-prompt skills
        index lists only ``name: description`` (progressive disclosure); call
        this to read the whole SKILL.md — instructions, steps, examples, and
        any bundled files — right before you act on the skill."""
        return await handlers.skill_view(name=name)

    async def skill_search(query: str, limit: int = 20) -> dict:
        """Find skills by a plain substring over name, description, and body.
        Use it when you don't know the exact skill name or want to discover
        which skill fits a task. Returns metadata per hit; follow up with
        ``skill_view`` to read the full body of the one you want."""
        return await handlers.skill_search(query=query, limit=limit)

    async def skill_manage(
        action: str,
        name: str,
        body: str | None = None,
        description: str | None = None,
        category: str | None = None,
    ) -> dict:
        """Create, update, or remove a skill on disk. ``action`` is
        ``create`` / ``update`` / ``remove``. For create/update, ``body`` is
        the markdown instructions — frontmatter (``name`` / ``description`` /
        ``category``) is generated for you, so do not include your own
        ``---`` block. Edits appear in the system-prompt skills index on the
        next boot/reload, not mid-session."""
        return await handlers.skill_manage(
            action=action, name=name, body=body,
            description=description, category=category,
        )

    return Toolkit(
        name="skills",
        tools=[skill_view, skill_search, skill_manage],
    )
