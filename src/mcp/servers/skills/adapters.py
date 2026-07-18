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
        """Create, update, remove, or archive a skill on disk. ``action`` is
        ``create`` / ``update`` / ``remove`` / ``archive``. For create/update,
        ``body`` is the markdown instructions — frontmatter (``name`` /
        ``description`` / ``category``) is generated for you, so do not include
        your own ``---`` block. Skills you create are stamped
        ``created_by: agent``. ``archive`` retires a skill (drops it from the
        prompt index) without deleting the file, so it is reversible. Edits
        appear in the system-prompt skills index on the next boot/reload, not
        mid-session."""
        return await handlers.skill_manage(
            action=action, name=name, body=body,
            description=description, category=category,
        )

    tools = [skill_view, skill_search, skill_manage]

    # Skills-Hub tools are a SECOND gate on top of ``skills.enabled``: they are
    # only EXPOSED when ``skills.hub.enabled`` is also true (mirrors how the
    # curator's scheduled task is only seeded when ``skills.curator_enabled``).
    # With hub off the toolkit is byte-identical to the three-tool original —
    # the hub tools never enter the tool list, so the schema the model sees is
    # unchanged.
    from src.core.config import load_config, skills_settings

    if skills_settings(load_config()).hub_enabled:
        from src.mcp.servers.skills import hub

        async def skill_hub_pull(tap: str, force: bool = False) -> dict:
            """Pull SKILL.md skills from a shared git *tap* into your skills
            dir. ``tap`` is any git remote (``https://``, ``git@…``, or
            ``file://…``). The remote is shallow-cloned into quarantine, every
            skill is safety-scanned, and only the ones that pass are installed —
            each stamped ``created_by: hub`` (so the curator leaves it alone)
            with its source ``hub_repo`` / ``hub_commit``. A malicious skill is
            refused: ``dangerous`` always, ``caution`` unless ``force=True``.
            New/updated skills appear in the skills index on the next
            boot/reload."""
            return await hub.skill_hub_pull(tap=tap, force=force)

        async def skill_hub_list() -> dict:
            """List the hub skills installed locally (``created_by: hub``),
            each with the source repo / commit / content hash it was pinned to
            in ``.hub/lock.json``. Use it to see what shared playbooks are
            installed and exactly where they came from."""
            return await hub.skill_hub_list()

        tools = tools + [skill_hub_pull, skill_hub_list]

    return Toolkit(name="skills", tools=tools)
