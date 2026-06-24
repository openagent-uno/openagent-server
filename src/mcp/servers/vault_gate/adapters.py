"""Provider adapter for the in-process ``vault-gate`` MCP.

Follows the runtime ``Toolkit`` pattern (see ``shell/adapters.py``): plain
async callables with type hints + docstrings, wrapped once. The runtime turns
the docstrings/signatures into the tool schema the model sees.
"""
from __future__ import annotations

from typing import Any

from src.mcp.servers.vault_gate import handlers


def build_runtime_toolkit() -> Any:
    from src.mcp._runtime import Toolkit

    async def vault_gate(strict: bool = False, limit: int = 40) -> dict:
        """Run the vault quality gate over every note. Reports missing
        frontmatter, over-long notes, broken wikilinks, orphan notes,
        disconnected islands, duplicates, and unanchored journal notes.
        ok=true means zero error-level issues. strict=true treats every issue
        as an error (the "0 errors" target)."""
        return await handlers.vault_gate(strict=strict, limit=limit)

    async def vault_doctor(apply: bool = False, limit: int = 40) -> dict:
        """Mechanically fix the issues a script can fix safely (collapse
        multi-line related, strip spaces in [[ ]], normalize dates, scaffold
        missing frontmatter, replace em dashes) and list harder issues
        (orphans, duplicates, over-long notes, broken links) as suggestions.
        apply=false is a dry run; apply=true writes the fixes."""
        return await handlers.vault_doctor(apply=apply, limit=limit)

    async def vault_validate_note(path: str, content: str) -> dict:
        """Validate a note's content BEFORE writing it: frontmatter, atomic
        size, wikilink format, dates, and whether every [[wikilink]] resolves.
        Returns ok plus a list of issues so you can fix it before saving."""
        return await handlers.vault_validate_note(path=path, content=content)

    async def vault_rename_note(old_path: str, new_path: str) -> dict:
        """Move or rename a note (or a whole folder) without breaking links:
        every [[wikilink]] to it is rewritten automatically (aliases and
        anchors preserved). Always prefer this over a raw move/rename."""
        return await handlers.vault_rename_note(old_path=old_path, new_path=new_path)

    async def vault_stats() -> dict:
        """Vault health at a glance: notes, links, broken links, orphans,
        connected components, notes per folder."""
        return await handlers.vault_stats()

    async def vault_search(query: str, limit: int = 20) -> dict:
        """Full-text search the vault (scales to 100k+ notes). Returns
        matching notes with a snippet."""
        return await handlers.vault_search(query=query, limit=limit)

    async def vault_backlinks(path: str) -> dict:
        """List the notes that link TO a note (its inbound links)."""
        return await handlers.vault_backlinks(path=path)

    async def vault_init() -> dict:
        """Scaffold the Company-Brain folder system: the eleven folders, the
        journal sub-tree, the canon workspace, and note/session/daily
        templates. Idempotent."""
        return await handlers.vault_init()

    async def vault_regenerate_derived() -> dict:
        """Regenerate llms.txt and _showcase/showcase.md from the notes.
        These are derived files — never edit them by hand."""
        return await handlers.vault_regenerate_derived()

    return Toolkit(
        name="vault-gate",
        tools=[
            vault_gate, vault_doctor, vault_validate_note, vault_rename_note,
            vault_init, vault_stats, vault_search, vault_backlinks,
            vault_regenerate_derived,
        ],
    )
