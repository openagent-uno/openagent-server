"""Provider adapter for the in-process ``vault-gate`` MCP.

Follows the runtime ``Toolkit`` pattern (see ``shell/adapters.py``): plain
async callables with type hints + docstrings, wrapped once. The runtime turns
the docstrings/signatures into the tool schema the model sees.
"""
from __future__ import annotations

from typing import Any

from src.mcp.servers.vault_gate import handlers, recall


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

    async def vault_search(query: str, limit: int = 20,
                           search_type: str = "content",
                           file_path: str | None = None) -> dict:
        """Search the vault. ``search_type`` controls what is searched:

        - ``"content"`` (default) — full-text search over title/summary/body (FTS5).
        - ``"filename"`` — search only note file names/paths.
        - ``"regex"`` — requires ``file_path``; searches within ONE note's content
          using Python regex. Returns line/column positions for each match.

        For ``search_type="regex"``, set ``file_path`` to the vault-relative path
        of the note (e.g. ``"projects/my-project/notes.md"``) and ``query`` to
        the regex pattern."""
        return await handlers.vault_search(
            query=query, limit=limit,
            search_type=search_type, file_path=file_path,
        )

    async def vault_backlinks(path: str) -> dict:
        """List the notes that link TO a note (its inbound links)."""
        return await handlers.vault_backlinks(path=path)

    async def vault_dream() -> dict:
        """Run a DREAM-MODE maintenance pass NOW: grade the vault, auto-fix
        the mechanical issues, regenerate llms.txt + showcase, commit it, and
        return open_suggestions — the harder issues (orphans, duplicates,
        over-long notes, missing summaries, broken links) for YOU to resolve.
        After it returns, fix those by writing/merging/linking notes, then run
        vault_gate again to confirm the vault improved."""
        return await handlers.vault_dream()

    async def vault_init() -> dict:
        """Scaffold the Company-Brain folder system: the eleven folders, the
        journal sub-tree, the canon workspace, and note/session/daily
        templates. Idempotent."""
        return await handlers.vault_init()

    async def vault_regenerate_derived() -> dict:
        """Regenerate llms.txt and _showcase/showcase.md from the notes.
        These are derived files — never edit them by hand."""
        return await handlers.vault_regenerate_derived()

    async def vault_recall_stats(limit: int = 20, since_days: int | None = 30,
                                 note_path: str | None = None) -> dict:
        """Which notes you actually READ, and how those runs ended. Per note:
        recalls, ok, errored, cancelled_excluded (barge-ins — never scored),
        and ok_rate over the scorable runs only.

        This is ASSOCIATION, NOT CAUSATION: ok_rate only says a run that read
        the note finished without raising — not that the note helped, and a
        run that read ten notes credits all ten. Use it to decide what to
        re-read or re-check, and during a dream pass to see which notes are
        load-bearing (often recalled) vs never recalled at all. Never use it
        alone as grounds to delete a note. Ignore rows with a low 'scorable'."""
        return await recall.vault_recall_stats(
            limit=limit, since_days=since_days, note_path=note_path,
        )

    return Toolkit(
        name="vault-gate",
        tools=[
            vault_gate, vault_doctor, vault_dream, vault_validate_note,
            vault_rename_note, vault_init, vault_stats, vault_search,
            vault_backlinks, vault_regenerate_derived, vault_recall_stats,
        ],
    )
