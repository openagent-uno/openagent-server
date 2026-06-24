"""Async handlers behind the ``vault-gate`` tools.

Thin wrappers over ``VaultService`` that shape results for a model: reports
are trimmed to a useful number of violations (with a truncation flag) so a
gate over a huge vault doesn't dump tens of thousands of lines into context.
"""
from __future__ import annotations

import dataclasses

from src.memory.vault.model import GateReport
from src.memory.vault.service import get_service
from src.memory.vault.vault_origin import recent_origin


def _origin(tool: str) -> dict:
    """Provenance for a tool-driven vault change: the current chat/workflow/
    task context (if any) plus the tool name."""
    return {**(recent_origin() or {"kind": "tool"}), "tool": tool}


def _compact_report(rep: GateReport, limit: int = 40) -> dict:
    grouped: dict[str, int] = {}
    for v in rep.violations:
        grouped[v.rule] = grouped.get(v.rule, 0) + 1
    shown = rep.violations[:limit]
    return {
        "summary": rep.summary_line(),
        "ok": rep.ok,
        "counts": {
            "error": rep.error_count,
            "warn": rep.warn_count,
            "info": rep.info_count,
        },
        "by_rule": grouped,
        "stats": {
            k: rep.stats.get(k)
            for k in ("notes", "gated_notes", "links", "broken_links",
                      "orphans", "components", "islands")
        },
        "violations": [v.to_dict() for v in shown],
        "violations_truncated": len(rep.violations) > limit,
        "total_violations": len(rep.violations),
        "elapsed_ms": rep.elapsed_ms,
    }


async def vault_gate(strict: bool = False, limit: int = 40) -> dict:
    """Run the vault quality gate over every note and return a structured
    report: missing frontmatter, over-long notes, broken wikilinks, orphan
    notes, disconnected islands, duplicates, journal notes not anchored to an
    entity, and more. Passes (``ok: true``) when there are zero error-level
    issues. Set ``strict=true`` to treat every issue as an error (the
    Company-Brain "0 errors" target)."""
    svc = get_service()
    cfg = dataclasses.replace(svc.config, strict=True) if strict else svc.config
    rep = await svc.gate(config=cfg)
    return _compact_report(rep, limit=limit)


async def vault_doctor(apply: bool = False, limit: int = 40) -> dict:
    """Mechanically fix the vault issues a script can fix safely (collapse
    multi-line ``related:``, strip spaces inside ``[[ ]]``, normalize dates to
    YYYY-MM-DD, scaffold missing frontmatter fields, replace em dashes), and
    list the harder issues (orphans, duplicates, over-long notes, broken
    links) as suggestions for you to resolve. ``apply=false`` is a dry run
    that only reports what would change; ``apply=true`` writes the fixes."""
    svc = get_service()
    result = await svc.doctor(apply=apply, origin=_origin("vault_doctor"))
    fix = result["fix"]
    after = result["after"] or result["before"]
    return {
        "applied": apply,
        "files_changed": fix["files_changed"],
        "mechanical_fixes": fix["fixed"],
        "open_suggestions": fix["suggestions"][:limit],
        "open_suggestion_count": len(fix["suggestions"]),
        "errors_before": result["before"]["error_count"],
        "errors_after": after["error_count"],
        "summary": after.get("ok"),
    }


async def vault_validate_note(path: str, content: str) -> dict:
    """Validate a single note's content BEFORE writing it: checks frontmatter
    completeness, atomic size, wikilink formatting, date format, and whether
    every ``[[wikilink]]`` resolves to an existing note. Returns ``ok`` plus a
    list of issues so you can fix the note before saving it."""
    svc = get_service()
    return await svc.validate_note(path, content)


async def vault_rename_note(old_path: str, new_path: str) -> dict:
    """Move or rename a note (or a whole folder) WITHOUT breaking links: every
    [[wikilink]] pointing at it — in any other note — is automatically
    rewritten to the new name, preserving aliases and anchors. Always use
    this instead of a raw move/rename, which would leave dangling links."""
    svc = get_service()
    return await svc.move(old_path, new_path, origin=_origin("vault_rename_note"))


async def vault_stats() -> dict:
    """Vault health at a glance: note count, total + broken wikilinks, orphan
    count, connected-component count, and notes per folder."""
    svc = get_service()
    return await svc.stats()


async def vault_search(query: str, limit: int = 20) -> dict:
    """Full-text search the vault (FTS5 over title/summary/body). Scales to
    hundreds of thousands of notes. Returns matching notes with a snippet."""
    svc = get_service()
    results = await svc.search(query, limit=limit)
    return {"query": query, "count": len(results), "results": results}


async def vault_backlinks(path: str) -> dict:
    """List the notes that link TO ``path`` (its backlinks / inbound links)."""
    svc = get_service()
    links = await svc.backlinks(path)
    return {"path": path, "backlinks": links, "count": len(links)}


async def vault_init() -> dict:
    """Scaffold the Company-Brain folder system in the vault: the eleven
    folders (self, areas, projects, sources, concepts, docs, entities, data,
    code, outputs, workspace), the journal sub-tree, the canon workspace, and
    note/session/daily templates. Idempotent — safe to run on an existing
    vault."""
    svc = get_service()
    return await svc.init_taxonomy(origin=_origin("vault_init"))


async def vault_regenerate_derived() -> dict:
    """Regenerate the derived artifacts from the notes: ``llms.txt`` (the
    per-folder index AIs read) and ``_showcase/showcase.md`` (the vault
    snapshot). These are derived — never edit them by hand."""
    svc = get_service()
    return await svc.regenerate_derived(origin=_origin("vault_regenerate_derived"))
