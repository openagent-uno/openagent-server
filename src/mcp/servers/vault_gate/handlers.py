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


_CANONICAL_PATH_MARKERS = (
    "/procedures/", "/dev/", "/known-issues/", "/support-playbooks/",
    "/features/", "/releases/", "/subsystems/", "support-grounding",
)
_RECEIPT_PATH_MARKERS = (
    "/receipts/", "receipts-archive", "/batch-runs/", "/cycle-logs/",
)


def _canonical_first(results: list[dict], limit: int) -> list[dict]:
    """Promote authoritative candidates without discarding relevant receipts.

    A support vault can contain thousands of triage receipts repeating the same
    customer vocabulary. Pure BM25 therefore returns ten historical examples
    before the one bug analysis or procedure that determines what is true now.
    Preserve search relevance inside each group, but put canonical candidates
    first and label receipts explicitly so a small local model does not mistake
    precedent for current state.
    """
    canonical: list[dict] = []
    remainder: list[dict] = []
    for item in results:
        path = "/" + str(item.get("path") or "").lower().lstrip("/")
        tagged = dict(item)
        is_receipt = any(marker in path for marker in _RECEIPT_PATH_MARKERS)
        is_grounding = "support-grounding" in path
        is_canonical = is_grounding or (
            not is_receipt
            and any(marker in path for marker in _CANONICAL_PATH_MARKERS)
        )
        if is_canonical:
            tagged["evidence_class"] = "canonical_candidate"
            canonical.append(tagged)
        else:
            if is_receipt:
                tagged["evidence_class"] = "historical_receipt"
            else:
                tagged["evidence_class"] = "supporting_context"
            remainder.append(tagged)
    return (canonical + remainder)[:max(1, limit)]


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


async def vault_search(query: str, limit: int = 20,
                      search_type: str = "content",
                      file_path: str | None = None) -> dict:
    """Search the vault. This is the DEFAULT way to consult memory: it runs
    over an incremental full-text index, so it stays sub-10ms on a vault of
    thousands of notes instead of re-reading every file from disk.

    ``search_type`` controls what is searched:

    - ``"content"`` (default) — full-text over filename/title/summary/body,
      best match first.
    - ``"filename"`` — search only note file names/paths.
    - ``"regex"`` — requires ``file_path``; searches within ONE note's content
      using Python regex. Returns line/column positions for each match.

    Query language for ``search_type="content"``:

    - Plain words are OR'd and RANKED — ask in natural words and read the
      top hits. A note's filename and title count for more than a passing
      mention in a body, which is what floats the note that is ABOUT your
      topic above the many that merely mention it. Do NOT pre-narrow out of
      caution: requiring every word measurably finds the WRONG note on
      question-shaped queries.
    - ``+term`` requires a term, ``"exact phrase"`` requires a phrase, and
      ``-term`` excludes one. Reach for these when plain words return a
      topic instead of the fact you need.
    - Punctuation is safe to type; it is never interpreted as syntax.

    Frontmatter (including ``tags``) is NOT in this index — use the vault
    MCP's ``search_notes`` with ``searchFrontmatter``/``pathPrefix`` for
    those, and ``list_all_tags`` to enumerate tags.

    Each result carries a ``snippet`` with the matched terms in ``[]``; it is
    sized to let you pick between hits WITHOUT opening each one — read it
    before reaching for ``vault_read_note``.

    For ``search_type="regex"``, set ``file_path`` to the vault-relative path
    of the note (e.g. ``"projects/my-project/notes.md"``) and ``query`` to
    the regex pattern."""
    svc = get_service()
    if search_type == "filename":
        results = await svc.search_files(query, limit=limit)
        return {"query": query, "count": len(results), "results": results,
                "search_type": "filename"}
    if search_type == "regex":
        import re as _re
        if not file_path:
            return {"error": "file_path is required for regex search",
                    "count": 0, "results": []}
        vault_root = svc.vault_root
        full = vault_root / file_path.lstrip("/")
        try:
            full = full.resolve()
            full.relative_to(vault_root.resolve())
        except (ValueError, OSError):
            return {"error": "Invalid or forbidden path", "count": 0, "results": []}
        if not full.exists() or not full.is_file():
            return {"error": "File not found", "count": 0, "results": []}
        try:
            content = full.read_text(errors="replace")
        except OSError as exc:
            return {"error": str(exc), "count": 0, "results": []}
        try:
            pattern = _re.compile(query)
        except _re.error as exc:
            return {"error": f"Invalid regex: {exc}", "count": 0, "results": []}
        matches = []
        for lineno, line in enumerate(content.split("\n"), start=1):
            for m in pattern.finditer(line):
                matches.append({
                    "line": lineno,
                    "col": m.start() + 1,
                    "text": line,
                })
        return {"query": query, "count": len(matches), "results": matches,
                "search_type": "regex", "file_path": file_path}
    # Default: full-text content search
    from src.core.execution_profile import lean_local_event_active

    if lean_local_event_active():
        # A cold clone has no FTS rows yet. The first sync builds the disposable
        # index; steady-state sync is incremental (measured ~250 ms / 7.5k
        # notes). Over-fetch so a canonical analysis buried under repeated
        # receipts remains available to the authority reranker.
        await svc.sync(force=False)
        requested = max(1, int(limit))
        results = await svc.search(query, limit=min(200, requested * 10))
        results = _canonical_first(results, requested)
        return {
            "query": query,
            "count": len(results),
            "results": results,
            "search_type": "content",
            "ranking_profile": "canonical_first",
        }
    results = await svc.search(query, limit=limit)
    return {"query": query, "count": len(results), "results": results,
            "search_type": "content"}


async def vault_backlinks(path: str) -> dict:
    """List the notes that link TO ``path`` (its backlinks / inbound links)."""
    svc = get_service()
    links = await svc.backlinks(path)
    return {"path": path, "backlinks": links, "count": len(links)}


async def vault_dream() -> dict:
    """Run a DREAM-MODE maintenance pass over the memory vault NOW (the same
    routine that normally runs on a schedule): grade every note, mechanically
    auto-fix what code safely can (formatting, dates, scaffold missing
    frontmatter), regenerate the derived llms.txt + showcase, and commit it
    all. Returns ``before``/``after`` health, what was auto-fixed, and
    ``open_suggestions`` — the HARDER issues that need YOUR judgement (orphans
    to link, duplicates to merge, over-long notes to split, missing summaries
    to write, broken links to fix). After this returns, RESOLVE those
    suggestions by writing/merging/linking notes, then call vault_gate (or
    vault_dream) again to confirm the vault improved."""
    svc = get_service()
    summary = await svc.maintenance(apply_fixes=True, regenerate=True)
    try:
        # Commit the mechanical fixes (the derived files were committed by the
        # pass) with dream provenance.
        await svc.autocommit(origin={**_origin("vault_dream"), "kind": "dream"})
    except Exception:  # noqa: BLE001
        pass
    return summary


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
