"""The vault quality gate — pure, synchronous, code-enforced.

This is the OpenAgent realisation of the Company-Brain "gate di qualità"
(tutorial Prompt 5): it passes every note through a fixed set of rules and
returns a structured report. It runs entirely off the ``VaultIndex`` (no
file re-reads), so a full gate over a 100k-note vault is a handful of SQL
queries plus one O(N) pass — not 100k disk reads.

What it does NOT do: pass judgement that needs a model (is this note *truly*
a duplicate of that one? should this 400-line note be split here or there?).
Those it surfaces as ``suggestion``-bearing violations for the doctor / dream
mode / the agent to act on. Code finds and reports; code fixes what code can;
the rest is handed up with enough structure to fix.

Rule coverage (every Company-Brain gate rule + extras):
  frontmatter, atomicity (<=max_lines), min_links (>=3 unique real),
  broken_link, orphan, connectivity (single component), filename
  (uniqueness + prefix), wikilink_format (related one-line / no inner
  spaces), date_format (YYYY-MM-DD), taxonomy (tags[0]==folder, channels as
  tags), duplicate (exact + title), journal_link (anchored to an entity),
  em_dash.
"""
from __future__ import annotations

import re
import time

from src.memory.vault import taxonomy
from src.memory.vault.index import VaultIndex
from src.memory.vault.model import (
    GateConfig,
    GateReport,
    Note,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    Violation,
)

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_MAX_ISLANDS = 100  # cap connectivity violations so a shattered vault can't flood


def _is_valid_iso(val: str) -> bool:
    """True for a real ``YYYY-MM-DD`` date — digit shape AND in-range month/
    day, so a bogus ``2024-13-04`` is still flagged."""
    m = _DATE_RE.match(val)
    if not m:
        return False
    _y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return 1 <= mo <= 12 and 1 <= d <= 31


def run_gate(index: VaultIndex, config: GateConfig | None = None) -> GateReport:
    cfg = config or GateConfig()
    t0 = time.monotonic()
    violations: list[Violation] = []

    def emit(rule: str, path: str, message: str, *, fixable: bool = False,
             line: int | None = None, suggestion: str | None = None,
             related_path: str | None = None) -> None:
        if not cfg.is_enabled(rule):
            return
        violations.append(Violation(
            rule=rule, severity=cfg.severity_for(rule), path=path,
            message=message, fixable=fixable, line=line,
            suggestion=suggestion, related_path=related_path,
        ))

    notes = index.all_notes()
    # The set of notes the structural rules apply to (skip raw material and
    # excluded folders, but keep journal notes — they get journal rules).
    gated: list[Note] = [
        n for n in notes
        if not taxonomy.is_excluded(n.path, cfg.excluded_folders)
        and not taxonomy.is_raw(n.path, cfg.raw_prefixes, cfg.journal_root)
    ]
    gated_paths = {n.path for n in gated}

    degree = index.resolved_degree()
    inlinks = index.inlink_counts()
    anchor_status = index.journal_anchor_status()

    # ── per-note rules ────────────────────────────────────────────────
    for n in gated:
        # R1 frontmatter completeness
        if n.missing_frontmatter_fields:
            miss = ", ".join(n.missing_frontmatter_fields)
            emit("frontmatter", n.path,
                 f"missing frontmatter field(s): {miss}", fixable=True,
                 suggestion=f"add {miss} to the YAML frontmatter")

        # R2 atomicity — one idea, under the line limit
        if n.line_count > cfg.max_lines:
            emit("atomicity", n.path,
                 f"{n.line_count} body lines exceeds the {cfg.max_lines}-line "
                 "limit — split into atomic notes",
                 suggestion="split this note into smaller atomic notes, one "
                            "idea each, and cross-link them")

        # R3 minimum outgoing links (content notes only)
        if not n.is_index and not n.is_journal:
            d = degree.get(n.path, 0)
            if d < cfg.min_outlinks:
                emit("min_links", n.path,
                     f"{d} resolved outgoing link(s); want >= {cfg.min_outlinks}",
                     suggestion="link this note to >=3 distinct existing notes "
                                "(its hub and neighbouring notes)")

        # R5 orphan (no inbound link) — hubs and journal entries exempt
        if not n.is_index and not n.is_journal and inlinks.get(n.path, 0) == 0:
            emit("orphan", n.path,
                 "orphan — no other note links here",
                 suggestion="link to this note from its hub (_index) or a "
                            "related note so it joins the graph")

        # R8 wikilink / related formatting
        #
        # This rule USED TO also fire on ``n.related_multiline``, demanding a
        # block-style ``related:`` list be collapsed onto one comma-separated
        # line. That requirement is deleted, not relaxed: ``related: [[a]],
        # [[b]]`` is not valid YAML, our own vault MCP's gray-matter reader
        # throws on it (and silently degrades the note to ``frontmatter: {}``),
        # ``write_note`` rejects it, and Obsidian Properties — which §5 says
        # the vault must render in — are strict YAML. The multi-line list the
        # rule flagged is the *correct* form. See ``doctor.py`` where
        # ``_collapse_related`` was removed for the measured damage (13 -> 21
        # permanent ``date_format`` violations on the owner's real vault).
        if n.spaced_wikilinks:
            sample = ", ".join(n.spaced_wikilinks[:3])
            emit("wikilink_format", n.path,
                 f"wikilink(s) with spaces inside the brackets: {sample}",
                 fixable=True, suggestion="remove spaces inside [[ ]]")

        # R9 date format
        for field_name in ("created", "updated"):
            val = getattr(n, field_name)
            if val and not _is_valid_iso(val):
                emit("date_format", n.path,
                     f"`{field_name}: {val}` is not an absolute YYYY-MM-DD date",
                     fixable=True,
                     suggestion=f"set {field_name} to a YYYY-MM-DD date")

        # R10 taxonomy (advisory; off unless enforce_taxonomy)
        if cfg.enforce_taxonomy and n.folder:
            if taxonomy.looks_like_channel(n.folder, cfg.channel_hints):
                emit("taxonomy", n.path,
                     f"`{n.folder}/` looks like a channel — channels should be "
                     f"tags (channel/{n.folder}), not folders")
            elif n.tags and n.tags[0] != n.folder:
                emit("taxonomy", n.path,
                     f"tags[0] is '{n.tags[0]}' but the folder is '{n.folder}' "
                     "— the first tag should be the folder name")
            if n.folder in taxonomy.CONTENT_FOLDERS and not taxonomy.has_domain_prefix(n.stem):
                emit("filename", n.path,
                     f"filename '{n.stem}' has no domain prefix "
                     f"(e.g. {taxonomy.suggested_prefix(n.folder)}…)")

        # R12 em dash (journal notes always; everywhere if configured)
        if n.body_has_em_dash and (n.is_journal or cfg.check_em_dash):
            emit("em_dash", n.path,
                 "body contains an em dash (—); prefer -- or a comma",
                 fixable=True)

        # R13 journal anchoring
        if n.is_journal and anchor_status.get(n.path) is False:
            emit("journal_link", n.path,
                 "journal note is not anchored to any static entity note",
                 suggestion="add a [[wikilink]] to a real entity (client, KPI, "
                            "project) this session touched")

    # ── cross-note rules ──────────────────────────────────────────────
    # R4 broken links
    for src, target in index.broken_links():
        if src in gated_paths:
            emit("broken_link", src,
                 f"wikilink [[{target}]] points to a note that does not exist",
                 suggestion=f"create the note, fix the target, or remove [[{target}]]",
                 related_path=target)

    # R7 filename uniqueness (stem collisions)
    for stem, paths in index.stem_collisions().items():
        relevant = [p for p in paths if p in gated_paths]
        if len(relevant) > 1:
            for p in relevant:
                others = ", ".join(o for o in relevant if o != p)
                emit("filename", p,
                     f"filename '{stem}' is not unique — also at: {others}",
                     suggestion="give each note a unique, domain-prefixed name",
                     related_path=relevant[0] if relevant[0] != p else relevant[-1])

    # R11 duplicates — exact (content hash) then title collisions. Track the
    # exact-dup "followers" (not the kept copy) so they aren't re-reported as
    # title dupes, WITHOUT dropping a title-only dup that merely shares a
    # title with one member of an exact-dup pair.
    hash_dup_followers: set[str] = set()
    for _h, paths in index.hash_collisions().items():
        relevant = [p for p in paths if p in gated_paths]
        if len(relevant) > 1:
            keep, *dups = relevant
            for p in dups:
                hash_dup_followers.add(p)
                emit("duplicate", p,
                     f"byte-identical duplicate of {keep}",
                     suggestion=f"merge into {keep} and delete this copy",
                     related_path=keep)
    for _t, paths in index.title_collisions().items():
        relevant = [p for p in paths if p in gated_paths]
        if len(relevant) > 1:
            keep = relevant[0]
            for p in relevant[1:]:
                if p in hash_dup_followers:
                    continue  # already reported as an exact duplicate
                emit("duplicate", p,
                     f"shares a title with {keep} — possible duplicate",
                     suggestion=f"merge with {keep} if they cover the same thing",
                     related_path=keep)

    # R6 connectivity — one violation per non-largest island, capped.
    # Islands made up entirely of journal notes are skipped: an unanchored
    # journal entry is already reported by journal_link, and an anchored one
    # joins the main graph through its entity, so it is never its own island.
    journal_paths = {n.path for n in gated if n.is_journal}
    comps = index.components()
    gated_comps = [[p for p in c if p in gated_paths] for c in comps]
    gated_comps = [c for c in gated_comps if c]
    # ``components()`` sorts by FULL-graph size (raw notes included), so after
    # filtering to gated notes the biggest gated cluster may not be first —
    # re-sort so [0] is the true main component and the rest are the islands.
    gated_comps.sort(key=len, reverse=True)
    reportable = [c for c in gated_comps[1:] if not set(c) <= journal_paths]
    island_count = len(reportable)
    for comp in reportable[:_MAX_ISLANDS]:
        rep = _component_representative(comp, notes)
        emit("connectivity", rep,
             f"disconnected island of {len(comp)} note(s) — not linked to "
             "the main graph",
             suggestion="link a note in this cluster to the main graph")

    # ── tally ─────────────────────────────────────────────────────────
    errors = sum(1 for v in violations if v.severity == SEVERITY_ERROR)
    warns = sum(1 for v in violations if v.severity == "warn")
    infos = sum(1 for v in violations if v.severity == SEVERITY_INFO)

    stats = index.stats()
    stats.update({
        "gated_notes": len(gated),
        "islands": island_count,
        "rules_run": sorted({v.rule for v in violations}),
    })

    return GateReport(
        ok=(errors == 0),
        error_count=errors,
        warn_count=warns,
        info_count=infos,
        note_count=len(gated),
        violations=violations,
        stats=stats,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


def _component_representative(component: list[str], notes: list[Note]) -> str:
    """Pick the most meaningful note to attach an island violation to: an
    _index hub if present, else the shortest path (usually the most central)."""
    index_notes = [p for p in component if taxonomy.is_index_note(p)]
    if index_notes:
        return sorted(index_notes, key=len)[0]
    return sorted(component, key=len)[0]
