"""Scaffold a vault's folder system and the canon workflow.

Company-Brain Prompt 1 sets up the eleven folders by hand; this does it in
one call, plus the journal sub-tree (Prompt 13) and the canon workspace
(Prompt 2). Idempotent — existing folders/files are left untouched.

The **canon** is the one place the methodology is inherently AI-driven, not
code-checkable: "extract the hard facts from ``sources/`` into a single
coherent canon, numbers must reconcile, invent nothing." A script cannot
verify that a sentence is faithful to a source document. So the contribution
here is *structure + guidance*: a place for the raw material (``sources/``),
a place for the working canon (``workspace/_canon/``), a README explaining
the flow, and templates — and then the gate guarantees the atomic notes
*derived* from the canon are well-formed (frontmatter, links, no orphans).
"""
from __future__ import annotations

from pathlib import Path

from src.memory.vault import taxonomy

_CANON_README = """\
# Canon — the single source of hard facts

This is a *working* folder, skipped by the quality gate. The workflow:

1. Drop raw material (docs, notes, exports) into `sources/`.
2. Extract the **hard facts** into `canon.md` here — one coherent picture:
   identity, products, people, clients (with numbers), KPIs. Rules:
   - Invent nothing. If it is not in `sources/`, it does not go in the canon.
   - Numbers must reconcile (client ARR sums to total ARR, etc.).
3. Turn the canon into **atomic notes** in the content folders (`self/`,
   `entities/`, `data/`, `concepts/`, …) — one idea per note, dense
   `[[wikilinks]]`, complete frontmatter. The gate (`vault_gate`) then keeps
   those notes honest: no orphans, no broken links, one connected graph.

The canon is the bridge from raw material to the brain; the notes are the
brain. Keep this file as the audit trail of where the facts came from.
"""

_NOTE_TEMPLATE = """\
---
title:
summary:
tags: [{folder}]
status: active
created:
updated:
related:
---

"""

_SESSION_TEMPLATE = """\
---
title:
summary:
tags: [workspace, type/session]
status: done
created:
updated:
related:
---

## Done

## Decided

## Open
"""

_DAILY_TEMPLATE = """\
---
title:
summary:
tags: [workspace, type/daily]
status: done
created:
updated:
related:
---

## Done

## Decided

## Open

## Sessions
"""


def scaffold(vault_root: Path, journal_root: str = "workspace/journal") -> dict:
    """Create the canonical folder system + canon + journal templates.
    Returns the list of paths created (idempotent — skips what exists)."""
    created: list[str] = []

    def _mkdir(rel: str) -> None:
        p = vault_root / rel
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(rel + "/")

    def _write(rel: str, content: str) -> None:
        p = vault_root / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            created.append(rel)

    for folder in taxonomy.ALL_FOLDERS:
        _mkdir(folder)

    # Journal sub-tree (Prompt 13) + reusable templates.
    _mkdir(f"{journal_root}/sessions")
    _mkdir(f"{journal_root}/daily")
    _write(f"{journal_root}/_templates/sessione.md", _SESSION_TEMPLATE)
    _write(f"{journal_root}/_templates/daily.md", _DAILY_TEMPLATE)

    # Canon workspace (Prompt 2).
    _mkdir("workspace/_canon")
    _write("workspace/_canon/README.md", _CANON_README)

    # A generic note template per content folder.
    _write("workspace/_templates/note.md", _NOTE_TEMPLATE.format(folder="self"))

    return {"created": created, "count": len(created)}
