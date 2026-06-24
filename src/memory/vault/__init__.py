"""OpenAgent vault quality subsystem.

The memory vault is a folder of human-readable Markdown notes (vision §5,
"Markdown over opaque stores"). This package adds the *evaluation* layer the
vision asks for but never enforced before: a code-first quality gate, an
incremental index that scales to hundreds of thousands of notes, derived
artifacts (``llms.txt`` / showcase), and an auto-doctor that mechanically
fixes the issues a script can fix and hands the rest to the AI / dream mode.

The methodology is the "Company Brain" second-brain discipline: atomic notes,
strict frontmatter, dense wikilinks, no orphans, no broken links, one
connected graph. The Markdown files remain the single source of truth — the
SQLite index here is a rebuildable cache, never authoritative.

Public surface:
- ``VaultService`` — async facade used by REST, the CLI, dream mode, and the
  native ``vault-gate`` MCP.
- ``run_gate`` / ``GateReport`` — the gate itself (pure, synchronous, fast).
- ``VaultIndex`` — the incremental, FTS5-backed index.
"""
from __future__ import annotations

from src.memory.vault.model import (
    GateConfig,
    GateReport,
    Note,
    RULES,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARN,
    Violation,
)

__all__ = [
    "GateConfig",
    "GateReport",
    "Note",
    "RULES",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARN",
    "Violation",
]
