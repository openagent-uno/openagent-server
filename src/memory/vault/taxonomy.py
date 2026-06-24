"""The Company-Brain folder taxonomy and path classification helpers.

The taxonomy is *advisory* in OpenAgent: existing vaults use free-form
folders (``projects/<repo>/...``, ``pending-automations/`` …), so the gate
never hard-fails on layout. These helpers let the taxonomy rule emit gentle
``info`` nudges and let the structural rules know which folders are raw
material (skip) vs. real memory (gate), and which notes are hubs / journal
entries (exempt from some rules).
"""
from __future__ import annotations

from pathlib import PurePosixPath

# The eleven Company-Brain folders, one per "job" the brain does.
ALL_FOLDERS: tuple[str, ...] = (
    "self", "areas", "projects", "sources", "concepts", "docs",
    "entities", "data", "code", "outputs", "workspace",
)

# Folders that hold finished, gateable notes (Company-Brain gates these nine;
# it skips sources/ raw material and workspace/ scratch).
CONTENT_FOLDERS: tuple[str, ...] = (
    "self", "areas", "projects", "concepts", "docs",
    "entities", "data", "code", "outputs",
)

# Conventional domain prefixes the tutorial uses for unique, self-describing
# filenames (advisory — used only to *suggest* a prefix, never to fail).
DOMAIN_PREFIXES: tuple[str, ...] = (
    "self-", "area-", "progetto-", "project-", "fonte-", "source-",
    "concetto-", "concept-", "doc-", "cliente-", "client-", "persona-",
    "person-", "prodotto-", "product-", "kpi-", "dato-", "data-",
    "entita-", "entity-", "strumento-", "tool-", "output-", "code-",
    "script-", "sessione-", "session-",
)


def top_folder(rel_path: str) -> str:
    """Return the top-level folder of a vault-relative path, or "" if the
    note sits at the vault root."""
    parts = PurePosixPath(rel_path).parts
    return parts[0] if len(parts) > 1 else ""


def stem_of(rel_path: str) -> str:
    """Filename without the ``.md`` extension (original case)."""
    name = PurePosixPath(rel_path).name
    return name[:-3] if name.lower().endswith(".md") else name


def is_index_note(rel_path: str) -> bool:
    """Hub note: ``_index.md`` or ``<folder>/_index.md`` or any ``*_index``.

    Hubs are map-notes; the gate exempts them from the orphan and
    minimum-links rules (they are linked *from* details, and link out by
    nature, but Company-Brain explicitly excludes them so the counts stay
    honest)."""
    stem = stem_of(rel_path).lower()
    return stem == "_index" or stem.endswith("_index") or stem == "index"


def is_journal_note(rel_path: str, journal_root: str = "workspace/journal") -> bool:
    norm = rel_path.replace("\\", "/").lstrip("/")
    root = journal_root.replace("\\", "/").strip("/")
    return norm == root or norm.startswith(root + "/")


def is_raw(rel_path: str, raw_prefixes: tuple[str, ...] = ("sources/", "workspace/"),
           journal_root: str = "workspace/journal") -> bool:
    """True for raw-material / scratch notes the structural gate skips.

    ``workspace/journal`` is carved back *in* — it is dynamic memory, not
    scratch, and earns the journal rule. So a note under ``workspace/`` is
    raw unless it is under the journal root."""
    norm = rel_path.replace("\\", "/").lstrip("/")
    if is_journal_note(norm, journal_root):
        return False
    return any(norm.startswith(p) for p in raw_prefixes)


def is_excluded(rel_path: str, excluded_folders: tuple[str, ...]) -> bool:
    """True for paths under an excluded folder (e.g. ``_showcase``,
    ``.obsidian``) — never gated and never indexed as content."""
    norm = rel_path.replace("\\", "/").lstrip("/")
    parts = norm.split("/")
    return any(part in excluded_folders for part in parts[:-1]) or (
        parts[0] in excluded_folders
    )


def looks_like_channel(folder: str, channel_hints: tuple[str, ...]) -> bool:
    return folder.lower() in channel_hints


def has_domain_prefix(stem: str) -> bool:
    low = stem.lower()
    return any(low.startswith(p) for p in DOMAIN_PREFIXES) or ("-" in stem)


def suggested_prefix(folder: str) -> str:
    """A reasonable filename prefix for a note in ``folder`` (advisory)."""
    mapping = {
        "self": "self-", "areas": "area-", "projects": "progetto-",
        "concepts": "concetto-", "docs": "doc-", "entities": "entita-",
        "data": "dato-", "code": "code-", "outputs": "output-",
    }
    return mapping.get(folder, (folder + "-") if folder else "")
