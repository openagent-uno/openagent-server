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
    ``.obsidian``) — never gated and never indexed as content.

    A DOT-DIRECTORY is excluded by rule, not by name. The named list already
    enumerated three of them (``.obsidian``, ``.git``, ``.openagent``) and had
    already drifted: Obsidian's ``.trash/`` was missing, so the Python gate
    graded deleted notes — a fact only the Node writer knew, via the
    ``.``-prefix half of the heuristic that unification replaced. An
    enumeration of "tooling folders" is a list that goes stale every time a
    tool is added; the convention it was approximating is simply that a
    dot-directory is machinery, never memory.
    """
    norm = rel_path.replace("\\", "/").lstrip("/")
    parts = norm.split("/")
    if any(part.startswith(".") for part in parts[:-1]):
        return True
    return any(part in excluded_folders for part in parts[:-1]) or (
        parts[0] in excluded_folders
    )


def is_in_quality_scope(rel_path: str, excluded_folders: tuple[str, ...],
                        raw_prefixes: tuple[str, ...],
                        journal_root: str) -> bool:
    """THE answer to "does the quality system apply to this note?".

    There used to be three answers and they disagreed. Measured on the owner's
    real 2,116-note vault:

      * ``validate.ts:shouldValidate`` skipped any path segment starting with
        ``_`` or ``.``, plus ``templates/``;
      * this predicate (as inlined by ``gate.py``) skipped only excluded +
        raw folders;
      * ``VaultService._enforce_write`` skipped nothing at all.

    So **413 notes — 20% of the vault — were skipped by the Node writer and
    graded by the Python gate**: 404 under ``_inherited-from-lyra/**`` and 16
    ``_index.md`` hubs (7 of them in both sets). The agent wrote those notes
    through a writer that never validated them, and the gate then failed them
    forever with no path to green. Worse, the ``_``-prefix heuristic was
    simply wrong about what it caught: ``_index.md`` hubs are first-class
    notes the gate has explicit support for (``is_index_note`` exempts them
    from orphan/min_links precisely BECAUSE it grades them), and
    ``_inherited-from-lyra/`` is 404 notes of real memory, not scratch.

    Folder semantics are a vault-taxonomy question, so the taxonomy owns the
    answer and the other two derive from it: Python calls this function, and
    the TypeScript writer reads ``scope.generated.ts``, which is rendered from
    the same values by ``render_scope_typescript`` and pinned by
    ``test_vault_twins``.
    """
    return (not is_excluded(rel_path, excluded_folders)
            and not is_raw(rel_path, raw_prefixes, journal_root))


_TS_BANNER = """\
// AUTO-GENERATED by src/memory/vault/taxonomy.py -- DO NOT EDIT BY HAND.
//
// Regenerate:  .venv/bin/python -m src.memory.vault.taxonomy
// Pinned by:   scripts/tests/test_vault_twins.py (CI fails on drift)
//
// WHY THIS FILE EXISTS: "which notes does the quality system apply to?" is a
// vault-taxonomy question, and the taxonomy is declared in Python. This file
// is the derived copy the Node writer reads, because it runs as a separate
// process in another language and cannot call the declaration directly. It is
// generated rather than hand-kept: the hand-kept version drifted, and 413
// notes (20% of the owner's real vault) were skipped by this writer while the
// Python gate graded and failed them.
"""


def render_scope_typescript(excluded_folders: tuple[str, ...],
                            raw_prefixes: tuple[str, ...],
                            journal_root: str) -> str:
    """Render the scope declaration as the TypeScript module the vendored
    vault MCP imports. Pure string building — the test regenerates this and
    byte-compares it against what is on disk, so drift cannot land."""
    import json

    def arr(vals: tuple[str, ...]) -> str:
        return ", ".join(json.dumps(v) for v in vals)

    return _TS_BANNER + f'''
export const EXCLUDED_FOLDERS: readonly string[] = [{arr(excluded_folders)}];
export const RAW_PREFIXES: readonly string[] = [{arr(raw_prefixes)}];
export const JOURNAL_ROOT = {json.dumps(journal_root)};

const norm = (p: string): string => p.replace(/\\\\/g, "/").replace(/^\\/+/, "");

/** Port of taxonomy.is_journal_note. */
export function isJournalNote(relPath: string): boolean {{
  const p = norm(relPath);
  const root = JOURNAL_ROOT.replace(/\\\\/g, "/").replace(/^\\/+|\\/+$/g, "");
  return p === root || p.startsWith(root + "/");
}}

/** Port of taxonomy.is_excluded. A dot-directory is excluded by rule, not by
 *  name: an enumeration of tooling folders goes stale (it had already missed
 *  Obsidian's .trash/). */
export function isExcluded(relPath: string): boolean {{
  const parts = norm(relPath).split("/");
  const parents = parts.slice(0, -1);
  if (parents.some((s) => s.startsWith("."))) return true;
  return (
    parents.some((s) => EXCLUDED_FOLDERS.includes(s)) ||
    EXCLUDED_FOLDERS.includes(parts[0] ?? "")
  );
}}

/** Port of taxonomy.is_raw. The journal is carved back in: it is dynamic
 *  memory, not scratch. */
export function isRaw(relPath: string): boolean {{
  const p = norm(relPath);
  if (isJournalNote(p)) return false;
  return RAW_PREFIXES.some((prefix) => p.startsWith(prefix));
}}

/** Port of taxonomy.is_in_quality_scope. */
export function isInQualityScope(relPath: string): boolean {{
  return !isExcluded(relPath) && !isRaw(relPath);
}}
'''


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


# The scope values themselves are declared once, as the ``GateConfig``
# defaults in ``model.py``; this renders them for the Node side. Imported
# lazily inside ``__main__`` so ``taxonomy`` stays dependency-free for
# ``parser`` / ``model``.
SCOPE_TS_PATH = "src/mcp/servers/vault/src/scope.generated.ts"


def _generated_scope_typescript() -> str:
    from src.memory.vault.model import GateConfig
    cfg = GateConfig()
    return render_scope_typescript(
        cfg.excluded_folders, cfg.raw_prefixes, cfg.journal_root)


if __name__ == "__main__":  # pragma: no cover - developer entry point
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[3]
    out = repo_root / SCOPE_TS_PATH
    out.write_text(_generated_scope_typescript())
    print(f"wrote {out}")
