"""In-memory scan + index render for the file-backed skills directory.

Deliberately small: a ``SkillsRegistry`` globs ``<root>/*/SKILL.md``, parses
each file's YAML frontmatter (REUSING the vault parser so one code path
answers "is this frontmatter valid?"), and renders a compact index grouped
by category.

Two invariants matter:

  * **No coupling to VaultService / VaultIndex.** Those carry FTS / git /
    quality-gate machinery skills do not want. We only borrow the two pure
    frontmatter functions.
  * **Byte-stable index.** ``render_skills_index`` is injected into the
    CACHED system prefix, so it must be identical across every session /
    turn on a box and carry NO volatile tokens (timestamps, session ids).
    It derives solely from on-disk file contents and is cached until an
    explicit :meth:`load` / :meth:`reload`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.memory.vault.parser import (
    FrontmatterSyntaxError,
    load_frontmatter_yaml,
    split_frontmatter,
)

logger = logging.getLogger(__name__)

# Fallback bucket for a skill whose frontmatter names no ``category``.
DEFAULT_CATEGORY = "general"


@dataclass(frozen=True)
class SkillMeta:
    """Frontmatter-derived metadata for one skill (never the full body).

    ``created_by`` and ``status`` back the skill-curator's two invariants:

      * **Provenance.** ``created_by`` records who authored the skill —
        ``"agent"`` for skills the agent wrote via ``skill_manage``, ``None``
        for seed/user skills. The curator only ever touches agent-authored
        skills (see :meth:`SkillsRegistry.agent_authored`), so a consolidation
        pass can never merge or archive curated seed content.
      * **Archival.** ``status == "archived"`` retires a skill WITHOUT deleting
        it: the file stays on disk (auditable, reversible) but it is dropped
        from :meth:`SkillsRegistry.render_skills_index`, so an archived skill
        never reaches the frozen system-prompt index.

    Neither field is rendered into the index — the index derives solely from
    ``name`` / ``description`` / ``category`` — so provenance/status changes
    never perturb the byte-stable cached prompt.
    """
    name: str
    description: str
    category: str
    path: Path  # the SKILL.md file itself
    created_by: str | None = None
    status: str | None = None

    @property
    def directory(self) -> Path:
        """The skill's folder — where bundled files live alongside SKILL.md."""
        return self.path.parent

    @property
    def is_agent_authored(self) -> bool:
        """True when the AGENT wrote this skill (frontmatter ``created_by:
        agent``). Seed/user skills carry no such stamp and return False."""
        return (self.created_by or "").strip().lower() == "agent"

    @property
    def is_archived(self) -> bool:
        """True when this skill has been retired (frontmatter ``status:
        archived``). Archived skills are kept out of the active index."""
        return (self.status or "").strip().lower() == "archived"

    @property
    def is_hub(self) -> bool:
        """True when this skill was pulled from a Skills-Hub tap (frontmatter
        ``created_by: hub``). Hub skills are owned by their external upstream:
        because the curator boundary is ``created_by == "agent"``, they are
        automatically excluded from :meth:`SkillsRegistry.agent_authored` and
        the curator can never merge or archive one. This property is a LABEL
        only — nothing in the index render depends on it."""
        return (self.created_by or "").strip().lower() == "hub"


def parse_skill_file(md_path: Path) -> SkillMeta | None:
    """Parse one ``SKILL.md`` into a :class:`SkillMeta`, or ``None`` when it
    is malformed and must be skipped GRACEFULLY (never crash the scan):

      * unreadable file,
      * no ``---`` frontmatter fence,
      * frontmatter that is not valid YAML,
      * frontmatter with a missing / empty ``name``.

    A skipped file is logged at debug and simply does not appear in the
    index — one broken skill can never take the registry down.
    """
    try:
        content = md_path.read_text(errors="replace")
    except OSError as exc:
        logger.debug("skills: unreadable %s (%s) — skipping", md_path, exc)
        return None

    raw_fm, _body = split_frontmatter(content)
    if raw_fm is None:
        logger.debug("skills: %s has no frontmatter — skipping", md_path)
        return None

    try:
        meta = load_frontmatter_yaml(raw_fm)
    except FrontmatterSyntaxError as exc:
        logger.debug("skills: %s has malformed frontmatter (%s) — skipping",
                     md_path, exc)
        return None
    if not isinstance(meta, dict):
        return None

    name = str(meta.get("name") or "").strip()
    if not name:
        logger.debug("skills: %s frontmatter has no name — skipping", md_path)
        return None

    description = str(meta.get("description") or "").strip()
    category = (str(meta.get("category") or "").strip() or DEFAULT_CATEGORY)
    created_by = str(meta.get("created_by") or "").strip() or None
    status = str(meta.get("status") or "").strip() or None
    return SkillMeta(name=name, description=description,
                     category=category, path=md_path,
                     created_by=created_by, status=status)


class SkillsRegistry:
    """A cached scan of ``<skills_root>/*/SKILL.md``.

    Mirrors the MCP pool's catalog-caching discipline: the rendered index
    is cached in ``_index_cache`` and invalidated only on an explicit
    :meth:`load` / :meth:`reload` — never per turn.
    """

    def __init__(self, skills_root: str | Path):
        self.skills_root = Path(skills_root)
        self._skills: list[SkillMeta] | None = None
        self._index_cache: str | None = None

    # ── scan ──────────────────────────────────────────────────────────
    def _scan(self) -> list[SkillMeta]:
        root = self.skills_root
        if not root.is_dir():
            return []
        out: list[SkillMeta] = []
        # sorted() gives a stable order → a byte-stable index.
        for md in sorted(root.glob("*/SKILL.md")):
            meta = parse_skill_file(md)
            if meta is not None:
                out.append(meta)
        return out

    def _ensure_loaded(self) -> None:
        if self._skills is None:
            self._skills = self._scan()

    def load(self) -> None:
        """(Re)scan disk NOW and invalidate the cached index. Call this on
        boot and whenever the skills directory may have changed on disk."""
        self._skills = self._scan()
        self._index_cache = None

    # ``reload`` reads better at the invalidation call sites.
    reload = load

    # ── reads ─────────────────────────────────────────────────────────
    def skills(self) -> list[SkillMeta]:
        self._ensure_loaded()
        return list(self._skills or [])

    def get(self, name: str) -> SkillMeta | None:
        """Case-insensitive lookup by frontmatter ``name``."""
        self._ensure_loaded()
        key = (name or "").strip().lower()
        for s in self._skills or []:
            if s.name.lower() == key:
                return s
        return None

    def agent_authored(self) -> list[SkillMeta]:
        """The skills the skill-curator MAY touch: those the agent itself
        wrote (frontmatter ``created_by: agent``).

        This is the provenance boundary. Seed and user skills carry no
        ``created_by`` stamp, so they are excluded here and a consolidation
        pass — which discovers its work set through this filter — can never
        merge away or archive hand-curated seed content. Archived agent
        skills are still returned (the curator may legitimately re-open or
        delete one it retired earlier); callers that want only the LIVE set
        can filter on :attr:`SkillMeta.is_archived`.
        """
        self._ensure_loaded()
        return [s for s in (self._skills or []) if s.is_agent_authored]

    # ── index render (goes into the CACHED system prefix) ─────────────
    def render_skills_index(self) -> str:
        """Compact, byte-stable index grouped by category. Cached."""
        if self._index_cache is None:
            self._index_cache = self._build_index()
        return self._index_cache

    def _build_index(self) -> str:
        self._ensure_loaded()
        # Archived skills are retired: their file stays on disk (auditable,
        # reversible) but it is dropped from the index so the frozen
        # system-prompt prefix only ever advertises live skills. This is a
        # deterministic function of disk contents, so byte-stability holds.
        skills = [s for s in (self._skills or []) if not s.is_archived]
        if not skills:
            return "_(no skills installed.)_"

        by_cat: dict[str, list[SkillMeta]] = {}
        for s in skills:
            by_cat.setdefault(s.category, []).append(s)

        lines: list[str] = []
        for cat in sorted(by_cat):
            lines.append(f"**{cat}**")
            for s in sorted(by_cat[cat], key=lambda x: x.name.lower()):
                desc = s.description or "(no description)"
                lines.append(f"- ``{s.name}``: {desc}")
            lines.append("")  # blank line between categories
        return "\n".join(lines).rstrip()
