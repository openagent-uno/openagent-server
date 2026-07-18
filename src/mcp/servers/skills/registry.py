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
    """Frontmatter-derived metadata for one skill (never the full body)."""
    name: str
    description: str
    category: str
    path: Path  # the SKILL.md file itself

    @property
    def directory(self) -> Path:
        """The skill's folder — where bundled files live alongside SKILL.md."""
        return self.path.parent


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
    return SkillMeta(name=name, description=description,
                     category=category, path=md_path)


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

    # ── index render (goes into the CACHED system prefix) ─────────────
    def render_skills_index(self) -> str:
        """Compact, byte-stable index grouped by category. Cached."""
        if self._index_cache is None:
            self._index_cache = self._build_index()
        return self._index_cache

    def _build_index(self) -> str:
        self._ensure_loaded()
        skills = self._skills or []
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
