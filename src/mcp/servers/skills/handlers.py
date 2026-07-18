"""Async handlers behind the ``skills`` MCP tools.

Progressive disclosure: the framework prompt carries only the skills
*index* (category → ``name: description``). These handlers are the
on-demand layer — load a full body (``skill_view``), find a skill
(``skill_search``), or create/update/remove one on disk (``skill_manage``).

Each call resolves the skills directory itself via
``paths.default_skills_path()`` (which honours ``OPENAGENT_SKILLS_PATH``),
so the in-process MCP needs no config wiring. Reads build a fresh
``SkillsRegistry`` per call: the registry the AGENT holds is a frozen
snapshot for the cached prompt and is deliberately NOT mutated here — a
``skill_manage`` write lands on disk and surfaces in the prompt index on
the next boot/reload, exactly like a vault note never rewrites the live
system prompt mid-session.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from src.core.paths import default_skills_path
from src.memory.vault.parser import split_frontmatter
from src.mcp.servers.skills.registry import SkillsRegistry, parse_skill_file

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# The provenance stamp written into the frontmatter of every skill the AGENT
# creates. It is the boundary the skill-curator respects: skills WITHOUT this
# stamp are seed/user content and are off-limits to consolidation. A constant
# (never a timestamp), so it can safely live in a file that feeds the index —
# the index render ignores it, and it never changes between writes.
AGENT_PROVENANCE = "agent"


def _skills_root() -> Path:
    return default_skills_path()


def _registry() -> SkillsRegistry:
    reg = SkillsRegistry(_skills_root())
    reg.load()
    return reg


def _slug(name: str) -> str:
    """Directory-safe slug for a skill name. The frontmatter ``name`` is
    what identifies a skill; the folder is just where its files live."""
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or "skill"


def _bundled_files(skill_dir: Path) -> list[str]:
    """Names of files bundled alongside SKILL.md (relative to the folder)."""
    out: list[str] = []
    for p in sorted(skill_dir.rglob("*")):
        if p.is_file() and p.name != "SKILL.md":
            out.append(p.relative_to(skill_dir).as_posix())
    return out


# ── skill_view ────────────────────────────────────────────────────────

async def skill_view(name: str) -> dict:
    """Load the FULL body of one skill by its frontmatter ``name``. This is
    the on-demand half of progressive disclosure: the system-prompt index
    lists only ``name: description``; call this to read the whole SKILL.md
    (instructions, steps, examples) plus any bundled files, right before you
    act on the skill."""
    reg = _registry()
    meta = reg.get(name)
    if meta is None:
        available = sorted(s.name for s in reg.skills())
        return {
            "ok": False,
            "error": f"No skill named {name!r}.",
            "available": available,
        }
    content = meta.path.read_text(errors="replace")
    _fm, body = split_frontmatter(content)
    return {
        "ok": True,
        "name": meta.name,
        "description": meta.description,
        "category": meta.category,
        "path": str(meta.path),
        "body": body.strip("\n"),
        "content": content,
        "bundled_files": _bundled_files(meta.directory),
    }


# ── skill_search ──────────────────────────────────────────────────────

async def skill_search(query: str, limit: int = 20) -> dict:
    """Find skills by a plain substring over name, description, AND body.
    Use it when you don't know the exact skill name, or to discover which
    skill covers a task. Returns metadata + a short snippet per hit; call
    ``skill_view`` to read the full body of the one you want."""
    q = (query or "").strip().lower()
    reg = _registry()
    results: list[dict] = []
    for meta in reg.skills():
        try:
            body = meta.path.read_text(errors="replace")
        except OSError:
            body = ""
        haystacks = {
            "name": meta.name.lower(),
            "description": meta.description.lower(),
            "body": body.lower(),
        }
        where = [field for field, text in haystacks.items() if q and q in text]
        if not q or where:
            results.append({
                "name": meta.name,
                "description": meta.description,
                "category": meta.category,
                "matched_in": where,
                "path": str(meta.path),
            })
        if len(results) >= limit:
            break
    return {"query": query, "count": len(results), "results": results}


# ── skill_manage ──────────────────────────────────────────────────────

def _skill_markdown(
    name: str, description: str, category: str, body: str,
    *, extra: dict[str, str] | None = None,
) -> str:
    """Render a well-formed SKILL.md. Frontmatter is emitted via
    ``yaml.safe_dump`` so a colon / quote in a description can never produce
    invalid YAML (which the registry would then skip).

    ``extra`` carries provenance/lifecycle keys (``created_by`` / ``status``)
    AFTER the three core keys, so ``name`` / ``description`` / ``category``
    stay first in the file. ``None`` values are dropped — never written as a
    literal ``null``."""
    data: dict[str, str] = {
        "name": name, "description": description, "category": category,
    }
    for k, v in (extra or {}).items():
        if v is not None:
            data[k] = v
    fm = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    body = (body or "").strip("\n")
    return f"---\n{fm}\n---\n\n{body}\n" if body else f"---\n{fm}\n---\n"


def _preserved_provenance(existing) -> dict[str, str]:
    """Frontmatter keys carried forward when an existing skill is rewritten:
    its ``created_by`` provenance and any ``status``. Losing ``created_by``
    on an update would silently strip a skill of its agent-authored mark and
    push it out of the curator's reach — so it must survive every rewrite."""
    out: dict[str, str] = {}
    if existing is None:
        return out
    if existing.created_by:
        out["created_by"] = existing.created_by
    if existing.status:
        out["status"] = existing.status
    return out


async def skill_manage(
    action: str,
    name: str,
    body: str | None = None,
    description: str | None = None,
    category: str | None = None,
) -> dict:
    """Create, update, remove, or archive a skill on disk. ``action`` is one
    of ``create`` / ``update`` / ``remove`` / ``archive``. For create/update,
    ``body`` is the markdown instructions (frontmatter is generated from
    ``name`` / ``description`` / ``category`` — do NOT include your own
    ``---`` block). Skills you create are stamped ``created_by: agent`` in
    their frontmatter. ``archive`` retires a skill (sets ``status: archived``,
    drops it from the prompt index) WITHOUT deleting the file, so the change
    is reversible and auditable. Changes take effect in the system-prompt
    skills index on the next boot/reload, not mid-session."""
    action = (action or "").strip().lower()
    root = _skills_root()

    if action == "remove":
        existing = _registry().get(name)
        target = existing.directory if existing else (root / _slug(name))
        if not target.is_dir():
            return {"ok": False, "error": f"No skill named {name!r} to remove."}
        shutil.rmtree(target, ignore_errors=True)
        return {"ok": True, "action": "remove", "name": name, "path": str(target)}

    if action == "archive":
        existing = _registry().get(name)
        if existing is None:
            return {"ok": False, "error": f"No skill named {name!r} to archive."}
        # Retire in place: keep name/description/category/body verbatim, flip
        # status to archived, and carry provenance forward. The registry
        # drops archived skills from the index render on the next reload.
        content = existing.path.read_text(errors="replace")
        _fm, existing_body = split_frontmatter(content)
        extra = _preserved_provenance(existing)
        extra["status"] = "archived"
        existing.path.write_text(_skill_markdown(
            name=existing.name,
            description=existing.description,
            category=existing.category,
            body=(existing_body or "").strip("\n"),
            extra=extra,
        ))
        meta = parse_skill_file(existing.path)
        return {
            "ok": meta is not None,
            "action": "archive",
            "name": name,
            "status": "archived",
            "path": str(existing.path),
        }

    if action in ("create", "update"):
        existing = _registry().get(name)
        if action == "create" and existing is not None:
            return {
                "ok": False,
                "error": f"Skill {name!r} already exists — use action='update'.",
                "path": str(existing.path),
            }
        if action == "update" and existing is None:
            return {
                "ok": False,
                "error": f"No skill named {name!r} to update — use action='create'.",
            }
        skill_dir = existing.directory if existing else (root / _slug(name))
        skill_dir.mkdir(parents=True, exist_ok=True)
        md_path = skill_dir / "SKILL.md"
        # Stamp provenance. A fresh create is authored by the agent; an update
        # preserves whatever provenance/status the file already carried (so a
        # seed skill the user asked the agent to edit is NOT re-labelled as
        # agent-authored, and an agent skill keeps its mark).
        if action == "create":
            extra = {"created_by": AGENT_PROVENANCE}
        else:
            extra = _preserved_provenance(existing)
        md_path.write_text(_skill_markdown(
            name=name,
            description=(description or "").strip(),
            category=(category or "").strip() or "general",
            body=body or "",
            extra=extra,
        ))
        # Re-parse to confirm the written file is valid (defensive).
        meta = parse_skill_file(md_path)
        return {
            "ok": meta is not None,
            "action": action,
            "name": name,
            "path": str(md_path),
        }

    return {
        "ok": False,
        "error": (
            f"Unknown action {action!r}. Use create, update, remove, or archive."
        ),
    }
