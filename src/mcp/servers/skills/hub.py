"""Skills-Hub — pull SKILL.md playbooks from a shared git remote.

A *tap* is any git remote (``https://…``, ``git@…``, or ``file://…``). Pulling
one shallow-clones it into a QUARANTINE dir, scans every ``*/SKILL.md`` folder
with :mod:`hub_guard`, and copies only the ones that pass into the live skills
directory — stamping each with ``created_by: hub`` provenance so it is owned by
the external upstream, not this agent.

Two invariants make this safe to bolt onto the existing subsystem:

  * **Provenance = ``hub``.** The skill-curator's boundary is
    ``created_by == "agent"`` (see ``registry.agent_authored``). A hub skill is
    ``created_by: hub``, so it is automatically OUTSIDE that set — the curator
    can never merge or archive an upstream-owned skill, with zero curator code
    changes. Hub is the external source of truth; the agent must not rewrite it.
  * **Quarantine + scan before copy.** Pulled text reaches the model (index +
    ``skill_view``), so nothing lands in the live dir until it clears
    :func:`hub_guard.scan_skill`. ``dangerous`` is refused always; ``caution``
    is refused unless ``force=True``.

A lockfile at ``<skills_root>/.hub/lock.json`` pins ``{repo, commit,
content_hash}`` per installed hub skill, so an operator can see exactly what is
installed and from where. It lives OUTSIDE the ``*/SKILL.md`` glob, so it never
perturbs the byte-stable prompt index.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from src.core.paths import default_skills_path
from src.memory.vault.parser import (
    FrontmatterSyntaxError,
    load_frontmatter_yaml,
    split_frontmatter,
)
from src.mcp.servers.skills import hub_guard
from src.mcp.servers.skills.registry import SkillsRegistry, parse_skill_file

# The provenance stamp for a hub skill. NOT ``agent`` on purpose: the curator's
# work set is ``created_by == "agent"``, so this value keeps hub skills off-
# limits to consolidation. NOT ``None`` either — a hub skill must be
# distinguishable from a hand-written seed skill for ``skill_hub_list`` and the
# ``is_hub`` label.
HUB_PROVENANCE = "hub"

_LOCK_REL = Path(".hub") / "lock.json"

# Core frontmatter keys that stay first in a rewritten SKILL.md, for humans.
_CORE_KEYS = ("name", "description", "category")


def _skills_root() -> Path:
    return default_skills_path()


def _lock_path(root: Path) -> Path:
    return root / _LOCK_REL


def _load_lock(root: Path) -> dict:
    p = _lock_path(root)
    if not p.is_file():
        return {"skills": {}}
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        # A corrupt lockfile is not authoritative over the on-disk skills; the
        # registry (created_by==hub) remains the ground truth. Start fresh so a
        # pull can rewrite it rather than wedging on a bad file.
        return {"skills": {}}
    if not isinstance(data, dict) or not isinstance(data.get("skills"), dict):
        return {"skills": {}}
    return data


def _save_lock(root: Path, lock: dict) -> None:
    p = _lock_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys → deterministic file, easy to diff / review.
    p.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def _git(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _restamp_frontmatter(content: str, tap: str, commit: str) -> tuple[str, str, str] | None:
    """Rewrite a SKILL.md's frontmatter with hub provenance.

    Returns ``(stamped_markdown, name, body)`` or ``None`` when the file has no
    valid frontmatter / no ``name`` (the registry would skip it anyway, so it
    is not installable). All original frontmatter keys are preserved; only the
    provenance/lineage keys are forced:

      * ``created_by: hub``   — the curator boundary (never ``agent``),
      * ``hub_repo: <tap>``   — where it came from,
      * ``hub_commit: <sha>`` — the exact upstream commit pinned.
    """
    raw_fm, body = split_frontmatter(content)
    if raw_fm is None:
        return None
    try:
        meta = load_frontmatter_yaml(raw_fm)
    except FrontmatterSyntaxError:
        return None
    if not isinstance(meta, dict):
        return None
    name = str(meta.get("name") or "").strip()
    if not name:
        return None

    # Preserve original keys, in a human-friendly order: the three core keys
    # first, then any extra upstream keys, then the forced hub lineage last.
    ordered: dict = {}
    for k in _CORE_KEYS:
        if k in meta:
            ordered[k] = meta[k]
    for k, v in meta.items():
        if k in _CORE_KEYS or k in ("created_by", "hub_repo", "hub_commit"):
            continue
        ordered[k] = v
    ordered["created_by"] = HUB_PROVENANCE
    ordered["hub_repo"] = tap
    ordered["hub_commit"] = commit

    fm = yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True).strip()
    body_clean = (body or "").strip("\n")
    stamped = f"---\n{fm}\n---\n\n{body_clean}\n" if body_clean else f"---\n{fm}\n---\n"
    return stamped, name, body_clean


def _content_hash(body: str) -> str:
    return hashlib.sha256((body or "").strip().encode("utf-8")).hexdigest()


def _existing_is_hub(dest: Path) -> bool | None:
    """For a folder that already exists: ``True`` if it holds a hub skill,
    ``False`` if a non-hub (seed/agent/user) skill, ``None`` if not a skill."""
    meta = parse_skill_file(dest / "SKILL.md")
    if meta is None:
        return None
    return meta.is_hub


async def skill_hub_pull(tap: str, force: bool = False) -> dict:
    """Pull SKILL.md skills from a shared git *tap* into the local skills dir.

    ``tap`` is any git remote (``https://``, ``git@…``, or ``file://…``). The
    remote is shallow-cloned into a quarantine dir, every ``*/SKILL.md`` folder
    is safety-scanned, and only the ones that pass are copied into the live
    skills directory — each stamped ``created_by: hub`` (so the skill-curator
    leaves it alone) plus ``hub_repo`` / ``hub_commit`` lineage. A malicious
    skill is refused: ``dangerous`` always, ``caution`` unless ``force=True``.
    Returns a summary of what was pulled / skipped / rejected.
    """
    tap = (tap or "").strip()
    if not tap:
        return {"ok": False, "error": "no tap given"}
    if shutil.which("git") is None:
        return {"ok": False, "error": "git is not installed / not on PATH"}

    root = _skills_root()
    quarantine = Path(tempfile.mkdtemp(prefix="skillhub-"))
    clone = quarantine / "repo"
    try:
        cloned = _git(["clone", "--depth", "1", tap, str(clone)])
        if cloned.returncode != 0:
            return {
                "ok": False,
                "error": f"git clone failed for {tap!r}",
                "stderr": (cloned.stderr or "").strip()[-500:],
            }
        head = _git(["rev-parse", "HEAD"], cwd=str(clone))
        if head.returncode != 0:
            return {
                "ok": False,
                "error": "could not resolve cloned HEAD commit",
                "stderr": (head.stderr or "").strip()[-500:],
            }
        commit = head.stdout.strip()

        skill_mds = sorted(clone.glob("*/SKILL.md"))
        if not skill_mds:
            return {"ok": False, "error": f"no */SKILL.md skills found in {tap!r}",
                    "commit": commit}

        lock = _load_lock(root)
        pulled: list[dict] = []
        skipped: list[dict] = []
        rejected: list[dict] = []

        for md in skill_mds:
            src_dir = md.parent
            folder = src_dir.name

            verdict = hub_guard.scan_skill(src_dir)
            if verdict.blocks(force):
                rejected.append({
                    "folder": folder,
                    "verdict": verdict.level,
                    "reasons": verdict.reasons,
                })
                continue

            stamped = _restamp_frontmatter(md.read_text(errors="replace"), tap, commit)
            if stamped is None:
                rejected.append({
                    "folder": folder,
                    "verdict": "invalid",
                    "reasons": ["no valid frontmatter / missing name"],
                })
                continue
            stamped_md, name, body = stamped

            dest = root / folder
            if dest.exists():
                owned = _existing_is_hub(dest)
                if owned is False:
                    # Never clobber a hand-written seed/agent/user skill.
                    skipped.append({
                        "folder": folder, "name": name,
                        "reason": "destination is a non-hub skill — not overwriting",
                    })
                    continue
                shutil.rmtree(dest, ignore_errors=True)

            # Copy the whole folder (bundled files included), then overwrite
            # SKILL.md with the provenance-stamped version.
            shutil.copytree(src_dir, dest)
            (dest / "SKILL.md").write_text(stamped_md)

            meta = parse_skill_file(dest / "SKILL.md")
            if meta is None or not meta.is_hub:
                # The stamped file must parse and carry the hub mark, else the
                # curator boundary / index would be wrong — back it out.
                shutil.rmtree(dest, ignore_errors=True)
                rejected.append({
                    "folder": folder, "name": name,
                    "verdict": "invalid",
                    "reasons": ["stamped SKILL.md failed to re-parse as a hub skill"],
                })
                continue

            lock["skills"][name] = {
                "repo": tap,
                "commit": commit,
                "content_hash": _content_hash(body),
                "folder": folder,
            }
            entry = {"folder": folder, "name": name, "verdict": verdict.level}
            if verdict.level != "safe":
                entry["reasons"] = verdict.reasons
            pulled.append(entry)

        if pulled:
            _save_lock(root, lock)

        return {
            "ok": True,
            "tap": tap,
            "commit": commit,
            "pulled": pulled,
            "skipped": skipped,
            "rejected": rejected,
            "counts": {
                "pulled": len(pulled),
                "skipped": len(skipped),
                "rejected": len(rejected),
            },
        }
    finally:
        shutil.rmtree(quarantine, ignore_errors=True)


async def skill_hub_list() -> dict:
    """List the hub skills installed locally (``created_by: hub``).

    Cross-references the on-disk registry (the ground truth for the prompt
    index) with the ``.hub/lock.json`` lockfile (which pins the source repo /
    commit / content hash per skill), so you can see what is installed and
    exactly where it came from."""
    root = _skills_root()
    lock = _load_lock(root)
    locked = lock.get("skills", {})

    reg = SkillsRegistry(root)
    reg.load()

    out: list[dict] = []
    for meta in reg.skills():
        if not meta.is_hub:
            continue
        pin = locked.get(meta.name, {})
        out.append({
            "name": meta.name,
            "description": meta.description,
            "category": meta.category,
            "repo": pin.get("repo"),
            "commit": pin.get("commit"),
            "content_hash": pin.get("content_hash"),
            "path": str(meta.path),
        })
    out.sort(key=lambda e: e["name"].lower())
    return {"ok": True, "count": len(out), "skills": out}
