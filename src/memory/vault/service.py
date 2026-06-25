"""Async facade over the vault subsystem.

Every surface — the gateway REST API, the ``openagent vault`` CLI, dream-mode
maintenance, and the native ``vault-gate`` MCP — goes through ``VaultService``.
It owns one ``VaultIndex`` per vault and runs the synchronous, CPU-bound index
/ gate / doctor work inside ``asyncio.to_thread`` so the event loop never
blocks while a 100k-note vault is reconciled.

Services are cached by vault root so the long-lived gateway reuses one open
index connection instead of reopening (and re-validating) it on every call.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from src.memory.vault.derived import generate_llms_txt, generate_showcase
from src.memory.vault.doctor import apply_mechanical_fixes
from src.memory.vault.gate import run_gate
from src.memory.vault.index import VaultIndex
from src.memory.vault.model import GateConfig, GateReport
from src.memory.vault.parser import parse_note_text

_DATE_RE = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")


def resolve_vault_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("OPENAGENT_VAULT_PATH")
    if env:
        return Path(env).expanduser().resolve()
    from src.core.paths import default_vault_path
    return default_vault_path().resolve()


def default_index_path(vault_root: Path) -> Path:
    """The index cache lives outside the vault (vision: keep the Markdown
    folder pristine) — under the agent data dir, keyed by vault root so two
    vaults don't share one cache."""
    import hashlib
    from src.core.paths import data_dir
    h = hashlib.sha1(str(vault_root).encode()).hexdigest()[:10]
    return data_dir() / f"vault_index_{h}.db"


class VaultService:
    def __init__(self, vault_root: str | Path, index_path: str | Path | None = None,
                 journal_root: str = "workspace/journal",
                 config: GateConfig | None = None):
        self.vault_root = Path(vault_root)
        self.index_path = Path(index_path) if index_path else default_index_path(self.vault_root)
        self.journal_root = journal_root
        self.config = config or GateConfig.from_env()
        self._index: Optional[VaultIndex] = None
        self._init_lock = asyncio.Lock()
        self._git: Any = None  # VaultGit | False (tried, unavailable) | None
        # Serializes "change files + commit" against the background autocommit
        # sweep, so an OpenAgent write is always captured by its OWN precise
        # per-change commit and never swept into a generic-provenance one.
        self._mutation_lock = asyncio.Lock()

    async def _ensure_index(self) -> VaultIndex:
        if self._index is None:
            async with self._init_lock:
                if self._index is None:
                    self._index = await asyncio.to_thread(
                        VaultIndex, self.vault_root, self.index_path, self.journal_root
                    )
        return self._index

    async def close(self) -> None:
        if self._index is not None:
            await asyncio.to_thread(self._index.close)
            self._index = None

    # ── git history (every change is a tracked commit) ────────────────

    @staticmethod
    def _git_enabled() -> bool:
        return os.environ.get("OPENAGENT_VAULT_GIT_ENABLED", "1").strip().lower() in (
            "1", "true", "yes", "on")

    async def _ensure_git(self):
        """Lazily build the VaultGit and initialise the repo. Returns the
        VaultGit, or ``None`` when git is disabled/unavailable."""
        if self._git is None:
            if not self._git_enabled():
                self._git = False
            else:
                from src.memory.vault.gitrepo import VaultGit
                g = VaultGit(self.vault_root)
                ok = await asyncio.to_thread(g.ensure_repo)
                self._git = g if ok else False
        return self._git or None

    async def commit_paths(self, paths, summary: str, origin: dict | None = None) -> Optional[str]:
        """Commit exactly ``paths`` with provenance (one change, one commit)."""
        g = await self._ensure_git()
        if not g or not paths:
            return None
        from src.memory.vault.vault_origin import trailers
        return await asyncio.to_thread(g.commit, list(paths), summary, trailers(origin))

    async def autocommit(self, origin: dict | None = None,
                         summary: str | None = None) -> Optional[str]:
        """Commit ALL pending changes — the sweep that captures edits made
        outside OpenAgent's own tools (the external vault MCP, a human in
        Obsidian). Runs under the mutation lock so it never races an
        OpenAgent write (which would steal that write's precise provenance).

        Attribution is best-effort: a sweep can fold edits from several
        sources into one commit, so it is labelled ``Origin: external`` with
        the most recent activity as an ``Around:`` hint, rather than falsely
        claiming a specific session authored the edit."""
        g = await self._ensure_git()
        if not g:
            return None
        from src.memory.vault.vault_origin import recent_origin, trailers
        if origin is not None:
            o = origin
        else:
            recent = recent_origin()
            o = {"kind": "external"}
            if recent:
                around = recent.get("session") or recent.get("workflow") or recent.get("task")
                if around:
                    o["around"] = f"{recent.get('kind', '?')}:{around}"
        msg = summary or f"vault: autocommit ({o.get('kind', 'external')})"
        async with self._mutation_lock:
            return await asyncio.to_thread(g.commit_all, msg, trailers(o))

    @staticmethod
    def _enforce_writes() -> bool:
        """Whether REST/service writes go through the quality gate (auto-fix +
        block). Shares the flag the vendored vault MCP uses, default ON for
        the in-process service so the app/CLI get the same enforcement as the
        agent. Set OPENAGENT_VAULT_VALIDATE_WRITES=0 to fall back to the old
        warn-only behaviour."""
        return os.environ.get(
            "OPENAGENT_VAULT_VALIDATE_WRITES", "1").strip().lower() in (
            "1", "true", "yes", "on")

    async def _enforce_write(self, rel: str, content: str,
                             is_new: bool) -> tuple[str, list, list, list]:
        """Auto-fix the mechanically-fixable issues and return any blocking
        errors + warnings, mirroring the vendored vault MCP's write gate
        (validate.ts). Pure CPU; runs in a thread. Returns
        ``(fixed_content, errors, warnings, applied)``."""
        import datetime
        import yaml
        from src.memory.vault.doctor import fix_note_content, _FIXABLE_RULES
        from src.memory.vault.parser import split_frontmatter

        def work() -> tuple[str, list, list, list]:
            note = parse_note_text(rel, content, journal_root=self.journal_root)
            today = datetime.date.today().isoformat()
            fixed, applied = fix_note_content(
                content, note, set(_FIXABLE_RULES), today)
            errors: list = []
            warnings: list = []
            raw_fm, _ = split_frontmatter(fixed)
            if raw_fm is not None and raw_fm.strip():
                try:
                    yaml.safe_load(raw_fm)
                except Exception as e:  # noqa: BLE001
                    errors.append({"rule": "frontmatter", "severity": "error",
                                   "message": f"frontmatter is not valid YAML: {e}"})
            note2 = parse_note_text(rel, fixed, journal_root=self.journal_root)
            if is_new and note2.line_count > self.config.max_lines:
                errors.append({
                    "rule": "atomicity", "severity": "error",
                    "message": (f"note is {note2.line_count} body lines "
                                f"(> {self.config.max_lines}); split it into "
                                "atomic notes, one idea each, and link them")})
            if not note2.summary:
                warnings.append({
                    "rule": "frontmatter", "severity": "warn",
                    "message": ("missing 'summary' — add a one-sentence "
                                "summary so the note is self-describing")})
            return fixed, errors, warnings, applied

        return await asyncio.to_thread(work)

    async def write_note(self, rel_path: str, content: str,
                         origin: dict | None = None,
                         validate: bool = True) -> dict:
        """Write a note, index it, validate it, and commit it — all atomically
        under the mutation lock so the change lands in history with its own
        precise provenance. With the quality gate on (default), mechanical
        issues are auto-fixed and a structurally-broken note is rejected
        (``ok: False`` + ``errors``, nothing written). Returns ``{ok, existed,
        warnings, applied, commit}`` or ``{ok: False, blocked, errors}``."""
        rel = rel_path.replace("\\", "/").lstrip("/")
        abs_path = self.vault_root / rel
        async with self._mutation_lock:
            existed = abs_path.exists()
            warnings: list = []
            applied: list = []

            gated = validate and rel.lower().endswith(".md") and self._enforce_writes()
            if gated:
                try:
                    content, errors, warnings, applied = await self._enforce_write(
                        rel, content, is_new=not existed)
                except Exception:  # noqa: BLE001 — a validator fault must never block
                    errors = []
                if errors:
                    return {"ok": False, "blocked": True, "existed": existed,
                            "errors": errors, "warnings": warnings}

            def _write() -> None:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(content)

            await asyncio.to_thread(_write)  # critical: the note must land
            try:
                await self.index_note(rel, content)
                if not gated and validate and rel.lower().endswith(".md"):
                    v = await self.validate_note(rel, content)
                    warnings = v.get("issues", [])
            except Exception:  # noqa: BLE001 — index/validate is best-effort
                pass
            commit = await self.commit_paths(
                [rel], f"vault: {'update' if existed else 'create'} {rel}", origin)
            return {"ok": True, "existed": existed, "warnings": warnings,
                    "applied": applied, "commit": commit}

    async def delete_note(self, rel_path: str, origin: dict | None = None) -> dict:
        """Delete a note, de-index it, and commit the removal atomically."""
        rel = rel_path.replace("\\", "/").lstrip("/")
        abs_path = self.vault_root / rel
        async with self._mutation_lock:
            existed = abs_path.exists()
            if existed:
                await asyncio.to_thread(abs_path.unlink)
            await self.deindex_note(rel)
            commit = await self.commit_paths(
                [rel], f"vault: delete {rel}", origin)
            return {"existed": existed, "commit": commit}

    async def git_log(self, limit: int = 20, path: str | None = None) -> list[dict]:
        g = await self._ensure_git()
        if not g:
            return []
        return await asyncio.to_thread(g.log, limit, path)

    async def git_show(self, ref: str) -> dict | None:
        """The changes a single commit introduced (files + unified diff)."""
        g = await self._ensure_git()
        if not g:
            return None
        return await asyncio.to_thread(g.show, ref)

    async def restore_to(self, ref: str, origin: dict | None = None) -> dict:
        """Non-destructively roll the vault back to the state at ``ref`` — a
        NEW commit whose content matches it, so every later commit stays in
        history and this is itself revertable. Pending edits are checkpointed
        first (nothing is lost), and the index is rebuilt from disk after.
        Returns ``{ok, commit, restored_from, changed}``."""
        g = await self._ensure_git()
        if not g:
            return {"error": "git is disabled for this vault"}
        from src.memory.vault.vault_origin import trailers
        o = dict(origin or {})
        o.setdefault("kind", "tool")
        o["action"] = "restore"
        o["restored_from"] = ref
        async with self._mutation_lock:
            if await asyncio.to_thread(g.has_pending):
                await asyncio.to_thread(
                    g.commit_all, "vault: checkpoint before restore",
                    trailers({"kind": "system", "action": "pre-restore"}))
            commit = await asyncio.to_thread(
                g.restore_to, ref, f"vault: restore to {ref}", trailers(o))
            idx = await self._ensure_index()
            await asyncio.to_thread(idx.sync, True)
        return {"ok": True, "commit": commit, "restored_from": ref,
                "changed": commit is not None}

    async def reset_to(self, ref: str, origin: dict | None = None) -> dict:
        """DESTRUCTIVELY make ``ref`` the latest commit, deleting every commit
        after it (only when ``ref`` is in the current history). The index is
        rebuilt from disk after. Returns ``{ok, head, deleted}`` or
        ``{error}``."""
        g = await self._ensure_git()
        if not g:
            return {"error": "git is disabled for this vault"}
        async with self._mutation_lock:
            result = await asyncio.to_thread(g.reset_to, ref)
            if result.get("ok"):
                idx = await self._ensure_index()
                await asyncio.to_thread(idx.sync, True)
        return result

    # ── reconcile + read ──────────────────────────────────────────────

    async def sync(self, force: bool = False) -> dict:
        idx = await self._ensure_index()
        stats = await asyncio.to_thread(idx.sync, force)
        return stats.to_dict()

    async def stats(self, sync: bool = True) -> dict:
        idx = await self._ensure_index()
        if sync:
            await asyncio.to_thread(idx.sync, False)
        return await asyncio.to_thread(idx.stats)

    async def search(self, query: str, limit: int = 25) -> list[dict]:
        idx = await self._ensure_index()
        return await asyncio.to_thread(idx.search, query, limit)

    async def backlinks(self, rel_path: str) -> list[str]:
        idx = await self._ensure_index()
        return await asyncio.to_thread(idx.backlinks, rel_path)

    # ── gate ──────────────────────────────────────────────────────────

    async def gate(self, sync: bool = True,
                   config: GateConfig | None = None) -> GateReport:
        idx = await self._ensure_index()
        if sync:
            await asyncio.to_thread(idx.sync, False)
        return await asyncio.to_thread(run_gate, idx, config or self.config)

    async def validate_note(self, rel_path: str, content: str) -> dict:
        """Validate a single note (parsed in-memory) against the single-note
        rules + broken-link resolution. Cheap enough for the write path; does
        not recompute the global graph. Returns ``{ok, issues}``."""
        idx = await self._ensure_index()
        note = parse_note_text(rel_path, content, journal_root=self.journal_root)
        cfg = self.config
        issues: list[dict] = []

        def add(rule: str, message: str, *, fixable: bool = False) -> None:
            issues.append({
                "rule": rule, "severity": cfg.severity_for(rule),
                "message": message, "fixable": fixable,
            })

        if note.missing_frontmatter_fields:
            add("frontmatter",
                "missing frontmatter: " + ", ".join(note.missing_frontmatter_fields),
                fixable=True)
        if note.line_count > cfg.max_lines:
            add("atomicity",
                f"{note.line_count} body lines exceeds {cfg.max_lines}")
        if note.related_multiline:
            add("wikilink_format", "related spans multiple lines", fixable=True)
        if note.spaced_wikilinks:
            add("wikilink_format", "wikilink(s) contain inner spaces", fixable=True)
        for fld in ("created", "updated"):
            val = getattr(note, fld)
            if val and not _DATE_RE.match(val):
                add("date_format", f"{fld} '{val}' is not YYYY-MM-DD", fixable=True)
        # broken links — resolve against the current index
        for target in note.outlinks:
            resolved = await asyncio.to_thread(idx.resolve_link, target)
            if not resolved and target.strip().lower() != note.stem.lower():
                add("broken_link", f"[[{target}]] does not resolve to a note")

        ok = not any(i["severity"] == "error" for i in issues)
        return {"ok": ok, "path": note.path, "issues": issues}

    # ── doctor (auto-fix) ─────────────────────────────────────────────

    async def doctor(self, apply: bool = False,
                     config: GateConfig | None = None,
                     origin: dict | None = None) -> dict:
        idx = await self._ensure_index()
        cfg = config or self.config
        await asyncio.to_thread(idx.sync, False)
        before = await asyncio.to_thread(run_gate, idx, cfg)
        after = None
        commit = None
        if not apply:
            fix = await asyncio.to_thread(
                apply_mechanical_fixes, self.vault_root, before, idx, False)
        else:
            # Hold the lock across the file writes AND the commit so the
            # autocommit sweep can't grab the fixes first.
            async with self._mutation_lock:
                fix = await asyncio.to_thread(
                    apply_mechanical_fixes, self.vault_root, before, idx, True)
                if fix.files_changed:
                    await asyncio.to_thread(idx.sync, False)
                    after = await asyncio.to_thread(run_gate, idx, cfg)
                    commit = await self.commit_paths(
                        [f["path"] for f in fix.fixed], "vault: doctor auto-fix",
                        origin or {"kind": "doctor"})
        return {
            "before": before.to_dict(),
            "fix": fix.to_dict(),
            "after": after.to_dict() if after else None,
            "commit": commit,
        }

    # ── derived artifacts ─────────────────────────────────────────────

    async def regenerate_derived(self, origin: dict | None = None) -> dict:
        idx = await self._ensure_index()
        await asyncio.to_thread(idx.sync, False)
        llms = await asyncio.to_thread(generate_llms_txt, idx)
        showcase = await asyncio.to_thread(generate_showcase, idx)

        def _write() -> None:
            self.vault_root.mkdir(parents=True, exist_ok=True)
            (self.vault_root / "llms.txt").write_text(llms)
            sc_dir = self.vault_root / "_showcase"
            sc_dir.mkdir(parents=True, exist_ok=True)
            (sc_dir / "showcase.md").write_text(showcase)

        async with self._mutation_lock:
            await asyncio.to_thread(_write)
            commit = await self.commit_paths(
                ["llms.txt", "_showcase/showcase.md"],
                "vault: regenerate derived artifacts", origin or {"kind": "derived"})
        return {
            "llms_txt": "llms.txt",
            "showcase": "_showcase/showcase.md",
            "llms_bytes": len(llms),
            "showcase_bytes": len(showcase),
            "commit": commit,
        }

    # ── taxonomy scaffolding ──────────────────────────────────────────

    async def init_taxonomy(self, origin: dict | None = None) -> dict:
        """Create the Company-Brain folder system (11 folders + journal tree
        + canon workspace + templates). Idempotent."""
        from src.memory.vault.scaffold import scaffold
        async with self._mutation_lock:
            res = await asyncio.to_thread(scaffold, self.vault_root, self.journal_root)
            await self.sync()
            if res.get("created"):
                res["commit"] = await self.commit_paths(
                    [c.rstrip("/") for c in res["created"]],
                    "vault: scaffold folder system", origin or {"kind": "init"})
        return res

    # ── move / rename with automatic link rewriting ──────────────────

    async def move(self, old_rel: str, new_rel: str,
                   origin: dict | None = None) -> dict:
        """Move or rename a note OR a folder, rewriting every wikilink that
        points at it so no link breaks (the thing Obsidian does on rename).

        For a folder, every note underneath moves and all inbound links to
        any of them are rewritten. Link style is preserved (path stays a
        path, bare stem stays a stem) along with ``|alias`` / ``#anchor``."""
        from src.memory.vault.parser import rewrite_wikilinks

        idx = await self._ensure_index()
        await asyncio.to_thread(idx.sync, False)

        old_rel = old_rel.replace("\\", "/").strip("/")
        new_rel = new_rel.replace("\\", "/").strip("/")
        old_abs = self.vault_root / old_rel
        new_abs = self.vault_root / new_rel
        if not old_abs.exists():
            return {"error": f"source not found: {old_rel}"}
        if new_abs.exists():
            return {"error": f"target already exists: {new_rel}"}

        # Build the old->new mapping (1 entry for a file, N for a folder).
        mapping: dict[str, str] = {}
        if old_abs.is_dir():
            for p in sorted(old_abs.rglob("*.md")):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.vault_root).as_posix()
                mapping[rel] = new_rel + "/" + rel[len(old_rel):].lstrip("/")
        else:
            mapping[old_rel] = new_rel

        def _apply() -> dict:
            # 1. Rewrite inbound links across every affected backlink note
            #    (still at their current paths). Done before the move so the
            #    resolver still resolves the old targets.
            affected: set[str] = set()
            for old in mapping:
                affected.update(idx.backlinks(old))
            rewritten: list[dict] = []
            for bl in sorted(affected):
                bl_abs = self.vault_root / bl
                try:
                    text = bl_abs.read_text(errors="replace")
                except OSError:
                    continue
                new_text, n = rewrite_wikilinks(text, idx.resolve_link, mapping)
                if n:
                    bl_abs.write_text(new_text)
                    rewritten.append({"path": bl, "links": n})

            # 2. Move the file(s) on disk.
            for old, new in mapping.items():
                src = self.vault_root / old
                dst = self.vault_root / new
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dst.write_text(src.read_text(errors="replace"))
                    src.unlink()
                except OSError:
                    continue
            # prune now-empty source dirs (folder move)
            if old_abs.is_dir() or not (self.vault_root / old_rel).exists():
                try:
                    for d in sorted(old_abs.rglob("*"), reverse=True):
                        if d.is_dir() and not any(d.iterdir()):
                            d.rmdir()
                    if old_abs.is_dir() and not any(old_abs.iterdir()):
                        old_abs.rmdir()
                except OSError:
                    pass
            return {"rewritten": rewritten}

        async with self._mutation_lock:
            result = await asyncio.to_thread(_apply)
            # 3. Reconcile the index (picks up the moved files + rebuilds graph).
            await asyncio.to_thread(idx.sync, False)
            # 4. Commit the move + every rewritten backlink with provenance.
            changed = (list(mapping.keys()) + list(mapping.values())
                       + [r["path"] for r in result["rewritten"]])
            commit = await self.commit_paths(
                changed, f"vault: move {old_rel} -> {new_rel}", origin)
        return {
            "moved": {"from": old_rel, "to": new_rel},
            "notes_moved": len(mapping),
            "notes_updated": len(result["rewritten"]),
            "links_rewritten": sum(r["links"] for r in result["rewritten"]),
            "updated": result["rewritten"],
            "commit": commit,
        }

    # ── incremental write hooks (validate-on-write) ───────────────────

    async def index_note(self, rel_path: str, content: str) -> None:
        idx = await self._ensure_index()
        await asyncio.to_thread(idx.update_note, rel_path, content)

    async def deindex_note(self, rel_path: str) -> None:
        idx = await self._ensure_index()
        await asyncio.to_thread(idx.remove_note, rel_path)

    # ── dream-mode maintenance ────────────────────────────────────────

    async def maintenance(self, apply_fixes: bool = True,
                          regenerate: bool = True) -> dict:
        """One maintenance pass for dream mode: reconcile the index, run the
        gate, mechanically fix what code can, regenerate derived artifacts,
        and return a summary (the curator turns this into a dream-log)."""
        idx = await self._ensure_index()
        await asyncio.to_thread(idx.sync, False)
        report = await asyncio.to_thread(run_gate, idx, self.config)
        fix = await asyncio.to_thread(
            apply_mechanical_fixes, self.vault_root, report, idx, apply_fixes
        )
        after = None
        if apply_fixes and fix.files_changed:
            await asyncio.to_thread(idx.sync, False)
            after = await asyncio.to_thread(run_gate, idx, self.config)
        derived = await self.regenerate_derived() if regenerate else None
        final = after or report
        return {
            "before": report.summary_line(),
            "after": final.summary_line(),
            "files_changed": fix.files_changed,
            "fixes": fix.fixed,
            "open_suggestions": fix.suggestions[:50],
            "open_suggestion_count": len(fix.suggestions),
            "derived": derived,
            "stats": final.stats,
        }


# ── service cache (one open index per vault, reused by the gateway) ────

_SERVICES: dict[str, VaultService] = {}


def get_service(vault_root: str | Path | None = None,
                journal_root: str = "workspace/journal",
                config: GateConfig | None = None) -> VaultService:
    root = resolve_vault_root(vault_root)
    key = str(root)
    svc = _SERVICES.get(key)
    if svc is None:
        svc = VaultService(root, journal_root=journal_root, config=config)
        _SERVICES[key] = svc
    return svc


async def close_all() -> None:
    for svc in list(_SERVICES.values()):
        await svc.close()
    _SERVICES.clear()
