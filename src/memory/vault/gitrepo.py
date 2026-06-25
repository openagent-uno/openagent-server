"""Git-backed memory vault.

The vault is a git repository so every change to it is tracked with full
history — Company-Brain Prompt 9 ("the time machine for your brain"), made
automatic. The system commits changes itself; the agent is never asked to
run git. Each commit carries provenance trailers describing what produced the
change (a chat session, a workflow, a scheduled task, …) so the history is
auditable.

Git is not a hard requirement to run OpenAgent: ``resolve_git_bin`` finds a
git, ``src.setup.bootstrap.ensure_git`` installs one at setup if missing, and
if none can be obtained every operation here degrades to a logged no-op — the
vault still works, just without history.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GITIGNORE = """\
# OpenAgent memory vault — managed git repository.
# Obsidian per-machine UI state + OS / cache noise. Everything else
# (notes, .obsidian config, graph) is versioned on purpose.
.DS_Store
Thumbs.db
.obsidian/workspace.json
.obsidian/workspace-mobile.json
.obsidian/cache
.trash/
.openagent/
"""


def resolve_git_bin() -> Optional[str]:
    """Locate a usable ``git``: an explicit override, a git shipped inside
    the frozen bundle, then the system PATH."""
    override = os.environ.get("OPENAGENT_GIT_BIN")
    if override and Path(override).exists():
        return override
    try:
        from src._frozen import bundle_dir, is_frozen
        if is_frozen():
            for cand in (bundle_dir() / "bin" / "git", bundle_dir() / "git" / "bin" / "git"):
                if cand.exists():
                    return str(cand)
    except Exception:  # noqa: BLE001
        pass
    return shutil.which("git")


def _author() -> tuple[str, str]:
    name = os.environ.get("OPENAGENT_VAULT_GIT_NAME", "OpenAgent")
    email = os.environ.get("OPENAGENT_VAULT_GIT_EMAIL", "agent@openagent.local")
    return name, email


class VaultGit:
    """Thin, defensive wrapper around the git CLI for one vault. All methods
    are best-effort: a git failure is logged and swallowed, never raised into
    a vault write or an agent turn."""

    def __init__(self, vault_root: Path, git_bin: Optional[str] = None):
        self.vault_root = Path(vault_root)
        self.git_bin = git_bin if git_bin is not None else resolve_git_bin()
        self._lock = threading.RLock()
        self._repo_ready = False
        # A private, self-contained global gitconfig. It trusts the vault dir
        # regardless of ownership — essential in containers where OpenAgent
        # runs as root but the vault volume is owned by another uid, which
        # otherwise makes git refuse every operation with "dubious
        # ownership". ``safe.directory`` is ONLY honoured from system/global
        # config (not -c / env), so we point GIT_CONFIG_GLOBAL at this file.
        self._gitconfig = (
            Path(tempfile.gettempdir())
            / f"openagent-vaultgit-{hashlib.sha1(str(self.vault_root).encode()).hexdigest()[:12]}.gitconfig"
        )

    @property
    def available(self) -> bool:
        return bool(self.git_bin)

    # ── low-level runner ──────────────────────────────────────────────

    def _ensure_gitconfig(self) -> None:
        try:
            if not self._gitconfig.exists():
                self._gitconfig.write_text(
                    "[safe]\n\tdirectory = *\n[init]\n\tdefaultBranch = main\n"
                )
        except Exception:  # noqa: BLE001 — best-effort
            pass

    def _git(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        name, email = _author()
        self._ensure_gitconfig()
        # GIT_CONFIG_GLOBAL → our trust-the-vault config; GIT_CONFIG_SYSTEM →
        # devnull so a restrictive system config can't re-impose the
        # ownership check. Identity still passed via -c.
        env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": str(self._gitconfig),
            "GIT_CONFIG_SYSTEM": os.devnull,
        }
        cmd = [
            self.git_bin, "-C", str(self.vault_root),
            "-c", f"user.name={name}",
            "-c", f"user.email={email}",
            "-c", "commit.gpgsign=false",
            "-c", "core.autocrlf=false",
            "-c", "advice.detachedHead=false",
            *args,
        ]
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )

    def is_repo(self) -> bool:
        return (self.vault_root / ".git").exists()

    # ── repo lifecycle ────────────────────────────────────────────────

    def ensure_repo(self) -> bool:
        """Make sure the vault is a git repo with a committer identity and a
        ``.gitignore``. Initialises one (and commits the existing notes) if
        the folder is not yet a repo. Idempotent; returns whether a usable
        repo is present."""
        if not self.available:
            return False
        with self._lock:
            if self._repo_ready and self.is_repo():
                return True
            try:
                self.vault_root.mkdir(parents=True, exist_ok=True)
                fresh = not self.is_repo()
                if fresh:
                    r = self._git("init", "-b", "main")
                    if r.returncode != 0:
                        # older git without -b: fall back, then rename.
                        self._git("init")
                        self._git("symbolic-ref", "HEAD", "refs/heads/main")
                # Pin the identity into the repo's local config too, so a
                # human running git in the vault sees the same author.
                name, email = _author()
                self._git("config", "user.name", name)
                self._git("config", "user.email", email)
                gi = self.vault_root / ".gitignore"
                if not gi.exists():
                    gi.write_text(_GITIGNORE)
                if fresh:
                    self.commit_all(
                        "vault: initialise git repository",
                        ["Origin: system", "Action: git-init"],
                    )
                self._repo_ready = True
                return True
            except Exception as e:  # noqa: BLE001
                logger.warning("vault git ensure_repo failed: %s", e)
                return False

    # ── commits ───────────────────────────────────────────────────────

    def _message(self, summary: str, trailers: list[str]) -> str:
        body = "\n".join(t for t in (trailers or []) if t)
        return f"{summary}\n\n{body}\n" if body else summary + "\n"

    def has_pending(self) -> bool:
        if not self.available or not self.is_repo():
            return False
        try:
            r = self._git("status", "--porcelain")
            return bool((r.stdout or "").strip())
        except Exception:  # noqa: BLE001
            return False

    def commit(self, paths: list[str], summary: str,
               trailers: list[str]) -> Optional[str]:
        """Commit ONLY the given vault-relative paths (so a per-change commit
        carries exactly that change's provenance and doesn't sweep up
        unrelated pending edits). Returns the short hash, or ``None`` when
        there was nothing to commit / git is unavailable."""
        if not self.ensure_repo() or not paths:
            return None
        with self._lock:
            try:
                self._git("add", "-A", "--", *paths)
                # Anything staged for these paths?
                check = self._git("diff", "--cached", "--quiet", "--", *paths)
                if check.returncode == 0:
                    return None  # no change for these paths
                r = self._git("commit", "-m", self._message(summary, trailers),
                              "--only", "--", *paths)
                if r.returncode != 0:
                    logger.debug("vault git commit (paths) noop/failed: %s",
                                 (r.stderr or r.stdout or "").strip()[:200])
                    return None
                return self._head()
            except Exception as e:  # noqa: BLE001
                logger.warning("vault git commit failed: %s", e)
                return None

    def commit_all(self, summary: str, trailers: list[str]) -> Optional[str]:
        """Stage and commit ALL pending changes (the autocommit sweep that
        catches edits made outside OpenAgent's own tools — the external vault
        MCP, or a human in Obsidian). Returns the short hash or ``None``."""
        if not self.ensure_repo():
            return None
        with self._lock:
            try:
                self._git("add", "-A")
                check = self._git("diff", "--cached", "--quiet")
                if check.returncode == 0:
                    return None
                r = self._git("commit", "-m", self._message(summary, trailers))
                if r.returncode != 0:
                    logger.debug("vault git commit_all noop/failed: %s",
                                 (r.stderr or r.stdout or "").strip()[:200])
                    return None
                return self._head()
            except Exception as e:  # noqa: BLE001
                logger.warning("vault git commit_all failed: %s", e)
                return None

    def _head(self) -> Optional[str]:
        try:
            r = self._git("rev-parse", "--short", "HEAD")
            return (r.stdout or "").strip() or None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _provenance(body: str) -> dict:
        prov: dict[str, str] = {}
        for line in (body or "").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip() and v.strip():
                    prov[k.strip().lower()] = v.strip()
        return prov

    def log(self, limit: int = 20, path: Optional[str] = None) -> list[dict]:
        """Recent commits as ``{hash, subject, date, author, provenance}``.
        With ``path``, only commits that touched that note/folder. The
        provenance dict is parsed from the commit-message trailers (Origin,
        Session, Workflow, Task, Tool, …)."""
        if not self.available or not self.is_repo():
            return []
        try:
            fmt = "--pretty=format:%h%x1f%s%x1f%cI%x1f%an%x1f%b%x1e"
            args = ["log", f"-{max(1, limit)}", fmt]
            if path:
                args += ["--", path]
            r = self._git(*args)
            out: list[dict] = []
            for rec in (r.stdout or "").split("\x1e"):
                rec = rec.strip("\n")
                if not rec.strip():
                    continue
                parts = rec.split("\x1f")
                if len(parts) < 4:
                    continue
                h, subject, date, author = parts[0], parts[1], parts[2], parts[3]
                body = parts[4] if len(parts) > 4 else ""
                out.append({
                    "hash": h, "subject": subject, "date": date,
                    "author": author, "provenance": self._provenance(body),
                })
            return out
        except Exception:  # noqa: BLE001
            return []

    def _commit_exists(self, ref: str) -> bool:
        """True when ``ref`` resolves to a real commit in this repo."""
        if not ref or any(c.isspace() for c in ref):
            return False
        r = self._git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        return r.returncode == 0 and bool((r.stdout or "").strip())

    def show(self, ref: str, max_diff_lines: int = 600) -> Optional[dict]:
        """The changes a single commit introduced: metadata, the list of
        files (with status), and the unified diff (capped at
        ``max_diff_lines`` so a huge commit can't blow up the response)."""
        if not self.available or not self.is_repo() or not self._commit_exists(ref):
            return None
        try:
            meta = self._git(
                "show", "-s",
                "--pretty=format:%H%x1f%h%x1f%s%x1f%cI%x1f%an%x1f%b", ref)
            parts = (meta.stdout or "").split("\x1f")
            if len(parts) < 5:
                return None
            full, short, subject, date, author = parts[:5]
            body = parts[5] if len(parts) > 5 else ""
            ns = self._git("show", "--name-status", "--pretty=format:", ref)
            files: list[dict] = []
            for line in (ns.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                bits = line.split("\t")
                status = bits[0][:1]
                fpath = bits[-1]
                files.append({"status": status, "path": fpath})
            patch = self._git("show", "--no-color", "--pretty=format:", ref)
            lines = (patch.stdout or "").splitlines()
            truncated = len(lines) > max_diff_lines
            diff = "\n".join(lines[:max_diff_lines])
            return {
                "hash": short.strip(), "full_hash": full.strip(),
                "subject": subject, "date": date, "author": author,
                "provenance": self._provenance(body),
                "files": files, "diff": diff, "diff_truncated": truncated,
            }
        except Exception:  # noqa: BLE001
            return None

    def restore_to(self, ref: str, summary: str,
                   trailers: list[str]) -> Optional[str]:
        """Non-destructively bring the working tree back to the state at
        ``ref`` by committing a NEW commit whose content matches it. History
        is fully preserved (every later commit stays reachable and this is
        itself revertable). Returns the new short hash, or ``None`` when the
        tree already matched ``ref`` (nothing to do) or on failure.

        Assumes a clean tree — the caller commits any pending edits first so
        ``reset --hard`` can't silently drop them."""
        if not self.ensure_repo() or not self._commit_exists(ref):
            return None
        with self._lock:
            try:
                saved = (self._git("rev-parse", "HEAD").stdout or "").strip()
                if not saved:
                    return None
                if self._git("reset", "--hard", ref).returncode != 0:
                    return None
                # Move the branch pointer back to the real tip; index +
                # worktree stay at ``ref`` so the next commit records the
                # difference as a forward-moving "restore".
                self._git("reset", "--soft", saved)
                if self._git("diff", "--cached", "--quiet").returncode == 0:
                    return None  # already at that state
                r = self._git("commit", "-m", self._message(summary, trailers))
                if r.returncode != 0:
                    # Best-effort recovery: re-point HEAD at the real tip.
                    self._git("reset", "--hard", saved)
                    return None
                return self._head()
            except Exception as e:  # noqa: BLE001
                logger.warning("vault git restore_to failed: %s", e)
                return None

    def reset_to(self, ref: str) -> dict:
        """DESTRUCTIVELY make ``ref`` the latest commit, permanently deleting
        every commit after it (``git reset --hard``). Only allowed when
        ``ref`` is an ancestor of HEAD — you can roll the history back, never
        rewrite a divergent line. Returns ``{ok, head, deleted}`` or
        ``{error}``."""
        if not self.ensure_repo():
            return {"error": "git unavailable"}
        if not self._commit_exists(ref):
            return {"error": "unknown commit"}
        with self._lock:
            try:
                if self._git("merge-base", "--is-ancestor", ref,
                             "HEAD").returncode != 0:
                    return {"error": "commit is not in the current history"}
                cnt = (self._git("rev-list", "--count",
                                 f"{ref}..HEAD").stdout or "0").strip()
                if self._git("reset", "--hard", ref).returncode != 0:
                    return {"error": "reset failed"}
                return {"ok": True, "head": self._head(),
                        "deleted": int(cnt or 0)}
            except Exception as e:  # noqa: BLE001
                logger.warning("vault git reset_to failed: %s", e)
                return {"error": str(e)}
