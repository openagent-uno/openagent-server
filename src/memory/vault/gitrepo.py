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

    def log(self, limit: int = 20) -> list[dict]:
        """Recent commits as ``{hash, subject}`` — for inspection / tests."""
        if not self.available or not self.is_repo():
            return []
        try:
            r = self._git("log", f"-{limit}", "--pretty=format:%h\t%s")
            out = []
            for line in (r.stdout or "").splitlines():
                if "\t" in line:
                    h, s = line.split("\t", 1)
                    out.append({"hash": h, "subject": s})
            return out
        except Exception:  # noqa: BLE001
            return []
