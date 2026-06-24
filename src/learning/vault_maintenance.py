"""Dream-mode vault maintenance.

Vision §12: the agent runs a scheduled "dream" that maintains its memory
vault — consolidating duplicates, strengthening links, pruning broken cross-
references, and writing a dream-log of what it changed. This module is the
vault half of that. On its interval it:

  1. reconciles the index with the notes on disk,
  2. runs the quality gate,
  3. mechanically fixes everything code can fix (the doctor),
  4. regenerates the derived ``llms.txt`` / showcase,
  5. optionally asks a cheap model to propose fixes for the issues code
     cannot fix (orphans, duplicates, over-long notes), and
  6. writes a dream-log note so the agent — and the user — can see what was
     done and what still needs a human/AI decision.

Independent of the curator loop and opt-in via
``memory.vault.maintenance.enabled`` (default off). Everything degrades
gracefully: no model key → no AI suggestions, just the mechanical pass.
"""
from __future__ import annotations

import asyncio
import datetime
import os
from typing import Any, Optional

from src.core.logging import elog

_DEFAULT_INTERVAL_H = 12


def _is_enabled() -> bool:
    return (
        os.environ.get("OPENAGENT_VAULT_MAINTENANCE_ENABLED", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


async def run_once(db: Any = None) -> dict:
    """One maintenance pass. Safe to call manually (tests, CLI)."""
    from src.memory.vault.service import get_service

    svc = get_service()
    autofix = _bool_env("OPENAGENT_VAULT_MAINTENANCE_AUTOFIX", True)
    derived = _bool_env("OPENAGENT_VAULT_MAINTENANCE_DERIVED", True)
    summary = await svc.maintenance(apply_fixes=autofix, regenerate=derived)

    ai_note = await _ai_suggestions(db, summary)
    try:
        await _write_dream_log(svc, summary, ai_note)
    except Exception as e:  # noqa: BLE001
        elog("vault_maintenance.dreamlog_error", error=str(e)[:200])

    # Commit everything this dream pass changed (fixes + derived + dream-log)
    # in one provenance-stamped commit.
    try:
        summary["commit"] = await svc.autocommit(
            origin={"kind": "dream", "task": "vault-maintenance"},
            summary="vault: dream-mode maintenance")
    except Exception as e:  # noqa: BLE001
        elog("vault_maintenance.commit_error", error=str(e)[:200])

    elog(
        "vault_maintenance.run",
        before=summary["before"],
        after=summary["after"],
        files_changed=summary["files_changed"],
        open_suggestions=summary["open_suggestion_count"],
    )
    return summary


async def _ai_suggestions(db: Any, summary: dict) -> Optional[str]:
    """Ask a cheap model to propose concrete fixes for the issues code could
    not fix. Best-effort: returns ``None`` offline or without a key."""
    if db is None:
        return None
    suggestions = summary.get("open_suggestions") or []
    if not suggestions:
        return None
    try:
        from src.learning import _groq
    except Exception:
        return None
    lines = [
        f"- [{s['rule']}] {s['path']}: {s['message']}"
        for s in suggestions[:20]
    ]
    prompt = (
        "You maintain a Markdown knowledge vault of atomic, wiki-linked "
        "notes. A script already fixed the mechanical issues. Below are the "
        "issues it could NOT fix automatically. For each, give ONE short, "
        "concrete action (e.g. which existing note an orphan should link to, "
        "whether two notes should be merged and which to keep, where to split "
        "an over-long note). Be terse — one line per issue.\n\n"
        + "\n".join(lines)
    )
    try:
        return await _groq.complete(
            db=db, prompt=prompt, log_event="vault_maintenance.ai",
            timeout_s=25.0, max_tokens=900,
        )
    except Exception:
        return None


async def _write_dream_log(svc, summary: dict, ai_note: Optional[str]) -> None:
    """Write a dream-log note under ``workspace/_dream/`` (the gate skips
    ``workspace/`` scratch, so the log never trips its own rules)."""
    now = datetime.datetime.now()
    stamp = now.strftime("%Y-%m-%d-%H%M")
    date = now.strftime("%Y-%m-%d")
    rel = f"workspace/_dream/dream-{stamp}.md"

    fixes = summary.get("fixes") or []
    fix_lines = [
        f"- `{f['path']}`: {', '.join(f['fixes'])}" for f in fixes[:50]
    ] or ["- (no mechanical fixes were needed)"]
    sugg = summary.get("open_suggestions") or []
    sugg_lines = [
        f"- [{s['rule']}] `{s['path']}` — {s['message']}" for s in sugg[:50]
    ] or ["- (none)"]

    body = [
        "---",
        f"title: Dream log {stamp}",
        f"summary: Vault maintenance pass on {date}.",
        "tags: [workspace, type/dream]",
        "status: done",
        f"created: {date}",
        f"updated: {date}",
        "---",
        "",
        f"# Dream log — {stamp}",
        "",
        f"- Before: {summary.get('before')}",
        f"- After: {summary.get('after')}",
        f"- Files changed: {summary.get('files_changed')}",
        f"- Open suggestions: {summary.get('open_suggestion_count')}",
        "",
        "## Mechanical fixes applied",
        *fix_lines,
        "",
        "## Open issues (need a human / AI decision)",
        *sugg_lines,
    ]
    if ai_note:
        body += ["", "## AI-proposed fixes", "", ai_note.strip()]

    abs_path = svc.vault_root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text("\n".join(body) + "\n")
    try:
        await svc.index_note(rel, abs_path.read_text())
    except Exception:  # noqa: BLE001
        pass


async def _loop(db: Any) -> None:
    while True:
        try:
            await run_once(db)
        except Exception as e:  # noqa: BLE001
            elog("vault_maintenance.loop_error", error=str(e)[:200])
        interval_h = _int_env(
            "OPENAGENT_VAULT_MAINTENANCE_INTERVAL_HOURS", _DEFAULT_INTERVAL_H
        )
        try:
            await asyncio.sleep(max(60, interval_h * 3600))
        except asyncio.CancelledError:
            elog("vault_maintenance.loop_cancelled")
            raise


def start(db: Any) -> Optional[asyncio.Task]:
    """Schedule the vault-maintenance loop. Returns the task, or ``None``
    when the feature is off (caller treats that as success)."""
    if not _is_enabled():
        return None
    elog(
        "vault_maintenance.start",
        interval_hours=_int_env(
            "OPENAGENT_VAULT_MAINTENANCE_INTERVAL_HOURS", _DEFAULT_INTERVAL_H
        ),
    )
    return asyncio.create_task(_loop(db), name="vault-maintenance")
