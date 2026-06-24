"""Background vault autocommit loop.

OpenAgent commits its OWN vault writes immediately (per-change, with precise
provenance). But it cannot hook every writer — the external vault MCP and a
human editing in Obsidian both touch the files directly. This loop is the
safety net: on a short interval it commits any pending changes so that *every*
change to the vault ends up in history, attributed (via ``vault_origin``) to
whatever was active when it happened.

On by default when git is available; opt out with
``memory.vault.git.enabled: false`` or set the interval to 0.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from src.core.logging import elog

_DEFAULT_INTERVAL_S = 25


def _enabled() -> bool:
    return os.environ.get("OPENAGENT_VAULT_GIT_ENABLED", "1").strip().lower() in (
        "1", "true", "yes", "on")


def _interval() -> int:
    try:
        return int(os.environ.get("OPENAGENT_VAULT_GIT_AUTOCOMMIT_SECONDS",
                                  _DEFAULT_INTERVAL_S))
    except (TypeError, ValueError):
        return _DEFAULT_INTERVAL_S


async def _loop() -> None:
    from src.memory.vault.service import get_service
    while True:
        try:
            interval = _interval()
            if interval <= 0:
                await asyncio.sleep(60)
                continue
            await asyncio.sleep(interval)
            svc = get_service()
            commit = await svc.autocommit()
            if commit:
                elog("vault_autocommit.commit", commit=commit)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            elog("vault_autocommit.error", error=str(e)[:200])
            await asyncio.sleep(_interval() or 60)


def start() -> Optional[asyncio.Task]:
    """Start the autocommit loop, ensuring the repo exists first. Returns the
    task, or ``None`` when git is disabled/unavailable."""
    if not _enabled() or _interval() <= 0:
        return None
    from src.memory.vault.gitrepo import resolve_git_bin
    if not resolve_git_bin():
        elog("vault_autocommit.skipped", reason="git_not_found")
        return None
    elog("vault_autocommit.start", interval_s=_interval())
    return asyncio.create_task(_loop(), name="vault-autocommit")
