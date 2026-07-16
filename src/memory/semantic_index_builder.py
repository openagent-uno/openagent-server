"""Background builder for the semantic recall index.

The on-turn auto-recall hook is time-boxed (``OPENAGENT_AUTO_RECALL_TIMEOUT``,
4s) so it can only ever SEARCH + top up a handful of vectors — building the whole
2000+ note index there silently times out and stores nothing (the symptom that
made recall return 0 hits in prod despite the embedder working). So the BUILD
lives here instead: a background loop, off the turn path and un-time-boxed, that
embeds the vault + sessions to completion, then re-syncs periodically to pick up
new notes. Mirrors ``curator`` / ``vault.autocommit``: a ``start()`` that returns
a task, a no-op when the semantic layer is inert (no embedding model configured).
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from src.core.logging import elog

# One bounded ``sync`` embeds up to _MAX_ITEMS_PER_SYNC (128); loop until nothing
# is pending so the first pass fully builds the index, then re-sync on this
# cadence to pick up notes/sessions written since.
_RESYNC_SECONDS = 300
_FIRST_BUILD_MAX_PASSES = 200  # 200 * 128 = 25.6k items — far above any real vault


def _interval() -> int:
    raw = (os.environ.get("OPENAGENT_SEMANTIC_RESYNC_SECONDS") or "").strip()
    try:
        return max(30, int(raw)) if raw else _RESYNC_SECONDS
    except ValueError:
        return _RESYNC_SECONDS


async def _loop(db_path: str, vault_root: Optional[str], providers_config: Any) -> None:
    from src.memory.semantic_index import SemanticIndex, resolve_embedder

    embedder = resolve_embedder(providers_config)
    if embedder is None:
        return  # inert — no embedding model configured, recall falls back to FTS
    try:
        idx = SemanticIndex(db_path, vault_root=vault_root, embedder=embedder)
    except Exception as exc:  # noqa: BLE001
        elog("semantic.builder_open_error", level="warning",
             error=str(exc) or type(exc).__name__)
        return
    if not idx.active:
        return

    first = True
    while True:
        try:
            passes = 0
            while passes < _FIRST_BUILD_MAX_PASSES:
                passes += 1
                stats = await asyncio.to_thread(idx.sync)
                if (stats["vault"].pending == 0 and stats["sessions"].pending == 0):
                    break
            st = idx.stats()
            if first or st.get("notes"):
                elog("semantic.index_built", notes=st.get("notes", 0),
                     sessions=st.get("sessions", 0), passes=passes, first=first)
            first = False
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001 — a build error must not kill the loop
            elog("semantic.index_build_error", level="warning",
                 error=str(exc) or type(exc).__name__)
        try:
            await asyncio.sleep(_interval())
        except asyncio.CancelledError:
            break
    try:
        idx.close()
    except Exception:  # noqa: BLE001
        pass


def start(db_path: str, vault_root: Optional[str],
          providers_config: Any = None) -> Optional[asyncio.Task]:
    """Start the background index builder, or return None when the layer is inert.

    Cheap to call unconditionally: ``resolve_embedder`` returns None (and the loop
    exits immediately) when no embedding model is configured, so a deployment that
    never turned on semantic recall pays nothing.
    """
    if not db_path:
        return None
    try:
        return asyncio.create_task(_loop(db_path, vault_root, providers_config))
    except RuntimeError:
        return None  # no running loop (e.g. a unit test calling start() directly)
