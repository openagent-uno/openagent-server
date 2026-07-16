"""Core Agent class: orchestrates model, MCP pool, and memory."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import threading
from typing import Any, AsyncIterator, Callable, Awaitable

from src.models.base import BaseModel, ModelResponse
from src.memory.db import MemoryDB
from src.mcp.pool import MCPPool
from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT, build_mcp_catalog_summary
from src.models.runtime import wire_model_runtime

from src.core.logging import elog


def _now_local():
    """Current wall-clock time in the agent's configured timezone.

    Uses ``scheduler.timezone`` when set (so the date the agent records matches
    the operator's day, not the container's UTC), falling back to host local
    time. Imported lazily to keep this module's import graph light and to
    avoid a cycle through the scheduler package at load time.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.memory.schedule import default_timezone_name

    tz = default_timezone_name()
    if tz:
        try:
            return datetime.now(ZoneInfo(tz))
        except Exception:  # noqa: BLE001 — a bad tz name must never break a turn
            pass
    return datetime.now()

logger = logging.getLogger(__name__)

_FROZEN_RUNTIME_PRELOADS = (
    "src.models.discovery",
    "src.channels.voice",
    "src.channels.tts_local",
    # Runtime submodules that ``native_provider`` (and ``mcp.pool``)
    # import lazily on first use. Like the src modules above, they
    # live in the PyInstaller archive; a sibling-service binary swap
    # on performa breaks the deferred zlib extraction and surfaces as
    # ``zlib.error: Error -3 ... incorrect header check`` raised out
    # of ``_ensure_team``/``_dispatch`` and reported as
    # ``agent.run.error``.
    "src.core._runner.agent",
    "src.core._runner.team",
    "src.memory.store.sqlite",
    "src.memory.store.base",
    "src.memory.sessions.agent",
    "src.core._run_state.agent",
    "src.core._run_state.base",
    "src.core._run_state.team",
    "src.models.providers.utils",
    "src.models.providers.message",
    "src.mcp._runtime",
    "src.mcp._runtime.mcp",
)


def _format_run_error(e: BaseException) -> str:
    """Produce a chat-renderable error string for any agent-run failure.

    Two shapes:
      - ``NativeProviderError`` carries an already-clean provider message
        (e.g. "API status error from OpenAI API: 403 - You are not
        allowed to sample from this model"). Prefix with a stable
        marker so bridges and the app can detect it as an error and
        style it accordingly.
      - Anything else falls back to ``Error: <ClassName>: <repr>`` so
        the user sees *something* even on novel exception types.
    """
    from src.models.native_provider import NativeProviderError

    if isinstance(e, NativeProviderError):
        return f"⚠️ Model provider error\n\n{e}"
    msg = str(e) or repr(e)
    return f"⚠️ {type(e).__name__}: {msg}" if msg else f"⚠️ {type(e).__name__}"


def _preload_frozen_runtime_modules() -> None:
    """Eagerly import late-loaded modules in frozen builds.

    PyInstaller onefile processes still lazy-read code from the on-disk
    executable when a module hasn't been imported yet. Performa runs
    multiple services from the same shared binary, so replacing that file
    during a deploy can break a long-lived sibling later when background
    warmup tasks finally import voice/discovery modules. Preloading the
    known late imports up front pins them in ``sys.modules`` before any
    sibling can swap the executable.
    """
    try:
        from src._frozen import is_frozen
    except Exception:  # noqa: BLE001
        return
    if not is_frozen():
        return
    for module_name in _FROZEN_RUNTIME_PRELOADS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            elog(
                "frozen.preload_error",
                level="warning",
                module=module_name,
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
                exc_info=True,
            )


def _format_shell_reminder(events) -> str:
    """Format terminal shell events into a <system-reminder> block."""
    lines = ["Background shell status update since your last message:"]
    for ev in events:
        if ev.kind == "completed":
            detail = f"completed with exit_code={ev.exit_code}"
        elif ev.kind == "timed_out":
            detail = "timed_out"
        else:
            detail = f"killed ({ev.signal or 'unknown'})"
        lines.append(
            f"- shell_id={ev.shell_id}: {detail}. stdout_bytes={ev.bytes_stdout}, "
            f"stderr_bytes={ev.bytes_stderr}. Call shell_output to read."
        )
    lines.append(
        "The user has not sent a new message; continue the task from where "
        "you left off, or summarise and stop if the work is complete."
    )
    body = "\n".join(lines)
    return f"<system-reminder>\n{body}\n</system-reminder>"


_AGNO_IMAGE_MIMES = frozenset({
    "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp",
    "image/bmp", "image/tiff", "image/tif", "image/avif", "image/heic", "image/heif",
})
_AGNO_AUDIO_MIMES = frozenset({
    "audio/wav", "audio/wave", "audio/mp3", "audio/mpeg", "audio/ogg",
    "audio/mp4", "audio/m4a", "audio/aac", "audio/flac", "audio/webm",
})
_AGNO_VIDEO_MIMES = frozenset({
    "video/x-flv", "video/quicktime", "video/mpeg", "video/mpegs", "video/mpgs",
    "video/mpg", "video/mp4", "video/webm", "video/wmv", "video/3gpp",
})

# Extension → MIME fallback for clients that don't send Content-Type.
_EXT_TO_MIME = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
    "tiff": "image/tiff", "tif": "image/tiff", "avif": "image/avif",
    "heic": "image/heic", "heif": "image/heif",
    "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
    "m4a": "audio/m4a", "aac": "audio/aac", "flac": "audio/flac",
    "mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm",
    "mkv": "video/x-matroska", "wmv": "video/wmv",
    "pdf": "application/pdf", "json": "application/json",
    "txt": "text/plain", "md": "text/markdown", "csv": "text/csv",
    "html": "text/html", "css": "text/css", "xml": "text/xml",
    "rtf": "text/rtf", "py": "text/x-python", "js": "text/javascript",
    "yaml": "text/plain", "yml": "text/plain", "log": "text/plain",
}


def _infer_mime(filename: str | None, declared: str | None) -> str | None:
    """MIME inference matching AgentOS's permissive shape.

    Prefers the upload's declared Content-Type when present; falls back
    to the filename extension. Returns ``None`` only when we can't
    guess at all — the caller decides what to do with that.
    """
    if declared:
        return declared
    if not filename or "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    return _EXT_TO_MIME.get(ext)


def _build_runtime_media(
    attachments: list[dict] | None,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Convert wire-format attachments into runtime media objects, AgentOS-style.

    Mirrors the per-MIME dispatch in
    ``the runtime media-ingest path:process_image/audio/video/document``:
    reads each upload's bytes once and constructs typed runtime media
    objects with ``content=bytes`` (NOT ``filepath=...``). Returns four
    parallel lists — ``(images, audios, videos, files)`` — ready to be
    routed to ``arun``'s separate kwargs.

    Why content=bytes instead of filepath: the runtime's model adapters
    consume bytes directly (base64 → multimodal API content). The
    filepath shape exists only as a back-compat input form, and it
    forces every downstream consumer to think about sandbox
    allow-lists. With bytes in
    memory, downstream code can inline text, base64-encode for an API,
    or write to a known-safe location — its choice.

    Anything we can't classify by MIME is skipped with a structured
    warning so a flaky upload doesn't tank the whole turn.
    """
    if not attachments:
        return ([], [], [], [])
    from src.stream.media import Audio, File as _RuntimeFile, Image, Video

    images: list[Any] = []
    audios: list[Any] = []
    videos: list[Any] = []
    files: list[Any] = []

    for a in attachments:
        a_path = a.get("path") or ""
        a_name = a.get("filename") or (a_path.rsplit("/", 1)[-1] if a_path else None)
        a_mime = _infer_mime(a_name, a.get("mime_type"))
        if not a_path:
            elog("agent.media.skip", level="warning",
                 filename=a_name, reason="no_path")
            continue

        content: bytes | None
        try:
            with open(a_path, "rb") as fh:
                content = fh.read()
        except OSError as exc:
            # The bytes weren't readable (file not on disk yet, permission,
            # NFS hiccup, …). Don't drop the attachment — fall back to
            # ``filepath=`` so the runtime still sees it. Downstream
            # consumers that need bytes can re-read on demand; tools that
            # just need a path (file-search MCPs, shell, etc.) still work.
            elog("agent.media.read_skip", level="warning",
                 filename=a_name, path=a_path, error=str(exc))
            content = None

        ext = (a_name.rsplit(".", 1)[-1].lower() if a_name and "." in a_name else None)

        try:
            if a_mime in _AGNO_IMAGE_MIMES:
                if content is not None:
                    images.append(Image(content=content, format=ext, mime_type=a_mime))
                else:
                    images.append(Image(filepath=a_path, format=ext, mime_type=a_mime))
            elif a_mime in _AGNO_AUDIO_MIMES:
                if content is not None:
                    audios.append(Audio(content=content, format=ext, mime_type=a_mime))
                else:
                    audios.append(Audio(filepath=a_path, format=ext, mime_type=a_mime))
            elif a_mime in _AGNO_VIDEO_MIMES:
                if content is not None:
                    videos.append(Video(content=content, format=ext, mime_type=a_mime))
                else:
                    videos.append(Video(filepath=a_path, format=ext, mime_type=a_mime))
            else:
                # File path. The runtime's ``File`` accepts ``content``
                # AND ``filepath`` together (unlike Image/Audio/Video,
                # which require exactly one source). We pass both when
                # the bytes are available so downstream code can pick
                # whichever it prefers — multimodal APIs consume content,
                # file-search tools follow the path.
                file_kwargs: dict[str, Any] = {
                    "filepath": a_path,
                    "filename": a_name,
                    "format": ext,
                }
                if content is not None:
                    file_kwargs["content"] = content
                try:
                    if a_mime in _RuntimeFile.valid_mime_types():
                        file_kwargs["mime_type"] = a_mime
                except Exception:  # noqa: BLE001
                    pass
                files.append(_RuntimeFile(**file_kwargs))
        except Exception as exc:  # noqa: BLE001
            elog("agent.media.build_skip", level="warning",
                 filename=a_name, mime=a_mime, error=str(exc) or type(exc).__name__)

    return (images, audios, videos, files)


_VAULT_WRITE_TOOLS = frozenset({
    "vault_write_note",
    "vault_patch_note",
    "vault_update_frontmatter",
    "vault_delete_note",
    "vault_move_note",
    "vault_manage_tags",
})

# Every name here MUST be a live tool key — ``test_vault_recall`` asserts it
# against the real registrations, because this set has already rotted once.
# ``vault_list_notes`` and ``vault_get_backlinks`` sat here from the day it
# shipped and matched NOTHING: the browse leaf is ``vault_list_directory`` and
# the backlinks tool is ``vault_backlinks``. So ``vault_reads`` undercounted
# for its whole life, missing the two tools the agent browses the vault with.
#
# The trap is that two servers spell their keys differently: ``vault`` is a
# Node SUBPROCESS, so the pool prefixes its keys (``vault_read_note``), while
# ``vault-gate`` is IN-PROCESS, so its keys are the bare python function names
# (``vault_backlinks`` — not ``vault_gate_backlinks``). Guessing from the
# server name is what produced the phantoms.
_VAULT_READ_TOOLS = frozenset({
    # ``vault`` (Node subprocess → prefixed keys)
    "vault_read_note",
    "vault_read_multiple_notes",
    "vault_search_notes",
    "vault_list_directory",
    "vault_get_frontmatter",
    "vault_get_notes_info",
    "vault_list_all_tags",
    "vault_get_vault_stats",
    # ``vault-gate`` (in-process → bare function-name keys)
    "vault_backlinks",
    "vault_search",
    "vault_stats",
})


def _emit_tool_call_summary(
    response: Any, *, session_id: str | None, iter_count: int,
) -> None:
    """Log per-iteration tool call breakdown to events.jsonl.

    Best-effort: silently no-ops when the provider didn't populate
    ``tool_names_called``.

    DO NOT TRUST THESE COUNTS AS A VAULT BASELINE. This was written to measure
    how often the agent writes to the vault, and it has never once measured it:
    the production log holds ZERO ``agent.turn.tool_calls`` entries across
    11,360 events (2026-05-18 → 2026-07-14). ``tool_names_called`` is only
    populated by the NON-streaming providers, and production streams —
    ``runtime.generate`` fired 11 times in that window against 697 streamed
    turns. So this covers ~2% of traffic, and a prompt tweak "evaluated"
    against it would be evaluated against noise. Same defect class as the one
    ``src/models/stream_usage.py`` was written to fix.

    Fixing it here is not possible from inside this function: the streaming
    loop (``_run_inner_stream``) has no ``ModelResponse`` to hand it, only
    deltas. Vault-recall attribution therefore hooks the tool executions
    themselves, on both paths — see ``src/core/vault_recall.py``. This hook is
    left as-is because it is the only per-ITERATION tool breakdown there is,
    and it is honest about non-streamed runs.
    """
    tool_names = list(getattr(response, "tool_names_called", None) or [])
    if not tool_names:
        return
    by_server: dict[str, int] = {}
    vault_writes = 0
    vault_reads = 0
    for name in tool_names:
        server = name.split("_", 1)[0] if "_" in name else name
        by_server[server] = by_server.get(server, 0) + 1
        if name in _VAULT_WRITE_TOOLS:
            vault_writes += 1
        elif name in _VAULT_READ_TOOLS:
            vault_reads += 1
    elog(
        "agent.turn.tool_calls",
        session_id=session_id,
        iter=iter_count,
        by_server=by_server,
        vault_writes=vault_writes,
        vault_reads=vault_reads,
        total=len(tool_names),
    )


async def _with_vault_reminder(db: Any, session_id: str | None, text: str) -> str:
    """Prepend the periodic memory-checkpoint nudge to a turn's input.

    Hooked here — on the shared run path — rather than in any one channel,
    because this is where chat, delegation, the scheduler, workflow AI blocks
    and event runs all converge (every origin reaches the model through
    ``Agent.run`` / ``run_stream``; see ``core/child_session.py``). It used to
    live in ``bridges/base.py``, which meant a desktop or gateway-only install
    got no nudge at all and no automated run ever did. Vision §15: "There is
    no reduced or alternate baseline for non-interactive execution paths — the
    agent is the same agent wherever it runs"; §7 says a scheduled task
    "writes any results back to the vault… exactly as it would for a turn the
    user typed in person".

    Deliberately unconditional: ``maybe_render_reminder`` owns the enabled
    check and the cadence, and returns ``None`` when off or when this turn
    isn't a checkpoint. Re-checking the flag here is what broke the feature
    last time — see ``learning/vault_reminder.py``.

    Never raises: a memory nudge must not be able to fail a turn.
    """
    if db is None or not session_id:
        return text
    try:
        from src.learning.vault_reminder import maybe_render_reminder

        reminder = await maybe_render_reminder(db, session_id)
    except Exception as exc:  # noqa: BLE001
        elog(
            "vault_reminder.hook_error",
            level="warning",
            session_id=session_id,
            error_type=type(exc).__name__,
            error=str(exc) or repr(exc),
        )
        return text
    return f"{reminder}\n\n{text}" if reminder else text


# ── Auto-recall: semantic memory surfaced before a turn ───────────────
#
# This is Layer B — "recall automatici". Before a turn we run a cheap semantic
# search (``src/memory/semantic_index.py``) over the user's message and surface
# the top hits so the agent sees possibly-relevant notes/sessions it would
# otherwise have to think to go look for. It is also the single most dangerous
# change in this area, so it is built SAFE, not naive:
#
#   * OFF BY DEFAULT and INERT WITHOUT AN EMBEDDING MODEL. With no
#     ``OPENAGENT_EMBEDDING_MODEL`` the embedder resolves to ``None`` and this
#     returns the turn text byte-identical — retrieval falls back to FTS,
#     exactly as before the semantic layer existed (§17).
#   * THRESHOLDED. Only hits at/above ``min_score`` are surfaced; a weak match
#     injects NOTHING. The real eSound vault has ~1,167 orphans and unreconciled
#     contradictions — blindly injecting a stale note every turn would make the
#     agent answer confidently wrong, the exact hallucination this is meant to
#     PREVENT. A floor is what keeps noise out.
#   * FRAMED, NOT ASSERTED. The block is a ``<system-reminder>`` that says these
#     are UNVERIFIED leads to CHECK against current state, never established
#     fact. Sibling of ``_with_vault_reminder``; same wrapper, same rationale.
#   * BOUNDED. Top-K small and a hard char cap, because this fires on EVERY turn
#     including every sub-agent and cron firing (§15), so its cost multiplies.
#   * CACHE-SAFE. Prepended to the USER-MESSAGE path (like the vault reminder),
#     NEVER to the cached system prefix. Per-turn content in the ~10.8k-token
#     cached prefix busts the cache every turn — the exact regression the
#     ``<session-id>`` split in ``_combined_system_prompt`` guards against.
#   * NEVER RAISES and is TIME-BOXED. It embeds the query (one network call) off
#     the event loop under a timeout; a slow/unreachable endpoint degrades to
#     "inject nothing", never to a stalled or failed turn.
#
# Outcome-weighting (prefer notes that preceded good runs, via
# ``vault_recall_stats``) is DEFERRED — see ``_recall_block``. For now recency
# is the tie-breaker the search already applies through ``updated``.

# One SemanticIndex per source DB, shared across agent instances (same db = same
# cache file). Built lazily on first recall; the inert (no-embedder) case is
# NOT cached, so enabling a model later — a restart, like a provider key —
# takes effect without stale state.
_RECALL_INDEX_CACHE: dict[str, Any] = {}
_RECALL_INDEX_LOCK = threading.Lock()
# One-shot guard so a missing-numpy (or other import) failure is logged once,
# not per turn — see ``_get_recall_index``.
_RECALL_IMPORT_WARNED = False

# ── Hybrid recall (Layer A ∪ Layer B) ─────────────────────────────────
# Semantic search matches MEANING but its cosine scores compress into a narrow
# band (nomic ~0.59–0.83) where relevant notes and noise OVERLAP, so no single
# ``min_score`` cleanly separates them — a refund-policy note scored 0.592
# (below the floor) while a generic thread scored 0.604 (above it). FTS keyword
# search does the opposite: it nails EXACT terms ("rimborso" → the refund rule)
# that semantic ranks below the floor. So we run BOTH and fuse them: an FTS hit
# is injected regardless of its semantic score, which is precisely the note the
# semantic floor was dropping. Fusion is Reciprocal Rank Fusion (RRF, k=60) —
# rank-based so it needs no score calibration between the two very different
# scales, and a note found by both sides is boosted. Degrades cleanly (§17):
# embedder down → FTS-only; FTS index unavailable → semantic-only.
_FTS_INDEX_CACHE: dict[str, Any] = {}
_FTS_INDEX_LOCK = threading.Lock()
_FTS_IMPORT_WARNED = False
_RRF_K = 60  # standard RRF damping constant


def _recall_enabled() -> bool:
    return (
        os.environ.get("OPENAGENT_AUTO_RECALL_ENABLED", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )


def _recall_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _recall_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_recall_index(agent: Any) -> Any:
    """Return the shared :class:`SemanticIndex` for this agent's DB, or ``None``
    when the semantic layer is inert (no embedding model resolved)."""
    db = getattr(agent, "_db", None)
    db_path = getattr(db, "db_path", None)
    if not db_path:
        return None
    db_path = str(db_path)
    with _RECALL_INDEX_LOCK:
        cached = _RECALL_INDEX_CACHE.get(db_path)
        if cached is not None:
            return cached
        try:
            from src.memory.semantic_index import SemanticIndex, resolve_embedder
        except Exception as exc:  # noqa: BLE001 — numpy/module issue must not break turns
            # Don't fail the turn, but DON'T fail silently either: a missing
            # numpy in a frozen build disables all of semantic recall, and a
            # silent ``return None`` made that look like "recall found nothing"
            # for hours. Log once so it's diagnosable.
            global _RECALL_IMPORT_WARNED
            if not _RECALL_IMPORT_WARNED:
                _RECALL_IMPORT_WARNED = True
                elog("auto_recall.import_failed", level="warning",
                     error=str(exc) or type(exc).__name__,
                     hint="semantic recall is INERT — is numpy in the build?")
            return None
        embedder = resolve_embedder(getattr(agent, "_providers_config", None))
        if embedder is None:
            return None  # inert; not cached so a later config takes effect
        try:
            vault_root = agent._resolve_vault_path()
        except Exception:  # noqa: BLE001
            vault_root = None
        try:
            idx = SemanticIndex(db_path, vault_root=vault_root, embedder=embedder)
        except Exception as exc:  # noqa: BLE001
            elog("auto_recall.index_open_error", level="warning",
                 error=str(exc) or type(exc).__name__)
            return None
        _RECALL_INDEX_CACHE[db_path] = idx
        return idx


def _hybrid_enabled() -> bool:
    """Hybrid FTS∪semantic recall. Default ON; set to 0/false for semantic-only
    (the pre-hybrid behaviour), which is what the tests pin as the fallback."""
    return (
        os.environ.get("OPENAGENT_AUTO_RECALL_HYBRID", "1").strip().lower()
        in ("1", "true", "yes", "on")
    )


def _get_vault_fts_index(agent: Any) -> Any:
    """Return a cached FTS :class:`VaultIndex` over this agent's vault, or
    ``None`` when it can't be opened. Mirrors ``_get_recall_index`` (cache keyed
    by vault root; open failures degrade to None so recall stays semantic-only).

    Synced once on open so a cold index is usable; steady-state freshness rides
    the shared WAL db that the gateway's ``VaultService`` keeps reconciled. This
    runs from the recall worker thread — the sync is a cheap stat scan (no
    embeddings), bounded by the caller's ``OPENAGENT_AUTO_RECALL_TIMEOUT``."""
    try:
        vault_root = agent._resolve_vault_path()
    except Exception:  # noqa: BLE001
        vault_root = None
    if not vault_root:
        return None
    key = str(vault_root)
    with _FTS_INDEX_LOCK:
        cached = _FTS_INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from src.memory.vault.index import VaultIndex
            from src.memory.vault.service import default_index_path
        except Exception as exc:  # noqa: BLE001 — an import issue must not break turns
            global _FTS_IMPORT_WARNED
            if not _FTS_IMPORT_WARNED:
                _FTS_IMPORT_WARNED = True
                elog("auto_recall.fts_import_failed", level="warning",
                     error=str(exc) or type(exc).__name__)
            return None
        try:
            idx = VaultIndex(vault_root, default_index_path(vault_root))
            idx.sync()  # populate a cold index once; warm re-opens are a stat scan
        except Exception as exc:  # noqa: BLE001
            elog("auto_recall.fts_open_error", level="warning",
                 error=str(exc) or type(exc).__name__)
            return None
        _FTS_INDEX_CACHE[key] = idx
        return idx


def _rrf_merge(sem_hits: list[dict], fts_hits: list[dict], limit: int) -> list[dict]:
    """Fuse semantic hits (already ``min_score``-filtered) with FTS note hits by
    Reciprocal Rank Fusion, deduped by identity, best first, capped at ``limit``.

    The whole point: an FTS-matched note is admitted even when its semantic score
    is below the floor (it isn't in ``sem_hits`` at all) — that is how the
    exact-term policy note the floor dropped gets back in. A note found by both
    sides accumulates both RRF contributions and rises.
    """
    fused: dict[tuple, dict[str, Any]] = {}

    def _key(h: dict) -> tuple:
        return ("note", h.get("path")) if h.get("kind") != "session" \
            else ("session", h.get("session_id"))

    for rank, h in enumerate(sem_hits):
        k = _key(h)
        fused[k] = {"hit": dict(h), "rrf": 1.0 / (_RRF_K + rank)}

    for rank, h in enumerate(fts_hits):
        k = ("note", h.get("path"))
        if k in fused:
            fused[k]["rrf"] += 1.0 / (_RRF_K + rank)
            fused[k]["hit"]["fts_matched"] = True
        else:
            # FTS-only: no semantic score (below the floor or unembedded).
            fused[k] = {
                "hit": {"kind": "note", "path": h.get("path"),
                        "title": h.get("title") or "", "score": None,
                        "fts_matched": True},
                "rrf": 1.0 / (_RRF_K + rank),
            }

    ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    return [e["hit"] for e in ranked[:limit]]


def _format_recall_block(hits: list[dict], max_chars: int) -> str:
    """Render recall hits as a verify-framed ``<system-reminder>`` block.

    The FRAMING is the safety property, not decoration: a note surfaced as fact
    that turns out stale is a confident hallucination. So the block says plainly
    that these are unverified leads to check, and how to check them.
    """
    header = (
        "Possibly-relevant memory (semantic match on the user's message). "
        "These are UNVERIFIED and may be stale, superseded, or contradicted — "
        "treat each as a LEAD to check against current state before relying on "
        "it (read the note with vault_read_note, or open the session), never as "
        "established fact. If none is actually relevant, ignore this block."
    )
    lines = [header]
    for h in hits:
        score = h.get("score")
        if h.get("kind") == "note":
            label = f"note `{h.get('path', '')}`"
            upd = h.get("updated")
            if upd:
                label += f" (updated {upd})"
        else:
            title = h.get("title") or "untitled"
            label = f"past session `{h.get('session_id', '')}` — {title}"
        # How the hit was found: semantic similarity, exact keyword, or both.
        if score is None:
            tag = "keyword match"
        elif h.get("fts_matched"):
            tag = f"similarity {score} + keyword"
        else:
            tag = f"similarity {score}"
        lines.append(f"- {label}  [{tag}]")
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + " …"
    return f"<system-reminder>\n{body}\n</system-reminder>"


def _recall_block(agent: Any, query: str, session_id: str | None = None) -> str:
    """Sync worker (runs off the event loop): warm, search, format. Returns the
    ``<system-reminder>`` string, or ``""`` when nothing clears the threshold.

    Outcome-weighting via ``vault_recall_stats`` is deferred here: the honest
    tie-breaker today is recency (the index carries each note's ``updated``),
    and the threshold is what does the real quality-gating. Wiring the recall
    ledger in — prefer notes with a good measured ok_rate — is the follow-up.
    """
    k = max(1, _recall_int("OPENAGENT_AUTO_RECALL_TOP_K", 3))
    floor = _recall_float("OPENAGENT_AUTO_RECALL_MIN_SCORE", 0.75)
    # SEMANTIC side (Layer B). Runs only when an embedder is wired; when it isn't,
    # sem_hits stays empty and recall rides FTS alone — that IS the §17 fallback,
    # so we do NOT bail here the way the pre-hybrid code did.
    idx = _get_recall_index(agent)
    semantic_active = idx is not None and idx.active
    sem_hits: list[dict] = []
    if semantic_active:
        # Warm a bounded number of changed items so a cold index becomes useful
        # over the first few turns without an unbounded burst of embedding calls.
        # Steady state embeds only the query below. 0 disables warming.
        warm = _recall_int("OPENAGENT_AUTO_RECALL_WARM_BUDGET", 24)
        if warm > 0:
            try:
                idx.sync(max_items=warm)
            except Exception:  # noqa: BLE001 — a warm failure must not block recall
                pass
        sem_hits = idx.search(query, scope="all", limit=k, min_score=floor)
    hits = sem_hits
    # FTS side (Layer A) — fuse in exact-term keyword hits so a note the semantic
    # floor dropped still surfaces. Independent of the embedder: this is what
    # makes recall degrade to FTS-only when the embedder is down, and it never
    # blocks recall (open/search failures → semantic-only).
    fts_used = False
    if _hybrid_enabled():
        fts_idx = _get_vault_fts_index(agent)
        if fts_idx is not None:
            try:
                fts_k = max(1, _recall_int("OPENAGENT_AUTO_RECALL_FTS_TOP_K", 3))
                fts_hits = fts_idx.search(query, limit=fts_k)
            except Exception:  # noqa: BLE001 — FTS failure → semantic-only
                fts_hits = []
            if fts_hits:
                fts_used = True
                # Cap the fused set a little above k so FTS-only policy notes get
                # room without unbounding the injected block.
                extra = max(0, _recall_int("OPENAGENT_AUTO_RECALL_FTS_EXTRA", 2))
                hits = _rrf_merge(sem_hits, fts_hits, limit=k + extra)
    # Inert: neither layer could run (no embedder AND no FTS) — byte-identical to
    # pre-recall, and nothing to record.
    if not semantic_active and not fts_used:
        return ""
    # Quality monitor (opt-in): record this turn's recall outcome — hit-rate and
    # top-score feed the aggregate and the min_score tuning signal. No-op when
    # the monitor is off; safe from this worker thread (logging is thread-safe).
    # top_score is the strongest SEMANTIC cosine among the fused hits; FTS-only
    # hits carry no cosine and don't contribute to it.
    try:
        from src.core import quality_monitor
        _top = max((h["score"] for h in hits if h.get("score") is not None),
                   default=0.0)
        quality_monitor.note_recall(
            session_id, used=True, hits=len(hits), top_score=_top)
    except Exception:  # noqa: BLE001 — a metric must never block recall
        pass
    if not hits:
        return ""
    max_chars = max(200, _recall_int("OPENAGENT_AUTO_RECALL_MAX_TOKENS", 400) * 4)
    return _format_recall_block(hits, max_chars)


async def _with_recall(agent: Any, session_id: str | None, query: str,
                       text: str) -> str:
    """Prepend a semantic-recall ``<system-reminder>`` to a turn's input.

    Sibling of :func:`_with_vault_reminder`, hooked on the same shared run path
    so chat, delegation, the scheduler and workflow AI blocks all get it (§15).
    ``query`` is the RAW user message (what to embed); ``text`` is what to
    prepend to (already carrying the vault reminder). Cache-safe: the result is
    the user-message string, never the system prompt.

    Never raises, and time-boxed: the embed + brute-force cosine run off the
    event loop under ``OPENAGENT_AUTO_RECALL_TIMEOUT`` seconds. A miss, an error,
    or a slow endpoint all degrade to returning ``text`` unchanged.
    """
    if not _recall_enabled() or not query or not query.strip():
        return text
    try:
        timeout = _recall_float("OPENAGENT_AUTO_RECALL_TIMEOUT", 4.0)
        block = await asyncio.wait_for(
            asyncio.to_thread(_recall_block, agent, query, session_id), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — recall must never fail a turn
        elog("auto_recall.hook_error", level="warning",
             session_id=session_id, error_type=type(exc).__name__,
             error=str(exc) or repr(exc))
        return text
    return f"{block}\n\n{text}" if block else text


# Status callback type: async def on_status(status: str) -> None
StatusCallback = Callable[[str], Awaitable[None]]


class Agent:
    """Main agent class. Ties together a model, MCP pool, and memory.

    OpenAgent owns the *product* layer (catalog, pricing, gateway, channels,
    memory vault, dormant-MCP detection). Tool execution and the per-call
    tool loop are delegated to the active provider:

      - ``NativeProvider`` consumes ``MCPPool.runtime_toolkits`` (``MCPTools``
        instances) and the runtime ``Agent`` runs the loop internally, including
        proper image-artifact handling for binary tool results.

    ``Agent.run`` is a single ``model.generate`` call — the
    provider returns the final content after running its own tool loop.

    Long-term memory lives in the Obsidian-style vault exposed through MCP.
    The SQLite database is used for runtime state such as scheduler tasks,
    platform-managed chat sessions, and usage tracking.

    Usage:
        agent = Agent(
            name="assistant",
            model=NativeProvider(model="anthropic:claude-sonnet-4-20250514"),
            system_prompt="You are a helpful assistant.",
            mcp_pool=None,  # ``initialize`` rebuilds from the ``mcps`` DB table
            memory=MemoryDB("agent.db"),
        )
        async with agent:
            response = await agent.run("Hello!", user_id="user-1")
    """

    def __init__(
        self,
        name: str = "agent",
        model: BaseModel | None = None,
        system_prompt: str = "You are a helpful assistant.",
        mcp_pool: MCPPool | None = None,
        memory: MemoryDB | str | None = None,
        config: dict | None = None,
        fallback_config: Any | None = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.config = config or {}
        self.fallback_config = fallback_config

        # MCPPool — owns the lifecycle of all MCP servers for the process.
        # Pass an empty pool if not provided so dormant detection / system
        # prompt building still work without crashing.
        self._mcp = mcp_pool if mcp_pool is not None else MCPPool([])

        # Runtime DB; the long-term knowledge base still lives in the Obsidian vault via MCP.
        if isinstance(memory, str):
            self._db = MemoryDB(memory)
        elif isinstance(memory, MemoryDB):
            self._db = memory
        else:
            self._db = None

        self._initialized = False
        self._idle_cleanup_task: asyncio.Task | None = None
        self._runtime_models: list[BaseModel] = []
        self._last_response_meta: dict[str, dict[str, Any]] = {}

        # Materialised provider+model catalog from the SQLite ``providers`` /
        # ``models`` tables. Populated by ``_hydrate_providers_from_db`` at
        # boot and on every hot-reload tick; the yaml config is never
        # consulted for this state.
        self._providers_config: list[dict[str, Any]] = []

        # Per-model in-flight counters + drain events. Keyed by id(model).
        # Used by swap_model() to hold old models alive until their last
        # generate() call returns, then shutdown them asynchronously.
        self._inflight_counts: dict[int, int] = {}
        self._drain_events: dict[int, asyncio.Event] = {}

    @property
    def memory_db(self) -> MemoryDB | None:
        """Expose the runtime DB. Public accessor for REST handlers
        and manager MCPs so they don't poke at ``_db`` directly.
        """
        return self._db

    @staticmethod
    def _response_meta_key(session_id: str | None) -> str:
        return session_id or "__default__"

    def _store_response_meta(self, session_id: str | None, response: ModelResponse | None) -> None:
        key = self._response_meta_key(session_id)
        if response is None or not response.model:
            self._last_response_meta.pop(key, None)
            return
        self._last_response_meta[key] = {"model": response.model}

    def last_response_meta(self, session_id: str | None) -> dict[str, Any]:
        return dict(self._last_response_meta.get(self._response_meta_key(session_id), {}))

    def _register_runtime_model(self, model: BaseModel | None) -> None:
        """Track every model instance that may need lifecycle management."""
        if model is None:
            return
        if any(existing is model for existing in self._runtime_models):
            return
        self._runtime_models.append(model)

    def _unregister_runtime_model(self, model: BaseModel | None) -> None:
        """Remove *model* from the runtime registry (no-op if absent)."""
        if model is None:
            return
        self._runtime_models = [m for m in self._runtime_models if m is not model]

    def _prepare_model_runtime(self, model: BaseModel | None) -> None:
        """Wire shared runtime dependencies into models that support them."""
        if model is None:
            return
        self._register_runtime_model(model)
        wire_model_runtime(
            model,
            db=self._db,
            mcp_pool=self._mcp,
            fallback_config=self.fallback_config,
        )

    def _acquire_model_slot(self, model: BaseModel | None) -> BaseModel | None:
        """Increment the in-flight counter for *model*. Returns *model* unchanged."""
        if model is None:
            return None
        key = id(model)
        self._inflight_counts[key] = self._inflight_counts.get(key, 0) + 1
        return model

    def _release_model_slot(self, model: BaseModel | None) -> None:
        """Decrement the in-flight counter for *model*; fire drain event at zero."""
        if model is None:
            return
        key = id(model)
        remaining = self._inflight_counts.get(key, 0) - 1
        if remaining <= 0:
            self._inflight_counts.pop(key, None)
            ev = self._drain_events.pop(key, None)
            if ev is not None:
                ev.set()
        else:
            self._inflight_counts[key] = remaining

    def swap_model(self, new_model: BaseModel) -> tuple[BaseModel | None, asyncio.Event]:
        """Atomically replace ``self.model`` with *new_model*.

        Returns ``(old_model, drain_event)``. The caller should
        ``await drain_event.wait()`` in a background task and then call
        ``old_model.shutdown()`` to release its resources after its last
        in-flight ``generate()`` call has completed.

        If the old model had no in-flight calls, ``drain_event`` is already
        set so the caller can shut down immediately.
        """
        old = self.model
        self._prepare_model_runtime(new_model)
        self.model = new_model
        self._ensure_idle_cleanup_task()

        if old is None or old is new_model:
            ev = asyncio.Event()
            ev.set()
            return old, ev

        key = id(old)
        if self._inflight_counts.get(key, 0) <= 0:
            ev = asyncio.Event()
            ev.set()
        else:
            ev = self._drain_events.setdefault(key, asyncio.Event())

        # Keep *old* in the runtime registry so Agent.shutdown() will
        # still clean it up if the process exits before drain completes.
        # Caller must call _unregister_runtime_model(old) after shutdown.
        return old, ev

    def _ensure_idle_cleanup_task(self) -> None:
        """Start the idle cleanup loop if any runtime model supports it."""
        if self._idle_cleanup_task and not self._idle_cleanup_task.done():
            return
        if any(callable(getattr(model, "cleanup_idle", None)) for model in self._runtime_models):
            self._idle_cleanup_task = asyncio.create_task(self._run_idle_cleanup())

    async def release_session(
        self,
        session_id: str | None,
        *,
        model_override: BaseModel | None = None,
    ) -> None:
        """Release live runtime resources tied to one session, if supported."""
        if not session_id:
            return
        model = model_override or self.model
        if model is None:
            return
        self._prepare_model_runtime(model)
        close_session = getattr(model, "close_session", None)
        if not callable(close_session):
            return
        await close_session(session_id)
        try:
            from src.mcp.servers.shell.handlers import get_hub
            await get_hub().purge_session(session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("shell hub purge for %s failed: %s", session_id, e)

    def known_model_session_ids(
        self, *, model_override: BaseModel | None = None
    ) -> list[str]:
        """Return every session_id the primary model has resume state for.

        Used by the gateway's ``/clear`` code path to reach past its own
        in-memory SessionManager (which starts empty after a restart) and
        forget conversations whose bridge session ids were hydrated back
        into the model from disk. Also includes sessions persisted in the
        ``sessions`` table so that Claude CLI sessions without a live
        SDK session mapping still appear in the list.
        """
        import sqlite3

        model = model_override or self.model
        sids: set[str] = set()
        if model is not None:
            known = getattr(model, "known_session_ids", None)
            if callable(known):
                try:
                    sids.update(known())
                except Exception:
                    pass
        if self._db is not None:
            db_path = getattr(self._db, "db_path", None)
            if db_path:
                try:
                    conn = sqlite3.connect(str(db_path), timeout=0.2)
                    try:
                        rows = conn.execute(
                            "SELECT session_id FROM sessions"
                        ).fetchall()
                        sids.update(str(r[0]) for r in rows if r and r[0])
                    finally:
                        conn.close()
                except Exception:
                    pass
        return list(sids)

    async def request_cancel(self, session_id: str) -> bool:
        """Cooperatively cancel the in-flight run for ``session_id``.

        Forwards to ``self.model`` (runtime providers register the run in
        agno's cancellation registry so it persists with partial messages).
        Returns True if a run was registered for cancellation; False if the
        model can't cooperatively cancel (caller hard-cancels instead).
        """
        if not session_id or self.model is None:
            return False
        fn = getattr(self.model, "request_cancel", None)
        if not callable(fn):
            return False
        try:
            return bool(await fn(session_id))
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.debug(
                "request_cancel on %s failed: %s",
                type(self.model).__name__, e,
            )
            return False

    async def forget_session(
        self,
        session_id: str | None,
        *,
        model_override: BaseModel | None = None,
    ) -> None:
        """Erase all resume state for ``session_id`` so the next run starts fresh.

        Stronger than :meth:`release_session`: also drops the provider-native
        session id mapping, so the next message spawns a new subprocess
        without ``--resume``. Gateway ``/clear`` and ``/new`` call this so
        users can actually wipe the conversation.
        """
        if not session_id:
            return
        model = model_override or self.model
        if model is None:
            return
        self._prepare_model_runtime(model)
        forget_session = getattr(model, "forget_session", None)
        if callable(forget_session):
            await forget_session(session_id)
        else:
            # Fallback: release live resources even if provider lacks explicit
            # forget support — best-effort; SDK-side resume state may linger.
            close_session = getattr(model, "close_session", None)
            if callable(close_session):
                await close_session(session_id)
        try:
            from src.mcp.servers.shell.handlers import get_hub
            await get_hub().purge_session(session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("shell hub purge for %s failed: %s", session_id, e)

    async def initialize(self) -> None:
        """Connect MCP servers and initialize memory DB.

        The ``mcps`` / ``providers`` / ``models`` SQLite tables are the
        sole sources of truth at runtime. ``ensure_builtin_mcps`` runs
        every boot to backfill any missing builtin rows (forward compat
        + safety net); the MCP pool is then (re)built from the DB via
        ``MCPPool.from_db`` so the runtime can hot-reload entries
        without a process restart (see ``reload_mcps_if_changed``).
        """
        if self._initialized:
            return
        elog("agent.initialize.start", agent=self.name, model_class=type(self.model).__name__)
        if self._db:
            await self._db.connect()

        # Hydrate providers/models from the DB and swap to the DB-backed
        # MCP pool. Skipped when there is no DB (pure in-memory tests);
        # in that case we fall back to whatever pool the caller passed in.
        if self._db is not None:
            try:
                from src.memory.bootstrap import ensure_builtin_mcps
                # Every boot: re-seed any BUILTIN_MCP_SPECS entry that
                # doesn't have a row yet (forward-compat for future
                # builtins + safety net against manual DB tampering).
                # Existing rows — including disabled ones — are untouched.
                await ensure_builtin_mcps(self._db)
                # Provider keys and the model catalog are DB-backed. Pull
                # the rows into ``self._providers_config`` so ModelDispatcher
                # / NativeProvider see the materialised view.
                await self._hydrate_providers_from_db()
                self._providers_last_updated = await self._db.providers_max_updated()
                self._models_last_updated = await self._db.models_max_updated()
                # Hand the freshly-hydrated list to every live runtime
                # model. ModelDispatcher was constructed with an empty
                # providers_config; without this push it would keep that
                # empty reference until the first hot-reload tick — which
                # only fires on gateway messages, so scheduler turns that
                # run before any user chat would see an empty catalog and
                # reject with "no_enabled_model".
                providers_config = self._providers_config
                for model in list(self._runtime_models) + [self.model]:
                    if model is None:
                        continue
                    rebuild = getattr(model, "rebuild_routing", None)
                    if callable(rebuild):
                        try:
                            rebuild(providers_config)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("rebuild_routing on boot failed: %s", exc)
            except Exception as exc:  # noqa: BLE001 — bootstrap must not block startup
                elog("bootstrap.error", level="warning", error=str(exc))

            try:
                db_path = getattr(self._db, "db_path", None)
                new_pool = await MCPPool.from_db(self._db, db_path=db_path)
                self._mcp = new_pool
                self._mcps_last_updated = await self._db.mcps_max_updated()
            except Exception as exc:  # noqa: BLE001 — leave the existing pool untouched
                elog("pool.from_db_error", level="warning", error=str(exc))

        _preload_frozen_runtime_modules()
        await self._mcp.connect_all()

        self._prepare_model_runtime(self.model)
        self._ensure_idle_cleanup_task()

        # Prime OpenRouter's catalog in the background so ``get_model_pricing``
        # has live rates before the first cost attribution, without blocking
        # startup on a network call. Errors are swallowed — the catalog has a
        # bundled offline backstop.
        async def _prime_openrouter() -> None:
            try:
                from src.models.discovery import _fetch_openrouter_catalog
                await _fetch_openrouter_catalog()
            except Exception as exc:  # noqa: BLE001
                # Some shutdown-time exceptions stringify to "" — also
                # capture the type and full traceback so events.jsonl
                # has something to triage from.
                elog(
                    "openrouter.prefetch_error",
                    level="warning",
                    error=str(exc) or type(exc).__name__,
                    error_type=type(exc).__name__,
                    exc_info=True,
                )

        # Warm the local Whisper model in the background so the first
        # voice-tab utterance doesn't pay the 60s+ download/load tax
        # (small ≈ 464 MB; ~10s cold-load even when cached locally).
        # By the time the user records anything, the model is in RAM.
        # Errors swallowed — transcribe() lazy-loads as a fallback.
        async def _prime_whisper() -> None:
            try:
                from src.channels.voice import _load_local_model
                await _load_local_model()
                elog("whisper.prefetch_done")
            except Exception as exc:  # noqa: BLE001
                elog(
                    "whisper.prefetch_error",
                    level="warning",
                    error=str(exc) or type(exc).__name__,
                    error_type=type(exc).__name__,
                    exc_info=True,
                )

        # Same idea for Piper: cold-load is ~10s for the ONNX model
        # plus a one-time ~25 MB voice-file download. Prefetch so the
        # first reply doesn't sit silent for 12s before audio plays.
        async def _prime_piper() -> None:
            try:
                from src.channels import tts_local
                if not tts_local.is_available():
                    return
                # Resolve to the configured default voice and load it.
                # ``_load_voice`` is the exact path synth uses, so a
                # successful prefetch guarantees the next synth is warm.
                voice = tts_local._resolve_voice_name(None)
                loaded = await tts_local._load_voice(voice)
                if loaded is not None:
                    elog("piper.prefetch_done", voice=voice)
            except Exception as exc:  # noqa: BLE001
                elog(
                    "piper.prefetch_error",
                    level="warning",
                    error=str(exc) or type(exc).__name__,
                    error_type=type(exc).__name__,
                    exc_info=True,
                )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_prime_openrouter())
            loop.create_task(_prime_whisper())
            loop.create_task(_prime_piper())
        except RuntimeError:
            # No running loop (sync entry point) — skip; all three
            # backends lazy-load on first request.
            pass

        self._initialized = True
        elog(
            "agent.initialize.done",
            agent=self.name,
            model_class=type(self.model).__name__,
            mcp_servers=self._mcp.server_count,
            tools=self._mcp.total_tool_count,
            has_db=bool(self._db),
        )

    async def refresh_registries(self) -> tuple[bool, int]:
        """Combined hot-reload probe for the gateway's dispatcher.

        One SQLite round-trip (``registry_status``) returns the max
        timestamps for the mcps / models / providers tables plus the
        enabled model count. We then reload whatever is stale and
        return the count so the caller can short-circuit when zero
        models are enabled. Returns ``(reloaded_anything, enabled_models)``.

        Provider edits (``api_key``, ``base_url``) invalidate the cached
        ``providers_config`` dict that ModelDispatcher hands to NativeProvider
        — without this hook, adding a key would require a restart.
        """
        if self._db is None:
            return False, -1
        try:
            mcps_updated, models_updated, enabled_count, providers_updated = (
                await self._db.registry_status()
            )
        except Exception as exc:  # noqa: BLE001 — never gate a message on this probe
            logger.debug("registry_status probe failed: %s", exc)
            return False, -1

        reloaded = False
        if mcps_updated > getattr(self, "_mcps_last_updated", 0.0):
            self._mcps_last_updated = mcps_updated
            try:
                await self._mcp.reload()
                for model in list(self._runtime_models):
                    wire_model_runtime(model, db=self._db, mcp_pool=self._mcp)
                reloaded = True
            except Exception as exc:  # noqa: BLE001
                elog("mcps.reload_error", level="warning", error=str(exc))

        providers_changed = providers_updated > getattr(self, "_providers_last_updated", 0.0)
        if providers_changed:
            self._providers_last_updated = providers_updated
            try:
                await self._hydrate_providers_from_db()
                elog("providers.reload")
                reloaded = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("providers hydrate failed: %s", exc)

        # Models or providers changed → rebuild router. Providers affect
        # routing because NativeProvider's api_key lookup goes through
        # ``providers_config``; models affect it because the classifier
        # picks from the materialised models list.
        models_changed = models_updated > getattr(self, "_models_last_updated", 0.0)
        if models_changed or providers_changed:
            self._models_last_updated = max(
                models_updated, getattr(self, "_models_last_updated", 0.0) or 0.0
            )
            if models_changed and not providers_changed:
                # Providers hydrate already ran above; re-run only when
                # models alone changed so the materialised catalog is fresh.
                try:
                    await self._hydrate_providers_from_db()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("models hydrate failed: %s", exc)
            providers_config = self._providers_config
            for model in list(self._runtime_models):
                rebuild = getattr(model, "rebuild_routing", None)
                if callable(rebuild):
                    try:
                        rebuild(providers_config)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("rebuild_routing failed: %s", exc)
            elog("models.reload")
            reloaded = True

        return reloaded, enabled_count

    async def _hydrate_providers_from_db(self) -> None:
        """Pull provider + model rows from the DB into ``self._providers_config``.

        The DB is the source of truth for provider keys AND the model
        catalog. ModelDispatcher / NativeProvider consume the v0.12 flat-list
        shape — each entry already carries its ``framework`` and its
        nested ``models`` list. Delegates the SQL materialisation to
        MemoryDB so
        smoke-test endpoints can reuse the same shape.
        """
        if self._db is None:
            return
        try:
            self._providers_config = await self._db.materialise_providers_config(
                enabled_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("providers hydrate failed: %s", exc)
            self._providers_config = []

    async def _run_idle_cleanup(self) -> None:
        """Periodically release idle provider resources."""
        while True:
            await asyncio.sleep(60)
            for model in list(self._runtime_models):
                cleanup_idle = getattr(model, "cleanup_idle", None)
                if not callable(cleanup_idle):
                    continue
                try:
                    released_ids = await cleanup_idle()
                    if released_ids:
                        try:
                            from src.mcp.servers.shell.handlers import get_hub
                            for sid in released_ids:
                                await get_hub().purge_session(sid)
                        except Exception as e:  # noqa: BLE001
                            logger.debug("shell hub purge on idle cleanup failed: %s", e)
                except Exception as e:
                    logger.debug("Idle cleanup error: %s", e)

    async def shutdown(self) -> None:
        """Close all connections."""
        elog("agent.shutdown.start", agent=self.name)
        if self._idle_cleanup_task:
            self._idle_cleanup_task.cancel()
            self._idle_cleanup_task = None
        # Persistent model runtimes may need an explicit shutdown to
        # release subprocesses or cached sessions cleanly.
        seen: set[int] = set()
        for model in [self.model, *self._runtime_models]:
            if model is None or id(model) in seen:
                continue
            seen.add(id(model))
            shutdown = getattr(model, "shutdown", None)
            if callable(shutdown):
                try:
                    await shutdown()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Model shutdown error: %s", e)
        await self._mcp.close_all()
        try:
            from src.mcp.servers.shell.handlers import get_hub
            await get_hub().shutdown()
        except Exception as e:  # noqa: BLE001
            logger.debug("shell hub shutdown failed: %s", e)
        if self._db:
            await self._db.close()
        self._initialized = False
        self._runtime_models.clear()
        elog("agent.shutdown.done", agent=self.name)

    async def run(
        self,
        message: str,
        user_id: str = "",
        session_id: str | None = None,
        attachments: list[dict] | None = None,
        on_status: StatusCallback | None = None,
        model_override: BaseModel | None = None,
        author: dict | None = None,
    ) -> str:
        """Run the agent with a user message. Returns the final text response.

        Args:
            session_id: Session key passed through to whichever history mode
                the active model uses.
            on_status: Optional async callback for live status updates.
                Called with status strings like "Thinking...", "Using shell_exec...", etc.
                Channels use this to update a live status message.
            author: Optional per-message author for this turn (a human handle,
                or an agent-self seed for delegated/scheduled/workflow runs).
                Stamped onto the user message and persisted in the runs JSON;
                never sent to the model. See src.core.identity_context.
        """
        if not self.model:
            raise RuntimeError("No model configured. Set agent.model before calling run().")

        await self.initialize()
        self._prepare_model_runtime(model_override)
        self._ensure_idle_cleanup_task()

        async def _status(msg: str) -> None:
            if on_status:
                try:
                    await on_status(msg)
                except Exception:
                    pass

        try:
            self._store_response_meta(session_id, None)
            elog(
                "agent.run.start",
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
                model_class=type(model_override or self.model).__name__,
                attachments=len(attachments or []),
            )
            result = await self._run_inner(message, attachments, _status, session_id=session_id, model_override=model_override, author=author)
            # Quality monitor (opt-in, sampled): grade this completed turn off
            # the reply path — fire-and-forget, so the judge's latency/cost never
            # sit on the response. Zero allocation + no task when disabled.
            try:
                from src.core import quality_monitor
                quality_monitor.spawn_scoring(self, session_id, message, result)
            except Exception:  # noqa: BLE001 — monitoring must never affect the turn
                pass
            return result
        except asyncio.CancelledError:
            # Shutdown or task-level cancellation is NOT a fatal error — it's
            # the runtime telling us to stop cleanly. Log it as such, tell the
            # caller something useful (empty ``str(CancelledError)`` used to
            # surface as "Error:" with nothing after), and re-raise so the
            # caller's cancellation semantics are preserved.
            elog(
                "agent.run.cancelled",
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
            )
            logger.info("Agent.run() cancelled for session %s", session_id)
            raise
        except BaseException as e:
            # Include error_type so we can tell a KeyError from a
            # ConnectionResetError from a RuntimeError. The old format
            # swallowed the type for exceptions whose ``__str__`` is "".
            elog(
                "agent.run.error",
                level="error",
                exc_info=True,
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
            return _format_run_error(e)

    async def _run_inner(
        self,
        message: str,
        attachments: list[dict] | None,
        _status,
        session_id: str | None = None,
        model_override: BaseModel | None = None,
        author: dict | None = None,
    ) -> str:
        """Run a single agent turn, continuing the session automatically when
        background shells complete during or shortly after it.

        Providers handle the internal tool-loop (API-based via its Agent, Claude
        SDK via its native MCP support), so each call to ``model.generate``
        returns post-tool-loop content. This method adds a wrapper loop
        above ``generate`` that:

        1. After each turn, drains the shell hub for terminal events
           (shell_exec+run_in_background=True) for ``session_id``.
        2. If any terminal event landed, formats it as a ``<system-reminder>``
           and re-enters ``generate`` on the same session — same subprocess
           (Claude), same runtime history — so the model sees the completion
           mid-conversation.
        3. If no events landed but shells are still running, awaits
           ``hub.wait`` up to ``shell.wake_wait_window_seconds`` before
           giving up and returning to the caller.
        4. Caps at ``shell.autoloop_cap`` iterations to prevent runaway
           chains, logged via ``agent.run.autoloop_cap_hit``.

        Returns the final ``ModelResponse.content`` after the loop settles.
        """
        await _status("Loading context...")

        # Combine OpenAgent's framework-level guidelines with the user's
        # project-specific system prompt from src.yaml. Passing
        # ``session_id`` appends a ``<session-id>`` tag so the LLM can
        # call tools that operate on its own session (e.g. pin_session).
        system = self._combined_system_prompt(session_id=session_id)

        # AgentOS-aligned media handling: split attachments by MIME and
        # construct typed runtime media objects (Image / Audio / Video /
        # File) with ``content=bytes``, then pass each list to ``arun``'s
        # corresponding kwarg. API-native model adapters consume the
        # bytes directly (multimodal API content).
        media_images, media_audios, media_videos, media_files = _build_runtime_media(attachments)

        # Images still get a textual prepend — but non-image attachments
        # are now routed natively through the runtime's ``files=`` kwarg
        # so the leader doesn't paraphrase synthetic file-info blocks into
        # delegation tasks.
        if attachments:
            from src.channels.base import build_attachment_context, prepend_context_block
            image_atts = [a for a in attachments if (a.get("type") or "file") == "image"]
            if image_atts:
                files_info: list[str] = []
                for a in image_atts:
                    a_name = a.get("filename", "")
                    a_path = a.get("path", "")
                    if a_path:
                        files_info.append(f"- image: {a_name} — local path: {a_path}")
                    else:
                        files_info.append(f"- image: {a_name}")
                message = prepend_context_block(
                    message,
                    build_attachment_context(
                        files_info,
                        read_hint=(
                            "Use the Read tool (or an MCP tool) with the local path to inspect each image. "
                            "For images, Read returns the image content for you to see directly."
                        ),
                    ),
                )

        from src.mcp.servers.shell.handlers import get_hub
        from src.mcp.servers.shell.adapters import set_session_context, reset_session_context
        from src.mcp.servers.delegation.handlers import (
            install_context as install_delegation_context,
            reset_context as reset_delegation_context,
        )
        from src.core.identity_context import (
            install_author_context, reset_author_context, owner_handle_of,
        )
        from src.core.config import shell_settings

        hub = get_hub()
        settings = shell_settings(getattr(self, "config", None) or {})
        wake_window = settings.wake_wait_window_seconds
        cap = settings.autoloop_cap

        active_model = self._acquire_model_slot(model_override or self.model)

        # Applied to the first input only: the autoloop's shell-reminder
        # re-entries below reassign ``current_input``, so a long tool-driven
        # turn doesn't re-pay the nudge on every iteration.
        current_input = await _with_vault_reminder(self._db, session_id, message)
        # Semantic auto-recall, on the same user-message path (cache-safe).
        # ``message`` (not ``current_input``) is embedded so the recall query is
        # the user's actual words, not the reminder prose wrapped around them.
        current_input = await _with_recall(self, session_id, message, current_input)
        last_response = None
        iter_count = 0

        pending = hub.drain(session_id)
        if pending:
            pre = _format_shell_reminder(pending)
            current_input = f"{pre}\n\n{current_input}"

        try:
            while True:
                iter_count += 1
                if iter_count > cap:
                    elog(
                        "agent.run.autoloop_cap_hit",
                        session_id=session_id,
                        cap=cap,
                    )
                    break

                # In-session compaction (vision §2): before driving the
                # model, check whether the cumulative stored history is
                # about to breach the model's context budget. If so,
                # fold the oldest runs into a recap row first so the
                # next ``add_history_to_context=True`` reload stays
                # under the limit. The reactive ContextWindowExceeded
                # fallback (src/models/providers/fallback.py) still
                # backstops this in case the heuristic underestimates.
                if iter_count == 1 and session_id:
                    try:
                        from src.core.compaction import should_compact, compact
                        if should_compact(session_id, active_model, agent=self):
                            await compact(
                                session_id, active_model, self,
                                on_status=_status,
                            )
                    except Exception as exc:  # noqa: BLE001 — never block a turn
                        elog(
                            "runtime.compaction.error",
                            level="warning",
                            session_id=session_id,
                            error_type=type(exc).__name__,
                            error=str(exc) or repr(exc),
                        )

                messages: list[dict[str, Any]] = [{"role": "user", "content": current_input}]
                await _status("Thinking...")

                token = set_session_context(session_id)
                # Record the active chat session so the vault autocommit can
                # attribute out-of-band note writes (external vault MCP) to it.
                try:
                    from src.memory.vault.vault_origin import note_activity
                    note_activity(kind="chat", session=session_id)
                except Exception:  # noqa: BLE001
                    pass
                # ``delegate_task`` MCP routes through ``dispatcher.run_delegated``,
                # which only ``ModelDispatcher`` (the canonical ``self.model``)
                # implements — a per-turn ``TeamRouterProvider`` built by
                # ``model_override`` is pinned to one entry runtime and can't
                # dispatch to other models. Live chat happens to pass because
                # ``active_model is self.model`` there; the workflow ai-prompt
                # block with ``model_override`` is the path that breaks.
                delegation_dispatcher = (
                    self.model
                    if hasattr(self.model, "run_delegated")
                    else active_model
                )
                delegation_tokens = install_delegation_context(
                    session_id=session_id,
                    pool=self._mcp,
                    db=self._db,
                    dispatcher=delegation_dispatcher,
                    agent=self,
                    # Owner handle for any child session spawned this turn. Only
                    # a human author carries one; agent-self / automation runs
                    # resolve to None and child_session inherits from the parent row.
                    owner_handle=owner_handle_of(author),
                )
                author_token = install_author_context(author)
                try:
                    # ``files`` is forwarded native to the runtime's ``arun(files=...)``
                    # by NativeProvider / TeamRouterProvider. Only attach on
                    # the first iteration so shell-reminder re-entries don't
                    # re-send the same files.
                    # Only forward media on the first iteration so a
                    # shell-reminder re-entry doesn't re-attach the same
                    # files (the agent already saw them).
                    first = iter_count == 1
                    response = await active_model.generate(
                        messages,
                        system=system,
                        on_status=_status,
                        session_id=session_id,
                        files=media_files if first else None,
                        images=media_images if first else None,
                        audio=media_audios if first else None,
                        videos=media_videos if first else None,
                    )
                finally:
                    reset_session_context(token)
                    reset_delegation_context(delegation_tokens)
                    reset_author_context(author_token)

                last_response = response

                _emit_tool_call_summary(
                    response, session_id=session_id, iter_count=iter_count,
                )

                events = hub.drain(session_id)
                if not events:
                    if not hub.has_running(session_id):
                        break
                    if wake_window > 0:
                        events = await hub.wait(session_id, timeout=wake_window)
                    if not events:
                        break

                elog(
                    "agent.run.autoloop_iter",
                    session_id=session_id,
                    iter=iter_count,
                    events=len(events),
                )
                current_input = _format_shell_reminder(events)
        finally:
            self._release_model_slot(active_model)

        self._store_response_meta(session_id, last_response)
        elog(
            "agent.run.done",
            agent=self.name,
            session_id=session_id,
            model_class=type(active_model).__name__,
            response_len=len((last_response.content if last_response else "") or ""),
        )
        return (last_response.content if last_response else "") or "(Done — no final message was returned.)"

    async def run_stream(
        self,
        message: str,
        user_id: str = "",
        session_id: str | None = None,
        attachments: list[dict] | None = None,
        on_status: StatusCallback | None = None,
        model_override: BaseModel | None = None,
        author: dict | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming sibling of :meth:`run` for voice-mode replies.

        Yields events as plain dicts so the gateway/orchestrator can
        consume them without import gymnastics:

        - ``{"kind": "delta", "text": "..."}`` — incremental text from
          the LLM. Feed this to a sentence chunker → TTS pipeline.
        - ``{"kind": "iteration_break"}`` — emitted between autoloop
          iterations (a tool finished and the agent re-enters
          ``model.stream`` with a shell-reminder). The chunker should
          ``flush()`` here so a sentence split by a tool call isn't
          re-narrated. (Risk #1 in the voice-chat plan.)
        - ``{"kind": "done", "text": "<full text>"}`` — final event with
          the assembled response text (post-marker-strip is up to the
          caller; we just emit raw text).

        Cancellation propagates exactly like :meth:`run`: a
        ``CancelledError`` from the model layer is logged and re-raised.
        """
        if not self.model:
            raise RuntimeError("No model configured. Set agent.model before calling run_stream().")

        await self.initialize()
        self._prepare_model_runtime(model_override)
        self._ensure_idle_cleanup_task()

        async def _status(msg: str) -> None:
            if on_status:
                try:
                    await on_status(msg)
                except Exception:
                    pass

        elog(
            "agent.run_stream.start",
            agent=self.name,
            user_id=user_id,
            session_id=session_id,
            model_class=type(model_override or self.model).__name__,
            attachments=len(attachments or []),
        )

        try:
            async for event in self._run_inner_stream(
                message, attachments, _status,
                session_id=session_id, model_override=model_override,
                author=author,
            ):
                yield event
        except asyncio.CancelledError:
            elog(
                "agent.run_stream.cancelled",
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
            )
            raise
        # ``Exception`` (not ``BaseException``) so we don't catch
        # ``GeneratorExit`` — that's how Python tells us the consumer
        # stopped iterating early (the orchestrator's ``break`` after a
        # ``done`` event triggers ``aclose`` → ``GeneratorExit`` here).
        # Catching it and yielding from the cleanup path is illegal —
        # Python raises ``RuntimeError("async generator ignored
        # GeneratorExit")`` and asyncio leaves the cleanup task
        # un-retrieved. Letting it propagate is the right thing.
        except Exception as e:
            elog(
                "agent.run_stream.error",
                level="error",
                exc_info=True,
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
            yield {"kind": "done", "text": _format_run_error(e)}

    async def _run_inner_stream(
        self,
        message: str,
        attachments: list[dict] | None,
        _status,
        session_id: str | None = None,
        model_override: BaseModel | None = None,
        author: dict | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming variant of :meth:`_run_inner`.

        Mirrors the autoloop logic line-for-line but calls
        ``active_model.stream(...)`` instead of ``generate(...)``.
        Providers that don't override ``stream`` fall back to a single
        post-hoc yield via ``BaseModel.stream`` — voice-chat still works,
        just without the time-to-first-audio win.
        """
        await _status("Loading context...")
        system = self._combined_system_prompt(session_id=session_id)

        # AgentOS-aligned media: see ``_run_inner`` above for the full
        # rationale. Same per-MIME split + content=bytes construction.
        media_images, media_audios, media_videos, media_files = _build_runtime_media(attachments)

        # Same images-prepend / files=-passthrough split as ``_run_inner``.
        if attachments:
            from src.channels.base import build_attachment_context, prepend_context_block
            image_atts = [a for a in attachments if (a.get("type") or "file") == "image"]
            if image_atts:
                files_info: list[str] = []
                for a in image_atts:
                    a_name = a.get("filename", "")
                    a_path = a.get("path", "")
                    if a_path:
                        files_info.append(f"- image: {a_name} — local path: {a_path}")
                    else:
                        files_info.append(f"- image: {a_name}")
                message = prepend_context_block(
                    message,
                    build_attachment_context(
                        files_info,
                        read_hint=(
                            "Use the Read tool (or an MCP tool) with the local path to inspect each image. "
                            "For images, Read returns the image content for you to see directly."
                        ),
                    ),
                )

        from src.mcp.servers.shell.handlers import get_hub
        from src.mcp.servers.shell.adapters import set_session_context, reset_session_context
        from src.mcp.servers.delegation.handlers import (
            install_context as install_delegation_context,
            reset_context as reset_delegation_context,
        )
        from src.core.identity_context import (
            install_author_context, reset_author_context, owner_handle_of,
        )
        from src.core.config import shell_settings

        hub = get_hub()
        settings = shell_settings(getattr(self, "config", None) or {})
        wake_window = settings.wake_wait_window_seconds
        cap = settings.autoloop_cap

        active_model = self._acquire_model_slot(model_override or self.model)

        # Streaming twin of the reminder hook in ``_run_inner`` — see there.
        current_input = await _with_vault_reminder(self._db, session_id, message)
        current_input = await _with_recall(self, session_id, message, current_input)
        accumulated: list[str] = []
        iter_count = 0

        pending = hub.drain(session_id)
        if pending:
            pre = _format_shell_reminder(pending)
            current_input = f"{pre}\n\n{current_input}"

        # When the streaming autoloop yields zero deltas (tool-only turns,
        # or the runtime when no RunContentEvent fires), we fall back to a
        # one-shot generate() so callers always receive text. The real
        # ModelResponse from that call wins for last_response_meta()
        # over the synthetic placeholder.
        fallback_response: ModelResponse | None = None
        try:
            while True:
                iter_count += 1
                if iter_count > cap:
                    elog("agent.run_stream.autoloop_cap_hit",
                         session_id=session_id, cap=cap)
                    break

                # Streaming twin of the in-session compaction call in
                # _run_inner. See that branch (or src/core/compaction.py)
                # for the rationale — vision §2 "the session compacts in
                # place". Only runs on the first iteration so shell-
                # reminder re-entries don't pay the threshold check on
                # every loop.
                if iter_count == 1 and session_id:
                    try:
                        from src.core.compaction import should_compact, compact
                        if should_compact(session_id, active_model, agent=self):
                            await compact(
                                session_id, active_model, self,
                                on_status=_status,
                            )
                    except Exception as exc:  # noqa: BLE001 — never block a turn
                        elog(
                            "runtime.compaction.error",
                            level="warning",
                            session_id=session_id,
                            error_type=type(exc).__name__,
                            error=str(exc) or repr(exc),
                        )

                messages: list[dict[str, Any]] = [{"role": "user", "content": current_input}]
                await _status("Thinking...")

                token = set_session_context(session_id)
                # Record the active chat session so the vault autocommit can
                # attribute out-of-band note writes (external vault MCP) to it.
                try:
                    from src.memory.vault.vault_origin import note_activity
                    note_activity(kind="chat", session=session_id)
                except Exception:  # noqa: BLE001
                    pass
                # See ``_run_inner`` above: delegation must route through the
                # canonical ``ModelDispatcher`` (``self.model``), not the
                # per-turn ``active_model`` which may be a ``TeamRouterProvider``
                # built by ``model_override`` and lacks ``run_delegated``.
                delegation_dispatcher = (
                    self.model
                    if hasattr(self.model, "run_delegated")
                    else active_model
                )
                delegation_tokens = install_delegation_context(
                    session_id=session_id,
                    pool=self._mcp,
                    db=self._db,
                    dispatcher=delegation_dispatcher,
                    agent=self,
                    # Owner handle for any child session spawned this turn. Only
                    # a human author carries one; agent-self / automation runs
                    # resolve to None and child_session inherits from the parent row.
                    owner_handle=owner_handle_of(author),
                )
                author_token = install_author_context(author)
                try:
                    # Pass session_id and on_status so ``ModelDispatcher.stream``
                    # can run the same entry-model resolution that ``generate``
                    # uses: per-session pin -> ``is_classifier``-flagged model ->
                    # first enabled. (No classifier LLM call is involved despite
                    # the flag's name — see ``models/catalog.py``.) The pin lives
                    # in ``pinned_sessions``, so without ``session_id`` there is
                    # nothing to look it up by and voice turns fell through to
                    # "first enabled api-based model" instead of the session's
                    # pinned one — which 403'd on users whose first api-based
                    # model was an OpenAI model their key couldn't access.
                    #
                    # Introspect once instead of try/except TypeError around
                    # the iteration body — a catch-all TypeError swallows
                    # errors raised mid-iteration and the silent retry
                    # without ``session_id`` collides on the ``"default"``
                    # session, which then yields zero deltas → fallback
                    # at line 1089 fires → caller sees ONE giant delta.
                    stream_kwargs: dict[str, Any] = {"system": system}
                    try:
                        sig_params = inspect.signature(
                            active_model.stream
                        ).parameters
                    except (TypeError, ValueError):
                        # Builtins / C-coded callables don't expose a
                        # signature. Skip the introspection — call with
                        # only the always-supported args.
                        sig_params = {}
                    if "session_id" in sig_params:
                        stream_kwargs["session_id"] = session_id
                    if "on_status" in sig_params:
                        stream_kwargs["on_status"] = _status
                    # Only attach media on iteration 1: shell-reminder
                    # re-entries reuse the same the runtime session, which
                    # already has the prior files in its run history.
                    if iter_count == 1:
                        if "files" in sig_params and media_files:
                            stream_kwargs["files"] = media_files
                        if "images" in sig_params and media_images:
                            stream_kwargs["images"] = media_images
                        if "audio" in sig_params and media_audios:
                            stream_kwargs["audio"] = media_audios
                        if "videos" in sig_params and media_videos:
                            stream_kwargs["videos"] = media_videos
                    async for delta in active_model.stream(
                        messages, **stream_kwargs,
                    ):
                        if not delta:
                            continue
                        accumulated.append(delta)
                        yield {"kind": "delta", "text": delta}
                finally:
                    reset_session_context(token)
                    reset_delegation_context(delegation_tokens)
                    reset_author_context(author_token)

                events = hub.drain(session_id)
                if not events:
                    if not hub.has_running(session_id):
                        break
                    if wake_window > 0:
                        events = await hub.wait(session_id, timeout=wake_window)
                    if not events:
                        break

                # Force-flush any in-progress sentence before re-entering the
                # model — a sentence split by a tool call must not be
                # re-narrated. Risk #1 from the voice-chat plan.
                yield {"kind": "iteration_break"}

                elog(
                    "agent.run_stream.autoloop_iter",
                    session_id=session_id, iter=iter_count, events=len(events),
                )
                current_input = _format_shell_reminder(events)

            # Empty-stream safety net: some providers emit zero deltas
            # for tool-only turns, empty completions, or non-streamable
            # backends. Without this
            # fallback voice mode (and the soon-to-be-streaming web
            # chat) would surface a confusing "(no output)" message
            # while ``Agent.run()`` worked fine for the same prompt.
            if not accumulated:
                # A pending cancellation here is ALWAYS a barge-in. The
                # only code that calls ``cancel()`` on this turn's task is
                # ``StreamSession._cancel_active_turn`` (a user interrupt /
                # a new message preempting the turn) and session teardown —
                # nothing else increments ``task.cancelling()``. When the
                # provider's stream swallows that ``CancelledError``
                # internally and returns "cleanly" with zero deltas, the
                # task is left cancellation-poisoned (``cancelling() > 0``)
                # but the stream looks empty.
                #
                # The old behaviour ``uncancel()``-ed that poison and ran a
                # full ``generate()`` to completion. That defeated the
                # barge-in AND — because the runner's caller is blocked on
                # ``await task`` while holding the session's dispatch lock —
                # froze the whole session for the duration of the recovered
                # generate. After a quick burst of messages each preempt
                # stacked another blocking generate and the agent stopped
                # responding entirely. Honour the interrupt instead: let the
                # cancellation propagate promptly so the caller can dispatch
                # the next (merged) turn.
                task = asyncio.current_task()
                if (
                    task is not None
                    and hasattr(task, "cancelling")
                    and task.cancelling() > 0
                ):
                    elog(
                        "agent.run_stream.barge_in",
                        session_id=session_id,
                        pending_cancels=task.cancelling(),
                    )
                    raise asyncio.CancelledError()

                # Genuine empty stream (tool-only / empty completion, no
                # cancel pending): fall back to a one-shot ``generate()`` so
                # the caller doesn't surface a confusing "(no output)".
                elog(
                    "agent.run_stream.fallback_to_generate",
                    session_id=session_id,
                    reason="no_deltas_yielded",
                )
                try:
                    generate_kwargs: dict[str, Any] = {
                        "system": system,
                        "tools": None,
                    }
                    try:
                        gen_params = inspect.signature(
                            active_model.generate
                        ).parameters
                    except (TypeError, ValueError):
                        gen_params = {}
                    if "session_id" in gen_params:
                        generate_kwargs["session_id"] = session_id
                    if "on_status" in gen_params:
                        generate_kwargs["on_status"] = _status

                    fallback_task = asyncio.create_task(
                        active_model.generate(
                            [{"role": "user", "content": message}],
                            **generate_kwargs,
                        ),
                        name=f"generate-fallback:{session_id or 'default'}",
                    )
                    try:
                        fallback_response = await fallback_task
                    except asyncio.CancelledError:
                        # A barge-in arriving during the fallback generate
                        # must stop it too — cancel the child and propagate
                        # rather than running it to completion.
                        fallback_task.cancel()
                        raise
                    _emit_tool_call_summary(
                        fallback_response,
                        session_id=session_id,
                        iter_count=iter_count,
                    )
                    fallback_text = (fallback_response.content or "").strip()
                    if fallback_text:
                        accumulated.append(fallback_text)
                        # Yield as a final delta so SentenceChunker /
                        # TTS / streaming clients see it the same way
                        # they would a normal delta.
                        yield {"kind": "delta", "text": fallback_text}
                except Exception as e:  # noqa: BLE001 — surface in log, return empty
                    fallback_response = None
                    elog(
                        "agent.run_stream.generate_fallback_failed",
                        level="warning",
                        session_id=session_id,
                        error_type=type(e).__name__,
                        error=str(e) or repr(e),
                    )
        finally:
            self._release_model_slot(active_model)

        full_text = "".join(accumulated)
        # Prefer the real ModelResponse from the generate() fallback so
        # last_response_meta() has accurate model + usage. Otherwise
        # synthesize a minimal stand-in from the accumulated text and
        # the *effective* model id — for ModelDispatcher this is the
        # runtime actually picked for the session, not a generic
        # instance attribute. ``getattr(active_model, "model_name",
        # None)`` (the previous code) returned ``None`` for every
        # provider in tree (NativeProvider exposes ``self.model``;
        # ModelDispatcher exposes neither), which silently dropped the
        # model badge from the chat UI after the streaming migration.
        # ``effective_model_id`` is the provider-aware accessor.
        if fallback_response is not None:
            self._store_response_meta(session_id, fallback_response)
        else:
            model_id = active_model.effective_model_id(session_id)
            synthetic = ModelResponse(content=full_text, model=model_id)
            self._store_response_meta(session_id, synthetic)
        elog(
            "agent.run_stream.done",
            agent=self.name,
            session_id=session_id,
            model_class=type(active_model).__name__,
            response_len=len(full_text),
            used_fallback=fallback_response is not None,
        )
        yield {"kind": "done", "text": full_text}

    def _resolve_vault_path(self) -> str:
        """Return the on-disk path the vault MCP is actually using.

        Mirrors the gateway's resolution order
        ([openagent/gateway/api/vault.py]): a YAML-level
        ``memory.vault_path`` override wins, otherwise falls back to
        ``default_vault_path()`` (which already honours ``--agent-dir``
        via the ``_agent_dir`` global in :mod:`openagent.core.paths`).
        Returned as a string ready to splice into the framework prompt.
        """
        from pathlib import Path
        from src.core.paths import default_vault_path

        cfg_path = (
            (self.config or {}).get("memory", {}).get("vault_path")
        )
        if cfg_path:
            return str(Path(cfg_path).expanduser().resolve())
        return str(default_vault_path())

    def _resolve_db_path(self) -> str:
        """Return the SQLite DB path backing runtime state for this agent."""
        from pathlib import Path
        from src.core.paths import default_db_path

        cfg_path = (
            (self.config or {}).get("memory", {}).get("db_path")
        )
        if cfg_path:
            return str(Path(cfg_path).expanduser().resolve())
        db_path = getattr(self._db, "db_path", None)
        if db_path:
            return str(Path(str(db_path)).expanduser().resolve())
        return str(default_db_path())

    def _combined_system_prompt(self, session_id: str | None = None) -> str:
        """Concatenate the framework prompt with the user's project-specific one.

        Substitutes ``{{OPENAGENT_VAULT_PATH}}`` and
        ``{{OPENAGENT_DB_PATH}}`` in the framework prompt with the
        resolved on-disk paths so the agent sees the exact vault and
        SQLite stores for this deployment. Per-agent because each agent
        runs in its own process with its own ``--agent-dir`` (and
        optional ``memory.*_path`` YAML overrides).

        When ``session_id`` is provided we append a ``<session-id>`` tag
        so the LLM can learn its own id and pass it to tools that
        operate on "this session" — e.g.
        ``model-manager.pin_session(session_id=..., runtime_id=...)``.
        The tag is stripped of whitespace and comes last so project
        prompts read cleanly above it.

        Its position is now load-bearing for cost, not just readability.
        ``Claude._build_system`` splits this string at the tag and emits the
        tag as an uncached trailing block, so everything above it is a
        byte-identical prefix across every session on the box and the ~10.8k
        framework prompt is cached once rather than once per session. Moving
        the tag, or appending anything after it, silently makes the cached
        prefix per-session again — a quiet ~1.25x-write-per-session
        regression with no test-visible symptom other than the bill.
        ``src/models/native_provider.py`` and ``src/models/dispatcher.py``
        also match this tag (to key their Agent caches), so its shape is a
        contract, not a formatting choice.
        """
        framework = FRAMEWORK_SYSTEM_PROMPT.replace(
            "{{OPENAGENT_VAULT_PATH}}", self._resolve_vault_path()
        ).replace(
            "{{OPENAGENT_DB_PATH}}", self._resolve_db_path()
        ).replace(
            "{{MCP_CATALOG_SUMMARY}}",
            build_mcp_catalog_summary(self._mcp),
        )

        user = (self.system_prompt or "").strip()
        if not user:
            combined = framework
        else:
            combined = (
                framework
                + "\n\n── User-specific identity and project context ──\n\n"
                + user
            )

        # Tell the agent what day it is. Without this it does not know: the
        # prompt asks it for "absolute date" note fields and deadlines, and the
        # model fills them from its training cutoff — a live dream-log came out
        # dated 2025 while the agent ran in 2026. Every note's ``created:``,
        # every ``dream-log-YYYY-MM-DD.md`` filename, every "<date>: symptom"
        # receipt was a guess.
        #
        # This lands INSIDE the cached prefix, and that is fine — the date is
        # the same for every session on a given day, so the prefix stays
        # byte-identical fleet-wide and caches once PER DAY per box. The cost is
        # one ~10.8k-token prefix rewrite at each midnight boundary, then reads
        # all day: negligible. That is a different thing from the per-SESSION
        # invalidation the ``<session-id>`` split guards against — there, every
        # new session would pay the write. A daily date does not; a per-turn
        # value (a clock time, the session id) would, which is why only the
        # date goes here and the time-of-day is deliberately omitted.
        _now = _now_local()
        combined += (
            f"\n\nThe current date is {_now:%Y-%m-%d} ({_now:%A}). Use it for "
            "any absolute date you record — never guess the year from memory."
        )
        if session_id:
            combined += f"\n\n<session-id>{session_id}</session-id>"
        return combined

    async def stream_run(
        self,
        message: str,
        user_id: str = "",
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream the agent's response. Does not support tool use in streaming mode."""
        if not self.model:
            raise RuntimeError("No model configured.")

        await self.initialize()

        system = self._combined_system_prompt(session_id=session_id)
        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]

        async for chunk in self.model.stream(messages, system=system):
            yield chunk

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()
