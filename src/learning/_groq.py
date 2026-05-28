"""Shared Groq HTTP client used by the learning hooks.

Lives outside ``models/native_provider.py`` because the learning workers
need a stripped-down completion call (single prompt → JSON or plain text)
without the Agno MCP toolkits, history management, or streaming surface.
A bare ``groq`` SDK request is ~200 ms; layering Agno adds 1-2 s of setup
overhead per call which would make the post-turn flush noticeable on
Telegram even though it runs async.

The API key is resolved on every call (cheap dict read) so a rotation
via ``UPDATE providers ... WHERE name='groq'`` takes effect on the next
flush without a process restart. The Groq SDK *client itself* is cached
at module level — instantiating ``AsyncGroq`` opens a fresh httpx
session and pays a TLS handshake, which we'd otherwise burn on every
classifier + profile flush + skill detect call (3 round-trips per
post-turn cycle).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from src.core.logging import elog

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Cached AsyncGroq clients keyed by api_key. Keyed (rather than singleton)
# so a key rotation lands cleanly — the old client stays connected for
# in-flight requests, the new key spins up its own session. Same client
# instance is reused across N concurrent ``complete()`` calls; the
# underlying httpx pool handles per-request concurrency.
_groq_client_cache: dict[str, Any] = {}


async def _resolve_api_key(db: Any) -> Optional[str]:
    """Pull the Groq API key from the ``providers`` table, falling back
    to ``GROQ_API_KEY`` env. Returns ``None`` when neither path yields a
    key — the caller skips the LLM step and the learning hook becomes a
    no-op until the operator configures Groq.
    """
    env = (os.environ.get("GROQ_API_KEY") or "").strip()
    if env:
        return env
    if db is None:
        return None
    try:
        rows = await db.list_providers()
    except Exception:
        return None
    for row in rows or []:
        # Accept any framework — Groq under Agno carries the key in the
        # same ``api_key`` column regardless of which dispatch layer
        # wraps it. ``provider.name == "groq"`` is the canonical match
        # per ``catalog.SUPPORTED_PROVIDERS``.
        if (row.get("name") or "").strip().lower() == "groq":
            key = (row.get("api_key") or "").strip()
            if key:
                return key
    return None


async def complete(
    *,
    db: Any,
    prompt: str,
    system: str | None = None,
    model: str = _DEFAULT_MODEL,
    timeout_s: float = 20.0,
    max_tokens: int = 1024,
    log_event: str = "learning.groq_call",
    session_id: str | None = None,
) -> Optional[str]:
    """Single-shot Groq chat completion. Returns the assistant text or
    ``None`` on any failure (no exceptions propagate to the caller's
    turn loop). Failures are logged via ``elog`` so an operator can see
    the learning hook silently degrading instead of mysteriously not
    updating profiles / skills.
    """
    api_key = await _resolve_api_key(db)
    if not api_key:
        elog(log_event + ".skipped", reason="no_api_key", session_id=session_id)
        return None
    try:
        # Imported here so the rest of the package loads even when the
        # groq SDK is missing — keeps learning a soft-optional feature.
        from groq import AsyncGroq  # type: ignore
    except Exception as e:  # pragma: no cover — declared core dep
        elog(log_event + ".skipped", reason="groq_import_failed", error=str(e))
        return None

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        client = _groq_client_cache.get(api_key)
        if client is None:
            client = AsyncGroq(api_key=api_key)
            _groq_client_cache[api_key] = client
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        elog(log_event + ".timeout", session_id=session_id, model=model, timeout_s=timeout_s)
        return None
    except Exception as e:
        elog(log_event + ".error", session_id=session_id, model=model, error=str(e)[:200])
        return None

    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        elog(log_event + ".malformed_response", session_id=session_id)
        return None
