"""Provider-agnostic single-shot completion for the background learning loops.

Replaces ``_groq.py`` (deleted), which pinned every background learning call to
``groq:llama-3.3-70b-versatile`` and a ``GROQ_API_KEY``. That made Groq a
*structural requirement* for the agent to maintain its own memory — an agent
configured with only a local model could never dream (§12), no matter how many
other providers it had. Vision §17: "Removing any single provider — any LLM
API, any MCP, any channel — must leave the agent operational with what
remains. This rules out architectures in which a particular cloud service is
structurally required."

The model is named by explicit config, never inferred:

    OPENAGENT_LEARNING_MODEL=<provider>:<model>     # e.g. ollama:llama3.1

Why not auto-pick a cheap row? Both candidate signals lie:

* ``models.is_classifier`` does NOT mean "cheap". ``dispatcher._resolve_entry_model``
  reads it as the user's persistent "default team leader" — the model ENTRY
  turns route to (per-session pin → is_classifier → first enabled). On a
  typical setup that is the user's *best* model, so auto-selecting it would
  point an unattended nightly loop at the priciest row in the catalog. It
  would also couple two unrelated subsystems: flipping the flag to move the
  team leader would silently move the dream summariser too.
* ``tier_hint`` is free-form prose ("fast and cheap general-purpose chat").
  Vision §3 is explicit: "Scopes over hardcoded routing. A model's role is a
  sentence the router reads, not a switch statement in code." Grepping
  /cheap|mini|fast/ over that sentence IS the switch statement it forbids, and
  it mis-fires on the first user who writes "not for cheap work".

``src.core.compaction`` reached the same conclusion independently and for the
same reasons; see ``_SUMMARY_MODEL_ENV`` there. This module deliberately
mirrors its shape so there is one way to say "do this cheap bit of AI on a
model the operator named".

Unset → ``None``, and the caller skips its AI step and keeps its mechanical
half. This is where we diverge from compaction, which falls back to the
primary model when unconfigured: compaction is load-bearing for the turn (it
summarises or the turn breaks), whereas every caller here is optional
enrichment running unattended on a timer. A background loop that silently
bills the user's premium model every 12h is the cost bug we are avoiding, not
an acceptable default.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from src.core.logging import elog

_LEARNING_MODEL_ENV = "OPENAGENT_LEARNING_MODEL"


async def _resolve_model(db: Any, *, log_event: str) -> Optional[Any]:
    """Build a bare provider for the configured learning model, or ``None``.

    Strict but never fatal: the id must match an enabled, api-based row in the
    agent's own DB-backed catalog. Anything unresolvable logs once and returns
    ``None`` — a learning hook that can't find its model is a degraded feature,
    never a crash in the caller's loop.

    The provider is built WITHOUT MCP toolkits: these calls take one prompt and
    return prose, so tool schemas would be pure token overhead on exactly the
    call we are trying to keep cheap.
    """
    configured = os.environ.get(_LEARNING_MODEL_ENV, "").strip()
    if not configured:
        return None
    try:
        from src.models.catalog import FRAMEWORK_API_BASED, iter_configured_models
        from src.models.native_provider import NativeProvider

        providers_config = await db.materialise_providers_config(enabled_only=True)
        match = next(
            (
                entry
                for entry in iter_configured_models(providers_config)
                if entry.runtime_id == configured
                and not entry.disabled
                and entry.framework == FRAMEWORK_API_BASED
            ),
            None,
        )
        if match is None:
            elog(
                log_event + ".model_unresolved",
                level="warning",
                configured=configured,
                reason="no_enabled_api_based_row",
            )
            return None
        db_path = getattr(db, "db_path", None)
        return NativeProvider(
            model=match.runtime_id,
            providers_config=providers_config,
            db_path=str(db_path) if db_path else None,
        )
    except Exception as exc:  # noqa: BLE001 — a background nicety must never raise
        elog(
            log_event + ".model_failed",
            level="warning",
            configured=configured,
            error_type=type(exc).__name__,
            error=str(exc) or repr(exc),
        )
        return None


async def complete(
    *,
    db: Any,
    prompt: str,
    system: str | None = None,
    timeout_s: float = 25.0,
    log_event: str = "learning.model_call",
) -> Optional[str]:
    """Single-shot completion on the configured learning model.

    Returns the assistant text, or ``None`` on any failure — no exceptions
    propagate to the caller. Failures are logged via ``elog`` so an operator
    can see the hook degrading rather than mysteriously doing nothing.
    """
    if not prompt or db is None:
        return None
    model = await _resolve_model(db, log_event=log_event)
    if model is None:
        elog(log_event + ".skipped", reason="no_learning_model")
        return None
    try:
        response = await asyncio.wait_for(
            model.generate(
                [{"role": "user", "content": prompt}],
                system=system,
                # No session_id: this is out-of-band housekeeping, not a
                # conversation. Passing one would append a row to a real
                # session's transcript.
                session_id=None,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        elog(log_event + ".timeout", timeout_s=timeout_s)
        return None
    except Exception as exc:  # noqa: BLE001
        elog(
            log_event + ".error",
            error_type=type(exc).__name__,
            error=str(exc)[:200] or repr(exc),
        )
        return None
    return (getattr(response, "content", None) or "").strip() or None
