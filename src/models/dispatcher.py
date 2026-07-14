"""Unified model dispatcher — entry resolution + Team-as-router.

This module is the single top-level model layer wired into the Agent.
It owns two complementary responsibilities, colocated for clarity:

- :class:`ModelDispatcher` — picks the entry runtime_id for each
  session (pin → first-enabled) and routes the turn through the
  matching :class:`TeamRouterProvider`. Also drives lifecycle
  (cleanup_idle, shutdown, close/forget_session) and budget telemetry.

- :class:`TeamRouterProvider` — for one entry runtime_id, builds a
  runtime ``Team(mode=coordinate)`` per session whose leader is that entry
  model and whose members are every OTHER enabled LLM model in the
  catalog. The runtime's TeamMode.coordinate
  lets the leader fire MULTIPLE ``delegate_task_to_member`` tool calls
  per turn — the runtime gathers them with ``asyncio.gather``, so
  independent sub-tasks run in parallel. The leader then synthesizes
  the member outputs into the final user-facing reply.

History is unified in the runtime's sessions SqliteDb, so there's no
need to lock a session to one framework once it's been served.

``SmartRouter`` (the pre-rename name) stays available as an alias at
the bottom of this module for transitional compatibility with tests
and external callers.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from src.core.logging import elog
from src.models import stream_usage
from src.models.base import BaseModel, ModelResponse
from src.models.budget import BudgetTracker
from src.models.catalog import (
    CatalogModel,
    FRAMEWORK_API_BASED,
    FULL_SESSION_HISTORY_RUNS,
    RUNTIME_SESSION_USER_ID,
    framework_of,
    iter_configured_models,
)
from src.models.runtime import wire_model_runtime

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Module-level helpers (shared by TeamRouterProvider + fallback)
# ════════════════════════════════════════════════════════════════════


def _last_user_prompt(messages: list[dict[str, Any]] | None) -> str:
    """Render the latest user turn as a plain string for ``arun``.

    The runtime owns persisted history, so we only push the LATEST user
    message; earlier turns are replayed from the sessions store. Content
    blocks (text, image_url) are flattened to their text portions — image
    parts are dropped because ``arun``'s prompt arg accepts only text.
    """
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(p for p in parts if p)
        return str(content) if content is not None else ""
    return ""


def _runtime_db_path(db: Any) -> str | None:
    """Resolve OpenAgent's SQLite path so the runtime's ``SqliteDb`` lands on
    the same file (the sessions table is shared)."""
    if db is None:
        return None
    path = getattr(db, "db_path", None)
    return str(path) if path else None


_SESSION_ID_TAG_RE = re.compile(r"\n*<session-id>[^<]*</session-id>\s*$")


def _system_cache_key(system: str | None) -> str:
    """Stable cache key for a system prompt that ignores the per-session
    ``<session-id>`` tag the orchestrator appends.

    Without this, every session generates a unique system prompt string
    — so any cache keyed on the raw system text misses on every session
    AND grows unbounded with session count.
    """
    if not system:
        return ""
    return _SESSION_ID_TAG_RE.sub("", system).strip()


def _compose_member_system(system: str | None, role_blurb: str) -> str | None:
    """Compose the per-member system prompt: framework prompt + persona
    on top, a short ``Role`` block on the bottom that pins the
    specialist to its area.

    Mirrors the pattern at ``native_provider.py``'s internal Team (see
    ``_ensure_team``) so both Team paths follow the same shape. Returns
    ``None`` when no framework prompt is supplied — the single-model
    classifier-only generate path can call into the dispatcher with
    ``system=None`` and we mustn't synthesise a prompt out of thin air
    in that case.
    """
    base = (system or "").strip()
    if not base:
        return None
    role = (role_blurb or "").strip()
    if not role:
        return base
    return (
        f"{base}\n\n── Role ──\n"
        f"You are this team's {role} specialist. Prefer tasks that fit "
        f"that role; defer to the team leader when the request is "
        f"outside your area."
    )


def _build_role_blurb(entry: CatalogModel) -> str:
    """Compose the natural-language description the team leader uses
    to route a turn to a specialist member.

    Prefers ``tier_hint`` (e.g. "best for coding"); falls back to a
    generic "specialist using <model>" line so a row with no hint still
    parses.
    """
    hint = (entry.tier_hint or "").strip()
    return hint or f"specialist using {entry.model_id}"


def _team_member_sessions_enabled() -> bool:
    """Whether a coordinate-team member delegation runs as its own durable
    child session (navigable card) rather than nested in the team session.
    Default ON (the chosen design); set
    ``OPENAGENT_TEAM_MEMBER_SESSIONS=0`` to revert to the legacy nested
    ``member_responses`` behavior."""
    return os.environ.get("OPENAGENT_TEAM_MEMBER_SESSIONS", "1").strip() not in ("0", "false", "no")


def _member_identifier(runtime_id: str) -> str:
    """Build a url-safe runtime member identifier from a runtime_id.

    The runtime's ``get_member_id`` runs the agent's ``name`` (when no explicit
    id is set) through ``url_safe_string`` — which strips colons. Feeding
    it a colon-laden ``specialist:anthropic:claude-opus-4.7``
    collapses the id to ``specialistanthropicclaude-opus-4.7``
    while the system message still surfaces the original name, so the
    leader LLM sees two divergent identifiers for the same member and
    hallucinates placeholder ids when delegating.

    Dashes pass through ``url_safe_string`` unchanged, so a colon-to-dash
    swap yields an id that matches the name byte-for-byte.
    """
    return runtime_id.replace(":", "-")


def _extract_delegated_member_id(tool_call: Any) -> str | None:
    """If a ``delegate_task_to_member`` tool call carries a ``member_id``
    argument, return it. Used by both the collect and stream paths to
    surface the delegated specialist as the chat UI's model badge —
    otherwise the badge stays on the team leader even when the actual
    response came from a specialist.
    """
    name = getattr(tool_call, "tool_name", None)
    if name != "delegate_task_to_member":
        return None
    args = getattr(tool_call, "tool_args", None)
    if isinstance(args, dict):
        member_id = args.get("member_id")
        if isinstance(member_id, str) and member_id:
            return member_id
    return None


async def _arun_runtime_collect(
    runtime: Any,
    *,
    prompt: str,
    session_id: str,
    user_id: str,
    error_event: str,
    entry_runtime_id: str,
    on_delegate: Callable[[str], None] | None = None,
    files: list[Any] | None = None,
    images: list[Any] | None = None,
    audio: list[Any] | None = None,
    videos: list[Any] | None = None,
) -> ModelResponse:
    """Drive a runtime ``Team`` / ``Agent``-shaped runtime in non-stream
    mode and translate the resulting ``RunOutput`` into the
    ``ModelResponse`` shape the surrounding pipeline expects.

    Used by ``TeamRouterProvider.generate`` (the Team path) — it calls
    ``runtime.arun(prompt, stream=False)`` and funnels the same five
    fields (content, tool names, input/output tokens, model) into
    ``ModelResponse``.

    Media kwargs (``files``/``images``/``audio``/``videos``) are
    forwarded to ``runtime.arun(..., files=..., images=..., ...)``.
    The runtime propagates the same media to members during
    ``delegate_task_to_member`` — see ``_runner/team/_default_tools.py:713``.
    Splitting by type matches AgentOS's ``process_image/audio/video/document``
    convention.
    """
    arun_kwargs: dict[str, Any] = {
        "session_id": session_id, "user_id": user_id, "stream": False,
    }
    if files:
        arun_kwargs["files"] = files
    if images:
        arun_kwargs["images"] = images
    if audio:
        arun_kwargs["audio"] = audio
    if videos:
        arun_kwargs["videos"] = videos
    try:
        run_output = await runtime.arun(prompt, **arun_kwargs)
    except Exception as e:  # noqa: BLE001
        elog(error_event, session_id=session_id, error=str(e))
        raise
    content = str(getattr(run_output, "content", "") or "")
    tools_used: list[str] = []
    for tc in getattr(run_output, "tools", None) or []:
        name = getattr(tc, "tool_name", None)
        if name:
            tools_used.append(str(name))
        if on_delegate is not None:
            member_id = _extract_delegated_member_id(tc)
            if member_id:
                on_delegate(member_id)
    metrics = getattr(run_output, "metrics", None)
    input_tokens = int(getattr(metrics, "input_tokens", 0) or 0) if metrics else 0
    output_tokens = int(getattr(metrics, "output_tokens", 0) or 0) if metrics else 0
    return ModelResponse(
        content=content,
        tool_names_called=tools_used,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=entry_runtime_id,
    )


async def _arun_runtime_stream(
    runtime: Any,
    *,
    prompt: str,
    session_id: str,
    user_id: str,
    on_status: Callable[[str], Awaitable[None]] | None,
    error_event: str,
    on_delegate: Callable[[str], None] | None = None,
    on_run_id: Callable[[str], None] | None = None,
    files: list[Any] | None = None,
    images: list[Any] | None = None,
    audio: list[Any] | None = None,
    videos: list[Any] | None = None,
) -> AsyncIterator[str]:
    """Drive a runtime ``Team`` / ``Agent``-shaped runtime in stream mode
    and yield string deltas, optionally surfacing tool-call status
    pings via ``on_status``.

    Used by ``TeamRouterProvider.stream`` — it calls
    ``runtime.arun(prompt, stream=True)``.

    Critical: the runtime publishes ``RunContentEvent`` / ``ToolCallStartedEvent``
    under TWO distinct modules — one for plain ``Agent`` runs and one for
    ``Team(mode=route)`` runs. They are NOT the same class, so isinstance
    against just one of them swallows the other side's deltas and triggers
    the ``run_stream`` "no_deltas_yielded" fallback to ``generate()``, which
    then runs the whole turn AGAIN and persists a duplicate row to the
    sessions table. Union both modules below so Team deltas actually
    reach the WS pipe.

    ``IntermediateRunContentEvent`` (team-only) carries deltas from a
    delegated member back to the team leader — also string content,
    also worth surfacing to the user.

    Tool-call frames are emitted as **structured JSON status** (the
    same envelope ``NativeProvider._emit_agno_tool_status`` uses for
    single-agent runs and the rehydration endpoint reconstructs from
    stored ``runs[].tools``). Without this, team-mode turns sent the
    legacy ``"⚙ <name>"`` plain text and the universal app rendered
    them as text-only tool messages while rehydrated sessions for the
    SAME turn produced structured tool chips — the asymmetry users
    reported as "team-member usages sometimes shown as tool components,
    sometimes as text response".
    """
    from src.core._run_state.agent import (
        RunCompletedEvent as _AgentRunCompletedEvent,
        RunContentEvent as _AgentRunContentEvent,
        ToolCallCompletedEvent as _AgentToolCallCompletedEvent,
        ToolCallErrorEvent as _AgentToolCallErrorEvent,
        ToolCallStartedEvent as _AgentToolCallStartedEvent,
    )
    from src.core._run_state.team import (
        IntermediateRunContentEvent as _TeamIntermediateRunContentEvent,
        RunCompletedEvent as _TeamRunCompletedEvent,
        RunContentEvent as _TeamRunContentEvent,
        ToolCallCompletedEvent as _TeamToolCallCompletedEvent,
        ToolCallErrorEvent as _TeamToolCallErrorEvent,
        ToolCallStartedEvent as _TeamToolCallStartedEvent,
    )
    from src.models import stream_usage
    from src.models._tool_status import emit_tool_status

    run_completed_event_types = (
        _AgentRunCompletedEvent,
        _TeamRunCompletedEvent,
    )

    content_event_types = (
        _AgentRunContentEvent,
        _TeamRunContentEvent,
        _TeamIntermediateRunContentEvent,
    )
    tool_start_event_types = (
        _AgentToolCallStartedEvent,
        _TeamToolCallStartedEvent,
    )
    tool_complete_event_types = (
        _AgentToolCallCompletedEvent,
        _TeamToolCallCompletedEvent,
    )
    tool_error_event_types = (
        _AgentToolCallErrorEvent,
        _TeamToolCallErrorEvent,
    )

    async def _emit_status(
        tool: Any, *, error_text: str | None = None, phase: str | None = None,
    ) -> None:
        await emit_tool_status(on_status, tool, error_text=error_text, phase=phase)

    stream_iter = None
    try:
        # ``stream_events=True`` is REQUIRED to receive
        # ``ToolCallStartedEvent`` for the leader's ``delegate_task_to_member``
        # call — without it the runtime only yields ``RunContentEvent`` deltas
        # (the specialist's text) and ``on_delegate`` would never fire,
        # leaving the chat-UI badge pinned to the team leader even when
        # a specialist actually wrote the response.
        #
        # ``files`` is forwarded to ``runtime.arun(..., files=...)``. The
        # runtime propagates the same files to members during
        # ``delegate_task_to_member`` (see
        # ``_runner/team/_default_tools.py:713``).
        stream_kwargs: dict[str, Any] = {
            "session_id": session_id, "user_id": user_id, "stream": True,
            "stream_events": True,
        }
        if files:
            stream_kwargs["files"] = files
        if images:
            stream_kwargs["images"] = images
        if audio:
            stream_kwargs["audio"] = audio
        if videos:
            stream_kwargs["videos"] = videos
        member_sessions_on = _team_member_sessions_enabled()
        yielded_content = False
        suppressed_content: list[str] = []
        stream_iter = runtime.arun(prompt, **stream_kwargs)
        async for event in stream_iter:
            # Capture the agno run_id (first event wins) so a barge-in can
            # cooperatively cancel THIS run and let the runner persist it.
            if on_run_id is not None:
                rid = getattr(event, "run_id", None)
                if rid:
                    on_run_id(rid)
                    on_run_id = None
            # Token accounting. The streaming path used to record NOTHING, so a
            # streamed turn — which is every turn that matters — never reached
            # usage_log. Take the metrics off the PARENT run's completion event
            # only: a delegated member's run belongs to its own child session
            # and reports there. (src/models/stream_usage.py)
            if isinstance(event, run_completed_event_types):
                ev_sid = getattr(event, "session_id", None)
                if ev_sid is None or ev_sid == session_id:
                    inp, out = stream_usage.metrics_to_tokens(
                        getattr(event, "metrics", None)
                    )
                    stream_usage.record(input_tokens=inp, output_tokens=out)
            # Suppress a delegated member's OWN content + nested tool events
            # from the PARENT live stream when the member runs in its own child
            # session. That work belongs to the child session (navigable via the
            # delegation card) — streaming it into the parent floods the parent
            # transcript with the sub-agent's reasoning + tool calls (the
            # reported "mid messages"). A member event is identifiable by a
            # session_id that differs from this (parent) run; the team also
            # relays the member's text as ``IntermediateRunContentEvent``
            # (parent session_id) — drop that too. Legacy nested mode
            # (team-member-sessions off) keeps the old inline behavior so the
            # live view still matches its rehydration.
            if member_sessions_on:
                ev_sid = getattr(event, "session_id", None)
                is_member = (
                    isinstance(event, _TeamIntermediateRunContentEvent)
                    or (ev_sid is not None and session_id is not None and ev_sid != session_id)
                )
                if is_member:
                    # Stash the member's text — the silent-leader fallback below
                    # surfaces it ONLY if the leader emitted no synthesis of its
                    # own, so a delegate-and-stop turn still yields content
                    # rather than tripping run_stream's no-deltas fallback
                    # (which would re-run the turn and spawn a DUPLICATE child).
                    c = getattr(event, "content", None)
                    if c:
                        suppressed_content.append(c)
                    continue
            if isinstance(event, content_event_types):
                delta = event.content or ""
                if delta:
                    yielded_content = True
                    yield delta
            elif isinstance(event, tool_start_event_types):
                tool = getattr(event, "tool", None)
                if on_delegate is not None and tool is not None:
                    member_id = _extract_delegated_member_id(tool)
                    if member_id:
                        on_delegate(member_id)
                await _emit_status(tool, phase="started")
            elif isinstance(event, tool_complete_event_types):
                tool = getattr(event, "tool", None)
                await _emit_status(tool)
            elif isinstance(event, tool_error_event_types):
                tool = getattr(event, "tool", None)
                err_text = getattr(event, "error", None)
                await _emit_status(tool, error_text=err_text)

        # Silent-leader safety net: a turn that delegated but whose leader
        # emitted NO synthesis of its own would otherwise yield zero content
        # (member text suppressed above) → run_stream's no-deltas fallback
        # re-runs the whole turn via generate(), spawning a DUPLICATE child
        # session. Surface the member's text in that (rare) case so the turn
        # has content. The normal case (leader synthesises) never hits this.
        if member_sessions_on and not yielded_content and suppressed_content:
            yield "".join(suppressed_content)
    except Exception as e:  # noqa: BLE001
        elog(error_event, session_id=session_id, error=str(e))
        raise
    finally:
        # aclose() the underlying async generator so a mid-stream
        # consumer cancellation (WS close, downstream raise) doesn't
        # leak the cursor / leader SqliteDb transaction that holds
        # writer locks open.
        if stream_iter is not None:
            aclose = getattr(stream_iter, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass


# ════════════════════════════════════════════════════════════════════
#  TeamRouterProvider — per-entry-model Team builder
# ════════════════════════════════════════════════════════════════════


class TeamRouterProvider(BaseModel):
    """BaseModel facade over a runtime ``Team(mode=coordinate)`` per session.

    Construction is lazy: the first ``generate`` / ``stream`` for a
    session builds the team. Subsequent turns reuse the cached team.
    Changing ``providers_config`` (hot reload) invalidates the cache so
    the next turn picks up new members / role blurbs.

    The team includes:

    - **Leader**: the entry model for the session. ``team.model`` (the
      routing classifier) is the leader's own model.
    - **Members**: every OTHER enabled LLM model in the DB. Each member
      is a runtime ``Agent`` (api-based via ``NativeProvider``) whose
      ``role`` comes from the DB row's ``tier_hint``.

    When there's no other enabled LLM model (single-model deployment),
    the team is skipped and the leader runs as a single agent.
    """

    history_mode = "platform"

    def __init__(
        self,
        entry_runtime_id: str,
        providers_config: Any = None,
        *,
        history_runs: int = FULL_SESSION_HISTORY_RUNS,
        budget: Any = None,
    ):
        self._entry_runtime_id = entry_runtime_id
        self._providers_config = providers_config if providers_config is not None else []
        self._db: Any = None
        self._mcp_pool: Any = None
        self._fallback_config: Any = None
        self._history_runs = history_runs
        # Cost telemetry lives HERE, not in ModelDispatcher, because not every
        # run goes through the dispatcher: a model-PINNED run (an Events
        # delivery, a scheduled task with a `model`, a workflow ai-prompt
        # override) is handed this provider directly by
        # ``ModelDispatcher.build_override_model`` and never touches the
        # dispatcher's generate/stream. That is why usage_log held zero rows
        # for the entire Replio webhook lane while it burned 412M input tokens.
        self._budget = budget
        self._session_runtime: dict[str, Any] = {}  # session_id → Team or fallback
        # session_id → stable hash of the system prompt the cached runtime
        # was built with; used to invalidate when the system prompt
        # changes (e.g. a workflow ai-prompt block with different system).
        self._session_system_key: dict[str, str] = {}
        self._session_handles: dict[str, str] = {}
        # session_id → {member_id (dash form) → runtime_id (canonical
        # colon form)}. Populated at team build time so the per-turn
        # ``delegate_task_to_member`` tool call can be translated back to
        # a runtime_id for the UI badge.
        self._session_member_map: dict[str, dict[str, str]] = {}
        # session_id → runtime_id that handled the most recent turn.
        # Empty until the first delegation fires; ``effective_model_id``
        # prefers this over the entry pick so the chat badge shows the
        # specialist that actually wrote the response, not the leader.
        self._last_delegation_by_session: dict[str, str] = {}
        # In-flight agno run_id per session is tracked by ``BaseModel`` (see
        # ``_track_run_id`` / ``_coop_cancel``); the TEAM path captures it via
        # the ``on_run_id`` callback wired into ``_arun_runtime_stream``.

    @property
    def model(self) -> str | None:
        return self._entry_runtime_id

    # ── wiring ────────────────────────────────────────────────────────

    def set_db(self, db: Any) -> None:
        if db is self._db:
            return
        self._db = db
        self._invalidate_session_cache()

    def set_mcp_pool(self, pool: Any) -> None:
        if pool is self._mcp_pool:
            return
        self._mcp_pool = pool
        self._invalidate_session_cache()

    def set_fallback_config(self, fallback_config: Any) -> None:
        if fallback_config is self._fallback_config:
            return
        self._fallback_config = fallback_config
        self._invalidate_session_cache()

    def set_session_handle(self, session_id: str, handle: str | None) -> None:
        # NOTE: the runtime ``sessions``-row owner is the single stable
        # ``RUNTIME_SESSION_USER_ID`` sentinel (see ``generate``/``stream``),
        # NOT this handle. Binding it here used to drive that column, which
        # made the owner flip mid-session (the gateway wrote the handle, the
        # runtime fell back to "openagent" before this propagated) and the
        # agent forgot the conversation. The map is retained for callers
        # that want the authenticated handle for other purposes; it no
        # longer gates history read/write.
        if not session_id:
            return
        if handle and handle.strip():
            self._session_handles[session_id] = handle.strip()
        else:
            self._session_handles.pop(session_id, None)

    def rebuild_routing(self, providers_config: Any = None) -> None:
        """Drop cached teams so the next turn picks up the fresh catalog."""
        if providers_config is not None and providers_config is self._providers_config:
            return
        if providers_config is not None:
            self._providers_config = providers_config
        self._invalidate_session_cache()

    def _invalidate_session_cache(self) -> None:
        """Clear all per-session caches so the next turn rebuilds from
        the current db / pool / providers_config.
        """
        self._session_runtime.clear()
        self._session_system_key.clear()
        self._session_member_map.clear()
        self._last_delegation_by_session.clear()
        self._run_ids.clear()

    # ── catalog → runtime objects ─────────────────────────────────────

    def _enabled_llm_models(self) -> list[CatalogModel]:
        """Return every enabled api-based LLM model.

        Each joins the Team as a runtime ``Agent`` (via ``NativeProvider``)
        so it passes the runtime's ``isinstance(member, Agent)`` check and
        can serve as a Team member or leader.
        """
        out: list[CatalogModel] = []
        for entry in iter_configured_models(self._providers_config):
            if entry.disabled:
                continue
            if framework_of(entry.runtime_id) != FRAMEWORK_API_BASED:
                continue
            out.append(entry)
        return out

    def _build_api_agent_for(
        self,
        entry: CatalogModel,
        *,
        name: str,
        role: str | None,
        system: str | None = None,
        db: Any = None,
    ) -> Any:
        """Construct a single api-based runtime ``Agent`` for ``entry``.

        Uses the existing ``NativeProvider`` machinery to resolve model
        class + API key + base URL from the providers_config, then
        builds an ``Agent`` configured for membership in a Team — with
        the tool-search toolkit attached so members can actually invoke
        MCP tools after the leader delegates to them.

        ``system`` is the OpenAgent framework prompt + user persona
        prompt (composed upstream by ``Agent._combined_system_prompt``).
        Vision §15 requires that this prompt be injected into every
        agent the user can reach — leader and every team member alike —
        so when the Team coordinator delegates to a member, the member
        still runs with the full OpenAgent identity (vault, MCPs,
        framework guidelines) and the user's persona.
        """
        from src.models.native_provider import NativeProvider

        provider = NativeProvider(
            model=entry.runtime_id,
            providers_config=self._providers_config,
            db_path=_runtime_db_path(self._db),
        )
        provider.set_fallback_config(self._fallback_config)
        member_toolkits: list[Any] = []
        if self._mcp_pool is not None:
            member_toolkits = list(self._mcp_pool.runtime_toolkits_tool_search_only())
            provider.set_mcp_toolkits(member_toolkits)
        runtime_model = provider.build_runtime_model()
        from src.core._runner.agent import Agent as RuntimeAgent

        return RuntimeAgent(
            name=name,
            model=runtime_model,
            tools=member_toolkits or None,
            role=role,
            system_message=system or None,
            fallback_config=self._fallback_config,
            markdown=False,
            # When a db is supplied (members in team-member-session mode), the
            # member persists its run to its OWN child session row — so a
            # delegation surfaces as a navigable child session + card. The
            # leader is built without a db (its turn persists via the Team's
            # own db on the parent session). ``add_history_to_context`` stays
            # False so a fresh per-delegation child id never loads stale state.
            db=db,
        )

    def _build_agent_for(
        self,
        entry: CatalogModel,
        *,
        name: str,
        role: str | None,
        system: str | None = None,
        db: Any = None,
    ) -> Any:
        """Build a runtime ``Agent`` for ``entry``.

        The per-framework member-build seam — currently only the
        ``api-based`` framework ships, so this delegates straight to
        ``_build_api_agent_for``.
        """
        return self._build_api_agent_for(
            entry, name=name, role=role, system=system, db=db,
        )

    def _ensure_runtime(self, session_id: str, system: str | None) -> Any:
        """Return the cached Team / single-agent for ``session_id``, building
        it when absent.

        Skips the team build when there's only one enabled LLM model
        (single-model deployment) — the lone leader runs as a single agent.

        Cache key is ``(session_id, system_hash)`` so changing the
        framework prompt mid-session forces a fresh build instead of
        silently reusing the cached Team's baked-in ``instructions``.
        """
        sys_key = _system_cache_key(system)
        cached = self._session_runtime.get(session_id)
        if cached is not None and self._session_system_key.get(session_id) == sys_key:
            return cached

        catalog = self._enabled_llm_models()
        entry_runtime = self._entry_runtime_id or (catalog[0].runtime_id if catalog else None)
        if not entry_runtime:
            raise RuntimeError(
                "TeamRouterProvider has no enabled LLM models to dispatch to."
            )
        entry = next(
            (e for e in catalog if e.runtime_id == entry_runtime),
            catalog[0] if catalog else None,
        )
        if entry is None:
            raise RuntimeError(
                f"TeamRouterProvider entry {entry_runtime!r} not in enabled catalog."
            )

        members_catalog = [e for e in catalog if e.runtime_id != entry.runtime_id]
        entry_framework = framework_of(entry.runtime_id)

        # No other LLM model to delegate to — a Team of one is degenerate
        # (nothing to route TO). Skip Team and dispatch the lone leader as
        # a single agent via NativeProvider.
        if not members_catalog:
            from src.models.native_provider import NativeProvider

            provider = NativeProvider(
                model=entry.runtime_id,
                providers_config=self._providers_config,
                db_path=_runtime_db_path(self._db),
            )
            provider.set_fallback_config(self._fallback_config)
            if self._mcp_pool is not None:
                provider.set_mcp_toolkits(self._mcp_pool.runtime_toolkits_tool_search_only())
            if self._db is not None:
                provider.set_db(self._db)
            self._session_runtime[session_id] = provider
            self._session_system_key[session_id] = sys_key
            elog(
                "team_router.single_agent",
                session_id=session_id,
                entry=entry.runtime_id,
                reason="no_other_llm_models",
            )
            return provider

        # Resolve ``team.model`` — the leader's own model (the Team always
        # invokes team.model for routing decisions).
        leader_agent = self._build_api_agent_for(
            entry, name="leader", role=None, system=system,
        )
        team_model = leader_agent.model

        # Multi-model: build Team(mode=coordinate). See the ``mode=``
        # kwarg below for why coordinate (not route).
        from src.memory.store.sqlite import SqliteDb
        from src.core._runner.team import Team, TeamMode

        # The shared runtime DB. Built BEFORE the members so it can be handed
        # to each one: in team-member-session mode a delegated member persists
        # its run to its OWN child session row (a real, navigable session +
        # sidebar card) rather than nested in the parent. The leader stays
        # db-less — its turn is persisted via the Team's own db below.
        db_path = _runtime_db_path(self._db)
        team_db = SqliteDb(db_file=db_path) if db_path else None
        member_db = team_db if _team_member_sessions_enabled() else None

        # Vision §15: every member must run with the framework prompt +
        # user persona injected as its own system_message — the Team's
        # ``instructions=`` only feed the coordinator (team.model), they
        # are NOT inherited by members at delegation time. Each member
        # also gets a short Role suffix so the delegated specialist
        # stays in its lane; the role blurb itself is *additionally*
        # passed via the runtime's ``role=`` so the coordinator's
        # ``<team_members>`` block keeps surfacing it for routing.
        members = [
            self._build_agent_for(
                e,
                name=_member_identifier(e.runtime_id),
                role=_build_role_blurb(e),
                system=_compose_member_system(system, _build_role_blurb(e)),
                db=member_db,
            )
            for e in members_catalog
        ]

        # Stamp each member with child-session linkage so that when a
        # delegation runs the member in its OWN session row (see
        # ``delegate_task_to_member``), ``read_or_create_session`` carries
        # this metadata onto the new row: the app then shows the member run
        # as a delegation card under the team session. ``client_id`` (owner)
        # is intentionally omitted — this sync build site can't resolve it;
        # ``list_all_sessions`` instead inherits visibility from the parent
        # team session (so the child still lands in the owner's list).
        if _team_member_sessions_enabled():
            for member, e in zip(members, members_catalog):
                try:
                    member.metadata = {
                        "parent_session_id": session_id,
                        "origin": "delegation",
                        "kind": e.runtime_id,
                        "title": f"Sub-agent · {e.runtime_id}",
                    }
                except Exception:  # noqa: BLE001 — never block team build on this
                    pass

        # Record member_id → runtime_id so the per-turn delegation tool
        # call can be reflected back to the chat UI as the specialist's
        # canonical runtime_id. ``leader`` maps to the entry runtime so
        # a self-handled turn still surfaces the right badge.
        self._session_member_map[session_id] = {
            _member_identifier(entry.runtime_id): entry.runtime_id,
            "leader": entry.runtime_id,
            **{_member_identifier(e.runtime_id): e.runtime_id for e in members_catalog},
        }
        # ``team_db`` / ``db_path`` were built above (before the members).
        # Pass our OpenAgent system prompt as ``instructions`` — NOT as
        # ``system_message``. The runtime's get_system_message returns early
        # when system_message is set, skipping the <team_members> block
        # AND the mode-specific "how to delegate" instructions.
        # ``coordinate`` mode lets the leader fire multiple delegations
        # per turn (gathered by the runtime via asyncio.gather).
        # Tool-search at the team level so the leader can invoke MCPs
        # directly via ``tool_search_call_tool`` without delegating;
        # each member also has its own toolkit copy.
        team_tools = (
            list(self._mcp_pool.runtime_toolkits_tool_search_only())
            if self._mcp_pool is not None else None
        )
        team = Team(
            mode=TeamMode.coordinate,
            model=team_model,
            members=[leader_agent, *members],
            db=team_db,
            tools=team_tools,
            instructions=[system] if system else None,
            fallback_config=self._fallback_config,
            add_member_tools_to_context=True,
            add_history_to_context=True,
            num_history_runs=self._history_runs,
            enable_session_summaries=True,
            markdown=False,
        )
        self._session_runtime[session_id] = team
        self._session_system_key[session_id] = sys_key
        elog(
            "team_router.built",
            session_id=session_id,
            mode="coordinate",
            leader=entry.runtime_id,
            leader_framework=entry_framework,
            routing_model_id=getattr(team_model, "id", None),
            routing_model_provider=getattr(team_model, "provider", None),
            members=[e.runtime_id for e in members_catalog],
        )
        return team

    # ── lifecycle ────────────────────────────────────────────────────

    async def close_session(self, session_id: str) -> None:
        self._drop_session_state(session_id)
        # The runtime's Team / Agent don't expose a per-session close in
        # this adapter shape; the SqliteDb persists the session for next
        # boot. Nothing else to tear down.

    async def forget_session(self, session_id: str) -> None:
        self._drop_session_state(session_id)
        # Erase the persisted session row so the next turn starts fresh.
        db_path = _runtime_db_path(self._db)
        if not db_path:
            return
        try:
            from src.memory.store.sqlite import SqliteDb

            SqliteDb(db_file=db_path).delete_session(session_id=session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("TeamRouterProvider.forget_session %s: %s", session_id, e)

    def _drop_session_state(self, session_id: str) -> None:
        """Remove every per-session entry for ``session_id``. Without
        this, ``_session_member_map`` / ``_last_delegation_by_session``
        keep stale references after close/forget and slowly leak memory.
        """
        if not session_id:
            return
        self._session_runtime.pop(session_id, None)
        self._session_system_key.pop(session_id, None)
        self._session_member_map.pop(session_id, None)
        self._last_delegation_by_session.pop(session_id, None)
        self._clear_run_id(session_id)

    async def request_cancel(self, session_id: str) -> bool:
        # Cooperative cancel so the runner persists the interrupted run. The
        # TEAM path streams through ``_arun_runtime_stream`` (run_id captured
        # here); the single-model path streams through the NativeProvider
        # runtime (run_id captured there). Try the team-captured id first,
        # else forward to the runtime.
        sid = session_id or "default"
        if await self._coop_cancel(session_id):
            return True
        runtime = self._session_runtime.get(sid)
        fn = getattr(runtime, "request_cancel", None) if runtime else None
        if callable(fn):
            try:
                return bool(await fn(session_id))
            except Exception as e:  # noqa: BLE001
                logger.debug("TeamRouterProvider.request_cancel: %s", e)
        return False

    async def cleanup_idle(self) -> None:
        await self._fan_out_runtimes("cleanup_idle")

    async def shutdown(self) -> None:
        await self._fan_out_runtimes("shutdown")
        self._invalidate_session_cache()

    async def _fan_out_runtimes(self, method_name: str) -> None:
        import asyncio as _asyncio

        coros = []
        for runtime in list(self._session_runtime.values()):
            fn = getattr(runtime, method_name, None)
            if callable(fn):
                coros.append(fn())
        if not coros:
            return
        results = await _asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.debug("TeamRouterProvider.%s: %s", method_name, result)

    def known_session_ids(self) -> list[str]:
        return sorted(self._session_runtime.keys())

    def last_delegation_for_session(self, session_id: str | None) -> str | None:
        """Return the runtime_id of the specialist that handled the most
        recent turn for ``session_id``, or ``None`` if no delegation has
        fired yet. Used by ``ModelDispatcher.effective_model_id`` so the
        chat-UI badge follows the actual responder instead of staying
        pinned to the team leader.
        """
        if not session_id:
            return None
        return self._last_delegation_by_session.get(session_id)

    def _record_delegation(self, session_id: str, member_id: str) -> None:
        """Look up the runtime_id for ``member_id`` (runtime-derived dash form)
        in the per-session map built at team-build time, and stash it as
        the most recent delegation for badge lookup. Skips when the same
        delegation has already been recorded for this turn so the event
        log doesn't fill up on multi-step delegations to the same member.
        """
        member_map = self._session_member_map.get(session_id) or {}
        runtime_id = member_map.get(member_id)
        if not runtime_id:
            return
        if self._last_delegation_by_session.get(session_id) == runtime_id:
            return
        self._last_delegation_by_session[session_id] = runtime_id
        elog(
            "team_router.delegate",
            session_id=session_id,
            member_id=member_id,
            runtime_id=runtime_id,
        )

    # ── cost telemetry ────────────────────────────────────────────────

    async def record_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        session_id: str | None,
    ) -> None:
        """Write one call into ``usage_log`` — the canonical cost ledger.

        Every path that runs a model reaches this provider: the dispatcher's
        routed turns, AND the model-pinned ones that skip the dispatcher
        entirely. Accounting therefore belongs here, or it does not cover the
        traffic that turned out to be the expensive kind.
        """
        if not self._budget:
            return
        if not input_tokens and not output_tokens:
            return
        runtime_id = self._entry_runtime_id
        cost = BudgetTracker.compute_cost(runtime_id, input_tokens, output_tokens)
        try:
            await self._budget.record(
                model=runtime_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                session_id=session_id,
            )
            elog(
                "router.cost_recorded",
                session_id=session_id,
                model=runtime_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        except Exception as e:  # noqa: BLE001
            elog(
                "router.cost_record_error",
                session_id=session_id,
                model=runtime_id,
                error=str(e),
            )

    # ── turn ──────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
        files: list[Any] | None = None,
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
    ) -> ModelResponse:
        sid = session_id or "default"
        runtime = self._ensure_runtime(sid, system)
        # Plain-NativeProvider single-model path: forward the full args.
        if isinstance(runtime, BaseModel):
            resp = await runtime.generate(
                messages=messages,
                system=system,
                tools=tools,
                on_status=on_status,
                session_id=sid,
                model_override=model_override,
                files=files, images=images, audio=audio, videos=videos,
            )
            await self.record_cost(
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                session_id=session_id,
            )
            return resp

        # Runtime Team path. Translate to the Team's run signature.
        # Clear the previous turn's delegation memo so this turn's badge
        # falls back to the leader when no delegate_task_to_member fires —
        # otherwise the model attribution stays pinned to whichever
        # specialist handled the LAST delegation indefinitely.
        self._last_delegation_by_session.pop(sid, None)
        # Stable sessions-row owner for every surface (see catalog.py);
        # tenancy is carried by ``session_id``, not this column, so the
        # runtime can always read/write the row regardless of which
        # handle (if any) the gateway bound to the session.
        user_id = RUNTIME_SESSION_USER_ID
        resp = await _arun_runtime_collect(
            runtime,
            prompt=_last_user_prompt(messages),
            session_id=sid,
            user_id=user_id,
            error_event="team_router.generate_error",
            entry_runtime_id=self._entry_runtime_id,
            on_delegate=lambda mid: self._record_delegation(sid, mid),
            files=files, images=images, audio=audio, videos=videos,
        )
        await self.record_cost(
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            session_id=session_id,
        )
        return resp

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
        files: list[Any] | None = None,
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        sid = session_id or "default"
        runtime = self._ensure_runtime(sid, system)
        # Collect the streamed run's tokens. Whatever ends the stream —
        # completion, an error, a barge-in cancel — what was already spent
        # gets billed: an abandoned turn still cost its input tokens.
        sink, sink_token = stream_usage.open_sink()
        try:
            if isinstance(runtime, BaseModel):
                async for delta in runtime.stream(
                    messages,
                    system=system,
                    tools=tools,
                    on_status=on_status,
                    session_id=sid,
                    model_override=model_override,
                    files=files, images=images, audio=audio, videos=videos,
                ):
                    yield delta
                return

            # Runtime Team streaming. Clear the previous turn's delegation memo
            # so a non-delegating turn falls back to the leader's badge
            # instead of staying pinned to the last specialist.
            self._last_delegation_by_session.pop(sid, None)
            # Stable sessions-row owner for every surface (see catalog.py);
            # tenancy is carried by ``session_id``, not this column, so the
            # runtime can always read/write the row regardless of which
            # handle (if any) the gateway bound to the session.
            user_id = RUNTIME_SESSION_USER_ID
            try:
                async for delta in _arun_runtime_stream(
                    runtime,
                    prompt=_last_user_prompt(messages),
                    session_id=sid,
                    user_id=user_id,
                    on_status=on_status,
                    error_event="team_router.stream_error",
                    on_delegate=lambda mid: self._record_delegation(sid, mid),
                    on_run_id=lambda rid: self._track_run_id(sid, rid),
                    files=files, images=images, audio=audio, videos=videos,
                ):
                    yield delta
            finally:
                self._clear_run_id(sid)
        finally:
            stream_usage.close_sink(sink_token)
            await self.record_cost(
                input_tokens=sink.get("input_tokens", 0),
                output_tokens=sink.get("output_tokens", 0),
                session_id=session_id,
            )


# ════════════════════════════════════════════════════════════════════
#  ModelDispatcher — entry resolution + lifecycle + budget
# ════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RoutingDecision:
    """Per-call routing record."""
    reason: str
    primary_model: str
    candidates: list[str]


class ModelDispatcher(BaseModel):
    """Single dispatcher routing every turn through a TeamRouterProvider.

    Resolves the entry runtime_id for a session (pin → first-enabled),
    holds one cached :class:`TeamRouterProvider` per entry runtime_id,
    forwards lifecycle hooks (cleanup_idle, shutdown, close/forget
    session) and records cost telemetry against the api-based path.

    Entry-model resolution (no classifier LLM call):

      1. **Per-session pin** (``models.pin_session``) — if the user or
         the agent itself has chosen a specific runtime_id for this
         session, dispatch directly.
      2. **First enabled** — fresh sessions with no pin pick the first
         enabled model in catalog order.

    History is unified in the runtime's sessions SqliteDb across
    every framework, so there's no longer a need to lock a session to
    one framework once it's been served by that side.

    ``history_mode`` is intentionally ``None`` — the gateway's
    ``SessionManager.bind_history_mode`` bails on falsy modes; the
    dispatcher handles binding internally.
    """

    history_mode = None

    def __init__(self, providers_config: Any = None):
        # Providers_config is a flat list of provider entries (v0.12+).
        # Accept both shapes (list or legacy dict) so early-boot / tests
        # that still seed with a dict keep working.
        if providers_config is None:
            providers_config = []
        self._providers_config = providers_config

        self._budget: BudgetTracker | None = None
        self._db: Any = None
        self._mcp_pool: Any = None
        self._fallback_config: Any = None

        # Per-entry-model TeamRouterProvider. The provider's session
        # cache picks up where this one leaves off, so reusing a single
        # provider instance per entry-model lets us share the Team
        # construction cache across sessions that share an entry.
        self._team_providers: dict[str, BaseModel] = {}

        # Per-session memo of the last entry model we routed to — used
        # by ``effective_model_id`` so the chat UI's model badge always
        # has something to render.
        self._last_pick_by_session: dict[str, str] = {}

    def rebuild_routing(self, providers_config: Any = None) -> None:
        """Called by the hot-reload loop when the ``models`` DB table changes.

        Forwards the fresh providers_config to every cached
        TeamRouterProvider so they rebuild their member lists / role
        blurbs on the next turn.
        """
        if providers_config is not None:
            self._providers_config = providers_config
        for provider in self._team_providers.values():
            fn = getattr(provider, "rebuild_routing", None)
            if callable(fn):
                fn(providers_config=self._providers_config)
        elog("router.rebuilt")

    # ── runtime wiring ───────────────────────────────────────────────

    def set_db(self, db: Any) -> None:
        self._db = db
        self._budget = BudgetTracker(db, 0.0)
        for provider in self._team_providers.values():
            wire_model_runtime(provider, db=db)
            # A provider built before the db arrived would otherwise bill into
            # the void for the rest of the process's life.
            provider._budget = self._budget

    def set_mcp_pool(self, pool: Any) -> None:
        self._mcp_pool = pool
        for provider in self._team_providers.values():
            wire_model_runtime(provider, mcp_pool=pool)

    def set_fallback_config(self, fallback_config: Any) -> None:
        self._fallback_config = fallback_config
        for provider in self._team_providers.values():
            wire_model_runtime(provider, fallback_config=fallback_config)

    def set_session_handle(self, session_id: str, handle: str | None) -> None:
        """Bind a user handle to ``session_id`` across every cached
        TeamRouterProvider so subsequent runtime calls pass it as
        ``user_id``. Gateway calls this when accepting a WS frame.
        """
        for provider in self._team_providers.values():
            fn = getattr(provider, "set_session_handle", None)
            if callable(fn):
                fn(session_id, handle)

    async def cleanup_idle(self) -> None:
        await self._fan_out_async("cleanup_idle")

    async def shutdown(self) -> None:
        await self._fan_out_async("shutdown")

    async def close_session(self, session_id: str) -> None:
        if not session_id:
            return
        self._last_pick_by_session.pop(session_id, None)
        await self._fan_out_async("close_session", session_id)

    async def forget_session(self, session_id: str) -> None:
        if not session_id:
            return
        self._last_pick_by_session.pop(session_id, None)
        if self._db is not None:
            # delete_sdk_session is kept callable for back-compat with
            # gateway/clear code; the runtime-side persistence no longer
            # uses it, but legacy installs may still have rows.
            try:
                await self._db.delete_sdk_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("delete_sdk_session %s: %s", session_id, e)
        await self._fan_out_async("forget_session", session_id)

    async def request_cancel(self, session_id: str) -> bool:
        """Cooperatively cancel the session's in-flight runtime run so the
        runner persists it (and ``upsert_run`` backfills the interrupted turn
        into history) rather than leaving a poisoned RUNNING shell. Targets
        only the provider that handled the session. Returns True if a run was
        registered for cancellation.
        """
        if not session_id:
            return False
        provider = self._provider_for_session(session_id)
        fn = getattr(provider, "request_cancel", None) if provider else None
        if not callable(fn):
            return False
        try:
            return bool(await fn(session_id))
        except Exception as e:  # noqa: BLE001
            logger.debug("team provider request_cancel: %s", e)
            return False

    def _provider_for_session(self, session_id: str) -> BaseModel | None:
        """Return the TeamRouterProvider that handled ``session_id``,
        falling back to ``None`` when the session hasn't been routed
        yet (or the provider has since been evicted).
        """
        runtime_id = self._last_pick_by_session.get(session_id)
        if not runtime_id:
            return None
        return self._team_providers.get(runtime_id)

    async def _fan_out_async(self, method_name: str, *args: Any) -> None:
        """Invoke ``method_name`` on every cached TeamRouterProvider
        concurrently, swallowing per-provider errors so one bad provider
        can't block shutdown / cleanup of the rest.
        """
        import asyncio as _asyncio

        coros = []
        for provider in self._team_providers.values():
            fn = getattr(provider, method_name, None)
            if callable(fn):
                coros.append(fn(*args))
        if not coros:
            return
        results = await _asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.debug("dispatcher.%s: %s", method_name, result)

    def known_session_ids(self) -> list[str]:
        seen: set[str] = set()
        for provider in self._team_providers.values():
            fn = getattr(provider, "known_session_ids", None)
            if callable(fn):
                try:
                    seen.update(fn())
                except Exception as e:  # noqa: BLE001
                    logger.debug("known_session_ids team: %s", e)
        return sorted(seen)

    def effective_model_id(self, session_id: str | None = None) -> str | None:
        """Return the runtime_id of the model that actually handled the
        most recent turn for ``session_id``.

        Resolution order:
        1. Last delegation target recorded by any TeamRouterProvider for
           this session — i.e. the specialist the leader routed to.
        2. Last entry pick (the session's leader) — for turns the leader
           answered directly without delegating.

        Without step 1 the chat UI badge stays pinned to the leader
        (e.g. ``deepseek-v4-flash``) even when a heavyweight specialist
        actually wrote the response, masking the team-as-router from
        the user.
        """
        if session_id:
            for tp in self._team_providers.values():
                fn = getattr(tp, "last_delegation_for_session", None)
                if callable(fn):
                    delegated = fn(session_id)
                    if delegated:
                        return delegated
        key = session_id or "__default__"
        return self._last_pick_by_session.get(key)

    # ── dispatch plumbing ───────────────────────────────────────────

    def _enabled_catalog(self) -> list[CatalogModel]:
        out: list[CatalogModel] = []
        for entry in iter_configured_models(self._providers_config):
            if entry.disabled:
                continue
            out.append(entry)
        return out

    def build_override_model(self, runtime_id: str) -> BaseModel:
        """Construct a BaseModel bound to a specific runtime_id.

        Used by the workflow engine's ``ai-prompt`` block for explicit
        model overrides. Routes through the same dispatch as the
        regular path: every runtime gets a
        TeamRouterProvider keyed on that runtime so the team-as-router
        delegation still applies inside a workflow's ai-prompt block.
        """
        if not runtime_id:
            raise ValueError("runtime_id is required")
        known = {m.runtime_id for m in self._enabled_catalog()}
        if runtime_id not in known:
            raise ValueError(
                f"runtime_id {runtime_id!r} is not an enabled model. "
                f"Known ids: {sorted(known)}"
            )
        return self._get_team_provider(runtime_id)

    def _get_team_provider(self, entry_runtime_id: str) -> BaseModel:
        """Per-entry-model TeamRouterProvider. Cached so subsequent
        sessions on the same entry model reuse the same Team builder.
        """
        if entry_runtime_id not in self._team_providers:
            provider = TeamRouterProvider(
                entry_runtime_id=entry_runtime_id,
                providers_config=self._providers_config,
                # The provider bills, not the dispatcher: a model-pinned run
                # (Events delivery, scheduled task, workflow override) gets this
                # object straight from build_override_model and never touches
                # the dispatcher at all.
                budget=self._budget,
            )
            wire_model_runtime(
                provider,
                db=self._db,
                mcp_pool=self._mcp_pool,
                fallback_config=self._fallback_config,
            )
            self._team_providers[entry_runtime_id] = provider
        return self._team_providers[entry_runtime_id]

    async def _resolve_entry_model(
        self,
        session_id: str | None,
    ) -> RoutingDecision:
        """Pick the entry runtime_id for this session.

        Order: per-session pin → ``is_classifier``-flagged model → first
        enabled. The ``is_classifier`` flag is the user's persistent
        "default team leader" hint (set in the model_manager UI); the
        first-enabled fallback only kicks in when no model is flagged.
        """
        if session_id and self._db is not None:
            try:
                pinned_id = await self._db.get_session_pin(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("get_session_pin failed for %s: %s", session_id, e)
                pinned_id = None
            if pinned_id:
                enabled_ids = {entry.runtime_id for entry in self._enabled_catalog()}
                if pinned_id not in enabled_ids:
                    elog(
                        "router.pin_auto_heal",
                        session_id=session_id,
                        pinned_model=pinned_id,
                        reason="model_not_enabled",
                    )
                    try:
                        await self._db.unpin_session_model(session_id)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("unpin_session_model failed for %s: %s", session_id, e)
                    pinned_id = None
            if pinned_id:
                return RoutingDecision(
                    reason="session_pin",
                    primary_model=pinned_id,
                    candidates=[pinned_id],
                )

        catalog = self._enabled_catalog()
        if not catalog:
            return RoutingDecision(
                reason="no_enabled_model",
                primary_model="",
                candidates=[],
            )

        flagged = next((entry for entry in catalog if entry.is_classifier), None)
        if flagged is not None:
            return RoutingDecision(
                reason="router_flag",
                primary_model=flagged.runtime_id,
                candidates=[flagged.runtime_id],
            )

        primary = catalog[0].runtime_id
        return RoutingDecision(
            reason="first_enabled",
            primary_model=primary,
            candidates=[primary],
        )

    def _remember_pick(self, session_id: str | None, runtime_id: str) -> None:
        key = session_id or "__default__"
        self._last_pick_by_session[key] = runtime_id

    # ── turn ──────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        files: list[Any] | None = None,
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
    ) -> ModelResponse:
        decision = await self._resolve_entry_model(session_id)
        if not decision.primary_model:
            elog(
                "router.error",
                session_id=session_id,
                reason=decision.reason,
            )
            return ModelResponse(content="No model is currently enabled.", stop_reason="error")

        runtime_id = decision.primary_model
        self._remember_pick(session_id, runtime_id)
        elog(
            "router.route",
            session_id=session_id,
            reason=decision.reason,
            model=runtime_id,
        )

        provider = self._get_team_provider(runtime_id)
        resp = await provider.generate(
            messages, system=system, tools=tools,
            on_status=on_status, session_id=session_id,
            files=files, images=images, audio=audio, videos=videos,
        )

        # Cost telemetry is recorded by the provider (TeamRouterProvider
        # .record_cost) — it is the one object EVERY path reaches, including
        # the model-pinned runs that bypass this dispatcher entirely.

        if not resp.model:
            resp.model = runtime_id
        return resp

    async def run_delegated(
        self,
        *,
        model_id: str,
        task: str,
        parent_session_id: str | None,
        pool: Any,
        db: Any,
    ) -> str:
        """SUPERSEDED — delegation now spawns a full child session.

        This method used to run the sub-agent as a bare ``provider.generate``
        with ``system=None`` (no framework/persona prompt — a vision §15
        violation) on a synthetic, non-persisted ``{parent}::sub::{model}``
        id. The delegation MCP (``mcp/servers/delegation/handlers.py``) now
        routes ``delegate_task`` through ``core.child_session.run_child_session``
        instead, so the sub-agent runs through the ordinary ``Agent.run``
        path: full two-layer prompt, full tools, and a real durable session
        row the user can navigate to and continue.

        The method is kept only so the ``hasattr(model, "run_delegated")``
        dispatcher-selection guard (agent.py / workflow/executor.py) is
        undisturbed; it should never be invoked. If it is, fail loudly
        rather than silently re-introduce the §15 violation.
        """
        raise NotImplementedError(
            "ModelDispatcher.run_delegated is superseded by "
            "core.child_session.run_child_session; delegation now spawns a "
            "full child session via the Agent, not a system-less provider call."
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        files: list[Any] | None = None,
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        decision = await self._resolve_entry_model(session_id)
        if not decision.primary_model:
            elog(
                "router.stream.error",
                session_id=session_id,
                reason=decision.reason,
            )
            yield "No model is currently enabled."
            return

        runtime_id = decision.primary_model
        self._remember_pick(session_id, runtime_id)
        elog(
            "router.stream.route",
            session_id=session_id,
            reason=decision.reason,
            model=runtime_id,
        )

        # The provider opens its own usage sink and bills the turn — see
        # TeamRouterProvider.stream / .record_cost.
        provider = self._get_team_provider(runtime_id)
        async for chunk in provider.stream(
            messages, system=system, tools=tools,
            on_status=on_status, session_id=session_id,
            files=files, images=images, audio=audio, videos=videos,
        ):
            yield chunk


# Transitional alias so existing imports keep working until the
# follow-up pass cleans up callers (tests + src). Tests reference
# ``SmartRouter`` directly; the runtime entry point in
# ``runtime.create_model_from_spec`` uses ``ModelDispatcher`` after
# the rename.
SmartRouter = ModelDispatcher
