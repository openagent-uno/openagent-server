"""Shared infrastructure for the subscription-CLI backed agents.

This module hosts the bits ``ClaudeBackedAgent`` (drives
``claude_agent_sdk``) and ``CodexBackedAgent`` (drives ``openai_codex``)
share — roughly 70% of their scaffolding:

Agent-side base (consumed by ``Agent`` subclasses in ``claude_agent`` /
``codex_agent``):

- ``_NullModel`` placeholder Agent.__init__ accepts when the subclass
  drives the SDK directly and never reaches ``self.model.invoke``.
- ``_stringify_input`` message flattener — Agno's polymorphic ``arun``
  input is reduced to the plain string each SDK's per-turn entry
  point accepts.
- ``BaseSubscriptionBackedAgent`` — Agent subclass providing the
  scaffolding (session-id round-trip via ``session_data``, run dispatch
  by stream flag, sync ``run`` wrapper, metrics helper). Subclasses
  override only the SDK-specific bits (lazy import, options build,
  per-SDK message-walk loop).

CLI-side helpers (consumed by the registry providers):

- ``_last_user_prompt(messages)`` — message-list to plain string.
- ``_known_sdk_session_ids_by_key(db, json_key, …)`` — SQL probe for
  ``agno_sessions`` rows that have an SDK session id stamped, used by
  the gateway's ``/clear`` legacy-row fallback.
- ``_make_sqlite_db(db)`` — build an Agno ``SqliteDb`` pointing at the
  same file as the OpenAgent ``MemoryDB``.

This module does NOT import either SDK — the lazy-import sites stay
per-file so a slim install missing one SDK can still import the other.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional
from uuid import uuid4

from agno.agent import Agent
from agno.metrics import ModelMetrics, RunMetrics
from agno.models.base import Model

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Agent-side: _NullModel + BaseSubscriptionBackedAgent
# ════════════════════════════════════════════════════════════════════


@dataclass
class _NullModel(Model):
    """Stand-in for the ``Agent.model`` slot.

    ``Agent.__init__`` typing assumes a real ``Model``; the standard
    arun / run code paths invoke ``self.model.invoke`` /
    ``self.model.ainvoke``. Subscription-backed subclasses override
    ``arun`` BEFORE that machinery runs, so this class's methods are
    never reached — they raise loudly if a code path slips past the
    override.
    """

    id: str = "subscription-backed-null-model"
    provider: Optional[str] = "subscription-cli"

    def invoke(self, *args, **kwargs):
        raise NotImplementedError(
            f"_NullModel({self.id}).invoke called — subclass arun should "
            "have short-circuited before reaching self.model.invoke()."
        )

    async def ainvoke(self, *args, **kwargs):
        raise NotImplementedError(
            f"_NullModel({self.id}).ainvoke called — subclass arun should "
            "have short-circuited before reaching self.model.ainvoke()."
        )

    def invoke_stream(self, *args, **kwargs):
        raise NotImplementedError(
            f"_NullModel({self.id}).invoke_stream called — subclass owns streaming."
        )

    def ainvoke_stream(self, *args, **kwargs):
        raise NotImplementedError(
            f"_NullModel({self.id}).ainvoke_stream called — subclass owns streaming."
        )

    def _parse_provider_response(self, response: Any, **kwargs):
        raise NotImplementedError(
            f"_NullModel({self.id})._parse_provider_response called — subclass owns parsing."
        )

    def _parse_provider_response_delta(self, response: Any):
        raise NotImplementedError(
            f"_NullModel({self.id})._parse_provider_response_delta called — subclass owns parsing."
        )


def _stringify_input(value: Any) -> str:
    """Flatten an Agent ``arun`` input into a plain string.

    Agno passes either a string, a ``Message``, a list of ``Message``,
    a dict, or a pydantic model. The SDK transcript is owned by the
    subprocess via ``--resume`` (Claude) or ``thread_resume`` (Codex);
    we only push the LATEST user turn each call, so the leader's
    "decompose & delegate" turn arrives as a single string per ``arun``
    invocation.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # Message-shaped objects: prefer .content, fall back to str().
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _stringify_input(item)
            if text:
                parts.append(text)
        return "\n\n".join(parts)
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return str(value["content"])
        if isinstance(value.get("content"), list):
            return _stringify_input(value["content"])
        return str(value)
    return str(value)


class BaseSubscriptionBackedAgent(Agent):
    """``Agent`` subclass shared by ``ClaudeBackedAgent`` / ``CodexBackedAgent``.

    Subclasses MUST override:

    - ``SESSION_DATA_KEY``: JSON key in ``session.session_data`` that
      carries the SDK's session id (``"sdk_session_id"`` for Claude,
      ``"codex_session_id"`` for Codex).
    - ``PROVIDER_NAME``: framework name used in metrics
      (``FRAMEWORK_CLAUDE_CLI`` / ``FRAMEWORK_CODEX_CLI``).
    - ``NULL_MODEL_ID``: id stamped onto the ``_NullModel`` placeholder.
    - ``_arun_collect`` / ``_arun_stream``: SDK-specific message-walk
      translation (different SDK shapes — can't be shared).
    """

    SESSION_DATA_KEY: ClassVar[str] = ""
    PROVIDER_NAME: ClassVar[str] = ""
    NULL_MODEL_ID: ClassVar[str] = "subscription-backed-null-model"

    # Subclasses are free to override the error-class string in their own
    # _arun_collect / _arun_stream paths; this is the default elog key.
    ARUN_ERROR_EVENT: ClassVar[str] = "subscription_backed_agent.arun_error"
    STREAM_ERROR_EVENT: ClassVar[str] = "subscription_backed_agent.stream_error"

    def __init__(
        self,
        *,
        name: str,
        sdk_model_id: str,
        sdk_options: Optional[Dict[str, Any]] = None,
        db: Any = None,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            model=_NullModel(id=self.NULL_MODEL_ID, provider=self.PROVIDER_NAME),
            db=db,
            role=role,
            markdown=False,
            **kwargs,
        )
        self._sdk_model_id = sdk_model_id
        self._sdk_options: Dict[str, Any] = dict(sdk_options or {})
        # session_id (Agno) -> SDK session id (Claude subprocess /
        # Codex thread). Used to resume the SDK transcript across
        # within-process turns. Round-tripped via session_data so
        # restart-resume also works.
        self._sdk_session_ids: Dict[str, str] = {}

    # ── Session-id persistence ───────────────────────────────────────

    def _hydrate_from_db(self, session_id: Optional[str]) -> None:
        """Look up ``session.session_data[SESSION_DATA_KEY]`` and
        populate ``self._sdk_session_ids`` so the next SDK call resumes
        the correct subprocess / thread.

        Best-effort: when no DB is attached or the session row isn't
        present yet (first turn), this is a quiet no-op.
        """
        if not session_id or self.db is None:
            return
        if session_id in self._sdk_session_ids:
            return  # already known in-process
        try:
            session = self.db.get_session(session_id=session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("%s._hydrate_from_db: %s", type(self).__name__, e)
            return
        if session is None:
            return
        data = getattr(session, "session_data", None)
        if isinstance(data, dict):
            sdk_sid = data.get(self.SESSION_DATA_KEY)
            if sdk_sid:
                self._sdk_session_ids[session_id] = str(sdk_sid)

    def _save_to_db(self, session_id: Optional[str]) -> None:
        """Persist the freshly-minted SDK session id into Agno's
        ``session.session_data`` so a process restart can resume the
        same subprocess / thread.

        Errors here are non-fatal — the agent already produced its
        response; failing to persist the SDK id just means the next
        boot starts a fresh transcript.
        """
        if not session_id or self.db is None:
            return
        sdk_sid = self._sdk_session_ids.get(session_id)
        if not sdk_sid:
            return
        try:
            session = self.db.get_session(session_id=session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("%s._save_to_db (get): %s", type(self).__name__, e)
            return
        if session is None:
            return
        data = getattr(session, "session_data", None)
        if isinstance(data, dict):
            data[self.SESSION_DATA_KEY] = sdk_sid
        else:
            session.session_data = {self.SESSION_DATA_KEY: sdk_sid}
        try:
            self.db.upsert_session(session=session)
        except Exception as e:  # noqa: BLE001
            logger.debug("%s._save_to_db (upsert): %s", type(self).__name__, e)

    # ── arun (dispatch by stream flag) ──────────────────────────────

    def arun(  # type: ignore[override]
        self,
        input: Any,
        *,
        stream: bool = False,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Bypass ``Agent.arun_dispatch`` and drive the SDK directly.

        Matches Agno's contract: ``arun`` is a sync function that
        returns EITHER a coroutine (when ``stream=False`` — caller
        awaits) OR an async iterator (when ``stream=True`` — caller
        iterates). Returning an async iterator from a sync function
        lets callers do ``async for event in agent.arun(..., stream=True)``
        without an extra ``await``.

        When ``stream=True`` subclasses yield ``RunContentEvent`` /
        ``ToolCall*Event`` and, when ``yield_run_output=True`` (Agno
        Team passes this), a terminal ``RunOutput``.

        Extra ``**kwargs`` from Team (``stream_events``,
        ``session_state``, ``dependencies``, ``knowledge_filters``,
        ``metadata``, ``add_*_to_context``, media, ...) are accepted
        and discarded — the SDK owns its own context-building and
        tool wiring.
        """
        run_id = run_id or str(uuid4())
        if stream:
            return self._arun_stream(
                input,
                user_id=user_id,
                session_id=session_id,
                run_id=run_id,
                yield_run_output=bool(kwargs.get("yield_run_output")),
            )
        return self._arun_collect(
            input,
            user_id=user_id,
            session_id=session_id,
            run_id=run_id,
        )

    def run(self, *args, **kwargs):  # type: ignore[override]
        """Sync entry — Team's sync ``execute_task`` calls
        ``member.run(...)`` outside the event loop. Inside an event
        loop (any async caller) the user should be calling ``arun``;
        we error explicitly rather than the typical Agno fallback to
        ``asyncio.run()`` from inside an already-running loop.
        """
        stream = kwargs.get("stream", False)
        if stream:
            raise RuntimeError(
                f"{type(self).__name__}.run(stream=True) is not supported. "
                "Use arun(stream=True) — every OpenAgent caller is async."
            )
        coro = self.arun(*args, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            f"{type(self).__name__}.run() called from inside an event loop. "
            "Use arun() — Agno Team's async dispatcher already does this."
        )

    # ── Metrics helper ──────────────────────────────────────────────

    def _build_metrics(self, input_tokens: int, output_tokens: int) -> RunMetrics:
        """Build a minimal ``RunMetrics`` with one ``ModelMetrics`` entry
        so downstream telemetry (BudgetTracker is skipped for
        subscription-CLI, but session summaries / member-run aggregation
        still expect a ``metrics`` attribute on the RunOutput) sees a
        valid object.
        """
        mm = ModelMetrics(
            id=self._sdk_model_id or "",
            provider=self.PROVIDER_NAME,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return RunMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            details={"model": [mm]},
        )

    # ── Subclass hooks ──────────────────────────────────────────────

    async def _arun_collect(  # pragma: no cover — abstract hook
        self,
        input: Any,
        *,
        user_id: Optional[str],
        session_id: Optional[str],
        run_id: str,
    ) -> Any:
        raise NotImplementedError

    async def _arun_stream(  # pragma: no cover — abstract hook
        self,
        input: Any,
        *,
        user_id: Optional[str],
        session_id: Optional[str],
        run_id: str,
        yield_run_output: bool,
    ) -> Any:
        raise NotImplementedError
        yield  # type: ignore[unreachable]  # marker: this is an async generator hook


# ════════════════════════════════════════════════════════════════════
#  CLI-side helpers (used by ClaudeCLI / CodexCLI registries)
# ════════════════════════════════════════════════════════════════════


def _last_user_prompt(messages: list[dict[str, Any]] | None) -> str:
    """Render the latest user turn as a plain string for ``arun``.

    Provider-managed history (the SDK keeps its own transcript via
    ``--resume`` or ``thread_resume``) means we only push the LATEST
    user message; everything before it is already in the subprocess'
    context. Content blocks (text, image_url) are flattened to their
    text portions — image parts are dropped because the SDK's prompt
    arg accepts only text.
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


def _prepend_files_for_subscription_cli(prompt: str, files: list[Any]) -> str:
    """Inline ``files=`` into the prompt for subscription-CLI agents.

    Claude-CLI / Codex-CLI SDKs don't accept Agno's ``files=`` argument
    natively. Their tools see attachments via local path through the
    ``Read`` tool, so we fall back to the same context-block shape
    OpenAgent has historically used (see
    ``src.channels.base.build_attachment_context``).

    Logged at ``subscription_cli.files_prepended`` so the eventual native
    file-passing investigation can find the call sites.
    """
    if not files:
        return prompt
    # Local import to keep this module free of the channels dependency at
    # load time.
    from src.channels.base import build_attachment_context, prepend_context_block
    from src.core.logging import elog as _elog

    lines: list[str] = []
    for f in files:
        filename = getattr(f, "filename", None) or getattr(f, "name", None) or ""
        filepath = getattr(f, "filepath", None)
        if filepath:
            lines.append(f"- file: {filename} — local path: {filepath}")
        elif filename:
            lines.append(f"- file: {filename}")
    if not lines:
        return prompt
    _elog("subscription_cli.files_prepended", files=len(lines))
    return prepend_context_block(
        prompt,
        build_attachment_context(
            lines,
            read_hint=(
                "Use the Read tool (or an MCP tool) with the local path "
                "to inspect each file."
            ),
        ),
    )


def _known_sdk_session_ids_by_key(
    db: Any,
    *,
    json_key: str,
    extra_or_clauses: list[str] | None = None,
) -> list[str]:
    """Return ``agno_sessions.session_id`` rows for SDK-backed sessions.

    Shared SQL skeleton: every subscription-CLI ``*BackedAgent`` row
    stamps ``session_data[json_key]`` with the SDK's own session id once
    known. That's the durable marker the gateway's ``/clear`` legacy-row
    fallback can find after a cold restart — before any in-process
    registry / runtime has been re-populated by user activity.

    Claude's legacy ``BaseExternalAgent`` rows used a different marker
    (``agent_data.framework = 'claude-agent-sdk'``); pass
    ``extra_or_clauses=["json_extract(agent_data, '$.framework') = 'claude-agent-sdk'"]``
    to broaden the filter.
    """
    clauses = [f"json_extract(session_data, '$.{json_key}') IS NOT NULL"]
    if extra_or_clauses:
        clauses.extend(extra_or_clauses)
    return _query_sessions(db, " OR ".join(clauses))


def all_subscription_session_ids(db: Any) -> list[str]:
    """Return every ``agno_sessions`` row whose data identifies it as a
    subscription-CLI-backed session.

    Single SQL query coalescing every known marker — ``session_data``'s
    ``sdk_session_id`` / ``codex_session_id`` keys plus the legacy
    ``agent_data.framework = 'claude-agent-sdk'`` field. Used by
    :meth:`ModelDispatcher.known_session_ids` so ``/clear`` doesn't fan
    out into three full-table scans.
    """
    return _query_sessions(
        db,
        " OR ".join((
            "json_extract(session_data, '$.sdk_session_id') IS NOT NULL",
            "json_extract(session_data, '$.codex_session_id') IS NOT NULL",
            "json_extract(agent_data, '$.framework') = 'claude-agent-sdk'",
        )),
    )


def _query_sessions(db: Any, where_clause: str) -> list[str]:
    if db is None:
        return []
    db_path = getattr(db, "db_path", None)
    if not db_path:
        return []
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        try:
            cur = conn.execute(
                f"SELECT session_id FROM agno_sessions "
                f"WHERE {where_clause} "
                f"ORDER BY session_id"
            )
            return [str(r[0]) for r in cur.fetchall()]
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.debug("_query_sessions(%s): %s", where_clause, e)
        return []


def _make_sqlite_db(db: Any) -> Any | None:
    """Build an ``agno.db.sqlite.SqliteDb`` pointing at the same file as
    OpenAgent's ``MemoryDB``. Returns ``None`` when the DB isn't wired
    or Agno's SqliteDb isn't importable.

    Shared by both subscription-CLI registries so the agno-side session
    storage lands in the same file the rest of OpenAgent writes to.
    """
    if db is None:
        return None
    db_path = getattr(db, "db_path", None)
    if not db_path:
        return None
    try:
        from agno.db.sqlite import SqliteDb

        return SqliteDb(db_file=str(db_path))
    except ImportError as e:
        logger.debug("_make_sqlite_db: agno.db.sqlite import failed: %s", e)
        return None
