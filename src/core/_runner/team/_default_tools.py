"""Built-in tool factory functions for Team."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core._runner.team.team import Team

import asyncio
import contextlib
import json
import os
import uuid
from copy import copy


def _team_member_sessions_enabled() -> bool:
    """Mirror of ``models.dispatcher._team_member_sessions_enabled`` (kept
    local to avoid a dispatcher↔team import cycle). When ON, a delegated
    team member runs in its OWN durable child session row rather than nested
    in the team session, so it surfaces as a navigable card."""
    return os.environ.get("OPENAGENT_TEAM_MEMBER_SESSIONS", "1").strip() not in ("0", "false", "no")


def _member_run_session_id(team_session_id, member_agent) -> str:
    """The session id a member delegation runs on: a unique per-delegation
    child id (``{team}::member::{member}::{rand}``) when team-member sessions
    are enabled, else the shared team session id (legacy nested behavior).
    Unique per call so concurrent / repeated delegations never collide."""
    if not team_session_id or not _team_member_sessions_enabled():
        return team_session_id
    mid = getattr(member_agent, "id", None) or getattr(member_agent, "name", None) or "member"
    return f"{team_session_id}::member::{mid}::{uuid.uuid4().hex[:8]}"


def _child_session_meta(team_session, member_agent, *, child_run_id=None) -> dict:
    """Build a delegated member's child-session link metadata (parent /
    origin / inherited owner, + optionally the member ``child_run_id`` bridge).
    Shared by the at-START announce and the at-COMPLETION persist so the stub
    and the final row never drift."""
    meta = dict(getattr(member_agent, "metadata", None) or {})
    meta.setdefault("parent_session_id", getattr(team_session, "session_id", None))
    meta.setdefault("origin", "delegation")
    if child_run_id:
        meta["child_run_id"] = child_run_id
    parent_meta = getattr(team_session, "metadata", None) or {}
    if parent_meta.get("client_id") and not meta.get("client_id"):
        meta["client_id"] = parent_meta.get("client_id")
    return meta


async def _upsert_member_child_row(member_agent, sid, meta, runs) -> None:
    """Write the child ``sessions`` row via the lower-level ``aupsert_session``
    (no ``team_id`` gate — the runner's ``asave_session`` would otherwise refuse
    a member's own session). ``user_id`` stays the runtime sentinel; owner
    visibility rides on ``metadata.client_id``."""
    import time as _time

    from src.core._runner.agent import _storage
    from src.memory.sessions.agent import AgentSession

    now = int(_time.time())
    child = AgentSession(
        session_id=sid,
        agent_id=getattr(member_agent, "id", None) or getattr(member_agent, "name", None),
        user_id=(getattr(runs[-1], "user_id", None) if runs else None) or "openagent",
        metadata=meta,
        runs=list(runs),
        created_at=now,
        updated_at=now,
    )
    await _storage.aupsert_session(member_agent, session=child)


async def _announce_member_child_session(team_session, member_agent, member_session_id) -> None:
    """Pre-create the child session row + announce it the MOMENT a delegation
    starts — not when it finishes — so the sub-agent appears in the sidebar
    immediately and its live token stream (emitted below) has a durable row to
    land on while it works. Best-effort; never blocks delegation."""
    try:
        db = getattr(member_agent, "db", None)
        if db is None or not member_session_id:
            return
        if member_session_id == getattr(team_session, "session_id", None):
            return
        if not _team_member_sessions_enabled():
            return
        meta = _child_session_meta(team_session, member_agent)
        await _upsert_member_child_row(member_agent, member_session_id, meta, runs=[])
        # Announce so connected clients add the sub-agent to the sidebar live.
        try:
            from src.core.child_session import _notify_created
            _notify_created(member_session_id, {
                "owner": meta.get("client_id"),
                "origin": "delegation",
                "parent_session_id": meta.get("parent_session_id"),
                "title": meta.get("title"),
            })
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001 — announce is best-effort
        pass


async def _persist_member_child_session(team_session, member_agent, member_run) -> None:
    """Persist a delegated member's run as its OWN durable child session row
    once the run completes (overwriting the stub written at announce time with
    the full transcript). Records the member ``run_id`` so the parent delegate
    tool — which reliably carries the same value as ``child_run_id`` — links to
    this child session deterministically at rehydration (the runtime's team
    run_response object the leader mutates is NOT the one persisted, so the
    tool's own ``child_session_id`` can't be trusted). Best-effort.
    """
    try:
        db = getattr(member_agent, "db", None)
        sid = getattr(member_run, "session_id", None)
        if db is None or not sid or sid == getattr(team_session, "session_id", None):
            return
        if not _team_member_sessions_enabled():
            return
        meta = _child_session_meta(
            team_session, member_agent, child_run_id=getattr(member_run, "run_id", None),
        )
        await _upsert_member_child_row(member_agent, sid, meta, runs=[member_run])
    except Exception:  # noqa: BLE001 — persistence is best-effort
        pass


# Member run event classes (agent + nested-team variants), resolved once and
# memoized — the team runner imports them lazily to avoid an import cycle.
_member_event_types_cache: Optional[dict] = None


def _member_event_types() -> dict:
    global _member_event_types_cache
    if _member_event_types_cache is None:
        from src.core._run_state.agent import (
            RunContentEvent as _AC, ToolCallCompletedEvent as _ACd,
            ToolCallErrorEvent as _AE, ToolCallStartedEvent as _AS,
        )
        from src.core._run_state.team import (
            RunContentEvent as _TC, ToolCallCompletedEvent as _TCd,
            ToolCallErrorEvent as _TE, ToolCallStartedEvent as _TS,
        )
        _member_event_types_cache = {
            "content": (_AC, _TC),
            "tool_start": (_AS, _TS),
            "tool_done": (_ACd, _TCd),
            "tool_err": (_AE, _TE),
        }
    return _member_event_types_cache


async def _emit_member_card_link(run_response, team_session, member_id, member_session_id) -> None:
    """Stamp the child session id onto the leader's IN-PROGRESS delegate tool
    and re-stream its status so the parent's delegation card becomes clickable
    WHILE the sub-agent runs — otherwise the completed frame is the first
    carrier of ``child_session_id`` and the card stays inert until the
    sub-agent finishes. Best-effort."""
    try:
        parent_sid = getattr(team_session, "session_id", None)
        if not member_session_id or member_session_id == parent_sid:
            return
        target = None
        for t in (getattr(run_response, "tools", None) or []):
            if (getattr(t, "tool_name", None) or "").lower() != "delegate_task_to_member":
                continue
            if getattr(t, "child_session_id", None):
                continue  # already linked (a sibling delegation)
            args = getattr(t, "tool_args", None) or {}
            if member_id and args.get("member_id") and args.get("member_id") != member_id:
                continue
            target = t
            break
        if target is None:
            return
        target.child_session_id = member_session_id  # type: ignore[attr-defined]
        from src.models._tool_status import tool_exec_to_wire_json
        from src.stream.child_stream import emit_child_frame
        encoded = tool_exec_to_wire_json(target, phase="started")
        if encoded is not None:
            await emit_child_frame(parent_sid, "status", text=encoded)
    except Exception:  # noqa: BLE001 — card linkage is best-effort
        pass


async def _emit_member_event(member_session_id, event) -> None:
    """Mirror ONE member run event onto the child session's OWN live stream
    (tagged with the child ``session_id``) over the active turn's channel — so
    navigating into a running sub-agent shows its work in real time, exactly
    like any session. No-op when no streaming client drives the turn. The
    PARENT stream suppresses these same events (dispatcher); this is where they
    surface, on the child's channel."""
    try:
        if not member_session_id:
            return
        from src.stream.child_stream import current_child_stream_emitter, emit_child_frame
        if current_child_stream_emitter() is None:
            return
        types = _member_event_types()
        if isinstance(event, types["content"]):
            content = getattr(event, "content", None)
            if isinstance(content, str) and content:
                await emit_child_frame(member_session_id, "delta", text=content)
            return
        if isinstance(event, types["tool_start"] + types["tool_done"] + types["tool_err"]):
            from src.models._tool_status import tool_exec_to_wire_json
            tool = getattr(event, "tool", None)
            err = getattr(event, "error", None) if isinstance(event, types["tool_err"]) else None
            phase = "started" if isinstance(event, types["tool_start"]) else None
            encoded = tool_exec_to_wire_json(tool, phase=phase, error_text=err)
            if encoded is not None:
                await emit_child_frame(member_session_id, "status", text=encoded)
    except Exception:  # noqa: BLE001 — child mirroring is best-effort
        pass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

from pydantic import BaseModel

from src.core._runner.agent import Agent
from src.memory.store.base import AsyncBaseDb, BaseDb, SessionType
from src.core._runner._stubs import FilterExpr
from src.core._runner._stubs import KnowledgeFilter
from src.stream.media import Audio, File, Image, Video
from src.core._runner._stubs import MemoryManager
from src.models.providers.message import Message, MessageReferences
from src.core._run_state import RunContext
from src.core._run_state.agent import RunOutput, RunOutputEvent
from src.core._run_state.team import (
    TeamRunOutput,
    TeamRunOutputEvent,
)
from src.memory.sessions import TeamSession
from src.mcp._runtime.function import Function
from src.core._runner.utils.knowledge import get_agentic_or_user_search_filters
from src.core._runner.utils.log import (
    log_debug,
    log_info,
    log_warning,
    use_agent_logger,
    use_team_logger,
)
from src.core._runner.utils.merge_dict import merge_dictionaries
from src.core._runner.utils.response import (
    check_if_run_cancelled,
)
from src.core._runner.utils.team import (
    add_interaction_to_team_run_context,
    format_member_agent_task,
)
from src.core._runner.utils.timer import Timer


def _get_update_user_memory_function(team: "Team", user_id: Optional[str] = None, async_mode: bool = False) -> Function:
    def update_user_memory(task: str) -> str:
        """
        Use this function to submit a task to modify the Agent's memory.
        Describe the task in detail and be specific.
        The task can include adding a memory, updating a memory, deleting a memory, or clearing all memories.

        Args:
            task: The task to update the memory. Be specific and describe the task in detail.

        Returns:
            str: A string indicating the status of the update.
        """
        team.memory_manager = cast(MemoryManager, team.memory_manager)
        response = team.memory_manager.update_memory_task(task=task, user_id=user_id)
        return response

    async def aupdate_user_memory(task: str) -> str:
        """
        Use this function to submit a task to modify the Agent's memory.
        Describe the task in detail and be specific.
        The task can include adding a memory, updating a memory, deleting a memory, or clearing all memories.

        Args:
            task: The task to update the memory. Be specific and describe the task in detail.

        Returns:
            str: A string indicating the status of the update.
        """
        team.memory_manager = cast(MemoryManager, team.memory_manager)
        response = await team.memory_manager.aupdate_memory_task(task=task, user_id=user_id)
        return response

    if async_mode:
        update_memory_function = aupdate_user_memory
    else:
        update_memory_function = update_user_memory  # type: ignore

    return Function.from_callable(update_memory_function, name="update_user_memory")


def _get_chat_history_function(team: "Team", session: TeamSession, async_mode: bool = False):
    def get_chat_history(num_chats: Optional[int] = None) -> str:
        """
        Use this function to get the team chat history in reverse chronological order.
        Leave the num_chats parameter blank to get the entire chat history.
        Example:
            - To get the last chat, use num_chats=1
            - To get the last 5 chats, use num_chats=5
            - To get all chats, leave num_chats blank

        Args:
            num_chats: The number of chats to return.
                Each chat contains 2 messages. One from the team and one from the user.
                Default: None

        Returns:
            str: A JSON string containing a list of dictionaries representing the team chat history.
        """
        import json

        all_chats = session.get_messages(team_id=team.id)

        if len(all_chats) == 0:
            return ""

        history: List[Dict[str, Any]] = [chat.to_dict() for chat in all_chats]  # type: ignore

        if num_chats is not None:
            history = history[-num_chats:]

        return json.dumps(history)

    async def aget_chat_history(num_chats: Optional[int] = None) -> str:
        """
        Use this function to get the team chat history in reverse chronological order.
        Leave the num_chats parameter blank to get the entire chat history.
        Example:
            - To get the last chat, use num_chats=1
            - To get the last 5 chats, use num_chats=5
            - To get all chats, leave num_chats blank

        Args:
            num_chats: The number of chats to return.
                Each chat contains 2 messages. One from the team and one from the user.
                Default: None

        Returns:
            str: A JSON string containing a list of dictionaries representing the team chat history.
        """
        import json

        all_chats = session.get_messages(team_id=team.id)

        if len(all_chats) == 0:
            return ""

        history: List[Dict[str, Any]] = [chat.to_dict() for chat in all_chats]  # type: ignore

        if num_chats is not None:
            history = history[-num_chats:]

        return json.dumps(history)

    if async_mode:
        get_chat_history_func = aget_chat_history
    else:
        get_chat_history_func = get_chat_history  # type: ignore
    return Function.from_callable(get_chat_history_func, name="get_chat_history")


def _update_session_state_tool(team: "Team", run_context: RunContext, session_state_updates: dict) -> str:
    """
    Update the shared session state.  Provide any updates as a dictionary of key-value pairs.
    Example:
        "session_state_updates": {"shopping_list": ["milk", "eggs", "bread"]}

    Args:
        session_state_updates (dict): The updates to apply to the shared session state. Should be a dictionary of key-value pairs.
    """
    if run_context.session_state is None:
        run_context.session_state = {}
    session_state = run_context.session_state
    for key, value in session_state_updates.items():
        session_state[key] = value

    return f"Updated session state: {session_state}"


def _search_past_sessions_function(
    team: "Team",
    num_past_sessions_to_search: Optional[int] = None,
    num_past_session_runs_in_search: Optional[int] = None,
    user_id: Optional[str] = None,
    current_session_id: Optional[str] = None,
    async_mode: bool = False,
) -> Function:
    """Factory for search_past_sessions tool for Team."""

    from src.core._runner.agent._default_tools import _extract_session_preview
    from src.core._runner.team._init import _has_async_db

    _limit = num_past_sessions_to_search if num_past_sessions_to_search is not None else 20
    _num_runs = num_past_session_runs_in_search if num_past_session_runs_in_search is not None else 3

    def search_past_sessions() -> str:
        """List previous chat sessions with short previews.
        Use read_past_session to read the full conversation for a specific session.

        Returns:
            str: JSON list of session previews with session_id, created_at, and runs (user/assistant pairs).
        """
        if team.db is None:
            return json.dumps([])

        team.db = cast(BaseDb, team.db)
        selected_sessions = team.db.get_sessions(
            session_type=SessionType.TEAM,
            limit=_limit,
            user_id=user_id,
            sort_by="created_at",
            sort_order="desc",
        )

        results: list = []
        for session in selected_sessions:
            if not isinstance(session, TeamSession) or not session.runs:
                continue
            if current_session_id and session.session_id == current_session_id:
                continue
            results.append(_extract_session_preview(session, num_runs=_num_runs))

        return json.dumps(results)

    async def asearch_past_sessions() -> str:
        """List previous chat sessions with short previews.
        Use read_past_session to read the full conversation for a specific session.

        Returns:
            str: JSON list of session previews with session_id, created_at, and runs (user/assistant pairs).
        """
        if team.db is None:
            return json.dumps([])

        if _has_async_db(team):
            selected_sessions = await cast(AsyncBaseDb, team.db).get_sessions(  # type: ignore
                session_type=SessionType.TEAM,
                limit=_limit,
                user_id=user_id,
                sort_by="created_at",
                sort_order="desc",
            )
        else:
            selected_sessions = team.db.get_sessions(  # type: ignore
                session_type=SessionType.TEAM,
                limit=_limit,
                user_id=user_id,
                sort_by="created_at",
                sort_order="desc",
            )

        results: list = []
        for session in selected_sessions:
            if not isinstance(session, TeamSession) or not session.runs:
                continue
            if current_session_id and session.session_id == current_session_id:
                continue
            results.append(_extract_session_preview(session, num_runs=_num_runs))

        return json.dumps(results)

    if async_mode and _has_async_db(team):
        return Function.from_callable(asearch_past_sessions, name="search_past_sessions")
    return Function.from_callable(search_past_sessions, name="search_past_sessions")


def _read_past_session_function(
    team: "Team",
    user_id: Optional[str] = None,
    async_mode: bool = False,
) -> Function:
    """Factory for read_past_session tool for Team."""

    from src.core._runner.agent._default_tools import _get_message_text
    from src.core._runner.team._init import _has_async_db

    def read_past_session(session_id: str, num_runs: Optional[int] = None) -> str:
        """Read the full conversation from a previous session.
        Use search_past_sessions first to find relevant sessions.

        Args:
            session_id: The session ID to read (from search results).
            num_runs: Maximum number of runs to include. Default: all runs.

        Returns:
            str: The conversation formatted as User/Assistant message pairs.
        """
        if team.db is None:
            return "No database configured."

        team.db = cast(BaseDb, team.db)
        session = team.db.get_session(
            session_id=session_id,
            session_type=SessionType.TEAM,
            user_id=user_id,
        )

        if session is None or not isinstance(session, TeamSession) or not session.runs:
            return "Session not found."

        lines: list = []
        lines.append(f"Session: {session.session_id}")
        if session.created_at:
            lines.append(f"Created: {session.created_at}")
        lines.append("")

        runs = session.runs if num_runs is None else session.runs[:num_runs]
        for run in runs:
            for msg in run.messages or []:
                if msg.role not in ("user", "assistant"):
                    continue
                text = _get_message_text(msg)
                if text:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    lines.append(f"{role_label}: {text}")
                    lines.append("")

        return "\n".join(lines) if lines else "No messages found in session."

    async def aread_past_session(session_id: str, num_runs: Optional[int] = None) -> str:
        """Read the full conversation from a previous session.
        Use search_past_sessions first to find relevant sessions.

        Args:
            session_id: The session ID to read (from search results).
            num_runs: Maximum number of runs to include. Default: all runs.

        Returns:
            str: The conversation formatted as User/Assistant message pairs.
        """
        if team.db is None:
            return "No database configured."

        if _has_async_db(team):
            session = await cast(AsyncBaseDb, team.db).get_session(  # type: ignore
                session_id=session_id,
                session_type=SessionType.TEAM,
                user_id=user_id,
            )
        else:
            session = team.db.get_session(  # type: ignore
                session_id=session_id,
                session_type=SessionType.TEAM,
                user_id=user_id,
            )

        if session is None or not isinstance(session, TeamSession) or not session.runs:
            return "Session not found."

        lines: list = []
        lines.append(f"Session: {session.session_id}")
        if session.created_at:
            lines.append(f"Created: {session.created_at}")
        lines.append("")

        runs = session.runs if num_runs is None else session.runs[:num_runs]
        for run in runs:
            for msg in run.messages or []:
                if msg.role not in ("user", "assistant"):
                    continue
                text = _get_message_text(msg)
                if text:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    lines.append(f"{role_label}: {text}")
                    lines.append("")

        return "\n".join(lines) if lines else "No messages found in session."

    if async_mode and _has_async_db(team):
        return Function.from_callable(aread_past_session, name="read_past_session")
    return Function.from_callable(read_past_session, name="read_past_session")


def _get_delegate_task_function(
    team: "Team",
    run_response: TeamRunOutput,
    run_context: RunContext,
    session: TeamSession,
    team_run_context: Dict[str, Any],
    user_id: Optional[str] = None,
    stream: bool = False,
    stream_events: bool = False,
    async_mode: bool = False,
    input: Optional[str] = None,  # Used for determine_input_for_members=False
    images: Optional[List[Image]] = None,
    videos: Optional[List[Video]] = None,
    audio: Optional[List[Audio]] = None,
    files: Optional[List[File]] = None,
    add_history_to_context: Optional[bool] = None,
    add_dependencies_to_context: Optional[bool] = None,
    add_session_state_to_context: Optional[bool] = None,
    debug_mode: Optional[bool] = None,
) -> Function:
    from src.core._runner.team._init import _initialize_member
    from src.core._runner.team._run import _update_team_media
    from src.core._runner.team._tools import (
        _determine_team_member_interactions,
        _find_member_by_id,
        _get_history_for_member_agent,
        _propagate_member_pause,
    )

    if not images:
        images = []
    if not videos:
        videos = []
    if not audio:
        audio = []
    if not files:
        files = []

    def _setup_delegate_task_to_member(member_agent: Union[Agent, "Team"], task: str):
        # 1. Initialize the member agent

        _initialize_member(team, member_agent)

        # If team has send_media_to_model=False, ensure member agent also has it set to False
        # This allows tools to access files while preventing models from receiving them
        if not team.send_media_to_model:
            member_agent.send_media_to_model = False

        # 2. Handle respond_directly nuances
        if team.respond_directly:
            # Since we return the response directly from the member agent, we need to set the output schema from the team down.
            # Get output_schema from run_context
            team_output_schema = run_context.output_schema if run_context else None
            if not member_agent.output_schema and team_output_schema:
                member_agent.output_schema = team_output_schema

            # If the member will produce structured output, we need to parse the response
            if member_agent.output_schema is not None:
                team._member_response_model = member_agent.output_schema

        # 3. Handle enable_agentic_knowledge_filters on the member agent
        if team.enable_agentic_knowledge_filters and not member_agent.enable_agentic_knowledge_filters:
            member_agent.enable_agentic_knowledge_filters = team.enable_agentic_knowledge_filters

        # 4. Determine team context to send
        team_member_interactions_str = _determine_team_member_interactions(
            team, team_run_context, images=images, videos=videos, audio=audio, files=files
        )

        # 5. Get the team history
        team_history_str = None
        if team.add_team_history_to_members and session:
            team_history_str = session.get_team_history_context(num_runs=team.num_team_history_runs)

        # 6. Create the member agent task or use the input directly
        if team.determine_input_for_members is False:
            member_agent_task = input  # type: ignore
        else:
            member_agent_task = task

        if team_history_str or team_member_interactions_str:
            member_agent_task = format_member_agent_task(  # type: ignore
                task_description=member_agent_task or "",
                team_member_interactions_str=team_member_interactions_str,
                team_history_str=team_history_str,
            )

        # 7. Add member-level history for the member if enabled (because we won't load the session for the member, so history won't be loaded automatically)
        history = None
        if hasattr(member_agent, "add_history_to_context") and member_agent.add_history_to_context:
            history = _get_history_for_member_agent(team, session, member_agent)
            if history:
                if isinstance(member_agent_task, str):
                    history.append(Message(role="user", content=member_agent_task))

        return member_agent_task, history

    def _process_delegate_task_to_member(
        member_agent_run_response: Optional[Union[TeamRunOutput, RunOutput]],
        member_agent: Union[Agent, "Team"],
        member_agent_task: Union[str, Message],
        member_session_state_copy: Dict[str, Any],
    ):
        # Add team run id to the member run

        if member_agent_run_response is not None:
            member_agent_run_response.parent_run_id = run_response.run_id  # type: ignore

        # Update the top-level team run_response tool call to have the run_id of the member run
        if run_response.tools is not None and member_agent_run_response is not None:
            for tool in run_response.tools:
                if tool.tool_name and tool.tool_name.lower() == "delegate_task_to_member":
                    tool.child_run_id = member_agent_run_response.run_id  # type: ignore
                    # When the member ran in its OWN session (id differs from
                    # the team session), record it so the leader transcript can
                    # render a card that deep-links into the sub-agent's full
                    # session. No-op in legacy nested mode (same session id).
                    member_sid = getattr(member_agent_run_response, "session_id", None)
                    if member_sid and member_sid != session.session_id:
                        tool.child_session_id = member_sid  # type: ignore

        # Update the team run context
        member_name = member_agent.name if member_agent.name else member_agent.id if member_agent.id else "Unknown"
        if isinstance(member_agent_task, str):
            normalized_task = member_agent_task
        elif member_agent_task.content:
            normalized_task = str(member_agent_task.content)
        else:
            normalized_task = ""
        add_interaction_to_team_run_context(
            team_run_context=team_run_context,
            member_name=member_name,
            task=normalized_task,
            run_response=member_agent_run_response,  # type: ignore
        )

        # Add the member run to the team run response if enabled
        if run_response and member_agent_run_response:
            run_response.add_member_run(member_agent_run_response)

        # Scrub the member run based on that member's storage flags before storing
        if member_agent_run_response:
            if (
                not member_agent.store_media
                or not member_agent.store_tool_messages
                or not member_agent.store_history_messages
            ):
                from src.core._runner.agent._run import scrub_run_output_for_storage

                scrub_run_output_for_storage(member_agent, run_response=member_agent_run_response)  # type: ignore[arg-type]

            # Add the member run to the team session — but ONLY in legacy
            # nested mode. When the member ran in its OWN child session row
            # (id differs from the team session), persisting its full run into
            # the TEAM session too would (a) duplicate the transcript and
            # (b) surface the member's leader-authored task prompt as a
            # top-level "user" message in the parent chat (the reported
            # "my message shows the delegation prompt" bug). In child-session
            # mode the member's transcript lives in its own row; the parent
            # only needs the delegation card (child_session_id on the tool +
            # the member_responses linkage used at rehydration).
            _member_sid = getattr(member_agent_run_response, "session_id", None)
            if not (_member_sid and _member_sid != session.session_id
                    and _team_member_sessions_enabled()):
                session.upsert_run(member_agent_run_response)

        # Update team session state
        merge_dictionaries(run_context.session_state, member_session_state_copy)  # type: ignore

        # Update the team media
        if member_agent_run_response is not None:
            _update_team_media(team, member_agent_run_response)  # type: ignore

    def delegate_task_to_member(member_id: str, task: str) -> Iterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
        """Use this function to delegate a task to the selected team member.
        You must provide a clear and concise description of the task the member should achieve AND the expected output.

        Args:
            member_id (str): The ID of the member to delegate the task to. Use only the ID of the member, not the ID of the team followed by the ID of the member.
            task (str): A clear and concise description of the task the member should achieve.
        Returns:
            str: The result of the delegated task.
        """

        # Find the member agent using the helper function
        result = _find_member_by_id(team, member_id, run_context=run_context)
        if result is None:
            yield f"Member with ID {member_id} not found in the team or any subteams. Please choose the correct member from the list of members:\n\n{team.get_members_system_message_content(indent=0, run_context=run_context)}"
            return

        _, member_agent = result
        member_agent_task, history = _setup_delegate_task_to_member(member_agent=member_agent, task=task)

        # Make sure for the member agent, we are using the agent logger
        use_agent_logger()

        member_session_state_copy = copy(run_context.session_state)
        # A unique per-delegation session so the member runs as its own full
        # child session (navigable card); falls back to the team session id
        # when team-member sessions are disabled (legacy nested behavior).
        member_session_id = _member_run_session_id(session.session_id, member_agent)

        if stream:
            member_agent_run_response_stream = member_agent.run(
                input=member_agent_task if not history else history,
                user_id=user_id,
                session_id=member_session_id,
                session_state=member_session_state_copy,  # Send a copy to the agent
                images=images,
                videos=videos,
                audio=audio,
                files=files,
                stream=True,
                stream_events=stream_events or team.stream_member_events,
                debug_mode=debug_mode,
                dependencies=run_context.dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                metadata=run_context.metadata,
                add_session_state_to_context=add_session_state_to_context,
                knowledge_filters=run_context.knowledge_filters
                if not member_agent.knowledge_filters and member_agent.knowledge
                else None,
                yield_run_output=True,
            )
            member_agent_run_response = None
            for member_agent_run_output_event in member_agent_run_response_stream:
                # Do NOT break out of the loop, Iterator need to exit properly
                if isinstance(member_agent_run_output_event, (TeamRunOutput, RunOutput)):
                    member_agent_run_response = member_agent_run_output_event  # type: ignore
                    continue  # Don't yield TeamRunOutput or RunOutput, only yield events

                # Check if the run is cancelled
                check_if_run_cancelled(member_agent_run_output_event)

                # Yield the member event directly
                member_agent_run_output_event.parent_run_id = (
                    member_agent_run_output_event.parent_run_id or run_response.run_id
                )
                yield member_agent_run_output_event  # type: ignore
        else:
            member_agent_run_response = member_agent.run(  # type: ignore
                input=member_agent_task if not history else history,  # type: ignore
                user_id=user_id,
                session_id=member_session_id,
                session_state=member_session_state_copy,  # Send a copy to the agent
                images=images,
                videos=videos,
                audio=audio,
                files=files,
                stream=False,
                debug_mode=debug_mode,
                dependencies=run_context.dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=run_context.metadata,
                knowledge_filters=run_context.knowledge_filters
                if not member_agent.knowledge_filters and member_agent.knowledge
                else None,
            )

            check_if_run_cancelled(member_agent_run_response)  # type: ignore

        # Check if the member run is paused (HITL)
        if member_agent_run_response is not None and member_agent_run_response.is_paused:
            _propagate_member_pause(run_response, member_agent, member_agent_run_response)
            use_team_logger()
            _process_delegate_task_to_member(
                member_agent_run_response,
                member_agent,
                member_agent_task,  # type: ignore
                member_session_state_copy,  # type: ignore
            )
            yield f"Member '{member_agent.name}' requires human input before continuing."
            return

        if not stream:
            try:
                if member_agent_run_response.content is None and (  # type: ignore
                    member_agent_run_response.tools is None or len(member_agent_run_response.tools) == 0  # type: ignore
                ):
                    yield "No response from the member agent."
                elif isinstance(member_agent_run_response.content, str):  # type: ignore
                    content = member_agent_run_response.content.strip()  # type: ignore
                    if len(content) > 0:
                        yield content

                    # If the content is empty but we have tool calls
                    elif member_agent_run_response.tools is not None and len(member_agent_run_response.tools) > 0:  # type: ignore
                        tool_str = ""
                        for tool in member_agent_run_response.tools:  # type: ignore
                            if tool.result:
                                tool_str += f"{tool.result},"
                        yield tool_str.rstrip(",")

                elif issubclass(type(member_agent_run_response.content), BaseModel):  # type: ignore
                    yield member_agent_run_response.content.model_dump_json(indent=2)  # type: ignore
                else:
                    import json

                    yield json.dumps(member_agent_run_response.content, indent=2)  # type: ignore
            except Exception as e:
                yield str(e)

        # Afterward, switch back to the team logger
        use_team_logger()

        _process_delegate_task_to_member(
            member_agent_run_response,
            member_agent,
            member_agent_task,  # type: ignore
            member_session_state_copy,  # type: ignore
        )

    async def adelegate_task_to_member(
        member_id: str, task: str
    ) -> AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
        """Use this function to delegate a task to the selected team member.
        You must provide a clear and concise description of the task the member should achieve AND the expected output.

        Args:
            member_id (str): The ID of the member to delegate the task to. Use only the ID of the member, not the ID of the team followed by the ID of the member.
            task (str): A clear and concise description of the task the member should achieve.
        Returns:
            str: The result of the delegated task.
        """

        # Find the member agent using the helper function
        result = _find_member_by_id(team, member_id, run_context=run_context)
        if result is None:
            yield f"Member with ID {member_id} not found in the team or any subteams. Please choose the correct member from the list of members:\n\n{team.get_members_system_message_content(indent=0, run_context=run_context, async_mode=True)}"
            return

        _, member_agent = result
        member_agent_task, history = _setup_delegate_task_to_member(member_agent=member_agent, task=task)

        # Make sure for the member agent, we are using the agent logger
        use_agent_logger()

        member_session_state_copy = copy(run_context.session_state)
        # Unique per-delegation session → member runs as its own child session.
        member_session_id = _member_run_session_id(session.session_id, member_agent)

        # The member's seed message is the LEADER's task, not a human's words.
        # The per-turn author contextvar carries the human handle (so the
        # parent chat attributes the user correctly); a delegated member runs
        # inside that same context, so without this override its task prompt
        # would be stamped with the human's identity (the reported "my message
        # shows the delegation prompt" bug). Override to an agent-self author so
        # the child session renders the seed as a Mission block. Reset right
        # after the member run (the HITL/synthesis code below stamps nothing).
        _member_author_tok = None
        if member_session_id and member_session_id != session.session_id and _team_member_sessions_enabled():
            try:
                from src.core.identity_context import agent_author, install_author_context
                _member_author_tok = install_author_context(
                    agent_author(
                        f"Sub-agent · {getattr(member_agent, 'name', None) or member_id}",
                        agent_name=getattr(member_agent, "name", None),
                    )
                )
            except Exception:  # noqa: BLE001 — never block delegation on authorship
                _member_author_tok = None

        # A delegation IS a navigable sub-agent session. Create + announce it
        # the moment it STARTS (so it appears in the sidebar and its live
        # stream has a row to land on), and stamp + stream the child id onto
        # the leader's in-progress delegate tool so the card is clickable
        # immediately — both previously only happened at completion.
        if member_session_id and member_session_id != session.session_id and _team_member_sessions_enabled():
            await _announce_member_child_session(session, member_agent, member_session_id)
            await _emit_member_card_link(run_response, session, member_id, member_session_id)

        if stream:
            member_agent_run_response_stream = member_agent.arun(  # type: ignore
                input=member_agent_task if not history else history,
                user_id=user_id,
                session_id=member_session_id,
                session_state=member_session_state_copy,  # Send a copy to the agent
                images=images,
                videos=videos,
                audio=audio,
                files=files,
                stream=True,
                stream_events=stream_events or team.stream_member_events,
                debug_mode=debug_mode,
                dependencies=run_context.dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=run_context.metadata,
                knowledge_filters=run_context.knowledge_filters
                if not member_agent.knowledge_filters and member_agent.knowledge
                else None,
                yield_run_output=True,
            )
            member_agent_run_response = None
            async for member_agent_run_response_event in member_agent_run_response_stream:
                # Do NOT break out of the loop, AsyncIterator need to exit properly
                if isinstance(member_agent_run_response_event, (TeamRunOutput, RunOutput)):
                    member_agent_run_response = member_agent_run_response_event  # type: ignore
                    continue  # Don't yield TeamRunOutput or RunOutput, only yield events

                # Check if the run is cancelled
                check_if_run_cancelled(member_agent_run_response_event)

                # Yield the member event directly
                member_agent_run_response_event.parent_run_id = getattr(
                    member_agent_run_response_event, "parent_run_id", None
                ) or (run_response.run_id if run_response is not None else None)
                yield member_agent_run_response_event  # type: ignore

                # Mirror the event onto the child session's own live stream so
                # navigating into the running sub-agent shows it work in real
                # time. The parent stream suppresses these (dispatcher); this
                # surfaces them on the child's channel, tagged with its sid.
                await _emit_member_event(member_session_id, member_agent_run_response_event)
        else:
            member_agent_run_response = await member_agent.arun(  # type: ignore
                input=member_agent_task if not history else history,
                user_id=user_id,
                session_id=member_session_id,
                session_state=member_session_state_copy,  # Send a copy to the agent
                images=images,
                videos=videos,
                audio=audio,
                files=files,
                stream=False,
                debug_mode=debug_mode,
                dependencies=run_context.dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=run_context.metadata,
                knowledge_filters=run_context.knowledge_filters
                if not member_agent.knowledge_filters and member_agent.knowledge
                else None,
            )
            check_if_run_cancelled(member_agent_run_response)  # type: ignore

        # Pin the member RunOutput's session_id to the minted child id. The
        # runtime can leave ``RunOutput.session_id`` pointing at the team
        # session, which would (a) make ``_process`` skip the child_session_id
        # card link and (b) mislabel the child row — so make it authoritative
        # here for the persistence + linkage steps below.
        if (member_agent_run_response is not None
                and member_session_id and member_session_id != session.session_id):
            try:
                member_agent_run_response.session_id = member_session_id  # type: ignore
            except Exception:  # noqa: BLE001
                pass

        # Member run done — restore the human author context for the rest of
        # the leader's turn (the HITL / synthesis code below stamps nothing).
        if _member_author_tok is not None:
            try:
                from src.core.identity_context import reset_author_context
                reset_author_context(_member_author_tok)
            except Exception:  # noqa: BLE001
                pass
            _member_author_tok = None

        # Persist the member run as its OWN durable child session row (the
        # runner skips member-session saves, so we write it explicitly) — this
        # is what makes the delegation a navigable sub-agent session + sidebar
        # card rather than a dangling link.
        if member_agent_run_response is not None and not member_agent_run_response.is_paused:
            await _persist_member_child_session(session, member_agent, member_agent_run_response)
            # Finalize the child's live stream: the now-persisted row is
            # canonical, so tell the app the sub-agent's turn is done — it
            # reconciles to that transcript and settles the streaming bubble.
            if member_session_id and member_session_id != session.session_id:
                from src.stream.child_stream import emit_child_frame
                await emit_child_frame(member_session_id, "turn_complete")

        # Check if the member run is paused (HITL)
        if member_agent_run_response is not None and member_agent_run_response.is_paused:
            _propagate_member_pause(run_response, member_agent, member_agent_run_response)
            use_team_logger()
            _process_delegate_task_to_member(
                member_agent_run_response,
                member_agent,
                member_agent_task,  # type: ignore
                member_session_state_copy,  # type: ignore
            )
            yield f"Member '{member_agent.name}' requires human input before continuing."
            return

        if not stream:
            try:
                if member_agent_run_response.content is None and (  # type: ignore
                    member_agent_run_response.tools is None or len(member_agent_run_response.tools) == 0  # type: ignore
                ):
                    yield "No response from the member agent."
                elif isinstance(member_agent_run_response.content, str):  # type: ignore
                    if len(member_agent_run_response.content.strip()) > 0:  # type: ignore
                        yield member_agent_run_response.content  # type: ignore

                    # If the content is empty but we have tool calls
                    elif (
                        member_agent_run_response.tools is not None  # type: ignore
                        and len(member_agent_run_response.tools) > 0  # type: ignore
                    ):
                        yield ",".join([tool.result for tool in member_agent_run_response.tools if tool.result])  # type: ignore
                elif issubclass(type(member_agent_run_response.content), BaseModel):  # type: ignore
                    yield member_agent_run_response.content.model_dump_json(indent=2)  # type: ignore
                else:
                    import json

                    yield json.dumps(member_agent_run_response.content, indent=2)  # type: ignore
            except Exception as e:
                yield str(e)

        # Afterward, switch back to the team logger
        use_team_logger()

        _process_delegate_task_to_member(
            member_agent_run_response,
            member_agent,
            member_agent_task,  # type: ignore
            member_session_state_copy,  # type: ignore
        )

    # When the task should be delegated to all members
    def delegate_task_to_members(task: str) -> Iterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
        """
        Use this function to delegate a task to all the member agents and return a response.
        You must provide a clear and concise description of the task the member should achieve AND the expected output.

        Args:
            task (str): A clear and concise description of the task to send to member agents.
        Returns:
            str: The result of the delegated task.
        """
        from src.core._runner.utils.callables import get_resolved_members

        resolved_members = get_resolved_members(team, run_context) or []

        # Run all the members sequentially
        for _, member_agent in enumerate(resolved_members):
            member_agent_task, history = _setup_delegate_task_to_member(member_agent=member_agent, task=task)

            member_session_state_copy = copy(run_context.session_state)
            # Each broadcast member runs as its own child session.
            member_session_id = _member_run_session_id(session.session_id, member_agent)
            if stream:
                member_agent_run_response_stream = member_agent.run(
                    input=member_agent_task if not history else history,
                    user_id=user_id,
                    session_id=member_session_id,
                    session_state=member_session_state_copy,  # Send a copy to the agent
                    images=images,
                    videos=videos,
                    audio=audio,
                    files=files,
                    stream=True,
                    stream_events=stream_events or team.stream_member_events,
                    knowledge_filters=run_context.knowledge_filters
                    if not member_agent.knowledge_filters and member_agent.knowledge
                    else None,
                    debug_mode=debug_mode,
                    dependencies=run_context.dependencies,
                    add_dependencies_to_context=add_dependencies_to_context,
                    add_session_state_to_context=add_session_state_to_context,
                    metadata=run_context.metadata,
                    yield_run_output=True,
                )
                member_agent_run_response = None
                for member_agent_run_response_chunk in member_agent_run_response_stream:
                    # Do NOT break out of the loop, Iterator need to exit properly
                    if isinstance(member_agent_run_response_chunk, (TeamRunOutput, RunOutput)):
                        member_agent_run_response = member_agent_run_response_chunk  # type: ignore
                        continue  # Don't yield TeamRunOutput or RunOutput, only yield events

                    # Check if the run is cancelled
                    check_if_run_cancelled(member_agent_run_response_chunk)

                    # Yield the member event directly
                    member_agent_run_response_chunk.parent_run_id = member_agent_run_response_chunk.parent_run_id or (
                        run_response.run_id if run_response is not None else None
                    )
                    yield member_agent_run_response_chunk  # type: ignore

            else:
                member_agent_run_response = member_agent.run(  # type: ignore
                    input=member_agent_task if not history else history,
                    user_id=user_id,
                    session_id=member_session_id,
                    session_state=member_session_state_copy,  # Send a copy to the agent
                    images=images,
                    videos=videos,
                    audio=audio,
                    files=files,
                    stream=False,
                    knowledge_filters=run_context.knowledge_filters
                    if not member_agent.knowledge_filters and member_agent.knowledge
                    else None,
                    debug_mode=debug_mode,
                    dependencies=run_context.dependencies,
                    add_dependencies_to_context=add_dependencies_to_context,
                    add_session_state_to_context=add_session_state_to_context,
                    metadata=run_context.metadata,
                )

                check_if_run_cancelled(member_agent_run_response)  # type: ignore

            # Check if the member run is paused (HITL)
            if member_agent_run_response is not None and member_agent_run_response.is_paused:
                _propagate_member_pause(run_response, member_agent, member_agent_run_response)
                use_team_logger()
                _process_delegate_task_to_member(
                    member_agent_run_response,
                    member_agent,
                    member_agent_task,  # type: ignore
                    member_session_state_copy,  # type: ignore
                )
                yield f"Agent {member_agent.name}: Requires human input before continuing."
                continue

            if not stream:
                try:
                    if member_agent_run_response.content is None and (  # type: ignore
                        member_agent_run_response.tools is None or len(member_agent_run_response.tools) == 0  # type: ignore
                    ):
                        yield f"Agent {member_agent.name}: No response from the member agent."
                    elif isinstance(member_agent_run_response.content, str):  # type: ignore
                        if len(member_agent_run_response.content.strip()) > 0:  # type: ignore
                            yield f"Agent {member_agent.name}: {member_agent_run_response.content}"  # type: ignore
                        elif (
                            member_agent_run_response.tools is not None and len(member_agent_run_response.tools) > 0  # type: ignore
                        ):
                            yield f"Agent {member_agent.name}: {','.join([tool.result for tool in member_agent_run_response.tools])}"  # type: ignore
                    elif issubclass(type(member_agent_run_response.content), BaseModel):  # type: ignore
                        yield f"Agent {member_agent.name}: {member_agent_run_response.content.model_dump_json(indent=2)}"  # type: ignore
                    else:
                        import json

                        yield f"Agent {member_agent.name}: {json.dumps(member_agent_run_response.content, indent=2)}"  # type: ignore
                except Exception as e:
                    yield f"Agent {member_agent.name}: Error - {str(e)}"

            _process_delegate_task_to_member(
                member_agent_run_response,
                member_agent,
                member_agent_task,  # type: ignore
                member_session_state_copy,  # type: ignore
            )

        # After all the member runs, switch back to the team logger
        use_team_logger()

    # When the task should be delegated to all members
    async def adelegate_task_to_members(task: str) -> AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
        """Use this function to delegate a task to all the member agents and return a response.
        You must provide a clear and concise description of the task to send to member agents.

        Args:
            task (str): A clear and concise description of the task to send to member agents.
        Returns:
            str: The result of the delegated task.
        """
        from src.core._runner.utils.callables import get_resolved_members

        resolved_members = get_resolved_members(team, run_context) or []

        if stream:
            # Concurrent streaming: launch each member as a streaming worker and merge events
            done_marker = object()
            queue: "asyncio.Queue[Union[RunOutputEvent, TeamRunOutputEvent, str, object]]" = asyncio.Queue()

            async def stream_member(agent: Union[Agent, "Team"]) -> None:
                member_agent_task, history = _setup_delegate_task_to_member(member_agent=agent, task=task)  # type: ignore
                member_session_state_copy = copy(run_context.session_state)
                member_session_id = _member_run_session_id(session.session_id, agent)

                member_stream = agent.arun(  # type: ignore
                    input=member_agent_task if not history else history,
                    user_id=user_id,
                    session_id=member_session_id,
                    session_state=member_session_state_copy,  # Send a copy to the agent
                    images=images,
                    videos=videos,
                    audio=audio,
                    files=files,
                    stream=True,
                    stream_events=stream_events or team.stream_member_events,
                    debug_mode=debug_mode,
                    knowledge_filters=run_context.knowledge_filters
                    if not agent.knowledge_filters and agent.knowledge
                    else None,
                    dependencies=run_context.dependencies,
                    add_dependencies_to_context=add_dependencies_to_context,
                    add_session_state_to_context=add_session_state_to_context,
                    metadata=run_context.metadata,
                    yield_run_output=True,
                )
                member_agent_run_response = None
                try:
                    try:
                        async for member_agent_run_output_event in member_stream:
                            # Do NOT break out of the loop, AsyncIterator need to exit properly
                            if isinstance(member_agent_run_output_event, (TeamRunOutput, RunOutput)):
                                member_agent_run_response = member_agent_run_output_event  # type: ignore
                                continue  # Don't yield TeamRunOutput or RunOutput, only yield events

                            check_if_run_cancelled(member_agent_run_output_event)
                            member_agent_run_output_event.parent_run_id = (
                                member_agent_run_output_event.parent_run_id
                                or (run_response.run_id if run_response is not None else None)
                            )
                            await queue.put(member_agent_run_output_event)
                    finally:
                        # Check if the member run is paused (HITL)
                        if member_agent_run_response is not None and member_agent_run_response.is_paused:
                            _propagate_member_pause(run_response, agent, member_agent_run_response)
                            _process_delegate_task_to_member(
                                member_agent_run_response,
                                agent,
                                member_agent_task,  # type: ignore
                                member_session_state_copy,  # type: ignore
                            )
                            await queue.put(f"Agent {agent.name}: Requires human input before continuing.")
                        else:
                            _process_delegate_task_to_member(
                                member_agent_run_response,
                                agent,
                                member_agent_task,  # type: ignore
                                member_session_state_copy,  # type: ignore
                            )
                finally:
                    await queue.put(done_marker)

            # Initialize and launch all members
            tasks: List[asyncio.Task[None]] = []
            for member_agent in resolved_members:
                current_agent = member_agent
                _initialize_member(team, current_agent)
                tasks.append(asyncio.create_task(stream_member(current_agent)))

            # Drain queue until all members reported done
            completed = 0
            try:
                while completed < len(tasks):
                    item = await queue.get()
                    if item is done_marker:
                        completed += 1
                    else:
                        yield item  # type: ignore
            finally:
                # Ensure tasks do not leak on cancellation
                for t in tasks:
                    if not t.done():
                        t.cancel()
                # Await cancellation to suppress warnings
                for t in tasks:
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await t
        else:
            # Non-streaming concurrent run of members; collect results when done
            tasks = []
            for member_agent_index, member_agent in enumerate(resolved_members):
                current_agent = member_agent
                member_agent_task, history = _setup_delegate_task_to_member(member_agent=current_agent, task=task)

                async def run_member_agent(
                    member_agent=current_agent,
                    member_agent_task=member_agent_task,
                    history=history,
                    member_agent_index=member_agent_index,
                ) -> tuple[str, Optional[Union[Agent, "Team"]], Optional[Union[RunOutput, TeamRunOutput]]]:
                    member_session_state_copy = copy(run_context.session_state)
                    member_session_id = _member_run_session_id(session.session_id, member_agent)

                    member_agent_run_response = await member_agent.arun(
                        input=member_agent_task if not history else history,
                        user_id=user_id,
                        session_id=member_session_id,
                        session_state=member_session_state_copy,  # Send a copy to the agent
                        images=images,
                        videos=videos,
                        audio=audio,
                        files=files,
                        stream=False,
                        stream_events=stream_events,
                        debug_mode=debug_mode,
                        knowledge_filters=run_context.knowledge_filters
                        if not member_agent.knowledge_filters and member_agent.knowledge
                        else None,
                        dependencies=run_context.dependencies,
                        add_dependencies_to_context=add_dependencies_to_context,
                        add_session_state_to_context=add_session_state_to_context,
                        metadata=run_context.metadata,
                    )
                    check_if_run_cancelled(member_agent_run_response)

                    member_name = member_agent.name if member_agent.name else f"agent_{member_agent_index}"

                    # Check if the member run is paused (HITL) before processing
                    if member_agent_run_response is not None and member_agent_run_response.is_paused:
                        _process_delegate_task_to_member(
                            member_agent_run_response,
                            member_agent,
                            member_agent_task,  # type: ignore
                            member_session_state_copy,  # type: ignore
                        )
                        return (
                            f"Agent {member_name}: Requires human input before continuing.",
                            member_agent,
                            member_agent_run_response,
                        )

                    _process_delegate_task_to_member(
                        member_agent_run_response,
                        member_agent,
                        member_agent_task,  # type: ignore
                        member_session_state_copy,  # type: ignore
                    )

                    try:
                        if member_agent_run_response.content is None and (
                            member_agent_run_response.tools is None or len(member_agent_run_response.tools) == 0
                        ):
                            return (f"Agent {member_name}: No response from the member agent.", None, None)
                        elif isinstance(member_agent_run_response.content, str):
                            if len(member_agent_run_response.content.strip()) > 0:
                                return (f"Agent {member_name}: {member_agent_run_response.content}", None, None)
                            elif (
                                member_agent_run_response.tools is not None and len(member_agent_run_response.tools) > 0
                            ):
                                return (
                                    f"Agent {member_name}: {','.join([tool.result for tool in member_agent_run_response.tools])}",
                                    None,
                                    None,
                                )
                        elif issubclass(type(member_agent_run_response.content), BaseModel):
                            return (
                                f"Agent {member_name}: {member_agent_run_response.content.model_dump_json(indent=2)}",  # type: ignore
                                None,
                                None,
                            )
                        else:
                            import json

                            return (
                                f"Agent {member_name}: {json.dumps(member_agent_run_response.content, indent=2)}",
                                None,
                                None,
                            )
                    except Exception as e:
                        return (f"Agent {member_name}: Error - {str(e)}", None, None)

                    return (f"Agent {member_name}: No Response", None, None)

                tasks.append(run_member_agent)  # type: ignore

            results = await asyncio.gather(*[task() for task in tasks])  # type: ignore
            # Propagate pauses sequentially after all members complete
            for result_text, paused_agent, paused_run_response in results:
                if paused_agent is not None and paused_run_response is not None:
                    _propagate_member_pause(run_response, paused_agent, paused_run_response)
                yield result_text

        # After all the member runs, switch back to the team logger
        use_team_logger()

    if team.delegate_to_all_members:
        if async_mode:
            delegate_function = adelegate_task_to_members  # type: ignore
        else:
            delegate_function = delegate_task_to_members  # type: ignore

        delegate_func = Function.from_callable(delegate_function, name="delegate_task_to_members")
    else:
        if async_mode:
            delegate_function = adelegate_task_to_member  # type: ignore
        else:
            delegate_function = delegate_task_to_member  # type: ignore

        delegate_func = Function.from_callable(delegate_function, name="delegate_task_to_member")

    if team.respond_directly:
        delegate_func.stop_after_tool_call = True
        delegate_func.show_result = True

    return delegate_func


def add_to_knowledge(team: "Team", query: str, result: str) -> str:
    """Use this function to add information to the knowledge base for future use.

    Args:
        query (str): The query or topic to add.
        result (str): The actual content or information to store.

    Returns:
        str: A string indicating the status of the addition.
    """
    from src.core._runner.utils.callables import get_resolved_knowledge

    knowledge = get_resolved_knowledge(team, None)
    if knowledge is None:
        log_warning("Knowledge is not set, cannot add to knowledge")
        return "Knowledge is not set, cannot add to knowledge"

    insert_method = getattr(knowledge, "insert", None)
    if not callable(insert_method):
        log_warning("Knowledge base does not support adding content")
        return "Knowledge base does not support adding content"

    document_name = query.replace(" ", "_").replace("?", "").replace("!", "").replace(".", "")
    document_content = json.dumps({"query": query, "result": result})
    log_info(f"Adding document to Knowledge: {document_name}: {document_content}")
    from src.core._runner._stubs import TextReader

    insert_method(name=document_name, text_content=document_content, reader=TextReader())
    return "Successfully added to knowledge base"


def create_knowledge_search_tool(
    team: "Team",
    run_response: Optional[TeamRunOutput] = None,
    run_context: Optional[RunContext] = None,
    knowledge_filters: Optional[Union[Dict[str, Any], List[FilterExpr]]] = None,
    enable_agentic_filters: Optional[bool] = False,
    async_mode: bool = False,
) -> Function:
    """Create a unified search_knowledge_base tool for Team.

    Routes all knowledge searches through get_relevant_docs_from_knowledge(),
    which checks knowledge_retriever first and falls back to knowledge.search().
    """

    def _format_results(docs: Optional[List[Union[Dict[str, Any], str]]]) -> str:
        if not docs:
            return "No documents found"
        if team.references_format == "json":
            return json.dumps(docs, indent=2, default=str)
        else:
            import yaml

            return yaml.dump(docs, default_flow_style=False)

    def _track_references(docs: Optional[List[Union[Dict[str, Any], str]]], query: str, elapsed: float) -> None:
        if run_response is not None and docs:
            references = MessageReferences(
                query=query,
                references=docs,
                time=round(elapsed, 4),
            )
            if run_response.references is None:
                run_response.references = []
            run_response.references.append(references)

    def _resolve_filters(
        agentic_filters: Optional[List[Any]] = None,
    ) -> Optional[Union[Dict[str, Any], List[FilterExpr]]]:
        if agentic_filters:
            filters_dict: Dict[str, Any] = {}
            for filt in agentic_filters:
                if isinstance(filt, dict):
                    filters_dict.update(filt)
                elif hasattr(filt, "key") and hasattr(filt, "value"):
                    filters_dict[filt.key] = filt.value
            return get_agentic_or_user_search_filters(filters_dict, knowledge_filters)
        return knowledge_filters

    if enable_agentic_filters:

        def search_knowledge_base_with_filters(query: str, filters: Optional[List[KnowledgeFilter]] = None) -> str:
            """Use this function to search the knowledge base for information about a query.

            Args:
                query: The query to search for.
                filters (optional): The filters to apply to the search. This is a list of KnowledgeFilter objects.

            Returns:
                str: A string containing the response from the knowledge base.
            """
            retrieval_timer = Timer()
            retrieval_timer.start()
            try:
                docs = get_relevant_docs_from_knowledge(
                    team,
                    query=query,
                    filters=_resolve_filters(filters),
                    validate_filters=True,
                    run_context=run_context,
                )
            except Exception as e:
                log_warning(f"Knowledge search failed: {str(e)}")
                return f"Error searching knowledge base: {type(e).__name__}"
            _track_references(docs, query, retrieval_timer.elapsed)
            retrieval_timer.stop()
            log_debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")
            return _format_results(docs)

        async def asearch_knowledge_base_with_filters(
            query: str, filters: Optional[List[KnowledgeFilter]] = None
        ) -> str:
            """Use this function to search the knowledge base for information about a query.

            Args:
                query: The query to search for.
                filters (optional): The filters to apply to the search. This is a list of KnowledgeFilter objects.

            Returns:
                str: A string containing the response from the knowledge base.
            """
            retrieval_timer = Timer()
            retrieval_timer.start()
            try:
                docs = await aget_relevant_docs_from_knowledge(
                    team,
                    query=query,
                    filters=_resolve_filters(filters),
                    validate_filters=True,
                    run_context=run_context,
                )
            except Exception as e:
                log_warning(f"Knowledge search failed: {str(e)}")
                return f"Error searching knowledge base: {type(e).__name__}"
            _track_references(docs, query, retrieval_timer.elapsed)
            retrieval_timer.stop()
            log_debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")
            return _format_results(docs)

        if async_mode:
            return Function.from_callable(asearch_knowledge_base_with_filters, name="search_knowledge_base")
        return Function.from_callable(search_knowledge_base_with_filters, name="search_knowledge_base")

    else:

        def search_knowledge_base(query: str) -> str:
            """Use this function to search the knowledge base for information about a query.

            Args:
                query: The query to search for.

            Returns:
                str: A string containing the response from the knowledge base.
            """
            retrieval_timer = Timer()
            retrieval_timer.start()
            try:
                docs = get_relevant_docs_from_knowledge(
                    team,
                    query=query,
                    filters=knowledge_filters,
                    run_context=run_context,
                )
            except Exception as e:
                log_warning(f"Knowledge search failed: {str(e)}")
                return f"Error searching knowledge base: {type(e).__name__}"
            _track_references(docs, query, retrieval_timer.elapsed)
            retrieval_timer.stop()
            log_debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")
            return _format_results(docs)

        async def asearch_knowledge_base(query: str) -> str:
            """Use this function to search the knowledge base for information about a query.

            Args:
                query: The query to search for.

            Returns:
                str: A string containing the response from the knowledge base.
            """
            retrieval_timer = Timer()
            retrieval_timer.start()
            try:
                docs = await aget_relevant_docs_from_knowledge(
                    team,
                    query=query,
                    filters=knowledge_filters,
                    run_context=run_context,
                )
            except Exception as e:
                log_warning(f"Knowledge search failed: {str(e)}")
                return f"Error searching knowledge base: {type(e).__name__}"
            _track_references(docs, query, retrieval_timer.elapsed)
            retrieval_timer.stop()
            log_debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")
            return _format_results(docs)

        if async_mode:
            return Function.from_callable(asearch_knowledge_base, name="search_knowledge_base")
        return Function.from_callable(search_knowledge_base, name="search_knowledge_base")


def get_relevant_docs_from_knowledge(
    team: "Team",
    query: str,
    num_documents: Optional[int] = None,
    filters: Optional[Union[Dict[str, Any], List[FilterExpr]]] = None,
    validate_filters: bool = False,
    run_context: Optional[RunContext] = None,
    **kwargs,
) -> Optional[List[Union[Dict[str, Any], str]]]:
    """Return a list of references from the knowledge base"""
    from src.core._runner._stubs import Document
    from src.core._runner.utils.callables import get_resolved_knowledge

    knowledge = get_resolved_knowledge(team, run_context)

    # Extract dependencies from run_context if available
    dependencies = run_context.dependencies if run_context else None

    if num_documents is None and knowledge is not None:
        num_documents = getattr(knowledge, "max_results", None)

    # Validate the filters against known valid filter keys
    if knowledge is not None and filters is not None and validate_filters:
        validate_filters_method = getattr(knowledge, "validate_filters", None)
        if callable(validate_filters_method):
            valid_filters, invalid_keys = validate_filters_method(filters)

            # Warn about invalid filter keys
            if invalid_keys:
                log_warning(f"Invalid filter keys provided: {invalid_keys}. These filters will be ignored.")

                # Only use valid filters
                filters = valid_filters
                if not filters:
                    log_warning("No valid filters remain after validation. Search will proceed without filters.")

            if invalid_keys == [] and valid_filters == {}:
                log_debug("No valid filters provided. Search will proceed without filters.")
                filters = None

    if team.knowledge_retriever is not None and callable(team.knowledge_retriever):
        from inspect import signature

        try:
            sig = signature(team.knowledge_retriever)
            knowledge_retriever_kwargs: Dict[str, Any] = {}
            if "team" in sig.parameters:
                knowledge_retriever_kwargs = {"team": team}
            if "filters" in sig.parameters:
                knowledge_retriever_kwargs["filters"] = filters
            if "run_context" in sig.parameters:
                knowledge_retriever_kwargs["run_context"] = run_context
            elif "dependencies" in sig.parameters:
                # Backward compatibility: support dependencies parameter
                knowledge_retriever_kwargs["dependencies"] = dependencies
            knowledge_retriever_kwargs.update({"query": query, "num_documents": num_documents, **kwargs})
            return team.knowledge_retriever(**knowledge_retriever_kwargs)
        except Exception as e:
            log_warning(f"Knowledge retriever failed: {str(e)}")
            raise e
    # Use knowledge protocol's retrieve method
    try:
        if knowledge is None:
            return None

        # Use protocol retrieve() method if available
        retrieve_fn = getattr(knowledge, "retrieve", None)
        if not callable(retrieve_fn):
            log_debug("Knowledge does not implement retrieve()")
            return None

        if num_documents is None:
            num_documents = getattr(knowledge, "max_results", 10)

        log_debug(f"Retrieving from knowledge base with filters: {filters}")
        relevant_docs: List[Document] = retrieve_fn(query=query, max_results=num_documents, filters=filters)

        if not relevant_docs or len(relevant_docs) == 0:
            log_debug("No relevant documents found for query")
            return None

        return [doc.to_dict() for doc in relevant_docs]
    except Exception as e:
        log_warning(f"Error retrieving from knowledge base: {str(e)}")
        raise e


async def aget_relevant_docs_from_knowledge(
    team: "Team",
    query: str,
    num_documents: Optional[int] = None,
    filters: Optional[Union[Dict[str, Any], List[FilterExpr]]] = None,
    validate_filters: bool = False,
    run_context: Optional[RunContext] = None,
    **kwargs,
) -> Optional[List[Union[Dict[str, Any], str]]]:
    """Get relevant documents from knowledge base asynchronously."""
    from src.core._runner._stubs import Document
    from src.core._runner.utils.callables import get_resolved_knowledge

    knowledge = get_resolved_knowledge(team, run_context)

    # Extract dependencies from run_context if available
    dependencies = run_context.dependencies if run_context else None

    if num_documents is None and knowledge is not None:
        num_documents = getattr(knowledge, "max_results", None)

    # Validate the filters against known valid filter keys
    if knowledge is not None and filters is not None and validate_filters:
        avalidate_filters_method = getattr(knowledge, "avalidate_filters", None)
        if callable(avalidate_filters_method):
            valid_filters, invalid_keys = await avalidate_filters_method(filters)

            # Warn about invalid filter keys
            if invalid_keys:
                log_warning(f"Invalid filter keys provided: {invalid_keys}. These filters will be ignored.")

                # Only use valid filters
                filters = valid_filters
                if not filters:
                    log_warning("No valid filters remain after validation. Search will proceed without filters.")

            if invalid_keys == [] and valid_filters == {}:
                log_debug("No valid filters provided. Search will proceed without filters.")
                filters = None

    if team.knowledge_retriever is not None and callable(team.knowledge_retriever):
        from inspect import isawaitable, signature

        try:
            sig = signature(team.knowledge_retriever)
            knowledge_retriever_kwargs: Dict[str, Any] = {}
            if "team" in sig.parameters:
                knowledge_retriever_kwargs = {"team": team}
            if "filters" in sig.parameters:
                knowledge_retriever_kwargs["filters"] = filters
            if "run_context" in sig.parameters:
                knowledge_retriever_kwargs["run_context"] = run_context
            elif "dependencies" in sig.parameters:
                # Backward compatibility: support dependencies parameter
                knowledge_retriever_kwargs["dependencies"] = dependencies
            knowledge_retriever_kwargs.update({"query": query, "num_documents": num_documents, **kwargs})

            result = team.knowledge_retriever(**knowledge_retriever_kwargs)

            if isawaitable(result):
                result = await result

            return result
        except Exception as e:
            log_warning(f"Knowledge retriever failed: {str(e)}")
            raise e

    # Use knowledge protocol's retrieve method
    try:
        if knowledge is None:
            return None

        # Use protocol aretrieve() or retrieve() method if available
        aretrieve_fn = getattr(knowledge, "aretrieve", None)
        retrieve_fn = getattr(knowledge, "retrieve", None)

        if not callable(aretrieve_fn) and not callable(retrieve_fn):
            log_debug("Knowledge does not implement retrieve()")
            return None

        if num_documents is None:
            num_documents = getattr(knowledge, "max_results", 10)

        log_debug(f"Retrieving from knowledge base with filters: {filters}")

        if callable(aretrieve_fn):
            relevant_docs: List[Document] = await aretrieve_fn(query=query, max_results=num_documents, filters=filters)
        elif callable(retrieve_fn):
            relevant_docs = retrieve_fn(query=query, max_results=num_documents, filters=filters)
        else:
            return None

        if not relevant_docs or len(relevant_docs) == 0:
            log_debug("No relevant documents found for query")
            return None

        return [doc.to_dict() for doc in relevant_docs]
    except Exception as e:
        log_warning(f"Error retrieving from knowledge base: {str(e)}")
        raise e
