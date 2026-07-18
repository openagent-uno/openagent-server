"""``OpenAgentACPAgent`` — the ACP :class:`acp.Agent` that fronts OpenAgent.

Reuses the exact machinery ``POST /api/chat`` uses: a
:class:`~src.stream.session.StreamSession` in ``profile="batched"`` built on
the process's single OpenAgent :class:`~src.core.agent.Agent`. Each ACP
session owns one StreamSession, kept alive across prompts so conversation
history accumulates — the same lifetime model as the chat endpoint's
per-``(client_id, session_id)`` cache.

Turn handling is NOT reinvented here: ``prompt`` pushes a ``TextFinal`` onto
the session's inbound bus (like :class:`~src.stream.channel.BatchedChannel`)
and streams the resulting ``session.outbound`` events out as ACP
``session_update`` notifications via :func:`src.acp.events.stream_turn`.
``cancel`` injects an ``Interrupt``, which the session's dispatch loop turns
into a cooperative turn cancel.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import acp
from acp.exceptions import RequestError
from acp.schema import AgentCapabilities, Implementation, PromptCapabilities

import src
from src.acp.events import stream_turn
from src.stream.events import Interrupt, TextFinal, now_ms
from src.stream.session import StreamSession

logger = logging.getLogger(__name__)


class _SessionState:
    """Per-ACP-session bookkeeping: the live StreamSession + a cancel flag."""

    __slots__ = ("session", "cancel_requested")

    def __init__(self, session: StreamSession) -> None:
        self.session = session
        self.cancel_requested = False


def _extract_text(prompt: Any) -> str:
    """Join the text of every text content block in an ACP prompt.

    Blocks arrive as typed pydantic models (``TextContentBlock`` etc.); handle
    plain dicts too for robustness. Non-text blocks (image/audio/resource) are
    ignored — v1 advertises no image capability.
    """
    parts: list[str] = []
    for block in prompt or ():
        btype = getattr(block, "type", None)
        if btype is None and isinstance(block, dict):
            btype = block.get("type")
        if btype != "text":
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


class OpenAgentACPAgent(acp.Agent):
    """ACP agent surface over a single OpenAgent :class:`~src.core.agent.Agent`."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._conn: acp.Client | None = None
        self._sessions: dict[str, _SessionState] = {}

    # ── connection lifecycle ─────────────────────────────────────────────

    def on_connect(self, conn: acp.Client) -> None:
        """Capture the reverse connection used to push ``session_update``s."""
        self._conn = conn
        logger.info("acp: client connected")

    # ── handshake ────────────────────────────────────────────────────────

    async def initialize(
        self,
        protocol_version: int | None = None,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> acp.InitializeResponse:
        name = getattr(client_info, "name", None) or "unknown"
        logger.info("acp: initialize from %s (protocol v%s)", name, protocol_version)
        return acp.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(
                name="openagent",
                version=getattr(src, "__version__", "unknown"),
            ),
            agent_capabilities=AgentCapabilities(
                # v1 has no image/embedded-context/fs surface.
                prompt_capabilities=PromptCapabilities(image=False),
            ),
        )

    async def authenticate(
        self, method_id: str | None = None, **kwargs: Any
    ) -> acp.AuthenticateResponse | None:
        # ACP is stdio-only, local-trust — no auth methods advertised, so this
        # is a no-op that simply acknowledges the (unexpected) call.
        logger.debug("acp: authenticate(method_id=%s) — no-op", method_id)
        return None

    # ── session lifecycle ────────────────────────────────────────────────

    async def new_session(
        self,
        cwd: str | None = None,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> acp.NewSessionResponse:
        session_id = f"acp-{uuid.uuid4().hex}"
        session = StreamSession(
            self._agent,
            client_id="acp",
            session_id=session_id,
            profile="batched",
            speak_enabled=False,
        )
        await session.start()
        self._sessions[session_id] = _SessionState(session)
        logger.info("acp: new session %s (cwd=%s)", session_id, cwd)
        return acp.NewSessionResponse(session_id=session_id)

    async def prompt(
        self, session_id: str, prompt: Any, **kwargs: Any
    ) -> acp.PromptResponse:
        state = self._sessions.get(session_id)
        if state is None:
            raise RequestError.invalid_params({"session_id": session_id})
        if self._conn is None:
            # Should never happen — run_agent calls on_connect before dispatch.
            raise RequestError.internal_error({"reason": "no client connection"})

        text = _extract_text(prompt)
        session = state.session
        state.cancel_requested = False

        # Discard any frames a prior (timed-out / cancelled) turn left
        # un-drained, exactly like BatchedChannel — otherwise this turn would
        # read the previous turn's tail.
        outbound = session.outbound
        while not outbound.empty():
            try:
                outbound.get_nowait()
            except Exception:  # asyncio.QueueEmpty
                break

        await session.push_in(
            TextFinal(
                session_id=session_id,
                seq=session.next_seq(),
                ts_ms=now_ms(),
                text=text,
                source="user_typed",
            )
        )

        stop_reason = await stream_turn(session, self._conn, session_id)
        if state.cancel_requested:
            stop_reason = "cancelled"
        return acp.PromptResponse(stop_reason=stop_reason)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            logger.debug("acp: cancel for unknown session %s — ignored", session_id)
            return
        state.cancel_requested = True
        session = state.session
        await session.push_in(
            Interrupt(
                session_id=session_id,
                seq=session.next_seq(),
                ts_ms=now_ms(),
                reason="manual",
            )
        )
        logger.info("acp: cancel requested for session %s", session_id)

    # ── teardown ─────────────────────────────────────────────────────────

    async def close_session(self, session_id: str, **kwargs: Any) -> Any:
        state = self._sessions.pop(session_id, None)
        if state is not None:
            try:
                await state.session.close()
            except Exception:  # noqa: BLE001 — teardown must not raise back
                logger.debug("acp: error closing session %s", session_id, exc_info=True)
        return None
