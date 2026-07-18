"""Translate OpenAgent ``session.outbound`` events → ACP session updates.

Mirrors the pump shape of :class:`~src.stream.channel.RealtimeChannel`
(drain ``session.outbound`` and forward each event over the transport) but
speaks ACP's ``session_update`` vocabulary instead of the wire codec:

    OpenAgent event      →  ACP session update
    ────────────────────────────────────────────────
    OutTextDelta         →  agent_message_chunk   (typewriter text)
    OutTextFinal         →  agent_message_chunk   (only if no deltas streamed)
    OutReasoning(active) →  agent_thought_chunk   (see caveat below)
    OutToolStatus        →  tool_call             (a completed transcript entry)
    OutError             →  agent_message_chunk   (surfaced inline)
    TurnComplete         →  stop the drain (turn end)

Everything else on the bus (audio spans, video frames, context reports,
compaction notices) is ignored — none of it maps onto ACP's coding-agent
surface in v1.

Caveat on ``OutReasoning``: OpenAgent's reasoning event is a pure boolean
"the model is working" flag — it carries NO reasoning text (by design; the
server refuses to pick the client's UI copy, see ``OutReasoning`` docstring).
ACP's ``agent_thought_chunk`` needs text, so there is nothing faithful to
forward. We emit a single neutral "Thinking…" thought on each rising edge
(deduped so a think→tool→think turn does not spam), purely so ACP clients get
a thought affordance. This is the one intentionally-synthetic mapping.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import acp

from src.stream.events import (
    OutError,
    OutReasoning,
    OutTextDelta,
    OutTextFinal,
    OutToolStatus,
    TurnComplete,
)

logger = logging.getLogger(__name__)


async def stream_turn(session: Any, conn: acp.Client, acp_session_id: str) -> str:
    """Drain ``session.outbound`` until :class:`TurnComplete`, forwarding each
    event to the ACP client as a ``session_update`` notification.

    Returns the ACP ``stop_reason`` implied by the stream — always
    ``"end_turn"`` here; the caller overrides it with ``"cancelled"`` when a
    cancel was requested for this session.
    """
    saw_delta = False
    reasoning_active = False

    while True:
        evt = await session.outbound.get()

        if isinstance(evt, OutTextDelta):
            if evt.text:
                saw_delta = True
                await conn.session_update(
                    acp_session_id, acp.update_agent_message_text(evt.text)
                )

        elif isinstance(evt, OutTextFinal):
            # The canonical full reply. Only forward it when no deltas were
            # streamed (some providers skip the typewriter path), so ACP
            # clients never render the answer twice.
            if not saw_delta and evt.text:
                await conn.session_update(
                    acp_session_id, acp.update_agent_message_text(evt.text)
                )

        elif isinstance(evt, OutReasoning):
            # Rising-edge only (see module docstring): one thought marker per
            # think phase, never a stream of them.
            if evt.active and not reasoning_active:
                reasoning_active = True
                await conn.session_update(
                    acp_session_id, acp.update_agent_thought_text("Thinking…")
                )
            elif not evt.active:
                reasoning_active = False

        elif isinstance(evt, OutToolStatus):
            # A fire-and-forget progress hint ("Using bash…", "Read done").
            # There is no paired completion event on the bus, so render each
            # as its own completed tool-call entry rather than leaving a
            # spinner hanging in the editor.
            if evt.text:
                await conn.session_update(
                    acp_session_id,
                    acp.start_tool_call(
                        f"oa-tool-{uuid.uuid4().hex[:8]}",
                        evt.text,
                        kind="other",
                        status="completed",
                    ),
                )

        elif isinstance(evt, OutError):
            # Soft error surfaced inline — stream the message so the user sees
            # it. The turn still ends via TurnComplete; stop_reason stays
            # "end_turn" (ACP has no generic error stop reason).
            if evt.text:
                await conn.session_update(
                    acp_session_id, acp.update_agent_message_text(evt.text)
                )

        elif isinstance(evt, TurnComplete):
            return "end_turn"

        # else: audio / video / context-report / compaction — ignored in v1.
