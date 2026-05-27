"""Rehydration parity — live wire and stored-run replay must agree.

The user-visible bug: a session that streamed as tool chips during a
live chat sometimes reappeared as plain "Using …" lines after the
universal app reloaded the session from ``/api/sessions/{id}/runs``
(or vice-versa). Delegated team members were inconsistent too —
shown as tool components in one render, flattened into assistant text
in another.

Root cause: the team-router stream path emitted ``f"⚙ {name}"`` plain
text via ``on_status``, while the rehydration endpoint reconstructed a
structured ``toolInfo`` chip from the stored ``ToolExecution`` row. The
two routes produced different shapes for the SAME underlying event.
``member_responses`` (delegated specialist content + tool calls) was
also dropped on the rehydration walk.

This module locks in the parity: both routes go through the shared
:mod:`src.models._tool_status` encoder which emits Agno's native
``ToolExecution.to_dict()`` shape, and the rehydration walk recurses
into ``member_responses`` so delegated content surfaces with the
specialist's own model attribution.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from ._framework import TestContext, test


# ── Shared helpers ──────────────────────────────────────────────────────


def _seed_agno_runs(db_path: str, session_id: str, runs: list[dict]) -> None:
    """Seed an ``agno_sessions`` row carrying ``runs`` as JSON."""
    import json as _json

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agno_sessions (
                session_id TEXT PRIMARY KEY,
                session_type TEXT,
                agent_id TEXT,
                team_id TEXT,
                workflow_id TEXT,
                user_id TEXT,
                session_data TEXT,
                agent_data TEXT,
                team_data TEXT,
                workflow_data TEXT,
                metadata TEXT,
                runs TEXT,
                summary TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO agno_sessions "
            "(session_id, session_type, user_id, runs, "
            " created_at, updated_at) "
            "VALUES (?, 'agent', 'openagent', ?, 100, 200)",
            (session_id, _json.dumps(runs)),
        )
        conn.commit()
    finally:
        conn.close()


@dataclass
class _FakeToolExec:
    """Minimal stand-in for Agno's ``ToolExecution`` dataclass.

    Only the attributes the shared status encoder reads
    (:mod:`src.models._tool_status`) are populated; everything else
    falls through to ``None`` so the encoder treats it as a real Agno
    object.
    """
    tool_name: str
    tool_call_id: str | None = None
    tool_args: dict | None = None
    result: str | None = None
    tool_call_error: bool | None = None


# ── Tests ────────────────────────────────────────────────────────────────


@test("rehydration_parity", "shared encoder produces identical envelopes for live + stored")
async def t_shared_encoder_parity(ctx: TestContext) -> None:
    """The live STATUS frame's ``text`` (JSON-encoded) and the
    rehydration endpoint's ``toolInfo`` MUST agree on every field the
    UI's parser inspects. Both go through the same encoder so the
    universal app's renderer branches identically.
    """
    from src.models._tool_status import (
        stored_tool_to_wire,
        tool_exec_to_wire_json,
    )

    live_exec = _FakeToolExec(
        tool_name="vault_write_note",
        tool_call_id="call_abc123",
        tool_args={"path": "Daily/today.md", "content": "..."},
        result="ok",
        tool_call_error=False,
    )
    live_json = tool_exec_to_wire_json(live_exec)
    assert live_json is not None
    live_payload = json.loads(live_json)
    assert live_payload["tool_name"] == "vault_write_note"
    assert live_payload["tool_call_id"] == "call_abc123"
    assert live_payload["tool_args"] == {"path": "Daily/today.md", "content": "..."}
    assert live_payload["tool_call_error"] is False
    assert live_payload["result"] == "ok"

    # Same tool, same args, after Agno persisted it as a ToolExecution
    # dict — the rehydration walker reconstructs the SAME payload.
    stored = {
        "tool_name": "vault_write_note",
        "tool_call_id": "call_abc123",
        "tool_args": {"path": "Daily/today.md", "content": "..."},
        "result": "ok",
        "tool_call_error": False,
    }
    stored_payload = stored_tool_to_wire(stored)
    assert stored_payload == live_payload, (
        f"Live and stored envelopes diverged:\n"
        f"  live   : {live_payload}\n"
        f"  stored : {stored_payload}"
    )


@test("rehydration_parity", "encoder surfaces tool_call_error+result for failed live frames")
async def t_encoder_error_path(ctx: TestContext) -> None:
    """Failed tools must carry ``tool_call_error=True`` AND ``result``
    set to the error text on the wire, so the UI's local phase
    derivation lights up the "✗ foo failed" chip in both live and
    rehydration views.
    """
    from src.models._tool_status import (
        stored_tool_to_wire,
        tool_exec_to_wire_json,
    )

    live_json = tool_exec_to_wire_json(
        _FakeToolExec(tool_name="shell_run", tool_call_id="x", tool_args={}),
        error_text="exit 1",
    )
    assert live_json is not None
    live = json.loads(live_json)
    assert live["tool_call_error"] is True
    assert live["result"] == "exit 1"

    # Stored rows pass through verbatim — what Agno wrote is what the UI sees.
    stored = stored_tool_to_wire({
        "tool_name": "shell_run",
        "tool_call_id": "x",
        "tool_args": {},
        "tool_call_error": True,
        "result": None,
    })
    assert stored is not None
    assert stored["tool_call_error"] is True
    # The stored row only persists the bool — no error text. UI derives
    # phase="error" purely from tool_call_error.
    assert stored["result"] is None


@test("rehydration_parity", "stored_tool_to_wire is exact pass-through of Agno-native dict")
async def t_stored_pass_through(ctx: TestContext) -> None:
    """The rehydration encoder MUST NOT translate field names or drop
    keys — the UI consumes Agno's ``ToolExecution.to_dict()`` shape
    verbatim, including fields we don't render today (metrics,
    child_run_id, etc.). This locks the contract.
    """
    from src.models._tool_status import stored_tool_to_wire

    stored = {
        "tool_name": "x",
        "tool_call_id": "call_1",
        "tool_args": {"a": 1},
        "tool_call_error": False,
        "result": "ok",
        "metrics": {"input_tokens": 5},
        "child_run_id": "haiku-r1",
    }
    wire = stored_tool_to_wire(stored)
    assert wire == stored, (wire, stored)  # exact pass-through


@test("rehydration_parity", "_arun_agno_stream emits JSON status for tool start + completion + error")
async def t_dispatcher_stream_emits_json_status(ctx: TestContext) -> None:
    """Live team-mode turn: the dispatcher's tool-call frames must be
    structured JSON in Agno-native shape (so the UI's
    ``handleServerMessage`` parser hits the ``toolInfo`` branch)
    instead of the legacy ``f"⚙ {name}"`` plain text. Without this,
    the chip renders one way live and another way after rehydration.
    """
    from agno.run.team import (
        ToolCallCompletedEvent as TeamToolCallCompletedEvent,
        ToolCallErrorEvent as TeamToolCallErrorEvent,
        ToolCallStartedEvent as TeamToolCallStartedEvent,
        RunContentEvent as TeamRunContentEvent,
    )
    from src.models.dispatcher import _arun_agno_stream

    # The same ToolExecution mutates between start + completion in
    # real Agno — result is None at start, populated at done. Mirror
    # that since the wire shape is verbatim ``to_dict()``.
    delegate_tool = _FakeToolExec(
        tool_name="delegate_task_to_member",
        tool_call_id="call_delegate_1",
        tool_args={"member_id": "anthropic-claude-haiku", "task_description": "x"},
    )
    vault_tool = _FakeToolExec(
        tool_name="vault_read_note",
        tool_call_id="call_vault_1",
        tool_args={"path": "today.md"},
    )
    failed_tool = _FakeToolExec(
        tool_name="shell_run",
        tool_call_id="call_shell_1",
        tool_args={"cmd": "false"},
    )

    class _FakeRuntime:
        def arun(self, prompt, *, session_id, user_id, stream, stream_events=False):
            async def _iter():
                yield TeamToolCallStartedEvent(
                    tool=delegate_tool, run_id="r", session_id=session_id,
                )
                yield TeamRunContentEvent(
                    content="hi ", run_id="r", session_id=session_id,
                )
                delegate_tool.result = "haiku said hello"
                yield TeamToolCallCompletedEvent(
                    tool=delegate_tool, run_id="r", session_id=session_id,
                )
                yield TeamToolCallStartedEvent(
                    tool=vault_tool, run_id="r", session_id=session_id,
                )
                vault_tool.result = "note text"
                yield TeamToolCallCompletedEvent(
                    tool=vault_tool, run_id="r", session_id=session_id,
                )
                yield TeamToolCallStartedEvent(
                    tool=failed_tool, run_id="r", session_id=session_id,
                )
                yield TeamToolCallErrorEvent(
                    tool=failed_tool, run_id="r",
                    session_id=session_id, error="exit 1",
                )
                yield TeamRunContentEvent(
                    content="done", run_id="r", session_id=session_id,
                )
            return _iter()

    statuses: list[str] = []

    async def _on_status(s: str) -> None:
        statuses.append(s)

    deltas: list[str] = []
    async for d in _arun_agno_stream(
        _FakeRuntime(),
        prompt="x",
        session_id="sess-parity",
        user_id="openagent",
        on_status=_on_status,
        error_event="test.error",
    ):
        deltas.append(d)

    # Deltas pass through unchanged.
    assert deltas == ["hi ", "done"], deltas

    # Status frames — every one must parse as JSON in Agno-native shape.
    # Three tools × started + (completed | error) = 6 frames.
    assert len(statuses) == 6, (
        f"expected 6 status frames (2 per tool), got {len(statuses)}: {statuses}"
    )
    parsed = [json.loads(s) for s in statuses]
    # delegate_task_to_member: started (no result yet) → completed (result set)
    assert parsed[0]["tool_name"] == "delegate_task_to_member"
    assert parsed[0]["tool_call_id"] == "call_delegate_1"
    assert parsed[0]["tool_args"]["member_id"] == "anthropic-claude-haiku"
    # Started frame: tool_call_error must NOT be true (UI derives "running").
    assert not parsed[0].get("tool_call_error")
    assert parsed[1]["tool_name"] == "delegate_task_to_member"
    # Completed frame: result populated → UI derives "completed".
    assert parsed[1].get("result") == "haiku said hello"
    # vault_read_note: started → completed
    assert parsed[2]["tool_name"] == "vault_read_note"
    assert not parsed[2].get("tool_call_error")
    assert parsed[3]["tool_name"] == "vault_read_note"
    assert parsed[3].get("result") == "note text"
    # shell_run: started → error (error text rides in result, flag flipped)
    assert parsed[4]["tool_name"] == "shell_run"
    assert not parsed[4].get("tool_call_error")
    assert parsed[5]["tool_name"] == "shell_run"
    assert parsed[5]["tool_call_error"] is True
    assert parsed[5]["result"] == "exit 1"

    # None of the statuses are plain text.
    assert not any(s.startswith("⚙ ") for s in statuses), (
        f"legacy plain-text status leaked: {statuses}"
    )


@test("rehydration_parity", "rehydration walks member_responses and attributes specialist model")
async def t_rehydration_walks_member_responses(ctx: TestContext) -> None:
    """A stored Team run with one delegation must rehydrate as:
      user → tool(delegate_task_to_member) → assistant(specialist text,
      model=specialist) → assistant(leader synthesis, model=leader).

    Before the fix, ``member_responses`` was dropped and the
    specialist's text got buried inside the tool result. The chat UI
    then either showed nothing (when the toolInfo result wasn't
    expanded) or showed it with the wrong model badge.
    """
    from src.gateway.api.sessions import _expand_run_messages

    leader_model = "openai:gpt-4o-mini"
    specialist_model = "anthropic:claude-haiku-4-5"

    # Stored shape: a TeamRunOutput.to_dict() with one delegate tool
    # call and one member response. Matches Agno's actual on-disk shape
    # (see agno.run.team.TeamRunOutput.to_dict and
    # agno.models.response.ToolExecution.to_dict).
    team_run = {
        "run_id": "team-run-1",
        "team_id": "openagent-test",
        "status": "completed",
        "model": leader_model,
        "created_at": 1234,
        "messages": [
            {"role": "user", "content": "Write a haiku about the moon."},
            # Tool-call carrier message — empty content by design.
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_delegate_1", "type": "function",
                     "function": {"name": "delegate_task_to_member"}},
                ],
            },
            # Tool result — content is the specialist's text (Agno
            # stores it here for the leader to use).
            {
                "role": "tool",
                "content": "Silver disk above /\nThe night breathes in quiet light /\nMoonglow on still snow",
                "tool_call_id": "call_delegate_1",
                "name": "delegate_task_to_member",
            },
            # Leader's synthesis.
            {
                "role": "assistant",
                "content": "Here's your haiku:\n\nSilver disk above /\nThe night breathes in quiet light /\nMoonglow on still snow",
            },
        ],
        "tools": [
            {
                "tool_name": "delegate_task_to_member",
                "tool_call_id": "call_delegate_1",
                "tool_args": {
                    "member_id": "anthropic-claude-haiku-4-5",
                    "task_description": "Write a haiku about the moon.",
                },
                "result": "Silver disk above /\nThe night breathes in quiet light /\nMoonglow on still snow",
                "tool_call_error": False,
                "child_run_id": "haiku-member-run-1",
            },
        ],
        "member_responses": [
            {
                "run_id": "haiku-member-run-1",
                "agent_id": "anthropic-claude-haiku-4-5",
                "status": "completed",
                "model": specialist_model,
                "messages": [
                    {"role": "user", "content": "Write a haiku about the moon."},
                    {
                        "role": "assistant",
                        "content": "Silver disk above /\nThe night breathes in quiet light /\nMoonglow on still snow",
                    },
                ],
                "tools": [],
            },
        ],
    }

    msg_counter = [0]
    out = _expand_run_messages(team_run, timestamp=1234, msg_counter=msg_counter)

    # Pluck just the fields that matter for parity (using Agno-native field names).
    summary = [
        (m["role"], (m.get("toolInfo", {}) or {}).get("tool_name"),
         bool((m.get("toolInfo", {}) or {}).get("tool_call_error")),
         m.get("model"))
        for m in out
    ]

    # Expected order:
    #   user → tool(delegate, no error) → assistant(specialist text,
    #   model=specialist) → assistant(leader synthesis, model=leader)
    #
    # The member run's first user-role message is the leader-generated
    # task prompt — an internal Agno artifact, not the human's input.
    # ``is_member_run=True`` on recursion filters it out so the chat
    # transcript shows only the user's REAL prompt at the top.
    assert summary[0] == ("user", None, False, None), summary
    assert summary[1] == ("tool", "delegate_task_to_member", False, None), summary
    assert summary[2] == ("assistant", None, False, specialist_model), summary
    assert summary[3] == ("assistant", None, False, leader_model), summary

    # toolInfo on the delegate row must carry Agno's native fields.
    delegate_msg = out[1]
    assert delegate_msg["toolInfo"]["tool_call_id"] == "call_delegate_1"
    assert delegate_msg["toolInfo"]["tool_args"]["member_id"] == "anthropic-claude-haiku-4-5"
    # Completion is implicit: result present + tool_call_error false.
    assert delegate_msg["toolInfo"]["result"]
    assert delegate_msg["toolInfo"]["tool_call_error"] is False


@test("rehydration_parity", "rehydration walks nested member tool calls (specialist's own MCP usage)")
async def t_rehydration_nested_tools(ctx: TestContext) -> None:
    """When the delegated specialist itself calls a tool (vault_read,
    shell, etc.) the chip MUST appear in the rehydrated transcript
    with the specialist's tool args + result.

    Live path: the IntermediateRunContentEvent + ToolCallStartedEvent
    fires for the member's own tools through ``_arun_agno_stream``.
    Rehydration path: those tools live in ``member_responses[*].tools``
    and the recursive expansion surfaces them at the right slot.
    """
    from src.gateway.api.sessions import _expand_run_messages

    team_run = {
        "run_id": "team-run-2",
        "team_id": "openagent-test",
        "status": "completed",
        "model": "openai:gpt-4o-mini",
        "created_at": 1234,
        "messages": [
            {"role": "user", "content": "What's in today's note?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_d1", "type": "function",
                             "function": {"name": "delegate_task_to_member"}}]},
            {"role": "tool", "content": "Note says: meeting at 3pm",
             "tool_call_id": "call_d1", "name": "delegate_task_to_member"},
            {"role": "assistant",
             "content": "Today's note says you have a meeting at 3pm."},
        ],
        "tools": [{
            "tool_name": "delegate_task_to_member",
            "tool_call_id": "call_d1",
            "tool_args": {"member_id": "claude-cli-vault-specialist",
                          "task_description": "Read today's note"},
            "result": "Note says: meeting at 3pm",
            "tool_call_error": False,
        }],
        "member_responses": [{
            "run_id": "vault-member-run",
            "agent_id": "claude-cli-vault-specialist",
            "status": "completed",
            "model": "anthropic:claude-opus-4-7",
            "messages": [
                {"role": "user", "content": "Read today's note"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_v1", "type": "function",
                                 "function": {"name": "vault_read_note"}}]},
                {"role": "tool",
                 "content": "meeting at 3pm",
                 "tool_call_id": "call_v1",
                 "name": "vault_read_note"},
                {"role": "assistant", "content": "Note says: meeting at 3pm"},
            ],
            "tools": [{
                "tool_name": "vault_read_note",
                "tool_call_id": "call_v1",
                "tool_args": {"path": "Daily/today.md"},
                "result": "meeting at 3pm",
                "tool_call_error": False,
            }],
        }],
    }

    msg_counter = [0]
    out = _expand_run_messages(team_run, timestamp=1234, msg_counter=msg_counter)

    tool_chips = [m for m in out if m["role"] == "tool"]
    # Two tool chips total: the leader's delegate + the specialist's vault read.
    assert len(tool_chips) == 2, (
        f"expected 2 tool chips, got {len(tool_chips)}: {tool_chips}"
    )
    assert tool_chips[0]["toolInfo"]["tool_name"] == "delegate_task_to_member"
    assert tool_chips[1]["toolInfo"]["tool_name"] == "vault_read_note"
    assert tool_chips[1]["toolInfo"]["tool_args"] == {"path": "Daily/today.md"}
    assert tool_chips[1]["toolInfo"]["result"] == "meeting at 3pm"

    # The specialist's assistant text carries the specialist's model.
    assistant_msgs = [m for m in out if m["role"] == "assistant"]
    assert any(m.get("model") == "anthropic:claude-opus-4-7" for m in assistant_msgs), (
        f"specialist's assistant text missing claude-opus model badge: {assistant_msgs}"
    )
    assert any(m.get("model") == "openai:gpt-4o-mini" for m in assistant_msgs), (
        f"leader's synthesis missing gpt-4o-mini model badge: {assistant_msgs}"
    )


@test("rehydration_parity", "rehydration endpoint returns envelope-matching toolInfo for a solo turn")
async def t_rehydration_solo_run_envelope(ctx: TestContext) -> None:
    """End-to-end on the DB layer: seed one solo RunOutput, fetch via
    list_session_runs + _expand_run_messages, and verify the produced
    toolInfo dict matches what a live STATUS frame would have carried.

    Hermetic — no gateway needed; we exercise the rehydration helpers
    directly against a seeded ``agno_sessions`` row.
    """
    from src.gateway.api.sessions import _expand_run_messages
    from src.memory.db import MemoryDB
    from src.models._tool_status import tool_exec_to_wire_json

    tmp_db = ctx.db_path.with_name(f"rehyd-{uuid.uuid4().hex[:8]}.db")
    try:
        session_id = "solo-run-session"
        run = {
            "run_id": "solo-run-1",
            "agent_id": "openagent-solo",
            "status": "completed",
            "model": "openai:gpt-4o",
            "created_at": 5000,
            "messages": [
                {"role": "user", "content": "Read note."},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_x1", "type": "function",
                                 "function": {"name": "vault_read_note"}}]},
                {"role": "tool", "content": "...", "tool_call_id": "call_x1",
                 "name": "vault_read_note"},
                {"role": "assistant", "content": "Here's the note."},
            ],
            "tools": [{
                "tool_name": "vault_read_note",
                "tool_call_id": "call_x1",
                "tool_args": {"path": "today.md"},
                "result": "meeting notes",
                "tool_call_error": False,
            }],
        }
        _seed_agno_runs(str(tmp_db), session_id, [run])

        db = MemoryDB(str(tmp_db))
        await db.connect()
        try:
            runs = await db.list_session_runs(session_id, limit=10)
            assert len(runs) == 1, runs
            msg_counter = [0]
            out = _expand_run_messages(
                runs[0], timestamp=5000, msg_counter=msg_counter,
            )
            tool_chip = next(m for m in out if m["role"] == "tool")
            stored_envelope = tool_chip["toolInfo"]

            # Reconstruct what the LIVE wire would have emitted for
            # the SAME tool execution and verify the rehydrated
            # envelope IS a superset of the live one (rehydration
            # preserves whatever Agno persisted; live has only the
            # subset _FakeToolExec exposes).
            live_envelope_json = tool_exec_to_wire_json(
                _FakeToolExec(
                    tool_name="vault_read_note",
                    tool_call_id="call_x1",
                    tool_args={"path": "today.md"},
                    result="meeting notes",
                    tool_call_error=False,
                ),
            )
            assert live_envelope_json is not None
            live_envelope = json.loads(live_envelope_json)
            # Field-by-field check on the keys the UI's renderer reads.
            for key in (
                "tool_name", "tool_call_id", "tool_args",
                "result", "tool_call_error",
            ):
                assert stored_envelope.get(key) == live_envelope.get(key), (
                    f"field drift on {key!r}:\n"
                    f"  live   : {live_envelope.get(key)!r}\n"
                    f"  stored : {stored_envelope.get(key)!r}"
                )

            # Assistant model attribution is the entry runtime id (no
            # member_responses on a solo run).
            assistant = next(m for m in out if m["role"] == "assistant" and m["text"])
            assert assistant["model"] == "openai:gpt-4o", assistant
        finally:
            await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("rehydration_parity", "cancelled runs stay excluded from the rehydrated transcript")
async def t_rehydration_skips_cancelled(ctx: TestContext) -> None:
    """Behaviour the old endpoint had — preserve it through the
    refactor. A cancelled run carries half-baked messages we don't
    want to replay.
    """
    from src.gateway.api.sessions import _expand_run_messages

    run = {
        "run_id": "cancelled-run",
        "status": "cancelled",
        "messages": [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "content": "starting..."},
        ],
    }
    msg_counter = [0]
    out = _expand_run_messages(run, timestamp=1234, msg_counter=msg_counter)
    assert out == [], out


@test("rehydration_parity", "tool chips emitted before assistant text in rehydration")
async def t_rehydration_preserves_order(ctx: TestContext) -> None:
    """Tool result rows in ``runs[].messages`` arrive in turn order
    (assistant tool_call carrier → tool result → assistant synthesis).
    The rehydration walk must preserve that order so the chat reads
    "user asks → tool fires → assistant replies", same as live.
    """
    from src.gateway.api.sessions import _expand_run_messages

    run = {
        "run_id": "ordered-run",
        "agent_id": "openagent-solo",
        "status": "completed",
        "model": "openai:gpt-4o",
        "created_at": 1234,
        "messages": [
            {"role": "user", "content": "Do A then B."},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_a", "type": "function",
                             "function": {"name": "tool_a"}}]},
            {"role": "tool", "content": "ok A", "tool_call_id": "call_a",
             "name": "tool_a"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_b", "type": "function",
                             "function": {"name": "tool_b"}}]},
            {"role": "tool", "content": "ok B", "tool_call_id": "call_b",
             "name": "tool_b"},
            {"role": "assistant", "content": "Both done."},
        ],
        "tools": [
            {"tool_name": "tool_a", "tool_call_id": "call_a",
             "tool_args": {}, "result": "ok A"},
            {"tool_name": "tool_b", "tool_call_id": "call_b",
             "tool_args": {}, "result": "ok B"},
        ],
    }
    msg_counter = [0]
    out = _expand_run_messages(run, timestamp=1234, msg_counter=msg_counter)
    roles = [m["role"] for m in out]
    tools_in_order = [m.get("toolInfo", {}).get("tool_name")
                      for m in out if m["role"] == "tool"]

    assert roles == ["user", "tool", "tool", "assistant"], roles
    assert tools_in_order == ["tool_a", "tool_b"], tools_in_order


@test(
    "rehydration_parity",
    "member-run user-role messages (leader-generated task prompts) are filtered out",
)
async def t_member_user_prompt_filtered(ctx: TestContext) -> None:
    """Bug: when the leader delegates via ``delegate_task_to_member``,
    the leader generates a synthetic prompt (e.g. "Read the file at
    /tmp/X using whatever tool…") and passes it to the specialist.
    Agno stores that prompt as a ``user``-role message inside the
    specialist's nested run under ``member_responses``.

    Before the fix, ``_expand_run_messages`` recursed into the member
    run and emitted that synthetic prompt as if the actual human had
    typed it — burying the user's REAL message ("answer these emails")
    under an internal artifact. The fix passes ``is_member_run=True``
    on recursion and skips user-role messages when that flag is set.

    Live streaming was unaffected because the live wire emits per-event
    deltas; the synthetic prompt never appeared on the wire.
    """
    from src.gateway.api.sessions import _expand_run_messages

    leader_run = {
        "run_id": "team-1",
        "model": "openai:gpt-4o-mini",
        "status": "completed",
        "messages": [
            # The HUMAN's actual prompt — must survive.
            {"role": "user", "content": "Reply to these support emails."},
            # Leader's delegate_task_to_member tool call.
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_delegate", "type": "function",
                             "function": {"name": "delegate_task_to_member"}}]},
            {"role": "tool", "content": "ok",
             "tool_call_id": "call_delegate",
             "name": "delegate_task_to_member"},
            {"role": "assistant", "content": "Replied to all three."},
        ],
        "tools": [
            {"tool_name": "delegate_task_to_member",
             "tool_call_id": "call_delegate",
             "tool_args": {"member_id": "specialist-openai-gpt-4o",
                           "task": "Read the file at /tmp/x.json …"},
             "result": "ok",
             "child_run_id": "member-1"},
        ],
        "member_responses": [
            {
                "run_id": "member-1",
                "agent_id": "specialist-openai-gpt-4o",
                "model": "openai:gpt-4o",
                "status": "completed",
                "messages": [
                    # This is the leader-generated synthetic prompt —
                    # must NOT appear in the rehydrated transcript.
                    {"role": "user",
                     "content": "Read the file at /tmp/x.json using "
                                "whatever tool you have available …"},
                    {"role": "assistant",
                     "content": "Here are the three replies …"},
                ],
                "tools": [],
            }
        ],
    }
    msg_counter = [0]
    out = _expand_run_messages(leader_run, timestamp=1234, msg_counter=msg_counter)

    user_texts = [m["text"] for m in out if m["role"] == "user"]
    assistant_texts = [m["text"] for m in out if m["role"] == "assistant"]

    assert user_texts == ["Reply to these support emails."], (
        f"the user's actual prompt is the ONLY user-role message that "
        f"should surface — leader-generated task prompts are internal "
        f"artifacts. Got: {user_texts!r}"
    )
    # Specialist's assistant content + leader's synthesis both survive.
    assert "Here are the three replies …" in assistant_texts, assistant_texts
    assert "Replied to all three." in assistant_texts, assistant_texts


@test(
    "rehydration_parity",
    "FRAMEWORK_SYSTEM_PROMPT instructs list-iteration parallelism",
)
async def t_prompt_list_iteration_hint(ctx: TestContext) -> None:
    """The leader needs an explicit hint that "N similar items → N
    parallel delegations" (not one delegation handling the whole list).
    Without this, the model collapses lists into a single delegation —
    losing per-item parallelism.
    """
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT

    text = FRAMEWORK_SYSTEM_PROMPT.lower()
    assert "list iteration is parallel by default" in text, (
        "the list-iteration hint must remain visible in the framework "
        "prompt so the leader fires N delegations for N items"
    )


__all__ = []
