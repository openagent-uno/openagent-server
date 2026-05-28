"""End-to-end unified-flow tests — multi-member delegation + live↔rehydration parity.

Locks down six cross-cutting properties of the v0.14 + coordinate-mode
stack that no single existing test module exercises end-to-end:

1. **Multi-member delegation** — the Team leader fires N parallel
   ``delegate_task_to_member`` tool calls in one synthetic turn and the
   dispatcher emits N structured tool-status envelopes (one per
   delegation) via ``on_status`` with the member_id surviving the
   round-trip.

2. **Live → rehydration parity** — a synthetic turn driven through
   ``_arun_runtime_stream`` is persisted to ``agno_sessions`` and replayed
   via the gateway's ``_expand_run_messages`` rehydration helper. The
   tool envelopes from the live wire and the rehydrated transcript
   must match field-for-field; member responses must carry the
   specialist's model badge.

3. **Coordinate-mode parallelism wiring** — ``TeamRouterProvider``
   builds the Team in ``TeamMode.coordinate``, and Agno upstream still
   gathers parallel function calls via ``asyncio.gather`` at the path
   the architecture relies on (``agno.models.base.arun_function_calls``).
   If a future Agno upgrade removes the gather, our parallelism
   assumption breaks silently — this test catches that.

4. **No-models short-circuit unaffected** — coordinate-mode shouldn't
   have shifted the zero-enabled-models error path (covered in
   ``test_regression_v014.t_generate_no_models_returns_error`` /
   ``t_stream_no_models_returns_error``); we explicitly sanity-check
   the dispatcher's short-circuit here in the e2e context.

5. **generate() emits symmetric (started, completed) tool frames** —
   the non-streaming path must produce the same two-phase wire shape as
   the streaming path so telegram / voice bridges that drive
   ``NativeProvider.generate()`` get a running-state chip before the
   completed one.

6. **Nested specialist tool calls surface as chips in BOTH live and
   rehydration** — the user's primary contract: when a team leader
   delegates to a specialist who then fires its OWN tools, the live
   wire and rehydrated transcript must both render the nested tools as
   their own chips in the same order. Agno emits these as
   ``agno.run.agent.ToolCallStartedEvent`` / ``ToolCallCompletedEvent``
   (member is an Agent, NOT a Team), passed through unchanged by the
   team's iterator when ``stream_member_events`` is True (default).

No live LLM calls — every Agno surface is faked via the same shapes
already exercised in ``test_team_router`` / ``test_rehydration_parity``.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from ._framework import TestContext, test


# ── Shared fakes ───────────────────────────────────────────────────────


@dataclass
class _FakeToolExec:
    """Minimal stand-in for Agno's ``ToolExecution`` dataclass.

    Mirrors the shape ``src.models._tool_status`` reads. Tests that
    need a delegate-tool call set ``tool_name='delegate_task_to_member'``
    and stash the member_id inside ``tool_args``. ``tool_call_error``
    defaults to ``False`` so the UI's local phase derivation flags
    these as non-error frames.
    """
    tool_name: str
    tool_call_id: str | None = None
    tool_args: dict | None = None
    result: str | None = None
    tool_call_error: bool | None = False


def _seed_agno_runs(db_path: str, session_id: str, runs: list[dict]) -> None:
    """Seed an ``agno_sessions`` row carrying ``runs`` as JSON.

    Copied from ``test_rehydration_parity`` to keep this module
    self-contained — the schema/shape is the contract test_rehydration_parity
    locks down, we just replay against it.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
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
            "INSERT OR REPLACE INTO sessions "
            "(session_id, session_type, user_id, runs, "
            " created_at, updated_at) "
            "VALUES (?, 'agent', 'openagent', ?, 100, 200)",
            (session_id, json.dumps(runs)),
        )
        conn.commit()
    finally:
        conn.close()


# ── Test 1: multi-member parallel delegation through the stream path ──


@test("e2e_unified_flow", "three parallel delegations emit three structured envelopes")
async def t_multi_member_delegation_emits_parallel_envelopes(ctx: TestContext) -> None:
    """Coordinate-mode leader fires THREE ``delegate_task_to_member`` tool
    calls in one synthetic turn. Each must surface to the WS pipe as its
    own structured tool-status envelope carrying the member_id from
    ``tool_args``; the final synthesized text contains all three member
    outputs. Mirrors the Test 1 contract from the prompt.

    The Agno surface is faked at the ``_arun_runtime_stream`` seam so we
    don't depend on a real Team's classifier or LLM round-trip.
    """
    from src.core._run_state.team import (
        RunContentEvent as TeamRunContentEvent,
        ToolCallCompletedEvent as TeamToolCallCompletedEvent,
        ToolCallStartedEvent as TeamToolCallStartedEvent,
    )
    from src.models.dispatcher import _arun_runtime_stream

    # Three delegations to three distinct specialists, each with its own
    # tool_call_id so the wire's tool_call_id field disambiguates them.
    # In real Agno, the SAME ToolExecution mutates: result is None at
    # start, populated at completion. Mirror that so the wire shape
    # (which is verbatim ``to_dict()``) carries the right per-phase
    # values the UI's local phase derivation reads.
    delegation_specs = [
        ("anthropic-claude-haiku", "summarise Q1", "Q1: revenue up 12%"),
        ("openai-gpt-5-coding", "scan tests", "tests: all green"),
        ("anthropic-claude-marketing", "draft tagline",
         "tagline: ship faster, sleep better"),
    ]
    delegations = [
        _FakeToolExec(
            tool_name="delegate_task_to_member",
            tool_call_id=f"call_delegate_{i+1}",
            tool_args={
                "member_id": member_id,
                "task_description": task,
            },
            result=None,
        )
        for i, (member_id, task, _result) in enumerate(delegation_specs)
    ]
    results_by_id = {
        f"call_delegate_{i+1}": r
        for i, (_, _, r) in enumerate(delegation_specs)
    }

    class _FakeTeamRuntime:
        def arun(self, prompt, *, session_id, user_id, stream, stream_events=False):
            async def _iter():
                # Coordinate mode: leader fires all three delegations before
                # any of them complete — mimics how Agno's
                # ``arun_function_calls`` opens by yielding tool_call_started
                # for every call (see agno/models/base.py:2515).
                for d in delegations:
                    yield TeamToolCallStartedEvent(
                        tool=d, run_id="r", session_id=session_id,
                    )
                # Then completion events as asyncio.gather harvests results.
                # Set ``result`` on the shared exec before yielding so
                # the completed envelope on the wire carries it (this
                # matches Agno's in-place mutation).
                for d in delegations:
                    d.result = results_by_id[d.tool_call_id]
                    yield TeamToolCallCompletedEvent(
                        tool=d, run_id="r", session_id=session_id,
                    )
                # Leader's synthesis.
                yield TeamRunContentEvent(
                    content="Q1: revenue up 12%. ", run_id="r",
                    session_id=session_id,
                )
                yield TeamRunContentEvent(
                    content="tests: all green. ", run_id="r",
                    session_id=session_id,
                )
                yield TeamRunContentEvent(
                    content="tagline: ship faster, sleep better.",
                    run_id="r", session_id=session_id,
                )
            return _iter()

    statuses: list[str] = []
    delegated_member_ids: list[str] = []

    async def _on_status(s: str) -> None:
        statuses.append(s)

    def _on_delegate(member_id: str) -> None:
        delegated_member_ids.append(member_id)

    deltas: list[str] = []
    async for d in _arun_runtime_stream(
        _FakeTeamRuntime(),
        prompt="Do Q1 summary, scan tests, and draft a launch tagline.",
        session_id="sess-parallel",
        user_id="openagent",
        on_status=_on_status,
        error_event="test.parallel_delegate",
        on_delegate=_on_delegate,
    ):
        deltas.append(d)

    # Three delegations × (started + completed) = 6 status frames.
    assert len(statuses) == 6, (
        f"expected 6 status frames (3 delegations × 2 phases), "
        f"got {len(statuses)}: {statuses}"
    )
    parsed = [json.loads(s) for s in statuses]

    # First three are started envelopes; per coordinate-mode semantics
    # all delegations are fired before any completes. The UI derives
    # phase locally from tool_call_error + result presence — started
    # frames have neither set.
    started = parsed[:3]
    completed = parsed[3:]
    for env in started:
        assert env["tool_name"] == "delegate_task_to_member", env
        assert not env.get("tool_call_error"), env
        assert "member_id" in env["tool_args"], env
    for env in completed:
        assert env["tool_name"] == "delegate_task_to_member", env
        # Completion frames carry the result + tool_call_error=False
        # (or absent). UI derives phase=completed from result presence.
        assert env.get("result"), env
        assert not env.get("tool_call_error"), env

    # Each delegation surfaces its OWN member_id on the wire — that's
    # the load-bearing field for the chat-UI's badge update path.
    started_member_ids = {e["tool_args"]["member_id"] for e in started}
    assert started_member_ids == {
        "anthropic-claude-haiku",
        "openai-gpt-5-coding",
        "anthropic-claude-marketing",
    }, started_member_ids

    # tool_call_id mirrors the call_id so duplicate tool names in one turn
    # stay distinguishable across the started/completed pair.
    started_ids = {e["tool_call_id"] for e in started}
    assert started_ids == {
        "call_delegate_1",
        "call_delegate_2",
        "call_delegate_3",
    }, started_ids

    # ``on_delegate`` fires once per delegation so ``_record_delegation``
    # can update the per-session member map (the chat UI's model badge).
    assert delegated_member_ids == [
        "anthropic-claude-haiku",
        "openai-gpt-5-coding",
        "anthropic-claude-marketing",
    ], delegated_member_ids

    # Final synthesized text contains all three member outputs.
    final_text = "".join(deltas)
    assert "Q1: revenue up 12%" in final_text, final_text
    assert "tests: all green" in final_text, final_text
    assert "ship faster, sleep better" in final_text, final_text


# ── Test 2: live → rehydration parity for a multi-tool turn ───────────


@test("e2e_unified_flow", "live wire envelopes match rehydrated envelopes field-for-field")
async def t_live_rehydration_parity_for_synthetic_run(ctx: TestContext) -> None:
    """Drive a synthetic turn (2 tool calls + 1 delegate + content deltas)
    through ``_arun_runtime_stream``, then persist the equivalent run to
    ``agno_sessions`` and replay it via ``_expand_run_messages``. The
    tool envelopes from both paths must match field-for-field, and the
    member's assistant content must surface with the specialist's
    model badge.
    """
    from src.core._run_state.team import (
        IntermediateRunContentEvent as TeamIRCE,
        RunContentEvent as TeamRunContentEvent,
        ToolCallCompletedEvent as TeamToolCallCompletedEvent,
        ToolCallStartedEvent as TeamToolCallStartedEvent,
    )
    from src.gateway.api.sessions import _expand_run_messages
    from src.memory.db import MemoryDB
    from src.models.dispatcher import _arun_runtime_stream

    leader_model = "openai:gpt-4o-mini"
    specialist_model = "anthropic:claude-opus-4-7"
    specialist_member_id = "anthropic-claude-opus-4-7"

    # In real Agno, the same ToolExecution mutates between start +
    # completion — ``result`` is None at start and populated at done.
    # Mirror that by leaving result=None and setting it before yielding
    # the completed event, so the wire shape (verbatim ToolExecution
    # to_dict) matches what Agno would emit.
    vault_tool = _FakeToolExec(
        tool_name="vault_read_note",
        tool_call_id="call_vault",
        tool_args={"path": "today.md"},
    )
    delegate_tool = _FakeToolExec(
        tool_name="delegate_task_to_member",
        tool_call_id="call_delegate",
        tool_args={
            "member_id": specialist_member_id,
            "task_description": "summarise today's note",
        },
    )
    shell_tool = _FakeToolExec(
        tool_name="shell_run",
        tool_call_id="call_shell",
        tool_args={"cmd": "date"},
    )

    # ── Live path ──────────────────────────────────────────────────
    class _FakeTeamRuntime:
        def arun(self, prompt, *, session_id, user_id, stream, stream_events=False):
            async def _iter():
                yield TeamToolCallStartedEvent(
                    tool=vault_tool, run_id="r", session_id=session_id,
                )
                vault_tool.result = "meeting at 3pm"
                yield TeamToolCallCompletedEvent(
                    tool=vault_tool, run_id="r", session_id=session_id,
                )
                yield TeamToolCallStartedEvent(
                    tool=delegate_tool, run_id="r", session_id=session_id,
                )
                # IntermediateRunContentEvent — the specialist's content
                # streaming back through the leader. Lives in the team
                # module (verified by test_regression_v014).
                yield TeamIRCE(
                    content="You have a meeting at 3pm today.",
                    run_id="r", session_id=session_id,
                )
                delegate_tool.result = "You have a meeting at 3pm today."
                yield TeamToolCallCompletedEvent(
                    tool=delegate_tool, run_id="r", session_id=session_id,
                )
                yield TeamToolCallStartedEvent(
                    tool=shell_tool, run_id="r", session_id=session_id,
                )
                shell_tool.result = "Tue 2026-05-27"
                yield TeamToolCallCompletedEvent(
                    tool=shell_tool, run_id="r", session_id=session_id,
                )
                yield TeamRunContentEvent(
                    content="Today: Tue 2026-05-27. You have a meeting at 3pm.",
                    run_id="r", session_id=session_id,
                )
            return _iter()

    live_statuses: list[str] = []

    async def _on_status(s: str) -> None:
        live_statuses.append(s)

    async for _ in _arun_runtime_stream(
        _FakeTeamRuntime(),
        prompt="What's in today's note and what's today's date?",
        session_id="sess-parity-e2e",
        user_id="openagent",
        on_status=_on_status,
        error_event="test.parity_e2e",
    ):
        pass

    live_envelopes = [json.loads(s) for s in live_statuses]
    # 3 tools × (start + done) = 6 envelopes.
    assert len(live_envelopes) == 6, live_envelopes

    # Keep just the "completed" envelopes for parity comparison —
    # rehydration only carries completed tool results (the stored
    # ``runs[].tools[]`` row IS the completion record). The UI's
    # phase derivation: result present + tool_call_error not set → done.
    live_done_by_id = {
        env["tool_call_id"]: env
        for env in live_envelopes
        if env.get("result") is not None and not env.get("tool_call_error")
    }
    assert set(live_done_by_id) == {
        "call_vault", "call_delegate", "call_shell",
    }, live_done_by_id

    # ── Rehydration path ───────────────────────────────────────────
    # Build the on-disk shape Agno would have written. This is the same
    # contract test_rehydration_parity walks; we recompose it here as
    # the parallel record of the live turn above.
    team_run = {
        "run_id": "parity-team-run",
        "team_id": "openagent-test",
        "status": "completed",
        "model": leader_model,
        "created_at": 1234,
        "messages": [
            {"role": "user",
             "content": "What's in today's note and what's today's date?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_vault", "type": "function",
                             "function": {"name": "vault_read_note"}}]},
            {"role": "tool", "content": "meeting at 3pm",
             "tool_call_id": "call_vault", "name": "vault_read_note"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_delegate", "type": "function",
                             "function": {"name": "delegate_task_to_member"}}]},
            {"role": "tool",
             "content": "You have a meeting at 3pm today.",
             "tool_call_id": "call_delegate",
             "name": "delegate_task_to_member"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_shell", "type": "function",
                             "function": {"name": "shell_run"}}]},
            {"role": "tool", "content": "Tue 2026-05-27",
             "tool_call_id": "call_shell", "name": "shell_run"},
            {"role": "assistant",
             "content": "Today: Tue 2026-05-27. You have a meeting at 3pm."},
        ],
        "tools": [
            {"tool_name": "vault_read_note", "tool_call_id": "call_vault",
             "tool_args": {"path": "today.md"}, "result": "meeting at 3pm",
             "tool_call_error": False},
            {"tool_name": "delegate_task_to_member",
             "tool_call_id": "call_delegate",
             "tool_args": {"member_id": specialist_member_id,
                           "task_description": "summarise today's note"},
             "result": "You have a meeting at 3pm today.",
             "tool_call_error": False,
             "child_run_id": "specialist-member-run"},
            {"tool_name": "shell_run", "tool_call_id": "call_shell",
             "tool_args": {"cmd": "date"}, "result": "Tue 2026-05-27",
             "tool_call_error": False},
        ],
        "member_responses": [
            {
                "run_id": "specialist-member-run",
                "agent_id": specialist_member_id,
                "status": "completed",
                "model": specialist_model,
                "messages": [
                    {"role": "user",
                     "content": "summarise today's note"},
                    {"role": "assistant",
                     "content": "You have a meeting at 3pm today."},
                ],
                "tools": [],
            },
        ],
    }

    # Persist + round-trip through MemoryDB.list_session_runs so the
    # JSON serialise/deserialise step is part of the parity check —
    # that's where field drift would slip in (e.g. tuple→list, int→str).
    tmp_db = ctx.db_path.with_name(f"e2e-parity-{uuid.uuid4().hex[:8]}.db")
    try:
        _seed_agno_runs(str(tmp_db), "sess-parity-e2e", [team_run])
        db = MemoryDB(str(tmp_db))
        await db.connect()
        try:
            stored_runs = await db.list_session_runs(
                "sess-parity-e2e", limit=10,
            )
        finally:
            await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass

    assert len(stored_runs) == 1, stored_runs
    msg_counter = [0]
    rehydrated = _expand_run_messages(
        stored_runs[0], timestamp=1234, msg_counter=msg_counter,
    )

    # Tool chips from rehydration, keyed by tool_call_id.
    rehydrated_done_by_id = {
        m["toolInfo"]["tool_call_id"]: m["toolInfo"]
        for m in rehydrated
        if m["role"] == "tool" and m.get("toolInfo")
    }
    assert set(rehydrated_done_by_id) == set(live_done_by_id), (
        f"tool_call_id sets differ:\n"
        f"  live       : {sorted(live_done_by_id)}\n"
        f"  rehydrated : {sorted(rehydrated_done_by_id)}"
    )

    # Field-for-field equality on the keys the UI's renderer reads.
    # Rehydration may surface additional Agno-native fields (metrics,
    # child_run_id, etc.) that the live path's _FakeToolExec doesn't
    # expose — assert the UI-load-bearing subset matches.
    ui_fields = ("tool_name", "tool_call_id", "tool_args", "result",
                 "tool_call_error")
    for tid, live_env in live_done_by_id.items():
        rehyd_env = rehydrated_done_by_id[tid]
        for key in ui_fields:
            assert live_env.get(key) == rehyd_env.get(key), (
                f"envelope drift for tool_call_id={tid!r} on {key!r}:\n"
                f"  live       : {live_env.get(key)!r}\n"
                f"  rehydrated : {rehyd_env.get(key)!r}"
            )

    # Specialist's content carries the specialist's model badge.
    assistant_msgs = [
        m for m in rehydrated if m["role"] == "assistant" and m["text"]
    ]
    specialist_msgs = [
        m for m in assistant_msgs
        if m.get("model") == specialist_model
    ]
    assert specialist_msgs, (
        f"no specialist-attributed assistant msg in rehydration: "
        f"{assistant_msgs}"
    )
    assert any(
        "meeting at 3pm" in m["text"] for m in specialist_msgs
    ), specialist_msgs

    # Leader's synthesis carries the leader's model badge.
    leader_msgs = [
        m for m in assistant_msgs if m.get("model") == leader_model
    ]
    assert leader_msgs, (
        f"no leader-attributed assistant msg in rehydration: "
        f"{assistant_msgs}"
    )


# ── Test 3: coordinate-mode parallelism wiring lock-down ──────────────


@test("e2e_unified_flow", "Team built in coordinate mode and Agno still gathers in parallel")
async def t_coordinate_mode_wired_and_agno_still_gathers(ctx: TestContext) -> None:
    """Two-part contract:

    1. ``TeamRouterProvider`` builds the Team with ``mode=coordinate`` so
       the leader can fire multiple ``delegate_task_to_member`` calls in
       a single turn (vs ``route`` which caps the leader at one).
    2. Agno upstream still gathers parallel function calls via
       ``asyncio.gather`` at ``agno.models.base.arun_function_calls``.
       This is the wire-level mechanism that makes the parallelism real
       — if an Agno upgrade ever serialises function calls, our
       multi-domain decomposition silently degrades to sequential.

    A failure here is a flag: either we built the Team wrong, or the
    Agno SDK changed and the parallelism architecture is no longer
    real. Both deserve a loud test failure.
    """
    import inspect

    from src.core._runner.team import Team, TeamMode

    from src.models.dispatcher import TeamRouterProvider

    # (1) TeamRouterProvider wires the Team in coordinate mode.
    providers = [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "gpt-4o-mini", "enabled": True,
                 "tier_hint": "fast leader"},
                {"id": 11, "model": "gpt-5-coding", "enabled": True,
                 "tier_hint": "best for coding"},
                {"id": 12, "model": "gpt-4o-vision", "enabled": True,
                 "tier_hint": "vision and image analysis"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    team = provider._ensure_runtime("sess-coord", system="MARKER")
    assert isinstance(team, Team), type(team).__name__
    assert team.mode == TeamMode.coordinate, (
        f"Team mode must be coordinate so leader can dispatch to "
        f"multiple members per turn; got {team.mode!r}. If this trips, "
        f"check that TeamRouterProvider._ensure_runtime still passes "
        f"mode=TeamMode.coordinate to Team(...)."
    )

    # (2) Agno's arun_function_calls still uses asyncio.gather. We
    # inspect the source rather than calling it (calling would require a
    # real Model + function executor; the source-level check is what
    # locks the contract).
    import src.models.providers.base as agno_base

    assert hasattr(agno_base.Model, "arun_function_calls"), (
        "agno.models.base.Model.arun_function_calls disappeared — the "
        "parallel-delegation architecture has lost its underlying "
        "primitive. Re-evaluate the coordinate-mode wiring."
    )
    src = inspect.getsource(agno_base.Model.arun_function_calls)
    assert "asyncio.gather" in src, (
        "agno.models.base.Model.arun_function_calls no longer uses "
        "asyncio.gather — parallel ``delegate_task_to_member`` calls "
        "would now serialise, defeating coordinate-mode's multi-member "
        "decomposition. Pin Agno or rebuild parallelism in OpenAgent."
    )


# ── Test 4: no-models short-circuit unaffected by coordinate-mode ─────


@test("e2e_unified_flow", "no-models short-circuit survives coordinate-mode change")
async def t_no_models_short_circuit_intact(ctx: TestContext) -> None:
    """The empty-catalog safety net (``ModelDispatcher.generate`` and
    ``stream`` returning a user-facing error instead of crashing) is
    covered in detail by ``test_regression_v014``. We re-assert here
    in the e2e module so a future coordinate-mode tweak that
    accidentally moves the empty-catalog branch is caught alongside
    the parallelism / parity locks.

    Two short-circuits, both must hold:
      - ``generate`` returns ``ModelResponse(stop_reason='error', content
        contains 'no model')``.
      - ``stream`` yields chunks whose joined text contains 'no model'.
    """
    from src.models.dispatcher import ModelDispatcher

    # Both paths share the same ``_resolve_entry_model`` short-circuit;
    # exercise each so a half-fix that only patches one is caught.
    router = ModelDispatcher(providers_config=[])

    resp = await router.generate(
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess-empty-1",
    )
    assert resp.stop_reason == "error", resp
    assert "no model" in resp.content.lower(), resp.content

    chunks: list[str] = []
    async for chunk in router.stream(
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess-empty-2",
    ):
        chunks.append(chunk)
    assert chunks, "stream must yield at least one chunk (the error text)"
    assert "no model" in "".join(chunks).lower(), chunks


# ── Test 5: generate() symmetrises tool frames with the streaming path ──


@test("e2e_unified_flow",
      "generate() emits (started, completed) frames per tool symmetric with stream")
async def t_generate_emits_symmetric_started_completed_per_tool(ctx: TestContext) -> None:
    """Non-streaming ``NativeProvider.generate()`` must emit a (started,
    completed) pair per tool so bridges that drive it (telegram, voice)
    see the same wire shape as the streaming path. Started frame has
    ``result=None`` (UI's phase derivation reads it as "running"),
    completed frame carries the final result.

    Mirrors the wire contract locked in by Test 1 + the
    ``_arun_runtime_stream`` filter set. Hermetic — we drive the provider's
    inner ``runner.arun`` with a fake response carrying one ToolExecution.
    """
    from src.models.native_provider import NativeProvider

    # Bypass __init__: it pulls heavy Agno/MCP wiring. We just need
    # _emit_agno_tool_status's bound logic + a fake runner.
    p = NativeProvider.__new__(NativeProvider)
    p.model = "openai:gpt-4o-mini"
    p._providers_config = {}
    # Stub the cost-mirror so generate() doesn't crash on missing catalog.
    p._compute_and_mirror_cost = lambda **_: 0.0  # type: ignore[attr-defined]

    # Fake runner.arun returns a response carrying one completed tool.
    completed_tool = _FakeToolExec(
        tool_name="vault_read_note",
        tool_call_id="call_v_gen",
        tool_args={"path": "today.md"},
        result="meeting at 3pm",
        tool_call_error=False,
    )

    class _FakeMetrics:
        # Empty dict-like; _metrics_to_dict will see no attrs.
        pass

    class _FakeResponse:
        content = "Today: meeting at 3pm."
        tools = [completed_tool]
        metrics = _FakeMetrics()

    class _FakeRunner:
        async def arun(self, *_a, **_k):
            return _FakeResponse()

    # _ensure_team / _ensure_agent return our fake runner.
    p._ensure_team = lambda **_: None  # type: ignore[attr-defined]
    p._ensure_agent = lambda **_: _FakeRunner()  # type: ignore[attr-defined]
    p._compatible_mcp_toolkits = lambda: ([], [])  # type: ignore[attr-defined]
    p._flatten_messages = lambda msgs: msgs[-1]["content"]  # type: ignore[attr-defined]
    p._metrics_to_dict = lambda _m: {}  # type: ignore[attr-defined]
    p._extract_metric = lambda *_a, **_k: 0  # type: ignore[attr-defined]
    p._is_session_corruption_error = lambda _e: False  # type: ignore[attr-defined]
    p._rewrite_provider_error_detail = lambda s: s  # type: ignore[attr-defined]

    statuses: list[str] = []

    async def _on_status(s: str) -> None:
        statuses.append(s)

    await p.generate(
        messages=[{"role": "user", "content": "Read today's note."}],
        on_status=_on_status,
        session_id="sess-gen-sym",
    )

    # The "Thinking..." preface is emitted before the run; drop it for
    # the tool-frame parity check (it's an unstructured ping).
    tool_frames = [s for s in statuses if s != "Thinking..."]
    assert len(tool_frames) == 2, (
        f"expected 2 tool frames (started + completed), got "
        f"{len(tool_frames)}: {tool_frames}"
    )
    started = json.loads(tool_frames[0])
    completed = json.loads(tool_frames[1])
    assert started["tool_name"] == "vault_read_note", started
    assert started["result"] is None, (
        f"started frame must have result=None for the UI's local phase "
        f"derivation; got {started!r}"
    )
    assert completed["tool_name"] == "vault_read_note", completed
    assert completed["result"] == "meeting at 3pm", completed
    assert not completed.get("tool_call_error"), completed


# ── Test 6: nested specialist tool calls surface in BOTH live + rehydration ──


@test("e2e_unified_flow",
      "delegation: live + rehydration both surface nested tool chips")
async def t_delegation_parity_with_nested_tools(ctx: TestContext) -> None:
    """The user's contract: when the team leader fires
    ``delegate_task_to_member`` and the specialist then fires its OWN
    tools (shell.exec, vault.search), BOTH the live stream and the
    rehydrated transcript must render the specialist's nested tools
    as their own chips — same order, same envelope.

    Audit finding: Agno's ``Team`` yields member events through its
    iterator when ``stream_member_events=True`` (default). A member
    Agent's tool call fires ``agno.run.agent.ToolCallStartedEvent`` /
    ``ToolCallCompletedEvent`` — exactly the types the dispatcher
    already includes in ``tool_start_event_types`` /
    ``tool_complete_event_types``. So the live path SHOULD already
    surface them; this test pins that contract.

    Hermetic: a fake team runtime yields the leader's delegate +
    nested member tool events in the same order Agno would.
    """
    from src.core._run_state.agent import (
        ToolCallCompletedEvent as AgentToolCallCompletedEvent,
        ToolCallStartedEvent as AgentToolCallStartedEvent,
    )
    from src.core._run_state.team import (
        IntermediateRunContentEvent as TeamIRCE,
        RunContentEvent as TeamRunContentEvent,
        ToolCallCompletedEvent as TeamToolCallCompletedEvent,
        ToolCallStartedEvent as TeamToolCallStartedEvent,
    )
    from src.gateway.api.sessions import _expand_run_messages
    from src.memory.db import MemoryDB
    from src.models.dispatcher import _arun_runtime_stream

    leader_model = "openai:gpt-4o-mini"
    specialist_model = "anthropic:claude-opus-4-7"
    specialist_member_id = "anthropic-claude-opus-4-7"

    # Team leader's delegate tool.
    delegate_tool = _FakeToolExec(
        tool_name="delegate_task_to_member",
        tool_call_id="call_delegate",
        tool_args={
            "member_id": specialist_member_id,
            "task_description": "investigate today's state",
        },
    )
    # Specialist's nested tools (these fire INSIDE the delegated arun,
    # so they emit Agent-typed events — NOT Team-typed).
    shell_tool = _FakeToolExec(
        tool_name="shell.exec",
        tool_call_id="call_shell_nested",
        tool_args={"cmd": "ls"},
    )
    vault_tool = _FakeToolExec(
        tool_name="vault.search",
        tool_call_id="call_vault_nested",
        tool_args={"query": "today"},
    )

    class _FakeTeamRuntime:
        def arun(self, prompt, *, session_id, user_id, stream, stream_events=False):
            async def _iter():
                # 1. Leader fires delegate (Team-typed event).
                yield TeamToolCallStartedEvent(
                    tool=delegate_tool, run_id="r", session_id=session_id,
                )
                # 2. Specialist's nested tools — Agent-typed events
                # because the member is an Agent. Agno's team iterator
                # passes them through unmodified when stream_member_events
                # is True (the default).
                yield AgentToolCallStartedEvent(
                    tool=shell_tool, run_id="r-member",
                    session_id=session_id,
                )
                shell_tool.result = "file1.txt\nfile2.txt"
                yield AgentToolCallCompletedEvent(
                    tool=shell_tool, run_id="r-member",
                    session_id=session_id,
                )
                yield AgentToolCallStartedEvent(
                    tool=vault_tool, run_id="r-member",
                    session_id=session_id,
                )
                vault_tool.result = "found 2 notes"
                yield AgentToolCallCompletedEvent(
                    tool=vault_tool, run_id="r-member",
                    session_id=session_id,
                )
                # 3. Specialist's text delta back through the leader's
                # iterator (IntermediateRunContentEvent).
                yield TeamIRCE(
                    content="found 2 notes, both from today.",
                    run_id="r", session_id=session_id,
                )
                # 4. Delegate completes.
                delegate_tool.result = "found 2 notes, both from today."
                yield TeamToolCallCompletedEvent(
                    tool=delegate_tool, run_id="r", session_id=session_id,
                )
                # 5. Leader's synthesis.
                yield TeamRunContentEvent(
                    content="Specialist confirmed: 2 notes from today.",
                    run_id="r", session_id=session_id,
                )
            return _iter()

    # ── Live path ──────────────────────────────────────────────────
    live_statuses: list[str] = []

    async def _on_status(s: str) -> None:
        live_statuses.append(s)

    async for _ in _arun_runtime_stream(
        _FakeTeamRuntime(),
        prompt="What's in today's vault?",
        session_id="sess-nested-parity",
        user_id="openagent",
        on_status=_on_status,
        error_event="test.nested_parity",
    ):
        pass

    live_envelopes = [json.loads(s) for s in live_statuses]
    # 3 tools × 2 phases = 6 frames (delegate started, shell start/done,
    # vault start/done, delegate done).
    assert len(live_envelopes) == 6, (
        f"expected 6 live frames (3 tools × 2 phases); got "
        f"{len(live_envelopes)}: {[e.get('tool_name') for e in live_envelopes]}"
    )
    live_tool_order = [e["tool_name"] for e in live_envelopes]
    # The wire order Agno produces: delegate-start → shell-start/done →
    # vault-start/done → delegate-done. Pins the user's contract that
    # nested chips appear BETWEEN the delegate start + done (the
    # delegate envelope wraps the specialist's work).
    assert live_tool_order == [
        "delegate_task_to_member",
        "shell.exec", "shell.exec",
        "vault.search", "vault.search",
        "delegate_task_to_member",
    ], live_tool_order

    # Each nested tool surfaces as ONE started (result=None) + ONE
    # completed envelope on the wire. UI derives phase locally.
    shell_envs = [e for e in live_envelopes if e["tool_name"] == "shell.exec"]
    assert shell_envs[0]["result"] is None, shell_envs[0]
    assert shell_envs[1]["result"] == "file1.txt\nfile2.txt", shell_envs[1]
    vault_envs = [e for e in live_envelopes if e["tool_name"] == "vault.search"]
    assert vault_envs[0]["result"] is None, vault_envs[0]
    assert vault_envs[1]["result"] == "found 2 notes", vault_envs[1]

    # ── Rehydration path ───────────────────────────────────────────
    team_run = {
        "run_id": "nested-team-run",
        "team_id": "openagent-test",
        "status": "completed",
        "model": leader_model,
        "created_at": 1234,
        "messages": [
            {"role": "user", "content": "What's in today's vault?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_delegate", "type": "function",
                             "function": {"name": "delegate_task_to_member"}}]},
            {"role": "tool",
             "content": "found 2 notes, both from today.",
             "tool_call_id": "call_delegate",
             "name": "delegate_task_to_member"},
            {"role": "assistant",
             "content": "Specialist confirmed: 2 notes from today."},
        ],
        "tools": [{
            "tool_name": "delegate_task_to_member",
            "tool_call_id": "call_delegate",
            "tool_args": {"member_id": specialist_member_id,
                          "task_description": "investigate today's state"},
            "result": "found 2 notes, both from today.",
            "tool_call_error": False,
            "child_run_id": "specialist-nested-run",
        }],
        "member_responses": [{
            "run_id": "specialist-nested-run",
            "agent_id": specialist_member_id,
            "status": "completed",
            "model": specialist_model,
            "messages": [
                {"role": "user", "content": "investigate today's state"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_shell_nested", "type": "function",
                                 "function": {"name": "shell.exec"}}]},
                {"role": "tool", "content": "file1.txt\nfile2.txt",
                 "tool_call_id": "call_shell_nested", "name": "shell.exec"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_vault_nested", "type": "function",
                                 "function": {"name": "vault.search"}}]},
                {"role": "tool", "content": "found 2 notes",
                 "tool_call_id": "call_vault_nested", "name": "vault.search"},
                {"role": "assistant",
                 "content": "found 2 notes, both from today."},
            ],
            "tools": [
                {"tool_name": "shell.exec", "tool_call_id": "call_shell_nested",
                 "tool_args": {"cmd": "ls"},
                 "result": "file1.txt\nfile2.txt",
                 "tool_call_error": False},
                {"tool_name": "vault.search",
                 "tool_call_id": "call_vault_nested",
                 "tool_args": {"query": "today"},
                 "result": "found 2 notes",
                 "tool_call_error": False},
            ],
        }],
    }

    tmp_db = ctx.db_path.with_name(f"e2e-nested-{uuid.uuid4().hex[:8]}.db")
    try:
        _seed_agno_runs(str(tmp_db), "sess-nested-parity", [team_run])
        db = MemoryDB(str(tmp_db))
        await db.connect()
        try:
            stored_runs = await db.list_session_runs(
                "sess-nested-parity", limit=10,
            )
        finally:
            await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass

    msg_counter = [0]
    rehydrated = _expand_run_messages(
        stored_runs[0], timestamp=1234, msg_counter=msg_counter,
    )

    # Rehydration produces tool chips for the three tools in the same
    # order as the live wire. Rehydration only carries completed rows
    # (each stored ToolExecution IS the completion record) — so we
    # compare against the COMPLETED frames from the live path.
    rehydrated_tool_order = [
        m["toolInfo"]["tool_name"]
        for m in rehydrated
        if m["role"] == "tool" and m.get("toolInfo")
    ]
    assert rehydrated_tool_order == [
        "delegate_task_to_member",
        "shell.exec",
        "vault.search",
    ], rehydrated_tool_order

    live_done = [
        e for e in live_envelopes
        if e.get("result") is not None and not e.get("tool_call_error")
    ]
    live_done_by_name = {e["tool_name"]: e for e in live_done}
    rehydrated_chips = {
        m["toolInfo"]["tool_name"]: m["toolInfo"]
        for m in rehydrated
        if m["role"] == "tool" and m.get("toolInfo")
    }
    for tool_name in ("delegate_task_to_member", "shell.exec", "vault.search"):
        live_env = live_done_by_name[tool_name]
        rehyd_env = rehydrated_chips[tool_name]
        for key in ("tool_name", "tool_call_id", "tool_args", "result",
                    "tool_call_error"):
            assert live_env.get(key) == rehyd_env.get(key), (
                f"envelope drift for {tool_name!r} on {key!r}:\n"
                f"  live       : {live_env.get(key)!r}\n"
                f"  rehydrated : {rehyd_env.get(key)!r}"
            )

    # The specialist's assistant text appears with the specialist's
    # model badge in the rehydrated transcript (the live path's
    # IntermediateRunContentEvent puts the same text in the leader's
    # delta stream but rehydration stores it under the member_response).
    assistant_msgs = [
        m for m in rehydrated if m["role"] == "assistant" and m["text"]
    ]
    specialist_msgs = [
        m for m in assistant_msgs if m.get("model") == specialist_model
    ]
    assert specialist_msgs, (
        f"no specialist-attributed assistant msg in rehydration: "
        f"{assistant_msgs}"
    )


# ── Test 7: non-image attachments route via Agno's native files= ──────


@test("e2e_unified_flow",
      "non-image attachments route via files= not prepend (clean stored user msg)")
async def t_attachments_native_files(ctx: TestContext) -> None:
    """Non-image attachments must reach Agno via the native ``files=``
    parameter on ``Team.arun`` / ``Agent.arun`` — NOT as a synthetic
    "The user attached files: …" prepend in the user's text.

    The prior behavior (prepend a file-info block) caused the team
    leader to paraphrase the synthetic path into delegation tasks
    like "Read the file at /tmp/X.json using whatever tool…". Routing
    through ``files=`` lets Agno propagate the attachment to members
    natively via ``delegate_task_to_member`` (see
    ``agno/team/_default_tools.py:713``).

    Hermetic: fake the Agno runtime at the ``_arun_runtime_stream`` seam
    and capture both ``input`` and ``files`` it receives. We assert the
    bare user text survived (no synthetic prepend) and that the
    ``files`` sequence carries one Agno ``File`` per non-image upload.
    """
    from src.stream.media import File as AgnoFile

    from src.core.agent import _build_runtime_media
    from src.models.dispatcher import _arun_runtime_stream

    def _files_only(attachments):
        """Pluck the files element from the (images, audios, videos, files) tuple."""
        result = _build_runtime_media(attachments)
        if not result:
            return None
        files_list = result[3]
        return files_list or None

    # ── 1. The converter agent.py uses returns four parallel lists; this
    # test pins the files slot specifically.
    atts = [
        {"type": "image", "filename": "chart.png", "path": "/tmp/chart.png"},
        {"type": "file",  "filename": "report.json", "path": "/tmp/report.json"},
        {"type": "file",  "filename": "notes.csv",   "path": "/tmp/notes.csv"},
    ]
    files = _files_only(atts)
    assert files is not None and len(files) == 2, (
        f"image must be dropped from files=; got {files!r}"
    )
    assert all(isinstance(f, AgnoFile) for f in files), files
    by_name = {f.filename: f for f in files}
    assert set(by_name) == {"report.json", "notes.csv"}, list(by_name)
    assert str(by_name["report.json"].filepath) == "/tmp/report.json"
    assert str(by_name["notes.csv"].filepath) == "/tmp/notes.csv"

    # Empty / image-only inputs → no files so callers can skip threading.
    assert _files_only(None) is None
    assert _files_only([]) is None
    assert _files_only(
        [{"type": "image", "filename": "x.png", "path": "/tmp/x.png"}]
    ) is None

    # ── 2. ``_arun_runtime_stream`` forwards files= to runtime.arun ──
    captured: dict[str, Any] = {}

    class _FakeRuntime:
        def arun(self, prompt, *, session_id, user_id, stream,
                 stream_events=False, files=None, **kwargs):
            captured["prompt"] = prompt
            captured["files"] = files
            captured["session_id"] = session_id
            async def _iter():
                from src.core._run_state.agent import RunContentEvent
                yield RunContentEvent(
                    content="ok", run_id="r", session_id=session_id,
                )
            return _iter()

    deltas: list[str] = []
    async for d in _arun_runtime_stream(
        _FakeRuntime(),
        prompt="what's in the report?",
        session_id="sess-files",
        user_id="openagent",
        on_status=None,
        error_event="test.files_routing",
        files=files,
    ):
        deltas.append(d)

    # The user's bare text survives — no synthetic prepend.
    assert captured["prompt"] == "what's in the report?", captured
    assert "The user attached files:" not in (captured["prompt"] or ""), (
        f"synthetic prepend leaked into Agno input: {captured['prompt']!r}"
    )
    # The Agno File sequence arrived natively on the ``files=`` kwarg.
    assert captured["files"] is not None, captured
    assert len(captured["files"]) == 2, captured
    routed_names = {f.filename for f in captured["files"]}
    assert routed_names == {"report.json", "notes.csv"}, routed_names


__all__: list[str] = []
