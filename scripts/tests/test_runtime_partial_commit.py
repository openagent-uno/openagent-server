"""Interrupted-turn history backfill — ``upsert_run`` + ``synth_interrupted_messages``.

When the user stops a turn the runner persists the run as ``status=CANCELLED``
with the user's question in ``input`` but no ``messages`` — and history skips
CANCELLED runs, so the stopped turn would vanish and the next turn ("continua")
would lose all context. ``Session.upsert_run`` backfills the user message and
re-marks the run COMPLETED so it stays in history. These tests pin that.
"""
from __future__ import annotations

import json
import time
import uuid

from ._framework import TestContext, TestSkip, test


def _imports():
    from src.memory.sessions.agent import AgentSession
    from src.memory.sessions.team import TeamSession
    from src.core._run_state.agent import RunOutput, RunInput
    from src.core._run_state.team import TeamRunOutput
    from src.core._run_state.base import RunStatus
    from src.models.providers.message import Message
    return AgentSession, TeamSession, RunOutput, RunInput, TeamRunOutput, RunStatus, Message


@test("runtime_partial_commit", "AgentSession.upsert_run backfills a cancelled turn into history")
async def t_agent_backfill(_ctx: TestContext) -> None:
    try:
        AgentSession, _T, RunOutput, RunInput, _TR, RunStatus, _M = _imports()
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    now = int(time.time())
    s = AgentSession(session_id="sid", agent_id="a", user_id="u", created_at=now, updated_at=now)
    s.upsert_run(RunOutput(
        run_id="r-1", agent_id="a", session_id="sid", user_id="u",
        input=RunInput(input_content="scrivi un poema sul mare"),
        status=RunStatus.cancelled,  # stopped mid-stream, no messages assembled
    ))
    run = s.runs[0]
    assert run.status == RunStatus.completed, run.status  # promoted so history keeps it
    roles = [(m.role, m.content) for m in (run.messages or [])]
    assert roles == [("user", "scrivi un poema sul mare")], roles


@test("runtime_partial_commit", "TeamSession.upsert_run backfills a cancelled turn into history")
async def t_team_backfill(_ctx: TestContext) -> None:
    try:
        _A, TeamSession, _R, RunInput, TeamRunOutput, RunStatus, _M = _imports()
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    now = int(time.time())
    s = TeamSession(session_id="sid", team_id="t", user_id="u", created_at=now, updated_at=now)
    s.upsert_run(TeamRunOutput(
        run_id="r-1", team_id="t", session_id="sid", user_id="u",
        input=RunInput(input_content="conta da 1 a 300"),
        status=RunStatus.cancelled,
    ))
    run = s.runs[0]
    assert run.status == RunStatus.completed, run.status
    roles = [(m.role, m.content) for m in (run.messages or [])]
    assert roles == [("user", "conta da 1 a 300")], roles


@test("runtime_partial_commit", "a completed run with messages is left untouched")
async def t_completed_untouched(_ctx: TestContext) -> None:
    try:
        AgentSession, _T, RunOutput, _RI, _TR, RunStatus, Message = _imports()
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    now = int(time.time())
    s = AgentSession(session_id="sid", agent_id="a", user_id="u", created_at=now, updated_at=now)
    original = [Message(role="user", content="q"), Message(role="assistant", content="a")]
    s.upsert_run(RunOutput(
        run_id="r-1", agent_id="a", session_id="sid", user_id="u",
        content="a", messages=list(original), status=RunStatus.completed,
    ))
    run = s.runs[0]
    assert run.status == RunStatus.completed
    assert [(m.role, m.content) for m in run.messages] == [("user", "q"), ("assistant", "a")]


@test("runtime_partial_commit", "synth is a no-op when there is no user input to recover")
async def t_synth_noop(_ctx: TestContext) -> None:
    try:
        _A, _T, RunOutput, _RI, _TR, RunStatus, _M = _imports()
        from src.memory.sessions._synth import synth_interrupted_messages
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    # No input → nothing to recover; the run (and a RUNNING in-flight run) is
    # left exactly as-is.
    assert synth_interrupted_messages(RunOutput(run_id="x", status=RunStatus.cancelled)) is None

    from src.memory.sessions.agent import AgentSession
    from src.core._run_state.agent import RunInput
    now = int(time.time())
    s = AgentSession(session_id="sid", agent_id="a", user_id="u", created_at=now, updated_at=now)
    # RUNNING (in-flight) with input but no messages must NOT be backfilled/promoted.
    s.upsert_run(RunOutput(
        run_id="r-1", agent_id="a", session_id="sid", user_id="u",
        input=RunInput(input_content="still going"), status=RunStatus.running,
    ))
    assert s.runs[0].status == RunStatus.running, s.runs[0].status
    assert not s.runs[0].messages


@test("runtime_partial_commit", "a stopped run that DID assemble messages is promoted to COMPLETED")
async def t_cancelled_with_messages_promoted(_ctx: TestContext) -> None:
    try:
        AgentSession, _T, RunOutput, _RI, _TR, RunStatus, Message = _imports()
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    now = int(time.time())
    s = AgentSession(session_id="sid", agent_id="a", user_id="u", created_at=now, updated_at=now)
    # Stopped late enough that the runner already assembled messages (incl. the
    # streamed partial). It must STILL be promoted so history keeps it.
    s.upsert_run(RunOutput(
        run_id="r-1", agent_id="a", session_id="sid", user_id="u",
        content="1\n2\n3", status=RunStatus.cancelled,
        messages=[Message(role="user", content="conta"), Message(role="assistant", content="1\n2\n3")],
    ))
    run = s.runs[0]
    assert run.status == RunStatus.completed, run.status
    assert [(m.role, m.content) for m in run.messages] == [("user", "conta"), ("assistant", "1\n2\n3")]


@test("runtime_partial_commit", "synth keeps the partial reply but skips the cancel sentinels")
async def t_synth_partial_and_sentinels(_ctx: TestContext) -> None:
    try:
        _A, _T, RunOutput, RunInput, _TR, RunStatus, _M = _imports()
        from src.memory.sessions._synth import synth_interrupted_messages
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    def shell(content):
        return RunOutput(run_id="rid", status=RunStatus.cancelled,
                         input=RunInput(input_content="scrivi una poesia"), content=content)

    # Real partial → kept as the assistant message.
    got = synth_interrupted_messages(shell("Tramonto sul mare\nQuando il giorno"))
    assert [(m.role, m.content) for m in got] == [
        ("user", "scrivi una poesia"), ("assistant", "Tramonto sul mare\nQuando il giorno")]
    # Sentinels (nothing was generated) → user only, no bogus assistant.
    assert [m.role for m in synth_interrupted_messages(shell("Run rid was cancelled"))] == ["user"]
    assert [m.role for m in synth_interrupted_messages(shell("Operation cancelled by user"))] == ["user"]


@test("runtime_partial_commit", "stale RUNNING input and partial are recovered into history")
async def t_recover_stale_running_dict(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"stale-session-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        now = int(time.time())
        session_id = "stale-session"
        run_id = "stale-run"
        runs = [{
            "run_id": run_id,
            "session_id": session_id,
            "agent_id": "a",
            "user_id": "owner",
            "status": "RUNNING",
            "input": {"input_content": "investigate the database lock"},
            "content": "I found the writer that held",
            "messages": [],
            "created_at": now,
        }]
        conn = await db._ensure_connected()
        await conn.execute(
            "INSERT INTO sessions "
            "(session_id, session_type, agent_id, user_id, metadata, runs, created_at, updated_at) "
            "VALUES (?, 'agent', 'a', 'owner', '{}', ?, ?, ?)",
            (session_id, json.dumps(runs), now, now),
        )
        await conn.commit()

        assert await db.recover_stale_session_runs(session_id) == 1
        recovered = (await db.list_session_runs(session_id))[0]
        assert recovered["status"] == "COMPLETED", recovered
        assert [(m["role"], m["content"]) for m in recovered["messages"]] == [
            ("user", "investigate the database lock"),
            ("assistant", "I found the writer that held"),
        ]
        # Recovery is idempotent and the normalized transcript sees the same
        # repaired conversation, not a permanently-streaming shell.
        assert await db.recover_stale_session_runs(session_id) == 0
        run_row = await (
            await conn.execute(
                "SELECT status FROM session_runs WHERE session_id=?",
                (session_id,),
            )
        ).fetchone()
        assert run_row is not None and run_row[0] == "success", run_row
        messages = await (
            await conn.execute(
                "SELECT role, text FROM session_messages WHERE session_id=? ORDER BY sequence",
                (session_id,),
            )
        ).fetchall()
        assert [(m[0], m[1]) for m in messages] == [
            ("user", "investigate the database lock"),
            ("assistant", "I found the writer that held"),
        ], messages
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("runtime_partial_commit", "startup recovery finds projected running sessions")
async def t_recover_all_stale_running(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"stale-startup-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        now = int(time.time())
        conn = await db._ensure_connected()
        await conn.execute(
            "INSERT INTO sessions "
            "(session_id, session_type, agent_id, user_id, metadata, runs, created_at, updated_at) "
            "VALUES ('startup-stale', 'agent', 'a', 'owner', '{}', ?, ?, ?)",
            (json.dumps([{
                "run_id": "startup-run",
                "status": "RUNNING",
                "input": {"input_content": "continue this task"},
                "messages": [],
                "created_at": now,
            }]), now, now),
        )
        await db._project_operational_session("startup-stale")
        await conn.commit()

        assert await db.recover_all_stale_session_runs() == 1
        recovered = (await db.list_session_runs("startup-stale"))[0]
        assert recovered["status"] == "COMPLETED", recovered
        assert recovered["messages"][0]["content"] == "continue this task"
        assert await db.recover_all_stale_session_runs() == 0
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("runtime_partial_commit", "journal witness restores history lost after compaction")
async def t_recover_missing_journal_history(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"journal-recovery-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        now = int(time.time())
        session_id = "journal-lost"
        surviving = [{
            "run_id": "recent-run",
            "status": "COMPLETED",
            "content": "recent answer",
            "messages": [
                {
                    "role": "user",
                    "content": "first lost request",
                    "from_history": True,
                },
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
            ],
            "created_at": now,
        }]
        conn = await db._ensure_connected()
        await conn.execute(
            "INSERT INTO sessions "
            "(session_id, session_type, agent_id, user_id, metadata, runs, created_at, updated_at) "
            "VALUES (?, 'agent', 'a', 'owner', '{}', ?, ?, ?)",
            (session_id, json.dumps(surviving), now, now),
        )
        await conn.commit()
        for event_type, text in [
            ("user/message", "first lost request"),
            ("assistant/message", "first lost answer"),
            ("user/message", "only continue now after validation"),
            ("assistant/message", "validation is still pending"),
            ("user/message", "second lost decision"),
            ("assistant/message", "second lost answer"),
            ("user/message", "recent question"),
            ("assistant/message", "recent answer"),
            ("user/message", "continue now"),
        ]:
            await db.append_session_event(session_id, event_type, {"text": text})

        recovered_count = await db.recover_session_from_journal(
            session_id, current_text="[language hint]\ncontinue now",
        )
        assert recovered_count == 6, recovered_count
        runs = await db.list_session_runs(session_id, limit=10)
        recovery = next(
            run for run in runs
            if (run.get("metadata") or {}).get("continuity_recovery")
        )
        assert [m["content"] for m in recovery["messages"]] == [
            "first lost request",
            "first lost answer",
            "only continue now after validation",
            "validation is still pending",
            "second lost decision",
            "second lost answer",
        ], recovery
        assert all(
            message.get("content") != "continue now"
            for message in recovery["messages"]
        )
        assert await db.recover_session_from_journal(
            session_id, current_text="continue now",
        ) == 0

        projected = await (
            await conn.execute(
                "SELECT role, text FROM session_messages "
                "WHERE session_id=? ORDER BY sequence",
                (session_id,),
            )
        ).fetchall()
        projected_pairs = [(row[0], row[1]) for row in projected]
        assert ("user", "first lost request") in projected_pairs
        assert ("assistant", "second lost answer") in projected_pairs
        assert ("user", "recent question") in projected_pairs
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("runtime_partial_commit", "journal recovery reads the newest bounded event window")
async def t_recover_latest_journal_window(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"journal-window-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        now = int(time.time())
        session_id = "journal-window"
        conn = await db._ensure_connected()
        await conn.execute(
            "INSERT INTO sessions "
            "(session_id, session_type, agent_id, user_id, metadata, runs, created_at, updated_at) "
            "VALUES (?, 'agent', 'a', 'owner', '{}', ?, ?, ?)",
            (session_id, json.dumps([{
                "run_id": "survivor",
                "status": "COMPLETED",
                "content": "surviving answer",
                "messages": [
                    {"role": "user", "content": "surviving request"},
                    {"role": "assistant", "content": "surviving answer"},
                ],
                "created_at": now,
            }]), now, now),
        )
        await conn.executemany(
            "INSERT INTO session_events (session_id, seq, ts_ms, type, data) "
            "VALUES (?, ?, ?, 'tool/status', '{\"text\":\"progress\"}')",
            [(session_id, seq, now * 1000) for seq in range(1, 2006)],
        )
        await conn.executemany(
            "INSERT INTO session_events (session_id, seq, ts_ms, type, data) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    session_id, 2006, now * 1000, "user/message",
                    json.dumps({"text": "newest lost request"}),
                ),
                (
                    session_id, 2007, now * 1000, "assistant/message",
                    json.dumps({"text": "newest lost answer"}),
                ),
                (
                    session_id, 2008, now * 1000, "user/message",
                    json.dumps({"text": "continue"}),
                ),
            ],
        )
        await conn.commit()

        assert await db.recover_session_from_journal(
            session_id, current_text="continue",
        ) == 2
        runs = await db.list_session_runs(session_id, limit=10)
        recovery = next(
            run for run in runs
            if (run.get("metadata") or {}).get("continuity_recovery")
        )
        assert [message["content"] for message in recovery["messages"]] == [
            "newest lost request",
            "newest lost answer",
        ]
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
