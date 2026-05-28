"""Session-continuity tests — make sure runs accumulate across turns.

Background
----------

In ~/my-agent (2026-05-28) two consecutive ``sessions`` rows had
``num_runs=1`` despite many user messages over hours. The LLM appeared
to have no memory of the previous turn because the previous turn's run
was overwritten by the next one. Yesterday's pre-agno→inline DBs had
sessions with 4-30 runs each, so something in the new persistence
layer was silently dropping rows.

These tests pin down the contract so the regression can't slip back:

1. ``AgentSession.from_dict`` / ``TeamSession.from_dict`` MUST tolerate
   ``runs == []`` (the IndexError on ``runs[0]`` used to bubble up and
   trigger the "create new session" fallback that overwrote the row).

2. Runs containing BOTH an ``agent_id`` key (with ``None`` value) and a
   truthy ``team_id`` must dispatch as ``TeamRunOutput``. The previous
   logic checked ``if "agent_id" in run`` first and would mis-classify
   team runs as agent runs, dropping their team_id-based filtering in
   ``get_messages``.

3. The combined ``MemoryDB.upsert_session`` (metadata-only) +
   ``SqliteDb.upsert_session`` (full runtime upsert) interleave must
   NOT lose runs. The gateway writes the bare row first (session_type=
   'agent', runs=NULL); the runtime later writes the TeamSession with
   runs — the on_conflict_do_update path must accumulate them.

4. A TeamSession written, then read, then written again must end up
   with 2 runs in the DB row — not 1.

5. The dispatcher's ``TeamRouterProvider._ensure_runtime`` Team-build
   path must preserve session.runs across cache invalidations (a new
   Team built with the same session_id reads the existing runs before
   writing).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import uuid

from ._framework import TestContext, TestSkip, test


@test("session_continuity", "AgentSession.from_dict tolerates runs=[]")
async def t_agent_from_dict_empty_runs(_ctx: TestContext) -> None:
    """Empty list ``runs=[]`` must not raise IndexError. The bug used to
    propagate as ``IndexError: list index out of range`` from
    ``isinstance(runs[0], dict)``, get swallowed by ``read_session``'s
    blanket ``except``, and surface as "session not found" — making the
    next turn create a fresh session that wiped the row.
    """
    from src.memory.sessions.agent import AgentSession

    session = AgentSession.from_dict({
        "session_id": "S1",
        "agent_id": "A",
        "runs": [],          # the empty-list payload that used to crash
        "created_at": 1,
        "updated_at": 1,
    })
    assert session is not None
    assert session.runs == []


@test("session_continuity", "TeamSession.from_dict tolerates runs=[]")
async def t_team_from_dict_empty_runs(_ctx: TestContext) -> None:
    """Same contract as the AgentSession test above, for TeamSession."""
    from src.memory.sessions.team import TeamSession

    session = TeamSession.from_dict({
        "session_id": "S1",
        "team_id": "T",
        "runs": [],
        "created_at": 1,
        "updated_at": 1,
    })
    assert session is not None
    assert session.runs == []


@test(
    "session_continuity",
    "from_dict dispatches runs with truthy team_id to TeamRunOutput "
    "even when agent_id key is present (with None value)",
)
async def t_from_dict_dispatch_mixed_keys(_ctx: TestContext) -> None:
    """A run dict may carry BOTH ``agent_id`` and ``team_id`` keys —
    ``agent_id: None`` and ``team_id: "team-1"``. The dispatch must look
    at TRUTHY values, not just key presence, otherwise team runs get
    silently mis-classified as agent runs (and then their team_id-based
    filtering in ``get_messages`` drops them from history)."""
    from src.memory.sessions.agent import AgentSession
    from src.core._run_state.team import TeamRunOutput

    session = AgentSession.from_dict({
        "session_id": "S1",
        "agent_id": "A",
        "runs": [
            {
                "run_id": "r1",
                "agent_id": None,            # present but falsy
                "team_id": "team-leader",    # truthy → TeamRunOutput
                "session_id": "S1",
                "content": "hello",
            }
        ],
        "created_at": 1,
        "updated_at": 1,
    })
    assert session is not None
    assert len(session.runs) == 1
    assert isinstance(session.runs[0], TeamRunOutput), (
        f"expected TeamRunOutput, got {type(session.runs[0]).__name__}"
    )


@test(
    "session_continuity",
    "SqliteDb round-trip: two consecutive turns accumulate two runs",
)
async def t_sqlite_two_turn_round_trip(_ctx: TestContext) -> None:
    """The simplest possible regression: write a TeamSession with one
    run, read it back, append another run, write back, read again. The
    second read MUST show 2 runs. Pre-fix, certain interleavings with
    the gateway's MemoryDB.upsert_session left only the most recent run
    in the row."""
    from src.memory.store.sqlite import SqliteDb
    from src.memory.store.base import SessionType
    from src.memory.sessions import TeamSession
    from src.core._run_state.team import TeamRunOutput
    from src.models.providers.message import Message

    fd, db_path = tempfile.mkstemp(prefix="oa_continuity_", suffix=".db")
    os.close(fd)
    try:
        db = SqliteDb(db_file=db_path, session_table="sessions")
        sid = "S-" + uuid.uuid4().hex[:8]
        now = int(time.time())

        # Turn 1
        s1 = TeamSession(
            session_id=sid, team_id="T", user_id="openagent", created_at=now,
        )
        s1.upsert_run(TeamRunOutput(
            run_id="r1", team_id="T", session_id=sid, user_id="openagent",
            content="reply 1", messages=[Message(role="assistant", content="r1")],
        ))
        db.upsert_session(s1)

        # Turn 2: load, append, save.
        s2 = db.get_session(session_id=sid, session_type=SessionType.TEAM)
        assert s2 is not None, "second-turn load returned None"
        assert s2.runs and len(s2.runs) == 1, (
            f"second-turn load lost the first turn's run; got "
            f"{len(s2.runs or [])} runs"
        )
        s2.upsert_run(TeamRunOutput(
            run_id="r2", team_id="T", session_id=sid, user_id="openagent",
            content="reply 2", messages=[Message(role="assistant", content="r2")],
        ))
        db.upsert_session(s2)

        # Final read
        s_final = db.get_session(session_id=sid, session_type=SessionType.TEAM)
        assert s_final is not None
        assert len(s_final.runs or []) == 2, (
            f"runs failed to accumulate across turns; got "
            f"{len(s_final.runs or [])} runs, expected 2"
        )
        ids = {r.run_id for r in s_final.runs}
        assert ids == {"r1", "r2"}, f"got run_ids={ids}"
    finally:
        os.unlink(db_path)


@test(
    "session_continuity",
    "gateway pre-creates row (session_type='agent', runs=NULL) — "
    "runtime writes TeamSession with runs, runs ACCUMULATE across turns",
)
async def t_gateway_runtime_interleave(_ctx: TestContext) -> None:
    """The exact interleave from the user's broken sessions:

    1. Gateway's ``MemoryDB.upsert_session(session_id, ...)`` creates a
       bare row with ``session_type='agent'``, ``runs=NULL``.
    2. Runtime's ``SqliteDb.upsert_session(team_session)`` writes the
       first turn's run.
    3. Gateway's ``MemoryDB.upsert_session(session_id, ...)`` UPDATES the
       metadata field (no runs touched). [This is the suspicious step
       that used to be assumed safe but needs explicit coverage.]
    4. Runtime's ``SqliteDb.upsert_session(team_session)`` writes the
       second turn's run.

    After step 4 the row must contain BOTH runs.
    """
    from src.memory.store.sqlite import SqliteDb
    from src.memory.store.base import SessionType
    from src.memory.sessions import TeamSession
    from src.core._run_state.team import TeamRunOutput
    from src.models.providers.message import Message

    fd, db_path = tempfile.mkstemp(prefix="oa_gw_interleave_", suffix=".db")
    os.close(fd)
    try:
        sid = "S-" + uuid.uuid4().hex[:8]
        now = int(time.time())

        # Step 1: emulate MemoryDB's metadata-only insert
        conn = sqlite3.connect(db_path)
        conn.executescript("""
        CREATE TABLE sessions (
            session_id   TEXT PRIMARY KEY,
            session_type TEXT NOT NULL,
            agent_id     TEXT, team_id TEXT, workflow_id TEXT,
            user_id      TEXT,
            session_data TEXT, agent_data TEXT, team_data TEXT, workflow_data TEXT,
            metadata     TEXT, runs TEXT, summary TEXT,
            created_at   INTEGER NOT NULL,
            updated_at   INTEGER
        );
        """)
        conn.execute(
            "INSERT INTO sessions (session_id, session_type, user_id, metadata, created_at, updated_at) "
            "VALUES (?, 'agent', 'openagent', '{}', ?, ?)",
            (sid, now, now),
        )
        conn.commit()
        conn.close()

        # Step 2: runtime writes turn-1 run.
        db = SqliteDb(db_file=db_path, session_table="sessions")
        s1 = db.get_session(session_id=sid, session_type=SessionType.TEAM)
        # Gateway pre-creates with NULL team_id → runtime loads with empty runs.
        if s1 is None:
            s1 = TeamSession(session_id=sid, team_id="T", user_id="openagent", created_at=now)
        s1.upsert_run(TeamRunOutput(
            run_id="r1", team_id="T", session_id=sid, user_id="openagent",
            content="reply 1", messages=[Message(role="assistant", content="r1")],
        ))
        db.upsert_session(s1)

        # Step 3: emulate MemoryDB's metadata UPDATE (no runs touched).
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE sessions SET metadata = ?, updated_at = ? WHERE session_id = ?",
            ('{"title":"chat 1"}', int(time.time()), sid),
        )
        conn.commit()
        conn.close()

        # Step 4: runtime writes turn-2 run.
        s2 = db.get_session(session_id=sid, session_type=SessionType.TEAM)
        assert s2 is not None, "metadata UPDATE wiped the session row"
        assert s2.runs and len(s2.runs) == 1, (
            f"metadata UPDATE clobbered runs; got "
            f"{len(s2.runs or [])} runs after the UPDATE"
        )
        s2.upsert_run(TeamRunOutput(
            run_id="r2", team_id="T", session_id=sid, user_id="openagent",
            content="reply 2", messages=[Message(role="assistant", content="r2")],
        ))
        db.upsert_session(s2)

        # Final read: both runs survive.
        s_final = db.get_session(session_id=sid, session_type=SessionType.TEAM)
        assert s_final is not None
        assert len(s_final.runs or []) == 2, (
            f"runs failed to accumulate across the gateway/runtime "
            f"interleave; got {len(s_final.runs or [])} runs, expected 2"
        )
    finally:
        os.unlink(db_path)


@test(
    "session_continuity",
    "session_type mismatch (row='agent', read-as=TEAM) preserves runs",
)
async def t_session_type_mismatch_preserves_runs(_ctx: TestContext) -> None:
    """The user's sessions had ``session_type='agent'`` (created by the
    gateway's MemoryDB) but ran through a Team (which loads as
    SessionType.TEAM). The mismatch must not corrupt runs — the read
    should yield a TeamSession with all stored runs, even though the
    row's ``session_type`` column says 'agent'.
    """
    from src.memory.store.sqlite import SqliteDb
    from src.memory.store.base import SessionType
    from src.memory.sessions import AgentSession, TeamSession
    from src.core._run_state.team import TeamRunOutput
    from src.models.providers.message import Message

    fd, db_path = tempfile.mkstemp(prefix="oa_mismatch_", suffix=".db")
    os.close(fd)
    try:
        sid = "S-" + uuid.uuid4().hex[:8]
        now = int(time.time())

        db = SqliteDb(db_file=db_path, session_table="sessions")

        # First write as AgentSession (session_type='agent').
        s_agent = AgentSession(
            session_id=sid, agent_id=None, user_id="openagent", created_at=now,
        )
        s_agent.upsert_run(TeamRunOutput(
            run_id="r1", team_id="T", session_id=sid, user_id="openagent",
            content="hi", messages=[Message(role="assistant", content="r1")],
        ))
        db.upsert_session(s_agent)

        # Re-read as TEAM — should still see the run.
        s_team = db.get_session(session_id=sid, session_type=SessionType.TEAM)
        assert isinstance(s_team, TeamSession), f"got {type(s_team).__name__}"
        assert s_team.runs and len(s_team.runs) == 1, (
            f"team read of agent-row dropped runs; got "
            f"{len(s_team.runs or [])} runs"
        )
    finally:
        os.unlink(db_path)


@test(
    "session_continuity",
    "get_session(user_id='X') matches rows with user_id=NULL (unclaimed)",
)
async def t_get_session_matches_null_user(_ctx: TestContext) -> None:
    """The 2026-05-28 PROD bug: the read filter used to require
    ``user_id == 'alessandro'`` strictly, so a row with ``user_id=NULL``
    (post-migration / gateway pre-create) was invisible to the runtime.
    The runtime then built an empty TeamSession and its UPSERT
    overwrote the existing runs — the LLM 'forgot' every previous turn.

    Fix: ``get_session`` now matches ``user_id == X OR user_id IS NULL``
    so an authenticated read can claim an unowned row without losing
    its prior runs."""
    from src.memory.store.sqlite import SqliteDb
    from src.memory.store.base import SessionType
    from src.memory.sessions import TeamSession
    from src.core._run_state.team import TeamRunOutput
    from src.models.providers.message import Message

    fd, db_path = tempfile.mkstemp(prefix="oa_read_null_", suffix=".db")
    os.close(fd)
    try:
        sid = "S-" + uuid.uuid4().hex[:8]
        now = int(time.time())

        # Seed the broken-state row: user_id=NULL, has one prior run.
        db = SqliteDb(db_file=db_path, session_table="sessions")
        seed = TeamSession(
            session_id=sid, team_id="T-seed", user_id=None, created_at=now,
        )
        seed.upsert_run(TeamRunOutput(
            run_id="r-old", team_id="T-seed", session_id=sid, user_id=None,
            content="old reply",
            messages=[Message(role="assistant", content="old")],
        ))
        db.upsert_session(seed)

        # Authenticated read with a real handle must SEE the existing row.
        loaded = db.get_session(
            session_id=sid, session_type=SessionType.TEAM, user_id="alessandro",
        )
        assert loaded is not None, (
            "authenticated read couldn't see the NULL-user row; the "
            "runtime would have built an empty session and overwritten "
            "the runs column on the next save — this is the prod bug "
            "the user hit on 2026-05-28"
        )
        assert len(loaded.runs or []) == 1
        assert loaded.runs[0].run_id == "r-old"
    finally:
        os.unlink(db_path)


@test(
    "session_continuity",
    "Team load-then-save claims unowned row + preserves prior runs",
)
async def t_team_claim_unowned_preserves_runs(_ctx: TestContext) -> None:
    """The end-to-end version of the prod bug: load an unowned (NULL
    user_id) TeamSession, append a new turn, save back, verify both
    runs survive AND the row's user_id is now the caller's handle (so
    subsequent strict-equality WHEREs match)."""
    from src.memory.store.sqlite import SqliteDb
    from src.memory.store.base import SessionType
    from src.memory.sessions import TeamSession
    from src.core._run_state.team import TeamRunOutput
    from src.models.providers.message import Message

    fd, db_path = tempfile.mkstemp(prefix="oa_team_claim_", suffix=".db")
    os.close(fd)
    try:
        sid = "S-" + uuid.uuid4().hex[:8]
        now = int(time.time())

        db = SqliteDb(db_file=db_path, session_table="sessions")
        seed = TeamSession(
            session_id=sid, team_id="T-seed", user_id=None, created_at=now,
        )
        seed.upsert_run(TeamRunOutput(
            run_id="r-old", team_id="T-seed", session_id=sid, user_id=None,
            content="old", messages=[Message(role="assistant", content="old")],
        ))
        db.upsert_session(seed)

        # Mirror what the runtime's _aread_or_create_session does now:
        # read with the caller's user_id, then claim the row by
        # stamping user_id when the loaded session is unowned.
        loaded = db.get_session(
            session_id=sid, session_type=SessionType.TEAM, user_id="alessandro",
        )
        assert loaded is not None
        if not loaded.user_id:
            loaded.user_id = "alessandro"
        loaded.upsert_run(TeamRunOutput(
            run_id="r-new", team_id="T-new", session_id=sid,
            user_id="alessandro", content="new",
            messages=[Message(role="assistant", content="new")],
        ))
        db.upsert_session(loaded)

        # Final state: 2 runs, row claimed by 'alessandro'.
        # Reload through the real deserializer — ``json_array_length``
        # on the raw column gives 0 because SQLAlchemy's JSON type
        # round-trips the column with one extra encode/decode layer
        # on top of our ``serialize_session_json_fields`` call.
        final = db.get_session(
            session_id=sid, session_type=SessionType.TEAM, user_id="alessandro",
        )
        assert final is not None, "final read missed the row"
        assert final.user_id == "alessandro", (
            f"row should be claimed by 'alessandro'; got user_id={final.user_id!r}"
        )
        runs = final.runs or []
        assert len(runs) == 2, (
            f"prior run must survive — expected 2 runs, got {len(runs)}"
        )
        ids = {r.run_id for r in runs}
        assert ids == {"r-old", "r-new"}, f"unexpected run_ids: {ids}"
    finally:
        os.unlink(db_path)


@test(
    "session_continuity",
    "user_id sentinel 'openagent' is migrated to NULL on bootstrap",
)
async def t_user_id_sentinel_migration(_ctx: TestContext) -> None:
    """Pre-existing rows with ``user_id='openagent'`` (the old gateway
    sentinel) must be migrated to NULL so the runtime can UPSERT them.
    The runtime's WHERE clause is ``user_id == new OR user_id IS NULL``;
    a stale sentinel blocks every subsequent UPDATE and the runs
    silently never accumulate."""
    import asyncio
    from src.memory.db import MemoryDB

    fd, db_path = tempfile.mkstemp(prefix="oa_sentinel_", suffix=".db")
    os.close(fd)
    try:
        # Seed with the broken-state row: user_id='openagent' on a
        # legitimately authenticated session.
        conn = sqlite3.connect(db_path)
        conn.executescript("""
        CREATE TABLE sessions (
            session_id   TEXT PRIMARY KEY,
            session_type TEXT NOT NULL,
            agent_id     TEXT, team_id TEXT, workflow_id TEXT,
            user_id      TEXT,
            session_data TEXT, agent_data TEXT, team_data TEXT, workflow_data TEXT,
            metadata     TEXT, runs TEXT, summary TEXT,
            created_at   INTEGER NOT NULL,
            updated_at   INTEGER
        );
        """)
        now = int(time.time())
        conn.execute(
            "INSERT INTO sessions (session_id, session_type, user_id, metadata, created_at, updated_at) "
            "VALUES ('S-broken', 'agent', 'openagent', '{}', ?, ?)",
            (now, now),
        )
        # Defensive: real-handle rows must NOT be touched by the migration.
        conn.execute(
            "INSERT INTO sessions (session_id, session_type, user_id, metadata, created_at, updated_at) "
            "VALUES ('S-real', 'agent', 'alessandro', '{}', ?, ?)",
            (now, now),
        )
        conn.commit()
        conn.close()

        # MemoryDB.connect() runs the migration on bootstrap.
        db = MemoryDB(db_path=db_path)
        await db.connect()
        try:
            row1 = await (
                await db._conn.execute(
                    "SELECT user_id FROM sessions WHERE session_id = 'S-broken'"
                )
            ).fetchone()
            row2 = await (
                await db._conn.execute(
                    "SELECT user_id FROM sessions WHERE session_id = 'S-real'"
                )
            ).fetchone()
        finally:
            await db.close()

        assert row1[0] is None, (
            f"sentinel 'openagent' should migrate to NULL; got {row1[0]!r}"
        )
        assert row2[0] == "alessandro", (
            f"real handle must not be touched; got {row2[0]!r}"
        )
    finally:
        os.unlink(db_path)


@test(
    "session_continuity",
    "gateway INSERT uses handle as user_id (no more 'openagent' sentinel)",
)
async def t_gateway_insert_uses_handle(_ctx: TestContext) -> None:
    """The fixed gateway must write ``user_id=<handle>`` (not the
    ``'openagent'`` sentinel) so the runtime's UPSERT can update the
    row on subsequent turns. A handle-less call (no client_id) writes
    NULL, which is also runtime-claimable."""
    from src.memory.db import MemoryDB

    fd, db_path = tempfile.mkstemp(prefix="oa_gw_insert_", suffix=".db")
    os.close(fd)
    try:
        db = MemoryDB(db_path=db_path)
        await db.connect()
        try:
            await db.upsert_session(
                "S-with-handle", client_id="alessandro", title="t1",
            )
            await db.upsert_session(
                "S-no-handle", title="t2",
            )
            row1 = await (
                await db._conn.execute(
                    "SELECT user_id FROM sessions WHERE session_id = 'S-with-handle'"
                )
            ).fetchone()
            row2 = await (
                await db._conn.execute(
                    "SELECT user_id FROM sessions WHERE session_id = 'S-no-handle'"
                )
            ).fetchone()
        finally:
            await db.close()

        assert row1[0] == "alessandro", (
            f"gateway INSERT must use handle as user_id; got {row1[0]!r}"
        )
        assert row2[0] is None, (
            f"handle-less INSERT must use NULL (NOT 'openagent'); "
            f"got {row2[0]!r}"
        )
    finally:
        os.unlink(db_path)


@test(
    "session_continuity",
    "Team across three turns: runs accumulate in the sessions row",
)
async def t_team_three_turns_accumulate(_ctx: TestContext) -> None:
    """End-to-end: build a Team with a stub model, run three turns with
    the same ``session_id``, then assert the DB row carries all three
    runs. The fake model errors mid-call (response shape isn't valid),
    but the runtime still calls ``acleanup_and_store`` which appends the
    run — that's the contract we're locking down here, not the success
    of the model call itself.
    """
    try:
        from src.core._runner.agent import Agent as RuntimeAgent
        from src.core._runner.team import Team, TeamMode
        from src.memory.store.sqlite import SqliteDb
        from src.memory.store.base import SessionType
        from src.models.providers.base import Model
        from src.models.providers.response import ModelResponse
    except ImportError as exc:
        raise TestSkip(f"runtime imports failed: {exc}")

    from dataclasses import dataclass

    @dataclass
    class _FakeModel(Model):
        id: str = "fake"
        provider: str = "Fake"
        canned: str = "ok"

        def invoke(self, *args, **kwargs):
            raise NotImplementedError

        async def ainvoke(self, *args, **kwargs):
            class FakeChoice:
                message = type("M", (), {
                    "content": self.canned, "role": "assistant", "tool_calls": None,
                })()
                finish_reason = "stop"

            class FakeUsage:
                input_tokens = 5; output_tokens = 5; total_tokens = 10
                completion_tokens = 5; prompt_tokens = 5
            return type(
                "R", (),
                {"choices": [FakeChoice()], "usage": FakeUsage(), "id": "x"},
            )()

        def invoke_stream(self, *args, **kwargs):
            raise NotImplementedError

        async def ainvoke_stream(self, *args, **kwargs):
            class FakeDelta:
                role = "assistant"; content = self.canned; tool_calls = None
            yield type(
                "Chunk", (),
                {
                    "choices": [type("Ch", (), {"delta": FakeDelta(), "finish_reason": "stop"})()],
                    "usage": None,
                },
            )()

        def _parse_provider_response(self, response, **kwargs):
            msg = response.choices[0].message
            return ModelResponse(role="assistant", content=msg.content)

        def _parse_provider_response_delta(self, response, **kwargs):
            return ModelResponse(role="assistant", content="")

    fd, db_path = tempfile.mkstemp(prefix="oa_team3_", suffix=".db")
    os.close(fd)
    try:
        db = SqliteDb(db_file=db_path, session_table="sessions")
        team = Team(
            mode=TeamMode.coordinate,
            model=_FakeModel(canned="coord"),
            members=[
                RuntimeAgent(name="m1", model=_FakeModel(canned="m1"), markdown=False),
                RuntimeAgent(name="m2", model=_FakeModel(canned="m2"), markdown=False),
            ],
            db=db,
            add_history_to_context=True,
            num_history_runs=20,
        )
        sid = "S-team3-" + uuid.uuid4().hex[:6]
        for i in range(1, 4):
            try:
                await team.arun(
                    input=f"user-msg-{i}",
                    session_id=sid,
                    user_id="openagent",
                )
            except Exception:
                # Fake model can error inside the runtime parser — we
                # still expect the run to be persisted via the cleanup
                # path. If a future runtime change skips persistence on
                # error, this assertion catches it.
                pass

        s = db.get_session(session_id=sid, session_type=SessionType.TEAM)
        assert s is not None
        assert len(s.runs or []) == 3, (
            f"expected 3 runs after 3 Team turns, got {len(s.runs or [])}; "
            "the runtime is dropping runs across turns — see "
            "_acleanup_and_store / asave_session in core/_runner/team/_run.py"
        )
    finally:
        os.unlink(db_path)
