"""AgnoProvider.commit_partial_assistant — synth-run injection round-trip.

Verifies that a barge-in commit appends a synthetic ``RunOutput``
carrying the partial assistant text to Agno's session row, so the next
turn's ``add_history_to_context=True`` reload sees ``user → assistant
(interrupted) → user`` instead of two adjacent user turns.

Uses Agno's real ``SqliteDb`` against a throwaway file so the upsert /
get round-trip exercises the actual schema rather than a mocked stand-in.
"""
from __future__ import annotations

import os
import tempfile
import time

from ._framework import TestContext, TestSkip, test


@test("agno_partial_commit", "synth run lands in the agno_sessions row")
async def t_synth_run_round_trip(_ctx: TestContext) -> None:
    try:
        from agno.db.sqlite import SqliteDb
        from agno.db.base import SessionType
        from agno.session.agent import AgentSession
        from agno.run.agent import RunOutput
        from agno.run.base import RunStatus
        from agno.models.message import Message
    except ImportError as e:
        raise TestSkip(f"agno not installed: {e}")

    from openagent.models.agno_provider import AgnoProvider

    fd, db_path = tempfile.mkstemp(prefix="oa_agno_commit_", suffix=".db")
    os.close(fd)
    try:
        # Seed the DB with a session that has one prior run, like a normal
        # multi-turn conversation would. ``created_at`` is required by
        # the agno_sessions schema (NOT NULL in v2.x).
        db = SqliteDb(db_file=db_path)
        now = int(time.time())
        seeded_session = AgentSession(
            session_id="sid-commit",
            agent_id="test-agent",
            user_id="u",
            runs=[RunOutput(
                run_id="r-1",
                agent_id="test-agent",
                session_id="sid-commit",
                user_id="u",
                content="prior reply",
                messages=[
                    Message(role="user", content="prior question"),
                    Message(role="assistant", content="prior reply"),
                ],
                status=RunStatus.completed,
            )],
            created_at=now,
            updated_at=now,
        )
        db.upsert_session(seeded_session, deserialize=False)

        # Build a provider pointing at the same db_path. We don't need an
        # API key — commit_partial_assistant only does DB IO.
        provider = AgnoProvider(
            model="agno:openai:gpt-4o-mini",
            api_key=None,
            db_path=db_path,
        )

        await provider.commit_partial_assistant(
            "sid-commit",
            "I was about to explain when",
        )

        # Re-read and verify the synth run landed.
        out = db.get_session(
            session_id="sid-commit",
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        assert isinstance(out, AgentSession), type(out)
        assert out.runs is not None
        assert len(out.runs) == 2, f"expected 2 runs after commit; got {len(out.runs)}"
        synth = out.runs[-1]
        assert synth.status == RunStatus.cancelled, synth.status
        assert synth.content == "I was about to explain when", synth.content
        # The synth run carries one assistant message with the same text.
        assert synth.messages and synth.messages[-1].role == "assistant"
        assert synth.messages[-1].content == "I was about to explain when"
    finally:
        os.unlink(db_path)


@test("agno_partial_commit", "missing session row is a silent no-op")
async def t_missing_session_noop(_ctx: TestContext) -> None:
    try:
        from agno.db.sqlite import SqliteDb  # noqa: F401
    except ImportError as e:
        raise TestSkip(f"agno not installed: {e}")

    from openagent.models.agno_provider import AgnoProvider

    fd, db_path = tempfile.mkstemp(prefix="oa_agno_noop_", suffix=".db")
    os.close(fd)
    try:
        provider = AgnoProvider(
            model="agno:openai:gpt-4o-mini",
            api_key=None,
            db_path=db_path,
        )
        # No prior session row — must not raise.
        await provider.commit_partial_assistant("never-seen", "partial")
    finally:
        os.unlink(db_path)


@test("agno_partial_commit", "empty inputs short-circuit")
async def t_empty_inputs_short_circuit(_ctx: TestContext) -> None:
    try:
        from agno.db.sqlite import SqliteDb  # noqa: F401
    except ImportError as e:
        raise TestSkip(f"agno not installed: {e}")

    from openagent.models.agno_provider import AgnoProvider

    provider = AgnoProvider(
        model="agno:openai:gpt-4o-mini",
        api_key=None,
        db_path=":memory:",
    )
    # No DB call at all — both arguments degenerate.
    await provider.commit_partial_assistant("", "x")
    await provider.commit_partial_assistant("sid", "")
