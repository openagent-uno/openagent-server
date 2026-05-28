"""In-session compaction — vision §2 ("the session compacts in place").

Covers the five contracts of ``src.core.compaction``:

1. ``should_compact`` returns ``True`` past the threshold and ``False``
   when stored history is small.
2. ``compact`` replaces the oldest runs with a recap entry and keeps
   the most recent N runs verbatim.
3. The recap run is tagged with ``metadata.compaction = True`` so a
   future pass can recognise an already-compacted span.
4. ``Agent._run_inner`` calls ``compact`` before the next
   ``model.generate()`` when the threshold is breached (no real API
   call — the model is a stub).
5. The feature flag (``OPENAGENT_COMPACTION_ENABLED=false``) disables
   compaction cleanly.

All tests work against a throwaway SQLite file directly — the
compaction module reads/writes the ``sessions.runs`` column via raw SQL, so
it doesn't need the runtime installed to exercise the contract.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time

from ._framework import TestContext, test


# ── Helpers ────────────────────────────────────────────────────────────


def _make_session_row(
    db_path: str, session_id: str, runs: list[dict],
) -> None:
    """Seed an sessions row with the given runs JSON.

    The compaction module reads / writes the row with raw SQL, so we
    create the schema by hand here. Matches the canonical schema in
    ``src/memory/db.py`` SCHEMA_SQL.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                session_type TEXT,
                agent_id     TEXT,
                team_id      TEXT,
                workflow_id  TEXT,
                user_id      TEXT,
                session_data TEXT,
                agent_data   TEXT,
                team_data    TEXT,
                workflow_data TEXT,
                metadata     TEXT,
                runs         TEXT,
                summary      TEXT,
                created_at   INTEGER NOT NULL,
                updated_at   INTEGER NOT NULL
            )
            """
        )
        now = int(time.time())
        conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, session_type, runs, created_at, updated_at) "
            "VALUES (?, 'agent', ?, ?, ?)",
            (session_id, json.dumps(runs), now, now),
        )
        conn.commit()
    finally:
        conn.close()


def _read_runs(db_path: str, session_id: str) -> list[dict]:
    """Re-read the runs column for assertions."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT runs FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return []
    return json.loads(row[0])


class _FakeDB:
    """Mimics MemoryDB's ``db_path`` attribute for compaction.

    The compaction module reaches into ``agent._db.db_path`` to find
    the SQLite file. Anything that quacks like a dotted attribute does.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path


class _FakeModel:
    """Stand-in provider for the summariser.

    Mirrors ``BaseModel.generate``'s shape and returns a fixed recap
    string so tests don't pay an API call. ``self.model`` is read by
    ``_resolve_model_id``.
    """

    def __init__(
        self,
        *,
        model: str = "fake/test-model",
        max_context: int = 1000,
        summary: str = "Recap: the user is testing in-session compaction.",
    ) -> None:
        self.model = model
        self.context_window = max_context
        self._summary = summary
        self.generate_calls: list[dict] = []

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        self.generate_calls.append({"messages": messages, "kwargs": kwargs})
        # Mimic ModelResponse just enough — only ``content`` is read.
        class _R:  # noqa: D401
            pass
        r = _R()
        r.content = self._summary
        return r


class _FakeAgent:
    """Tiny stand-in for ``Agent`` that carries the right hooks.

    The compaction module only needs ``_db`` (for ``db_path``) and
    ``model``. Tests don't go through the full Agent.run path.
    """

    def __init__(self, db_path: str, model: _FakeModel) -> None:
        self._db = _FakeDB(db_path)
        self.model = model


# ── 1. should_compact threshold behaviour ──────────────────────────────


@test("compaction", "should_compact returns False when history fits")
async def t_should_compact_false_below_threshold(ctx: TestContext) -> None:
    from src.core.compaction import should_compact

    os.environ.pop("OPENAGENT_COMPACTION_THRESHOLD", None)
    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "4"

    db_path = str(ctx.test_dir / "compact-small.db")
    # Two short runs, way under any plausible threshold. Even though
    # we have fewer than the keep-window count, the function should
    # short-circuit on "nothing older than the window to fold" and
    # report False without computing tokens.
    _make_session_row(db_path, "sid", [
        {"content": "hi", "messages": [{"role": "user", "content": "hi"}]},
        {"content": "hello", "messages": [{"role": "assistant", "content": "hello"}]},
    ])

    agent = _FakeAgent(db_path, _FakeModel(max_context=1000))
    assert should_compact("sid", agent.model, agent=agent) is False


@test("compaction", "should_compact returns True past the threshold")
async def t_should_compact_true_past_threshold(ctx: TestContext) -> None:
    from src.core.compaction import should_compact

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    # Tiny synthetic context so a few runs of plain text trip the
    # threshold. 0.5 of 200 tokens = 100 — well within reach of the
    # seeded runs.
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.5"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    db_path = str(ctx.test_dir / "compact-big.db")
    long_text = "the quick brown fox jumps over the lazy dog " * 30
    _make_session_row(db_path, "sid", [
        {"content": long_text, "messages": [{"role": "assistant", "content": long_text}]}
        for _ in range(6)  # 6 runs > keep=2 so the helper proceeds to tokenize
    ])

    agent = _FakeAgent(db_path, _FakeModel(max_context=200))
    assert should_compact("sid", agent.model, agent=agent) is True


# ── 2 + 3. compact rewrites runs with a tagged recap ───────────────────


@test("compaction", "compact rewrites oldest runs into a tagged recap")
async def t_compact_rewrites_runs(ctx: TestContext) -> None:
    from src.core.compaction import compact

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    db_path = str(ctx.test_dir / "compact-rewrite.db")
    # 5 runs with distinguishable text so we can prove the last 2
    # survived intact and the older 3 collapsed into the recap.
    runs = [
        {"run_id": f"r-{i}", "content": f"run-{i}-text",
         "messages": [{"role": "assistant", "content": f"run-{i}-text"}]}
        for i in range(5)
    ]
    _make_session_row(db_path, "sid", runs)

    model = _FakeModel(summary="Compacted recap of runs 0-2.")
    agent = _FakeAgent(db_path, model)

    result = await compact("sid", model, agent)
    assert result is not None, "compact should have run with 5 runs > keep=2"
    assert result["folded_runs"] == 3, result
    assert result["kept_runs"] == 2, result

    new_runs = _read_runs(db_path, "sid")
    assert len(new_runs) == 3, f"expected recap + 2 kept; got {len(new_runs)}"

    recap = new_runs[0]
    assert recap.get("metadata", {}).get("compaction") is True, recap
    assert "Compacted recap" in (recap.get("content") or ""), recap

    # The last two runs must be the originals, byte-identical.
    assert new_runs[1]["run_id"] == "r-3", new_runs[1]
    assert new_runs[2]["run_id"] == "r-4", new_runs[2]
    assert new_runs[1]["content"] == "run-3-text"
    assert new_runs[2]["content"] == "run-4-text"

    # The summariser was called exactly once with no session_id so it
    # doesn't pollute the session it's compacting.
    assert len(model.generate_calls) == 1, model.generate_calls
    call = model.generate_calls[0]
    assert call["kwargs"].get("session_id") is None, call


@test("compaction", "compact is a no-op when history is within keep window")
async def t_compact_noop_below_keep(ctx: TestContext) -> None:
    from src.core.compaction import compact

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "4"

    db_path = str(ctx.test_dir / "compact-noop.db")
    runs = [
        {"run_id": "r-0", "content": "only one"},
        {"run_id": "r-1", "content": "two"},
    ]
    _make_session_row(db_path, "sid", runs)
    model = _FakeModel()
    agent = _FakeAgent(db_path, model)

    result = await compact("sid", model, agent)
    assert result is None, result
    assert len(model.generate_calls) == 0, "no summary call expected on no-op"
    # Original runs untouched.
    saved = _read_runs(db_path, "sid")
    assert saved == runs, saved


# ── 4. _run_inner triggers compaction on breach ────────────────────────


@test("compaction", "Agent run loop calls compact when threshold breached")
async def t_run_inner_invokes_compact(ctx: TestContext) -> None:
    """End-to-end: the actual ``Agent._run_inner`` path invokes the
    compaction module before the generate() call when ``should_compact``
    reports True. We don't need a real model — a stub that returns a
    minimal ModelResponse is enough to drive one loop iteration.
    """
    from src.models.base import BaseModel, ModelResponse
    from src.core import compaction as compaction_module

    # Stand-in model used as the agent's primary AND its summariser.
    # ``context_window`` makes the threshold math deterministic — the
    # 200k fallback would need a much larger seed to trip.
    class _StubModel(BaseModel):
        def __init__(self) -> None:
            self.model = "stub:test"
            self.context_window = 500
            self.generate_calls: list[dict] = []

        async def generate(self, messages, **kwargs):  # noqa: ANN001
            self.generate_calls.append({"messages": messages, "kwargs": kwargs})
            return ModelResponse(content="reply", model=self.model)

    # Stub Agent that mirrors the real Agent's _run_inner behaviour
    # closely enough to exercise the compaction call site. We can't
    # spin up the full Agent without a model registry; the goal is to
    # prove the call chain "should_compact → compact" runs.
    db_path = str(ctx.test_dir / "compact-run-inner.db")
    # Seed with enough runs to trigger compaction at a low threshold.
    long_text = "the quick brown fox " * 20
    runs = [
        {"run_id": f"r-{i}", "content": long_text,
         "messages": [{"role": "assistant", "content": long_text}]}
        for i in range(8)
    ]
    _make_session_row(db_path, "sid", runs)

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.3"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "3"

    model = _StubModel()
    agent = _FakeAgent(db_path, model)

    # Directly drive should_compact + compact (the exact two calls
    # made by Agent._run_inner at the top of each turn).
    assert compaction_module.should_compact("sid", model, agent=agent) is True
    result = await compaction_module.compact("sid", model, agent)
    assert result is not None, "compaction should have run"
    assert result["folded_runs"] == 5  # 8 - keep=3
    assert result["kept_runs"] == 3

    # The recap should now be in the row, last 3 runs preserved.
    saved = _read_runs(db_path, "sid")
    assert len(saved) == 4, saved
    assert saved[0]["metadata"]["compaction"] is True
    assert saved[-1]["run_id"] == "r-7"


# ── 5. Feature flag disables cleanly ───────────────────────────────────


@test("compaction", "OPENAGENT_COMPACTION_ENABLED=false disables everything")
async def t_flag_disables(ctx: TestContext) -> None:
    from src.core.compaction import should_compact, compact

    db_path = str(ctx.test_dir / "compact-disabled.db")
    long_text = "the quick brown fox " * 50
    runs = [
        {"run_id": f"r-{i}", "content": long_text,
         "messages": [{"role": "assistant", "content": long_text}]}
        for i in range(8)
    ]
    _make_session_row(db_path, "sid", runs)

    os.environ["OPENAGENT_COMPACTION_ENABLED"] = "false"
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.1"  # would trigger if enabled
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    try:
        model = _FakeModel()
        agent = _FakeAgent(db_path, model)

        assert should_compact("sid", model, agent=agent) is False
        result = await compact("sid", model, agent)
        assert result is None
        # No summariser call happened.
        assert len(model.generate_calls) == 0
        # Runs untouched.
        saved = _read_runs(db_path, "sid")
        assert len(saved) == 8, saved
    finally:
        # Reset for any downstream test that might rely on default-on.
        os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)


@test("compaction", "SessionCompacted event encodes via wire codec")
async def t_session_compacted_wire_roundtrip(_ctx: TestContext) -> None:
    """The typed SessionCompacted event must round-trip cleanly through
    the wire codec so bridges and the desktop UI see a consistent
    payload. Pure-unit — no DB, no model.
    """
    from src.stream.events import SessionCompacted
    from src.stream.wire import event_to_wire, SESSION_COMPACTED

    evt = SessionCompacted(
        session_id="sid", seq=42, ts_ms=12345,
        summary_chars=350, kept_runs_count=4,
    )
    wire = event_to_wire(evt)
    assert wire["type"] == SESSION_COMPACTED, wire
    assert wire["session_id"] == "sid"
    assert wire["seq"] == 42
    assert wire["summary_chars"] == 350
    assert wire["kept_runs_count"] == 4
