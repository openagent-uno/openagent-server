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
        self.forgotten_sessions: list[str] = []

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        self.generate_calls.append({"messages": messages, "kwargs": kwargs})
        # Mimic ModelResponse just enough — only ``content`` is read.
        class _R:  # noqa: D401
            pass
        r = _R()
        r.content = self._summary
        return r

    async def forget_session(self, session_id: str) -> None:
        self.forgotten_sessions.append(session_id)


class _FakeAgent:
    """Tiny stand-in for ``Agent`` that carries the right hooks.

    The compaction module only needs ``_db`` (for ``db_path``) and
    ``model``. Tests don't go through the full Agent.run path.
    """

    def __init__(self, db_path: str, model: _FakeModel) -> None:
        self._db = _FakeDB(db_path)
        self.model = model


class _BlockingModel(_FakeModel):
    """Provider that never returns unless its task is cancelled."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        self.generate_calls.append({"messages": messages, "kwargs": kwargs})
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@test("compaction", "start-of-turn repairs stale history before measuring it")
async def t_start_of_turn_recovers_before_compaction(ctx: TestContext) -> None:
    from src.core import compaction

    class _RecoveringDB:
        db_path = str(ctx.test_dir / "missing-recovery.db")

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.journal_calls: list[tuple[str, str | None]] = []

        async def recover_stale_session_runs(self, session_id: str) -> int:
            self.calls.append(session_id)
            return 1

        async def recover_session_from_journal(
            self, session_id: str, *, current_text: str | None = None,
        ) -> int:
            self.journal_calls.append((session_id, current_text))
            return 2

    model = _FakeModel()
    db = _RecoveringDB()
    agent = _FakeAgent(db.db_path, model)
    agent._db = db

    registered = await compaction.run_start_of_turn(
        "sid", model, agent, None, current_message="continue",
    )
    assert registered is True
    assert db.calls == ["sid"]
    assert db.journal_calls == [("sid", "continue")]
    compaction.mark_turn_done("sid")


@test("compaction", "an overlapping live turn is never mistaken for stale")
async def t_start_of_turn_does_not_recover_live_overlap(ctx: TestContext) -> None:
    from src.core import compaction

    class _RecoveringDB:
        db_path = str(ctx.test_dir / "missing-overlap.db")

        def __init__(self) -> None:
            self.calls = 0

        async def recover_stale_session_runs(self, _session_id: str) -> int:
            self.calls += 1
            return 1

    model = _FakeModel()
    db = _RecoveringDB()
    agent = _FakeAgent(db.db_path, model)
    agent._db = db
    compaction.mark_turn_active("overlap")
    try:
        registered = await compaction.run_start_of_turn(
            "overlap", model, agent, None,
        )
        assert registered is True
        assert db.calls == 0
    finally:
        # Balance the manually-seeded first turn and run_start_of_turn's second.
        compaction.mark_turn_done("overlap")
        compaction.mark_turn_done("overlap")


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
    assert recap.get("status") == "COMPLETED", recap
    assert "Compacted recap" in (recap.get("content") or ""), recap

    # The last two runs must be the originals, byte-identical.
    assert new_runs[1]["run_id"] == "r-3", new_runs[1]
    assert new_runs[2]["run_id"] == "r-4", new_runs[2]
    assert new_runs[1]["content"] == "run-3-text"
    assert new_runs[2]["content"] == "run-4-text"

    # The summariser must run in a unique disposable session.  Passing None
    # used to resolve to the shared 9.8 MB ``default`` runtime session in
    # production, silently injecting that history into this tiny prompt.
    assert len(model.generate_calls) == 1, model.generate_calls
    call = model.generate_calls[0]
    summary_sid = call["kwargs"].get("session_id")
    assert summary_sid.startswith("compaction:"), call
    assert summary_sid not in {"default", "sid"}, call
    assert model.forgotten_sessions == [summary_sid], model.forgotten_sessions


@test("compaction", "summary transcript excludes system and copied history")
async def t_extracts_only_current_turn(_ctx: TestContext) -> None:
    from src.core.compaction import _extract_run_text

    system_prompt = "FRAMEWORK INSTRUCTIONS " * 200
    text = _extract_run_text({
        "input": {"input_content": "Keep the blue launch date."},
        "content": "Decision saved.",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Old copied question", "from_history": True},
            {"role": "assistant", "content": "Old copied answer", "from_history": True},
            {"role": "user", "content": "Keep the blue launch date."},
            {"role": "assistant", "content": "Decision saved."},
        ],
    })

    assert "FRAMEWORK INSTRUCTIONS" not in text, text[:500]
    assert "Old copied question" not in text, text
    assert "Old copied answer" not in text, text
    assert text.count("Keep the blue launch date.") == 1, text
    assert text.count("Decision saved.") == 1, text


@test("compaction", "compaction refreshes the normalized client transcript")
async def t_compaction_projects_operational_history(ctx: TestContext) -> None:
    from src.core.compaction import compact
    from src.memory.db import MemoryDB

    db_path = str(ctx.test_dir / "compact-project-v2.db")
    db = MemoryDB(db_path)
    await db.connect()
    now = int(time.time())
    runs = [
        {
            "run_id": f"r-{i}",
            "status": "COMPLETED",
            "content": f"answer-{i}",
            "messages": [
                {"role": "user", "content": f"question-{i}"},
                {"role": "assistant", "content": f"answer-{i}"},
            ],
            "created_at": now + i,
        }
        for i in range(5)
    ]
    conn = await db._ensure_connected()
    await conn.execute(
        "INSERT INTO sessions "
        "(session_id, session_type, agent_id, user_id, metadata, runs, created_at, updated_at) "
        "VALUES ('sid', 'agent', 'a', 'owner', '{}', ?, ?, ?)",
        (json.dumps(runs), now, now),
    )
    await db._project_operational_session("sid")
    await conn.commit()

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"
    model = _FakeModel(summary="Recap of questions zero through two.")
    agent = _FakeAgent(db_path, model)
    agent._db = db
    result = await compact("sid", model, agent)
    assert result is not None

    rows = await (
        await conn.execute(
            "SELECT role, text FROM session_messages "
            "WHERE session_id='sid' ORDER BY sequence",
        )
    ).fetchall()
    projected = [(row[0], row[1]) for row in rows]
    # The operational repository stores the recap's synthetic user/assistant
    # pair; the history API later renders its metadata as one compaction card.
    assert projected[1] == (
        "assistant", "Recap of questions zero through two.",
    ), projected
    assert not any("question-0" in (text or "") for _, text in projected), projected
    assert ("user", "question-3") in projected
    assert ("assistant", "answer-4") in projected
    await db.close()


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


# ── 6. Live status plumbing: running → done envelopes ──────────────────


@test("compaction", "compact emits running→done status envelopes with stats")
async def t_compact_status_envelopes(ctx: TestContext) -> None:
    """``compact`` must fire a ``phase=running`` hint BEFORE the summariser
    round-trip and a ``phase=done`` hint after, carrying the run/token
    stats — that's what lets a client show "Compacting conversation" →
    "Compacted conversation" instead of unexplained latency. The same
    stats must persist in the recap row's metadata so a reopened session
    rebuilds the identical card.
    """
    from src.core.compaction import compact
    from src.channels.base import parse_compaction_status

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    db_path = str(ctx.test_dir / "compact-status.db")
    runs = [
        {"run_id": f"r-{i}", "content": f"run-{i}-text " * 5,
         "messages": [{"role": "assistant", "content": f"run-{i}-text " * 5}]}
        for i in range(5)
    ]
    _make_session_row(db_path, "sid", runs)
    model = _FakeModel(summary="Recap of the earlier turns.")
    agent = _FakeAgent(db_path, model)

    seen: list[dict] = []

    async def on_status(raw: str) -> None:
        # compact() must only ever push compaction envelopes through
        # on_status — never a bare tool line — so parsing must succeed.
        parsed = parse_compaction_status(raw)
        assert parsed is not None, raw
        seen.append(parsed)

    result = await compact("sid", model, agent, on_status=on_status)
    assert result is not None

    assert [p["phase"] for p in seen] == ["running", "done"], seen
    running, done = seen
    assert running["folded_runs"] == 3, running
    assert running["kept_runs_count"] == 2, running
    assert running["tokens_before"] > 0, running
    assert done["folded_runs"] == 3, done
    assert done["summary_chars"] == len("Recap of the earlier turns."), done
    assert done["tokens_after"] > 0, done

    # The returned dict exposes the token stats too (manual /compact reads it).
    assert result["tokens_before"] == running["tokens_before"], result
    assert result["tokens_after"] == done["tokens_after"], result

    # Recap metadata persists every stat for reopen/reconcile parity.
    saved = _read_runs(db_path, "sid")
    meta = saved[0]["metadata"]
    assert meta["compaction"] is True, meta
    assert meta["folded_runs"] == 3, meta
    assert meta["kept_runs_count"] == 2, meta
    assert meta["tokens_before"] == running["tokens_before"], meta
    assert meta["tokens_after"] == done["tokens_after"], meta
    assert meta["summary_chars"] == done["summary_chars"], meta


@test("compaction", "compact emits running→error when the summary is empty")
async def t_compact_status_error(ctx: TestContext) -> None:
    """An empty summary aborts the fold — but only AFTER we've announced
    ``running``. The terminal ``error`` hint must fire so a client that
    showed "Compacting…" doesn't hang on it. The runs stay untouched.
    """
    from src.core.compaction import compact
    from src.channels.base import parse_compaction_status

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    db_path = str(ctx.test_dir / "compact-status-err.db")
    runs = [{"run_id": f"r-{i}", "content": f"c{i} " * 5} for i in range(5)]
    _make_session_row(db_path, "sid", runs)
    model = _FakeModel(summary="")  # empty recap → skip the fold
    agent = _FakeAgent(db_path, model)

    phases: list[str] = []

    async def on_status(raw: str) -> None:
        parsed = parse_compaction_status(raw)
        if parsed is not None:
            phases.append(parsed["phase"])

    result = await compact("sid", model, agent, on_status=on_status)
    assert result is None
    assert phases == ["running", "error"], phases
    # Nothing folded — the row is byte-identical.
    assert len(_read_runs(db_path, "sid")) == 5


@test("compaction", "a failed DB rewrite cannot report compaction done")
async def t_compact_save_failure_is_error(ctx: TestContext) -> None:
    from src.core import compaction
    from src.channels.base import parse_compaction_status

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"
    db_path = str(ctx.test_dir / "compact-save-failed.db")
    runs = [{"run_id": f"r-{i}", "content": f"turn {i}"} for i in range(5)]
    _make_session_row(db_path, "sid", runs)
    model = _FakeModel(summary="A valid summary that cannot be committed.")
    agent = _FakeAgent(db_path, model)
    phases: list[str] = []

    async def on_status(raw: str) -> None:
        parsed = parse_compaction_status(raw)
        if parsed is not None:
            phases.append(parsed["phase"])

    original_save = compaction._save_runs
    compaction._save_runs = lambda *_args, **_kwargs: False
    try:
        result = await compaction.compact("sid", model, agent, on_status=on_status)
    finally:
        compaction._save_runs = original_save

    assert result is None
    assert phases == ["running", "error"], phases
    assert _read_runs(db_path, "sid") == runs


@test("compaction", "parse_compaction_status normalises phase + stats")
async def t_parse_compaction_status(_ctx: TestContext) -> None:
    """The shared envelope parser is the seam the turn runner and the
    bridges rely on. It must coerce stats to ints, default a missing
    phase to ``done`` (legacy pre-phase envelopes), and reject anything
    that isn't a compaction envelope.
    """
    from src.channels.base import parse_compaction_status

    full = parse_compaction_status(json.dumps({
        "kind": "session.compacted", "phase": "done",
        "folded_runs": 3, "kept_runs_count": 2, "summary_chars": 120,
        "tokens_before": 900, "tokens_after": 60,
    }))
    assert full == {
        "phase": "done", "folded_runs": 3, "kept_runs_count": 2,
        "summary_chars": 120, "tokens_before": 900, "tokens_after": 60,
    }, full

    # Legacy envelope without a phase → treated as done.
    legacy = parse_compaction_status(json.dumps({
        "kind": "session.compacted", "summary_chars": 5, "kept_runs_count": 4,
    }))
    assert legacy["phase"] == "done", legacy
    assert legacy["kept_runs_count"] == 4, legacy

    assert parse_compaction_status(json.dumps({
        "kind": "session.compacted", "phase": "running",
    }))["phase"] == "running"
    # Unknown phase coerces to done rather than leaking through.
    assert parse_compaction_status(json.dumps({
        "kind": "session.compacted", "phase": "weird",
    }))["phase"] == "done"

    # Non-envelopes → None (a tool event, a plain status, empty, other kind).
    assert parse_compaction_status(json.dumps({"tool_name": "bash"})) is None
    assert parse_compaction_status("Thinking...") is None
    assert parse_compaction_status("") is None
    assert parse_compaction_status(json.dumps({"kind": "other"})) is None


@test("compaction", "SessionCompacted round-trips phase + token stats")
async def t_session_compacted_full_roundtrip(_ctx: TestContext) -> None:
    """The extended SessionCompacted fields (phase + folded_runs + token
    counts) must survive both directions of the wire codec — the CLI and
    bridge listeners parse the frame back via ``wire_to_event``.
    """
    from src.stream.events import SessionCompacted
    from src.stream.wire import event_to_wire, wire_to_event, SESSION_COMPACTED

    evt = SessionCompacted(
        session_id="sid", seq=7, ts_ms=999,
        phase="done", folded_runs=5, kept_runs_count=3,
        summary_chars=420, tokens_before=1800, tokens_after=90,
    )
    wire = event_to_wire(evt)
    assert wire["type"] == SESSION_COMPACTED, wire
    assert wire["phase"] == "done", wire
    assert wire["folded_runs"] == 5, wire
    assert wire["tokens_before"] == 1800, wire
    assert wire["tokens_after"] == 90, wire

    back = wire_to_event(wire)
    assert isinstance(back, SessionCompacted), back
    assert back.phase == "done"
    assert back.folded_runs == 5
    assert back.kept_runs_count == 3
    assert back.summary_chars == 420
    assert back.tokens_before == 1800
    assert back.tokens_after == 90

    # A running frame (partial stats) round-trips too.
    r = wire_to_event(event_to_wire(SessionCompacted(
        session_id="s", phase="running", folded_runs=2, tokens_before=500,
    )))
    assert r.phase == "running", r
    assert r.folded_runs == 2, r
    assert r.tokens_before == 500, r


@test("compaction", "recap run rehydrates as a compaction message")
async def t_recap_rehydrates_as_compaction(_ctx: TestContext) -> None:
    """On reopen/reconcile the recap run must surface as a ``compaction``
    message (the same card the live frame draws) — NOT as a bare
    assistant bubble leaking the recap paragraph. Exercises the gateway's
    ``_expand_run_messages`` directly.
    """
    from src.gateway.api.sessions import _expand_run_messages

    recap = {
        "run_id": "compaction-1",
        "content": "Recap paragraph.",
        "messages": [
            {"role": "system", "content": "[compacted session recap]"},
            {"role": "assistant", "content": "Recap paragraph."},
        ],
        "metadata": {
            "compaction": True, "folded_runs": 3, "kept_runs_count": 2,
            "summary_chars": 16, "tokens_before": 900, "tokens_after": 40,
        },
    }
    out = _expand_run_messages(recap, timestamp=123, msg_counter=[0])
    # Exactly one message, and it's the compaction card — the assistant
    # summary must NOT also render as a bubble.
    assert len(out) == 1, out
    m = out[0]
    assert m["role"] == "compaction", m
    comp = m["compaction"]
    assert comp["phase"] == "done", comp
    assert comp["folded_runs"] == 3, comp
    assert comp["kept_runs_count"] == 2, comp
    assert comp["summary_chars"] == 16, comp
    assert comp["tokens_before"] == 900, comp
    assert comp["tokens_after"] == 40, comp


@test("compaction", "manual /compact on a short conversation emits a no-op feedback frame")
async def t_compact_noop_emits_feedback(ctx: TestContext) -> None:
    """A hand-typed /compact on a conversation too short to fold must still
    tell the UI something happened — the app reads the typed frame, not the
    command_result, so a silent early-return leaves the command looking
    broken. ``compact`` emits a terminal ``done`` frame with
    ``folded_runs=0`` (rendered as "Already compact — nothing to fold")
    while still doing no summariser work.
    """
    from src.core.compaction import compact
    from src.channels.base import parse_compaction_status

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "4"

    db_path = str(ctx.test_dir / "compact-noop-feedback.db")
    _make_session_row(db_path, "sid", [
        {"run_id": f"r-{i}", "content": f"c{i}"} for i in range(2)
    ])
    model = _FakeModel()
    agent = _FakeAgent(db_path, model)

    seen: list[dict] = []

    async def on_status(raw: str) -> None:
        parsed = parse_compaction_status(raw)
        if parsed is not None:
            seen.append(parsed)

    result = await compact("sid", model, agent, on_status=on_status)
    assert result is None
    # No summariser round-trip — there was nothing to fold.
    assert len(model.generate_calls) == 0, model.generate_calls
    # But exactly one terminal feedback frame lands: done, zero folded.
    assert len(seen) == 1, seen
    assert seen[0]["phase"] == "done", seen
    assert seen[0]["folded_runs"] == 0, seen
    assert seen[0]["kept_runs_count"] == 2, seen


# ── 7. Claude-Code-style manual /compact (keep=0 folds everything) ─────


@test("compaction", "manual /compact (keep=0) folds the WHOLE conversation")
async def t_compact_keep_zero_folds_all(ctx: TestContext) -> None:
    """The manual /compact command passes ``keep=0`` so it behaves like
    Claude Code: it folds the entire conversation into one recap and keeps
    nothing verbatim, even when the chat is far shorter than the automatic
    keep window (which would otherwise no-op).
    """
    from src.core.compaction import compact

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    # A keep window that WOULD no-op the automatic path on 2 runs...
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "4"

    db_path = str(ctx.test_dir / "compact-keep0.db")
    runs = [
        {"run_id": f"r-{i}", "content": f"turn-{i} text " * 5,
         "messages": [{"role": "assistant", "content": f"turn-{i} text " * 5}]}
        for i in range(2)
    ]
    _make_session_row(db_path, "sid", runs)
    model = _FakeModel(summary="Whole-conversation recap.")
    agent = _FakeAgent(db_path, model)

    # ...but keep=0 folds them anyway.
    result = await compact("sid", model, agent, keep=0)
    assert result is not None, "keep=0 must fold even a short conversation"
    assert result["folded_runs"] == 2, result
    assert result["kept_runs"] == 0, result

    # The session is now JUST the recap — nothing kept verbatim.
    saved = _read_runs(db_path, "sid")
    assert len(saved) == 1, saved
    assert saved[0]["metadata"]["compaction"] is True, saved
    assert "Whole-conversation recap" in (saved[0]["content"] or ""), saved


@test("compaction", "manual /compact (keep=0) no-ops when only a recap remains")
async def t_compact_keep_zero_noop_on_recap(ctx: TestContext) -> None:
    """Folding an already-compacted session (only a recap run) into another
    recap gains nothing, so keep=0 must no-op rather than loop — otherwise
    a repeated /compact would summarise the summary forever.
    """
    from src.core.compaction import compact

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)

    db_path = str(ctx.test_dir / "compact-keep0-recap.db")
    _make_session_row(db_path, "sid", [
        {"run_id": "compaction-1", "content": "existing recap",
         "metadata": {"compaction": True, "folded_runs": 3}},
    ])
    model = _FakeModel()
    agent = _FakeAgent(db_path, model)

    result = await compact("sid", model, agent, keep=0)
    assert result is None, "folding a lone recap into a recap must no-op"
    assert len(model.generate_calls) == 0, model.generate_calls
    saved = _read_runs(db_path, "sid")
    assert len(saved) == 1 and saved[0]["run_id"] == "compaction-1", saved


# ── 6. the cost ceiling: a huge context window must not license a huge bill ──


@test("compaction", "the cost ceiling compacts a 1M-window model long before its window")
async def t_cost_ceiling_bites_before_the_window(ctx: TestContext) -> None:
    """The regression behind the 2026-07-13 burn.

    The threshold is a fraction of the MODEL's context window, which answers
    "will this overflow?" and not "what will this cost?". On a 1M-token model
    that licensed ~786k tokens of accumulated history — and history is re-sent
    on every step of the agentic loop and every delivery bound to the session.
    Sessions reached 16M input tokens for one support thread.

    So compaction now trips on whichever comes first: the window, or the cost
    ceiling. Same session, same model: unbounded ceiling → no compaction;
    realistic ceiling → compaction.
    """
    from src.core.compaction import should_compact

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.75"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    db_path = str(ctx.test_dir / "compact-ceiling.db")
    # ~10k tokens of history: nowhere near 75% of a 1M window (750k), but well
    # past a 5k-token cost ceiling.
    long_text = "the quick brown fox jumps over the lazy dog " * 400
    _make_session_row(db_path, "sid", [
        {"content": long_text, "messages": [{"role": "assistant", "content": long_text}]}
        for _ in range(6)
    ])
    agent = _FakeAgent(db_path, _FakeModel(max_context=1_000_000))

    try:
        # A 1M-token model's own window would never trip on this history.
        os.environ["OPENAGENT_COMPACTION_MAX_HISTORY_TOKENS"] = "10000000"
        assert should_compact("sid", agent.model, agent=agent) is False

        # With a realistic ceiling, the same session compacts.
        os.environ["OPENAGENT_COMPACTION_MAX_HISTORY_TOKENS"] = "5000"
        assert should_compact("sid", agent.model, agent=agent) is True
    finally:
        os.environ.pop("OPENAGENT_COMPACTION_MAX_HISTORY_TOKENS", None)


@test("compaction", "history tool-result elision is OFF by default (byte-identical)")
async def t_history_elide_off_by_default(ctx: TestContext) -> None:
    from src.core.compaction import _trim_kept_tool_results

    os.environ.pop("OPENAGENT_COMPACTION_HISTORY_TOOL_RESULT_CHARS", None)
    kept = [
        {"messages": [{"role": "tool", "tool_name": "vault_search",
                       "content": "X" * 500_000}]},
        {"messages": [{"role": "tool", "tool_name": "vault_search",
                       "content": "Y" * 500_000}]},
    ]
    out, n, chars = _trim_kept_tool_results(kept)
    assert out is kept and n == 0 and chars == 0, "default must be a no-op"


@test("compaction", "history tool-result elision trims OLD kept runs, spares the most recent")
async def t_history_elide_trims_old_keeps_recent(ctx: TestContext) -> None:
    from src.core.compaction import _trim_kept_tool_results

    os.environ["OPENAGENT_COMPACTION_HISTORY_TOOL_RESULT_CHARS"] = "1000"
    try:
        big = "X" * 50_000
        kept = [
            {"run_id": "old", "messages": [
                {"role": "assistant", "content": "let me search"},
                {"role": "tool", "tool_name": "vault_search",
                 "tool_call_id": "c1", "content": big},
            ]},
            {"run_id": "recent", "messages": [
                {"role": "tool", "tool_name": "vault_search",
                 "tool_call_id": "c2", "content": big},
            ]},
        ]
        out, n, chars = _trim_kept_tool_results(kept)
        assert n == 1 and chars == 50_000, (n, chars)
        old_tool = out[0]["messages"][1]
        # role + tool_call_id preserved (the assistant/tool pairing must survive)
        assert old_tool["role"] == "tool" and old_tool["tool_call_id"] == "c1"
        assert "elided from history" in old_tool["content"]
        assert "re-run `vault_search`" in old_tool["content"]  # re-fetch pointer
        assert len(old_tool["content"]) < 1000                 # collapsed small
        assert out[0]["messages"][0]["content"] == "let me search"  # non-tool untouched
        # the MOST RECENT run keeps its full tool output (still live context)
        assert out[1]["messages"][0]["content"] == big
    finally:
        os.environ.pop("OPENAGENT_COMPACTION_HISTORY_TOOL_RESULT_CHARS", None)


@test("compaction", "history tool-result elision keeps normal results + non-text blocks")
async def t_history_elide_preserves_normal_and_blocks(ctx: TestContext) -> None:
    from src.core.compaction import _trim_kept_tool_results, _elide_tool_content

    os.environ["OPENAGENT_COMPACTION_HISTORY_TOOL_RESULT_CHARS"] = "1000"
    try:
        kept = [
            {"messages": [{"role": "tool", "tool_name": "t", "content": "small ok"}]},
            {"messages": [{"role": "user", "content": "next"}]},
        ]
        out, n, _ = _trim_kept_tool_results(kept)
        assert n == 0 and out[0]["messages"][0]["content"] == "small ok"  # small = untouched
        # list content-block form: image block PRESERVED, oversized text collapsed
        blocks = [{"type": "image", "source": "…"}, {"type": "text", "text": "Z" * 5000}]
        new_c, elided = _elide_tool_content(blocks, "docs_search", 1000)
        assert elided == 5000
        assert any(isinstance(b, dict) and b.get("type") == "image" for b in new_c), \
            "image block was dropped — data loss"
        assert any("elided from history" in b.get("text", "") for b in new_c)
    finally:
        os.environ.pop("OPENAGENT_COMPACTION_HISTORY_TOOL_RESULT_CHARS", None)


# ── 8. Proactive (background) compaction ───────────────────────────────
#
# The reactive path above blocks the USER on the summariser. The proactive
# path (``compact_after_turn`` + the per-session lock / active-turn guard)
# moves the fold OFF the critical path AFTER a turn completes, while
# guaranteeing a background fold never overlaps a turn's history read/write for
# the same session. These tests pin that contract directly (no full Agent).


def _seed_breaching(db_path: str, session_id: str, *, runs: int = 6) -> None:
    """Seed a session whose history trips should_compact at the tiny-context
    settings used below (max_context=200, threshold=0.5, keep=2)."""
    long_text = "the quick brown fox jumps over the lazy dog " * 30
    _make_session_row(db_path, session_id, [
        {"content": long_text,
         "messages": [{"role": "assistant", "content": long_text}]}
        for _ in range(runs)
    ])


class _SlotAgent(_FakeAgent):
    """``_FakeAgent`` that also implements the model-slot counter, so we can
    prove the background pass keeps the model alive across the summariser call
    and releases it exactly once."""

    def __init__(self, db_path: str, model: _FakeModel) -> None:
        super().__init__(db_path, model)
        self.acquired = 0
        self.released = 0

    def _acquire_model_slot(self, model):  # noqa: ANN001
        self.acquired += 1
        return model

    def _release_model_slot(self, model):  # noqa: ANN001
        self.released += 1


@test("compaction", "proactive: compact_after_turn folds in the background when breached")
async def t_compact_after_turn_folds(ctx: TestContext) -> None:
    from src.core import compaction

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.5"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    sid = "proactive-fold"
    db_path = str(ctx.test_dir / "proactive-fold.db")
    _seed_breaching(db_path, sid)
    model = _FakeModel(max_context=200, summary="Background recap.")
    agent = _FakeAgent(db_path, model)

    task = compaction.compact_after_turn(sid, model, agent)
    assert task is not None, "a breaching session must schedule a background pass"
    await task  # drain the background work the user never waited on

    # The fold landed: recap + last 2 kept, summariser called once in an
    # isolated disposable runtime session.
    saved = _read_runs(db_path, sid)
    assert len(saved) == 3, saved
    assert saved[0].get("metadata", {}).get("compaction") is True, saved[0]
    assert len(model.generate_calls) == 1, model.generate_calls
    summary_sid = model.generate_calls[0]["kwargs"].get("session_id")
    assert summary_sid.startswith("compaction:"), summary_sid
    assert summary_sid not in {"default", sid}, summary_sid
    assert model.forgotten_sessions == [summary_sid], model.forgotten_sessions
    # The in-flight guard cleared itself so the next turn can schedule again.
    assert sid not in compaction._INFLIGHT_SESSIONS


@test("compaction", "proactive: compact_after_turn no-ops when disabled or session-less")
async def t_compact_after_turn_noops(ctx: TestContext) -> None:
    from src.core import compaction

    db_path = str(ctx.test_dir / "proactive-noop.db")
    _seed_breaching(db_path, "sid")
    model = _FakeModel(max_context=200)
    agent = _FakeAgent(db_path, model)

    # No session id → nothing to key on.
    assert compaction.compact_after_turn(None, model, agent) is None

    # Feature flag off → inert, allocates no task.
    os.environ["OPENAGENT_COMPACTION_ENABLED"] = "false"
    try:
        assert compaction.compact_after_turn("sid", model, agent) is None
        assert len(model.generate_calls) == 0
    finally:
        os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)


@test("compaction", "proactive: background pass SKIPS while a turn is active (no race)")
async def t_background_skips_active_turn(ctx: TestContext) -> None:
    """Invariant #2: a background fold must not run while a turn for the same
    session is reading/writing history — that turn will fire its own post-turn
    pass. We simulate an in-progress turn with ``mark_turn_active`` and prove
    the scheduled pass observes it and leaves the runs byte-identical."""
    from src.core import compaction

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.5"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    sid = "proactive-active"
    db_path = str(ctx.test_dir / "proactive-active.db")
    _seed_breaching(db_path, sid)
    before = _read_runs(db_path, sid)
    model = _FakeModel(max_context=200, summary="should not be written")
    agent = _FakeAgent(db_path, model)

    # A turn is active for this session (as the turn loop would register it).
    compaction.mark_turn_active(sid)
    try:
        task = compaction.compact_after_turn(sid, model, agent)
        assert task is not None
        await task
        # Skipped: no summariser call, runs untouched.
        assert len(model.generate_calls) == 0, model.generate_calls
        assert _read_runs(db_path, sid) == before
    finally:
        compaction.mark_turn_done(sid)


@test("compaction", "proactive: only one background compaction per session at a time")
async def t_background_dedup(ctx: TestContext) -> None:
    from src.core import compaction

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.5"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    sid = "proactive-dedup"
    db_path = str(ctx.test_dir / "proactive-dedup.db")
    _seed_breaching(db_path, sid)
    model = _FakeModel(max_context=200, summary="Dedup recap.")
    agent = _FakeAgent(db_path, model)

    # Two fire-and-forget calls before the first task runs: the second must
    # dedup to None because the session is already in-flight.
    first = compaction.compact_after_turn(sid, model, agent)
    second = compaction.compact_after_turn(sid, model, agent)
    assert first is not None
    assert second is None, "a second concurrent pass for one session must dedup"
    await first
    # Exactly one summariser round-trip happened for the two calls.
    assert len(model.generate_calls) == 1, model.generate_calls


@test("compaction", "proactive: an incoming turn preempts a stuck background fold")
async def t_incoming_turn_preempts_background_fold(ctx: TestContext) -> None:
    """Regression: provider Retry-After used to hold the session lock for
    minutes, making Telegram appear frozen on "Compacting conversation"."""
    from src.core import compaction

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.5"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"
    os.environ["OPENAGENT_COMPACTION_SUMMARY_TIMEOUT_SECONDS"] = "30"

    sid = "proactive-preempt"
    db_path = str(ctx.test_dir / "proactive-preempt.db")
    _seed_breaching(db_path, sid)
    before = _read_runs(db_path, sid)
    model = _BlockingModel(max_context=200)
    agent = _FakeAgent(db_path, model)

    background = compaction.compact_after_turn(sid, model, agent)
    assert background is not None
    await asyncio.wait_for(model.started.wait(), timeout=1)
    assert compaction.session_lock(sid).locked()

    try:
        registered = await asyncio.wait_for(
            compaction.run_start_of_turn(sid, model, agent, None),
            timeout=1,
        )
        assert registered is True
        assert model.cancelled.is_set(), "the provider call must be cancelled"
        assert background.cancelled(), "the detached fold must be cancelled"
        assert len(model.generate_calls) == 1, "the turn must not retry the fold"
        assert _read_runs(db_path, sid) == before, "a cancelled fold writes nothing"
        assert sid not in compaction._INFLIGHT_SESSIONS
        assert sid not in compaction._BACKGROUND_TASKS
    finally:
        compaction.mark_turn_done(sid)
        os.environ.pop("OPENAGENT_COMPACTION_SUMMARY_TIMEOUT_SECONDS", None)


@test("compaction", "summary provider calls stop at the maintenance timeout")
async def t_summary_provider_timeout(ctx: TestContext) -> None:
    from src.core import compaction

    primary = _BlockingModel(max_context=200)
    agent = _FakeAgent(str(ctx.test_dir / "summary-timeout.db"), primary)
    orig_pick = compaction._pick_summary_model
    orig_fb = compaction._summary_fallback_model
    compaction._pick_summary_model = lambda a, *, fallback: primary
    compaction._summary_fallback_model = lambda a, *, exclude_provider: None
    os.environ["OPENAGENT_COMPACTION_SUMMARY_TIMEOUT_SECONDS"] = "0.01"
    try:
        out = await asyncio.wait_for(
            compaction._summarize_runs(_FOLD_RUNS, primary, agent),
            timeout=1,
        )
    finally:
        compaction._pick_summary_model = orig_pick
        compaction._summary_fallback_model = orig_fb
        os.environ.pop("OPENAGENT_COMPACTION_SUMMARY_TIMEOUT_SECONDS", None)

    assert out == ""
    assert primary.cancelled.is_set(), "wait_for must cancel the provider call"
    assert len(primary.generate_calls) == 1


@test("compaction", "proactive: mark_turn_done balances the count and prunes the idle lock")
async def t_mark_turn_done_prunes_lock(_ctx: TestContext) -> None:
    from src.core import compaction

    sid = "proactive-prune"
    # A turn registers (creating the lock on first acquire, which the turn loop
    # does; here we materialise it explicitly) then completes.
    _ = compaction.session_lock(sid)
    compaction.mark_turn_active(sid)
    assert compaction._ACTIVE_TURNS.get(sid) == 1
    assert sid in compaction._SESSION_LOCKS

    compaction.mark_turn_done(sid)
    # Count balanced back to nothing and the idle lock dropped so the map
    # stays bounded across many short-lived sessions.
    assert sid not in compaction._ACTIVE_TURNS
    assert sid not in compaction._SESSION_LOCKS


@test("compaction", "proactive: background pass acquires and releases the model slot")
async def t_background_uses_model_slot(ctx: TestContext) -> None:
    from src.core import compaction

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.5"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    sid = "proactive-slot"
    db_path = str(ctx.test_dir / "proactive-slot.db")
    _seed_breaching(db_path, sid)
    model = _FakeModel(max_context=200, summary="Slot recap.")
    agent = _SlotAgent(db_path, model)

    task = compaction.compact_after_turn(sid, model, agent)
    assert task is not None
    await task
    # The model was pinned for the whole pass and released exactly once, so a
    # concurrent swap/shutdown can't tear it down mid-summary.
    assert agent.acquired == 1, agent.acquired
    assert agent.released == 1, agent.released


# ── 9. Contention progress notice (Telegram/channel) ───────────────────
#
# The owner's UX rule: a BACKGROUND fold (nobody waiting) is INVISIBLE, but a
# turn that arrives WHILE a fold is mid-flight — so it blocks on
# session_lock(session_id) — gets ONE brief "optimizing…" notice explaining the
# pause. run_start_of_turn owns that: it emits the compaction envelope only on
# lock contention, and compact_after_turn (the background path) carries no
# on_status so it can never reach a channel.


def _phase_recorder():
    """Return ``(recorder_list, on_status)`` where on_status parses each
    session.compacted envelope and appends its phase — a stand-in for the
    bridge collector that would render "🗜 Compacting…" → "🗜 Compacted…"."""
    from src.channels.base import parse_compaction_status

    phases: list[str] = []

    async def on_status(raw: str) -> None:
        parsed = parse_compaction_status(raw)
        if parsed is not None:
            phases.append(parsed["phase"])

    return phases, on_status


@test("compaction", "notice: a CONTENDED turn emits exactly one progress notice")
async def t_contended_turn_emits_one_notice(ctx: TestContext) -> None:
    """A turn that starts while a background fold holds the session lock must
    surface exactly one notice bracketing its wait: a single ``running`` then a
    single ``done``, and nothing more (compact()'s own envelopes are suppressed
    so a contended turn never doubles up)."""
    from src.core import compaction

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.5"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    sid = "notice-contended"
    db_path = str(ctx.test_dir / "notice-contended.db")
    _seed_breaching(db_path, sid)
    model = _FakeModel(max_context=200, summary="Recap.")
    agent = _FakeAgent(db_path, model)

    phases, on_status = _phase_recorder()

    # Simulate an in-flight background fold by holding the session lock.
    lock = compaction.session_lock(sid)
    await lock.acquire()
    try:
        turn = asyncio.create_task(
            compaction.run_start_of_turn(sid, model, agent, on_status))
        # The turn detects contention, emits ONE "running", then blocks on the
        # held lock. Let it reach that point.
        for _ in range(200):
            if phases:
                break
            await asyncio.sleep(0)
        assert phases == ["running"], phases
    finally:
        lock.release()

    registered = await turn
    assert registered is True, "a registered turn must report True for its finally"
    # One notice lifecycle: running → done, and NOTHING extra.
    assert phases == ["running", "done"], phases
    compaction.mark_turn_done(sid)  # balance the registration


@test("compaction", "notice: an UNCONTENDED turn stays silent")
async def t_uncontended_turn_is_silent(ctx: TestContext) -> None:
    """The common proactive case: the lock is free and nothing is over
    threshold, so the turn emits no channel notice at all."""
    from src.core import compaction

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "4"

    sid = "notice-silent"
    db_path = str(ctx.test_dir / "notice-silent.db")
    # A short, under-threshold session: no fold, no contention.
    _make_session_row(db_path, sid, [
        {"content": "hi", "messages": [{"role": "user", "content": "hi"}]},
        {"content": "hello", "messages": [{"role": "assistant", "content": "hello"}]},
    ])
    model = _FakeModel(max_context=200)
    agent = _FakeAgent(db_path, model)

    phases, on_status = _phase_recorder()
    registered = await compaction.run_start_of_turn(sid, model, agent, on_status)
    assert registered is True
    assert phases == [], f"an uncontended turn must be silent, got {phases}"
    assert len(model.generate_calls) == 0, "no summariser call on an uncontended turn"
    compaction.mark_turn_done(sid)


@test("compaction", "notice: a BACKGROUND fold with no waiter emits nothing to the channel")
async def t_background_fold_is_silent(ctx: TestContext) -> None:
    """compact_after_turn (the post-turn background trigger, called exactly as
    the turn loop does — with NO on_status) folds without ever routing a status
    to a channel. We spy on the envelope emitter and prove every emission during
    the fold carried a None channel, so nothing could reach the user."""
    from src.core import compaction

    os.environ.pop("OPENAGENT_COMPACTION_ENABLED", None)
    os.environ["OPENAGENT_COMPACTION_THRESHOLD"] = "0.5"
    os.environ["OPENAGENT_COMPACTION_KEEP_RUNS"] = "2"

    sid = "notice-bg-silent"
    db_path = str(ctx.test_dir / "notice-bg-silent.db")
    _seed_breaching(db_path, sid)
    model = _FakeModel(max_context=200, summary="Background recap.")
    agent = _FakeAgent(db_path, model)

    seen: list[tuple[bool, str]] = []
    orig = compaction._emit_compaction_status

    async def spy(on_status, session_id, fields):  # noqa: ANN001
        seen.append((on_status is None, fields.get("phase")))
        return await orig(on_status, session_id, fields)

    compaction._emit_compaction_status = spy
    try:
        task = compaction.compact_after_turn(sid, model, agent)  # no on_status
        assert task is not None
        await task
    finally:
        compaction._emit_compaction_status = orig

    # The fold really happened (recap landed) ...
    saved = _read_runs(db_path, sid)
    assert saved and saved[0].get("metadata", {}).get("compaction") is True, saved
    # ... and every status emission during it had a None channel — invisible.
    assert seen, "compact() should have emitted running/done envelopes internally"
    assert all(is_none for (is_none, _phase) in seen), seen


# ── 6. summariser fallback (proxy-exhaustion resilience) ────────────────
# When the primary summariser call raises (typically an OAuth-limited proxy
# row exhausted under load), compaction must retry ONCE with a distinct
# provider instead of skipping the fold and letting the session balloon.


class _FailModel:
    """Summariser stand-in whose generate() always raises — mimics the
    'No available Claude OAuth accounts' failure of an exhausted proxy row."""

    def __init__(self, model: str = "anthropic:claude-proxy") -> None:
        self.model = model
        self.generate_calls: list[dict] = []

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        self.generate_calls.append({"messages": messages, "kwargs": kwargs})
        raise RuntimeError("No available Claude OAuth accounts")


_FOLD_RUNS = [
    {"content": "hi", "messages": [
        {"role": "user", "content": "the user asked about a billing charge"}]},
]


@test("compaction", "summarise retries with fallback when the primary summariser fails")
async def t_summary_fallback_on_primary_failure(ctx: TestContext) -> None:
    from src.core import compaction

    primary = _FailModel()
    fallback = _FakeModel(model="deepseek:deepseek-chat", summary="Fallback recap.")
    agent = _FakeAgent(str(ctx.test_dir / "fb1.db"), primary)

    orig_pick = compaction._pick_summary_model
    orig_fb = compaction._summary_fallback_model
    compaction._pick_summary_model = lambda a, *, fallback: primary
    compaction._summary_fallback_model = lambda a, *, exclude_provider: fallback
    try:
        out = await compaction._summarize_runs(_FOLD_RUNS, primary, agent)
    finally:
        compaction._pick_summary_model = orig_pick
        compaction._summary_fallback_model = orig_fb

    assert out == "Fallback recap.", out
    assert len(primary.generate_calls) == 1, "primary tried once"
    assert len(fallback.generate_calls) == 1, "fallback tried once"
    # The fallback call, like the primary, must not read/pollute shared history.
    summary_sid = fallback.generate_calls[0]["kwargs"].get("session_id")
    assert summary_sid.startswith("compaction:"), summary_sid
    assert summary_sid != "default", summary_sid
    assert fallback.forgotten_sessions == [summary_sid]


@test("compaction", "summarise returns '' when the primary fails and no fallback exists")
async def t_summary_no_fallback_available(ctx: TestContext) -> None:
    from src.core import compaction

    primary = _FailModel()
    agent = _FakeAgent(str(ctx.test_dir / "fb2.db"), primary)

    orig_pick = compaction._pick_summary_model
    orig_fb = compaction._summary_fallback_model
    compaction._pick_summary_model = lambda a, *, fallback: primary
    compaction._summary_fallback_model = lambda a, *, exclude_provider: None
    try:
        out = await compaction._summarize_runs(_FOLD_RUNS, primary, agent)
    finally:
        compaction._pick_summary_model = orig_pick
        compaction._summary_fallback_model = orig_fb

    assert out == "", "no fallback → skip compaction (original contract)"
    assert len(primary.generate_calls) == 1


@test("compaction", "summarise returns '' when both primary and fallback fail")
async def t_summary_fallback_also_fails(ctx: TestContext) -> None:
    from src.core import compaction

    primary = _FailModel(model="anthropic:claude-proxy")
    fallback = _FailModel(model="deepseek:deepseek-chat")
    agent = _FakeAgent(str(ctx.test_dir / "fb3.db"), primary)

    orig_pick = compaction._pick_summary_model
    orig_fb = compaction._summary_fallback_model
    compaction._pick_summary_model = lambda a, *, fallback: primary
    compaction._summary_fallback_model = lambda a, *, exclude_provider: fallback
    try:
        out = await compaction._summarize_runs(_FOLD_RUNS, primary, agent)
    finally:
        compaction._pick_summary_model = orig_pick
        compaction._summary_fallback_model = orig_fb

    assert out == "", "both failed → skip compaction, never crash the turn"
    assert len(primary.generate_calls) == 1
    assert len(fallback.generate_calls) == 1


@test("compaction", "_provider_of extracts the provider from a runtime id")
async def t_provider_of(ctx: TestContext) -> None:
    from src.core.compaction import _provider_of

    class _M:
        pass

    m = _M(); m.model = "anthropic:claude-haiku-4-5"
    assert _provider_of(m) == "anthropic"
    m2 = _M(); m2.model = "deepseek:deepseek-chat"
    assert _provider_of(m2) == "deepseek"
    m3 = _M(); m3.model = "plainmodel-no-colon"
    assert _provider_of(m3) is None
    assert _provider_of(object()) is None


@test("compaction", "summary fallback picks a distinct-provider row and never the failed one")
async def t_summary_fallback_picker(ctx: TestContext) -> None:
    from src.core import compaction
    import src.models.native_provider as np_mod

    providers_config = [
        {"id": 1, "name": "anthropic", "framework": "api-based", "enabled": True,
         "models": [{"id": 101, "model": "claude-haiku-4-5"}]},
        {"id": 2, "name": "deepseek", "framework": "api-based", "enabled": True,
         "models": [{"id": 201, "model": "deepseek-chat"}]},
    ]
    agent = _FakeAgent(str(ctx.test_dir / "pk.db"), _FakeModel())
    agent._providers_config = providers_config

    captured: dict = {}

    class _FakeNP:
        def __init__(self, *, model, providers_config, db_path):  # noqa: ANN001
            captured["model"] = model
            self.model = model

    orig_np = np_mod.NativeProvider
    np_mod.NativeProvider = _FakeNP
    os.environ.pop(compaction._SUMMARY_FALLBACK_MODEL_ENV, None)
    try:
        # exclude the failed anthropic proxy → prefers deepseek (never OAuth-limited)
        fb = compaction._summary_fallback_model(agent, exclude_provider="anthropic")
        assert fb is not None
        assert captured.get("model") == "deepseek:deepseek-chat", captured

        # explicit operator override wins when it is a distinct provider
        captured.clear()
        os.environ[compaction._SUMMARY_FALLBACK_MODEL_ENV] = "deepseek:deepseek-chat"
        compaction._summary_fallback_model(agent, exclude_provider="anthropic")
        assert captured.get("model") == "deepseek:deepseek-chat", captured
        os.environ.pop(compaction._SUMMARY_FALLBACK_MODEL_ENV, None)

        # only one provider, and it IS the excluded one → no distinct fallback
        captured.clear()
        agent._providers_config = [providers_config[1]]  # deepseek only
        fb_none = compaction._summary_fallback_model(agent, exclude_provider="deepseek")
        assert fb_none is None, "no distinct-provider row → give up (caller skips)"
    finally:
        np_mod.NativeProvider = orig_np
        os.environ.pop(compaction._SUMMARY_FALLBACK_MODEL_ENV, None)


# ── 6. the fold is bounded by the SUMMARISER's window, not the primary's ──


@test("compaction", "summary transcript is clamped to the summariser's context window")
async def t_summary_transcript_clamped_to_summariser_window(ctx: TestContext) -> None:
    """A fold sized for the primary must not overflow a smaller summariser.

    Regression for the 2026-07-30 production break: the trigger budget is
    computed against the PRIMARY's window, the fold is sent to the summariser
    (``OPENAGENT_COMPACTION_MODEL`` exists to make that a different, cheaper,
    smaller-window row), and nothing reconciled the two — the provider rejected
    the call with ``prompt is too long: 213760 tokens > 200000 maximum`` and
    every long session silently stopped compacting.
    """
    from src.core import compaction

    # 40 turns of ~400 chars each: comfortably past a 1000-token window.
    runs = [
        {"content": "", "messages": [
            {"role": "user", "content": f"turn {i} " + ("lorem ipsum " * 30)}]}
        for i in range(40)
    ]
    # Primary has a huge window; the summariser is the small one that matters.
    primary = _FakeModel(model="big/primary", max_context=1_000_000)
    summariser = _FakeModel(model="small/summariser", max_context=1000,
                            summary="Recap.")
    agent = _FakeAgent(str(ctx.test_dir / "clamp.db"), primary)

    orig_pick = compaction._pick_summary_model
    compaction._pick_summary_model = lambda a, *, fallback: summariser
    try:
        out = await compaction._summarize_runs(runs, primary, agent)
    finally:
        compaction._pick_summary_model = orig_pick

    assert out == "Recap.", out
    assert len(summariser.generate_calls) == 1, "summariser called once"
    sent = summariser.generate_calls[0]["messages"][0]["content"]
    budget = int(1000 * compaction._summary_input_fraction())
    sent_tokens = compaction._estimate_text_tokens(sent, "small/summariser")
    # The wrapper text rides along, so allow it — what must NOT happen is the
    # whole 40-turn transcript going out at ~4000 tokens against a 1000 window.
    assert sent_tokens < 1000, f"prompt still overflows the window: {sent_tokens}"
    assert sent_tokens <= budget * 2, f"prompt far past budget {budget}: {sent_tokens}"
    # Oldest dropped, newest kept: the recap must describe where we ARE.
    assert "turn 39" in sent, "most recent turn must survive the clamp"
    assert "turn 0" not in sent, "oldest turn should have been dropped"


@test("compaction", "no clamp when the transcript already fits the summariser")
async def t_summary_transcript_untouched_when_it_fits(ctx: TestContext) -> None:
    from src.core import compaction

    runs = [{"content": "", "messages": [
        {"role": "user", "content": "short turn about a billing charge"}]}]
    primary = _FakeModel(model="big/primary", max_context=1_000_000)
    summariser = _FakeModel(model="small/summariser", max_context=200_000,
                            summary="Recap.")
    agent = _FakeAgent(str(ctx.test_dir / "noclamp.db"), primary)

    orig_pick = compaction._pick_summary_model
    compaction._pick_summary_model = lambda a, *, fallback: summariser
    try:
        out = await compaction._summarize_runs(runs, primary, agent)
    finally:
        compaction._pick_summary_model = orig_pick

    assert out == "Recap."
    sent = summariser.generate_calls[0]["messages"][0]["content"]
    assert "billing charge" in sent, "a fitting transcript must pass through intact"


@test("compaction", "summary transcript has an absolute provider-admission ceiling")
async def t_summary_transcript_absolute_ceiling(ctx: TestContext) -> None:
    """A huge context window must not license a huge subscription-proxy call.

    Regression from Friday: a 200k-window Claude row received one ~77k-token
    compaction request, the OAuth account rate-limited it immediately, and the
    live Telegram turn then had no provider. One run alone was larger than the
    desired clamp, so the fitter must suffix-trim that boundary run too.
    """
    from src.core import compaction

    runs = [
        {"content": "", "messages": [{
            "role": "assistant",
            "content": f"turn {i} " + ("tool output and decisions " * 3000),
        }]}
        for i in range(8)
    ]
    primary = _FakeModel(model="local:claude-sonnet-4-6", max_context=200_000)
    summariser = _FakeModel(
        model="local:claude-sonnet-4-6", max_context=200_000, summary="Recap.")
    agent = _FakeAgent(str(ctx.test_dir / "absolute-clamp.db"), primary)

    orig_pick = compaction._pick_summary_model
    compaction._pick_summary_model = lambda a, *, fallback: summariser
    os.environ[compaction._SUMMARY_MAX_INPUT_TOKENS_ENV] = "2000"
    try:
        out = await compaction._summarize_runs(runs, primary, agent)
    finally:
        compaction._pick_summary_model = orig_pick
        os.environ.pop(compaction._SUMMARY_MAX_INPUT_TOKENS_ENV, None)

    assert out == "Recap."
    sent = summariser.generate_calls[0]["messages"][0]["content"]
    sent_tokens = compaction._estimate_text_tokens(
        sent, "local:claude-sonnet-4-6")
    # Wrapper prose adds a small amount beyond the transcript ceiling.
    assert sent_tokens < 2300, sent_tokens
    assert "[Turn 8]" in sent, "the newest run marker must survive"
    assert "turn 0" not in sent, "oldest runs must yield first"
    assert "Earlier content from this turn" in sent, (
        "a single oversized newest run must be suffix-trimmed, not sent whole"
    )


# ── 7. sizing the fold: per-model windows and an honest token count ─────
# The fold is bounded by the summariser's window (§6 above) — which is only
# worth anything if BOTH numbers are right. Two ways they were wrong at once,
# and together they kept a support agent from compacting for four days:
#
#   * every model resolved to the same 200k fallback, because the window
#     lookup only answered for ids in OpenRouter's catalog and a proxy-served
#     id ("local:claude-haiku-4-5") is never in it; and
#   * the token count degraded to len//4 for any model without a tokenizer —
#     an English-prose ratio applied to tool JSON, which undercounted a real
#     transcript by 2.5x (112_088 estimated, 278_002 counted).
#
# So the fold was cut to a budget that was not the summariser's, using a size
# that was not the transcript's, and the provider rejected every attempt.


@test("compaction", "the context window is per-model, not one fallback for all")
async def t_context_window_is_per_model(_ctx: TestContext) -> None:
    from src.core import compaction

    class _Bare:
        """A provider row that exposes only its id — the proxy-served shape."""

        def __init__(self, model: str) -> None:
            self.model = model

    wide = compaction._resolve_max_context(_Bare("local:claude-opus-5"))
    narrow = compaction._resolve_max_context(_Bare("local:claude-haiku-4-5"))
    tiny = compaction._resolve_max_context(_Bare("local:deepseek-v4-pro"))

    assert wide == 1_000_000, f"1M row resolved to {wide}"
    assert narrow == 200_000, f"200k row resolved to {narrow}"
    assert tiny < 200_000, f"deepseek resolved to {tiny}"
    assert wide != narrow, "two different models must not share one window"

    # An id we publish nothing about still falls back — conservatively.
    unknown = compaction._resolve_max_context(_Bare("local:some-new-model"))
    assert unknown == compaction._FALLBACK_MAX_CONTEXT, unknown


@test("compaction", "a stand-in tokenizer is trusted less than the model's own")
async def t_token_estimate_margins_a_foreign_tokenizer(_ctx: TestContext) -> None:
    """``count_text_tokens`` never admits it is guessing — so we have to.

    For an id it doesn't recognise, ``_get_tiktoken_encoding`` quietly returns
    ``o200k_base`` instead of saying "no tokenizer for this model". The count
    then looks as authoritative as any other while being an OpenAI-shaped guess
    at Anthropic's tokenizer. Measured on a real fold, that guess was 2.5x low
    and every fold overflowed.
    """
    from src.core import compaction

    # A transcript-shaped sample: JSON, ids, paths, non-English — not prose.
    text = json.dumps({
        "tool": "replio_thread_brief",
        "thread_id": "5e547cd3-b137-4add-8130-cf8261655405",
        "body": "porque me sale la suscripción cancelada si tiene que estar "
                "activa, me podéis ayudar" * 40,
        "path": "/data/agent/logs/events.jsonl",
    }) * 20

    foreign = compaction._estimate_text_tokens(text, "local:claude-haiku-4-5")
    native = compaction._estimate_text_tokens(text, "gpt-4o")

    assert not compaction._is_native_tokenizer("local:claude-haiku-4-5")
    assert compaction._is_native_tokenizer("gpt-4o")
    # Same bytes, same encoder underneath — the only difference is how much we
    # are entitled to believe it.
    assert foreign > native, (
        f"a foreign tokenizer got no margin: {foreign} vs {native}")
    assert foreign == int(native * compaction._PROXY_COUNT_MARGIN), (
        f"unexpected margin: {foreign} vs {native}")


@test("compaction", "the provider's own count replaces our guess, for good")
async def t_learned_density_overrides_the_margin(_ctx: TestContext) -> None:
    from src.core import compaction

    model = "local:claude-test-density"
    text = "some transcript content " * 500
    orig = dict(compaction._MEASURED_DENSITY)
    try:
        before = compaction._estimate_text_tokens(text, model)
        # The provider rejected a fold and stated its real count: 3x ours.
        compaction._learn_density(model, counted=300_000, estimated=100_000)
        after = compaction._estimate_text_tokens(text, model)
        assert after > before, f"learning had no effect: {before} -> {after}"
        assert abs(after / before - 3.0) < 0.05, f"ratio {after / before}"

        # Ratcheted: a smaller correction must not undo a bigger measured one.
        compaction._learn_density(model, counted=110_000, estimated=100_000)
        assert compaction._estimate_text_tokens(text, model) == after
    finally:
        compaction._MEASURED_DENSITY.clear()
        compaction._MEASURED_DENSITY.update(orig)


@test("compaction", "a provider's size rejection is parsed; other failures are not")
async def t_parse_too_long(_ctx: TestContext) -> None:
    from src.core import compaction

    anthropic = RuntimeError(
        "{'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'prompt is too long: 278002 tokens > 200000 maximum'}}")
    assert compaction._parse_too_long(anthropic) == (278002, 200000)

    openai = RuntimeError(
        "This model's maximum context length is 104856 tokens. However, "
        "your messages resulted in 131204 tokens")
    assert compaction._parse_too_long(openai) == (131204, 104856)

    # A timeout or an exhausted account must NEVER be "fixed" by shrinking
    # the transcript — that would silently throw away context to cure a
    # problem that has nothing to do with size.
    assert compaction._parse_too_long(RuntimeError("Request timed out.")) == (None, None)
    assert compaction._parse_too_long(
        RuntimeError("No available Claude OAuth accounts")) == (None, None)


class _TooLongOnceModel:
    """Rejects the first fold for size, then accepts a smaller one.

    Mimics the real failure exactly: the provider states its own count and its
    own maximum, which is the only honest measurement of the transcript we
    ever get.
    """

    def __init__(self, model: str = "small/summariser", limit_chars: int = 4000):
        self.model = model
        self.max_context = 200_000
        self.limit_chars = limit_chars
        self.generate_calls: list[dict] = []

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        self.generate_calls.append({"messages": messages, "kwargs": kwargs})
        body = messages[0]["content"]
        if len(body) > self.limit_chars:
            raise RuntimeError(
                "prompt is too long: "
                f"{len(body)} tokens > {self.limit_chars} maximum")

        class _R:
            content = "Recap."
        return _R()


@test("compaction", "a fold rejected for size is re-cut and retried on the same model")
async def t_summary_refits_after_too_long(ctx: TestContext) -> None:
    """The whole point: a fold that overflows is a fold WE sized wrong.

    Before, an oversized fold fell through to a different provider carrying
    the identical oversized prompt (failing again) or returned "" — which
    left the session uncompacted, so the next turn was bigger, breached the
    threshold again, and burned another oversized attempt. Four days of that
    is what exhausted a weekly model quota.
    """
    from src.core import compaction

    runs = [
        {"content": "", "messages": [
            {"role": "user", "content": f"turn {i} " + ("lorem ipsum " * 40)}]}
        for i in range(30)
    ]
    primary = _FakeModel(model="big/primary", max_context=1_000_000)
    summariser = _TooLongOnceModel()
    agent = _FakeAgent(str(ctx.test_dir / "refit.db"), primary)

    orig_pick = compaction._pick_summary_model
    orig_density = dict(compaction._MEASURED_DENSITY)
    compaction._pick_summary_model = lambda a, *, fallback: summariser
    try:
        out = await compaction._summarize_runs(runs, primary, agent)
    finally:
        compaction._pick_summary_model = orig_pick
        compaction._MEASURED_DENSITY.clear()
        compaction._MEASURED_DENSITY.update(orig_density)

    assert out == "Recap.", f"the retry should have produced a recap, got {out!r}"
    assert len(summariser.generate_calls) == 2, (
        "expected exactly one re-cut retry, got "
        f"{len(summariser.generate_calls)} calls")
    first = summariser.generate_calls[0]["messages"][0]["content"]
    second = summariser.generate_calls[1]["messages"][0]["content"]
    assert len(second) < len(first), "the retry must be SMALLER, not identical"
    # Still the recent end of the conversation, which is the part that matters.
    assert "turn 29" in second, "the newest turn must survive the re-cut"
