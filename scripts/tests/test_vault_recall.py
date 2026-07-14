"""Vault recall attribution: note → run → outcome.

The tests that matter here are the ones that drive the REAL streaming path
(``TeamRouterProvider.stream`` → ``_arun_runtime_stream``) with genuine runtime
event objects, rather than asserting against a mock of it. That is deliberate.
The feature this replaces — ``agent.turn.tool_calls`` — had tests that passed
for its entire lifetime while it emitted nothing at all in production, because
they exercised the non-streaming path that production does not take. A unit
test of the extractor alone would have reproduced exactly that mistake.
"""
from __future__ import annotations

import asyncio

from ._framework import TestContext, test


# ── the hand-kept tool lists must name REAL tools ───────────────────


def _live_tool_keys():
    """Ground truth: every tool key actually registered, from the live
    adapters + the vendored Node server.

    Imported INSIDE the test rather than at module scope on purpose: the
    driver's ``_TEST_MODULES`` order is significant (tests register in import
    order), and importing a module that sits LATER in that tuple from one that
    sits earlier would pull its registrations forward and silently reorder the
    suite. By the time a test body runs, every module is already imported, so
    this is a cache hit.
    """
    from .test_prompt_tool_names import _registered_tool_keys

    keys, notes = _registered_tool_keys()
    assert len(keys) > 50, f"ground truth looks broken: {len(keys)} keys, {notes}"
    return keys


@test("vault_recall", "every attributed tool name is a REAL registration")
async def t_recall_tools_are_real(ctx: TestContext) -> None:
    """A hand-kept tool list is the thing that rots. It already did: both
    ``_VAULT_READ_TOOLS`` and this module shipped with ``vault_get_backlinks``
    and ``vault_list_notes``, neither of which has ever existed — so
    ``vault_reads`` undercounted for its entire life. Assert against the live
    registrations, not against a second hand-kept list, or this test rots the
    same way the code did.

    The trap being guarded: ``vault`` is a Node SUBPROCESS (keys prefixed →
    ``vault_read_note``) while ``vault-gate`` is IN-PROCESS (keys are the bare
    function names → ``vault_backlinks``, NOT ``vault_gate_backlinks``). Two
    servers, two spelling rules.
    """
    from src.core.vault_recall import _RECALL_TOOL_ARGS

    keys = _live_tool_keys()
    phantom = sorted(n for n in _RECALL_TOOL_ARGS if n not in keys)
    assert not phantom, (
        f"_RECALL_TOOL_ARGS names {phantom} — no such registered tool, so "
        f"those recalls can never be attributed."
    )


@test("vault_recall", "_VAULT_READ_TOOLS / _VAULT_WRITE_TOOLS are REAL registrations")
async def t_vault_tool_sets_are_real(ctx: TestContext) -> None:
    """Guards the sets ``_emit_tool_call_summary`` counts against. These held
    two phantoms since the day they shipped."""
    from src.core.agent import _VAULT_READ_TOOLS, _VAULT_WRITE_TOOLS

    keys = _live_tool_keys()
    for label, names in (
        ("_VAULT_READ_TOOLS", _VAULT_READ_TOOLS),
        ("_VAULT_WRITE_TOOLS", _VAULT_WRITE_TOOLS),
    ):
        phantom = sorted(n for n in names if n not in keys)
        assert not phantom, f"{label} names non-existent tool(s): {phantom}"

    # The two that actually rotted. Pinned by name so a regression says what
    # went wrong instead of just "a set changed".
    assert "vault_list_directory" in _VAULT_READ_TOOLS, "the browse leaf must be counted"
    assert "vault_backlinks" in _VAULT_READ_TOOLS
    assert "vault_list_notes" not in _VAULT_READ_TOOLS, "phantom is back"
    assert "vault_get_backlinks" not in _VAULT_READ_TOOLS, "phantom is back"


@test("vault_recall", "reads and writes stay disjoint")
async def t_read_write_disjoint(ctx: TestContext) -> None:
    """A tool counted as both would make vault_writes and vault_reads
    double-count the same call."""
    from src.core.agent import _VAULT_READ_TOOLS, _VAULT_WRITE_TOOLS
    from src.core.vault_recall import _RECALL_TOOL_ARGS

    assert not (_VAULT_READ_TOOLS & _VAULT_WRITE_TOOLS)
    # A recall is a strict subset of a read: you cannot recall a note without
    # reading it, and a write is never a recall.
    assert set(_RECALL_TOOL_ARGS) <= _VAULT_READ_TOOLS, (
        f"recall tools not counted as reads: "
        f"{sorted(set(_RECALL_TOOL_ARGS) - _VAULT_READ_TOOLS)}"
    )
    assert not (set(_RECALL_TOOL_ARGS) & _VAULT_WRITE_TOOLS)


# ── the pure pieces ─────────────────────────────────────────────────


@test("vault_recall", "note paths extracted only from vault READ tools")
async def t_extract_paths(ctx: TestContext) -> None:
    from src.core import vault_recall as vr

    assert vr.note_paths_from_tool(
        "vault_read_note", {"path": "Projects/alpha.md"},
    ) == ["Projects/alpha.md"]
    assert vr.note_paths_from_tool(
        "vault_read_multiple_notes", {"paths": ["a.md", "b.md"]},
    ) == ["a.md", "b.md"]

    # A write is not a recall: it says what the agent learned, not what it
    # consulted. Scoring writes would credit a note for the run that created it.
    assert vr.note_paths_from_tool(
        "vault_write_note", {"path": "a.md", "content": "x"},
    ) == []
    assert vr.note_paths_from_tool(
        "vault_get_frontmatter", {"path": "c.md"},
    ) == ["c.md"]

    # Non-vault tools are the overwhelming majority of calls.
    assert vr.note_paths_from_tool("shell_exec", {"path": "/etc/passwd"}) == []
    # Search/browse surface note NAMES in the RESULT; the model may ignore
    # every hit, so a hit is not a read.
    assert vr.note_paths_from_tool("vault_search_notes", {"query": "x"}) == []
    assert vr.note_paths_from_tool("vault_list_directory", {"path": "Projects"}) == []
    # Backlinks is a real vault READ but not a RECALL: it returns the names of
    # OTHER notes pointing at this one, never this note's own text.
    assert vr.note_paths_from_tool("vault_backlinks", {"path": "a.md"}) == []
    # Shape drift must cost a counter, never a turn.
    assert vr.note_paths_from_tool(None, None) == []
    assert vr.note_paths_from_tool("vault_read_note", "not-a-dict") == []
    assert vr.note_paths_from_tool("vault_read_note", {}) == []


@test("vault_recall", "a note BODY can never reach the sink as a path")
async def t_no_body_in_sink(ctx: TestContext) -> None:
    """The token/memory hazard: ``tool_args`` can carry tens of KB of model
    text. Only short path-shaped strings are ever retained."""
    from src.core import vault_recall as vr

    body = "x" * 50_000
    # Even on a read tool, an over-long value is not a path.
    assert vr.note_paths_from_tool("vault_read_note", {"path": body}) == []

    sink, token = vr.open_sink()
    try:
        vr.record_tool("vault_write_note", {"path": "a.md", "content": body})
        vr.record_tool("vault_read_note", {"path": "Real/note.md"})
        recorded = vr.recorded_paths(sink)
    finally:
        vr.close_sink(token)
    assert list(recorded) == ["Real/note.md"]
    assert all(len(p) < 512 for p in recorded)


@test("vault_recall", "a barge-in classifies as cancelled, never errored")
async def t_outcome_classification(ctx: TestContext) -> None:
    """The 294/294 trap: on the production log every errored run was also a
    cancelled one. Cancel MUST win over error, or the scorer learns that a
    user interrupting is a defect (§2 calls barge-in first-class)."""
    from src.core import vault_recall as vr

    assert vr.outcome_for_exception(None) == vr.OUTCOME_OK
    assert vr.outcome_for_exception(asyncio.CancelledError()) == vr.OUTCOME_CANCELLED
    assert vr.outcome_for_exception(GeneratorExit()) == vr.OUTCOME_CANCELLED
    assert vr.outcome_for_exception(RuntimeError("boom")) == vr.OUTCOME_ERRORED
    # And the scorable set excludes cancels by construction, not by a filter
    # a caller has to remember to apply.
    assert vr.OUTCOME_CANCELLED not in vr.SCORABLE
    assert vr.SCORABLE == {vr.OUTCOME_OK, vr.OUTCOME_ERRORED}


@test("vault_recall", "one recall per note per run, even when re-read")
async def t_dedupe(ctx: TestContext) -> None:
    """A retry loop re-reading the same note must not weight it by how badly
    the turn went."""
    from src.core import vault_recall as vr

    sink, token = vr.open_sink()
    try:
        for _ in range(5):
            vr.record_tool("vault_read_note", {"path": "a.md"})
        assert vr.recorded_paths(sink) == {"a.md": "vault_read_note"}
    finally:
        vr.close_sink(token)


@test("vault_recall", "record_tool is a no-op with no sink open")
async def t_no_sink(ctx: TestContext) -> None:
    """A provider used outside the dispatcher (a test, a direct call) must not
    blow up on telemetry."""
    from src.core import vault_recall as vr

    vr.record_tool("vault_read_note", {"path": "a.md"})  # must not raise


# ── the DB half ─────────────────────────────────────────────────────


@test("vault_recall", "ok_rate excludes barge-ins from the denominator")
async def t_db_stats_exclude_cancelled(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        async def rec(note, outcome, cost=0.001):
            await db.record_vault_recall(
                session_id="s1", note_path=note, tool="vault_read_note",
                outcome=outcome, model="test:m", cost=cost,
            )

        # 3 ok, 1 errored, 6 cancelled. If barge-ins counted as failures the
        # rate would be 3/10 = 0.3 — the exact lie the 294/294 correlation
        # would have taught.
        for _ in range(3):
            await rec("Good.md", "ok")
        await rec("Good.md", "errored")
        for _ in range(6):
            await rec("Good.md", "cancelled")

        rows = await db.get_vault_recall_stats(note_path="Good.md")
        assert len(rows) == 1, rows
        r = rows[0]
        assert r["recalls"] == 10, r
        assert r["ok"] == 3 and r["errored"] == 1 and r["cancelled"] == 6, r
        assert r["scorable"] == 4, r
        assert r["ok_rate"] == 0.75, f"barge-ins leaked into the rate: {r}"

        # A note recalled ONLY in cancelled runs has no scorable evidence at
        # all. It must report None, not 0.0 — "no data" is not "always fails".
        for _ in range(4):
            await rec("Unknown.md", "cancelled")
        rows = await db.get_vault_recall_stats(note_path="Unknown.md")
        assert rows[0]["ok_rate"] is None, rows[0]
        assert rows[0]["scorable"] == 0, rows[0]
    finally:
        await db.close()


@test("vault_recall", "stats window + ordering + cost aggregate")
async def t_db_stats_window(ctx: TestContext) -> None:
    import time

    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        for _ in range(3):
            await db.record_vault_recall(
                session_id="s", note_path="Win/hot.md", tool="vault_read_note",
                outcome="ok", model="m", cost=0.002,
            )
        await db.record_vault_recall(
            session_id="s", note_path="Win/cold.md", tool="vault_read_note",
            outcome="ok", model="m", cost=0.5,
        )
        # Cost aggregates per note, and is summed from the per-run values the
        # writer captured — NOT joined from usage_log, which is keyed by
        # session and would smear a whole session's spend over every note.
        hot = (await db.get_vault_recall_stats(note_path="Win/hot.md"))[0]
        assert abs(hot["cost"] - 0.006) < 1e-9, hot

        # Most-recalled sorts first. Scoped to this test's own notes: the
        # suite shares one DB, so an unscoped ordering assert would be a
        # function of whichever tests ran before it.
        rows = await db.get_vault_recall_stats(limit=100)
        mine = [r["note_path"] for r in rows if r["note_path"].startswith("Win/")]
        assert mine == ["Win/hot.md", "Win/cold.md"], mine

        # A window in the future excludes everything (no crash, no rows).
        assert await db.get_vault_recall_stats(since=time.time() + 60) == []
    finally:
        await db.close()


# ── the REAL call path ──────────────────────────────────────────────


class _FakeRuntime:
    """Stands in for the agno Team. Emits GENUINE runtime event objects so the
    isinstance dispatch inside ``_arun_runtime_stream`` is the real one — a
    hand-rolled event shape would pass while production silently didn't match.
    """

    def __init__(self, note: str, *, fail: bool = False, hang: bool = False):
        self._note = note
        self._fail = fail
        self._hang = hang

    def arun(self, prompt, **kwargs):
        from src.core._run_state.agent import (
            RunContentEvent, ToolCallCompletedEvent,
        )
        from src.core._run_state.requirement import ToolExecution

        note, fail, hang = self._note, self._fail, self._hang

        async def _gen():
            yield ToolCallCompletedEvent(
                tool=ToolExecution(
                    tool_name="vault_read_note", tool_args={"path": note},
                ),
            )
            yield ToolCallCompletedEvent(
                tool=ToolExecution(
                    tool_name="shell_exec", tool_args={"cmd": "ls"},
                ),
            )
            yield RunContentEvent(content="hello")
            if fail:
                raise RuntimeError("provider exploded")
            if hang:
                await asyncio.sleep(30)
            yield RunContentEvent(content=" world")

        return _gen()


def _provider(db, note, *, fail=False, hang=False):
    """A TeamRouterProvider whose runtime is faked but whose stream(), sink
    lifecycle, outcome classification and flush are all the real ones.

    Every caller passes a UNIQUE note: the suite shares one ``ctx.db_path``
    across all tests, so a shared note name would let one test's rows leak
    into another's assertions.
    """
    from src.models.dispatcher import TeamRouterProvider

    p = TeamRouterProvider("test:model")
    p._db = db
    p._ensure_runtime = lambda sid, system: _FakeRuntime(note, fail=fail, hang=hang)
    return p


async def _rows(db, note):
    return await db.get_vault_recall_stats(note_path=note)


@test("vault_recall", "REAL stream path: a completed turn attributes its notes")
async def t_stream_path_ok(ctx: TestContext) -> None:
    """Drives TeamRouterProvider.stream end to end. This is the test that
    would have caught ``agent.turn.tool_calls`` being inert: it asserts a row
    lands in the DB from the STREAMING path, which is the only path production
    uses (697 streamed turns vs 11 non-streamed, on the real log)."""
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        note = "Stream/ok.md"
        p = _provider(db, note)
        out = [d async for d in p.stream(
            [{"role": "user", "content": "hi"}], session_id="sess-ok",
        )]
        assert "".join(out) == "hello world", out

        rows = await _rows(db, note)
        assert len(rows) == 1, f"no attribution row from the real stream: {rows}"
        assert rows[0]["ok"] == 1 and rows[0]["errored"] == 0, rows[0]

        # The non-vault tool in the same turn must NOT have been recorded —
        # only vault reads are recalls.
        assert await _rows(db, "shell_exec") == []
        assert await _rows(db, "ls") == []
    finally:
        await db.close()


@test("vault_recall", "REAL stream path: a provider error books errored")
async def t_stream_path_errored(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        note = "Stream/errored.md"
        p = _provider(db, note, fail=True)
        try:
            async for _ in p.stream(
                [{"role": "user", "content": "hi"}], session_id="sess-err",
            ):
                pass
        except Exception:
            pass  # _arun_runtime_stream may absorb/re-raise; we assert the row

        rows = await _rows(db, note)
        assert len(rows) == 1, f"error turn lost its attribution: {rows}"
        assert rows[0]["cancelled"] == 0, (
            f"a provider error must not be booked as a barge-in: {rows[0]}"
        )
    finally:
        await db.close()


@test("vault_recall", "REAL stream path: a barge-in books cancelled, not errored")
async def t_stream_path_barge_in(ctx: TestContext) -> None:
    """The production-shaped case: the user interrupts mid-answer. 294 of the
    294 errored entries on the real log were this. It must land as cancelled
    and stay out of ok_rate."""
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        note = "Stream/bargein.md"
        p = _provider(db, note, hang=True)
        agen = p.stream([{"role": "user", "content": "hi"}], session_id="sess-bi")

        async def _drive():
            async for _ in agen:
                pass

        task = asyncio.create_task(_drive())
        # Let the tool event through, then barge in while the model "talks".
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # The generator's finally may need a tick to flush.
        await asyncio.sleep(0.1)

        rows = await _rows(db, note)
        assert len(rows) == 1, f"barge-in lost its attribution entirely: {rows}"
        r = rows[0]
        assert r["cancelled"] == 1, f"barge-in not booked as cancelled: {r}"
        assert r["errored"] == 0, (
            f"barge-in booked as a FAILURE — this is the 294/294 trap: {r}"
        )
        assert r["ok_rate"] is None, f"barge-in leaked into ok_rate: {r}"
    finally:
        await db.close()


@test("vault_recall", "attribution never breaks a turn when the DB is gone")
async def t_flush_never_raises(ctx: TestContext) -> None:
    """A bookkeeping miss must cost a counter, never a turn."""
    class _BadDB:
        async def record_vault_recall(self, **kw):
            raise RuntimeError("disk on fire")

    p = _provider(_BadDB(), "Stream/baddb.md")
    out = [d async for d in p.stream(
        [{"role": "user", "content": "hi"}], session_id="sess-baddb",
    )]
    assert "".join(out) == "hello world", "a DB failure killed the turn"

    # And with no db wired at all.
    p2 = _provider(None, "Stream/nodb.md")
    out2 = [d async for d in p2.stream(
        [{"role": "user", "content": "hi"}], session_id="sess-nodb",
    )]
    assert "".join(out2) == "hello world"


# ── the read-back tool ──────────────────────────────────────────────


@test("vault_recall", "vault_recall_stats is reachable on the vault-gate toolkit")
async def t_tool_registered(ctx: TestContext) -> None:
    """An unregistered tool is an inert feature — the exact failure mode this
    whole change exists to correct."""
    from src.mcp.servers.vault_gate.adapters import build_runtime_toolkit

    tk = build_runtime_toolkit()
    names = set(getattr(tk, "functions", {}) or {}) | set(
        getattr(tk, "async_functions", {}) or {}
    )
    assert "vault_recall_stats" in names, sorted(names)

    # vault-gate's exact tool keys are inlined into the framework prompt
    # (prompts.py:_INLINE_TOOL_KEYS_SERVERS) up to a cap of 24. Past it the key
    # silently stops being surfaced and the model has to guess it.
    assert len(names) <= 24, f"vault-gate exceeded the inline cap: {len(names)}"


@test("vault_recall", "the tool reports counts and refuses to claim causation")
async def t_tool_payload(ctx: TestContext) -> None:
    import os

    from src.memory.db import MemoryDB
    from src.mcp.servers.vault_gate import recall

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.record_vault_recall(
            session_id="s", note_path="A.md", tool="vault_read_note",
            outcome="ok", model="m", cost=0.01,
        )
        await db.record_vault_recall(
            session_id="s", note_path="A.md", tool="vault_read_note",
            outcome="cancelled", model="m", cost=0.01,
        )
    finally:
        await db.close()

    prev = os.environ.get("OPENAGENT_DB_PATH")
    os.environ["OPENAGENT_DB_PATH"] = str(ctx.db_path)
    try:
        res = await recall.vault_recall_stats(limit=10)
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_DB_PATH", None)
        else:
            os.environ["OPENAGENT_DB_PATH"] = prev

    note = next(n for n in res["notes"] if n["note"] == "A.md")
    assert note["recalls"] == 2, note
    assert note["cancelled_excluded"] == 1, note
    assert note["ok_rate"] == 1.0, note  # 1 ok / 1 scorable

    # The caveat is the honesty of the feature, not decoration: without it a
    # model reads ok_rate as a quality verdict.
    caveat = res["caveat"].lower()
    assert "association" in caveat and "not causation" in caveat, res["caveat"]
    assert "delete" in caveat, "must warn against deleting on a low rate"
    assert "interrupt" in caveat, "must say barge-ins are excluded, and why"
