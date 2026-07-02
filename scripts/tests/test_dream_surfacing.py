"""Dream-mode surfacing + live streaming regression tests.

Two user-visible bugs are pinned here:

  1. A dream-mode firing never appeared in the app's sidebar "Recent"
     activity feed. The feed is built by fanning out over
     ``GET /api/scheduled-tasks`` → each task's ``/runs``; the gateway hid
     the built-in ``dream-mode`` task from BOTH endpoints, so its firings
     were invisible even though they were recorded in ``task_runs``. The
     fix exposes built-ins read-only (``?include_builtin=1`` on the list,
     plus get-by-id / runs) while keeping every mutation rejected.

  2. Running dream-mode via the ``run_dream_mode`` MCP tool (the "run it
     manually from a chat" path) produced no live stream — the run screen
     stayed empty until the whole pass finished. The tool spawned its child
     session via ``run_child_session`` WITHOUT ``stream=True`` (the cron
     path passes it), so it took the non-streaming ``agent.run`` branch that
     emits no child frames. The fix passes ``stream=True`` so it streams
     token-by-token like the nightly firing.

These drive the REAL patched handlers against a temp ``MemoryDB`` + a stub
agent — no gateway lifecycle, no live model.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

from aiohttp.test_utils import make_mocked_request

from ._framework import TestContext, test


# ── shared fakes ──────────────────────────────────────────────────────


class _Capture:
    """A child-stream emitter that records every frame it's handed."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def __call__(self, frame: dict) -> None:
        self.frames.append(frame)


class _StreamAgent:
    """Minimal agent whose ``run_stream`` yields a couple of deltas — enough
    to prove the dream-mode child takes the streaming path."""

    name = "spy"
    model = None

    def __init__(self) -> None:
        self.released: list[str] = []

    async def run_stream(self, *, message, user_id, session_id,
                         model_override=None, author=None, on_status=None):
        if on_status is not None:
            await on_status("Consolidating…")
        for piece in ("Merged ", "3 notes."):
            yield {"kind": "delta", "text": piece}
        yield {"kind": "done", "text": "Merged 3 notes."}

    async def release_session(self, session_id, *, model_override=None) -> None:
        self.released.append(session_id)


def _make_request(db, *, method: str = "GET", path: str = "/x",
                  match_info: dict | None = None):
    """Build a minimal aiohttp request the scheduled-tasks handlers read.

    They only touch ``request.app['gateway']._scheduler.db``,
    ``request.query`` (parsed from ``path``), ``request.match_info``, and —
    for mutations that get past the built-in guard — ``gw.broadcast_resource``
    (a no-op here; the built-in tests never reach it)."""

    async def _noop_broadcast(*_a, **_k) -> None:
        return None

    gw = SimpleNamespace(
        _scheduler=SimpleNamespace(db=db),
        broadcast_resource=_noop_broadcast,
    )
    return make_mocked_request(
        method, path, match_info=match_info or {}, app={"gateway": gw},
    )


# ── Issue #2: run_dream_mode streams live ─────────────────────────────


@test("dream_surfacing", "run_dream_mode streams the firing live (stream=True)")
async def t_manual_dream_streams(ctx: TestContext) -> None:
    from src.core.builtin_tasks import DREAM_MODE_TASK_NAME
    from src.mcp.servers.delegation import handlers as dh
    from src.memory.db import MemoryDB
    from src.stream.child_stream import (
        install_child_stream_emitter, reset_child_stream_emitter,
    )
    from src.stream.resource_events import set_resource_event_sink

    tmp_db = ctx.db_path.with_name(f"dream-stream-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        # The built-in scheduled_tasks row must exist for run_dream_mode to
        # record a task_runs row against it (seeded at boot when dream mode is
        # enabled; we seed it directly here).
        dream_task_id = await db.add_task(DREAM_MODE_TASK_NAME, "0 3 * * *", "seed")

        agent = _StreamAgent()
        cap = _Capture()
        # Capture the resource events the tool fires so we can assert the
        # sidebar "Recent" feed is nudged to refresh (like the cron path).
        events: list[tuple] = []
        set_resource_event_sink(lambda res, act, _id=None: events.append((res, act, _id)))
        # Mimic being INSIDE a chat turn: a turn-scoped child emitter is bound
        # (the realistic "user asked me to run dream mode" scenario). The tool
        # must stream its child-tagged frames onto that channel.
        ctx_tokens = dh.install_context(
            session_id="chat-1", pool=None, db=db, dispatcher=None,
            agent=agent, owner_handle="alessandro",
        )
        etok = install_child_stream_emitter(cap)
        try:
            result = await dh.run_dream_mode()
        finally:
            reset_child_stream_emitter(etok)
            dh.reset_context(ctx_tokens)
            set_resource_event_sink(None)

        assert result.get("status") == "ok", result
        sid = result.get("child_session_id")
        assert sid and sid.startswith(f"scheduler:{DREAM_MODE_TASK_NAME}:"), sid

        # THE FIX: live frames were emitted DURING the run (pre-fix this list
        # was empty — the child took the non-streaming agent.run branch).
        assert cap.frames, "no child frames emitted — run_dream_mode is not streaming"
        assert all(f["session_id"] == sid for f in cap.frames), cap.frames
        kinds = [f["kind"] for f in cap.frames]
        assert kinds[0] == "seed", kinds            # mission block first
        assert "delta" in kinds, kinds              # token-by-token body
        assert "turn_complete" in kinds, kinds       # commits the bubble
        # The seed frame carries the agent-self author (renders as a Mission,
        # not a human "You" bubble).
        assert cap.frames[0].get("author", {}).get("kind") == "agent", cap.frames[0]

        # A task_runs row was recorded + linked to the child session so the
        # activity feed / run screen can find it, and finalized success.
        runs = await db.list_task_runs(dream_task_id)
        assert len(runs) == 1, runs
        assert runs[0]["status"] == "success", runs[0]
        assert runs[0]["session_id"] == sid, runs[0]
        # The child runtime was released (durable row kept), not forgotten.
        assert agent.released == [sid], agent.released
        # The feed was nudged to refresh at run open AND close (matches the
        # cron path's scheduled_task broadcasts) so a manual run appears in the
        # sidebar live, not only on the next reload.
        sched_events = [e for e in events if e[0] == "scheduled_task"]
        assert len(sched_events) >= 2, events
        assert all(e[1] == "updated" and e[2] == dream_task_id for e in sched_events), events
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("dream_surfacing", "run_dream_mode self-heals the missing builtin row so a manual firing surfaces")
async def t_manual_dream_self_heals(ctx: TestContext) -> None:
    """The real my-agent case: dream mode was never enabled, so no dream-mode
    ``scheduled_tasks`` row exists. A manual firing must create it (disabled)
    so the run records a ``task_runs`` row and shows up in the "Recent" feed —
    instead of silently running as an unlinked child session."""
    from src.core.builtin_tasks import DREAM_MODE_TASK_NAME
    from src.mcp.servers.delegation import handlers as dh
    from src.memory.db import MemoryDB
    from src.stream.resource_events import set_resource_event_sink

    tmp_db = ctx.db_path.with_name(f"dream-heal-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        # NO dream-mode row — mirrors an agent where dream mode was never enabled.
        pre = [t for t in await db.get_tasks() if t["name"] == DREAM_MODE_TASK_NAME]
        assert pre == [], pre

        agent = _StreamAgent()
        events: list[tuple] = []
        set_resource_event_sink(lambda r, a, i=None: events.append((r, a, i)))
        ctx_tokens = dh.install_context(
            session_id="chat-1", pool=None, db=db, dispatcher=None,
            agent=agent, owner_handle="alessandro",
        )
        try:
            result = await dh.run_dream_mode()
        finally:
            dh.reset_context(ctx_tokens)
            set_resource_event_sink(None)

        assert result.get("status") == "ok", result
        # The builtin row now exists, DISABLED (so the scheduler never auto-fires
        # it), and carries the manual firing.
        dream = next(
            (t for t in await db.get_tasks() if t["name"] == DREAM_MODE_TASK_NAME), None,
        )
        assert dream is not None, "self-heal did not create the dream-mode row"
        assert dream["enabled"] == 0, dream
        runs = await db.list_task_runs(dream["id"])
        assert len(runs) == 1 and runs[0]["status"] == "success", runs
        assert runs[0]["session_id"] == result["child_session_id"], runs[0]
        # The feed was nudged to refresh so the run appears live.
        assert any(e[0] == "scheduled_task" for e in events), events
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


# ── Issue #1: built-ins surface read-only in the feed data ────────────


async def _seed_feed_db(ctx: TestContext):
    """A temp DB with a seeded, enabled dream-mode built-in (with one recorded
    firing) plus a normal user task."""
    from src.core.builtin_tasks import DREAM_MODE_TASK_NAME
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"dream-feed-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    await db.connect()
    dream_id = await db.add_task(DREAM_MODE_TASK_NAME, "0 3 * * *", "p")
    user_id = await db.add_task("My Task", "0 9 * * *", "p")
    child_sid = f"scheduler:{DREAM_MODE_TASK_NAME}:r1"
    run_id = await db.add_task_run(task_id=dream_id, session_id=child_sid)
    await db.update_task_run(run_id, status="success", finished_at=1.0, output="ok")
    return db, tmp_db, dream_id, user_id, child_sid


@test("dream_surfacing", "default list still hides built-ins; ?include_builtin=1 reveals them")
async def t_list_include_builtin(ctx: TestContext) -> None:
    from src.gateway.api.scheduled_tasks import handle_list

    db, tmp_db, dream_id, user_id, _sid = await _seed_feed_db(ctx)
    try:
        # Default: the management list stays clean (built-ins hidden).
        resp = await handle_list(_make_request(db, path="/api/scheduled-tasks"))
        assert resp.status == 200, resp.status
        names = {t["name"] for t in json.loads(resp.body)["tasks"]}
        assert "dream-mode" not in names, names
        assert "My Task" in names, names

        # Opt-in: the activity feed sees the built-in with its real id.
        resp = await handle_list(
            _make_request(db, path="/api/scheduled-tasks?include_builtin=1"),
        )
        assert resp.status == 200, resp.status
        tasks = json.loads(resp.body)["tasks"]
        by_name = {t["name"]: t for t in tasks}
        assert "dream-mode" in by_name, by_name
        assert by_name["dream-mode"]["id"] == dream_id, by_name["dream-mode"]
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("dream_surfacing", "built-in is readable by id + its run history (no 404)")
async def t_get_and_runs_readable(ctx: TestContext) -> None:
    from src.gateway.api.scheduled_tasks import handle_get, handle_runs_list

    db, tmp_db, dream_id, _user_id, child_sid = await _seed_feed_db(ctx)
    try:
        # GET by id — the run-screen title read used to 404 for built-ins.
        resp = await handle_get(
            _make_request(db, path=f"/api/scheduled-tasks/{dream_id}",
                          match_info={"id": dream_id}),
        )
        assert resp.status == 200, (resp.status, resp.body)
        assert json.loads(resp.body)["name"] == "dream-mode"

        # Run history — the feed reads firings here; used to 404 for built-ins.
        resp = await handle_runs_list(
            _make_request(db, path=f"/api/scheduled-tasks/{dream_id}/runs",
                          match_info={"id": dream_id}),
        )
        assert resp.status == 200, (resp.status, resp.body)
        runs = json.loads(resp.body)["runs"]
        assert len(runs) == 1, runs
        assert runs[0]["session_id"] == child_sid, runs[0]
        assert runs[0]["status"] == "success", runs[0]
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("dream_surfacing", "built-in stays non-editable — every mutation is 403")
async def t_builtin_mutations_rejected(ctx: TestContext) -> None:
    from src.gateway.api.scheduled_tasks import (
        handle_delete, handle_run, handle_stop, handle_update,
    )

    db, tmp_db, dream_id, _user_id, _sid = await _seed_feed_db(ctx)
    try:
        for handler, method in (
            (handle_run, "POST"),
            (handle_stop, "POST"),
            (handle_update, "PATCH"),
            (handle_delete, "DELETE"),
        ):
            resp = await handler(
                _make_request(db, method=method,
                              path=f"/api/scheduled-tasks/{dream_id}",
                              match_info={"id": dream_id}),
            )
            assert resp.status == 403, f"{handler.__name__} → {resp.status}"
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
