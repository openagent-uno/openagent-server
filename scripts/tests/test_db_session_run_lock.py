"""``add_session_run`` per-session lock — concurrent writes don't drop runs.

Pre-fix, the read-modify-write of ``agno_sessions.runs`` was
unprotected. Two ``_persist_turn`` writes interleaving on the same
session both read the pre-state JSON, each appended their own run,
and the later commit clobbered the earlier one — silently losing
turns. The fix is a per-session ``asyncio.Lock`` that serialises the
RMW. This test fires N concurrent ``add_session_run`` calls and
asserts none vanish.
"""
from __future__ import annotations

import asyncio
import uuid

from ._framework import TestContext, test


@test("db_session_run_lock", "20 concurrent add_session_run calls all land")
async def t_concurrent_writes(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"runlock-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        sid = f"runlock-{uuid.uuid4().hex[:8]}"
        N = 20

        async def _write(i: int) -> str:
            return await db.add_session_run(
                session_id=sid,
                messages=[{"role": "user", "content": f"msg-{i}"}],
                model="claude-sonnet-4-6",
                run_id=f"run-{i}",
            )

        # Launch every write as a task simultaneously; under
        # contention any unsynchronised write would clobber some.
        run_ids = await asyncio.gather(*[_write(i) for i in range(N)])
        assert len(set(run_ids)) == N, "run_ids should be unique"

        runs = await db.list_session_runs(sid, limit=N + 5)
        persisted_ids = {r.get("run_id") for r in runs}
        # Every started run must show up — losses indicate the RMW
        # was concurrent unsafe again.
        expected = {f"run-{i}" for i in range(N)}
        missing = expected - persisted_ids
        assert not missing, f"missing runs after concurrent writes: {missing}"
        assert len(runs) == N, f"expected {N} runs, got {len(runs)}"
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("db_session_run_lock", "different sessions don't serialise through one lock")
async def t_per_session_isolation(ctx: TestContext) -> None:
    """Per-session locking — not global. Two concurrent writes against
    different session_ids must each get their own row, no shared
    blocking."""
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"runlock-iso-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        sid_a = f"isoA-{uuid.uuid4().hex[:8]}"
        sid_b = f"isoB-{uuid.uuid4().hex[:8]}"

        async def _write_to(sid: str) -> str:
            return await db.add_session_run(
                session_id=sid,
                messages=[{"role": "user", "content": "x"}],
                model="claude-sonnet-4-6",
            )

        a_rid, b_rid = await asyncio.gather(_write_to(sid_a), _write_to(sid_b))
        assert a_rid != b_rid

        a_runs = await db.list_session_runs(sid_a, limit=5)
        b_runs = await db.list_session_runs(sid_b, limit=5)
        assert len(a_runs) == 1 and len(b_runs) == 1
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
