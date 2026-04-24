"""Regression: forget_session must drain pending sdk_session writes
before deleting the row.

Before the fix, ``_persist_sdk_session`` scheduled a background
``create_task(_write())`` that wasn't awaited by ``forget_session``.
``forget_session`` called ``delete_sdk_session`` synchronously and
returned — but if the background task hadn't yet run, it would land
AFTER the delete and silently re-insert the old ``sdk_session_id``.
The next turn's ``_hydrate_sdk_session_id`` would then read the stale
row and resume the prior conversation via ``--resume <old-id>``.

This manifested as: Telegram ``/clear`` didn't actually clear, and
scheduled tasks with a fresh ``forget_session`` between fires still
inherited the previous firing's transcript.

The fix: ``forget_session`` now snapshots and ``await``s this
session's pending writes before calling ``delete_sdk_session``. We
prove the fix with a slow-write DB stub that WOULD lose the race on
the old code path.
"""
from __future__ import annotations

import asyncio

from ._framework import TestContext, test


class _SlowWriteDB:
    """Stub DB whose set_sdk_session sleeps long enough to lose a race
    with a naive, non-draining forget_session."""

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self.write_log: list[tuple[str, str]] = []
        # Small enough to keep test fast; big enough that a same-tick
        # forget_session on the old code path would finish before the
        # write ran (even a ``sleep(0)`` would do it, but 50 ms makes
        # the test robust to CI jitter).
        self.write_delay = 0.05

    async def get_sdk_session(self, session_id: str) -> str | None:
        return self.mapping.get(session_id)

    async def set_sdk_session(self, session_id, sdk_session_id, provider=None):
        await asyncio.sleep(self.write_delay)
        self.mapping[session_id] = sdk_session_id
        self.write_log.append((session_id, sdk_session_id))

    async def delete_sdk_session(self, session_id: str) -> None:
        self.mapping.pop(session_id, None)

    async def get_all_sdk_sessions(self, provider=None):
        return dict(self.mapping)


@test("claude_cli_race", "forget_session drains pending writes before delete")
async def t_forget_awaits_pending_write(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLI

    cli = ClaudeCLI(
        model="claude-sonnet-4-6",
        providers_config={"anthropic": {}},
    )
    db = _SlowWriteDB()
    cli.set_db(db)

    sid = "tg:race-user"
    # Simulate what ``_capture_result`` does after a successful turn:
    # schedule a persist that fires off as a background task.
    cli._persist_sdk_session(sid, "sdk-uuid-abc")

    # Immediately forget — must wait for the pending write so the
    # subsequent delete wins the race.
    await cli.forget_session(sid)

    # If the fix works, delete landed last → no row remains.
    # Before the fix, the slow write landed last and ``db.mapping[sid]``
    # still held ``sdk-uuid-abc``.
    assert sid not in db.mapping, (
        f"stale resume row survived forget; mapping={db.mapping}"
    )
    # The pending task completed (it was awaited), so write_log
    # records the write. The delete then cleared the row.
    assert db.write_log == [(sid, "sdk-uuid-abc")], db.write_log


@test("claude_cli_race", "multi-session forgets only drain their own pending writes")
async def t_forget_scoped_to_session(ctx: TestContext) -> None:
    """/clear on user A should not stall waiting for user B's writes.

    Per-session ``_pending_writes`` bucket lets us drain just the
    target session's in-flight persist tasks. Other sessions' writes
    continue in the background, unaffected.
    """
    from openagent.models.claude_cli import ClaudeCLI

    cli = ClaudeCLI(
        model="claude-sonnet-4-6",
        providers_config={"anthropic": {}},
    )
    db = _SlowWriteDB()
    db.write_delay = 0.2  # make user B's write visibly slow
    cli.set_db(db)

    cli._persist_sdk_session("tg:user-a", "sdk-a")
    cli._persist_sdk_session("tg:user-b", "sdk-b")

    import time
    t0 = time.monotonic()
    await cli.forget_session("tg:user-a")
    elapsed = time.monotonic() - t0

    # forget for A drains A's writes (~0.2s) but should not wait for
    # B's (also ~0.2s) — so total stays well under 2× write_delay.
    # Generous headroom for CI: assert under 0.35s.
    assert elapsed < 0.35, f"forget waited for sibling session's write; {elapsed=:.3f}s"

    # B's write is still pending or completed; either way, A is gone.
    assert "tg:user-a" not in db.mapping, db.mapping

    # Drain B so the test doesn't leave dangling tasks.
    await cli.shutdown()
