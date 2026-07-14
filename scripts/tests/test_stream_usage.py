"""Token accounting on the STREAMING path.

``ModelDispatcher.generate`` has always written a ``usage_log`` row. ``stream``
wrote nothing — and every surface that matters streams: chat, the channel
bridges, scheduled tasks, and inbound webhook events. On 2026-07-13/14 the
Replio webhook lane burned ~412M input tokens across two agents and the ledger
held **zero** rows for it. The burn wasn't merely missed; it was invisible by
construction, and it ended only when the provider ran out of credit.

These guard the pieces: the sink collects what a streamed run reports, and the
dispatcher turns that into a real usage_log row.
"""
from __future__ import annotations

import time

from ._framework import TestContext, test


@test("stream_usage", "the sink collects tokens from a streamed run")
async def t_sink_collects(_ctx: TestContext) -> None:
    from src.models import stream_usage

    sink, token = stream_usage.open_sink()
    try:
        stream_usage.record(input_tokens=1200, output_tokens=80, model="local:claude-sonnet-4-6")
        stream_usage.record(input_tokens=340, output_tokens=20)
    finally:
        stream_usage.close_sink(token)

    assert sink["input_tokens"] == 1540
    assert sink["output_tokens"] == 100
    assert sink["model"] == "local:claude-sonnet-4-6"


@test("stream_usage", "recording outside a sink is a no-op, never a crash")
async def t_record_without_sink(_ctx: TestContext) -> None:
    from src.models import stream_usage

    stream_usage.record(input_tokens=999, output_tokens=1)  # must not raise


@test("stream_usage", "metrics_to_tokens reads the runtime's shapes")
async def t_metrics_shapes(_ctx: TestContext) -> None:
    from src.models.stream_usage import metrics_to_tokens

    assert metrics_to_tokens(None) == (0, 0)
    assert metrics_to_tokens({"input_tokens": 10, "output_tokens": 2}) == (10, 2)
    # OpenAI-style aliases
    assert metrics_to_tokens({"prompt_tokens": 7, "completion_tokens": 3}) == (7, 3)

    class _Pydanticish:
        def model_dump(self):
            return {"input_tokens": 5, "output_tokens": 1}

    assert metrics_to_tokens(_Pydanticish()) == (5, 1)

    class _Broken:
        def model_dump(self):
            raise RuntimeError("shape changed again")

    assert metrics_to_tokens(_Broken()) == (0, 0)  # a bad metrics object costs a log line, not a turn


@test("stream_usage", "a streamed turn lands in usage_log")
async def t_streamed_turn_is_billed(ctx: TestContext) -> None:
    """End-to-end on the dispatcher: stream a turn, then read the ledger."""
    from src.memory.db import MemoryDB
    from src.models import stream_usage
    from src.models.budget import BudgetTracker
    from src.models.dispatcher import ModelDispatcher, RoutingDecision

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        class _Provider:
            """Stands in for TeamRouterProvider: streams deltas and reports its
            run's tokens through the sink, exactly as the runtime's
            RunCompletedEvent does."""

            async def stream(self, messages, **kwargs):
                yield "hel"
                yield "lo"
                stream_usage.record(input_tokens=12_000, output_tokens=300)

        dispatcher = ModelDispatcher.__new__(ModelDispatcher)
        dispatcher._budget = BudgetTracker(db, 0.0)
        dispatcher._get_team_provider = lambda rid: _Provider()
        dispatcher._remember_pick = lambda sid, rid: None

        async def _resolve(_sid):
            return RoutingDecision(
                reason="test", primary_model="local:claude-sonnet-4-6",
                candidates=["local:claude-sonnet-4-6"],
            )

        dispatcher._resolve_entry_model = _resolve

        sid = f"event:{int(time.time() * 1000)}"
        chunks = [
            c async for c in dispatcher.stream(
                [{"role": "user", "content": "hi"}], session_id=sid,
            )
        ]
        assert "".join(chunks) == "hello"

        conn = await db._ensure_connected()
        cursor = await conn.execute(
            "SELECT model, input_tokens, output_tokens FROM usage_log "
            "WHERE session_id = ?",
            (sid,),
        )
        rows = [dict(r) for r in await cursor.fetchall()]
        assert len(rows) == 1, (
            f"a streamed turn must produce exactly one usage row, got {rows}"
        )
        row = rows[0]
        assert row["input_tokens"] == 12_000
        assert row["output_tokens"] == 300
        assert row["model"] == "local:claude-sonnet-4-6"
    finally:
        await db.close()
