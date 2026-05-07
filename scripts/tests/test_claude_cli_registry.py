"""ClaudeCLIRegistry — multi-model routing without losing --resume state.

The registry holds one ClaudeCLI per model id so concurrent sessions can
use different Claude models. Tests here do NOT spawn the claude binary;
they monkey-patch the instances so the control flow is observable
without external dependencies.
"""
from __future__ import annotations

import uuid
from typing import Any

from ._framework import TestContext, test


class _FakeResponse:
    """Stand-in for ModelResponse (only ``.model`` is inspected by callers)."""

    def __init__(self, model: str):
        self.content = "ok"
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason = "stop"
        self.model = model


async def _fake_generate_factory():
    """Build a fake ClaudeCLI.generate that records (session, override) calls.

    Returns a response whose ``.model`` echoes the override it received
    so callers can assert the registry forwarded the right model to the
    per-session instance.
    """

    calls: list[tuple[str, str | None]] = []

    async def _fake_generate(
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Any = None,
        session_id: str | None = None,
        model_override: str | None = None,
    ) -> _FakeResponse:
        calls.append((session_id or "", model_override))
        return _FakeResponse(f"claude-cli/{model_override or 'unset'}")

    return calls, _fake_generate


@test("claude_cli_registry", "pin_session then generate forwards to the right instance")
async def t_registry_pin_and_dispatch(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLIRegistry

    registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")
    registry.pin_session("sess-a", "claude-sonnet-4-6")
    registry.pin_session("sess-b", "claude-haiku-4-5")

    # The registry now keys by session_id (one ClaudeCLI per session, with
    # in-place model switching via set_model). Pre-spawn the per-session
    # instances with their initial models and replace their generate with
    # a fake so no claude binary is needed.
    calls_a, fake_a = await _fake_generate_factory()
    calls_b, fake_b = await _fake_generate_factory()
    inst_a = registry._get_or_create("sess-a", "claude-sonnet-4-6")
    inst_b = registry._get_or_create("sess-b", "claude-haiku-4-5")
    inst_a.generate = fake_a  # type: ignore[assignment]
    inst_b.generate = fake_b  # type: ignore[assignment]

    resp_a = await registry.generate([{"role": "user", "content": "hi"}], session_id="sess-a")
    resp_b = await registry.generate([{"role": "user", "content": "hi"}], session_id="sess-b")

    assert resp_a.model.endswith("claude-sonnet-4-6")
    assert resp_b.model.endswith("claude-haiku-4-5")
    assert calls_a == [("sess-a", "claude-sonnet-4-6")]
    assert calls_b == [("sess-b", "claude-haiku-4-5")]


@test("claude_cli_registry", "model_override beats default")
async def t_registry_override_wins(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLIRegistry

    registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")

    calls, fake = await _fake_generate_factory()
    # Pre-spawn the per-session instance (initial model doesn't matter —
    # the override on generate() drives the actual pin).
    inst = registry._get_or_create("sess-new", "claude-sonnet-4-6")
    inst.generate = fake  # type: ignore[assignment]

    resp = await registry.generate(
        [{"role": "user", "content": "hi"}],
        session_id="sess-new",
        model_override="claude-cli/claude-opus-4-6",
    )
    assert resp.model.endswith("claude-opus-4-6")
    assert calls == [("sess-new", "claude-opus-4-6")]


@test("claude_cli_registry", "fan-out: set_db applies to every instance")
async def t_fan_out_set_db(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLIRegistry

    registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")
    inst_a = registry._get_or_create("sess-a", "claude-sonnet-4-6")
    inst_b = registry._get_or_create("sess-b", "claude-haiku-4-5")

    class FakeDB:
        pass

    db = FakeDB()
    registry.set_db(db)
    assert inst_a._db is db
    assert inst_b._db is db


@test("claude_cli_registry", "forget_session deletes DB-backed resume rows even without a live instance")
async def t_registry_forget_without_live_instance_clears_db(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB
    from openagent.models.claude_cli import ClaudeCLIRegistry

    tmp = ctx.db_path.with_name(f"claude-reg-forget-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        await db.set_sdk_session("tg:db-only", "sdk-db-only", provider="claude-cli")

        registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")
        registry.set_db(db)

        await registry.forget_session("tg:db-only")

        assert await db.get_sdk_session("tg:db-only") is None
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("claude_cli_registry", "known_session_ids includes DB-backed resume rows")
async def t_registry_known_session_ids_include_db_rows(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB
    from openagent.models.claude_cli import ClaudeCLIRegistry

    tmp = ctx.db_path.with_name(f"claude-reg-known-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        await db.set_sdk_session("tg:db-known", "sdk-known", provider="claude-cli")

        registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")
        registry.set_db(db)

        assert "tg:db-known" in registry.known_session_ids()
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
