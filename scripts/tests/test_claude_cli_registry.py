"""ClaudeCLIRegistry — multi-model routing without losing --resume state.

The registry holds one ClaudeCLI per model id so concurrent sessions can
use different Claude models. Tests here do NOT spawn the claude binary;
they monkey-patch the instances so the control flow is observable
without external dependencies.
"""
from __future__ import annotations

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


async def _fake_generate_factory(model_id: str):
    """Build a fake ClaudeCLI.generate that records calls and returns a stub."""

    calls: list[str] = []

    async def _fake_generate(
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Any = None,
        session_id: str | None = None,
    ) -> _FakeResponse:
        calls.append(session_id or "")
        return _FakeResponse(f"claude-cli/{model_id}")

    return calls, _fake_generate


@test("claude_cli_registry", "pin_session then generate forwards to the right instance")
async def t_registry_pin_and_dispatch(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLIRegistry

    registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")
    registry.pin_session("sess-a", "claude-sonnet-4-6")
    registry.pin_session("sess-b", "claude-haiku-4-5")

    # Pre-populate the internal instance map with fakes so generate doesn't
    # try to spawn the claude binary.
    calls_sonnet, fake_sonnet = await _fake_generate_factory("claude-sonnet-4-6")
    calls_haiku, fake_haiku = await _fake_generate_factory("claude-haiku-4-5")
    # Use _get_or_create so the registry sees them, then override generate.
    inst_s = registry._get_or_create("claude-sonnet-4-6")
    inst_h = registry._get_or_create("claude-haiku-4-5")
    inst_s.generate = fake_sonnet  # type: ignore[assignment]
    inst_h.generate = fake_haiku  # type: ignore[assignment]

    resp_a = await registry.generate([{"role": "user", "content": "hi"}], session_id="sess-a")
    resp_b = await registry.generate([{"role": "user", "content": "hi"}], session_id="sess-b")

    assert resp_a.model.endswith("claude-sonnet-4-6")
    assert resp_b.model.endswith("claude-haiku-4-5")
    assert calls_sonnet == ["sess-a"]
    assert calls_haiku == ["sess-b"]


@test("claude_cli_registry", "model_override beats default")
async def t_registry_override_wins(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLIRegistry

    registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")

    calls_opus, fake_opus = await _fake_generate_factory("claude-opus-4-6")
    inst_o = registry._get_or_create("claude-opus-4-6")
    inst_o.generate = fake_opus  # type: ignore[assignment]

    # No default-model instance needed; override should route past it.
    resp = await registry.generate(
        [{"role": "user", "content": "hi"}],
        session_id="sess-new",
        model_override="claude-cli/claude-opus-4-6",
    )
    assert resp.model.endswith("claude-opus-4-6")
    assert calls_opus == ["sess-new"]


@test("claude_cli_registry", "fan-out: set_db applies to every instance")
async def t_fan_out_set_db(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLIRegistry

    registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")
    inst_a = registry._get_or_create("claude-sonnet-4-6")
    inst_b = registry._get_or_create("claude-haiku-4-5")

    class FakeDB:
        pass

    db = FakeDB()
    registry.set_db(db)
    assert inst_a._db is db
    assert inst_b._db is db
