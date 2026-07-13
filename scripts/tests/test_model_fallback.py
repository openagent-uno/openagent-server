"""Model fallback: a rate-limit on the primary provider degrades the turn to
the configured secondary (e.g. DeepSeek) instead of failing the whole run.

`_build_agent` reads a ``fallback:`` config section and passes a
``FallbackConfig`` to the Agent; these guard the mechanism it relies on.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("model_fallback", "rate-limit error selects the on_rate_limit list")
async def t_rate_limit_uses_on_rate_limit(_ctx: TestContext) -> None:
    from src.core.runtime_errors import ModelRateLimitError
    from src.models.providers.fallback import FallbackConfig, get_fallback_models

    fc = FallbackConfig(on_rate_limit=["deepseek:deepseek-v4-pro"])
    assert fc.has_fallbacks  # property, not a method
    models = get_fallback_models(fc, ModelRateLimitError("rate limited"))
    assert models == ["deepseek:deepseek-v4-pro"]


@test("model_fallback", "no fallback config keeps today's no-fallback behaviour")
async def t_none_config_no_fallback(_ctx: TestContext) -> None:
    from src.core.runtime_errors import ModelRateLimitError
    from src.models.providers.fallback import get_fallback_models

    assert get_fallback_models(None, ModelRateLimitError("rate limited")) is None


@test("model_fallback", "_build_agent accepts and wires fallback config")
async def t_build_agent_wires_fallback_config(ctx: TestContext) -> None:
    from src.core.server import _build_agent

    agent = _build_agent({
        "name": "fallback-test",
        "memory": {"db_path": str(ctx.db_path)},
        "fallback": {
            "on_rate_limit": ["deepseek:deepseek-v4-pro"],
            "on_error": ["openai:gpt-4o-mini"],
        },
    })

    assert agent.fallback_config is not None
    assert agent.fallback_config.on_rate_limit == ["deepseek:deepseek-v4-pro"]
    assert agent.fallback_config.on_error == ["openai:gpt-4o-mini"]

    agent._prepare_model_runtime(agent.model)
    assert getattr(agent.model, "_fallback_config", None) is agent.fallback_config
