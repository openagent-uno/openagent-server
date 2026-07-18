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


@test("model_fallback",
      "sub-proxy 'no available accounts' (4xx) classifies as rate-limit and falls back")
async def t_account_exhaustion_falls_back(_ctx: TestContext) -> None:
    """The in-pod claude-sub-proxy returns a client status (e.g. 404) when all
    its rotating Claude accounts are cooling down. That is a TRANSIENT capacity
    signal ("Retry after ..."), not a config bug — it must degrade to the
    configured fallback (DeepSeek). Regression: it was hitting the "non-retryable
    4xx client error → return None" branch, so the run hard-failed (the
    skill-distiller ``failed`` with exactly this message) instead of falling
    back, and Claude-account saturation could take out live support turns."""
    from src.core.runtime_errors import ModelProviderError, ModelRateLimitError
    from src.models.providers.fallback import FallbackConfig, get_fallback_models

    fc = FallbackConfig(on_rate_limit=["deepseek:deepseek-v4-pro"])
    exhausted = ModelProviderError(
        "No available Claude OAuth accounts. Run `claude-sub-proxy login "
        "--priority 10` to add one. Last account error: HTTP 404: "
        '{"type":"error"}. Retry after about 171s.',
        status_code=404,
    )
    # Upgraded to a rate-limit despite the 4xx status …
    assert isinstance(ModelProviderError.classify(exhausted), ModelRateLimitError)
    # … so it degrades to the fallback instead of the hard-fail `return None`.
    assert get_fallback_models(fc, exhausted) == ["deepseek:deepseek-v4-pro"]

    # Guard against over-breadth: an UNRELATED real 404 (config bug) must STILL
    # be non-retryable — masking it with a silent fallback is the anti-pattern
    # the `return None` branch exists to prevent.
    real_404 = ModelProviderError("model `foo` not found", status_code=404)
    assert not isinstance(ModelProviderError.classify(real_404), ModelRateLimitError)
    assert get_fallback_models(fc, real_404) is None
