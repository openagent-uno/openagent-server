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
