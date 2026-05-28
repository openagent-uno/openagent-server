"""NativeProvider.stream — hermetic zero-delta fallback coverage.

Friday's stuck Telegram turns were hitting an Agno streamed turn that
finished with zero ``RunContentEvent`` deltas. The provider now falls back to
its own ``generate()`` before control returns to Agent.run_stream, which keeps
the reply on the provider side of the cancellation-poison boundary.
"""
from __future__ import annotations

from typing import Any

from ._framework import TestContext, test


@test("agno_stream", "stream zero-delta fallback yields generate() content")
async def t_agno_stream_zero_delta_fallback(_ctx: TestContext) -> None:
    from src.models.native_provider import NativeProvider
    from src.models.base import ModelResponse

    provider = NativeProvider(
        model="openai:gpt-4o-mini",
        api_key="x",
        providers_config=[],
    )

    class _EmptyRunner:
        def arun(self, *_args: Any, **_kwargs: Any):
            async def _gen():
                if False:
                    yield None
            return _gen()

    provider._compatible_mcp_toolkits = lambda: ([], [])  # type: ignore[assignment]
    provider._ensure_team = lambda system="": _EmptyRunner()  # type: ignore[assignment]
    provider._ensure_agent = lambda system=None: _EmptyRunner()  # type: ignore[assignment]

    generate_calls: list[dict[str, Any]] = []

    async def _fake_generate(
        messages,
        system=None,
        tools=None,
        on_status=None,
        session_id=None,
    ):
        generate_calls.append(
            {
                "messages": messages,
                "system": system,
                "session_id": session_id,
                "on_status": on_status is not None,
            }
        )
        return ModelResponse(content="Recovered from zero-delta stream", model=provider.model)

    provider.generate = _fake_generate  # type: ignore[assignment]

    yielded: list[str] = []
    async for chunk in provider.stream(
        [{"role": "user", "content": "hi"}],
        system="system-prompt",
        session_id="agno-zero-delta",
    ):
        yielded.append(chunk)

    assert yielded == ["Recovered from zero-delta stream"], yielded
    assert len(generate_calls) == 1, generate_calls
    assert generate_calls[0]["session_id"] == "agno-zero-delta", generate_calls
