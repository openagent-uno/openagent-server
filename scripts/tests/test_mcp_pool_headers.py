"""Regression test: ``spec.headers`` MUST be forwarded to the streamable-http
transport when an MCP has ``url`` set.

Bug history
-----------

**2026-05-25**: Field reproduction against ``https://mcp.mixout.ai/`` with a
known-good Bearer token. ``_ServerSpec.headers`` was being populated correctly
from YAML and from DB rows, but ``_build_and_enter_toolkit`` constructed
``MCPTools(url=..., transport="streamable-http", ...)`` *without* a
``server_params=`` argument — so the headers never reached the wire. Every
auth-gated remote MCP returned 401, the underlying anyio stream half-closed,
and ``session.initialize().send_nowait()`` raised
``anyio.ClosedResourceError`` inside the runtime's blanket-except, leaving the MCP
flagged dormant.

The fix passes a ``StreamableHTTPClientParams(url=..., headers=...)`` via
``server_params=``; this test asserts that those headers actually land on
the ``MCPTools`` constructor.
"""
from __future__ import annotations

import sys
import types
from typing import Any

from ._framework import TestContext, test


def _install_fake_runtime() -> tuple[dict[str, Any], type]:
    """Stub ``src.mcp._runtime.mcp`` and ``src.mcp._runtime.mcp.params`` so the lazy
    import inside ``_build_and_enter_toolkit`` resolves to our fakes.

    Returns ``(captured, FakeStreamableHTTPClientParams)``. ``captured``
    is filled with the kwargs the pool passes to ``MCPTools(...)``.
    """
    captured: dict[str, Any] = {}

    class FakeMCPTools:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeMCPTools":
            # Mimic the runtime's MCPTools enter: present an empty tool dict.
            self.functions = {}
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeStreamableHTTPClientParams:
        def __init__(self, *, url: str, headers: dict[str, str] | None = None,
                     **rest: Any) -> None:
            self.url = url
            self.headers = headers
            self.rest = rest

    # src.mcp._runtime.mcp.mcp
    fake_mcp_mod = types.ModuleType("src.mcp._runtime.mcp.mcp")
    fake_mcp_mod.MCPTools = FakeMCPTools  # type: ignore[attr-defined]

    # src.mcp._runtime.mcp.params
    fake_params_mod = types.ModuleType("src.mcp._runtime.mcp.params")
    fake_params_mod.StreamableHTTPClientParams = (  # type: ignore[attr-defined]
        FakeStreamableHTTPClientParams
    )

    sys.modules["src.mcp._runtime.mcp.mcp"] = fake_mcp_mod
    sys.modules["src.mcp._runtime.mcp.params"] = fake_params_mod
    return captured, FakeStreamableHTTPClientParams


def _restore_real_runtime(saved: dict[str, Any]) -> None:
    for key, mod in saved.items():
        if mod is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = mod


@test(
    "mcp_pool_headers",
    "spec.headers reach the streamable-http transport via server_params",
)
async def t_headers_forwarded_to_transport(ctx: TestContext) -> None:
    # Save originals so we don't poison the rest of the suite.
    saved = {
        "src.mcp._runtime.mcp.mcp": sys.modules.get("src.mcp._runtime.mcp.mcp"),
        "src.mcp._runtime.mcp.params": sys.modules.get("src.mcp._runtime.mcp.params"),
    }
    captured, fake_params_cls = _install_fake_runtime()
    try:
        from src.mcp.pool import MCPPool, _ServerSpec

        spec = _ServerSpec(
            name="mixout",
            url="https://example.invalid/",
            headers={"Authorization": "Bearer test"},
        )
        pool = MCPPool([spec])

        # Stub _safe_enter so we don't try to actually open a connection —
        # we only care that MCPTools was constructed with the right kwargs.
        async def _noop_safe_enter(self_pool: Any, toolkit: Any,
                                   _spec: Any) -> Any:
            return toolkit

        pool._safe_enter = _noop_safe_enter.__get__(pool, MCPPool)  # type: ignore

        result = await pool._build_and_enter_toolkit(spec)
        assert result is not None, "toolkit construction returned None"

        # The fix must route headers via server_params, not as a top-level
        # MCPTools kwarg.
        assert "server_params" in captured, (
            f"MCPTools was called without server_params; got kwargs={sorted(captured)}"
        )
        sp = captured["server_params"]
        assert isinstance(sp, fake_params_cls), (
            f"server_params is {type(sp).__name__}, expected "
            f"StreamableHTTPClientParams"
        )
        assert sp.url == "https://example.invalid/", sp.url
        assert sp.headers == {"Authorization": "Bearer test"}, sp.headers
        # The legacy top-level ``url=`` form must NOT be used — the bug was
        # that headers were dropped because they had no slot there.
        assert "url" not in captured, (
            f"MCPTools should not receive top-level url= when using "
            f"server_params; got url={captured.get('url')!r}"
        )
    finally:
        _restore_real_runtime(saved)
