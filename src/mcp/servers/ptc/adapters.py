"""Provider adapter for the in-process ``ptc`` MCP.

Follows the runtime ``Toolkit`` pattern (see ``tool_search/adapters.py`` and
``skills/adapters.py``): a plain async callable with type hints + docstring,
wrapped once. The runtime turns the docstring/signature into the schema the
model sees.

``build_runtime_toolkit`` takes a ``pool`` kwarg (injected by ``MCPPool`` when
it detects the signature accepts it — same mechanism as ``tool-search``); the
bridge needs it to dispatch through ``_call_tool_impl``. Runtime knobs
(``require_sandbox`` / ``allowed_tools`` / ``max_tool_calls`` / ``timeout_s``)
are read from ``ptc_settings(load_config())`` at build time; the ``enabled``
gate itself was already checked by ``config_gated_mcp_entries`` before this
factory ever runs.
"""
from __future__ import annotations

from typing import Any


def build_runtime_toolkit(*, pool: Any | None = None) -> Any:
    from src.mcp._runtime import Toolkit
    from src.core.config import load_config, ptc_settings
    from src.core.dry_run import is_dry_run
    from src.mcp.servers.ptc import handlers

    if pool is None:
        raise RuntimeError("ptc runtime adapter requires a pool kwarg")

    settings = ptc_settings(load_config())

    async def run_python(code: str) -> dict:
        """Run a Python script that reaches your OWN tools, returning its stdout.

        Write ordinary Python in ``code``. ``call_tool(server, tool, args)`` is
        already in scope (no import needed) and invokes any tool you have — the
        same ``server``/``tool`` names as ``tool_search_call_tool`` — returning
        its JSON result. The script runs in a sandbox; ONLY what it prints to
        stdout is returned to you.

        Reach for this to collapse a multi-step tool pipeline into ONE turn:
        fan out over many items, filter/aggregate in code, or join results from
        several tools, without paying a model round-trip per tool call. Print
        just the distilled answer — stdout is capped like any tool result.
        """
        return await handlers.run_python_impl(
            code, pool=pool, settings=settings, dry_run=is_dry_run(),
        )

    return Toolkit(name="ptc", tools=[run_python])
