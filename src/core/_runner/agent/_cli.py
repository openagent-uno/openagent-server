"""Terminal print helpers — inert stubs.

OpenAgent serves the agent over the gateway (WebSocket + REST). The
upstream framework also exposed an ``Agent.print_response()`` style API
for one-shot terminal usage; that path is unused here. The functions
below exist only so the agent's method bindings stay importable.
"""

from __future__ import annotations


def agent_print_response(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "Terminal print helpers are not available in the OpenAgent runtime. "
        "Stream responses through the gateway instead."
    )


async def agent_aprint_response(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "Terminal print helpers are not available in the OpenAgent runtime."
    )


def cli_app(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "Terminal CLI loop is not available in the OpenAgent runtime."
    )


async def acli_app(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "Terminal CLI loop is not available in the OpenAgent runtime."
    )
