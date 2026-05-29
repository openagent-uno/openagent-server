"""Terminal print helpers — inert stubs (team variant).

OpenAgent streams team activity through the gateway. Terminal-only print
helpers from the upstream framework are not used; the symbols stay to
keep ``Team.print_response()`` style method bindings importable.
"""

from __future__ import annotations


def team_print_response(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "Terminal print helpers are not available in the OpenAgent runtime. "
        "Stream team output through the gateway instead."
    )


async def team_aprint_response(*_args, **_kwargs):  # pragma: no cover
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


def _get_member_name(_team, entity_id):  # pragma: no cover
    return entity_id
