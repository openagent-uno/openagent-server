"""GET /api/commands — introspectable slash-command registry.

Exposes :data:`src.gateway.commands.COMMAND_SPECS` so rich clients (the
desktop app, the CLI) can build slash-command autocomplete and know which
commands take a *pickable* argument (``arg_source``) without hardcoding
the list. Thin bridges don't need this — they receive a ready-made picker
inside the ``command_result`` frame instead.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


async def handle_list(request: web.Request) -> web.Response:
    from aiohttp import web as _web
    from src.gateway.commands import COMMAND_SPECS

    commands = [
        {
            "name": spec.name,
            "description": spec.description,
            "help_text": spec.help_text,
            "menu_visible": spec.menu_visible,
            "help_visible": spec.help_visible,
            "arg_source": spec.arg_source,
        }
        for spec in COMMAND_SPECS
    ]
    return _web.json_response({"commands": commands})
