"""Auto-updater module — read-only sanity check.

Does not hit the network. Just confirms the symbols exist and the
current package reports a sane ``__version__``.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("updater", "updater symbols exist + current __version__ is sane")
async def t_updater_callable(ctx: TestContext) -> None:
    import openagent
    from openagent.updater import check_for_update, UpdateInfo, perform_self_update_sync
    assert openagent.__version__ and isinstance(openagent.__version__, str)
    assert callable(check_for_update)
    assert callable(perform_self_update_sync)
    fields = getattr(UpdateInfo, "_fields", None)
    assert fields and len(fields) >= 1
