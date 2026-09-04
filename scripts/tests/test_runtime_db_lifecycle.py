"""Cached runtime replacement must release every owned SQLite engine."""

from __future__ import annotations

from collections import OrderedDict

from ._framework import TestContext, test


class _Db:
    db_engine = object()
    Session = object()

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _Runtime:
    def __init__(self, db=None, members=None) -> None:
        self.db = db
        self.members = list(members or [])


@test("runtime_db_lifecycle", "team databases are closed once even when shared by members")
async def t_close_tree_deduplicates(ctx: TestContext) -> None:
    from src.models.runtime_db_lifecycle import close_runtime_databases

    shared = _Db()
    own = _Db()
    runtime = _Runtime(shared, [_Runtime(shared), _Runtime(own)])
    assert close_runtime_databases(runtime) == 2
    assert shared.closed == 1
    assert own.closed == 1


@test("runtime_db_lifecycle", "LRU eviction disposes the removed runtime")
async def t_native_eviction_closes(ctx: TestContext) -> None:
    from src.models.native_provider import _evict_oldest

    old_db = _Db()
    keep_db = _Db()
    cache = OrderedDict((
        ("old", _Runtime(old_db)),
        ("keep", _Runtime(keep_db)),
    ))
    _evict_oldest(cache, 1)
    assert list(cache) == ["keep"]
    assert old_db.closed == 1
    assert keep_db.closed == 0
