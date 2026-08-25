"""``PATCH /api/sessions/{id}`` — the title write must also claim an owner.

The app titles a chat from its first message, and that PATCH is often what
*creates* the ``sessions`` row. It used to write the title and nothing else,
so the row landed with ``metadata = {"title": ...}`` and no ``client_id`` —
and ``list_all_sessions`` filters on exactly that field. The row was therefore
invisible in every session listing from the moment it was born.

Nothing noticed while a turn was running (the client keeps its own copy in
memory) and the runtime later rewrote the row with an owner when it persisted
the runs. But a turn that DIED before that — the agent restarting mid-turn is
the everyday cause — left an ownerless, run-less row: the chat was on disk and
the user could never see it again. Observed on a live agent, 2026-08-25.
"""
from __future__ import annotations

import json

from ._framework import TestContext, test


class _FakeDB:
    def __init__(self):
        self.calls: list[dict] = []

    async def upsert_session(self, session_id, **kwargs):
        self.calls.append({"session_id": session_id, **kwargs})


class _FakeRequest:
    """Enough of an aiohttp request: match_info, a JSON body, and the
    Mapping ``get`` the auth middleware's values are read through."""

    def __init__(self, db, body=None, *, session_id="s1", values=None):
        class _Holder:
            pass

        agent = _Holder()
        agent.memory_db = db
        gateway = _Holder()
        gateway.agent = agent
        self.app = {"gateway": gateway}
        self.match_info = {"session_id": session_id}
        self._body = body
        self._read = False
        self._values = values or {}

    def get(self, key, default=None):
        return self._values.get(key, default)

    @property
    def can_read_body(self):
        return self._body is not None and not self._read

    async def json(self):
        self._read = True
        return self._body


async def _patch(db, body, values=None):
    from src.gateway.api import sessions as api

    resp = await api.handle_patch_metadata(_FakeRequest(db, body, values=values))
    return resp.status, json.loads(resp.body.decode())


@test("session_patch_owner", "the user handle is stamped as the row's owner")
async def t_handle_wins(ctx: TestContext) -> None:
    db = _FakeDB()
    status, _ = await _patch(
        db, {"title": "ciao"},
        values={"user_handle": "marco", "client_id": "deadbeef"},
    )
    assert status == 200
    # The handle, not the device pubkey: the listing is meant to be the same
    # on every device the user owns.
    assert db.calls == [{"session_id": "s1", "client_id": "marco",
                         "title": "ciao", "model": None}], db.calls


@test("session_patch_owner", "without a handle the device id owns the row")
async def t_device_fallback(ctx: TestContext) -> None:
    db = _FakeDB()
    status, _ = await _patch(db, {"title": "ciao"}, values={"client_id": "deadbeef"})
    assert status == 200
    assert db.calls[0]["client_id"] == "deadbeef", db.calls


@test("session_patch_owner", "no caller identity behaves exactly as before")
async def t_anonymous_unchanged(ctx: TestContext) -> None:
    # Trusted-proxy / single-owner deploys have neither value. The write must
    # still happen, just without claiming an owner.
    db = _FakeDB()
    status, _ = await _patch(db, {"title": "ciao", "model": "local:m"})
    assert status == 200
    assert db.calls[0]["client_id"] is None
    assert db.calls[0]["model"] == "local:m"


@test("session_patch_owner", "an empty patch is refused and writes nothing")
async def t_empty_patch(ctx: TestContext) -> None:
    db = _FakeDB()
    status, body = await _patch(db, {}, values={"user_handle": "marco"})
    assert status == 400 and "required" in body["error"]
    assert db.calls == [], "a refused patch must not touch the row"
