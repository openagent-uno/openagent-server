"""``PUT /api/sessions/{id}/model`` — pin a session to a model.

Regression guard for a bug that made the endpoint impossible to use: the
handler validated with ``db.get_model(runtime_id)``, but ``get_model`` takes
the surrogate row id and casts it with ``int()``, so every pin raised
``invalid literal for int()`` and returned 500. The endpoint had never
succeeded, which is why no client called it and why the desktop app kept its
model choice in device-local state instead.

The fake DB below reproduces exactly that distinction — ``get_model`` casts,
``list_models_enriched`` carries the derived ``runtime_id`` — so reverting to
the old lookup fails here rather than in front of a user.
"""
from __future__ import annotations

import json

from ._framework import TestContext, test

_ROWS = [
    {"id": 1, "runtime_id": "vendor:model-a", "enabled": 1,
     "provider_enabled": 1, "provider_name": "vendor"},
    {"id": 2, "runtime_id": "vendor:model-off", "enabled": 0,
     "provider_enabled": 1, "provider_name": "vendor"},
    {"id": 3, "runtime_id": "dead:model-c", "enabled": 1,
     "provider_enabled": 0, "provider_name": "dead"},
]


class _FakeDB:
    def __init__(self):
        self.pinned: dict[str, str] = {}

    async def get_model(self, model_id):
        # Mirrors the real method: a surrogate id, cast with int(). Handing it
        # a runtime_id is what used to blow up.
        int(model_id)
        return next((r for r in _ROWS if r["id"] == int(model_id)), None)

    async def list_models_enriched(self, **_):
        return _ROWS

    async def pin_session_model(self, session_id, runtime_id):
        self.pinned[session_id] = runtime_id

    async def unpin_session_model(self, session_id):
        self.pinned.pop(session_id, None)


class _FakeRequest:
    def __init__(self, db, body=None, session_id="s1"):
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

    @property
    def can_read_body(self):
        # aiohttp flips this once the body is consumed; modelling it keeps a
        # double-read bug from passing here.
        return self._body is not None and not self._read

    async def json(self):
        self._read = True
        return self._body


async def _pin(db, body):
    from src.gateway.api import sessions as api

    resp = await api.handle_pin(_FakeRequest(db, body))
    return resp.status, json.loads(resp.body.decode())


@test("rest_session_pin", "a runtime_id pins the session (the 500 is gone)")
async def t_pin_by_runtime_id(ctx: TestContext) -> None:
    db = _FakeDB()
    status, body = await _pin(db, {"runtime_id": "vendor:model-a"})
    assert status == 200, (status, body)
    assert body["pinned"] is True
    assert db.pinned["s1"] == "vendor:model-a"


@test("rest_session_pin", "unknown / disabled model and disabled provider are refused")
async def t_pin_refusals(ctx: TestContext) -> None:
    db = _FakeDB()

    status, _ = await _pin(db, {"runtime_id": "vendor:nope"})
    assert status == 404

    status, body = await _pin(db, {"runtime_id": "vendor:model-off"})
    assert status == 400 and "disabled" in body["error"]

    # A model whose PROVIDER is off cannot run either; pinning to it would
    # strand the session on a model the dispatcher skips.
    status, body = await _pin(db, {"runtime_id": "dead:model-c"})
    assert status == 400 and "provider" in body["error"]

    status, _ = await _pin(db, {})
    assert status == 400

    assert db.pinned == {}, "a refused pin must not write"


@test("rest_session_pin", "unpin clears the pin")
async def t_unpin(ctx: TestContext) -> None:
    from src.gateway.api import sessions as api

    db = _FakeDB()
    await _pin(db, {"runtime_id": "vendor:model-a"})
    resp = await api.handle_unpin(_FakeRequest(db))
    assert resp.status == 200
    assert "s1" not in db.pinned
