"""Pre-delivery precondition — the cheap "is there still work?" check.

A queued delivery runs against state that has moved on. Discovering that inside
the model turn costs a full turn to learn nothing: measured on a real support
webhook, ~22% of deliveries did exactly that. These tests pin the two things
that make the check safe to have at all — it skips only on an unambiguous
match, and every other outcome runs the delivery.
"""
from __future__ import annotations

import asyncio
import json

from ._framework import TestContext, test


def _event(spec, event_id: str = "evt_1") -> dict:
    return {"id": event_id, "name": "test event",
            "precondition_json": json.dumps(spec) if spec is not None else None}


_PAYLOAD = {"payload": {"thread_id": "th_abc123"}}


def _spec(**over) -> dict:
    spec = {
        "type": "http",
        "url": "http://replio/threads/{payload.payload.thread_id}",
        "headers": {"Authorization": "Bearer ${TEST_PRECOND_KEY}"},
        "skip_when": {"path": "waiting_for_team", "equals": False},
        "skip_output": "already handled",
    }
    spec.update(over)
    return spec


class _Fetch:
    """Stand-in for the HTTP GET. Records the call; returns or raises."""

    def __init__(self, body=None, exc: Exception | None = None):
        self.body, self.exc = body, exc
        self.calls: list[tuple[str, dict, float]] = []

    async def __call__(self, url, headers, timeout_s):
        self.calls.append((url, headers, timeout_s))
        if self.exc:
            raise self.exc
        return self.body


def _with_fetch(fetch, fn):
    from src.core import event_precondition as ep
    import os
    orig, os.environ["TEST_PRECOND_KEY"] = ep._fetch_json, "k-secret"
    ep._fetch_json = fetch
    try:
        return asyncio.get_event_loop().run_until_complete(fn(ep)) \
            if False else fn(ep)
    finally:
        ep._fetch_json = orig
        os.environ.pop("TEST_PRECOND_KEY", None)


@test("event-precondition", "no precondition configured → always runs")
async def t_no_spec(_ctx: TestContext) -> None:
    from src.core.event_precondition import should_skip
    assert await should_skip(_event(None), _PAYLOAD) == (False, "")
    assert await should_skip({"id": "e"}, _PAYLOAD) == (False, "")


@test("event-precondition", "condition matches → skip, with the configured reason")
async def t_skip_on_match(_ctx: TestContext) -> None:
    from src.core import event_precondition as ep
    import os
    fetch = _Fetch(body={"waiting_for_team": False, "status": "open"})
    orig, ep._fetch_json = ep._fetch_json, fetch
    os.environ["TEST_PRECOND_KEY"] = "k-secret"
    try:
        skip, reason = await ep.should_skip(_event(_spec()), _PAYLOAD)
    finally:
        ep._fetch_json = orig
        os.environ.pop("TEST_PRECOND_KEY", None)

    assert skip is True, "an unambiguous match must skip"
    assert reason == "already handled", reason
    url, headers, timeout = fetch.calls[0]
    assert url == "http://replio/threads/th_abc123", url          # payload interpolated
    assert headers["Authorization"] == "Bearer k-secret", headers  # env interpolated
    assert 0 < timeout <= 30


@test("event-precondition", "condition does not match → runs")
async def t_run_on_mismatch(_ctx: TestContext) -> None:
    from src.core import event_precondition as ep
    import os
    orig, ep._fetch_json = ep._fetch_json, _Fetch(body={"waiting_for_team": True})
    os.environ["TEST_PRECOND_KEY"] = "k"
    try:
        assert await ep.should_skip(_event(_spec()), _PAYLOAD) == (False, "")
    finally:
        ep._fetch_json = orig
        os.environ.pop("TEST_PRECOND_KEY", None)


@test("event-precondition", "every uncertain outcome fails OPEN (runs the delivery)")
async def t_fails_open(_ctx: TestContext) -> None:
    """The whole safety argument. Skipping a real customer message to save a
    model call is a far worse bug than the cost — so anything short of a clear
    match must run."""
    from src.core import event_precondition as ep
    import os
    os.environ["TEST_PRECOND_KEY"] = "k"
    orig = ep._fetch_json
    cases = {
        "endpoint unreachable": (_spec(), _Fetch(exc=TimeoutError("timed out"))),
        "endpoint 500":         (_spec(), _Fetch(exc=RuntimeError("HTTP 500"))),
        "field absent":         (_spec(), _Fetch(body={"status": "open"})),
        "body not an object":   (_spec(), _Fetch(body=["nope"])),
        "unknown spec type":    (_spec(type="carrier-pigeon"), _Fetch(body={})),
        "skip_when missing":    ({"type": "http", "url": "http://x"}, _Fetch(body={})),
        "no equals clause":     (_spec(skip_when={"path": "waiting_for_team"}),
                                 _Fetch(body={"waiting_for_team": False})),
        "payload ref missing":  (_spec(url="http://replio/{payload.nope.here}"),
                                 _Fetch(body={"waiting_for_team": False})),
    }
    try:
        for label, (spec, fetch) in cases.items():
            ep._fetch_json = fetch
            got = await ep.should_skip(_event(spec), _PAYLOAD)
            assert got == (False, ""), f"{label}: expected run, got {got}"
        # A malformed spec string must not raise either.
        ep._fetch_json = _Fetch(body={})
        bad = {"id": "e", "precondition_json": "{not json"}
        assert await ep.should_skip(bad, _PAYLOAD) == (False, "")
    finally:
        ep._fetch_json = orig
        os.environ.pop("TEST_PRECOND_KEY", None)


@test("event-precondition", "a missing credential does not leak an unauthenticated call")
async def t_missing_env_does_not_call(_ctx: TestContext) -> None:
    from src.core import event_precondition as ep
    import os
    os.environ.pop("TEST_PRECOND_KEY", None)
    fetch = _Fetch(body={"waiting_for_team": False})
    orig, ep._fetch_json = ep._fetch_json, fetch
    try:
        assert await ep.should_skip(_event(_spec()), _PAYLOAD) == (False, "")
    finally:
        ep._fetch_json = orig
    assert fetch.calls == [], "must not call out with an empty auth header"


@test("event-precondition", "a skipped delivery never reaches the model")
async def t_dispatcher_skips_without_a_turn(_ctx: TestContext) -> None:
    """End-to-end at the dispatcher: the delivery closes as `skipped`, and the
    agent is never invoked — which is the entire point of the feature."""
    from src.core import event_dispatcher as ed
    from src.core import event_precondition as ep
    import os

    updates: list[dict] = []

    class _DB:
        async def update_event_delivery(self, delivery_id, **kw):
            updates.append({"id": delivery_id, **kw})

    class _Agent:
        def __init__(self): self.called = False
        async def arun(self, *a, **k):  # noqa: ANN001
            self.called = True
            raise AssertionError("agent must not run on a skipped delivery")

    agent = _Agent()
    os.environ["TEST_PRECOND_KEY"] = "k"
    orig, ep._fetch_json = ep._fetch_json, _Fetch(body={"waiting_for_team": False})
    try:
        result = await ed.dispatch_event(
            agent=agent, db=_DB(), scheduler=None,
            event={**_event(_spec()), "action_kind": "prompt",
                   "prompt_template": "handle it"},
            payload=_PAYLOAD, delivery_id="dl_1", source="webhook",
        )
    finally:
        ep._fetch_json = orig
        os.environ.pop("TEST_PRECOND_KEY", None)

    assert result["status"] == "skipped", result
    assert agent.called is False
    assert [u["status"] for u in updates] == ["skipped"], updates
    assert "running" not in [u.get("status") for u in updates]
    assert updates[0]["output"] == "already handled"
    assert updates[0].get("finished_at"), "a terminal delivery needs a finish time"
