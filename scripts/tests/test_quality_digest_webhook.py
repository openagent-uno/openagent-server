"""Quality-digest alert WEBHOOK tests — the "reach a human" half.

Pins the new optional ``OPENAGENT_QUALITY_DIGEST_ALERT_WEBHOOK_URL``: when set,
each newly active ``quality.alert`` is POSTed (compact payload) IN ADDITION to
the existing ``elog``; when unset, nothing is POSTed and the ``elog`` behaviour
is byte-identical to before; a dead endpoint (POST raises) never propagates and
is logged as ``quality.webhook_error``; and a persisting condition is not
re-POSTed every cycle (edge-trigger de-dupe). Pure-unit: temp events.jsonl,
``elog`` captured, the HTTP client (``urllib`` ``urlopen``) mocked — no network.
"""
from __future__ import annotations

import contextlib
import json
import time

from ._framework import TestContext, test
# Reuse the sibling module's tiny fixtures (env / temp events file / elog
# capture / the canonical row set) so the two halves can't drift.
from .test_quality_digest import _capture, _env, _events_file, _rows


class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b""


@contextlib.contextmanager
def _mock_urlopen(sink: list, *, raise_exc: Exception | None = None):
    """Patch the module's ``urlopen`` seam. Records each request into ``sink``
    (url / decoded payload / method / timeout) or raises to simulate a dead
    endpoint."""
    import src.core.quality_digest as qd

    orig = qd._urllib_request.urlopen

    def fake(req, timeout=None):
        sink.append({
            "url": req.full_url,
            "method": req.get_method(),
            "timeout": timeout,
            "headers": {k.lower(): v for k, v in req.header_items()},
            "payload": json.loads(req.data.decode("utf-8")),
        })
        if raise_exc is not None:
            raise raise_exc
        return _FakeResp()

    qd._urllib_request.urlopen = fake
    try:
        yield
    finally:
        qd._urllib_request.urlopen = orig


@test("quality", "webhook set → alert POSTs the expected payload + still elogs")
async def t_webhook_posts(ctx: TestContext) -> None:
    import src.core.quality_digest as qd

    now = time.time()
    posts: list = []
    # _rows: avg 0.533 < 0.7 floor → quality_low; 1 fabrication → fabrication.
    with _events_file(_rows(now)) as p, _capture() as ev, _env(
        OPENAGENT_QUALITY_DIGEST_MIN_SCORE_ALERT="0.7",
        OPENAGENT_QUALITY_DIGEST_ALERT_WEBHOOK_URL="https://hook.example/quality",
    ):
        qd._alerted_kinds.clear()
        with _mock_urlopen(posts):
            digest, alerts = qd.run_once(3600, path=p)

    # The existing elog side is untouched — quality.alert still emitted.
    alert_logs = {kw["kind"] for n, kw in ev if n == "quality.alert"}
    assert "quality_low" in alert_logs and "fabrication" in alert_logs, ev

    # ...and the webhook POSTed one compact body per newly active alert.
    assert len(posts) == 2, posts
    assert all(pp["url"] == "https://hook.example/quality" for pp in posts), posts
    assert all(pp["method"] == "POST" for pp in posts), posts
    assert all(pp["timeout"] == qd._WEBHOOK_TIMEOUT_S for pp in posts), posts
    assert all(pp["headers"].get("content-type") == "application/json" for pp in posts), posts

    by_kind = {pp["payload"]["kind"]: pp["payload"] for pp in posts}
    assert set(by_kind) == {"quality_low", "fabrication"}, by_kind
    for body in by_kind.values():
        assert body["event"] == "quality.alert"
        assert body["severity"] == "warning"
        assert body["source"] == "quality_digest"
        assert isinstance(body["summary"], str) and body["summary"]
        assert isinstance(body["counts"], dict)
        assert isinstance(body["ts"], (int, float))
        assert body["window_hours"] == digest["window_hours"]
    # counts carries the alert's numeric detail (kind is promoted out of it).
    assert by_kind["fabrication"]["counts"].get("count") == 1, by_kind
    assert "kind" not in by_kind["fabrication"]["counts"], by_kind


@test("quality", "webhook unset → no POST, elog unchanged")
async def t_webhook_unset(ctx: TestContext) -> None:
    import src.core.quality_digest as qd

    now = time.time()
    posts: list = []
    with _events_file(_rows(now)) as p, _capture() as ev, _env(
        OPENAGENT_QUALITY_DIGEST_MIN_SCORE_ALERT="0.7",
        OPENAGENT_QUALITY_DIGEST_ALERT_WEBHOOK_URL=None,
    ):
        qd._alerted_kinds.clear()
        with _mock_urlopen(posts):
            qd.run_once(3600, path=p)

    assert posts == [], "unset webhook must POST nothing"
    # elog path is byte-identical to before — alerts still logged.
    assert any(n == "quality.alert" for n, _ in ev), ev


@test("quality", "webhook exception is swallowed + logged, never propagates")
async def t_webhook_error_contained(ctx: TestContext) -> None:
    import src.core.quality_digest as qd

    now = time.time()
    posts: list = []
    with _events_file(_rows(now)) as p, _capture() as ev, _env(
        OPENAGENT_QUALITY_DIGEST_MIN_SCORE_ALERT="0.7",
        OPENAGENT_QUALITY_DIGEST_ALERT_WEBHOOK_URL="https://hook.example/quality",
    ):
        qd._alerted_kinds.clear()
        with _mock_urlopen(posts, raise_exc=RuntimeError("connection refused")):
            # Must NOT raise even though every POST blows up.
            digest, alerts = qd.run_once(3600, path=p)

    assert alerts, "run_once must still return the alerts normally"
    assert any(n == "quality.webhook_error" for n, _ in ev), ev


@test("quality", "webhook is edge-triggered — a persisting alert isn't re-POSTed")
async def t_webhook_dedupe(ctx: TestContext) -> None:
    import src.core.quality_digest as qd

    now = time.time()
    posts: list = []
    with _events_file(_rows(now)) as p, _capture(), _env(
        OPENAGENT_QUALITY_DIGEST_MIN_SCORE_ALERT="0.7",
        OPENAGENT_QUALITY_DIGEST_ALERT_WEBHOOK_URL="https://hook.example/quality",
    ):
        qd._alerted_kinds.clear()
        with _mock_urlopen(posts):
            qd.run_once(3600, path=p)
            first = len(posts)
            qd.run_once(3600, path=p)  # same conditions still active
            second = len(posts)

    assert first == 2, posts
    assert second == first, "a persisting condition must not re-POST every cycle"
