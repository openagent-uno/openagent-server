"""Anti-fabrication reply guard — ``src.core.reply_guard``.

The guard rewrites a reply that promises human/team follow-up when NO backing
action tool ran this turn, and is fail-open everywhere else: disabled, no
promise, a promise backed by a real handoff tool, no tool visibility, or a
regeneration failure all return the reply unchanged.
"""
from __future__ import annotations

import os

from ._framework import TestContext, test


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    """Rewrite model whose generate() returns a fixed revised reply."""

    def __init__(self, revised: str = "Here's how you can sort this out yourself right now: open Settings > Account.") -> None:
        self._revised = revised
        self.calls: list[dict] = []

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _FakeResp(self._revised)


class _RaisingModel:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append(1)
        raise RuntimeError("model unavailable")


class _FakeAgent:
    def __init__(self, model) -> None:  # noqa: ANN001
        self.model = model


def _guard_on(on: bool) -> None:
    if on:
        os.environ["OPENAGENT_REPLY_GUARD_ENABLED"] = "1"
    else:
        os.environ.pop("OPENAGENT_REPLY_GUARD_ENABLED", None)


class _Trace:
    """Context-manager that patches tool_trace visibility + peek and restores."""

    def __init__(self, *, enabled: bool, rows) -> None:  # noqa: ANN001
        self._enabled = enabled
        self._rows = rows

    def __enter__(self):
        from src.core import tool_trace
        self._mod = tool_trace
        self._oe = tool_trace._enabled
        self._op = tool_trace.peek
        tool_trace._enabled = lambda: self._enabled
        tool_trace.peek = lambda sid: self._rows
        return self

    def __exit__(self, *exc):
        self._mod._enabled = self._oe
        self._mod.peek = self._op
        _guard_on(False)
        return False


# ── promise detection ───────────────────────────────────────────────────


@test("reply_guard", "promises_followup matches real handoff-promise phrasings")
async def t_promise_positive(ctx: TestContext) -> None:
    from src.core.reply_guard import promises_followup

    positives = [
        "Thanks! A teammate will personally verify your case and update you.",
        "No worries — a team member will follow up with you shortly.",
        "I've flagged this to the partnerships team for you.",
        "Our team will look into it and get back to you soon.",
        "A teammate has your case flagged and will take a look.",
        "Un collega ti ricontatterà al più presto.",
        "Il team verificherà la situazione e ti ricontatterà.",
        "Ho girato la tua richiesta al team dedicato.",
    ]
    for s in positives:
        assert promises_followup(s), f"should match: {s!r}"


@test("reply_guard", "promises_followup ignores benign / self-serve replies")
async def t_promise_negative(ctx: TestContext) -> None:
    from src.core.reply_guard import promises_followup

    negatives = [
        "You can re-enable Premium by updating to version 5.0.18 in the App Store.",
        "Here's how to fix it: open Settings > Account > Restore Purchases.",
        "Thanks for reaching out! Could you share your account email?",
        "I've refunded the duplicate charge; it should appear in 5-10 days.",
        "Grazie per la segnalazione, puoi riavviare l'app e riprovare?",
        "That feature lives under the equalizer tab on the player screen.",
    ]
    for s in negatives:
        assert not promises_followup(s), f"should NOT match: {s!r}"


# ── guard behaviour ─────────────────────────────────────────────────────


@test("reply_guard", "guard is a no-op when disabled")
async def t_guard_disabled(ctx: TestContext) -> None:
    from src.core import reply_guard

    _guard_on(False)
    model = _FakeModel()
    reply = "Thanks — a teammate will follow up with you shortly."
    out = await reply_guard.guard_reply(_FakeAgent(model), "sid", "help", reply)
    assert out == reply
    assert model.calls == [], "no regeneration when disabled"


@test("reply_guard", "guard rewrites an unbacked human-follow-up promise")
async def t_guard_rewrites_unbacked(ctx: TestContext) -> None:
    from src.core import reply_guard

    _guard_on(True)
    model = _FakeModel(revised="You can restore Premium yourself by updating to 5.0.18.")
    # visibility ON, and NO tools ran this turn (peek → None)
    with _Trace(enabled=True, rows=None):
        reply = "Thanks — a teammate will personally verify your case and update you."
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "I can't log in", reply,
        )
    assert out == "You can restore Premium yourself by updating to 5.0.18."
    assert len(model.calls) == 1, "regenerated exactly once"
    assert not reply_guard.promises_followup(out), "rewrite dropped the promise"
    # the rewrite call must not pollute the session
    assert model.calls[0]["kwargs"].get("session_id") is None


@test("reply_guard", "guard keeps the reply when the promise IS backed by a handoff tool")
async def t_guard_backed_promise(ctx: TestContext) -> None:
    from src.core import reply_guard

    _guard_on(True)
    model = _FakeModel()
    with _Trace(enabled=True, rows=[("mark_as_human", "queued"), ("replio_thread_brief", "…")]):
        reply = "Thanks — a team member will follow up with you shortly."
        out = await reply_guard.guard_reply(_FakeAgent(model), "sid", "billing", reply)
    assert out == reply, "a promise backed by a real handoff this turn is fine"
    assert model.calls == [], "no regeneration for a backed promise"


@test("reply_guard", "guard no-ops without tool visibility (cannot judge grounding)")
async def t_guard_no_visibility(ctx: TestContext) -> None:
    from src.core import reply_guard

    _guard_on(True)
    model = _FakeModel()
    with _Trace(enabled=False, rows=None):
        reply = "A teammate will follow up with you."
        out = await reply_guard.guard_reply(_FakeAgent(model), "sid", "x", reply)
    assert out == reply, "no tool-trace → fail-open, leave the reply alone"
    assert model.calls == []


@test("reply_guard", "guard is a no-op when the reply makes no promise")
async def t_guard_no_promise(ctx: TestContext) -> None:
    from src.core import reply_guard

    _guard_on(True)
    model = _FakeModel()
    with _Trace(enabled=True, rows=None):
        reply = "Update to 5.0.18 in the App Store to restore Premium."
        out = await reply_guard.guard_reply(_FakeAgent(model), "sid", "x", reply)
    assert out == reply
    assert model.calls == [], "no model call when there is no promise to fix"


@test("reply_guard", "guard fails open when regeneration raises")
async def t_guard_regen_raises(ctx: TestContext) -> None:
    from src.core import reply_guard

    _guard_on(True)
    model = _RaisingModel()
    with _Trace(enabled=True, rows=None):
        reply = "A teammate will get back to you soon."
        out = await reply_guard.guard_reply(_FakeAgent(model), "sid", "x", reply)
    assert out == reply, "regeneration error → keep the original"
    assert len(model.calls) == 1, "it tried once"


@test("reply_guard", "guard keeps the original when the rewrite still promises follow-up")
async def t_guard_regen_still_promises(ctx: TestContext) -> None:
    from src.core import reply_guard

    _guard_on(True)
    model = _FakeModel(revised="Our team will get back to you.")  # still a promise
    with _Trace(enabled=True, rows=None):
        reply = "A teammate will follow up with you."
        out = await reply_guard.guard_reply(_FakeAgent(model), "sid", "x", reply)
    assert out == reply, "never accept a rewrite that still fabricates a promise"


@test("reply_guard", "guard keeps original when the rewrite is empty")
async def t_guard_regen_empty(ctx: TestContext) -> None:
    from src.core import reply_guard

    _guard_on(True)
    model = _FakeModel(revised="   ")  # empty after strip
    with _Trace(enabled=True, rows=None):
        reply = "A teammate will follow up with you."
        out = await reply_guard.guard_reply(_FakeAgent(model), "sid", "x", reply)
    assert out == reply, "empty rewrite → keep the original"


@test("reply_guard", "backing-tool set is configurable via env")
async def t_backing_tools_env(ctx: TestContext) -> None:
    from src.core import reply_guard

    _guard_on(True)
    os.environ["OPENAGENT_REPLY_GUARD_BACKING_TOOLS"] = "my_custom_escalate,zzz"
    model = _FakeModel()
    try:
        with _Trace(enabled=True, rows=[("my_custom_escalate", "done")]):
            reply = "A teammate will follow up with you."
            out = await reply_guard.guard_reply(_FakeAgent(model), "sid", "x", reply)
        assert out == reply, "custom backing tool recognised → promise is backed"
        assert model.calls == []
    finally:
        os.environ.pop("OPENAGENT_REPLY_GUARD_BACKING_TOOLS", None)
        _guard_on(False)


@test("reply_guard", "enabled() is off by default")
async def t_enabled_default_off(ctx: TestContext) -> None:
    from src.core import reply_guard

    os.environ.pop("OPENAGENT_REPLY_GUARD_ENABLED", None)
    assert reply_guard.enabled() is False
