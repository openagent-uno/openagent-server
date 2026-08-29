"""Anti-fabrication reply guard — ``src.core.reply_guard``.

The guard rewrites a reply that promises human/team follow-up when NO backing
action tool ran this turn, and is fail-open everywhere else: disabled, no
promise, a promise backed by a real handoff tool, no tool visibility, or a
regeneration failure all return the reply unchanged.
"""
from __future__ import annotations

import json
import os
import re

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


@test("reply_guard", "future release detector matches F11 commitments precisely")
async def t_future_release_detector(ctx: TestContext) -> None:
    from src.core.reply_guard import promises_future_release

    positives = [
        "Both fixes will be included in the next app update.",
        "The fixes will be included in a future release.",
        "This is coming in the upcoming release.",
        "La correzione sarà inclusa nel prossimo aggiornamento.",
        "Il problema verrà risolto con il prossimo aggiornamento.",
    ]
    negatives = [
        "The fix is complete and awaiting release.",
        "Update to version 5.0.18, which is already available.",
        "I cannot promise a date or version.",
    ]
    for value in positives:
        assert promises_future_release(value), f"should match: {value!r}"
    for value in negatives:
        assert not promises_future_release(value), f"should not match: {value!r}"


@test("reply_guard", "lean local event rewrites a next-update promise")
async def t_local_future_release_rewrite(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    _guard_on(False)  # strict local profile enables the F11 net itself
    model = _FakeModel(revised="The issue is verified and currently tracked.")
    with lean_local_event_scope(True):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "When will it be fixed?",
            "Both fixes will be included in the next app update.",
        )
    assert out == "The issue is verified and currently tracked."
    assert len(model.calls) == 1


@test("reply_guard", "lean local event strips a forbidden sentence if rewrite fails")
async def t_local_future_release_fail_closed(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    _guard_on(False)
    model = _RaisingModel()
    draft = (
        "The issue has been documented. "
        "Both fixes will be included in the next app update."
    )
    with lean_local_event_scope(True):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "When?", draft,
        )
    assert out == "The issue has been documented."
    assert not reply_guard.promises_future_release(out)


@test("reply_guard", "lean dry-run removes fabricated completed actions (F9)")
async def t_local_dry_run_action_claim(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.dry_run import dry_run_scope
    from src.core.execution_profile import lean_local_event_scope

    assert reply_guard.claims_completed_action(
        "I've linked your report to the existing tracking tasks."
    )
    model = _FakeModel(revised="This report matches an issue documented in the vault.")
    with lean_local_event_scope(True), dry_run_scope(True):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "Please fix it",
            "I've linked your report to the existing tracking tasks.",
        )
    assert out == "This report matches an issue documented in the vault."
    assert not reply_guard.claims_completed_action(out)


@test("reply_guard", "lean dry-run rejects unverified account state")
async def t_local_dry_run_account_state(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.dry_run import dry_run_scope
    from src.core.execution_profile import lean_local_event_scope

    draft = (
        "```json\n"
        '{"language":"en","reply":"Your Premium subscription is active. '
        'Try restarting.","evidence_files":["receipt.md"]}'
        "\n```"
    )
    with lean_local_event_scope(True), dry_run_scope(True):
        out = await reply_guard.guard_reply(
            _FakeAgent(_RaisingModel()), "sid", "I still see ads", draft,
        )
    assert "subscription is active" not in out
    assert "account email" in out
    assert "Try restarting" not in out
    match = re.search(r"```json\s*(.*?)```", out, re.DOTALL)
    assert match
    json.loads(match.group(1))


@test("reply_guard", "lean live turn removes an unbacked completed action")
async def t_local_live_action_requires_receipt(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    model = _FakeModel(revised="I found the duplicate charge, but no refund was completed.")
    with lean_local_event_scope(True), _Trace(enabled=True, rows=[
        ("billingbear_detect_duplicate_subscriptions", '{"duplicatesFound":true}'),
    ]):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "I was charged twice",
            "I've refunded the duplicate charge.",
        )
    assert out == "I found the duplicate charge, but no refund was completed."
    assert len(model.calls) == 1


@test("reply_guard", "lean live turn keeps an action backed by a success receipt")
async def t_local_live_action_with_receipt(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    draft = "I've refunded the duplicate charge."
    model = _FakeModel()
    with lean_local_event_scope(True), _Trace(enabled=True, rows=[
        ("billingbear_refund_duplicate_subscriptions", '{"success":true,"refunded":1}'),
    ]):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "I was charged twice", draft,
        )
    assert out == draft
    assert model.calls == []


@test("reply_guard", "lean live account state requires a same-turn BillingBear read")
async def t_local_live_account_state_grounding(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    draft = "Your Premium subscription is active."
    with lean_local_event_scope(True), _Trace(enabled=True, rows=None):
        out = await reply_guard.guard_reply(
            _FakeAgent(_RaisingModel()), "sid", "I see ads", draft,
        )
    assert "subscription is active" not in out
    assert "account email" in out

    model = _FakeModel()
    with lean_local_event_scope(True), _Trace(enabled=True, rows=[
        ("billingbear_get_v1_customers_by_appUserId", '{"isPremium":true}'),
    ]):
        grounded = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "I see ads", draft,
        )
    assert grounded == draft
    assert model.calls == []


@test("reply_guard", "failed handoff tool does not back a human promise")
async def t_failed_handoff_is_not_backing(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    model = _FakeModel(revised="Please send the account email and order ID.")
    with lean_local_event_scope(True), _Trace(enabled=True, rows=[
        ("replio_threads_mark_for_human", "Error from MCP tool: HTTP 500 failed"),
    ]):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "Help", "A teammate will follow up.",
        )
    assert out == "Please send the account email and order ID."
    assert len(model.calls) == 1


@test("reply_guard", "lean event removes commercial commitments (F12)")
async def t_local_commercial_commitment(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    assert reply_guard.promises_commercial_value(
        "We can refund the purchase and give you free Premium."
    )
    model = _FakeModel(revised="A person must review any refund request.")
    with lean_local_event_scope(True):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "Refund me", "We can refund the purchase.",
        )
    assert out == "A person must review any refund request."


@test("reply_guard", "lean event rejects completed-fix status contradicted by evidence (F10)")
async def t_local_fix_status_contradiction(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    draft = (
        "This is a known issue. The correction has already been implemented "
        "and is awaiting release."
    )
    assert reply_guard.claims_completed_fix(draft)
    evidence = (
        "{'summary':'Root-cause analysis of wrong-audio-on-download. "
        "DRY RUN only - no code modified.', 'status':'analysis'}"
    )
    assert reply_guard._trace_contradicts_completed_fix([("vault_read_note", evidence)])
    model = _FakeModel(revised="This is a known issue under investigation.")
    with lean_local_event_scope(True), _Trace(
        enabled=True, rows=[("tool_search_call_tool", evidence)],
    ):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "Please fix it", draft,
        )
    assert out == "This is a known issue under investigation."
    assert not reply_guard.claims_completed_fix(out)


@test("reply_guard", "lean rewrite cannot introduce a different policy violation")
async def t_local_rewrite_cross_violation(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    draft = "The correction has already been implemented. The issue is tracked."
    evidence = "DRY RUN only - no code modified; status: analysis"
    model = _FakeModel(revised="A fix will be included in the next update.")
    with lean_local_event_scope(True), _Trace(
        enabled=True, rows=[("tool_search_call_tool", evidence)],
    ):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "Please fix it", draft,
        )
    assert not reply_guard.claims_completed_fix(out)
    assert not reply_guard.promises_future_release(out)
    assert out == (
        "The issue is tracked. Available evidence confirms tracking only; no "
        "remediation state or release date is established."
    )


@test("reply_guard", "strict fallback preserves valid fenced JSON")
async def t_local_stripper_preserves_json(ctx: TestContext) -> None:
    from src.core import reply_guard

    draft = (
        "This is a known issue. The correction has already been implemented "
        "and is awaiting release.\n\n"
        "```json\n"
        '{"language":"en","reply":"Known issue. The fix is implemented '
        'and awaiting release.","evidence_files":["bug.md"]}'
        "\n```"
    )
    out = reply_guard._strip_forbidden_sentences(
        draft, explain_unverified_fix=True,
    )
    assert not reply_guard.claims_completed_fix(out)
    match = re.search(r"```json\s*(.*?)```", out, re.DOTALL)
    assert match, out
    payload = json.loads(match.group(1))
    assert payload["language"] == "en"
    assert payload["evidence_files"] == ["bug.md"]
    assert payload["reply"] == (
        "Known issue. Available evidence confirms tracking only; no remediation "
        "state or release date is established."
    )


@test("reply_guard", "lean event never promotes a historical receipt to current fix status")
async def t_local_receipt_fix_status(ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    draft = (
        "```json\n"
        '{"language":"it","reply":"Problema noto. La correzione è già stata '
        'implementata e completata.","evidence_files":["old.md"]}'
        "\n```"
    )
    rows = [("tool_search_call_tool", "path: esound/receipts/old-task.md status: complete")]
    model = _RaisingModel()
    with lean_local_event_scope(True), _Trace(enabled=True, rows=rows):
        out = await reply_guard.guard_reply(
            _FakeAgent(model), "sid", "Succede ancora oggi", draft,
        )
    assert not reply_guard.claims_completed_fix(out)
    assert model.calls == [], "receipt guard is deterministic and does not regenerate"
    match = re.search(r"```json\s*(.*?)```", out, re.DOTALL)
    assert match
    payload = json.loads(match.group(1))
    assert "non è verificato" in payload["reply"]


@test("reply_guard", "tool trace capture is automatically enabled for lean events")
async def t_local_trace_enabled(ctx: TestContext) -> None:
    from src.core import tool_trace
    from src.core.execution_profile import lean_local_event_scope

    os.environ.pop("OPENAGENT_QUALITY_MONITOR_ENABLED", None)
    assert not tool_trace._enabled()
    with lean_local_event_scope(True):
        assert tool_trace._enabled()


@test("reply_guard", "tool trace records nested tool-search args with secrets redacted")
async def t_local_trace_nested_tool_args(ctx: TestContext) -> None:
    from src.core import tool_trace
    from src.core.execution_profile import lean_local_event_scope

    with lean_local_event_scope(True):
        sink, token = tool_trace.maybe_open()
        try:
            tool_trace.record_execution({
                "tool_name": "tool_search_call_tool",
                "tool_args": {
                    "server": "billingbear",
                    "tool": "billingbear_get_v1_customers_by_appUserId",
                    "args": {"appUserId": "abc123", "email": "person@example.com"},
                    "api_key": "must-not-leak",
                },
                "result": '{"isPremium":true}',
            })
        finally:
            tool_trace.close(token)
        tool_trace.publish("nested-sid", sink)
        rows = tool_trace.peek("nested-sid") or []
        tool_trace.take("nested-sid")

    assert len(rows) == 1
    _name, excerpt = rows[0]
    assert "billingbear_get_v1_customers_by_appUserId" in excerpt
    assert "abc123" in excerpt
    assert "person@example.com" not in excerpt
    assert "must-not-leak" not in excerpt
    assert excerpt.count("[redacted]") >= 2


@test("reply_guard", "an invented task id or amount never survives the guard")
async def t_ungrounded_identifiers(_ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    rows = [("tool_search_call_tool", 'result={"id": "86-local-created-esound"}')]

    # The two fabrications a local model actually produced in a dry run.
    assert reply_guard.unbacked_task_ids(
        "We're tracking it under ClickUp task #86cavv98q.", rows,
    ) == ["86cavv98q"]
    assert reply_guard.unbacked_money(
        "The refund of €9.99 will be issued.", rows,
    ) == ["€9.99"]
    # Tenant-agnostic on purpose: the same agent will serve Lyra, so the rule
    # keys on the context word, not on one workspace's id shape.
    assert reply_guard.unbacked_identifiers(
        "Lyra ticket LYR-4471 is tracking this.", rows,
    ) == ["LYR-4471"]
    assert reply_guard.unbacked_identifiers(
        "Il tuo ordine 12345-ABC non risulta.", rows,
    ) == ["12345-ABC"]
    # Parentheses and quotes separate a keyword from its id as often as a
    # space does; this exact sentence slipped through the first version.
    assert reply_guard.unbacked_identifiers(
        "A ClickUp task (86cb3fy30) is open and assigned to the team.", rows,
    ) == ["86cb3fy30"]
    assert reply_guard.claims_issue_tracked(
        "The issue is already known and being tracked."
    ) is True

    # And it must not cry wolf on ordinary prose, or it gets switched off.
    for innocuous in (
        "On iPhone16 with iOS 19 the app crashes.",
        "Update to version 5.0.18 and reopen the app.",
        "Please send the receipt and the order ID.",
        "Sign in with the same email used for the purchase.",
        "Sign in with the same email used for the purchase (the paying one).",
    ):
        assert reply_guard.unbacked_identifiers(innocuous, rows) == [], innocuous

    # An identifier the tools really returned is not a fabrication.
    assert reply_guard.unbacked_task_ids(
        "Linked to task 86-local-created-esound.", rows,
    ) == []
    assert reply_guard.unbacked_money(
        "Your last payment of 4.99 is on the receipt.",
        [("t", 'result={"lastPaymentAmount": 4.99}')],
    ) == []
    # Digits alone must not ground an amount: matching "999" inside a
    # timestamp would let an invented figure through.
    assert reply_guard.unbacked_money(
        "The refund of €9.99 will be issued.",
        [("t", 'result={"created_at": "1999-01-01", "n": 999}')],
    ) == ["€9.99"]

    # Passive and timeline promises count as commercial commitments.
    assert reply_guard.promises_commercial_value(
        "The refund will be issued within 5-10 business days."
    ) is True
    assert reply_guard.promises_commercial_value(
        "Il rimborso sarà elaborato entro 5 giorni lavorativi."
    ) is True
    assert reply_guard.promises_commercial_value(
        "Please send the receipt and the order ID."
    ) is False

    # End to end: the sentence carrying the invented id is removed.
    class _Agent:
        pass

    with lean_local_event_scope(True):
        cleaned = await reply_guard.guard_reply(
            _Agent(), None,
            "my app crashes",
            "Thanks for the report. We're tracking it under ClickUp task #86cavv98q.",
        )
    assert "86cavv98q" not in cleaned, cleaned


@test("reply_guard", "a tracking claim needs a task receipt, id or no id")
async def t_unbacked_tracking(_ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.execution_profile import lean_local_event_scope

    # Stripping the invented id alone was not enough: the model then makes the
    # same promise without one.
    for claim in (
        "A task already exists for this type of crash.",
        "We're tracking it internally.",
        "Your report is being tracked.",
        "Una segnalazione e' gia' aperta.",
    ):
        assert reply_guard.claims_issue_tracked(claim) is True, claim
    # "known issue" is deliberately out of scope: a vault analysis note can
    # ground it honestly, and firing on those would get the guard turned off.
    for fine in (
        "Please send the app version and the steps to reproduce.",
        "This is a known issue under investigation.",
    ):
        assert reply_guard.claims_issue_tracked(fine) is False, fine

    empty = [("clickup_get_workspace_tasks", 'result={"tasks": []}')]
    found = [("clickup_get_workspace_tasks", 'result={"tasks": [{"id": "86-real"}]}')]
    # An empty search proves the opposite of tracking.
    assert reply_guard._trace_supports_tracking(empty) is False
    assert reply_guard._trace_supports_tracking(found) is True

    class _Agent:
        pass

    with lean_local_event_scope(True):
        cleaned = await reply_guard.guard_reply(
            _Agent(), None, "the app crashes",
            "Thanks for the report. A task already exists for it. "
            "Please send your app version.",
        )
    assert "task already exists" not in cleaned.lower(), cleaned
    assert "app version" in cleaned.lower(), cleaned


@test("reply_guard", "a claim about diagnostic logs needs a receipt behind it")
async def t_diagnostics_claim_detected(_ctx: TestContext) -> None:
    """The three sentences a real Lyra thread received on 26-ago-2026.

    No capture had ever been switched on and no log existed; the customer
    performed the reproduction ritual twice for nothing.
    """
    from src.core import reply_guard

    for claimed in (
        "I've enabled diagnostic logging on your account.",
        "We've received your logs and created a tracking task.",
        "From your logs, we can see the technical requests look correct.",
        "Your logs show the requests are fine.",
        "Diagnostics have been enabled for your account.",
        "Ho attivato i log diagnostici sul tuo account.",
        "Abbiamo ricevuto i tuoi log.",
        "Dai tuoi log risulta che le richieste sono corrette.",
    ):
        assert reply_guard.claims_diagnostics(claimed) is True, claimed

    for fine in (
        "",
        "Could you send a screen recording of the error?",
        "I read your message, thanks for the detail.",
        "We are tracking this issue.",
        "Riproduci il problema una volta e rispondi qui.",
    ):
        assert reply_guard.claims_diagnostics(fine) is False, fine

    # An enable receipt is what makes the sentence sayable.
    backed = [(
        "lyra_admin_enable_diagnostics",
        'result={"ok": true, "categories": ["playback"]}',
    )]
    assert reply_guard._trace_supports_completed_action(
        "I enabled diagnostic capture on your account.", backed,
    ) is True
    failed = [(
        "lyra_admin_enable_diagnostics", 'result={"ok": false, "error": "not found"}',
    )]
    assert reply_guard._trace_supports_completed_action(
        "I enabled diagnostic capture on your account.", failed,
    ) is False
