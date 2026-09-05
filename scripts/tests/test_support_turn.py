"""Conversation evidence and delivery invariants, including failure paths."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from ._framework import TestContext, test
from .test_local_support_controller import _Doubles, _drive
from .test_local_support_controller import _Pool, _Toolkit
from src.core import local_support_controller as controller
from src.core.support_turn import delivery_state, missing_bug_fields, read_reported_turn


@test("support_turn", "resolved latest message does not reopen an old bug questionnaire")
async def t_resolved_followup(_ctx: TestContext) -> None:
    text = "Ho installato l'ultima versione, ora la musica è partita regolarmente."
    class Model:
        async def generate(self, **kw):
            return SimpleNamespace(content=json.dumps({"kind": "resolved_confirmation", "evidence": text}))
    state = controller.SupportState(thread_id="sim", customer_message=text, intent="bug",
        thread_customer_text="La musica non parte.\n" + text)
    await controller._read_reported_turn(SimpleNamespace(model=Model()), {}, state, "unit")
    assert state.intent == "resolved_confirmation"
    # A historical resolution cannot silence a new malfunction.
    state.customer_message = "Adesso ho un altro problema: la libreria è vuota."
    state.intent = "bug"
    await controller._read_reported_turn(SimpleNamespace(model=Model()), {}, state, "unit")
    assert state.intent == "request_review"
    assert state.facts["turn_reader"] == "invalid_latest_evidence"


@test("support_turn", "iOS availability has a guidance route even without a reader override")
async def t_ios_availability_route(_ctx: TestContext) -> None:
    result = await _drive(_Doubles(), "I switched from Android to iOS. Is the app available on iOS?")
    assert result["intent"] == "ios_availability"
    assert result["outcome"] == "guidance_answer"
    assert result["decision"] == "self_help"


@test("support_turn", "iOS topic words cannot suppress a newly reported playback bug")
async def t_ios_reader_overrides_old_topic(_ctx: TestContext) -> None:
    text = "Every song is unavailable on my iPhone, including newly searched songs."
    class Model:
        async def generate(self, **kw):
            return SimpleNamespace(content=json.dumps({"kind": "bug", "evidence": text}))
    state = controller.SupportState(thread_id="sim", customer_message=text, intent="ios_availability")
    await controller._read_reported_turn(SimpleNamespace(model=Model()), {}, state, "unit")
    assert state.intent == "bug"


@test("support_turn", "a media webhook uses the explanation that arrived before its brief")
async def t_media_burst_latest_text(_ctx: TestContext) -> None:
    latest = "Now it plays a voice explaining the song instead of the song. Is this a prank?"
    doubles = _Doubles(thread={"messages": [
        {"direction": "inbound", "body_text": "[1 attachment(s): video]"},
        {"direction": "inbound", "body_text": latest},
    ]})
    result = await _drive(doubles, "[1 attachment(s): video]", payload_extra={"channel_kind": "instagram_dm"})
    assert result["facts"]["message_source"] == "latest_thread_inbound"
    assert result["facts"]["language_signal"] == latest
    assert result["outcome"] != "attachment_unreadable"


@test("support_turn", "unmarked support quote cannot turn thanks into another bug report")
async def t_unmarked_support_echo(_ctx: TestContext) -> None:
    from src.core.support_email import without_support_echo
    prior = "Folders hold playlists and albums, not individual songs. Put the songs into a playlist first, then move that playlist into the folder."
    thanks = "So many thanks!\nI will follow your instructions later.\nCheers"
    mail = thanks + "\n\n" + prior.replace(" ", " \n ")
    assert without_support_echo(mail, [prior]) == thanks
    assert without_support_echo(prior, [prior]) == ""
    # Real bottom-posted and inline replies must not disappear.
    bottom = mail + "\nI tried that, but the folder is still empty."
    assert without_support_echo(bottom, [prior]) == bottom
    inline = "You said: " + prior
    assert without_support_echo(inline, [prior]) == inline
    assert without_support_echo(mail, ["An unrelated previous support answer."]) == mail
    history = {"messages": [{"direction": "outbound", "body_text": prior},
                             {"direction": "inbound", "body_text": mail}]}
    assert "Folders hold" not in controller._customer_text(history, mail)
    long_reply = prior + " More explanation of the folder and playlist behavior." * 20
    long_mail = thanks + "\n\n" + long_reply
    long_history = {"messages": [{"direction": "outbound", "body_text": long_reply},
                                  {"direction": "inbound", "body_text": long_mail}]}
    assert controller._customer_text(long_history, long_mail).strip() == thanks
    class Model:
        async def generate(self, **kw):
            return SimpleNamespace(content=json.dumps({"kind": "acknowledgement", "evidence": thanks}))
    state = controller.SupportState(thread_id="sim", customer_message=thanks, intent="bug")
    await controller._read_reported_turn(SimpleNamespace(model=Model()), {}, state, "unit")
    assert state.intent == "acknowledgement"


@test("support_turn", "lifecycle uses the guarded close and reasoned human queue APIs")
async def t_lifecycle_reserved_fields(_ctx: TestContext) -> None:
    doubles = _Doubles()
    with patch.dict(os.environ, {"OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES": "1"}):
        state = controller.SupportState(thread_id="sim", customer_message="It works now",
            intent="resolved_confirmation", outcome="resolved_confirmation", decision="noop")
        await controller._apply_lifecycle(doubles.pool(), state, "")
        assert doubles.args_for("replio_threads_patch")[-1]["patch"] == {"status": "closed"}
        state = controller.SupportState(thread_id="sim", customer_message="Still broken",
            intent="bug", decision="human", human_reason="The prior advice failed and needs technical investigation")
        assert await controller._queue_for_human(doubles.pool(), state)
        assert doubles.args_for("replio_threads_mark_for_human")
        assert all("waiting_for_team" not in call["patch"] for call in doubles.args_for("replio_threads_patch"))
        assert all("needs-human" not in call.get("tags", []) for call in doubles.args_for("replio_threads_tags_add"))


@test("support_turn", "blocked canned ads opener is removed without changing product facts")
async def t_ads_guard_progress(_ctx: TestContext) -> None:
    doubles = _Doubles(respond_results=[
        {"sent": False, "blocked": True, "retry_now": True, "category": "repeated_opening"},
        {"sent": True},
    ])
    result = await _drive(doubles, "There are too many ads and I cannot pay for Premium", payload_extra={"product": "lyra"})
    calls = doubles.args_for("replio_threads_respond")
    assert len(calls) == 2
    assert calls[0]["body_text"].split(". ", 1)[1] == calls[1]["body_text"]
    assert result["facts"]["delivery_state"] == "sent"
@test("support_turn", "referral reads require one resolved identity and matching product receipts")
async def t_referral_identity_and_scope(_ctx: TestContext) -> None:
    users = [{"identityId": "kratos-test", "id": "internal-test"}]
    receipt = {"product": "lyra", "identityId": "kratos-test", "verified": True,
               "eligiblePendingRewards": 1, "rewardsGranted": 0}
    calls = []
    async def search_users(**kwargs): return users
    async def read(**kwargs):
        calls.append(kwargs)
        return {"content": [{"type": "text", "text": json.dumps(receipt)}]}
    pool = _Pool({"lyra-admin": _Toolkit({"lyra_admin_search_users": search_users, "lyra_admin_get_referral_status": read})})
    state = controller.SupportState(thread_id="synthetic", customer_message="My invitation is missing", account_email="synthetic@example.test", tenant=controller._TENANTS["lyra"])
    result = await controller._read_referral_status(pool,state)
    assert result["rewardsGranted"] == 0 and result["eligiblePendingRewards"] == 1
    assert calls == [{"identityId": "kratos-test"}]
    receipt["product"] = "esound"
    assert not await controller._read_referral_status(pool,state)
    users.append({"identityId": "other"})
    before=len(calls)
    assert not await controller._read_referral_status(pool,state)
    assert len(calls)==before
    users[:]=[{"id":"internal-test"}]
    assert not await controller._read_referral_status(pool,state)
    assert len(calls)==before


@test("support_turn", "typographic quote normalization keeps meaning strict")
async def t_quote_typography(_ctx: TestContext) -> None:
    source = "Non so cos’è un log. La versione è 1.4.11."
    assert read_reported_turn({"kind": "other", "evidence": "Non so cos'è un log."}, source)
    assert read_reported_turn({"kind": "bug", "evidence": "La versione è 1.4.12."}, source) is None
    assert read_reported_turn({"kind": "bug", "evidence": "The log crashes"}, source) is None


@test("support_turn", "MCP text envelopes survive JSON decoding without erasing failure")
async def t_decoded_receipt(_ctx: TestContext) -> None:
    for receipt in ({"content": [{"type": "text", "text": '{"sent":true}'}]},
                    {"content": [{"type": "text", "text": {"sent": True}}]}):
        normalized = controller._jsonable(receipt)
        assert delivery_state([{"kind": "customer_reply", "success": True, "receipt": normalized}]) == "sent"
    failed = controller._jsonable({"isError": True, "structuredContent": {"sent": True}})
    assert not controller._succeeded(failed)
    assert delivery_state([{"kind": "customer_reply", "success": True, "receipt": failed}]) == "failed"
    guard = {"blocked": True, "retry_now": True, "category": "wrong_language"}
    assert controller._retryable_reply_guard(controller._jsonable({"content": [{"type": "text", "text": json.dumps(guard)}]})) == guard
    assert controller._retryable_reply_guard({"isError": True, "structuredContent": guard}) is None


@test("support_turn", "reported facts must quote customer text, not prior support claims")
async def t_reported_quotes(_ctx: TestContext) -> None:
    source = "Toco no ícone e fecha antes de mostrar a tela. Uso a versão 1.4.11."
    result = read_reported_turn({"kind": "bug", "evidence": "fecha antes de mostrar a tela",
        "reported": {"steps": "Toco no ícone", "observed": "fecha antes de mostrar a tela",
                     "app_version": "1.4.11", "os": "Android 16", "premium": "active"}}, source)
    assert result is not None
    assert result.reported == {"steps": "Toco no ícone", "observed": "fecha antes de mostrar a tela",
                               "app_version": "1.4.11"}
    assert missing_bug_fields(["app version", "device and OS", "steps to reproduce and exact behavior"], result) == ["device and OS"]
    assert read_reported_turn({"kind": "bug", "evidence": "The update fixed it"}, source) is None
    assert read_reported_turn({"kind": "delete_account", "evidence": source}, source) is None


@test("support_turn", "latest customer evidence survives the context budget")
async def t_latest_context(_ctx: TestContext) -> None:
    first = "x" * 22000
    current = "I tap the icon and it crashes before any UI."
    result = controller._customer_text({"messages": [{"direction": "inbound", "body_text": first}]}, current)
    assert len(result) == 20000 and result.endswith(current)


@test("support_turn", "transport, blocked, draft, simulated and sent outcomes stay distinct")
async def t_delivery_states(_ctx: TestContext) -> None:
    def state(receipt, **extra):
        return delivery_state([{"kind": "customer_reply", "success": True, "receipt": receipt, **extra}])
    assert state({"ok": True}) == "unknown"
    assert state({"sent": True}) == "sent"
    assert state({"sent": False}) == "held"
    assert state({"blocked": True, "sent": False}) == "blocked"
    assert state({"sent": True, "simulated": True}) == "simulated"
    assert state({"isError": True, "content": [{"text": '{"sent":true}'}]}) == "failed"
    assert state({"content": [{"text": '{"sent":false,"blocked":true}'}]}) == "blocked"
    assert state({}, planned=True) == "planned"
    assert state({}, kind="customer_draft") == "draft"
    assert delivery_state([]) == "not_attempted"
    assert delivery_state([
        {"kind": "customer_reply", "success": False, "receipt": {"blocked": True}},
        {"kind": "customer_reply", "success": True, "receipt": {"sent": True}},
    ]) == "sent"


@test("support_turn", "an uncertain send stays open and is not blindly retried")
async def t_unknown_send(_ctx: TestContext) -> None:
    doubles = _Doubles(respond_results=[{"ok": True}])
    result = await _drive(doubles, "I cannot download songs offline")
    assert len(doubles.args_for("replio_threads_respond")) == 1
    assert result["facts"]["delivery_state"] in {"unknown", "failed"}
    assert result["facts"]["delivery_handoff_confirmed"]
    assert not any(a["patch"].get("status") == "closed" for a in doubles.args_for("replio_threads_patch"))


@test("support_turn", "reader corrects a topic collision without authorizing account mutations")
async def t_reader_route_and_scope(_ctx: TestContext) -> None:
    from src.core.tool_scope import current_tool_allowlist
    text = "Three downloaded songs disappeared from my library."
    seen = []
    class Model:
        async def generate(self, **kw):
            seen.append(current_tool_allowlist())
            return SimpleNamespace(content=json.dumps({"kind": "library_loss", "evidence": text}))
    state = controller.SupportState(thread_id="sim", customer_message=text, intent="offline")
    await controller._read_reported_turn(SimpleNamespace(model=Model()), {}, state, "unit")
    assert state.intent == "bug"
    assert controller._reported_bug_route(state)[0] == "Fix missing content in the library"
    assert seen == [frozenset()]
    state.intent = "account_delete"
    await controller._read_reported_turn(SimpleNamespace(model=Model()), {}, state, "unit")
    assert state.intent == "account_delete" and len(seen) == 1


@test("support_turn", "missing or failing reader preserves bounded behavior")
async def t_reader_failure(_ctx: TestContext) -> None:
    class Model:
        async def generate(self, **kw):
            raise TimeoutError("unavailable")
    state = controller.SupportState(thread_id="sim", customer_message="cannot download offline", intent="offline")
    await controller._read_reported_turn(SimpleNamespace(model=Model()), {}, state, "unit")
    assert state.intent == "request_review" and state.reported_turn is None
    assert state.facts["turn_reader"] == "unavailable"


@test("support_turn", "a spliced citation has one repair attempt without relaxing evidence")
async def t_reader_citation_repair(_ctx: TestContext) -> None:
    text = "Nothing is happening, I switched phones and just asked about iOS availability."
    class Model:
        calls = 0
        valid_repair = True
        async def generate(self, **kw):
            self.calls += 1
            evidence = "Nothing is happening, just asked about iOS availability."
            if self.calls == 2 and self.valid_repair:
                evidence = "just asked about iOS availability"
            return SimpleNamespace(content=json.dumps({"kind": "guidance_question", "evidence": evidence}))
    for valid in (True, False):
        model = Model()
        model.valid_repair = valid
        state = controller.SupportState(thread_id="sim", customer_message=text, intent="ios_availability")
        await controller._read_reported_turn(SimpleNamespace(model=model), {}, state, "unit")
        assert model.calls == 2
        assert state.facts["turn_reader_citation_retry"] is True
        assert state.intent == ("guidance_question" if valid else "request_review")


@test("support_turn", "startup crash never requests an impossible in-app capture")
async def t_startup_capture(_ctx: TestContext) -> None:
    state = controller.SupportState(thread_id="sim", customer_message="The app crashes before any UI on startup", intent="bug")
    await controller._maybe_enable_bug_diagnostics(None, state)
    assert state.facts["diagnostics_skipped"] == "native_startup_capture_required"
    assert state.facts["required_capability"] == "native_startup_crash_read"
    assert not state.actions


@test("support_turn", "a quote-backed answer prevents a repeated model question")
async def t_known_question_guard(_ctx: TestContext) -> None:
    text = "My Samsung S21 runs Android 15 and app 1.4.11. It crashes."
    state = controller.SupportState(thread_id="sim", customer_message=text, intent="bug",
                                    outcome="bug_needs_evidence", decision="ask_information")
    state.facts.update(language="en", missing_evidence=["steps to reproduce and exact behavior"])
    state.reported_turn = read_reported_turn({"kind": "bug", "evidence": "It crashes",
         "reported": {"app_version": "1.4.11", "device": "Samsung S21", "os": "Android 15"}}, text)
    class Model:
        async def generate(self, **kw):
            return SimpleNamespace(content=json.dumps({"language": "en", "reply": "What app version and device are you using?"}))
    reply = await controller._compose_local(SimpleNamespace(model=Model()), {}, state, "unit")
    assert state.facts["reply_source"] == "deterministic:missing_evidence"
    assert "version" not in reply and "device" not in reply


@test("support_turn", "a selected custom evidence field cannot acquire invented UI options")
async def t_custom_question_no_options(_ctx: TestContext) -> None:
    state = controller.SupportState(thread_id="sim", customer_message="The widget is blank", intent="bug",
                                    outcome="bug_needs_evidence", decision="ask_information")
    state.facts.update(language="en", missing_evidence=["widget size"])
    class Model:
        async def generate(self, **kw):
            raise AssertionError("an already selected evidence question needs no free composition")
    reply = await controller._compose_local(SimpleNamespace(model=Model()), {}, state, "unit")
    assert "widget size" in reply and all(x not in reply for x in ("small", "medium", "large"))
    assert state.facts["reply_source"] == "deterministic:missing_evidence"

@test("support_turn", "native evidence is scoped, versioned and cannot turn a cohort into an individual diagnosis")
async def t_native_evidence_scope(_ctx: TestContext) -> None:
    calls=[]
    receipt={"verified":True,"product":"lyra","packageName":"com.fixture","platform":"android","individualCustomerMatch":False,"scope":"version_cohort","evidence":{"groups":[]}}
    async def read(**kwargs):
        calls.append(kwargs)
        return {"content":[{"type":"text","text":json.dumps(receipt)}]}
    pool=_Pool({"support-evidence":_Toolkit({"support_evidence_read_native_crashes":read,"support_evidence_read_release_status":read})})
    state=controller.SupportState(thread_id="synthetic",customer_message="crashes at startup",tenant=controller._TENANTS["lyra"])
    assert not await controller._read_support_evidence(pool,state,"native_crash")
    assert not calls
    # The controller's compact known-fields packet deliberately omits package.
    state.thread_customer_text = "The app crashes.\n---\npackage: com.fixture\nplatform: android"
    state.facts["already_known_from_form"]={"platform":"Android 15","native_version":"1.4.13 (71)"}
    got=await controller._read_support_evidence(pool,state,"native_crash")
    assert got["individualCustomerMatch"] is False
    assert calls[-1]=={"packageName":"com.fixture","platform":"android","version":"1.4.13","build":"71","days":7}
    receipt["product"]="esound"
    assert not await controller._read_support_evidence(pool,state,"native_crash")
    receipt["product"]="lyra";receipt["verified"]=False
    assert not await controller._read_support_evidence(pool,state,"native_crash")
    receipt["verified"]=True
    assert await controller._read_support_evidence(pool,state,"release")
    assert state.decision != "resolved"
    assert "particular task's fix" in state.instructions[-1]

@test("support_turn", "delivery diagnostics preserve only bounded operational labels")
async def t_compact_delivery_receipt(_ctx: TestContext) -> None:
    from src.core.support_delivery_receipts import summarize
    actions=[{"kind":"customer_reply","success":False,"receipt":{"isError":True,"content":[{"type":"text","text":"HTTPException: 409: Thread changed; re-read thread_brief. private@example.test token=private-secret"}]}}]
    result=summarize(actions)
    assert result['state']=='failed' and result['http_status']==409
    assert result['error_hints']==['newer_inbound']
    assert 'private' not in json.dumps(result)
    actions.append({'kind':'customer_reply','success':True,'receipt':{'sent':True,'body_text':'the customer says error, timeout and not found'}})
    result=summarize(actions)
    assert result['state']=='sent' and result['attempts']==2
    assert 'error_hints' not in result and 'body_text' not in json.dumps(result)
