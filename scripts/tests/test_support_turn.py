"""Conversation evidence and delivery invariants, including failure paths."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from ._framework import TestContext, test
from .test_local_support_controller import _Doubles, _drive
from src.core import local_support_controller as controller
from src.core.support_turn import delivery_state, missing_bug_fields, read_reported_turn


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
    assert "app_version" in state.facts["question_repair"]
    assert "version" not in reply and "device" not in reply
