"""Pure-unit tests for the deterministic eSound local support controller."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from ._framework import TestContext, test


@test("local_support_controller", "support model calls share one bounded lane")
async def t_support_model_calls_are_serialized(_ctx: TestContext) -> None:
    import asyncio
    from types import SimpleNamespace

    from src.core import local_support_controller as controller

    class SlowModel:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def generate(self, **_kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.02)
                return SimpleNamespace(content='{"reply":"ok"}')
            finally:
                self.active -= 1

    model = SlowModel()

    async def one(index: int):
        return await controller._generate_support_model(
            model,
            messages=[{"role": "user", "content": str(index)}],
            system="test",
            session_id=f"test:{index}",
            timeout_env="OPENAGENT_TEST_SUPPORT_MODEL_TIMEOUT",
            default_timeout="1",
        )

    await asyncio.gather(*(one(index) for index in range(4)))
    assert model.max_active == controller._SUPPORT_MODEL_CONCURRENCY == 1, (
        model.max_active,
        controller._SUPPORT_MODEL_CONCURRENCY,
    )


@test("local_support_controller", "real corpus preserves and filters product labels")
async def t_real_corpus_product_filter(_ctx: TestContext) -> None:
    from scripts.local_support_operational_dryrun import _cases_from_corpus

    corpus = [
        {
            "product": "esound", "channel_kind": "reddit",
            "messages": [{"direction": "inbound", "body_text": "eSound issue long enough"}],
        },
        {
            "product": "lyra", "channel_kind": "reddit",
            "messages": [{"direction": "inbound", "body_text": "Lyra issue long enough"}],
        },
        {
            "product": "lyra", "channel_kind": "email_imap",
            "messages": [{"direction": "inbound", "body_text": "Lyra email long enough"}],
        },
    ]

    esound = _cases_from_corpus(
        corpus, sample=20, seed=1, product="esound", channel="reddit",
    )
    lyra = _cases_from_corpus(
        corpus, sample=20, seed=1, product="lyra", channel="reddit",
    )

    assert len(esound) == 1 and esound[0].product == "esound", esound
    assert len(lyra) == 1 and lyra[0].product == "lyra", lyra
    assert esound[0].channel == lyra[0].channel == "reddit"


class _Toolkit:
    def __init__(self, functions: dict[str, Any]) -> None:
        self.functions = {
            name: SimpleNamespace(entrypoint=fn, parameters={
                "type": "object", "properties": {},
            })
            for name, fn in functions.items()
        }
        self.async_functions: dict[str, Any] = {}


class _Pool:
    def __init__(self, toolkits: dict[str, _Toolkit]) -> None:
        self._toolkit_by_name = toolkits

    def toolkit_by_name(self, name: str) -> Any:
        return self._toolkit_by_name.get(name)


class _Model:
    def __init__(self) -> None:
        self.saw_empty_tools = False
        self.saw_strict_local = False

    async def generate(self, **_kwargs: Any) -> Any:
        from src.core.execution_profile import strict_local_only_active
        from src.core.tool_scope import current_tool_allowlist

        self.saw_empty_tools = current_tool_allowlist() == frozenset()
        self.saw_strict_local = strict_local_only_active()
        return SimpleNamespace(content=json.dumps({
            "language": "en",
            "reply": "Premium is active. Sign in with the purchase email, then close and reopen the app.",
        }))


@test("local_support_controller", "offline beats generic bug routing")
async def t_offline_route(_ctx: TestContext) -> None:
    from src.core.local_support_controller import (
        _extract_app_user_id,
        _intent,
        _version_at_least,
    )

    assert _intent("The download button is not working offline") == "offline"
    assert _intent("The player crashes whenever I tap Play") == "bug"
    assert _intent(
        "Ho importato una playlist e la riproduzione si è bloccata al terzo brano"
    ) == "bug"
    assert _intent("I see two charges; refund the duplicate") == "duplicate_charge"
    assert _intent("I confirm cancellation of subscription sub-1") == "cancel_subscription"
    assert _intent("Thanks, it works now!") == "resolved_confirmation"
    assert _intent("Please add a playlist mixer") == "feature_request"
    assert _intent("Please delete my account") == "account_delete"
    assert _intent("I want a refund for my latest Paddle payment") == "refund"
    assert _intent("I forgot my password; reset it") == "account_change"
    assert _intent("I opened a card chargeback") == "billing_dispute"
    assert _intent("We have a business partnership proposal") == "business_request"
    assert _extract_app_user_id({}, "No app user ID was provided") == ""
    assert _extract_app_user_id({}, "appUserId: test-active") == "test-active"
    assert _version_at_least("5.0.18", "5.0.18") is True
    assert _version_at_least("5.0.20-beta", "5.0.18") is True
    assert _version_at_least("5.0.9", "5.0.18") is False
    assert _version_at_least("unknown", "5.0.18") is False


@test("local_support_controller", "controller is opt-in and scoped to Replio event")
async def t_controller_gate(_ctx: TestContext) -> None:
    from src.core.local_support_controller import enabled

    old = os.environ.get("OPENAGENT_ESOUND_SUPPORT_CONTROLLER")
    try:
        os.environ.pop("OPENAGENT_ESOUND_SUPPORT_CONTROLLER", None)
        assert enabled({"slug": "replio-thread"}) is False
        os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER"] = "1"
        assert enabled({"slug": "replio-thread"}) is True
        assert enabled({"slug": "some-other-event"}) is False
        os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER"] = "shadow"
        assert enabled({"slug": "replio-thread"}) is True
        assert enabled({"slug": "some-other-event"}) is False
    finally:
        if old is None:
            os.environ.pop("OPENAGENT_ESOUND_SUPPORT_CONTROLLER", None)
        else:
            os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER"] = old


@test(
    "local_support_controller",
    "explicit Replio controller is not bypassed by a cloud-family composer",
)
async def t_controller_outranks_lean_profile(ctx: TestContext) -> None:
    """The controller owns support even when the composer is cloud-family."""
    from src.core import local_support_controller as controller
    from src.core.event_dispatcher import _dispatch_prompt
    from src.memory.db import MemoryDB

    class _Agent:
        name = "support-test"
        model = None

    called: list[str] = []
    original_run = controller.run
    previous = os.environ.get("OPENAGENT_ESOUND_SUPPORT_CONTROLLER")

    async def fake_controller_run(**kwargs):
        called.append(str(kwargs["event"].get("model") or ""))
        return controller.ControllerResult(
            session_id=kwargs["session_id"], text='{"controller":"test"}',
        )

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    controller.run = fake_controller_run
    os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER"] = "execute"
    try:
        result = await _dispatch_prompt(
            agent=_Agent(), db=db,
            event={
                "id": "ev-replio", "name": "Replio support",
                "slug": "replio-thread", "model": "local:claude-haiku-4-5",
                "prompt_template": "support",
            },
            payload={"thread_id": "thread-test"},
            delivery_id="delivery-test", source="webhook",
        )
        assert result["status"] == "success", result
        assert called == ["local:claude-haiku-4-5"], called
        assert '"controller":"test"' in result["output"], result
    finally:
        controller.run = original_run
        if previous is None:
            os.environ.pop("OPENAGENT_ESOUND_SUPPORT_CONTROLLER", None)
        else:
            os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER"] = previous
        await db.close()


@test("local_support_controller", "active Premium uses BillingBear and deterministic policy wording")
async def t_active_premium(_ctx: TestContext) -> None:
    from src.core.local_support_controller import run

    calls: list[str] = []

    async def threads_get(thread_id: str) -> dict[str, Any]:
        calls.append("replio_get")
        return {"ok": True, "id": thread_id, "messages": []}

    async def vault_read_note(path: str) -> dict[str, Any]:
        calls.append("vault:" + path)
        return {"ok": True, "content": "canonical router"}

    async def customer(appUserId: str) -> dict[str, Any]:
        calls.append("billing:" + appUserId)
        return {
            "ok": True,
            "status": 200,
            "isPremium": True,
            "store": "paddle",
            "clientVersion": "5.0.18",
            "subscriptions": [{"id": "sub-1", "status": "active"}],
            "entitlements": [{"id": "premium", "status": "active"}],
        }

    pool = _Pool({
        "replio": _Toolkit({"replio_threads_get": threads_get}),
        "vault": _Toolkit({"vault_read_note": vault_read_note}),
        "billingbear": _Toolkit({
            "billingbear_get_v1_customers_by_appUserId": customer,
        }),
    })
    model = _Model()
    agent = SimpleNamespace(_mcp=pool, model=model)
    result = await run(
        agent=agent,
        event={"slug": "replio-thread", "model": ""},
        payload={
            "payload": {
                "thread_id": "thread-1",
                "message": {
                    "body_text": "appUserId: test-active I paid for Premium but still see ads",
                },
            },
        },
        session_id="test-session",
        delivery_id="test-delivery",
    )
    output = json.loads(result.text)
    assert output["outcome"] == "premium_active", output
    assert output["decision"] == "self_help", output
    assert output["facts"]["isPremium"] is True
    assert calls == [
        "replio_get",
        "vault:access.md",
        "vault:esound/procedures/customer-response/_routing.md",
        "billing:test-active",
    ], calls
    # Store-specific recovery is policy, so the model is intentionally not
    # called: it previously changed "reopen the app" into "reopen browser".
    assert output["facts"]["reply_source"] == "deterministic:billing_policy"
    assert model.saw_empty_tools is False
    assert not any("human" in call for call in calls)


@test("local_support_controller", "missing Premium identity asks without BillingBear or human")
async def t_missing_identity(_ctx: TestContext) -> None:
    from src.core.local_support_controller import run

    calls: list[str] = []

    async def threads_get(thread_id: str) -> dict[str, Any]:
        calls.append("replio_get")
        return {"ok": True, "id": thread_id}

    async def vault_read_note(path: str) -> dict[str, Any]:
        calls.append("vault")
        return {"ok": True, "content": "canonical router"}

    pool = _Pool({
        "replio": _Toolkit({"replio_threads_get": threads_get}),
        "vault": _Toolkit({"vault_read_note": vault_read_note}),
    })
    model = _Model()
    result = await run(
        agent=SimpleNamespace(_mcp=pool, model=model),
        event={"slug": "replio-thread", "model": ""},
        payload={
            "payload": {
                "thread_id": "thread-2",
                "message": {"body_text": "I paid for Premium but still see ads"},
            },
        },
        session_id="test-session-2",
        delivery_id="test-delivery-2",
    )
    output = json.loads(result.text)
    assert output["outcome"] == "premium_missing_identity", output
    assert output["decision"] == "ask_information", output
    assert calls == ["replio_get", "vault", "vault"], calls
    assert not any(action.get("success") for action in output["actions"]), output
    assert "email" in output["reply"].lower()


@test("local_support_controller", "an ads complaint gets free routes instead of a billing interrogation")
async def t_ads_policy_routes(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _is_ads_policy_complaint, run

    assert not _is_ads_policy_complaint(
        "appUserId test-active: I am Premium but ads are still showing"
    )
    assert not _is_ads_policy_complaint(
        "I paid for Premium but still see ads"
    )
    assert _is_ads_policy_complaint(
        "Amazing app, but ads pop up frequently while I use it."
    )
    assert _is_ads_policy_complaint(
        "L'introduction de la publicité rend l'application invivable."
    )

    calls: list[str] = []

    async def threads_get(thread_id: str) -> dict[str, Any]:
        calls.append("replio_get")
        return {"ok": True, "id": thread_id, "messages": []}

    async def vault_read_note(path: str) -> dict[str, Any]:
        calls.append("vault:" + path)
        return {"ok": True, "content": "current same-product rule"}

    pool = _Pool({
        "replio": _Toolkit({"replio_threads_get": threads_get}),
        "vault": _Toolkit({"vault_read_note": vault_read_note}),
    })

    async def one(product: str, text: str) -> dict[str, Any]:
        result = await run(
            agent=SimpleNamespace(_mcp=pool, model=_Model()),
            event={"slug": "replio-thread", "model": ""},
            payload={"payload": {
                "thread_id": "ads-" + product,
                "product": product,
                "message": {"body_text": text},
            }},
            session_id="ads-session-" + product,
            delivery_id="ads-delivery-" + product,
        )
        return json.loads(result.text)

    esound = await one(
        "esound",
        "Ci sono troppe pubblicità e non voglio pagare. Come le tolgo gratis?",
    )
    assert esound["outcome"] == "ads_policy_explained", esound
    assert esound["decision"] == "self_help", esound
    assert esound["facts"]["free_ad_routes"] == [
        "referral", "reward_video_if_customer_visible",
    ]
    assert "invita" in esound["reply"].lower(), esound["reply"]
    assert "video" in esound["reply"].lower(), esound["reply"]
    assert "ricevuta" not in esound["reply"].lower(), esound["reply"]

    lyra = await one(
        "lyra",
        "The ads are exhausting, but I cannot pay. What free options remove them?",
    )
    assert lyra["outcome"] == "ads_policy_explained", lyra
    assert lyra["facts"]["free_ad_routes"] == [
        "referral", "reward_video_if_customer_visible", "creator_if_eligible",
    ]
    low = lyra["reply"].lower()
    assert all(term in low for term in ("referral", "video", "creator", "premium")), lyra
    assert not any(call.startswith("billing:") for call in calls), calls

    implicit = await one(
        "esound",
        "Amazing app, but ads pop up frequently while I use it.",
    )
    assert implicit["outcome"] == "ads_policy_explained", implicit
    implicit_reply = implicit["reply"].lower()
    assert "friends" in implicit_reply and "video" in implicit_reply, implicit


@test("local_support_controller", "diagnostic proof is PII-free and dry runs never authorize a claim")
async def t_diagnostic_proof_envelope(_ctx: TestContext) -> None:
    from src.core.local_support_controller import (
        SupportState,
        _diagnostic_log_excerpt,
        _reply_verified_actions,
    )

    state = SupportState(thread_id="thread-1", customer_message="intermittent bug")
    state.account_email = "customer@example.com"
    state.facts["diagnostic_capture"] = {
        "category": "playback", "status": "enabled",
    }
    state.actions.append({
        "kind": "diagnostic_enable",
        "tool": "esound_admin_enable_diagnostics",
        "success": True,
        "receipt": {"ok": True, "simulated": False},
    })
    assert _reply_verified_actions(state) == [{
        "kind": "diagnostic_enable",
        "tool": "esound_admin_enable_diagnostics",
        "success": True,
        "simulated": False,
        "category": "playback",
    }]
    assert "customer@example.com" not in json.dumps(_reply_verified_actions(state))

    state.facts["simulation_only"] = True
    assert _reply_verified_actions(state)[0]["simulated"] is True

    excerpt = _diagnostic_log_excerpt({
        "content": "user=customer@example.com Authorization: Bearer-secret",
    })
    assert "customer@example.com" not in excerpt
    assert "Bearer-secret" not in excerpt
    assert "[email-redacted]" in excerpt
    assert "[credential-redacted]" in excerpt


@test("local_support_controller", "last outbound activates idempotency before vault/model")
async def t_already_answered(_ctx: TestContext) -> None:
    from src.core.local_support_controller import run

    calls: list[str] = []

    async def threads_get(thread_id: str) -> dict[str, Any]:
        calls.append("replio_get")
        return {
            "ok": True,
            "id": thread_id,
            "messages": [
                {"direction": "inbound", "sent_at": "2026-01-01T00:00:00Z"},
                {"direction": "outbound", "sent_at": "2026-01-01T00:01:00Z"},
            ],
        }

    pool = _Pool({"replio": _Toolkit({"replio_threads_get": threads_get})})
    model = _Model()
    result = await run(
        agent=SimpleNamespace(_mcp=pool, model=model),
        event={"slug": "replio-thread", "model": ""},
        payload={"payload": {
            "thread_id": "thread-3",
            "message": {"body_text": "One more message"},
        }},
        session_id="test-session-3",
        delivery_id="test-delivery-3",
    )
    output = json.loads(result.text)
    assert output["outcome"] == "already_answered", output
    assert output["reply"] == ""
    assert calls == ["replio_get"], calls
    assert model.saw_empty_tools is False


@test("local_support_controller", "ambiguous general reply cannot invent a product explanation")
async def t_general_local_composer(_ctx: TestContext) -> None:
    from src.core.local_support_controller import run

    async def threads_get(thread_id: str) -> dict[str, Any]:
        return {"ok": True, "id": thread_id}

    async def vault_read_note(path: str) -> dict[str, Any]:
        return {"ok": True, "content": path}

    pool = _Pool({
        "replio": _Toolkit({"replio_threads_get": threads_get}),
        "vault": _Toolkit({"vault_read_note": vault_read_note}),
    })
    model = _Model()
    result = await run(
        agent=SimpleNamespace(_mcp=pool, model=model),
        event={"slug": "replio-thread", "model": ""},
        payload={"payload": {
            "thread_id": "thread-general",
            "message": {"body_text": "Could you explain how this screen works?"},
        }},
        session_id="test-general",
        delivery_id="test-general-delivery",
    )
    output = json.loads(result.text)
    assert output["outcome"] == "general_needs_detail", output
    assert output["facts"]["reply_source"] == "deterministic:clarification", output
    # The bounded fallback classifier may still use the model to choose the
    # fixed `general` label; reply_source proves it did not author the answer.
    assert model.saw_empty_tools is True
    assert model.saw_strict_local is True
    assert "premium is active" not in output["reply"].lower(), output
    assert "more detail" in output["reply"].lower(), output

    known_result = await run(
        agent=SimpleNamespace(_mcp=pool, model=model),
        event={"slug": "replio-thread", "model": ""},
        payload={"payload": {
            "thread_id": "thread-general-known",
            "message": {"body_text": (
                "Music\n---\napp_version: 5.2.0\ndevice: realme RMX3231\n"
                "os: Android 11\nplatform: android"
            )},
        }},
        session_id="test-general-known",
        delivery_id="test-general-known-delivery",
    )
    known_output = json.loads(known_result.text)
    known_reply = known_output["reply"].lower()
    assert known_output["outcome"] == "general_needs_detail", known_output
    assert "already have" in known_reply, known_output
    assert "tell me exactly what happens" in known_reply, known_output


class _Doubles:
    """Replio/vault/BillingBear/ClickUp doubles that record every call.

    The controller is deterministic, so a route is proven by the exact call
    sequence and the receipts it produced, not by the wording it returned.
    """

    def __init__(
        self,
        *,
        thread: dict[str, Any] | None = None,
        customer: dict[str, Any] | None = None,
        tasks: list[dict[str, Any]] | None = None,
        attachment: dict[str, Any] | None = None,
        create_id: str = "86-created",
        create_ok: bool = True,
        link_ok: bool = True,
        respond_results: list[dict[str, Any]] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._thread = thread or {}
        self._customer = customer
        self._tasks = tasks or []
        self._attachment = attachment
        self._create_id = create_id
        self._create_ok = create_ok
        self._link_ok = link_ok
        self._respond_results = list(respond_results or [])
        self._links: dict[str, str] = {}

    def _log(self, tool: str, **args: Any) -> None:
        self.calls.append((tool, args))

    @property
    def names(self) -> list[str]:
        return [name for name, _args in self.calls]

    def args_for(self, name: str) -> list[dict[str, Any]]:
        return [args for called, args in self.calls if called == name]

    def pool(self) -> Any:
        async def threads_get(thread_id: str) -> dict[str, Any]:
            self._log("replio_threads_get", thread_id=thread_id)
            payload: dict[str, Any] = {"ok": True, "id": thread_id, "messages": []}
            payload.update(self._thread)
            if thread_id in self._links:
                payload["external_task_id"] = self._links[thread_id]
            return payload

        async def vault_read_note(path: str) -> dict[str, Any]:
            self._log("vault_read_note", path=path)
            return {"ok": True, "content": path}

        async def threads_respond(
            thread_id: str,
            body_text: str,
            expected_last_inbound_message_id: str | None = None,
            reply_to_message_id: str | None = None,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            self._log(
                "replio_threads_respond",
                thread_id=thread_id,
                body_text=body_text,
                expected_last_inbound_message_id=expected_last_inbound_message_id,
                reply_to_message_id=reply_to_message_id,
            )
            if self._respond_results:
                return self._respond_results.pop(0)
            return {"ok": True, "success": True, "simulated": True}

        async def threads_tags_add(thread_id: str, tags: Any) -> dict[str, Any]:
            self._log("replio_threads_tags_add", thread_id=thread_id, tags=tags)
            return {"ok": True, "success": True, "simulated": True}

        async def threads_patch(thread_id: str, patch: dict[str, Any]) -> dict[str, Any]:
            self._log("replio_threads_patch", thread_id=thread_id, patch=patch)
            return {"ok": True, "success": True, "simulated": True}

        async def mark_for_human(thread_id: str, reason: str) -> dict[str, Any]:
            self._log("replio_threads_mark_for_human", thread_id=thread_id, reason=reason)
            return {"ok": True, "success": True, "simulated": True}

        async def link_task(
            thread_id: str, task_provider_id: str, external_task_id: str,
        ) -> dict[str, Any]:
            self._log("replio_thread_link_task", thread_id=thread_id,
                      external_task_id=external_task_id)
            if self._link_ok:
                self._links[thread_id] = external_task_id
            return {"ok": self._link_ok, "success": self._link_ok, "simulated": True}

        async def read_attachment(thread_id: str, attachment_index: int = 0) -> dict[str, Any]:
            self._log("replio_thread_read_attachment", thread_id=thread_id)
            return self._attachment or {
                "ok": True, "text": "[Attachment placeholder: no image content was included]",
            }

        async def customers(appUserId: str) -> dict[str, Any]:
            self._log("billingbear_get_v1_customers_by_appUserId", appUserId=appUserId)
            return self._customer or {"ok": False, "status": 404}

        async def workspace_tasks(
            listId: str, query: str, includeClosed: bool = True,
        ) -> dict[str, Any]:
            self._log("clickup_get_workspace_tasks", listId=listId)
            return {"ok": True, "tasks": self._tasks if listId == "901512182215" else []}

        async def create_task(
            listId: str, name: str, description: str = "",
            priority: int = 3, tags: Any = None,
        ) -> dict[str, Any]:
            self._log("clickup_create_task", listId=listId, name=name, tags=tags)
            if not self._create_ok:
                return {"ok": False, "status": 500}
            return {"ok": True, "success": True, "id": self._create_id, "simulated": True}

        async def create_comment(task_id: str, comment_text: str) -> dict[str, Any]:
            self._log("clickup_create_task_comment", task_id=task_id)
            return {"ok": True, "success": True, "simulated": True}

        return _Pool({
            "replio": _Toolkit({
                "replio_threads_get": threads_get,
                "replio_threads_respond": threads_respond,
                "replio_threads_tags_add": threads_tags_add,
                "replio_threads_patch": threads_patch,
                "replio_threads_mark_for_human": mark_for_human,
                "replio_thread_link_task": link_task,
                "replio_thread_read_attachment": read_attachment,
            }),
            "vault": _Toolkit({"vault_read_note": vault_read_note}),
            "billingbear": _Toolkit({
                "billingbear_get_v1_customers_by_appUserId": customers,
            }),
            "clickup": _Toolkit({
                "clickup_get_workspace_tasks": workspace_tasks,
                "clickup_create_task": create_task,
                "clickup_create_task_comment": create_comment,
            }),
        })


async def _drive(
    doubles: _Doubles,
    message: str,
    *,
    thread_id: str = "t-1",
    payload_extra: dict[str, Any] | None = None,
    writes: bool = True,
) -> dict[str, Any]:
    """Run one delivery through the controller and return its JSON output."""
    from src.core.local_support_controller import run

    inner: dict[str, Any] = {
        "thread_id": thread_id,
        "message": {"body_text": message},
    }
    inner.update(payload_extra or {})
    previous = os.environ.get("OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES")
    os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES"] = "1" if writes else "0"
    try:
        result = await run(
            agent=SimpleNamespace(_mcp=doubles.pool(), model=_Model()),
            event={"slug": "replio-thread", "model": ""},
            payload={"payload": inner},
            session_id=f"unit:{thread_id}",
            delivery_id=f"unit:{thread_id}:1",
        )
    finally:
        if previous is None:
            os.environ.pop("OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES", None)
        else:
            os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES"] = previous
    return json.loads(result.text)


def _succeeded_kinds(output: dict[str, Any]) -> set[str]:
    return {
        action["kind"] for action in output["actions"]
        if action.get("success")
    }


@test(
    "local_support_controller",
    "an operational reply carries Replio freshness and pins a Reddit comment",
)
async def t_reply_contract_reaches_send(_ctx: TestContext) -> None:
    doubles = _Doubles(thread={
        "channel": "reddit",
        "reply_contract": {
            "expected_last_inbound_message_id": "reddit-comment-42",
        },
    })
    output = await _drive(
        doubles,
        "Could you explain how this option works?",
        payload_extra={"channel_kind": "reddit"},
    )
    assert "customer_reply" in _succeeded_kinds(output), output
    sent = doubles.args_for("replio_threads_respond")
    assert len(sent) == 1, sent
    assert sent[0]["expected_last_inbound_message_id"] == "reddit-comment-42", sent
    assert sent[0]["reply_to_message_id"] == "reddit-comment-42", sent


@test(
    "local_support_controller",
    "a Replio reconciliation event recovers the latest inbound from the brief",
)
async def t_reconcile_recovers_customer_message(_ctx: TestContext) -> None:
    doubles = _Doubles(thread={
        "messages": [{
            "direction": "inbound",
            "body_text": "Could you explain how this option works?",
        }],
        "reply_contract": {
            "expected_last_inbound_message_id": "latest-inbound-7",
        },
    })

    # The worker's safety-net event has a thread id but no payload.message.
    output = await _drive(doubles, "", thread_id="t-reconcile")

    assert output["outcome"] == "general_needs_detail", output
    assert output["facts"]["message_source"] == "thread_brief", output
    assert "customer_reply" in _succeeded_kinds(output), output
    sent = doubles.args_for("replio_threads_respond")
    assert len(sent) == 1, sent
    assert sent[0]["expected_last_inbound_message_id"] == "latest-inbound-7", sent


@test(
    "local_support_controller",
    "a guarded reply is not success and is rewritten once in the same turn",
)
async def t_reply_guard_retries_once(_ctx: TestContext) -> None:
    doubles = _Doubles(
        thread={"reply_contract": {
            "expected_last_inbound_message_id": "latest-inbound-8",
        }},
        respond_results=[
            {
                "sent": False,
                "blocked": True,
                "retry_now": True,
                "category": "canned_opening",
                "reason": "open directly on the issue",
            },
            {"sent": True, "blocked": False},
        ],
    )

    output = await _drive(
        doubles, "Could you explain how this option works?",
        thread_id="t-guard-retry",
    )

    replies = [
        action for action in output["actions"]
        if action.get("kind") == "customer_reply"
    ]
    assert [action["success"] for action in replies] == [False, True], replies
    assert len(doubles.args_for("replio_threads_respond")) == 2
    assert output["facts"]["delivery_guard_retry"] == "canned_opening"


@test(
    "local_support_controller",
    "a reconciled store review keeps its declared reviewer language",
)
async def t_reconcile_recovers_reviewer_language(_ctx: TestContext) -> None:
    doubles = _Doubles(thread={
        "messages": [{
            "direction": "inbound",
            "body_text": (
                "The app no longer opens.\n"
                "reviewer_language: es\nstore_country: ESP"
            ),
        }],
        "reply_contract": {
            "expected_last_inbound_message_id": "review-9",
        },
    })

    output = await _drive(
        doubles, "", thread_id="t-review-language",
        payload_extra={"channel_kind": "playstore_reviews"},
    )

    assert output["facts"]["message_source"] == "thread_brief", output
    assert output["facts"]["language"] == "es", output
    assert output["facts"]["language_source"] == "reviewer_language", output


@test("local_support_controller", "resolved confirmation closes the thread without replying")
async def t_resolved_confirmation(_ctx: TestContext) -> None:
    doubles = _Doubles()
    output = await _drive(doubles, "Thanks, it works now!", thread_id="t-resolved")

    assert output["outcome"] == "resolved_confirmation", output
    assert output["decision"] == "noop", output
    assert output["reply"] == "", output
    # A thank-you must not consume a customer reply, and must not be escalated.
    assert "replio_threads_respond" not in doubles.names, doubles.names
    assert "replio_threads_mark_for_human" not in doubles.names, doubles.names
    assert doubles.args_for("replio_threads_tags_add")[0]["tags"] == ["resolved"]
    assert doubles.args_for("replio_threads_patch")[0]["patch"] == {
        "waiting_for_team": False, "status": "closed",
    }
    # "thanks, but it still fails" is not a resolution.
    from src.core.local_support_controller import _resolved_confirmation

    assert _resolved_confirmation("Thanks, but it still crashes") is False


@test("local_support_controller", "unreadable attachment asks for text and never describes the image")
async def t_attachment_placeholder(_ctx: TestContext) -> None:
    doubles = _Doubles()
    output = await _drive(
        doubles, "", thread_id="t-att",
        payload_extra={"attachments": [{"name": "screenshot.png"}]},
    )

    assert output["intent"] == "attachment_only", output
    assert output["outcome"] == "attachment_unreadable", output
    assert output["decision"] == "ask_information", output
    assert output["facts"]["attachment_readable"] is False, output
    assert "replio_thread_read_attachment" in doubles.names, doubles.names
    assert "attachment-reading-gotcha.md" in " ".join(output["policy_paths"])
    low = output["reply"].lower()
    assert "attachment" in low and "screenshot shows" not in low, output["reply"]
    assert "replio_threads_mark_for_human" not in doubles.names, doubles.names


@test("local_support_controller", "readable receipt attachment is still not billing proof")
async def t_attachment_receipt(_ctx: TestContext) -> None:
    doubles = _Doubles(attachment={
        "ok": True,
        "text": "Purchase receipt. Order ID TEST-ORDER for the Premium plan.",
    })
    output = await _drive(
        doubles, "", thread_id="t-receipt",
        payload_extra={"attachments": [{"name": "receipt.png"}]},
    )

    assert output["outcome"] == "attachment_receipt_unverified", output
    assert output["decision"] == "ask_information", output
    # A receipt image never authorises an account-state claim on its own.
    assert "billingbear_get_v1_customers_by_appUserId" not in doubles.names, doubles.names
    assert output["facts"].get("billing_verified") is None, output


@test("local_support_controller", "feature request is grounded, never tasked or escalated")
async def t_feature_request(_ctx: TestContext) -> None:
    doubles = _Doubles()
    output = await _drive(
        doubles, "Please add a way to automatically mix two playlists together.",
        thread_id="t-feature",
    )

    assert output["intent"] == "feature_request", output
    assert output["outcome"] == "feature_needs_detail", output
    assert output["decision"] == "ask_information", output
    assert not any(name.startswith("clickup_") for name in doubles.names), doubles.names
    assert "replio_threads_mark_for_human" not in doubles.names, doubles.names
    low = output["reply"].lower()
    assert "next update" not in low, output["reply"]
    assert "roadmap includes" not in low and "we will add" not in low, output["reply"]


@test("local_support_controller", "deletion needs verified sender then explicit confirmation")
async def t_account_delete_gates(_ctx: TestContext) -> None:
    # Body prose is not ownership proof: no transport-verified sender, no gate.
    unverified = _Doubles()
    first = await _drive(
        unverified, "Please delete my account, my email is someone@example.com",
        thread_id="t-del-1",
    )
    assert first["outcome"] == "account_delete_identity_required", first
    assert first["decision"] == "ask_information", first
    assert "replio_threads_mark_for_human" not in unverified.names, unverified.names

    # Verified sender, but no explicit confirmation yet.
    verified = _Doubles(thread={"author_email": "owner@example.com"})
    second = await _drive(verified, "Please delete my account", thread_id="t-del-2")
    assert second["outcome"] == "account_delete_confirmation_required", second
    assert second["decision"] == "ask_information", second
    assert "replio_threads_mark_for_human" not in verified.names, verified.names
    assert "permanent" in second["reply"].lower(), second["reply"]

    # Verified sender plus confirmation: authority is unavailable, so a human
    # takes it - and the reply must never claim the account was deleted.
    confirmed = _Doubles(thread={"author_email": "owner@example.com"})
    third = await _drive(
        confirmed, "I confirm: delete my account permanently.", thread_id="t-del-3",
    )
    assert third["outcome"] == "account_delete_execution_human", third
    assert third["decision"] == "human", third
    assert "human_handoff" in _succeeded_kinds(third), third["actions"]
    _assert_human_tags(confirmed, "account")
    assert "deleted" not in third["reply"].lower(), third["reply"]


def doubles_tags(doubles: _Doubles) -> list[str]:
    """Every tag written, across calls: the tool takes one tag per call."""
    out: list[str] = []
    for args in doubles.args_for("replio_threads_tags_add"):
        out.extend(args["tags"])
    return out


def _assert_human_tags(doubles: _Doubles, *expected: str) -> None:
    """The queue tags policy requires, without pinning their order.

    "team-decision" is mandatory alongside "needs-human": the human queue is
    filtered on it.
    """
    tags = set(doubles_tags(doubles))
    missing = {*expected, "team-decision", "needs-human"} - tags
    assert not missing, (missing, tags)


@test("local_support_controller", "formal GDPR skips the confirmation gate and goes to a human")
async def t_formal_gdpr(_ctx: TestContext) -> None:
    doubles = _Doubles()
    output = await _drive(
        doubles, "This is a formal GDPR right to erasure request.", thread_id="t-gdpr",
    )

    assert output["outcome"] == "account_delete_formal_human", output
    assert output["decision"] == "human", output
    assert "human_handoff" in _succeeded_kinds(output), output["actions"]
    # A formal legal request must not be gated behind "please confirm".
    assert "anti-fabrication.md" in " ".join(output["policy_paths"]), output["policy_paths"]
    assert "legal advice" not in output["reply"].lower(), output["reply"]


@test("local_support_controller", "password reset is a blocked capability, not an invented action")
async def t_account_change(_ctx: TestContext) -> None:
    unverified = _Doubles()
    first = await _drive(unverified, "I forgot my password, please reset it", thread_id="t-ch-1")
    assert first["outcome"] == "account_change_identity_required", first
    assert first["decision"] == "ask_information", first

    verified = _Doubles(thread={"author_email": "owner@example.com"})
    second = await _drive(verified, "Please change my account email", thread_id="t-ch-2")
    assert second["outcome"] == "account_change_human", second
    assert second["decision"] == "human", second
    assert "human_handoff" in _succeeded_kinds(second), second["actions"]
    low = second["reply"].lower()
    assert "reset" not in low or "i have" not in low, second["reply"]


@test("local_support_controller", "chargeback and business requests are human, never billing writes")
async def t_dispute_and_business(_ctx: TestContext) -> None:
    dispute = _Doubles()
    first = await _drive(dispute, "I opened a card chargeback for this charge", thread_id="t-disp")
    assert first["outcome"] == "billing_dispute_human", first
    assert first["decision"] == "human", first
    assert not any(name.startswith("billingbear_") for name in dispute.names), dispute.names
    _assert_human_tags(dispute, "billing")

    business = _Doubles()
    second = await _drive(
        business, "We have a business partnership proposal for eSound", thread_id="t-biz",
    )
    assert second["outcome"] == "business_request_human", second
    _assert_human_tags(business, "business")


@test("local_support_controller", "expired messenger window never sends and never escalates")
async def t_messenger_window(_ctx: TestContext) -> None:
    doubles = _Doubles(thread={
        "channel": "messenger",
        "last_inbound_at": "2026-01-01T00:00:00Z",
    })
    output = await _drive(doubles, "Any news about my issue?", thread_id="t-expired")

    assert output["outcome"] == "undeliverable", output
    assert output["decision"] == "noop", output
    assert output["reply"] == "", output
    # Sending into a closed 24h window would fail; the thread is only tagged.
    assert "replio_threads_respond" not in doubles.names, doubles.names
    assert "replio_threads_mark_for_human" not in doubles.names, doubles.names
    assert doubles_tags(doubles) == ["messenger-window-expired"], doubles.calls


@test("local_support_controller", "bug routing picks the owning component, not the reporting app")
async def t_bug_symptom_route(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _CLICKUP_LISTS, _bug_symptom_route

    title, list_id, tag = _bug_symptom_route(
        "every time I open theme settings on iOS 19 the app crashes"
    )
    assert (title, list_id, tag) == (
        "Fix crash in theme settings", _CLICKUP_LISTS["esound"], "esound/app",
    )
    # Search and playback live in the shared client core, not the brand app.
    _t, list_id, tag = _bug_symptom_route("the app crashes when I use search")
    assert (list_id, tag) == (_CLICKUP_LISTS["client"], "esound/client-core")
    _t, list_id, tag = _bug_symptom_route("playback freezes after a few seconds")
    assert (list_id, tag) == (_CLICKUP_LISTS["client"], "esound/client-core")
    # Login and sync are backend-owned.
    _t, list_id, tag = _bug_symptom_route("login does not work at all")
    assert (list_id, tag) == (_CLICKUP_LISTS["backend"], "esound/backend-core")
    # Unknown symptom or unknown surface fails closed.
    assert _bug_symptom_route("the colors look strange to me") is None
    assert _bug_symptom_route("something crashed somewhere") is None


@test("local_support_controller", "a new bug task is created, linked, and link-verified")
async def t_bug_create_and_link(_ctx: TestContext) -> None:
    doubles = _Doubles(create_id="86-new-esound")
    output = await _drive(
        doubles,
        "On iPhone 16 with iOS 19 and eSound 5.1.2, every time I open theme settings "
        "the app crashes immediately.",
        thread_id="t-bug-new",
    )

    assert output["outcome"] == "bug_created", output
    assert output["decision"] == "bug_new_task", output
    created = doubles.args_for("clickup_create_task")[0]
    assert created["listId"] == "901512182215", created
    assert created["name"] == "Fix crash in theme settings", created
    assert "esound/app" in created["tags"], created
    # tags_add must precede the link: tagging can clear the external task id.
    order = doubles.names
    assert order.index("replio_threads_tags_add") < order.index("replio_thread_link_task"), order
    kinds = _succeeded_kinds(output)
    assert {"task_create", "task_link", "task_link_verify"} <= kinds, output["actions"]
    # A tracked bug leaves the thread open for the fix, it does not close it.
    patches = [args["patch"] for args in doubles.args_for("replio_threads_patch")]
    assert {"waiting_for_team": False, "status": "open"} in patches, patches
    assert "86-new-esound" in output["reply"], output["reply"]
    assert "release" not in output["reply"].lower() or "can’t" in output["reply"], output["reply"]


@test("local_support_controller", "an unverifiable Replio link never becomes a tracked claim")
async def t_bug_link_verification_failure(_ctx: TestContext) -> None:
    doubles = _Doubles(create_id="86-new-esound", link_ok=False)
    output = await _drive(
        doubles,
        "On iPhone 16 with iOS 19 and eSound 5.1.2, every time I open theme settings "
        "the app crashes immediately.",
        thread_id="t-bug-link-fail",
    )

    assert output["outcome"] == "bug_link_failed_human", output
    assert output["decision"] == "human", output
    low = output["reply"].lower()
    assert "known issue" not in low and "tracking" not in low, output["reply"]


@test("local_support_controller", "evidenced but unroutable bug asks for logs, not for the version again")
async def t_bug_fails_closed(_ctx: TestContext) -> None:
    doubles = _Doubles()
    output = await _drive(
        doubles,
        "On a Pixel 9 with Android 16 and eSound 5.1.2, every time I open the app "
        "it does not work properly.",
        thread_id="t-unrouted",
    )

    assert output["outcome"] == "bug_no_grounded_match", output
    assert output["decision"] == "ask_information", output
    assert "clickup_create_task" not in doubles.names, doubles.names
    low = output["reply"].lower()
    assert "log" in low or "recording" in low, output["reply"]
    # The customer already supplied version and device; asking again is wrong.
    assert "app version" not in low, output["reply"]


@test("local_support_controller", "intent and language survive Italian, Spanish and French")
async def t_multilingual_routing(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _intent, _language_hint

    # Substring matching used to misroute these: "change" contains "hang" and
    # "downloads" contains "ads".
    assert _intent("Please change my account email") == "account_change"
    assert _intent("The downloads never finish on my Pixel") == "general"
    assert _intent("I exchanged my phone and it does not start") == "general"

    expected = {
        "Ho pagato il Premium ma vedo ancora la pubblicità": ("premium", "it"),
        "Voglio cancellare il mio account": ("account_delete", "it"),
        "Grazie, ora funziona!": ("resolved_confirmation", "it"),
        "Non riesco a scaricare le canzoni": ("offline", "it"),
        "Vorrei che aggiungeste un equalizzatore": ("feature_request", "it"),
        "Hola, quiero cancelar mi suscripción": ("cancel_subscription", "es"),
        "Je veux résilier mon abonnement": ("cancel_subscription", "fr"),
        "I paid for premium but still see ads": ("premium", "en"),
    }
    for message, (intent, language) in expected.items():
        assert _intent(message) == intent, (message, _intent(message))
        assert _language_hint(message) == language, (message, _language_hint(message))


@test("local_support_controller", "an Italian thread is answered in Italian")
async def t_italian_reply(_ctx: TestContext) -> None:
    doubles = _Doubles(customer={
        "ok": True, "status": 200, "isPremium": True, "store": "paddle",
        "clientVersion": "5.0.18",
        "subscriptions": [{"id": "sub-1", "status": "active"}],
        "entitlements": [{"id": "premium", "status": "active"}],
    })
    output = await _drive(
        doubles,
        "appUserId: test-active — ho pagato il Premium ma vedo ancora la pubblicità.",
        thread_id="t-it",
    )

    assert output["outcome"] == "premium_active", output
    assert output["language"] == "it", output
    reply = output["reply"]
    assert "Accedi" in reply, reply
    assert "Sign in" not in reply, reply


@test("local_support_controller", "an invented refund in the passive voice is rejected")
async def t_passive_fabrication_guard(_ctx: TestContext) -> None:
    from src.core import reply_guard
    from src.core.local_support_controller import _amount_is_verified, SupportState

    # The active-voice pattern never saw this shape, and it is exactly the one
    # the local model produced for a Google Play refund that never happened.
    assert reply_guard.claims_completed_action(
        "Your refund request has been processed."
    ) is True
    assert reply_guard.claims_completed_action("Il rimborso è stato elaborato.") is True
    assert reply_guard.claims_completed_action(
        "The dry run simulated opening task 86-x; no real change was made."
    ) is False

    # A composed reply may not introduce a figure at all. The amount landing in
    # the verified facts is not consent to quote it: "you are eligible for a
    # refund of $4.99" reads as a commitment the agent cannot make.
    assert reply_guard.quotes_money("The amount of $4.99 will be credited.") is True
    assert reply_guard.quotes_money("Your plan is 4,99 EUR per month.") is True
    assert reply_guard.quotes_money("Please send the receipt and order ID.") is False

    state = SupportState(thread_id="t", customer_message="refund please")
    assert _amount_is_verified("The amount of $4.99 will be credited.", state) is False


@test("local_support_controller", "an MCP server name resolves case- and separator-insensitively")
async def t_server_name_resolution(_ctx: TestContext) -> None:
    from src.mcp.pool import MCPPool, _normalized_mcp_name

    assert _normalized_mcp_name("BillingBear") == _normalized_mcp_name("billingbear")
    assert _normalized_mcp_name("computer-control") == _normalized_mcp_name("computer_control")

    pool = MCPPool.__new__(MCPPool)
    marker = object()
    pool._toolkit_by_name = {"billingbear": marker, "computer_control": object()}
    # The exact name still wins, and a miss no longer costs a whole model
    # round-trip to recover from.
    assert pool.toolkit_by_name("billingbear") is marker
    assert pool.toolkit_by_name("BillingBear") is marker
    assert pool.toolkit_by_name("Billing-Bear") is marker
    assert pool.toolkit_by_name("nope") is None


@test("local_support_controller", "a rephrased receipt may change words, never claims")
async def t_rephrase_containment(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _introduces_claim, SupportState

    state = SupportState(thread_id="t", customer_message="refund")
    base = "The dry run simulated opening task 86-new-esound; no real change was made."

    # Pure rewording is allowed.
    assert _introduces_claim(
        "This was a dry run: task 86-new-esound was only simulated, nothing changed.",
        base, state,
    ) is False
    # A different task id is a new claim.
    assert _introduces_claim(
        "The dry run opened task 86-other-id; no real change was made.", base, state,
    ) is True
    # So is a contact channel that appeared from nowhere - the exact failure the
    # free-agent run produced with an invented security address.
    assert _introduces_claim(
        "No real change was made. Write to security@esound.app for details.",
        base, state,
    ) is True
    # A figure that is not in the sentence or the verified facts.
    assert _introduces_claim("You will be refunded $4.99.", base, state) is True
    state.facts["subscriptions"] = [{"lastPaymentAmount": 4.99}]
    assert _introduces_claim("The amount 4.99 is on the receipt.", base, state) is False


@test("local_support_controller", "live-thread shapes: acknowledgements, machine mail, non-English bugs")
async def t_real_thread_shapes(_ctx: TestContext) -> None:
    from src.core.local_support_controller import (
        _intent, _is_machine_mail, _language_hint,
    )

    # Measured on 60 real Replio threads: 20 of the 25 awaiting a reply landed
    # in "general" and got a clarification question. These are the shapes that
    # were missing.
    for text in (
        "Ok!", "ok", "Blz,obrigado", "Gracias por ayudarme espero su respuesta",
        "Okay I'll be waiting to hear back from you",
        "Bonjour, merci beaucoup à vous pour le geste",
    ):
        assert _intent(text) == "acknowledgement", (text, _intent(text))
    # A question is never a bare acknowledgement.
    assert _intent("ok, but which version should I install?") != "acknowledgement"

    bugs = {
        "cuando quiero entrar a la aplicacion se sale de la misma": "es",
        "O aplicativo apos a atualizacao nao esta funcionando, fecha": "pt",
        "my app still close when i turn off wifi or data": "en",
        "Everything is just buffering, nothing is playing": "en",
    }
    for text, language in bugs.items():
        assert _intent(text) == "bug", (text, _intent(text))
        assert _language_hint(text) == language, (text, _language_hint(text))
    assert _intent("je ne suis plus capable de telecharger des musiques") == "offline"

    # Our own guidance must never read as a crash report.
    assert _intent("Please close and reopen the app to see it") == "general"

    # A bounce must never receive a customer reply.
    assert _is_machine_mail(
        "This is the mail system at host ds-mx-o.uk.spicysparks.com",
        "Undelivered Mail Returned to Sender",
    ) is True
    assert _is_machine_mail("my app crashes on open", "Bug Report") is False


@test("local_support_controller", "language: writing system decides, unknown never becomes English")
async def t_language_all_scripts(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _language_hint

    # Real threads carried every one of these and each was answered in English.
    for text, code in (
        ("プレミアムは毎年いくらで使えるんですか？", "ja"),
        ("이사운드 오류 앱에 들어가자마자 자동으로 나가집니다", "ko"),
        ("После обновления перестало открываться", "ru"),
        ("你哋個app係咪壞左？無一首歌可以聽到", "zh"),
        ("حسابي مش عارفة استرجعه", "ar"),
        ("Hallo, ich kann keine Lieder mehr runterladen", "de"),
        ("Merhaba, uygulama çalışmıyor", "tr"),
        ("O aplicativo não está funcionando", "pt"),
    ):
        assert _language_hint(text) == code, (text, _language_hint(text))

    # A Latin-script language we do not list is "und", NOT English: the
    # composer is told to mirror the customer instead of switching them.
    assert _language_hint("Xyzzy plugh frotz blorple") == "und"
    assert _language_hint("I paid for premium but still see ads") == "en"


@test("local_support_controller", "form values are read, labels are not evidence")
async def t_form_fields_are_values(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _bug_evidence_missing, _form_fields

    filled = (
        "The app closes when I open playlists\n\n---\n"
        "app_version: 5.1.1\ndevice: lge LM-X430\nos: Android 10\n"
    )
    assert _form_fields(filled)["device"] == "lge LM-X430"
    # Nothing to ask for: the form already said it.
    assert _bug_evidence_missing(filled) == []

    # The label alone used to satisfy the check, so a form full of n/a looked
    # complete and the customer was never asked for the device.
    empty = filled.replace("5.1.1", "n/a").replace("lge LM-X430", "n/a") \
                  .replace("Android 10", "n/a")
    assert _form_fields(empty) == {}
    assert set(_bug_evidence_missing(empty)) >= {"app version", "device and OS"}


@test("local_support_controller", "an image the runtime cannot see is never 'read'")
async def t_attachment_vision_honesty(_ctx: TestContext) -> None:
    doubles = _Doubles(attachment="Image has been generated and added to the response.")
    output = await _drive(
        doubles, "", thread_id="t-img",
        payload_extra={"attachments": [{"name": "screenshot.png"}]},
    )

    # Replio ships the picture as an MCP image block; a text-only local model
    # receives only that sentence. Calling it readable would license a claim
    # about an image nobody saw.
    assert output["outcome"] == "attachment_unreadable", output
    assert output["facts"]["attachment_readable"] is False, output["facts"]
    assert output["facts"]["attachment_text_only_runtime"] is True, output["facts"]
    low = output["reply"].lower()
    assert "screenshot shows" not in low and "image shows" not in low, output["reply"]


@test("local_support_controller", "a phrase split by an email line break still routes")
async def t_hard_wrapped_phrases(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _intent

    # Real body: "I would like to delete my\naccount." Matching a literal
    # space read that as a feature request ("I would like ...") and the
    # deletion route was never taken. re.escape escapes the space itself, so
    # the token to relax is "\\ ", not " ".
    assert _intent("I would like to delete my\naccount. Please respond.") == "account_delete"
    assert _intent("I want to cancel my\nsubscription") == "cancel_subscription"
    assert _intent("There is a duplicate\ncharge on my card") == "duplicate_charge"
    assert _intent("The download\nbutton is missing") == "offline"
    assert _intent("the app is not\nworking") == "bug"
    # And an ordinary feature request still is one.
    assert _intent("I would like to add a sleep timer") == "feature_request"


@test("local_support_controller", "a five-star review is never answered with a question")
async def t_praise_is_not_a_ticket(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _intent

    for text in (
        "The app is great, I can even play along with the song, it's so much fun!!"
        "\n--- app_version: 5.0.18 os: Android 15",
        "very good\n--- app_version: 5.1.1 os: Android 12",
        "Ottima app, la uso da anni",
    ):
        assert _intent(text) == "praise", (text, _intent(text))
    # Praise WITH a complaint is still a complaint.
    assert _intent("Great app but it crashes every time I open a playlist") == "bug"
    assert _intent("Muy buena pero no funciona el premium") == "premium"


@test("local_support_controller", "Portuguese failure reports route as bugs, praise does not")
async def t_portuguese_coverage(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _intent

    # All four arrived on real threads and all four landed in the generic
    # bucket, so a Brazilian customer with a crash got "tell me more".
    for text in (
        "O app não está abrindo",
        "O app não está mais funcionando quando entro, apresentando falhas",
        "Não consigo mais entrar no aplicativo, ele se desconecta sozinho",
        "Estou tentando acessar o aplicativo e dá erros contínuos",
    ):
        assert _intent(text) == "bug", (text, _intent(text))
    # A download complaint is still the offline route, and praise is praise.
    assert _intent("Não consigo baixar as músicas") == "offline"
    assert _intent("O aplicativo está funcionando muito bem, adorei") == "praise"


@test("local_support_controller", "the composer persists nothing (no session row, no history)")
async def t_stateless_composition(_ctx: TestContext) -> None:
    from src.core.execution_profile import (
        stateless_completion_active, stateless_completion_scope,
    )

    assert stateless_completion_active() is False
    with stateless_completion_scope(True):
        assert stateless_completion_active() is True
    assert stateless_completion_active() is False

    # The controller must open that scope around every model call: the compose
    # and rephrase paths are one-shot and tool-less, and persisting a session
    # per call is what produced "database is locked" at 8 concurrent
    # deliveries (p95 22.2s, 5 composes lost). Stateless: p95 1.6s, none lost.
    seen: list[bool] = []

    class _Probe(_Model):
        async def generate(self, **kwargs):  # type: ignore[override]
            seen.append(stateless_completion_active())
            return await super().generate(**kwargs)

    doubles = _Doubles()
    from src.core.local_support_controller import run

    previous = os.environ.get("OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES")
    os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES"] = "1"
    try:
        await run(
            agent=SimpleNamespace(_mcp=doubles.pool(), model=_Probe()),
            event={"slug": "replio-thread", "model": ""},
            payload={"payload": {
                "thread_id": "t-stateless",
                "message": {"body_text": "Could you explain how this screen works?"},
            }},
            session_id="s", delivery_id="d",
        )
    finally:
        if previous is None:
            os.environ.pop("OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES", None)
        else:
            os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES"] = previous

    assert seen and all(seen), seen


@test("local_support_controller", "draft mode writes for real but reaches no customer")
async def t_draft_mode(_ctx: TestContext) -> None:
    from src.core import local_support_controller as lsc

    doubles = _Doubles()

    async def threads_draft(
        thread_id: str, body_text: str, confidence: Any = None,
        reasoning: Any = None, origin: str = "ai",
    ) -> dict[str, Any]:
        doubles._log("replio_threads_draft", thread_id=thread_id,
                     body_text=body_text, origin=origin)
        return {"ok": True, "success": True, "simulated": True}

    pool = doubles.pool()
    pool._toolkit_by_name["replio"].functions["replio_threads_draft"] = SimpleNamespace(
        entrypoint=threads_draft, parameters={"type": "object", "properties": {}},
    )

    previous = os.environ.get(lsc._DRAFTS_ENV)
    os.environ[lsc._DRAFTS_ENV] = "1"
    os.environ.pop(lsc._WRITES_ENV, None)
    try:
        assert lsc.drafts_enabled() is True
        # Drafting still counts as writing: the receipts are real.
        assert lsc.writes_enabled() is True
        result = await lsc.run(
            agent=SimpleNamespace(_mcp=pool, model=_Model()),
            event={"slug": "replio-thread", "model": ""},
            payload={"payload": {
                "thread_id": "t-draft",
                "message": {"body_text": "Could you explain how this screen works?"},
            }},
            session_id="s", delivery_id="d",
        )
    finally:
        if previous is None:
            os.environ.pop(lsc._DRAFTS_ENV, None)
        else:
            os.environ[lsc._DRAFTS_ENV] = previous

    output = json.loads(result.text)
    assert output["facts"]["delivered_as"] == "draft", output["facts"]
    assert "replio_threads_draft" in doubles.names, doubles.names
    # The customer-facing send must never happen on this rung, and a draft
    # must not tag or close the thread as if it had been answered.
    assert "replio_threads_respond" not in doubles.names, doubles.names
    assert "replio_threads_patch" not in doubles.names, doubles.names


@test("local_support_controller", "a store review is composed inside the channel's cap")
async def t_channel_reply_cap(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _fit_reply, _reply_cap

    # Replio hard-trims past these, and a trim lands mid-sentence.
    # The permanent 300-character rule binds everywhere; the store ceilings
    # only matter when they are STRICTER than it.
    assert _reply_cap("playstore_reviews") == 300
    assert _reply_cap("appstore_reviews") == 300
    assert _reply_cap("email_imap") == 300

    # Real prose: the trim lands on a sentence end, not mid-word.
    prose = (
        "Thanks for the report. Please update the app to the latest version "
        "and sign in with the same account you used for the purchase. "
        "If the problem persists, send us the exact error you see on screen "
        "and the model of your device so we can reproduce it properly. "
        "We will look at it as soon as we have those details from you."
    )
    fitted = _fit_reply(prose, 300)
    assert len(fitted) <= 300, len(fitted)
    assert fitted.endswith("."), fitted
    assert " ".join(fitted.split()) in " ".join(prose.split()), fitted
    # Short text is returned untouched.
    assert _fit_reply("Short reply.", 300) == "Short reply."
    # With no boundary in the second half, cut hard rather than discard most
    # of the reply - a store cap is a hard limit, not a suggestion.
    assert len(_fit_reply("A. " + "x" * 400, 300)) == 300


@test("local_support_controller", "the draft rung arms no other write")
async def t_draft_rung_is_narrow(_ctx: TestContext) -> None:
    from src.core import local_support_controller as lsc

    doubles = _Doubles(thread={"author_email": "owner@example.com"})

    async def threads_draft(thread_id: str, body_text: str, **_kw: Any) -> dict[str, Any]:
        doubles._log("replio_threads_draft", thread_id=thread_id)
        return {"ok": True, "success": True}

    pool = doubles.pool()
    pool._toolkit_by_name["replio"].functions["replio_threads_draft"] = SimpleNamespace(
        entrypoint=threads_draft, parameters={"type": "object", "properties": {}},
    )

    previous = os.environ.get(lsc._DRAFTS_ENV)
    os.environ[lsc._DRAFTS_ENV] = "1"
    os.environ.pop(lsc._WRITES_ENV, None)
    try:
        # A confirmed deletion request routes to a human, which on a full write
        # rung would tag the thread AND call mark_for_human for real.
        result = await lsc.run(
            agent=SimpleNamespace(_mcp=pool, model=_Model()),
            event={"slug": "replio-thread", "model": ""},
            payload={"payload": {
                "thread_id": "t-rung",
                "message": {"body_text": "I confirm: delete my account permanently."},
            }},
            session_id="s", delivery_id="d",
        )
    finally:
        if previous is None:
            os.environ.pop(lsc._DRAFTS_ENV, None)
        else:
            os.environ[lsc._DRAFTS_ENV] = previous

    output = json.loads(result.text)
    assert output["decision"] == "human", output
    # Nothing was executed: no handoff, no tag, no patch.
    assert not any(a.get("success") for a in output["actions"]), output["actions"]
    assert "replio_threads_mark_for_human" not in doubles.names, doubles.names
    assert "replio_threads_tags_add" not in doubles.names, doubles.names


@test("local_support_controller", "the subject rescues a fragment, but never overturns the message")
async def t_subject_as_signal(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _intent

    fragment = "Dans l appli on ne sait pas voir son adresse mail dom@example.com"
    # Alone the fragment says nothing; with the thread's subject it is a
    # premium renewal, and that is what triggers the BillingBear lookup.
    assert _intent(fragment) == "general"
    assert _intent(
        "Bonjour, j essaye de renouveler le premium mais je n y arrive pas. " + fragment
    ) == "premium"

    # An acknowledgement with a complaint attached is a complaint.
    assert _intent("pas de souci merci bcp") == "acknowledgement"
    assert _intent("nessun problema grazie") == "acknowledgement"
    assert _intent("pas de souci mais lapp ne fonctionne plus") == "bug"
    assert _intent("L application se ferme toute seule") == "bug"
    # A crash mentioned alongside Premium is a crash, not a billing case.
    assert _intent(
        "The app stop's and automatically close some bug? Pls help me im premium"
    ) == "bug"


@test("local_support_controller", "an eSound email resolves through Paddle, not the customer lookup")
async def t_paddle_email_lookup(_ctx: TestContext) -> None:
    from src.core.local_support_controller import (
        _BILLINGBEAR_PROJECT_ID, _customer_lookup_state, _paddle_verdict, run,
    )

    # The server answers "HTTP 200 OK\n{json}" and the verdict is definitive:
    # "never purchased via Paddle" is verification, not a failure.
    raw = ('HTTP 200 OK\n{"email":"x@y.com","configured":true,"found":false,'
           '"hasActiveSubscription":false,"hasCompletedPayment":false,'
           '"interpretation":"No Paddle customer"}')
    verdict = _paddle_verdict(raw)
    assert verdict is not None and verdict["isPremium"] is False, verdict
    # Paddle looks at PADDLE. Without the account-wide field its answer is
    # scope-limited: "nothing on the web channel", never "no subscription".
    # Inferring the latter is the incident where a Play Store subscriber was
    # told to buy again.
    assert verdict["paddle_scope_only"] is True, verdict
    assert _customer_lookup_state(verdict)[0] is False
    account_wide = _paddle_verdict(
        raw.replace('"found":false', '"found":true')
           .replace('"hasActiveSubscription":false',
                    '"hasActiveSubscription":true,"accountHasActivePremium":true')
    )
    assert account_wide["paddle_scope_only"] is False, account_wide
    assert _customer_lookup_state(account_wide) == (True, "paddle", "", [])
    # A non-Paddle payload is not mistaken for one.
    assert _paddle_verdict('{"ok":true,"isPremium":true}') is None

    # End to end: an email-only Premium thread must call the Paddle resolver,
    # with the address INSIDE `query` and the project on the path. The bare
    # customer-by-email tool answers 400/404 for eSound by design.
    seen: list[dict[str, Any]] = []

    async def paddle_lookup(projectId: str, query: dict[str, Any]) -> str:
        seen.append({"projectId": projectId, "query": query})
        return raw

    doubles = _Doubles()
    pool = doubles.pool()
    pool._toolkit_by_name["billingbear"].functions[
        "billingbear_get_v1_projects_by_projectId_paddle_lookup"
    ] = SimpleNamespace(entrypoint=paddle_lookup,
                        parameters={"type": "object", "properties": {}})

    output = json.loads((await run(
        agent=SimpleNamespace(_mcp=pool, model=_Model()),
        event={"slug": "replio-thread", "model": ""},
        payload={"payload": {
            "thread_id": "t-paddle",
            "message": {"body_text":
                        "I paid for premium but still see ads, my email is a@b.com"},
        }},
        session_id="s", delivery_id="d",
    )).text)

    assert seen, "the Paddle resolver was never called"
    assert seen[0]["projectId"] == _BILLINGBEAR_PROJECT_ID, seen
    assert seen[0]["query"] == {"email": "a@b.com"}, seen
    # A Paddle-only negative must NOT count as verified account state.
    assert output["facts"]["billing_verified"] is False, output["facts"]
    assert output["outcome"] == "premium_unverified_paddle_scope", output
    low = output["reply"].lower()
    assert "no subscription" not in low and "not premium" not in low, output["reply"]


@test("local_support_controller", "the web form's 32-hex id is never used as an appUserId")
async def t_account_user_id_is_not_appuserid(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _extract_app_user_id

    # The form posts a 32-hex obfuscated id; BillingBear keys on a 24-hex
    # Mongo ObjectId. Using the former guarantees a 404, and a 404 read as
    # "no subscription" is a fabricated verdict.
    body = ("account_email: a@b.com\n"
            "account_user_id: b3350871efde159a91d1d3a54eda2af3")
    assert _extract_app_user_id({}, body) == ""
    assert _extract_app_user_id({}, "appUserId: 68459bfd0a12b30a59b599db") == (
        "68459bfd0a12b30a59b599db"
    )


@test("local_support_controller", "the recovery step follows the store that took the money")
async def t_store_family_guidance(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _store_family

    for store in ("apple", "app_store", "ios", "google", "google_play",
                  "playstore", "play_store"):
        assert _store_family(store) == "iap", store
    for store in ("paddle", "stripe", "web", "desktop"):
        assert _store_family(store) == "web", store
    # Unknown must stay unknown: guessing the recovery step is worse than
    # asking, because the two are mutually exclusive.
    for store in ("", "unknown", "carrier-billing"):
        assert _store_family(store) == "", store
    assert _store_family(" Apple ") == "iap"


@test("local_support_controller", "an in-app purchase is never told to sign in with an email")
async def t_iap_premium_guidance(_ctx: TestContext) -> None:
    from src.core.local_support_controller import run

    async def customer(appUserId: str) -> dict[str, Any]:
        return {
            "ok": True, "status": 200, "isPremium": True, "store": "apple",
            "clientVersion": "5.1.1",
            "subscriptions": [{"id": "sub-a", "status": "active"}],
            "entitlements": [{"id": "premium", "status": "active"}],
        }

    doubles = _Doubles()
    pool = doubles.pool()
    pool._toolkit_by_name["billingbear"].functions[
        "billingbear_get_v1_customers_by_appUserId"
    ] = SimpleNamespace(entrypoint=customer,
                        parameters={"type": "object", "properties": {}})

    # Store recovery is deterministic; a web-style model answer must never be
    # consulted for an in-app purchase.
    output = json.loads((await run(
        agent=SimpleNamespace(_mcp=pool, model=_Model()),
        event={"slug": "replio-thread", "model": ""},
        payload={"payload": {
            "thread_id": "t-apple",
            "message": {"body_text":
                        "appUserId: test-apple I paid for premium but still see ads"},
        }},
        session_id="s", delivery_id="d",
    )).text)

    assert output["outcome"] == "premium_active", output
    assert output["facts"]["store_family"] == "iap", output["facts"]
    assert output["facts"]["reply_source"] == "deterministic:billing_policy", output["facts"]
    low = output["reply"].lower()
    assert "restore purchases" in low, output["reply"]
    assert "purchase email" not in low, output["reply"]


@test("local_support_controller", "no phrasing of 'the refund is done' survives without a receipt")
async def t_no_unbacked_refund_claim(_ctx: TestContext) -> None:
    """Adversarial: every way a model has actually claimed a refund."""
    from src.core import reply_guard

    claims = [
        "I have refunded your subscription.",
        "We've refunded the last payment.",
        "Your refund request has been processed.",
        "The refund has been issued to your card.",
        "The refund is being processed right now.",
        "Your refund will be issued within 5-10 business days.",
        "A credit will be applied to your account shortly.",
        "Ho rimborsato l'ultimo pagamento.",
        "Il rimborso è stato elaborato.",
        "Il rimborso sarà accreditato entro 5 giorni lavorativi.",
        "Ti rimborseremo l'ultimo addebito.",
    ]
    for claim in claims:
        flagged = (
            reply_guard.claims_completed_action(claim)
            or reply_guard.promises_commercial_value(claim)
        )
        assert flagged, f"un rimborso non provato è passato: {claim!r}"

    # An amount is never introduced by the model, even a true one.
    assert reply_guard.quotes_money("Your refund of €9.99 is on its way.") is True

    # And the wording the controller itself uses must stay clean, or the guard
    # would eat its own legitimate replies.
    for safe in (
        "Request the Apple refund at reportaproblem.apple.com; Apple manages the transaction directly.",
        "To process your refund, please provide the payment date for the subscription.",
        "Let's try to fix it first. Send me device, OS and app version.",
        "This report requires specialist human review.",
    ):
        assert not reply_guard.claims_completed_action(safe), safe
        assert not reply_guard.promises_commercial_value(safe), safe
        assert not reply_guard.quotes_money(safe), safe


@test("local_support_controller", "a refund asked because the app is broken is fixed first")
async def t_refund_malfunction_rule(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _intent, _refund_for_malfunction

    # Policy rule 5: resolve before refunding - but never DROP the refund.
    for text in (
        "I want a refund, the app crashes every time",
        "Quiero un reembolso, la app no funciona",
        "Voglio il rimborso, l'app si blocca sempre",
    ):
        assert _intent(text) == "refund", (text, _intent(text))
        assert _refund_for_malfunction(text) is True, text
    # A plain change-of-mind refund goes straight to the eligibility rules.
    assert _refund_for_malfunction("I want a refund, I changed my mind") is False
    # And a crash with no money-back request stays a bug.
    assert _intent("The app crashes every time I open it") == "bug"


@test("local_support_controller", "the refund threshold reads the catalogue, not a bare number")
async def t_amount_anomaly_is_currency_aware(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _amount_is_anomalous, _store_family

    # Real catalogue from BillingBear: Premium is 14.99/yr and 1.99/mo. A bare
    # ">30" was dead code (nothing costs that) AND currency-blind: 79.99 BRL
    # is about 13 EUR, i.e. the ordinary yearly plan.
    assert _amount_is_anomalous(14.99, "USD") is False
    assert _amount_is_anomalous(79.99, "BRL") is False
    assert _amount_is_anomalous(149.99, "USD") is True
    assert _amount_is_anomalous(31.00, "EUR") is True
    # No amount, or a currency we do not sell in: let the 14-day rule decide
    # rather than invent a conversion.
    assert _amount_is_anomalous(None, "USD") is False
    assert _amount_is_anomalous(500.0, "JPY") is False

    # The premium entitlement lists amazon and huawei as providers too; both
    # are in-app purchases and must not be collapsed into "google".
    for store in ("amazon", "amazon_appstore", "huawei", "appgallery"):
        assert _store_family(store) == "iap", store


@test("local_support_controller", "a legal or investment message is answered with silence")
async def t_legal_silence(_ctx: TestContext) -> None:
    from src.core import local_support_controller as lsc

    for text in (
        "I own the rights to this song, remove my content",
        "My attorney will be in touch about this infringement",
        "We are Universal Music and require a takedown",
        "unpaid royalties for my tracks",
        "We would like to discuss an acquisition of your company",
        "I want to invest, are you raising?",
    ):
        assert lsc._requires_legal_silence(text) is True, text
    for ordinary in ("the app crashes on open", "I want a refund", "premium is missing"):
        assert lsc._requires_legal_silence(ordinary) is False, ordinary

    notified: list[dict[str, Any]] = []

    async def send_telegram(chat_id: str, text: str) -> dict[str, Any]:
        notified.append({"chat_id": chat_id, "text": text})
        return {"ok": True, "success": True}

    doubles = _Doubles()
    pool = doubles.pool()
    pool._toolkit_by_name["messaging"] = _Toolkit(
        {"messaging_send_telegram": send_telegram}
    )

    previous = os.environ.get(lsc._WRITES_ENV)
    os.environ[lsc._WRITES_ENV] = "1"
    try:
        output = json.loads((await lsc.run(
            agent=SimpleNamespace(_mcp=pool, model=_Model()),
            event={"slug": "replio-thread", "model": ""},
            payload={"payload": {
                "thread_id": "t-legal",
                "message": {"body_text": "I own the rights, take down my content."},
            }},
            session_id="s", delivery_id="d",
        )).text)
    finally:
        if previous is None:
            os.environ.pop(lsc._WRITES_ENV, None)
        else:
            os.environ[lsc._WRITES_ENV] = previous

    assert output["outcome"] == "legal_silence", output
    assert output["reply"] == "", output
    # The customer is NEVER answered on a legal matter...
    assert "replio_threads_respond" not in doubles.names, doubles.names
    # ...but the owner wants every one of these queued for a human as well.
    assert "replio_threads_mark_for_human" in doubles.names, doubles.names
    # The owner is told, with the fixed payload and no analysis.
    assert notified, "the owner was never notified"
    body = notified[0]["text"]
    for field in ("source:", "thread_id:", "subject:", "excerpt:", "trigger:"):
        assert field in body, (field, body)


@test("local_support_controller", "a Play review is answered in the reviewer's language")
async def t_reviewer_language_wins(_ctx: TestContext) -> None:
    """Google translates the review to English before we ever see it.

    Measured on the corpus: 47 of 74 store reviews declare Spanish or
    Portuguese while the text handed to us is English, so detecting the text
    answered every one of those reviewers in a language they had not used.
    """
    from src.core import local_support_controller as lsc

    body = (
        "The app is good, but it takes a long time to load.\n\n"
        "---\napp_version: 5.1.1\nos: Android 15\nreviewer_language: pt\n"
    )
    assert lsc._form_fields(body).get("reviewer_language") == "pt"
    # The text really is English - that is the whole trap.
    assert lsc._language_hint(lsc._FORM_FIELD.sub("", body)) == "en"


@test("local_support_controller", "hostility is never filed as praise")
async def t_hostility_is_not_praise(_ctx: TestContext) -> None:
    """Silence is the worst possible reply to an angry customer."""
    from src.core import local_support_controller as lsc

    for insult in ("a mega piece of garbage", "esta app es una basura",
                   "app di merda", "this is the worst app", "useless",
                   "que porcaria de aplicativo"):
        assert lsc._is_praise(insult) is False, insult
    for real in ("phenomenal", "Highly recommended", "Superb and commercial-free"):
        assert lsc._is_praise(real) is True, real


@test("local_support_controller", "a cancellation needs three things, not the word yes")
async def t_cancellation_gate(_ctx: TestContext) -> None:
    """"Can you confirm?" used to cancel a paying customer's subscription."""
    from src.core import local_support_controller as lsc

    assert lsc._is_confirmed("confermo") is True
    assert lsc._is_confirmed("yes") is True
    # A question, a refusal, or a hedge is not a confirmation.
    for not_yes in ("Can you confirm what happens if I cancel?",
                    "no, non confermo", "wait, I confirm later?",
                    "I did not confirm"):
        assert lsc._is_confirmed(not_yes) is False, not_yes

    asked = [
        {"direction": "outbound", "body_text": "Can you confirm you want to cancel?"},
    ]
    ready = {"tags": ["billing", "subcancel-pending"], "messages": asked}
    assert lsc._cancellation_phase(ready, "confermo") == "execute"
    # Each missing leg drops it back to phase 1.
    assert lsc._cancellation_phase({"messages": asked}, "confermo") == "ask"
    assert lsc._cancellation_phase(
        {"tags": ["subcancel-pending"], "messages": []}, "confermo",
    ) == "ask"
    assert lsc._cancellation_phase(ready, "can you confirm?") == "ask"


@test("local_support_controller", "an expired entitlement is not premium")
async def t_entitlement_expiry(_ctx: TestContext) -> None:
    """The entitlement is the gate; the profile field can run ahead of it."""
    from src.core import local_support_controller as lsc

    assert lsc._entitlement_active({"expiresAt": "2099-01-01T00:00:00Z"}) is True
    assert lsc._entitlement_active({"expiresAt": "2020-01-01T00:00:00Z"}) is False
    # No expiry means it does not expire.
    assert lsc._entitlement_active({"id": "premium"}) is True
    # A revoked entitlement is not active whatever its date says.
    assert lsc._entitlement_active(
        {"status": "revoked", "expiresAt": "2099-01-01T00:00:00Z"},
    ) is False

    expired = {"entitlements": [{"id": "premium", "expiresAt": "2020-01-01T00:00:00Z"}]}
    assert lsc._customer_lookup_state(expired)[0] is False
    live = {"entitlements": [{"id": "premium", "expiresAt": "2099-01-01T00:00:00Z"}]}
    assert lsc._customer_lookup_state(live)[0] is True


@test("local_support_controller", "an appUserId is 24 hex, and the form's id is not one")
async def t_app_user_id_shape(_ctx: TestContext) -> None:
    """The 32-hex account_user_id looks up a customer that does not exist."""
    from src.core import local_support_controller as lsc

    assert lsc._APP_USER_ID.match("5f8349b40a31dbebd7063a5d") is not None
    # The web form's account_user_id: 32 hex, a different identifier entirely.
    assert lsc._APP_USER_ID.match("d9e7f794c4fb4db02fc2b2e9ce69df5d") is None


@test("local_support_controller", "severity follows the symptom, not the template")
async def t_bug_severity(_ctx: TestContext) -> None:
    """Stamping 'urgent' on every task is the same as stamping none."""
    from src.core import local_support_controller as lsc

    assert lsc._bug_severity("crash", False) == "urgent"
    assert lsc._bug_severity("missing audio", False) == "high"
    assert lsc._bug_severity("slow performance", False) == "normal"
    # An angry customer moves a cosmetic defect one step, never to urgent.
    assert lsc._bug_severity("slow performance", True) == "high"
    assert lsc._bug_severity("freeze", True) == "urgent"
    # An unknown symptom is normal, not urgent.
    assert lsc._bug_severity("something new", False) == "normal"
    for name, priority in (("urgent", 1), ("high", 2), ("normal", 3)):
        assert lsc._CLICKUP_PRIORITY[name] == priority


@test("local_support_controller", "a search hit is judged before it becomes 'known issue'")
async def t_dedup_is_judged(_ctx: TestContext) -> None:
    """Saying a problem is known when it is not is worse than a duplicate."""
    from src.core import local_support_controller as lsc

    right = {"name": "Fix missing audio in the player", "listId": "L1"}
    wrong = {"name": "Fix crash in the library", "listId": "L1"}
    unrelated = {"name": "Add dark mode", "listId": "L2"}

    picked = lsc._best_task_match(
        [unrelated, wrong, right], "missing audio", "the player", "L1",
    )
    assert picked is right, picked
    # Nothing convincing must return nothing - never the first row.
    assert lsc._best_task_match(
        [unrelated, wrong], "freeze", "the embed player", "L9",
    ) is None

    # When the symptom table cannot route the report, the customer's own
    # words decide - including across a plural.
    carplay = {"name": "Fix CarPlay disconnect", "listId": "L1"}
    assert lsc._best_task_match(
        [carplay], "", "", "", "every time I connect CarPlay the app disconnects",
    ) is carplay
    # One shared word is not a match.
    assert lsc._best_task_match(
        [{"name": "Fix crash in the library", "listId": "L1"}],
        "", "", "", "the sleep timer never stops the music",
    ) is None


@test("local_support_controller", "the classifier covers what real threads actually say")
async def t_classifier_real_shapes(_ctx: TestContext) -> None:
    """Every line here is a real message from the 1430-thread Replio corpus."""
    from src.core import local_support_controller as lsc

    cases = (
        # The app will not open or play: the largest cluster that used to sit
        # in the generic bucket.
        ("No me deja abrir la aplicacion y ya intente de todo", "bug"),
        ("Hola llevo dias sin poder reproducir nada de musica", "bug"),
        ("It's not letting me search my music or add any new music", "bug"),
        ("No puedo ingresar", "bug"),
        ("Hey! I think your Server is down... it doesnt work", "bug"),
        ("mi viene segnalato canzone non trovata", "bug"),
        ("nothing is syncing from PC to phone", "bug"),
        ("Every time I open the equalizer the preset resets", "bug"),
        ("Every time I connect CarPlay the app disconnects again", "bug"),
        # A support code is the most precise routing signal we ever get.
        ("Re: Dysfonctionnements et erreur [WC014] Compte payant", "bug"),
        ("ci sono problemi con il server (wc037) cosa significa?", "bug"),
        # One fixed answer, asked constantly.
        ("Is eSound available for iOS?", "ios_availability"),
        ("Hey will the app ever be available again on App Store?", "ios_availability"),
        # Non-Latin scripts: no word boundary can ever hold there.
        ("プレミアムは毎年いくらで使えるんですか？", "premium"),
        ("会员订阅怎么取消", "premium"),
        ("подписка не работает", "premium"),
        # A request wrapped in a compliment is a request.
        ("The app is good, but it should have the option to download "
         "directly to internal memory", "feature_request"),
        ("Put the Shuffle button back on or I'll uninstall this app",
         "feature_request"),
        # Short multilingual praise.
        ("phenomenal", "praise"),
        ("Highly recommended", "praise"),
        ("Superb and commercial-free", "praise"),
        ("Best music app I have ever used, thank you", "praise"),
        # The owner wants these in front of a person, never answered.
        ("We'd love to explore a partnership with your app", "business_request"),
        ("we are interested in a sponsorship for our media kit", "business_request"),
    )
    for text, expected in cases:
        assert lsc._intent(text) == expected, (text, lsc._intent(text), expected)

    # A refund must never be read as "bring back my money" feature talk.
    assert lsc._intent("I want my money back") == "refund"

    # An acquisition approach is handled one step earlier still: it never
    # reaches _intent, because it may not be answered at all.
    assert lsc._requires_legal_silence(
        "we represent a fund and would like to discuss an acquisition"
    ) is True


@test("local_support_controller", "an unstamped thread is not read as already answered")
async def t_already_answered_ordering(_ctx: TestContext) -> None:
    """Sorting on the tuple compared 'inbound' vs 'outbound' as text."""
    from src.core import local_support_controller as lsc

    unstamped = {"messages": [
        {"direction": "outbound", "body_text": "what is your account email?"},
        {"direction": "inbound", "body_text": "lina@example.com"},
    ]}
    assert lsc._thread_already_answered(unstamped) is False
    answered = {"messages": [
        {"direction": "inbound", "body_text": "premium is gone"},
        {"direction": "outbound", "body_text": "we restored it"},
    ]}
    assert lsc._thread_already_answered(answered) is True
    # Real timestamps still decide when they are present.
    stamped = {"messages": [
        {"direction": "outbound", "sent_at": "2026-08-01T10:00:00Z"},
        {"direction": "inbound", "sent_at": "2026-08-02T10:00:00Z"},
    ]}
    assert lsc._thread_already_answered(stamped) is False


@test("local_support_controller", "a bare email is an answer to us, not a new request")
async def t_identifier_only_reply(_ctx: TestContext) -> None:
    """Greeting someone who just sent what we asked for reads as mechanical."""
    from src.core import local_support_controller as lsc

    for fragment in ("lina.perret12@gmail.com", "Re: (no subject) 123456",
                     "  a1b2c3d4e5f6a7b8c9d0  "):
        assert lsc._identifier_only(fragment) is True, fragment
    # A message that says something as well as carrying an id is not a bare id.
    for real in ("the app crashes on open", "Hello", "ok thanks",
                 "my account is x@y.com and premium is gone"):
        assert lsc._identifier_only(real) is False, real

    # An id is not a language to mirror. Measured: the composer answered an
    # unknown customer in French because the only "signal" was their email.
    assert lsc._language_hint("lina.perret12@gmail.com") == "und"


@test("local_support_controller", "the fallback classifier can only pick a label")
async def t_model_classifier_is_constrained(_ctx: TestContext) -> None:
    """It has no tools, and anything off-list is discarded as 'general'."""
    from src.core import local_support_controller as lsc

    class _Reply:
        def __init__(self, content: str) -> None:
            self.content = content

    class _M:
        def __init__(self, content: str) -> None:
            self._content = content
            self.system: str | None = None

        async def generate(self, messages, system, session_id, **_kw):
            self.system = system
            return _Reply(self._content)

    async def label_for(content: str) -> str:
        return await lsc._classify_with_model(
            SimpleNamespace(model=_M(content)), {"model": ""}, "qualcosa", "s",
        )

    assert await label_for('{"label":"offline"}') == "offline"
    # An invented label, prose, or an empty answer all mean "general".
    for bogus in ('{"label":"urgent_vip"}', "the customer wants a refund",
                  "{}", "", '{"label":""}'):
        assert await label_for(bogus) == "general", bogus

    # Deterministic rules are never overridden: the model is only consulted
    # when the term lists returned "general".
    assert lsc._intent("the app keeps crashing") == "bug"

    # Turning it off must leave the deterministic answer standing.
    previous = os.environ.get(lsc._CLASSIFIER_ENV)
    os.environ[lsc._CLASSIFIER_ENV] = "0"
    try:
        assert await label_for('{"label":"offline"}') == "general"
    finally:
        if previous is None:
            os.environ.pop(lsc._CLASSIFIER_ENV, None)
        else:
            os.environ[lsc._CLASSIFIER_ENV] = previous


@test("local_support_controller", "an old review is answered, a not-found one is closed")
async def t_review_send_is_attempted(_ctx: TestContext) -> None:
    """Age is not a reason for silence: only the store's refusal is."""
    from src.core import local_support_controller as lsc

    assert lsc._is_review_channel("playstore_reviews") is True
    assert lsc._is_review_channel("email") is False

    # The store cannot retrieve the review -> terminal, stop retrying.
    for refusal in (
        {"ok": False, "status": 404, "error": "Could not find review"},
        {"ok": False, "error": "NOT_FOUND"},
        {"ok": False, "status": 422},
    ):
        assert lsc._review_send_unrepliable(refusal) is True, refusal
    # An ordinary transport hiccup is NOT a reason to close the review.
    for retryable in ({"ok": False, "status": 500}, {"ok": False, "error": "timeout"}):
        assert lsc._review_send_unrepliable(retryable) is False, retryable

    assert lsc._review_stars({"review_stars": 5}, None) == 5
    assert lsc._review_stars({}, {"review_stars": 1}) == 1
    assert lsc._review_stars({}, {}) is None


@test("local_support_controller", "a terminal verdict is closed, not just tagged")
async def t_terminal_outcomes_are_closed(_ctx: TestContext) -> None:
    """Leaving these open is what refires one thread ~25 times."""
    from src.core import local_support_controller as lsc

    previous = os.environ.get(lsc._WRITES_ENV)
    os.environ[lsc._WRITES_ENV] = "1"
    try:
        for message, outcome in (
            ("Thanks, it works now!", "resolved_confirmation"),
            ("Ok!", "acknowledgement_no_reply_needed"),
            ("very good", "praise_no_reply_needed"),
        ):
            doubles = _Doubles()
            output = json.loads((await lsc.run(
                agent=SimpleNamespace(_mcp=doubles.pool(), model=_Model()),
                event={"slug": "replio-thread", "model": ""},
                payload={"payload": {"thread_id": "t-term",
                                     "message": {"body_text": message}}},
                session_id="s", delivery_id="d",
            )).text)
            assert output["outcome"] == outcome, (message, output["outcome"])
            patches = [a["patch"] for a in doubles.args_for("replio_threads_patch")]
            assert {"waiting_for_team": False, "status": "closed"} in patches, (
                message, patches
            )
            assert "replio_threads_respond" not in doubles.names, message
    finally:
        if previous is None:
            os.environ.pop(lsc._WRITES_ENV, None)
        else:
            os.environ[lsc._WRITES_ENV] = previous


@test("local_support_controller", "the tenant decides product facts, never the policy")
async def t_tenant_resolution(_ctx: TestContext) -> None:
    from src.core.local_support_controller import (
        _CLICKUP_LISTS, _TENANTS, _bug_symptom_route, _is_other_brand, _tenant_for,
    )

    # Replio's own `product` field is the discriminator; an unknown or absent
    # value keeps today's behaviour rather than guessing.
    assert _tenant_for({}).key == "esound"
    assert _tenant_for({"product": "lyra"}).key == "lyra"
    assert _tenant_for({"product": "eSound"}).key == "esound"
    assert _tenant_for({"product": "unknown-brand"}).key == "esound"

    esound, lyra = _TENANTS["esound"], _TENANTS["lyra"]
    # The brand-app row follows the tenant...
    _t, list_id, tag = _bug_symptom_route("crash in theme settings", esound)
    assert (list_id, tag) == (_CLICKUP_LISTS["esound"], "esound/app")
    _t, list_id, tag = _bug_symptom_route("crash in theme settings", lyra)
    assert (list_id, tag) == (_CLICKUP_LISTS["lyra"], "lyra/app")
    # ...but a SHARED component stays put for both, only the tag changes.
    # Routing is by component, not by product.
    _t, list_id, tag = _bug_symptom_route("crash when I use search", lyra)
    assert (list_id, tag) == (_CLICKUP_LISTS["client"], "lyra/client-core")

    # Regola Zero is symmetric now that both brands are served: act on the
    # product the thread belongs to, stay off the other one's board.
    assert _is_other_brand("The Lyra Music app crashes", "", esound) is True
    assert _is_other_brand("The Lyra Music app crashes", "", lyra) is False
    assert _is_other_brand("eSound crashes on open", "", lyra) is True
    # Naming both is the documented shared-component carve-out.
    assert _is_other_brand("both eSound and Lyra crash", "", esound) is False


@test("local_support_controller", "a tenant with no billing project fails closed")
async def t_tenant_without_billing_project(_ctx: TestContext) -> None:
    from src.core.local_support_controller import _TENANTS, _billing_lookup

    # Every configured tenant must carry its own project: querying another
    # brand's would report a different product's customer.
    for tenant in _TENANTS.values():
        assert tenant.billingbear_project, tenant.key
    assert (_TENANTS["lyra"].billingbear_project
            != _TENANTS["esound"].billingbear_project)

    unconfigured = replace(_TENANTS["lyra"], billingbear_project="")
    doubles = _Doubles()
    try:
        await _billing_lookup(doubles.pool(), "", "a@b.com", unconfigured)
    except RuntimeError as exc:
        assert "BillingBear project" in str(exc), exc
    else:
        raise AssertionError("the lookup answered without a configured project")


@test("local_support_controller", "a correction is read back, but never a product claim")
async def t_corrections_are_procedural(_ctx: TestContext) -> None:
    """The loop was open at the far end: corrections were written for weeks
    and no code path ever loaded one."""
    from src.core import local_support_controller as lsc

    for procedural in (
        "Answer this customer in the language they wrote in.",
        "Acknowledge what they said before asking anything.",
        "Do not say an issue is tracked unless a task succeeded this turn.",
    ):
        assert lsc._LEARNING_IS_A_PRODUCT_CLAIM.search(procedural) is None, procedural
    # A learning that asserts what the product does would be injected into
    # every later reply on that thread - the most expensive mistake available.
    for claim in (
        "The app does not support offline playback.",
        "Cloud Import is not available on Android.",
        "l'app non supporta le playlist condivise",
    ):
        assert lsc._LEARNING_IS_A_PRODUCT_CLAIM.search(claim) is not None, claim


@test("quality_scorer", "a weak dimension earns a fixed procedural correction")
async def t_correction_for(_ctx: TestContext) -> None:
    """Fixed sentences on purpose: a correction reaches every later reply, so
    no model-written text is allowed into one."""
    from src.core.local_quality_scorer import (
        correction_for, verdict_for, weighted_score,
    )

    good = {"grounding": 1.0, "appropriateness": 1.0, "tone": 1.0,
            "f7_read": 1.0, "language": 1.0, "length": 1.0}
    assert correction_for(good) is None

    wrong_language = {**good, "language": 0.0}
    earned = correction_for(wrong_language)
    assert earned is not None and earned[0] == "correction: language"
    # It still earns one even though the weighted score reads as GOOD: a reply
    # in the wrong language is not a good reply.
    assert verdict_for(weighted_score(wrong_language), wrong_language) == "GOOD"

    invented = {**good, "grounding": 0.0}
    assert verdict_for(weighted_score(invented), invented) == "BAD"
    assert correction_for(invented)[0] == "correction: grounding"
