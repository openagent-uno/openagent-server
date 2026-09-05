"""Media must reach the model as images, remain bounded and fail closed."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

from ._framework import TestContext, test
from .test_local_support_controller import _Doubles
from src.core import local_support_controller as controller, support_attachments
from src.core.tool_scope import current_tool_allowlist


def _receipt():
    raw = (Path(__file__).parents[1] / "fixtures/support-vision-wc037.png").read_bytes()
    block = {"type": "image", "mimeType": "image/png", "data": base64.b64encode(raw).decode()}
    return raw, {"content": [{"type": "text", "text": "Image added to the response"}, block]}


@test("support_attachments", "image receipts deduplicate and enforce byte/count/error limits")
async def t_media_limits(_ctx: TestContext) -> None:
    raw, receipt = _receipt()
    receipt["content"].append(receipt["content"][-1])
    images, incomplete = support_attachments.images_from_receipt(receipt)
    assert len(images) == 1 and images[0].content == raw and not incomplete
    assert support_attachments.images_from_receipt(receipt, max_bytes=10) == ([], True)
    assert support_attachments.images_from_receipt(receipt, max_images=0) == ([], True)
    assert support_attachments.images_from_receipt({**receipt, "isError": True}) == ([], True)
    assert support_attachments.images_from_receipt({"images": [{"mime_type": "image/png", "base64_content": "?bad"}]}) == ([], True)


@test("support_attachments", "native image bytes reach the selected vision model without tool authority")
async def t_native_vision(_ctx: TestContext) -> None:
    raw, receipt = _receipt()
    doubles = _Doubles(attachment=receipt)
    seen = []
    class Model:
        def build_override_model(self, spec):
            seen.append(spec)
            return self
        async def generate(self, **kw):
            assert current_tool_allowlist() == frozenset()
            assert kw["images"][0].content == raw
            assert base64.b64encode(raw).decode() not in json.dumps(kw["messages"])
            return SimpleNamespace(content=json.dumps({"readable": True, "visible_text": "ERROR\nWC037", "observation": "The screenshot is displaying an error."}))
    state = controller.SupportState(thread_id="sim", customer_message="", intent="attachment_only")
    await controller._inspect_support_attachments(doubles.pool(), state, SimpleNamespace(model=Model()), {"vision_model": "registered:vision"}, "test")
    assert seen == ["registered:vision"]
    assert state.facts["attachment_images_processed"] == 1 and state.facts["attachment_readable"]
    assert state.attachment_visible_text == "ERROR\nWC037"
    assert "displaying" not in controller._bug_report_text(state)
    assert controller._bug_symptom_route("displaying an error") is None


@test("support_attachments", "unavailable vision override never claims image inspection succeeded")
async def t_unavailable_vision(_ctx: TestContext) -> None:
    _, receipt = _receipt()
    doubles = _Doubles(attachment=receipt)
    class Model:
        def build_override_model(self, spec):
            raise ValueError("model unavailable")
    state = controller.SupportState(thread_id="sim", customer_message="", intent="attachment_only")
    await controller._inspect_support_attachments(doubles.pool(), state, SimpleNamespace(model=Model()), {"vision_model": "registered:missing"}, "test")
    assert not state.facts["attachment_readable"]
    assert state.facts["attachment_inspection_incomplete"]
    assert state.facts["attachment_images_processed"] == 0


@test("support_attachments", "a receipt with known identity does not request identity or purchase proof again")
async def t_known_receipt_identity(_ctx: TestContext) -> None:
    state = controller.SupportState(thread_id="sim", customer_message="", intent="attachment_only",
                                    account_ref="test-account", attachment_observation="Purchase receipt TEST-ORDER")
    state.facts["attachment_readable"] = True
    await controller._route_attachment(_Doubles().pool(), state)
    assert state.decision == "human" and state.outcome == "attachment_billing_review"
    assert "email" not in controller._fallback_reply(state).lower()
    assert not state.facts.get("billing_verified")


@test("support_attachments", "historical media overflow does not erase readable current evidence")
async def t_partial_evidence(_ctx: TestContext) -> None:
    _, receipt = _receipt()
    doubles = _Doubles(attachment=receipt)
    class Model:
        async def generate(self, **kw):
            return SimpleNamespace(content=json.dumps({"readable": True, "visible_text": "Receipt TEST-ORDER", "observation": "A purchase document"}))
    state = controller.SupportState(thread_id="sim", customer_message="[1 attachment(s): image]", intent="attachment_only")
    state.facts["attachment_budget_exceeded"] = True
    await controller._inspect_support_attachments(doubles.pool(), state, SimpleNamespace(model=Model()), {}, "test")
    assert state.facts["attachment_readable"] and state.facts["attachment_inspection_incomplete"]
    assert "TEST-ORDER" in state.attachment_visible_text
    await controller._route_attachment(doubles.pool(), state)
    assert state.outcome == "attachment_partial_review" and state.decision == "human"
