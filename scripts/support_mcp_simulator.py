#!/usr/bin/env python3
"""Safe BillingBear/Replio/ClickUp MCP simulator for support evaluations.

The process never contacts external services and never persists customer or
thread state. It intentionally exposes a small, production-shaped tool surface
so an agent can be evaluated on discovery, argument construction, branching,
mutation choice, receipt handling, and human escalation without touching a
real subscription or support thread.
"""
from __future__ import annotations

import sys
from typing import Any

from mcp.server.fastmcp import FastMCP


ROLE = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
if ROLE not in {
    "billingbear", "replio", "clickup", "messaging", "esound-admin", "lyra-admin",
    "esound-identity",
}:
    raise SystemExit(
        "usage: support_mcp_simulator.py "
        "billingbear|replio|clickup|messaging|esound-admin|lyra-admin|esound-identity"
    )

mcp = FastMCP(ROLE)


if ROLE == "billingbear":

    @mcp.tool()
    async def get_v1_customers_by_appUserId(appUserId: str) -> dict[str, Any]:
        """Customer record by appUserId: returns isPremium, store, clientVersion, subscriptions and entitlements."""
        fixtures = {
            "test-active": {
                "isPremium": True,
                "store": "paddle",
                "clientVersion": "5.0.18",
                "subscriptions": [{"id": "sub-active", "status": "active", "willRenew": True}],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-expired": {
                "isPremium": True, "store": "paddle", "clientVersion": "5.1.1",
                "premiumExpiresAt": "2020-01-01T00:00:00+00:00",
                "subscriptions": [], "entitlements": [],
            },
            "test-active-email": {
                "isPremium": True, "store": "google", "clientVersion": "5.1.1",
                "premiumExpiresAt": "2099-01-01T00:00:00+00:00",
                "subscriptions": [{"id": "sub-g", "status": "active"}],
                "entitlements": [{"id": "premium", "expiresAt": "2099-01-01T00:00:00Z"}],
            },
            "test-cancel-web": {
                "isPremium": True, "store": "stripe", "clientVersion": "5.1.1",
                "subscriptions": [{"id": "sub-web", "status": "active", "willRenew": True}],
                "entitlements": [{"id": "premium", "expiresAt": "2099-01-01T00:00:00Z"}],
            },
            "test-cancelled-web": {
                "isPremium": True, "store": "stripe", "clientVersion": "5.1.1",
                "subscriptions": [{"id": "sub-web", "status": "active", "willRenew": False}],
                "entitlements": [{"id": "premium", "expiresAt": "2099-01-01T00:00:00Z"}],
            },
            "test-expired-entitlement": {
                "store": "paddle", "clientVersion": "5.1.1",
                "subscriptions": [],
                "entitlements": [{"id": "premium", "expiresAt": "2026-07-09T00:00:00Z"}],
            },
            "test-duplicate-overcap": {
                "isPremium": True, "store": "stripe", "clientVersion": "5.1.1",
                "subscriptions": [{"id": "sub-a", "status": "active"}],
                "entitlements": [{"id": "premium", "expiresAt": "2099-01-01T00:00:00Z"}],
            },
            "test-apple": {
                "isPremium": True, "store": "apple", "clientVersion": "5.1.1",
                "subscriptions": [{"id": "sub-apple", "status": "active"}],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-google-old": {
                "isPremium": True, "store": "google", "clientVersion": "5.0.9",
                "subscriptions": [{"id": "sub-google", "status": "active"}],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-stripe": {
                "isPremium": True, "store": "stripe", "clientVersion": "5.1.1",
                "subscriptions": [{"id": "sub-stripe", "status": "active"}],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-nostore": {
                "isPremium": True, "store": None, "clientVersion": "5.1.1",
                "subscriptions": [{"id": "sub-x", "status": "active"}],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-inactive": {
                "isPremium": False,
                "store": None,
                "clientVersion": "5.0.17",
                "subscriptions": [],
                "entitlements": [],
            },
            "test-duplicate": {
                "isPremium": True,
                "store": "stripe",
                "clientVersion": "5.0.18",
                "subscriptions": [
                    {"id": "sub-keep", "status": "active", "willRenew": True},
                    {"id": "sub-duplicate", "status": "active", "willRenew": True},
                ],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            # A yearly web plan with the OLD store subscription still live.
            # Different product ids, so the duplicate detector reports nothing
            # while the customer really is paying twice - the shape behind the
            # Spanish "Cobro denegado" thread of 28-Aug-2026.
            "test-other-store": {
                "isPremium": True,
                "store": "paddle",
                "clientVersion": "5.2.2",
                "subscriptions": [
                    {"id": "sub-web-yearly", "provider": "Paddle",
                     "productId": "pro_yearly", "status": "active",
                     "isActive": True, "willRenew": True,
                     "expiresAt": "2027-07-27T14:59:31+00:00"},
                    {"id": "sub-play-monthly", "provider": "Google",
                     "productId": "esoundpremium_m:p1m", "status": "active",
                     "isActive": True, "willRenew": True,
                     "expiresAt": "2027-01-29T14:48:23+00:00"},
                ],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-refund": {
                "isPremium": True,
                "store": "paddle",
                "clientVersion": "5.0.23",
                "subscriptions": [{
                    "id": "sub-refund", "status": "active", "willRenew": True,
                    "lastPaymentAt": "2026-08-18T10:00:00Z",
                    "lastPaymentAmount": 9.99,
                }],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-refund-old": {
                "isPremium": True, "store": "stripe", "clientVersion": "5.1.1",
                "subscriptions": [{
                    "id": "sub-old", "status": "active", "willRenew": True,
                    "lastPaymentAt": "2026-01-10T10:00:00Z",
                    "lastPaymentAmount": 9.99,
                }],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            # Real catalogue: Premium is 14.99/yr, 1.99/mo. An amount far
            # above the yearly plan is contradictory data, not a big plan.
            "test-refund-anomalous": {
                "isPremium": True, "store": "stripe", "clientVersion": "5.1.1",
                "subscriptions": [{
                    "id": "sub-odd", "status": "active", "willRenew": True,
                    "lastPaymentAt": "2026-08-19T10:00:00Z",
                    "lastPaymentAmount": 149.99, "currency": "USD",
                }],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            # 79.99 BRL is about 13 EUR - the yearly plan, not a large refund.
            "test-refund-brl": {
                "isPremium": True, "store": "stripe", "clientVersion": "5.1.1",
                "subscriptions": [{
                    "id": "sub-brl", "status": "active", "willRenew": True,
                    "lastPaymentAt": "2026-08-19T10:00:00Z",
                    "lastPaymentAmount": 79.99, "currency": "BRL",
                }],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-amazon": {
                "isPremium": True, "store": "amazon", "clientVersion": "5.1.1",
                "subscriptions": [{"id": "sub-amz", "status": "active"}],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-refund-nodate": {
                "isPremium": True, "store": "stripe", "clientVersion": "5.1.1",
                "subscriptions": [{"id": "sub-nodate", "status": "active"}],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
            "test-google": {
                "isPremium": True,
                "store": "google",
                "clientVersion": "5.0.23",
                "subscriptions": [{
                    "id": "sub-google", "status": "active",
                    "lastPaymentAt": "2026-08-18T10:00:00Z",
                    "lastPaymentAmount": 4.99,
                }],
                "entitlements": [{"id": "premium", "status": "active"}],
            },
        }
        if appUserId not in fixtures:
            return {"ok": False, "status": 404, "reason": "customer_not_found", "appUserId": appUserId}
        return {"ok": True, "status": 200, "appUserId": appUserId, **fixtures[appUserId]}

    @mcp.tool()
    async def get_entitlements_granted_by_appUserId(appUserId: str) -> dict[str, Any]:
        """Entitlements only for an appUserId: returns entitlements. Does NOT return store or clientVersion."""
        customer = await get_v1_customers_by_appUserId(appUserId)
        return {
            "ok": customer.get("ok", False),
            "status": customer.get("status"),
            "appUserId": appUserId,
            "entitlements": customer.get("entitlements", []),
        }

    @mcp.tool()
    async def get_v1_projects_by_projectId_paddle_lookup(
        projectId: str, query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Paddle-scoped resolver: answers for Paddle alone."""
        return {"ok": True, "status": 200, "projectId": projectId,
                "query": query or {}, "customers": [], "simulated": True,
                "paddle_scope_only": True}

    @mcp.tool()
    async def get_customer_by_email(
        projectId: str, query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Mirrors production: projectId + the address inside
        `query`, answering with appUserId / isPremium / premiumSource.

        The old version always answered 404, copying a vault note that turned
        out to be wrong. A simulator that lies about the one call the premium
        path depends on hides exactly the bug it exists to catch.
        """
        email = str((query or {}).get("email") or "").strip().lower()
        if not email or "@" not in email:
            return {"ok": False, "status": 400, "error": "valid email required",
                    "simulated": True}
        known = {
            "premium@example.com": {
                "appUserId": "test-active-email", "isPremium": True,
                "premiumExpiresAt": "2099-01-01T00:00:00+00:00",
                "premiumSource": "Google",
            },
            "expired@example.com": {
                "appUserId": "test-expired", "isPremium": True,
                "premiumExpiresAt": "2020-01-01T00:00:00+00:00",
                "premiumSource": "Paddle",
            },
            "cancel@example.com": {
                "appUserId": "test-cancel-web", "isPremium": True,
                "premiumExpiresAt": "2099-01-01T00:00:00+00:00",
                "premiumSource": "Stripe",
            },
        }
        found = known.get(email)
        if not found:
            return {"ok": False, "status": 404, "email": email,
                    "error": "customer not found", "simulated": True}
        return {"ok": True, "status": 200, "email": email, "simulated": True, **found}

    @mcp.tool()
    async def detect_duplicate_subscriptions(appUserId: str) -> dict[str, Any]:
        """Read-only duplicate detector. Returns a deterministic simulated plan."""
        found = appUserId == "test-duplicate"
        return {
            "ok": True,
            "status": 200,
            "appUserId": appUserId,
            "duplicatesFound": found or appUserId == "test-duplicate-overcap",
            "withinValueCap": found,
            "keepSubscriptionId": "sub-keep" if found else None,
            "refundSubscriptionIds": ["sub-duplicate"] if found else [],
            "simulated": True,
        }

    @mcp.tool()
    async def refund_duplicate_subscriptions(
        appUserId: str, dryRun: bool = True,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures the refund choice; never moves money."""
        found = appUserId == "test-duplicate"
        return {
            "ok": found,
            "success": found,
            "status": 200 if found else 409,
            "appUserId": appUserId,
            "dryRun": True,
            "simulated": True,
            "wouldRefund": ["sub-duplicate"] if found else [],
            "wouldKeepActive": "sub-keep" if found else None,
            "externalMutation": False,
        }

    @mcp.tool()
    async def post_v1_customer_center_by_appUserId_subscriptions_by_subscr(
        appUserId: str, subscriptionId: str, action: str = "cancel",
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a confirmed cancellation; never mutates billing."""
        valid = action == "cancel" and (
            (appUserId, subscriptionId) in {
                ("test-active", "sub-active"),
                ("test-cancelled-web", "sub-web"),
            }
        )
        return {
            "ok": valid,
            "success": valid,
            "status": 200 if valid else 409,
            "dryRun": True,
            "simulated": True,
            "wouldCancel": subscriptionId if valid else None,
            "externalMutation": False,
        }

    @mcp.tool()
    async def post_v1_customer_center_by_appUserId_subscriptions_by_subscr_4(
        appUserId: str, subscriptionId: str,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a policy-eligible single-subscription refund."""
        valid = (appUserId, subscriptionId) in {
            ("test-refund", "sub-refund"),
            ("test-refund-brl", "sub-brl"),
        }
        return {
            "ok": valid,
            "success": valid,
            "status": 202 if valid else 409,
            "dryRun": True,
            "simulated": True,
            "adjustmentStatus": "pending_approval" if valid else None,
            "wouldRefund": subscriptionId if valid else None,
            "externalMutation": False,
        }


if ROLE == "replio":
    _THREAD_LINKS: dict[str, str] = {}

    @mcp.tool()
    async def threads_get(thread_id: str) -> dict[str, Any]:
        """Read a simulated support thread."""
        result = {
            "ok": True,
            "id": thread_id,
            "status": "open",
            "waiting_for_team": False,
            "tags": [],
            "channel": "email",
            "messages": [],
        }
        if thread_id in {
            "sim-delete", "sim-delete-confirm", "sim-account-change", "sim-it-delete",
        }:
            result["author_email"] = "account@example.com"
        if thread_id == "sim-cancel-email":
            result["author_email"] = "cancel@example.com"
        if thread_id == "sim-email-premium":
            result["author_email"] = "premium@example.com"
        if thread_id == "sim-email-expired":
            result["author_email"] = "expired@example.com"
        if thread_id == "sim-diag-followup":
            result["author_email"] = "diag@example.com"
            result["tags"] = ["bug", "diagnostics-active"]
            result["external_task_id"] = "86-local-existing"
        if thread_id == "sim-lyra-diag-followup":
            result["author_email"] = "diaglyra@example.com"
            result["product"] = "lyra"
            result["tags"] = ["bug", "diagnostics-active"]
            result["external_task_id"] = "86-local-created-lyra"
        if thread_id == "sim-identity":
            result["messages"] = [
                {"direction": "outbound",
                 "body_text": "Could you send us the email on your account?"},
                {"direction": "inbound", "body_text": "lina.perret12@gmail.com"},
            ]
        if thread_id == "sim-cancel-phase2":
            result["tags"] = ["billing", "subcancel-pending"]
            result["messages"] = [
                {"direction": "inbound", "sent_at": "2026-08-20T09:00:00Z",
                 "body_text": "I want to cancel my subscription."},
                {"direction": "outbound", "sent_at": "2026-08-20T10:00:00Z",
                 "body_text": "Can you confirm you want to cancel?"},
                {"direction": "inbound", "sent_at": "2026-08-21T09:00:00Z",
                 "body_text": "confermo"},
            ]
        if thread_id == "sim-review-old":
            result["channel"] = "playstore_reviews"
            result["last_inbound_at"] = "2026-06-01T08:00:00Z"
            result["review_stars"] = 1
        if thread_id == "sim-review-5":
            result["channel"] = "playstore_reviews"
            result["last_inbound_at"] = "2026-08-20T08:00:00Z"
            result["review_stars"] = 5
        if thread_id == "sim-review-1":
            result["channel"] = "playstore_reviews"
            result["last_inbound_at"] = "2026-08-20T08:00:00Z"
            result["review_stars"] = 1
        if thread_id == "sim-expired":
            result["channel"] = "messenger"
            result["last_inbound_at"] = "2026-08-18T08:00:00Z"
        if thread_id in _THREAD_LINKS:
            result["external_task_id"] = _THREAD_LINKS[thread_id]
        return result

    @mcp.tool()
    async def thread_read_attachment(
        thread_id: str, attachment_index: int = 0,
    ) -> dict[str, Any]:
        """Read-only attachment fixture; default reproduces the vision placeholder gotcha."""
        if thread_id == "sim-receipt-readable":
            return {
                "ok": True,
                "text": "Purchase receipt. Order ID TEST-ORDER. Account email account@example.com",
                "attachment_index": attachment_index,
            }
        return {
            "ok": True,
            "text": "[Attachment placeholder: no image content was included]",
            "attachment_index": attachment_index,
        }

    @mcp.tool()
    async def threads_draft(
        thread_id: str, body_text: str, confidence: float | None = None,
        reasoning: str | None = None, origin: str = "ai",
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a pending draft; nothing reaches a customer."""
        return {"ok": True, "success": True, "dryRun": True, "simulated": True,
                "thread_id": thread_id, "captured_chars": len(body_text),
                "origin": origin, "externalMutation": False}

    @mcp.tool()
    async def threads_list(
        status: str = "", waiting_for_team: bool | None = None,
        channel_kind: str = "", limit: int = 50, order: str = "newest",
        tag: str = "", q: str = "", app_id: str = "", channel_id: str = "",
        compact: bool = False,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. A small, deterministic thread list."""
        waiting = [
            # Rightly waiting: a legal matter the bot may never answer.
            {"id": "sim-esc-legal", "status": "open", "waiting_for_team": True,
             "tags": ["legal", "needs-human"], "channel_kind": "email",
             "subject": "Copyright takedown"},
            # Avoidable: a plain download question the vault answers.
            {"id": "sim-esc-easy", "status": "open", "waiting_for_team": True,
             "tags": ["needs-human"], "channel_kind": "email",
             "subject": "How do I download songs for offline?"},
        ]
        closed = [
            # Under-escalated: a refund closed by the bot on its own.
            {"id": "sim-closed-refund", "status": "closed",
             "waiting_for_team": False, "tags": ["billing"],
             "channel_kind": "email", "subject": "I want my money back"},
        ]
        if waiting_for_team is True:
            items = waiting
        elif str(status).lower() == "closed":
            items = closed
        else:
            items = waiting + closed
        return {"ok": True, "count": len(items), "threads": items[:limit],
                "simulated": True}

    @mcp.tool()
    async def replies_to_score(product: str = "", limit: int = 20) -> dict[str, Any]:
        """SIMULATOR ONLY. Outbound replies awaiting a quality grade."""
        return {"ok": True, "count": 2, "items": [
            {
                "message_id": "m-good", "thread_id": "sim-q-good",
                "product": product or "esound", "channel_kind": "email",
                "last_inbound": "Non riesco a scaricare le canzoni, cosa devo fare?",
                "reply": "Per aiutarti, indicami la versione dell'app e il dispositivo.",
                "has_task": False, "escalated": False,
                "inbound_attachments": [], "attachment_read": False, "actions": [],
            },
            {
                "message_id": "m-bad", "thread_id": "sim-q-bad",
                "product": product or "esound", "channel_kind": "email",
                "last_inbound": "L'app si chiude da sola ogni volta che apro una playlist",
                "reply": "It is a known issue, the team is already tracking it and it "
                         "will be fixed in the next update.",
                "has_task": False, "escalated": False,
                "inbound_attachments": [], "attachment_read": False, "actions": [],
            },
        ], "simulated": True}

    @mcp.tool()
    async def quality_stats(window_days: int = 7, product: str = "") -> dict[str, Any]:
        """SIMULATOR ONLY. Deterministic quality aggregates."""
        return {"ok": True, "window_days": window_days, "product": product,
                "n": 1014, "avg_score": 0.82, "good": 719, "ok": 244,
                "bad": 51, "simulated": True}

    @mcp.tool()
    async def quality_record(
        message_id: str, thread_id: str, product: str = "", channel_kind: str = "",
        score: float = 0.0, verdict: str = "", dimensions: dict[str, Any] | None = None,
        grader: str = "", notes: str = "",
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a quality row; never writes to production.

        The real eSound Replio key is READ-ONLY, so the simulator refuses the
        write for that product too. A scorer that only works against a
        permissive simulator would tell us nothing about production.
        """
        if (product or "").lower() == "esound":
            return {"ok": False, "success": False, "status": 403,
                    "error": "MCP key is read-only", "simulated": True}
        return {"ok": True, "success": True, "simulated": True,
                "message_id": message_id, "score": score, "verdict": verdict,
                "dimensions": dimensions or {}, "grader": grader,
                "externalMutation": False}

    @mcp.tool()
    async def learnings_list(limit: int = 100) -> dict[str, Any]:
        """SIMULATOR ONLY. Recent learnings, with a dimension that recurs."""
        items = [
            {"kind": "correction", "title": "correction: language"},
            {"kind": "correction", "title": "correction: language"},
            {"kind": "correction", "title": "correction: language"},
            {"kind": "correction", "title": "correction: grounding"},
            {"kind": "correction", "title": "correction: tone"},
            {"kind": "correction", "title": "correction: length"},
            {"kind": "context", "title": "non una correzione"},
        ]
        return {"ok": True, "items": items[:limit], "simulated": True}

    @mcp.tool()
    async def thread_learnings(thread_id: str, limit: int = 10) -> dict[str, Any]:
        """SIMULATOR ONLY. Corrections previously recorded on this thread."""
        if thread_id != "sim-learned":
            return {"ok": True, "items": [], "simulated": True}
        return {"ok": True, "simulated": True, "items": [
            {"kind": "correction", "title": "lingua",
             "content": "Answer this customer in Italian: they wrote in Italian "
                        "and a previous reply went out in English."},
            {"kind": "correction", "title": "fatto di prodotto",
             "content": "The app does not support offline playback."},
            {"kind": "context", "title": "non una correzione",
             "content": "Customer prefers short replies."},
        ]}

    @mcp.tool()
    async def thread_learning_add(
        thread_id: str, title: str, content: str, kind: str = "note",
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a learning; never writes to production."""
        if thread_id == "sim-q-bad":
            # Lyra's key writes; eSound's does not. Let one through so the
            # loop can be observed end to end, and refuse the other so the
            # read-only path stays exercised too.
            return {"ok": True, "success": True, "simulated": True,
                    "thread_id": thread_id, "kind": kind, "externalMutation": False}
        if thread_id.startswith("sim-q-"):
            return {"ok": False, "success": False, "status": 403,
                    "error": "MCP key is read-only", "simulated": True}
        return {"ok": True, "success": True, "simulated": True,
                "thread_id": thread_id, "kind": kind, "externalMutation": False}

    @mcp.tool()
    async def threads_respond(
        thread_id: str,
        body_text: str,
        verified_actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a response; never sends to a customer."""
        if thread_id == "sim-review-old":
            # The store refuses: it cannot retrieve this review any more.
            return {"ok": False, "success": False, "status": 404,
                    "error": "Could not find review", "simulated": True}
        return {"ok": True, "success": True, "dryRun": True, "simulated": True,
                "thread_id": thread_id, "captured_chars": len(body_text),
                "verified_actions": verified_actions or [], "externalMutation": False}

    @mcp.tool()
    async def threads_mark_for_human(thread_id: str, reason: str) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a justified human handoff."""
        return {"ok": True, "success": True, "dryRun": True, "simulated": True,
                "thread_id": thread_id, "reason": reason, "externalMutation": False}

    @mcp.tool()
    async def threads_tags_add(thread_id: str, tags: list[str]) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures tags without changing a real thread."""
        return {"ok": True, "success": True, "dryRun": True, "simulated": True,
                "thread_id": thread_id, "tags": tags, "externalMutation": False}

    @mcp.tool()
    async def threads_tags_remove(thread_id: str, tags: list[str]) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures tag removal without changing a real thread."""
        return {"ok": True, "success": True, "dryRun": True, "simulated": True,
                "thread_id": thread_id, "tags": tags, "externalMutation": False}

    @mcp.tool()
    async def threads_patch(thread_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a lifecycle patch without changing a real thread."""
        return {"ok": True, "success": True, "dryRun": True, "simulated": True,
                "thread_id": thread_id, "patch": patch, "externalMutation": False}

    @mcp.tool()
    async def thread_link_task(
        thread_id: str, task_provider_id: str, external_task_id: str,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a task link without changing Replio."""
        _THREAD_LINKS[thread_id] = external_task_id
        return {
            "ok": True, "success": True, "dryRun": True, "simulated": True,
            "thread_id": thread_id, "task_provider_id": task_provider_id,
            "external_task_id": external_task_id, "externalMutation": False,
        }


if ROLE == "esound-admin":

    @mcp.tool()
    async def search_users(query: str) -> dict[str, Any]:
        """SIMULATOR ONLY. Read-only account lookup by email/username/id/authId."""
        known = {
            "cancel@example.com": {
                "id": 7900276, "authId": "5f8349b40a31dbebd7063a5d",
                "email": "cancel@example.com", "isPremium": True,
            },
            "diag@example.com": {
                "id": 7900999, "authId": "test-diag-auth",
                "email": "diag@example.com", "isPremium": False,
            },
        }
        user = known.get(str(query or "").strip().lower())
        return {"ok": True, "status": 200, "users": [user] if user else [],
                "simulated": True}

    @mcp.tool()
    async def list_diagnostic_categories() -> dict[str, Any]:
        """SIMULATOR ONLY. Product-supported diagnostic category names."""
        return {"ok": True, "categories": [
            "ads", "auth", "general", "library", "network", "playback",
            "playlists", "purchases", "search", "sync",
        ], "simulated": True}

    @mcp.tool()
    async def enable_diagnostics(
        userId: int, categories: list[str],
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures enable intent without changing an account."""
        return {
            "ok": True, "success": True, "userId": userId,
            "categories": categories, "dryRun": True, "simulated": True,
            "externalMutation": False,
        }

    @mcp.tool()
    async def list_diagnostic_streams(userId: int) -> dict[str, Any]:
        """SIMULATOR ONLY. One captured playback stream after reproduction."""
        return {"ok": True, "streams": [
            {"category": "playback", "sizeBytes": 512},
        ], "userId": userId, "simulated": True}

    @mcp.tool()
    async def read_diagnostic_stream(
        userId: int, category: str, tailBytes: int | None = None,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Synthetic diagnostic evidence."""
        return {"ok": True, "userId": userId, "category": category,
                "content": "player state=buffering source=remote retry=3",
                "tailBytes": tailBytes, "simulated": True}

    @mcp.tool()
    async def clear_diagnostic_streams(userId: int) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures cleanup without deleting anything."""
        return {"ok": True, "success": True, "userId": userId,
                "dryRun": True, "simulated": True, "externalMutation": False}


if ROLE == "lyra-admin":

    @mcp.tool()
    async def search_users(query: str) -> dict[str, Any]:
        """SIMULATOR ONLY. Read-only Lyra identity lookup."""
        known = {
            "diaglyra@example.com": {
                "identityId": "lyra-diag-identity",
                "email": "diaglyra@example.com",
            },
        }
        user = known.get(str(query or "").strip().lower())
        return {"ok": True, "status": 200, "users": [user] if user else [],
                "simulated": True}

    @mcp.tool()
    async def list_diagnostic_categories() -> dict[str, Any]:
        """SIMULATOR ONLY. Product-supported diagnostic category names."""
        return {"ok": True, "categories": [
            "auth", "general", "library", "network", "playback",
            "playlists", "purchases", "search", "sync",
        ], "simulated": True}

    @mcp.tool()
    async def enable_diagnostics(
        identityId: str, categories: list[str],
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures enable intent without changing an account."""
        return {
            "ok": True, "success": True, "identityId": identityId,
            "categories": categories, "dryRun": True, "simulated": True,
            "externalMutation": False,
        }

    @mcp.tool()
    async def list_diagnostic_logs(identityId: str) -> dict[str, Any]:
        """SIMULATOR ONLY. One captured general log after reproduction."""
        return {"ok": True, "items": [
            {"category": "general", "sizeBytes": 512},
        ], "identityId": identityId, "simulated": True}

    @mcp.tool()
    async def read_diagnostic_log(
        identityId: str, category: str, tailBytes: int | None = None,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Synthetic diagnostic evidence."""
        return {"ok": True, "identityId": identityId, "category": category,
                "content": "theme screen event=open result=crash code=simulated",
                "tailBytes": tailBytes, "simulated": True}

    @mcp.tool()
    async def clear_diagnostic_logs(identityId: str) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures cleanup without deleting anything."""
        return {"ok": True, "success": True, "identityId": identityId,
                "dryRun": True, "simulated": True, "externalMutation": False}


if ROLE == "clickup":

    @mcp.tool()
    async def get_workspace_tasks(
        listId: str, query: str, includeClosed: bool = True,
    ) -> dict[str, Any]:
        """Read-only simulated semantic task search."""
        low = query.lower()
        matched = any(term in low for term in (
            "playback", "endless", "infinite", "loading",
            "riproduzione", "caricamento",
        ))
        tasks = []
        if matched and listId == "901512182215":
            tasks.append({
                "id": "86-local-existing",
                "name": "Fix playback hanging on infinite loading",
                "status": "open",
                "listId": listId,
            })
        if "carplay" in low and listId == "901512182215":
            # A recurrence on a task that was already closed.
            tasks.append({
                "id": "86-closed-one", "name": "Fix CarPlay disconnect",
                "status": "closed", "listId": listId,
            })
        if "equalizer" in low and listId == "901512182215":
            tasks.append({
                "id": "86-seen-before", "name": "Fix equalizer preset reset",
                "status": "open", "listId": listId,
            })
        if "sleep" in low and listId == "901512182215":
            # Deliberately WRONG: the search shares a word with the report but
            # is a different defect. Claiming this one is "already tracked" is
            # exactly what the dedup judgment has to prevent.
            tasks.append({
                "id": "86-wrong-one", "name": "Fix crash in the library",
                "status": "open", "listId": listId,
            })
        return {
            "ok": True, "status": 200, "query": query,
            "includeClosed": includeClosed, "tasks": tasks,
        }

    _TASK_COMMENTS: dict[str, list[str]] = {
        # A task that already carries this thread's marker: the dedup protocol
        # must skip it instead of appending the same evidence again.
        "86-local-existing": [],
        "86-seen-before": [
            "<!-- source: support_email:sim-bug-seen -->\n\nprevious report"
        ],
    }

    @mcp.tool()
    async def get_task_comments(task_id: str) -> dict[str, Any]:
        """Read-only: the comments the dedup marker check inspects."""
        return {"ok": True, "task_id": task_id,
                "comments": _TASK_COMMENTS.get(task_id, [])}

    @mcp.tool()
    async def update_task(
        task_id: str, status: str | None = None, priority: int | None = None,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures a reopen/priority change."""
        return {"ok": True, "success": True, "dryRun": True, "simulated": True,
                "task_id": task_id, "status": status, "priority": priority,
                "externalMutation": False}

    @mcp.tool()
    async def create_task_comment(
        task_id: str, comment_text: str,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures an evidence comment."""
        return {
            "ok": True, "success": True, "dryRun": True, "simulated": True,
            "task_id": task_id, "captured_chars": len(comment_text),
            "externalMutation": False,
        }

    @mcp.tool()
    async def create_task(
        listId: str,
        name: str,
        description: str,
        priority: int = 3,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """SIMULATOR ONLY. Validates and captures a production-shaped task create."""
        valid = (
            listId in {
                "901512174103", "901512182180", "901519288689",
                "901512182025", "901512182215",
            }
            and 5 < len(name) <= 80
            and description.startswith("<!-- source:")
            and 1 <= priority <= 4
        )
        suffix = {
            "901512174103": "client",
            "901512182180": "backend",
            "901519288689": "bloom",
            "901512182025": "lyra",
            "901512182215": "esound",
        }.get(listId, "unknown")
        return {
            "ok": valid,
            "success": valid,
            "status": 201 if valid else 400,
            "dryRun": True,
            "simulated": True,
            # The id names the list so a benchmark can prove component routing,
            # not merely that some task was created.
            "id": f"86-local-created-{suffix}" if valid else None,
            "name": name,
            "listId": listId,
            "priority": priority,
            "tags": tags or [],
            "captured_chars": len(description),
            "externalMutation": False,
        }


if ROLE == "messaging":

    @mcp.tool()
    async def send_telegram(chat_id: str, text: str) -> dict[str, Any]:
        """SIMULATOR ONLY. Captures the owner notification; sends nothing."""
        return {"ok": True, "success": True, "dryRun": True, "simulated": True,
                "chat_id": chat_id, "captured_chars": len(text),
                "externalMutation": False}


if __name__ == "__main__":
    mcp.run()


if ROLE == "esound-identity":

    @mcp.tool()
    async def delete_account(email: str) -> dict[str, Any]:
        """SIMULATOR ONLY. Never destroys anything; mirrors the real contract.

        `deleted` is the only field that says what happened, and the no-account
        answer deliberately keeps `ok` false: the real endpoint used to answer
        a 404 with ok=true, which read as "your account has been deleted" to a
        caller trusting the generic success flag.
        """
        address = str(email or "").strip().lower()
        if address.endswith("@unknown.test"):
            return {"ok": False, "deleted": False, "reason": "no_account",
                    "email": address, "simulated": True}
        return {"ok": True, "deleted": True, "user_id": "sim-user",
                "email": address, "simulated": True, "externalMutation": False}
