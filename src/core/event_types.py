"""Webhook "type" presets for the Events channel.

The ``type`` on an event picks how an inbound HTTP delivery is authenticated
beyond the per-event secret, and how a provider delivery id is extracted for
idempotent de-duplication. Kept as plain data (plus a couple of stdlib-only
verifier helpers) so both the REST layer (``GET /api/event-types`` powers the
app's type picker) and the webhook-auth path share one source of truth.

Every type ALSO accepts the per-event secret directly (header
``X-OpenAgent-Event-Secret`` or ``Authorization: Bearer``), so a plain script
works against any event regardless of type. The provider presets add a second,
stronger check (HMAC over the raw body) that a real GitHub/Stripe/Slack sender
satisfies automatically.
"""

from __future__ import annotations

from typing import Any

# key → metadata shown to the UI and consumed by webhook_auth.
EVENT_TYPES: dict[str, dict[str, Any]] = {
    "generic": {
        "label": "Generic",
        "description": (
            "Any service that can send an HTTP POST with a secret header. "
            "Auth = the per-event secret in X-OpenAgent-Event-Secret or "
            "Authorization: Bearer."
        ),
        "signature": None,
        "external_id_header": None,
        "docs": "curl -X POST <url> -H 'X-OpenAgent-Event-Secret: <secret>' -d '{...}'",
    },
    "generic-hmac": {
        "label": "Generic HMAC",
        "description": (
            "Sender computes HMAC-SHA256 of the raw body with the secret and "
            "sends it in X-Signature-256 as 'sha256=<hex>'. Adds body-integrity "
            "on top of the shared secret."
        ),
        "signature": "hmac_sha256",
        "signature_header": "X-Signature-256",
        "signature_prefix": "sha256=",
        "external_id_header": "X-Delivery-Id",
        "docs": "sig = 'sha256=' + hmac_sha256(secret, raw_body)",
    },
    "github": {
        "label": "GitHub",
        "description": (
            "GitHub webhooks. Verifies X-Hub-Signature-256 (HMAC-SHA256 of the "
            "raw body with the secret) and dedupes on X-GitHub-Delivery. Set the "
            "same secret in the repo/org webhook settings."
        ),
        "signature": "hmac_sha256",
        "signature_header": "X-Hub-Signature-256",
        "signature_prefix": "sha256=",
        "external_id_header": "X-GitHub-Delivery",
        "event_name_header": "X-GitHub-Event",
        "docs": "Repo → Settings → Webhooks → Secret = <secret>, Content-Type application/json",
    },
    "stripe": {
        "label": "Stripe",
        "description": (
            "Stripe webhooks. Verifies the Stripe-Signature header (timestamped "
            "HMAC-SHA256) with a ±5min replay window. Use the endpoint's signing "
            "secret (whsec_…) as the event secret."
        ),
        "signature": "stripe",
        "signature_header": "Stripe-Signature",
        "external_id_header": None,   # extracted from the JSON body's "id"
        "external_id_json_path": "id",
        "docs": "Stripe Dashboard → Developers → Webhooks → Signing secret = <secret>",
    },
    "slack": {
        "label": "Slack",
        "description": (
            "Slack Events API. Verifies X-Slack-Signature (v0 HMAC of "
            "'v0:{timestamp}:{body}') with a ±5min replay window. Use the app's "
            "Signing Secret as the event secret."
        ),
        "signature": "slack",
        "signature_header": "X-Slack-Signature",
        "timestamp_header": "X-Slack-Request-Timestamp",
        "external_id_header": "X-Slack-Retry-Num",
        "docs": "Slack app → Basic Information → Signing Secret = <secret>",
    },
    "replio": {
        "label": "Replio",
        "description": (
            "Replio support platform. Verifies X-Replio-Signature (HMAC-SHA256 "
            "of the raw body with the secret) and dedupes on X-Replio-Delivery-Id. "
            "Replio keeps that id stable across its four retries, so a delivery "
            "it re-sends after a timeout cannot run the event — or answer a "
            "customer — twice. Use the same secret as the Replio webhook."
        ),
        "signature": "hmac_sha256",
        "signature_header": "X-Replio-Signature",
        "signature_prefix": "sha256=",
        "external_id_header": "X-Replio-Delivery-Id",
        "event_name_header": "X-Replio-Event",
        "docs": "Replio → Webhooks → new webhook, Secret = <secret>",
    },
}

DEFAULT_TYPE = "generic"

# How long (seconds) a signed request's timestamp may differ from now, for the
# providers that timestamp their signatures (Stripe, Slack). Blocks replay of
# a captured request outside this window.
REPLAY_WINDOW_S = 300


def is_valid_type(t: str | None) -> bool:
    return (t or DEFAULT_TYPE) in EVENT_TYPES


def public_types() -> list[dict[str, Any]]:
    """The list shape returned by ``GET /api/event-types``."""
    out = []
    for key, meta in EVENT_TYPES.items():
        out.append({
            "key": key,
            "label": meta["label"],
            "description": meta["description"],
            "signed": bool(meta.get("signature")),
            "docs": meta.get("docs"),
        })
    return out
