"""Authentication + validation for inbound webhook deliveries.

Pure functions over ``(event_row, raw_body, headers)`` — no aiohttp types —
so they are unit-testable in isolation and reusable by both the dedicated
webhook listener and the REST ``/trigger`` path if needed.

Two layers, in order:
1. **Secret** — every event carries a per-event secret (salted-hash stored).
   The caller presents it in ``X-OpenAgent-Event-Secret`` or
   ``Authorization: Bearer``. This is the baseline for ``type=generic``.
2. **Signature** — provider presets (github/stripe/slack/generic-hmac) ALSO
   verify an HMAC over the *raw* body (and, for timestamped schemes, a replay
   window). A real provider satisfies this automatically; it means a leaked
   secret alone is not enough to forge a body.

All comparisons are constant-time. Failures return a reason string that is
logged (never the secret) and mapped to an identical 401 by the caller, so a
probe can't distinguish "wrong secret" from "bad signature".
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Optional

from src.core.event_secret import decrypt_secret, verify_secret
from src.core.event_types import EVENT_TYPES, REPLAY_WINDOW_S


class WebhookAuthError(Exception):
    """Raised on any auth/validation failure. ``.reason`` is safe to log."""

    def __init__(self, reason: str, *, status: int = 401):
        super().__init__(reason)
        self.reason = reason
        self.status = status


def _lower_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


def _extract_secret(h: dict[str, str]) -> Optional[str]:
    secret = h.get("x-openagent-event-secret")
    if secret:
        return secret.strip()
    auth = h.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _hmac_sha256_hex(secret: str, raw: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def _verify_hmac_header(
    *, event_secret: str, raw: bytes, h: dict[str, str], meta: dict[str, Any],
) -> None:
    header_name = meta["signature_header"].lower()
    provided = (h.get(header_name) or "").strip()
    if not provided:
        raise WebhookAuthError("missing signature header")
    prefix = meta.get("signature_prefix", "")
    expected = prefix + _hmac_sha256_hex(event_secret, raw)
    if not hmac.compare_digest(provided, expected):
        raise WebhookAuthError("signature mismatch")


def _verify_stripe(*, event_secret: str, raw: bytes, h: dict[str, str], meta: dict[str, Any]) -> None:
    header = (h.get("stripe-signature") or "").strip()
    if not header:
        raise WebhookAuthError("missing Stripe-Signature")
    parts = dict(
        p.split("=", 1) for p in header.split(",") if "=" in p
    )
    ts = parts.get("t")
    v1 = parts.get("v1")
    if not ts or not v1:
        raise WebhookAuthError("malformed Stripe-Signature")
    try:
        ts_int = int(ts)
    except ValueError:
        raise WebhookAuthError("bad Stripe timestamp")
    if abs(time.time() - ts_int) > REPLAY_WINDOW_S:
        raise WebhookAuthError("Stripe timestamp outside replay window")
    signed_payload = f"{ts}.".encode("utf-8") + raw
    expected = _hmac_sha256_hex(event_secret, signed_payload)
    if not hmac.compare_digest(v1, expected):
        raise WebhookAuthError("Stripe signature mismatch")


def _verify_slack(*, event_secret: str, raw: bytes, h: dict[str, str], meta: dict[str, Any]) -> None:
    ts = (h.get("x-slack-request-timestamp") or "").strip()
    sig = (h.get("x-slack-signature") or "").strip()
    if not ts or not sig:
        raise WebhookAuthError("missing Slack signature headers")
    try:
        ts_int = int(ts)
    except ValueError:
        raise WebhookAuthError("bad Slack timestamp")
    if abs(time.time() - ts_int) > REPLAY_WINDOW_S:
        raise WebhookAuthError("Slack timestamp outside replay window")
    basestring = b"v0:" + ts.encode("utf-8") + b":" + raw
    expected = "v0=" + _hmac_sha256_hex(event_secret, basestring)
    if not hmac.compare_digest(sig, expected):
        raise WebhookAuthError("Slack signature mismatch")


def authenticate(
    *,
    event: dict[str, Any],
    raw_body: bytes,
    headers: dict[str, str],
    db_path: Optional[str],
) -> None:
    """Raise ``WebhookAuthError`` unless the request is authentic for ``event``.

    ``event`` must include the encrypted ``secret_enc`` (fetch it with
    ``db.get_event(..., include_secret=True)``); ``db_path`` locates the
    encryption key so the secret can be recovered for HMAC verification. Never
    call this with a public event dict — it fails closed, which is safe but
    useless.
    """
    h = _lower_headers(headers)
    secret_enc = event.get("secret_enc")
    etype = event.get("type") or "generic"
    meta = EVENT_TYPES.get(etype, EVENT_TYPES["generic"])
    scheme = meta.get("signature")

    presented = _extract_secret(h)

    # Recover the clear secret once (needed as the HMAC key for signed types,
    # and as the compare target for the raw-secret path).
    clear_secret: Optional[str] = None
    if secret_enc:
        try:
            clear_secret = decrypt_secret(secret_enc, db_path=db_path)
        except Exception:  # noqa: BLE001
            clear_secret = None

    # generic: the presented secret must match. Signed types: the HMAC IS the
    # proof of the secret, so a raw-secret header is optional (a real provider
    # won't send one) — but if one is presented it must be correct.
    if scheme is None:
        if not presented:
            raise WebhookAuthError("missing secret")
        if not verify_secret(presented, secret_enc=secret_enc, db_path=db_path):
            raise WebhookAuthError("bad secret")
        return

    if presented and not verify_secret(
        presented, secret_enc=secret_enc, db_path=db_path
    ):
        raise WebhookAuthError("bad secret")

    if not clear_secret:
        # Can't recover the HMAC key → cannot verify a provider signature.
        raise WebhookAuthError("event not configured for signature verification")

    if scheme == "hmac_sha256":
        _verify_hmac_header(event_secret=clear_secret, raw=raw_body, h=h, meta=meta)
    elif scheme == "stripe":
        _verify_stripe(event_secret=clear_secret, raw=raw_body, h=h, meta=meta)
    elif scheme == "slack":
        _verify_slack(event_secret=clear_secret, raw=raw_body, h=h, meta=meta)
    else:
        raise WebhookAuthError(f"unknown signature scheme {scheme!r}")


def extract_external_id(
    *, event: dict[str, Any], headers: dict[str, str], payload: dict[str, Any],
) -> Optional[str]:
    """Provider delivery id for idempotent de-dupe, per the event type."""
    h = _lower_headers(headers)
    meta = EVENT_TYPES.get(event.get("type") or "generic", EVENT_TYPES["generic"])
    hdr = meta.get("external_id_header")
    if hdr:
        val = h.get(hdr.lower())
        if val:
            return val.strip()
    json_path = meta.get("external_id_json_path")
    if json_path and isinstance(payload, dict):
        val = payload.get(json_path)
        if isinstance(val, (str, int)):
            return str(val)
    return None
