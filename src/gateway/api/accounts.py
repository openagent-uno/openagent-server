"""Accounts REST API — who is actually serving this agent's models.

GET /api/accounts  → { "providers": [ { provider, accounts: [...] } ] }

A model row says *which* model answers; it says nothing about *whose*
subscription pays for it. On a subscription-backed deployment that is the
question the operator actually has: which account is serving traffic, is it
rate-limited right now, and how much of its window is left. Nothing exposed
that, so the answer lived in `kubectl exec … curl localhost:8787/health`.

Two sources, merged behind one endpoint:

* **Subscription proxies.** A provider whose ``base_url`` points at a
  sub-proxy (Claude, Codex, anything speaking the same shape) exposes its
  accounts on ``<base>/health``. We probe it and pass the accounts through.
  Discovery is by URL, not by provider NAME — nothing here knows the word
  "codex", so a new proxy works without a code change.
* **OpenAgent's own credential pool.** When a provider declares
  ``metadata.accounts``, :mod:`src.models.credential_pool` rotates over them
  and tracks each one's health. That pool is in-process, so it is read
  directly rather than over HTTP.

The honest part: **quota is reported only when the upstream reports it.**
The Codex proxy returns a real ``used_percent`` against a real window; the
Claude proxy returns only "limited / not limited", because Anthropic does
not tell it more — it learns an account is spent by being told 429. So
``quota`` is ``null`` for those accounts and the client must say "not
reported" rather than draw a meter it invented. A fabricated headroom number
on this screen is worse than no number: it is the one an operator would plan
around.

Fail-soft throughout. A proxy that is down, slow, or not a proxy at all
yields ``reachable: false`` and an empty account list — never a 500, and
never a stall: the probes run concurrently under a tight timeout, because
this endpoint sits behind a UI that must paint.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import web

from src.gateway.api._common import gateway_db as _db

# A dead proxy must not hold the panel hostage. Short, and the probes run
# concurrently, so total latency is one timeout rather than N.
_PROBE_TIMEOUT_S = 2.5

# Fields copied out of a proxy's account entry. An explicit allowlist: the
# health payload is another program's shape and may grow fields we would
# rather not blindly relay to a client.
_ACCOUNT_FIELDS = (
    "id", "name", "priority", "plan", "managed",
    "limited", "limited_until_ms", "expires_at_ms", "has_refresh_token",
)

_QUOTA_FIELDS = (
    "plan", "active_limit",
    "primary_used_percent", "primary_window_minutes", "primary_reset_after_s",
    "secondary_used_percent", "secondary_window_minutes", "secondary_reset_after_s",
    "credits_balance",
)

# Pool metrics worth showing beside the accounts: how loaded the proxy is and
# whether it has been switching accounts (which is what rate-limiting looks
# like from the outside).
_METRIC_FIELDS = (
    "active", "queued", "rate_limited", "account_rate_limited",
    "account_switches", "upstream_errors",
)


def _health_url(base_url: str) -> str | None:
    """Turn a provider ``base_url`` into its proxy health URL.

    The base_url is an OpenAI-compatible endpoint (``…/v1``); the health
    endpoint sits at the root beside it. Only loopback/private-looking HTTP
    endpoints are probed — a provider pointing at a vendor's public API is
    not a proxy we own, and must never be poked with an unexpected request.
    """
    if not base_url or not isinstance(base_url, str):
        return None
    url = base_url.strip().rstrip("/")
    if not url.startswith("http://"):
        # A sub-proxy we host is reached over plain HTTP on the pod network.
        # An https:// base_url is somebody else's API — leave it alone.
        return None
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return f"{url}/health"


def _redact_url(url: str) -> str:
    """Strip any userinfo from a URL before it leaves the gateway.

    ``http://user:token@host/v1`` is a legal base_url and some deployments
    carry the credential there. This value is echoed to a UI, so the
    credential must not ride along — the host is the useful part.
    """
    if not url or "@" not in url:
        return url or ""
    try:
        scheme, rest = url.split("://", 1)
    except ValueError:
        return url
    if "@" not in rest:
        return url
    _userinfo, host = rest.rsplit("@", 1)
    return f"{scheme}://***@{host}"


def _pick(src: Any, fields: tuple[str, ...]) -> dict:
    if not isinstance(src, dict):
        return {}
    return {k: src[k] for k in fields if k in src}


def _normalise_account(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    out = _pick(raw, _ACCOUNT_FIELDS)
    if not out.get("id") and not out.get("name"):
        return None
    quota = _pick(raw.get("quota"), _QUOTA_FIELDS)
    # An empty dict would render as "quota known, all zeroes". Null means
    # "this upstream does not report quota", which is a different claim.
    out["quota"] = quota or None
    out["source"] = "proxy"
    return out


async def _probe(session, provider: dict) -> dict:
    """Ask one provider's proxy who is serving it. Never raises."""
    name = provider.get("name") or "?"
    base_url = provider.get("base_url") or ""
    entry = {
        "provider": name,
        "framework": provider.get("framework"),
        "base_url": _redact_url(base_url),
        "enabled": bool(provider.get("enabled", True)),
        "reachable": False,
        "accounts": [],
        "metrics": None,
        "error": None,
    }
    url = _health_url(base_url)
    if url is None:
        entry["error"] = "not a local proxy endpoint"
        return entry
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                entry["error"] = f"health returned {resp.status}"
                return entry
            payload = await resp.json(content_type=None)
    except Exception as exc:  # noqa: BLE001 — a down proxy is data, not a fault
        entry["error"] = str(exc)[:200]
        return entry

    if not isinstance(payload, dict):
        entry["error"] = "health payload was not an object"
        return entry
    raw_accounts = payload.get("accounts")
    if not isinstance(raw_accounts, list):
        # Reachable, but not a proxy that fronts accounts (a plain llama.cpp
        # health endpoint, say). Reachable is still worth reporting.
        entry["reachable"] = True
        entry["error"] = "endpoint reports no accounts"
        return entry

    entry["reachable"] = True
    entry["accounts"] = [a for a in map(_normalise_account, raw_accounts) if a]
    entry["metrics"] = _pick(payload.get("metrics"), _METRIC_FIELDS) or None
    return entry


def _pool_accounts(provider_name: str) -> list[dict]:
    """Accounts from OpenAgent's own rotation pool, if this provider has one.

    Inert on almost every deployment (a pool only exists at >= 2 configured
    accounts), so this usually returns []. Kept because when it IS configured
    it is the same question with a different answer source, and the client
    should not have to know which.
    """
    try:
        from src.models.credential_pool import _POOLS  # noqa: PLC0415

        pool = _POOLS.get(provider_name)
        if pool is None:
            return []
        return pool.snapshot()
    except Exception:  # noqa: BLE001
        return []


async def handle_list(request):
    """GET /api/accounts — the accounts serving each enabled provider."""
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "No database configured"}, status=503)

    providers = await db.list_providers()
    live = [p for p in providers if p.get("enabled")]

    import aiohttp

    timeout = aiohttp.ClientTimeout(total=_PROBE_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        entries = await asyncio.gather(
            *(_probe(session, p) for p in live), return_exceptions=True
        )

    out: list[dict] = []
    for provider, entry in zip(live, entries):
        if isinstance(entry, BaseException):
            entry = {
                "provider": provider.get("name"),
                "base_url": _redact_url(provider.get("base_url") or ""),
                "enabled": True,
                "reachable": False,
                "accounts": [],
                "metrics": None,
                "error": str(entry)[:200],
            }
        pooled = _pool_accounts(provider.get("name") or "")
        if pooled:
            entry["accounts"] = list(entry["accounts"]) + pooled
        out.append(entry)

    return web.json_response({"providers": out})
