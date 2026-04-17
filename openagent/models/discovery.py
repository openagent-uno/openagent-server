"""Dynamic LLM catalog per provider.

Given a user's configured API keys, return the list of models that
provider actually exposes. The returned model ids are the bare
provider-side values (``gpt-4o-mini``, ``claude-sonnet-4-6``, …) — the
caller wraps them with ``catalog.build_runtime_model_id`` to produce
the canonical ``openai:gpt-4o-mini`` / ``claude-cli/<id>`` forms used
in the ``models`` table.

Source chain (highest priority first):

  1. **Provider's own /v1/models** with the user's API key. Accurate
     and authoritative for that account. Anthropic, OpenAI, Groq,
     Mistral, xAI, DeepSeek, Cerebras, ZAI, OpenRouter all speak the
     same pattern; Google uses a query-param scheme.
  2. **OpenRouter's unauthenticated catalog** at
     ``https://openrouter.ai/api/v1/models``. Returns every model
     every major provider supports (prefixed like ``openai/gpt-4o``,
     ``anthropic/claude-sonnet-4.5``). Extracting a provider's list is
     a prefix filter. Free, no key needed, updated continuously.
  3. **Bundled fallback** derived from ``default_pricing.json`` — the
     same table that drives cost reporting. Used when offline / the
     live fetches error. Display names = bare ids.

Errors are absorbed — the gateway UX is "show what you can" — and
logged via ``elog`` so operators can diagnose key problems. Each tier
cached in-process with a 10-minute TTL.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from openagent.core.logging import elog
from openagent.models.catalog import _load_default_pricing


# Per-provider discovery endpoint config. Every OpenAI-compatible
# provider shares the same shape. Each tuple is ``(base_url, path)``.
_OPENAI_STYLE: dict[str, tuple[str, str]] = {
    "openai":     ("https://api.openai.com",            "/v1/models"),
    "anthropic":  ("https://api.anthropic.com",         "/v1/models"),
    "groq":       ("https://api.groq.com/openai",       "/v1/models"),
    "mistral":    ("https://api.mistral.ai",            "/v1/models"),
    "xai":        ("https://api.x.ai",                  "/v1/models"),
    "deepseek":   ("https://api.deepseek.com",          "/v1/models"),
    "cerebras":   ("https://api.cerebras.ai",           "/v1/models"),
    "zai":        ("https://api.z.ai/api/paas/v4",      "/models"),
    "openrouter": ("https://openrouter.ai/api",         "/v1/models"),
}

_GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
_OPENROUTER_CATALOG = "https://openrouter.ai/api/v1/models"
_ANTHROPIC_VERSION = "2023-06-01"

# OpenRouter prefixes each model with ``<vendor>/<id>``. Map vendor →
# our provider key so we can filter OpenRouter's response. ``openai``,
# ``anthropic``, ``google``, … match 1:1 except for a few edge cases
# (xai ↔ x-ai).
_OPENROUTER_VENDOR_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "mistralai": "mistral",
    "meta-llama": "groq",   # OpenRouter's meta-llama models run on Groq hosts we wrap
    "x-ai": "xai",
    "deepseek": "deepseek",
    "cerebras": "cerebras",
}

_CACHE_TTL_SECONDS = 600
_CACHE: dict[tuple[str, str], tuple[float, list[dict[str, Any]]]] = {}
# Separate cache entry for the OpenRouter catalog — it's the same for
# every provider lookup, so one fetch serves all.
_OPENROUTER_CACHE: tuple[float, list[dict[str, Any]]] | None = None


def _cache_key(provider: str, api_key: str) -> tuple[str, str]:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12] if api_key else ""
    return (provider, digest)


def _parse_openai_style(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    out: list[dict[str, Any]] = []
    for entry in payload.get("data") or []:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get("id") or "").strip()
        if not mid:
            continue
        out.append({"id": mid, "display_name": str(entry.get("name") or mid)})
    return out


def _parse_google(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    out: list[dict[str, Any]] = []
    for entry in payload.get("models") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if name.startswith("models/"):
            name = name[len("models/"):]
        if not name:
            continue
        out.append({
            "id": name,
            "display_name": str(entry.get("displayName") or name),
        })
    return out


async def _fetch_openai_style(
    provider: str, api_key: str, base_url: str | None
) -> list[dict[str, Any]]:
    import aiohttp

    base, path = _OPENAI_STYLE[provider]
    url = (base_url.rstrip("/") if base_url else base.rstrip("/")) + path
    if provider == "anthropic":
        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "accept": "application/json",
        }
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
        }
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"{provider} /v1/models returned {resp.status}")
            payload = await resp.json(content_type=None)
    return _parse_openai_style(payload)


async def _fetch_google(api_key: str) -> list[dict[str, Any]]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=10)
    url = f"{_GOOGLE_ENDPOINT}?key={api_key}"
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"google list models returned {resp.status}")
            payload = await resp.json(content_type=None)
    return _parse_google(payload)


async def _fetch_openrouter_catalog() -> list[dict[str, Any]]:
    """Hit OpenRouter's unauthenticated catalog. Returns raw entries.

    Cached in process for ``_CACHE_TTL_SECONDS`` because the payload is
    large (~300 models) and every per-provider lookup shares it.
    """
    global _OPENROUTER_CACHE
    if _OPENROUTER_CACHE and time.time() - _OPENROUTER_CACHE[0] < _CACHE_TTL_SECONDS:
        return list(_OPENROUTER_CACHE[1])
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(_OPENROUTER_CATALOG) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"openrouter catalog returned {resp.status}")
            payload = await resp.json(content_type=None)
    entries = payload.get("data") if isinstance(payload, dict) else payload
    entries = entries or []
    _OPENROUTER_CACHE = (time.time(), list(entries))
    return entries


def _openrouter_filter_for(
    provider: str, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extract bare-id models for ``provider`` from OpenRouter's catalog.

    OpenRouter ids look like ``<vendor>/<model>`` — strip the vendor
    prefix. Pricing (USD per token) is carried through so callers can
    surface it. Unknown vendors are skipped silently.
    """
    want_vendor = None
    for vendor, our_name in _OPENROUTER_VENDOR_MAP.items():
        if our_name == provider:
            want_vendor = vendor
            break
    if not want_vendor:
        return []
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_id = str(entry.get("id") or "").strip()
        if "/" not in raw_id:
            continue
        vendor, bare = raw_id.split("/", 1)
        if vendor != want_vendor:
            continue
        pricing = entry.get("pricing") or {}
        # OpenRouter reports $/token; convert to $/million for parity with
        # default_pricing.json.
        try:
            input_cost = float(pricing.get("prompt") or 0.0) * 1_000_000
            output_cost = float(pricing.get("completion") or 0.0) * 1_000_000
        except (TypeError, ValueError):
            input_cost = output_cost = 0.0
        out.append({
            "id": bare,
            "display_name": str(entry.get("name") or bare),
            "input_cost_per_million": input_cost or None,
            "output_cost_per_million": output_cost or None,
        })
    return out


def _bundled_fallback(provider: str) -> list[dict[str, Any]]:
    """Last-resort list derived from ``default_pricing.json`` keys.

    We already ship a pricing table with ``<provider>:<model>`` keys;
    reusing it avoids maintaining a second bundled file. Display name
    is the bare id since the pricing JSON has no human-readable names.
    """
    pricing = _load_default_pricing()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    prefix = f"{provider}:"
    for key, info in pricing.items():
        if not key.startswith(prefix):
            continue
        bare = key[len(prefix):]
        if bare in seen:
            continue
        seen.add(bare)
        out.append({
            "id": bare,
            "display_name": bare,
            "input_cost_per_million": info.get("input_cost_per_million") or None,
            "output_cost_per_million": info.get("output_cost_per_million") or None,
        })
    return out


async def list_provider_models(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Return ``[{id, display_name, input_cost_per_million, output_cost_per_million}]``.

    Always returns a list — never raises. Source priority: configured
    key → OpenRouter → bundled pricing table.
    """
    provider = provider.lower().strip()
    if not provider:
        return []

    if api_key:
        cached = _CACHE.get(_cache_key(provider, api_key))
        if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
            return list(cached[1])
        try:
            if provider == "google":
                result = await _fetch_google(api_key)
            elif provider in _OPENAI_STYLE:
                result = await _fetch_openai_style(provider, api_key, base_url)
            else:
                result = []
            if result:
                _CACHE[_cache_key(provider, api_key)] = (time.time(), list(result))
                elog("discovery.live", provider=provider, count=len(result), source="live")
                return result
            elog("discovery.live_empty", provider=provider)
        except Exception as e:  # noqa: BLE001 — fall through to OpenRouter
            elog("discovery.live_error", provider=provider, error=str(e))

    # OpenRouter is our dynamic cross-provider catalog. It covers most
    # vendors we care about and carries pricing too, so it beats the
    # static fallback.
    try:
        entries = await _fetch_openrouter_catalog()
        or_models = _openrouter_filter_for(provider, entries)
        if or_models:
            elog("discovery.openrouter", provider=provider, count=len(or_models))
            return or_models
    except Exception as e:  # noqa: BLE001 — network errors; fall back
        elog("discovery.openrouter_error", provider=provider, error=str(e))

    fallback = _bundled_fallback(provider)
    elog("discovery.bundled", provider=provider, count=len(fallback))
    return fallback


async def list_provider_models_cached(provider: str) -> list[dict[str, Any]]:
    """Provider discovery that reads the live openagent.yaml for the key."""
    import os
    from openagent.core.config import load_config
    try:
        providers_cfg = load_config(os.environ.get("OPENAGENT_CONFIG_PATH")).get("providers", {}) or {}
    except (FileNotFoundError, PermissionError, OSError):
        providers_cfg = {}
    except Exception as e:  # noqa: BLE001 — yaml errors, bad env refs, etc.
        elog("discovery.load_config_error", level="warning", error=str(e))
        providers_cfg = {}
    cfg = providers_cfg.get(provider, {}) or {}
    return await list_provider_models(
        provider, api_key=cfg.get("api_key") or None, base_url=cfg.get("base_url") or None,
    )
