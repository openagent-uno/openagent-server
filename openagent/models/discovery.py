"""Dynamic LLM catalog per provider.

Given a user's configured API keys, return the list of models that
provider actually exposes. The returned model ids are the bare
provider-side values (``gpt-4o-mini``, ``claude-sonnet-4-6``, …) — the
caller wraps them with ``catalog.build_runtime_model_id`` to produce
the canonical ``openai:gpt-4o-mini`` form used in the ``models`` table.

Source chain (highest priority first):

  1. **Provider's own /v1/models** with the user's API key. Accurate
     and authoritative for that account. Anthropic, OpenAI, Groq,
     Mistral, xAI, DeepSeek, Cerebras, ZAI, OpenRouter all speak
     OpenAI-compatible envelopes; Google uses a query-param scheme.
  2. **OpenRouter's unauthenticated catalog** at
     ``https://openrouter.ai/api/v1/models``. Every major vendor's
     model prefixed ``<vendor>/<id>``. Free, no key needed, updated
     continuously. Also carries pricing so a cache hit on this tier
     primes ``catalog._openrouter_pricing_lookup`` for cost reporting.

No bundled offline fallback: OpenRouter is reachable from any network
that can reach the provider APIs, and stale offline data causes more
confusion than a clean empty list. Errors are absorbed and logged via
``elog``; cached in-process with a 10-minute TTL.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from openagent.core.logging import elog


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

# Bundled catalogs for audio-only vendors that don't expose a public
# /v1/models endpoint. These models are picked by the same Add Model
# flow as LLM rows — the catalog seeds the dropdown so the user never
# types a model id by hand.
_AUDIO_BUNDLED: dict[str, list[dict[str, Any]]] = {
    "elevenlabs": [
        {"id": "eleven_flash_v2_5", "display_name": "ElevenLabs Flash v2.5"},
        {"id": "eleven_turbo_v2_5", "display_name": "ElevenLabs Turbo v2.5"},
        {"id": "eleven_multilingual_v2", "display_name": "ElevenLabs Multilingual v2"},
    ],
    "deepgram": [
        {"id": "nova-3", "display_name": "Deepgram Nova 3"},
        {"id": "nova-2", "display_name": "Deepgram Nova 2"},
    ],
}


def _kind_for_model(provider: str, model_id: str) -> str:
    """Heuristic: classify a model as ``llm`` / ``tts`` / ``stt`` from id.

    Audio-only vendors get a flat label; otherwise we pattern-match on
    the model id so OpenAI's /v1/models response gets correctly split
    into chat (``gpt-*``), TTS (``tts-1``, ``gpt-4o-mini-tts``), and
    STT (``whisper-1``, ``gpt-4o-transcribe``).
    """
    p = provider.lower()
    m = model_id.lower()
    if p == "elevenlabs":
        return "tts"
    if p == "deepgram":
        return "stt"
    # Order matters: ``gpt-4o-mini-tts`` contains ``gpt`` but is TTS,
    # ``gpt-4o-transcribe`` contains ``gpt`` but is STT.
    if "tts" in m or m.startswith("eleven_") or m == "playai-tts":
        return "tts"
    if "whisper" in m or "transcribe" in m or m.startswith("nova-"):
        return "stt"
    return "llm"

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
    """Extract ``[{id, display_name}]`` from any reasonable envelope.

    Accepts:
      * ``{"data": [{"id": "..."}]}``           — OpenAI canonical
      * ``{"models": [{"id": "..."}]}``         — z.ai, some self-hosts
      * ``[{"id": "..."}]``                     — bare list (OpenRouter legacy)
      * ``{"data": [{"model": "..."}]}``        — a few compatibility layers

    Robust envelope handling matters because OpenAI-compatible ≠ OpenAI-
    identical: z.ai returns ``models``, some providers return a bare
    array. Returning ``[]`` on an unknown shape means the UI silently
    surfaces nothing, which we debugged by hand once already.
    """
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("data") or payload.get("models") or []
    else:
        return []
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get("id") or entry.get("model") or entry.get("name") or "").strip()
        if not mid:
            continue
        out.append({"id": mid, "display_name": str(entry.get("name") or entry.get("display_name") or mid)})
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
        # the ``<runtime_id>:<pricing>`` format everything else uses.
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


async def list_provider_models(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Return ``[{id, display_name, input_cost_per_million, output_cost_per_million}]``.

    Always returns a list — never raises. Source priority: configured
    key → OpenRouter cross-vendor catalog → empty list.
    """
    provider = provider.lower().strip()
    if not provider:
        return []

    # Audio-only vendors short-circuit to the bundled catalog — they
    # don't expose a discovery endpoint and the model set is small and
    # stable enough that hand-curating beats live fetch.
    if provider in _AUDIO_BUNDLED:
        return _tagged(provider, [dict(e) for e in _AUDIO_BUNDLED[provider]])

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
                tagged = _tagged(provider, result)
                _CACHE[_cache_key(provider, api_key)] = (time.time(), list(tagged))
                elog("discovery.live", provider=provider, count=len(tagged), source="live")
                return tagged
            elog("discovery.live_empty", provider=provider)
        except Exception as e:  # noqa: BLE001 — fall through to OpenRouter
            elog("discovery.live_error", provider=provider, error=str(e))

    # OpenRouter cross-vendor catalog — free, unauthenticated, live.
    try:
        entries = await _fetch_openrouter_catalog()
        or_models = _openrouter_filter_for(provider, entries)
        elog("discovery.openrouter", provider=provider, count=len(or_models))
        return _tagged(provider, or_models)
    except Exception as e:  # noqa: BLE001 — network/DNS/etc.
        elog("discovery.openrouter_error", provider=provider, error=str(e))
        return []


def _tagged(provider: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each entry with its inferred ``kind`` in place."""
    for e in entries:
        e["kind"] = _kind_for_model(provider, str(e.get("id") or ""))
    return entries


