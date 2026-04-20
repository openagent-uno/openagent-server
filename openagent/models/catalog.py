"""Catalog helpers for configured providers, models, and pricing.

This module deliberately keeps product-facing provider/model metadata under
OpenAgent control instead of delegating it to the runtime. Agno is used as the
execution engine for API-backed models, while OpenAgent remains the source of
truth for:

- configured providers
- enabled/disabled models
- display/runtime model ids
- pricing used for reporting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openagent.core.logging import elog

# Dedup `catalog.pricing_resolved` events so each (runtime_id, source) pair only
# logs once per process. Without this every call to compute_cost emits a row;
# at 3+ lookups per chat turn that drowns the event log.
_LOGGED_PRICING: set[tuple[str, str]] = set()

# Provider-specific defaults. Keep here so they have a single home alongside
# the rest of the provider/model catalog.
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"

# OpenAgent vocabulary (since v0.10.0):
#   - **provider**  : the model's vendor / owner (anthropic, openai, google, …).
#   - **framework** : the runtime OpenAgent dispatches through — ``agno``
#                     (direct Agno SDK against the provider's API) or
#                     ``claude-cli`` (the local ``claude`` binary wrapping
#                     Anthropic models).
#   - **model**     : the bare model id (``gpt-4o-mini``, ``claude-sonnet-4-6``).
#
# ``runtime_id`` encodes all three. Layout:
#   - ``agno``        framework (the default): ``<provider>:<model>``
#                                              (backward-compat with v0.9.x).
#   - ``claude-cli``  framework             : ``claude-cli:<provider>:<model>``
#                                              where provider is always
#                                              ``anthropic`` in practice.
#
# Rationale: most provider+model pairs only run under agno, so keeping the
# two-part form for them keeps existing usage_log rows and user references
# valid. Only claude-cli entries gain the three-part form, which is the
# minimum needed to distinguish "anthropic via Agno API" from "anthropic
# via Claude CLI subscription".
SUPPORTED_PROVIDERS = [
    "anthropic",
    "openai",
    "google",
    "openrouter",
    "groq",
    "mistral",
    "xai",
    "deepseek",
    "cerebras",
    "zai",
    "local",
]

FRAMEWORK_AGNO = "agno"
FRAMEWORK_CLAUDE_CLI = "claude-cli"
SUPPORTED_FRAMEWORKS = (FRAMEWORK_AGNO, FRAMEWORK_CLAUDE_CLI)


@dataclass(frozen=True)
class CatalogModel:
    provider: str
    model_id: str
    runtime_id: str
    history_mode: str
    framework: str = FRAMEWORK_AGNO
    disabled: bool = False
    display_name: str | None = None
    tier_hint: str | None = None
    metadata: dict[str, Any] | None = None
    # v0.12 fields — the provider's surrogate id and (name, framework)
    # pair resolved at hydration time so routing doesn't have to re-split
    # the runtime_id back apart. ``provider_id = 0`` indicates a seed
    # entry from yaml routing hints (no backing provider row yet).
    provider_id: int = 0
    # When True, SmartRouter uses this row as its classifier model.
    # At most one row should carry the flag; the router picks the first
    # flagged entry from the catalog on turn 1 of a fresh session, and
    # falls back to the first enabled entry when no row is flagged.
    is_classifier: bool = False


def _entry_model_id(entry: Any) -> str:
    """Extract the bare vendor model id from a catalog entry.

    Entries come from two shapes:
    - v0.12 DB rows: ``{"id": 10, "model": "gpt-4o-mini", …}`` —
      ``id`` is the surrogate DB primary key (int); ``model`` is the id.
    - Legacy yaml / tests: ``{"id": "gpt-4o-mini"}`` or bare string.

    Prefer ``model`` / ``model_id`` first so DB rows resolve correctly.
    Only fall back to ``id`` when it's a string (legacy shape).
    """
    if isinstance(entry, dict):
        for key in ("model", "model_id"):
            value = entry.get(key)
            if value:
                return str(value).strip()
        raw_id = entry.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            return raw_id.strip()
        return ""
    return str(entry or "").strip()


def _entry_metadata(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return dict(entry)
    return {}


def build_runtime_model_id(
    provider_name: str,
    model_id: str,
    framework: str = FRAMEWORK_AGNO,
) -> str:
    """Canonical runtime_id for a (framework, provider, model) triple.

    Agno entries produce ``<provider>:<model>`` (preserved from v0.9.x).
    Claude-CLI entries produce ``claude-cli:<provider>:<model>`` — and
    if the caller passed ``provider=claude-cli`` (pre-v0.10 vocabulary)
    we treat it as legacy shorthand for ``provider=anthropic,
    framework=claude-cli``.
    """
    raw = str(model_id or "").strip()
    if not raw:
        return raw

    # Legacy input: user-written ``claude-cli/<model>`` or ``claude-cli:<...>``.
    if raw.startswith("claude-cli/"):
        _, rest = raw.split("/", 1)
        # If the tail already has a provider prefix, keep it; else assume anthropic.
        if ":" in rest:
            prov, model = rest.split(":", 1)
            return f"{FRAMEWORK_CLAUDE_CLI}:{prov}:{model}"
        return f"{FRAMEWORK_CLAUDE_CLI}:anthropic:{rest}"
    if raw.startswith(f"{FRAMEWORK_CLAUDE_CLI}:"):
        tail = raw[len(FRAMEWORK_CLAUDE_CLI) + 1:]
        # claude-cli:anthropic:model already canonical.
        if tail.count(":") >= 1:
            return raw
        # claude-cli:model → assume anthropic.
        return f"{FRAMEWORK_CLAUDE_CLI}:anthropic:{tail}"
    if framework == FRAMEWORK_CLAUDE_CLI:
        effective_provider = provider_name or "anthropic"
        if effective_provider == FRAMEWORK_CLAUDE_CLI:
            effective_provider = "anthropic"
        return f"{FRAMEWORK_CLAUDE_CLI}:{effective_provider}:{raw}"

    # Agno framework — legacy 2-part form.
    if provider_name == FRAMEWORK_CLAUDE_CLI:
        # Caller passed the deprecated pseudo-provider. Treat as framework hint.
        return f"{FRAMEWORK_CLAUDE_CLI}:anthropic:{raw}"
    if ":" in raw:
        return raw
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        return f"{prefix}:{rest}"
    return f"{provider_name}:{raw}"


def normalize_runtime_model_id(model_ref: str, providers_config: Any = None) -> str:
    raw = str(model_ref or "").strip()
    if not raw:
        return raw
    if is_claude_cli_model(raw):
        return raw
    if ":" in raw:
        return raw
    configured_names = _configured_provider_names(providers_config)
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        if prefix == FRAMEWORK_CLAUDE_CLI:
            return raw
        if prefix in SUPPORTED_PROVIDERS or prefix in configured_names:
            return f"{prefix}:{rest}"
        return raw
    # Try to resolve a bare model id by scanning configured entries.
    for entry in _iter_provider_entries(providers_config):
        provider_name = str(entry.get("name") or "").strip()
        if not provider_name:
            continue
        for raw_model in entry.get("models") or []:
            if _entry_model_id(raw_model) == raw:
                return build_runtime_model_id(
                    provider_name, raw, entry.get("framework") or FRAMEWORK_AGNO,
                )
    return raw


def _iter_provider_entries(providers_config: Any) -> list[dict[str, Any]]:
    """Yield a list of provider dicts regardless of the config shape.

    Accepts the v0.12 flat list, the pre-v0.12 name-keyed dict (including
    the special ``claude-cli`` bucket), or ``None``.
    """
    if providers_config is None:
        return []
    if isinstance(providers_config, list):
        return [e for e in providers_config if isinstance(e, dict)]
    if isinstance(providers_config, dict):
        out: list[dict[str, Any]] = []
        for name, cfg in providers_config.items():
            if not isinstance(cfg, dict):
                continue
            if name == FRAMEWORK_CLAUDE_CLI:
                out.append({
                    "name": "anthropic",
                    "framework": FRAMEWORK_CLAUDE_CLI,
                    **cfg,
                })
            else:
                out.append({
                    "name": name,
                    "framework": cfg.get("framework") or FRAMEWORK_AGNO,
                    **cfg,
                })
        return out
    return []


def _configured_provider_names(providers_config: Any) -> set[str]:
    """Return the set of provider names visible in ``providers_config``.

    Works against both the flat-list and the legacy dict shape.
    """
    names: set[str] = set()
    for entry in _iter_provider_entries(providers_config):
        name = str(entry.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def is_claude_cli_model(model_ref: str | None) -> bool:
    """True when ``model_ref`` is dispatched via the claude-cli framework.

    Matches both the v0.10 canonical form (``claude-cli:<provider>:<model>``)
    AND the legacy pre-v0.10 forms (``claude-cli``, ``claude-cli/<model>``).
    """
    raw = str(model_ref or "").strip()
    return (
        raw == FRAMEWORK_CLAUDE_CLI
        or raw.startswith(f"{FRAMEWORK_CLAUDE_CLI}:")
        or raw.startswith(f"{FRAMEWORK_CLAUDE_CLI}/")
    )


def framework_of(model_ref: str | None) -> str:
    """``"claude-cli"`` when the ref belongs to that framework, else ``"agno"``."""
    return FRAMEWORK_CLAUDE_CLI if is_claude_cli_model(model_ref) else FRAMEWORK_AGNO


def claude_cli_model_spec(model_id: str | None = None) -> str:
    """Build the canonical claude-cli runtime_id from a bare model id.

    Legacy callers (pre-v0.10) received ``claude-cli/<id>``; the new
    canonical form is ``claude-cli:anthropic:<id>``. Both are accepted
    downstream by ``is_claude_cli_model`` / ``split_runtime_id``; this
    helper emits the new form.
    """
    raw = str(model_id or "").strip()
    if not raw:
        return FRAMEWORK_CLAUDE_CLI
    return f"{FRAMEWORK_CLAUDE_CLI}:anthropic:{raw}"


def split_runtime_id(runtime_id: str) -> tuple[str, str]:
    """Split a runtime id into ``(provider, model_id)`` for billing / display.

    v0.10 forms:
      - ``<provider>:<model>``                  → (provider, model)
      - ``claude-cli:<provider>:<model>``       → (provider, model)
    Legacy forms still accepted:
      - ``claude-cli/<model>``                  → ("claude-cli", model)
      - ``claude-cli``                          → ("claude-cli", "claude-cli")
      - bare ``<id>``                           → (id, id)
    """
    if runtime_id.startswith(f"{FRAMEWORK_CLAUDE_CLI}:"):
        tail = runtime_id[len(FRAMEWORK_CLAUDE_CLI) + 1:]
        if ":" in tail:
            provider, model_id = tail.split(":", 1)
            return provider, model_id
        # claude-cli:<model> — legacy, assume anthropic.
        return "anthropic", tail
    if ":" in runtime_id:
        provider, model_id = runtime_id.split(":", 1)
        return provider, model_id
    if "/" in runtime_id:
        provider, model_id = runtime_id.split("/", 1)
        return provider, model_id
    return runtime_id, runtime_id


def model_id_from_runtime(runtime_id: str) -> str:
    """Return just the model id portion of a runtime id (no provider prefix)."""
    return split_runtime_id(runtime_id)[1]


def model_history_mode(model_ref: str, providers_config: Any = None) -> str:
    runtime_id = normalize_runtime_model_id(model_ref, providers_config)
    if is_claude_cli_model(runtime_id):
        return "provider"
    return "platform"


def iter_configured_models(
    providers_config: Any,
    *,
    include_disabled: bool = False,
    history_mode: str | None = None,
) -> list[CatalogModel]:
    """Flatten the providers_config into :class:`CatalogModel` records.

    v0.12 shape (preferred) — a flat ``list[dict]`` of provider entries:

    .. code-block:: python

        [
          {"id": 1, "name": "openai", "framework": "agno",
           "api_key": "sk-…", "base_url": None, "enabled": True,
           "models": [{"id": 10, "model": "gpt-4o-mini", …}, …]},
          {"id": 2, "name": "anthropic", "framework": "claude-cli",
           "api_key": None, "models": [{"id": 7, "model": "claude-opus-4-7"}]},
        ]

    Legacy shape (accepted for back-compat with yaml seed / old tests) —
    a ``dict`` keyed by provider name, with a special ``claude-cli``
    bucket treated as framework=claude-cli/provider=anthropic.
    """
    results: list[CatalogModel] = []
    seen: set[str] = set()

    if providers_config is None:
        return results

    normalised: list[dict[str, Any]]
    if isinstance(providers_config, list):
        normalised = [dict(entry) for entry in providers_config if isinstance(entry, dict)]
    elif isinstance(providers_config, dict):
        normalised = []
        for provider_name, cfg in providers_config.items():
            if not isinstance(cfg, dict):
                continue
            if provider_name == FRAMEWORK_CLAUDE_CLI:
                entry_name = "anthropic"
                entry_framework = FRAMEWORK_CLAUDE_CLI
            else:
                entry_name = provider_name
                entry_framework = cfg.get("framework") or FRAMEWORK_AGNO
            normalised.append({
                "id": cfg.get("id") or 0,
                "name": entry_name,
                "framework": entry_framework,
                "api_key": cfg.get("api_key"),
                "base_url": cfg.get("base_url"),
                "enabled": cfg.get("enabled", True),
                "models": list(cfg.get("models") or []),
                "disabled_models": list(cfg.get("disabled_models") or []),
            })
    else:
        return results

    for entry in normalised:
        provider_name = str(entry.get("name") or "").strip()
        provider_framework = entry.get("framework") or FRAMEWORK_AGNO
        if not provider_name:
            continue
        if entry.get("enabled") is False:
            # A disabled provider's models should never appear in the
            # routing catalog — the router uses enabled_only hydration in
            # normal operation, but the dict-shape back-compat path may
            # feed us stale rows during boot.
            continue
        provider_id = int(entry.get("id") or 0)
        disabled = {
            str(item).strip()
            for item in (entry.get("disabled_models") or [])
        }

        for raw_model in entry.get("models") or []:
            model_id = _entry_model_id(raw_model)
            if not model_id:
                continue
            model_metadata = _entry_metadata(raw_model)
            model_enabled = model_metadata.get("enabled", True)
            is_disabled = (not bool(model_enabled)) or (model_id in disabled)
            if is_disabled and not include_disabled:
                continue

            runtime_id = build_runtime_model_id(
                provider_name, model_id, provider_framework,
            )
            mode = (
                "provider"
                if provider_framework == FRAMEWORK_CLAUDE_CLI
                else "platform"
            )
            if history_mode and mode != history_mode:
                continue
            if runtime_id in seen:
                continue
            seen.add(runtime_id)

            tier_hint = model_metadata.get("tier_hint")
            display_name = (
                model_metadata.get("display_name")
                or model_metadata.get("name")
            )
            is_classifier = bool(model_metadata.get("is_classifier", False))
            results.append(
                CatalogModel(
                    provider=provider_name,
                    model_id=model_id,
                    runtime_id=runtime_id,
                    history_mode=mode,
                    framework=provider_framework,
                    disabled=is_disabled,
                    display_name=str(display_name) if display_name else None,
                    tier_hint=str(tier_hint) if tier_hint else None,
                    metadata=model_metadata or None,
                    provider_id=provider_id,
                    is_classifier=is_classifier,
                )
            )
    return results


def supported_providers(configured: Any = None) -> list[str]:
    provider_set = set(SUPPORTED_PROVIDERS)
    provider_set.update(_configured_provider_names(configured))
    return sorted(provider_set)


def get_default_model_for_provider(
    provider_name: str,
    providers_config: Any = None,
    *,
    framework: str | None = None,
) -> str | None:
    """Return the first configured runtime_id for ``provider_name``.

    When a provider is registered under both frameworks (anthropic+agno
    AND anthropic+claude-cli), pass ``framework=`` to disambiguate.
    """
    for entry in iter_configured_models(providers_config):
        if entry.provider != provider_name:
            continue
        if framework and entry.framework != framework:
            continue
        return entry.runtime_id
    return None


def get_model_pricing(model_ref: str, providers_config: dict | None = None) -> dict[str, float]:
    """Return ``{input_cost_per_million, output_cost_per_million}`` for a model.

    Lookup order (live, never stale):
      1. claude-cli models → zero (Claude Pro/Max subscription, not per-token).
      2. OpenRouter in-process cache — primed lazily on first miss so the
         next call hits warm cache.
      3. Zero pricing (logged as "missing") if OpenRouter is unreachable.

    Always returns a dict; never raises. ``providers_config`` is accepted
    for backward compat with callers that still pass it but is no longer
    consulted for pricing — the DB / yaml never carry authoritative cost
    anymore. Emits ``catalog.pricing_resolved`` so zero-cost events can
    be alerted on.
    """
    runtime_id = normalize_runtime_model_id(model_ref, providers_config)

    # 1. claude-cli is the local subprocess wrapping the user's Claude
    # Pro/Max subscription — no per-token billing, ever. Short-circuit
    # before any lookup so we don't accidentally attribute Anthropic API
    # pricing to a claude-cli session.
    if is_claude_cli_model(runtime_id):
        _log_pricing(model_ref, runtime_id, "claude_cli_subscription", 0.0, 0.0)
        return {"input_cost_per_million": 0.0, "output_cost_per_million": 0.0}

    # 2. Online catalog (OpenRouter). Resolved from a process-wide cache
    # populated by discovery.py; never blocks — returns None on cache miss.
    online = _openrouter_pricing_lookup(runtime_id)
    if online is not None:
        _log_pricing(
            model_ref, runtime_id, "openrouter",
            online["input_cost_per_million"], online["output_cost_per_million"],
        )
        return online

    # 2b. Cache miss — fire-and-forget a prime so the next lookup hits.
    # Doesn't block the current turn; we still return zero this time.
    _maybe_prime_openrouter_cache()

    # 3. Nothing — log so ops can alert on persistently-zero entries.
    _log_pricing(model_ref, runtime_id, "missing", 0.0, 0.0)
    return {"input_cost_per_million": 0.0, "output_cost_per_million": 0.0}


def _maybe_prime_openrouter_cache() -> None:
    """Schedule a background fetch of OpenRouter's catalog if cache is cold.

    Pricing lookups are sync, but the fetch is async; we hop into the
    running loop (when there is one) and fire-and-forget. The first call
    after process start returns zero; subsequent calls — once the prime
    lands ~1 s later — get live cost.
    """
    try:
        import asyncio

        from openagent.models import discovery
    except ImportError:
        return
    cache = getattr(discovery, "_OPENROUTER_CACHE", None)
    import time as _time
    if cache and _time.time() - cache[0] < discovery._CACHE_TTL_SECONDS:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _prime() -> None:
        try:
            await discovery._fetch_openrouter_catalog()
        except Exception as e:
            try:
                elog("catalog.openrouter_prime_error", level="warning", error=str(e))
            except Exception:
                pass

    loop.create_task(_prime())


def _openrouter_pricing_lookup(runtime_id: str) -> dict[str, float] | None:
    """Look up pricing for ``runtime_id`` in the OpenRouter cache.

    Reads ``discovery._OPENROUTER_CACHE`` without triggering a fetch —
    this is a hot path and must not block on network. The cache is
    primed the first time anyone hits ``/api/models/available`` or
    ``list_provider_models``; subsequent pricing lookups amortize for
    free. Returns ``None`` when the cache is empty or the model isn't
    in OpenRouter's catalog.
    """
    try:
        from openagent.models import discovery
    except ImportError:
        return None
    cache = getattr(discovery, "_OPENROUTER_CACHE", None)
    if not cache or ":" not in runtime_id:
        return None
    _ts, entries = cache
    provider, bare = runtime_id.split(":", 1)
    # Reverse the _OPENROUTER_VENDOR_MAP: our provider → OpenRouter's vendor prefix.
    want_prefix = None
    for vendor, our_name in discovery._OPENROUTER_VENDOR_MAP.items():
        if our_name == provider:
            want_prefix = vendor
            break
    if not want_prefix:
        return None
    target = f"{want_prefix}/{bare}"
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("id") or "") != target:
            continue
        pricing = entry.get("pricing") or {}
        try:
            input_cost = float(pricing.get("prompt") or 0.0) * 1_000_000
            output_cost = float(pricing.get("completion") or 0.0) * 1_000_000
        except (TypeError, ValueError):
            return None
        if input_cost <= 0 and output_cost <= 0:
            return None
        return {
            "input_cost_per_million": input_cost,
            "output_cost_per_million": output_cost,
        }
    return None


def _log_pricing(model_ref: str, runtime_id: str, source: str, input_cpm: float, output_cpm: float) -> None:
    """Log pricing resolution exactly once per ``(runtime_id, source)`` pair.

    ``compute_cost`` is a hot path (called multiple times per chat turn). Without
    deduplication the event log fills up with identical resolution rows. The
    first lookup confirms wiring; subsequent lookups for the same model+source
    add no information.
    """
    key = (runtime_id, source)
    if key in _LOGGED_PRICING:
        return
    _LOGGED_PRICING.add(key)
    try:
        elog(
            "catalog.pricing_resolved",
            model=model_ref,
            runtime_id=runtime_id,
            source=source,
            input_cost_per_million=input_cpm,
            output_cost_per_million=output_cpm,
        )
    except Exception:
        # Logging must never break a hot-path lookup.
        pass


def compute_cost(model_ref: str, input_tokens: int, output_tokens: int) -> float:
    pricing = get_model_pricing(model_ref)
    return (
        (pricing["input_cost_per_million"] * max(0, input_tokens)) / 1_000_000
        + (pricing["output_cost_per_million"] * max(0, output_tokens)) / 1_000_000
    )
