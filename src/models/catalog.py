"""Catalog helpers for configured providers, models, and pricing.

This module deliberately keeps product-facing provider/model metadata under
OpenAgent control instead of delegating it to the inlined runtime. The runtime is the
execution engine for both LLM paths — ``api-based`` (native ``Agent``) and
``claude-cli`` (``ClaudeAgent`` adapter) — while OpenAgent remains the
source of truth for:

- configured providers
- enabled/disabled models
- display/runtime model ids
- pricing used for reporting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.logging import elog

# Dedup `catalog.pricing_resolved` events so each (runtime_id, source) pair only
# logs once per process. Without this every call to compute_cost emits a row;
# at 3+ lookups per chat turn that drowns the event log.
_LOGGED_PRICING: set[tuple[str, str]] = set()

# Provider-specific defaults. Keep here so they have a single home alongside
# the rest of the provider/model catalog.
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
# OpenAI-compatible providers wired through ``OpenAILike`` (see
# ``native_provider.RUNTIME_PROVIDER_CLASSES``). Each value is the full
# versioned API root the OpenAI SDK appends ``/chat/completions`` to. A
# per-provider ``providers.base_url`` DB value overrides these (e.g. the
# Moonshot/DashScope China endpoints, or an OpenRouter proxy).
DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# OpenAgent vocabulary:
#   - **provider**  : the model's vendor / owner (anthropic, openai, google, …).
#   - **framework** : the adapter OpenAgent uses to instantiate the runtime
#                     agent — ``api-based`` (native runtime ``Agent`` with
#                     the provider's API key) or ``claude-cli`` (the runtime's
#                     Claude SDK adapter wrapping
#                     the local ``claude`` binary, no API key).
#   - **kind**      : ``llm`` / ``tts`` / ``stt``. The runtime dispatches by
#                     kind first; framework only matters for ``kind='llm'``.
#                     TTS/STT rows always have framework='api-based' and
#                     route through ``litellm.aspeech`` / ``atranscription``.
#   - **model**     : the bare model id (``gpt-4o-mini``, ``claude-sonnet-4-6``).
#
# ``runtime_id`` encodes provider+model (and framework for claude-cli). Layout:
#   - ``api-based`` framework (the default): ``<provider>:<model>``.
#   - ``claude-cli`` framework             : ``claude-cli:<provider>:<model>``
#                                              where provider is always
#                                              ``anthropic`` in practice.
#
# Rationale: most provider+model pairs only run via the api-based adapter,
# so keeping the two-part form for them keeps existing usage_log rows and
# user references valid. Only claude-cli entries gain the three-part form,
# which is the minimum needed to distinguish "anthropic via API" from
# "anthropic via Claude SDK subscription".
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
    "moonshot",
    "qwen",
    "local",
]

# Framework values written to ``providers.framework``. Three values:
#   - ``api-based``  → runtime ``Agent`` for LLM, ``litellm.aspeech`` /
#                      ``litellm.atranscription`` for TTS/STT.
#   - ``claude-cli`` → ``ClaudeBackedAgent`` (drives ``claude_agent_sdk``
#                      against the user's Claude Pro/Max subscription).
#   - ``codex-cli``  → ``CodexBackedAgent`` (drives ``openai_codex``
#                      against the user's ChatGPT Plus/Pro subscription).
FRAMEWORK_API_BASED = "api-based"
FRAMEWORK_CLAUDE_CLI = "claude-cli"
FRAMEWORK_CODEX_CLI = "codex-cli"

# Transitional aliases — the old framework names ``agno`` (legacy: LLM native runtime) and ``litellm`` (TTS/STT) collapse into the single ``api-based``
# value above. A DB migration rewrites existing rows; these aliases stay
# for one release so any stragglers in code or tests don't crash. Delete
# in a follow-up cleanup once the codebase is fully on FRAMEWORK_API_BASED.
FRAMEWORK_AGNO = FRAMEWORK_API_BASED
FRAMEWORK_LITELLM = FRAMEWORK_API_BASED

# LLM-dispatch frameworks. The router and LLM code paths iterate over
# these — TTS/STT providers (kind != 'llm') live in the same ``providers``
# table but are addressed by capability-specific code (``channels/tts.py``
# and ``channels/voice.py``) keyed off ``kind``, not the LLM dispatcher.
LLM_FRAMEWORKS = (FRAMEWORK_API_BASED, FRAMEWORK_CLAUDE_CLI, FRAMEWORK_CODEX_CLI)
SUPPORTED_FRAMEWORKS = LLM_FRAMEWORKS

# Subscription-CLI frameworks: ChatGPT/Claude-subscription-backed paths.
# Both share these traits the router cares about:
#   - billed against a user subscription, not per-token via API
#   - cannot serve as the runtime Team's ``team.model`` (the routing classifier)
#     because their backing ``Agent.model`` is a placeholder
# ``team_router._cheapest_api_based_model`` is consulted for these
# leaders, and pricing returns zero for them.
SUBSCRIPTION_CLI_FRAMEWORKS = (FRAMEWORK_CLAUDE_CLI, FRAMEWORK_CODEX_CLI)


@dataclass(frozen=True)
class CatalogModel:
    provider: str
    model_id: str
    runtime_id: str
    history_mode: str
    framework: str = FRAMEWORK_API_BASED
    disabled: bool = False
    display_name: str | None = None
    tier_hint: str | None = None
    metadata: dict[str, Any] | None = None
    # v0.12 fields — the provider's surrogate id and (name, framework)
    # pair resolved at hydration time so routing doesn't have to re-split
    # the runtime_id back apart. ``provider_id = 0`` indicates a seed
    # entry from yaml routing hints (no backing provider row yet).
    provider_id: int = 0
    # DB-level marker (mirror of ``models.is_classifier``) kept on the
    # catalog row so model_manager surfaces the flag in /api/models
    # responses. No router currently consumes this — SmartRouter was
    # retired in v0.14 in favour of the runtime's Team-based routing.
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
    framework: str = FRAMEWORK_API_BASED,
) -> str:
    """Canonical runtime_id for a (framework, provider, model) triple.

    api-based entries produce ``<provider>:<model>`` (preserved from v0.9.x).
    Subscription-CLI entries use a 3-part form:
      - ``claude-cli:<provider>:<model>`` — provider is always
        ``anthropic`` in practice.
      - ``codex-cli:<provider>:<model>``  — provider is always
        ``openai`` in practice.
    Legacy shorthands ``claude-cli/<model>`` / ``codex-cli/<model>`` are
    accepted and rewritten to the canonical form.
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

    # codex-cli mirror of the claude-cli logic above.
    if raw.startswith("codex-cli/"):
        _, rest = raw.split("/", 1)
        if ":" in rest:
            prov, model = rest.split(":", 1)
            return f"{FRAMEWORK_CODEX_CLI}:{prov}:{model}"
        return f"{FRAMEWORK_CODEX_CLI}:openai:{rest}"
    if raw.startswith(f"{FRAMEWORK_CODEX_CLI}:"):
        tail = raw[len(FRAMEWORK_CODEX_CLI) + 1:]
        if tail.count(":") >= 1:
            return raw
        return f"{FRAMEWORK_CODEX_CLI}:openai:{tail}"
    if framework == FRAMEWORK_CODEX_CLI:
        effective_provider = provider_name or "openai"
        if effective_provider == FRAMEWORK_CODEX_CLI:
            effective_provider = "openai"
        return f"{FRAMEWORK_CODEX_CLI}:{effective_provider}:{raw}"

    # api-based framework — 2-part form.
    if provider_name == FRAMEWORK_CLAUDE_CLI:
        # Caller passed the deprecated pseudo-provider. Treat as framework hint.
        return f"{FRAMEWORK_CLAUDE_CLI}:anthropic:{raw}"
    if provider_name == FRAMEWORK_CODEX_CLI:
        return f"{FRAMEWORK_CODEX_CLI}:openai:{raw}"
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
    if is_claude_cli_model(raw) or is_codex_cli_model(raw):
        return raw
    if ":" in raw:
        return raw
    configured_names = _configured_provider_names(providers_config)
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        if prefix in (FRAMEWORK_CLAUDE_CLI, FRAMEWORK_CODEX_CLI):
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
                    provider_name, raw, entry.get("framework") or FRAMEWORK_API_BASED,
                )
    return raw


def _iter_provider_entries(providers_config: Any) -> list[dict[str, Any]]:
    """Yield a list of provider dicts regardless of the config shape.

    Accepts the v0.12 flat list, the pre-v0.12 name-keyed dict (including
    the special ``claude-cli`` / ``codex-cli`` buckets), or ``None``.
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
            elif name == FRAMEWORK_CODEX_CLI:
                out.append({
                    "name": "openai",
                    "framework": FRAMEWORK_CODEX_CLI,
                    **cfg,
                })
            else:
                out.append({
                    "name": name,
                    "framework": cfg.get("framework") or FRAMEWORK_API_BASED,
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


def is_codex_cli_model(model_ref: str | None) -> bool:
    """True when ``model_ref`` is dispatched via the codex-cli framework.

    Canonical form ``codex-cli:<provider>:<model>``; legacy shorthand
    ``codex-cli`` / ``codex-cli/<model>`` also accepted.
    """
    raw = str(model_ref or "").strip()
    return (
        raw == FRAMEWORK_CODEX_CLI
        or raw.startswith(f"{FRAMEWORK_CODEX_CLI}:")
        or raw.startswith(f"{FRAMEWORK_CODEX_CLI}/")
    )


def is_subscription_cli_model(model_ref: str | None) -> bool:
    """True for any subscription-CLI framework (claude-cli OR codex-cli)."""
    return is_claude_cli_model(model_ref) or is_codex_cli_model(model_ref)


def framework_of(model_ref: str | None) -> str:
    """Return the framework name for ``model_ref``.

    ``claude-cli`` / ``codex-cli`` when the ref carries the matching
    prefix; otherwise ``api-based``.
    """
    if is_claude_cli_model(model_ref):
        return FRAMEWORK_CLAUDE_CLI
    if is_codex_cli_model(model_ref):
        return FRAMEWORK_CODEX_CLI
    return FRAMEWORK_API_BASED


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


def codex_cli_model_spec(model_id: str | None = None) -> str:
    """Build the canonical codex-cli runtime_id from a bare model id."""
    raw = str(model_id or "").strip()
    if not raw:
        return FRAMEWORK_CODEX_CLI
    return f"{FRAMEWORK_CODEX_CLI}:openai:{raw}"


def split_runtime_id(runtime_id: str) -> tuple[str, str]:
    """Split a runtime id into ``(provider, model_id)`` for billing / display.

    v0.10+ forms:
      - ``<provider>:<model>``                  → (provider, model)
      - ``claude-cli:<provider>:<model>``       → (provider, model)
      - ``codex-cli:<provider>:<model>``        → (provider, model)
    Legacy forms still accepted:
      - ``claude-cli/<model>``                  → ("claude-cli", model)
      - ``claude-cli``                          → ("claude-cli", "claude-cli")
      - ``codex-cli/<model>``                   → ("codex-cli", model)
      - ``codex-cli``                           → ("codex-cli", "codex-cli")
      - bare ``<id>``                           → (id, id)
    """
    if runtime_id.startswith(f"{FRAMEWORK_CLAUDE_CLI}:"):
        tail = runtime_id[len(FRAMEWORK_CLAUDE_CLI) + 1:]
        if ":" in tail:
            provider, model_id = tail.split(":", 1)
            return provider, model_id
        # claude-cli:<model> — legacy, assume anthropic.
        return "anthropic", tail
    if runtime_id.startswith(f"{FRAMEWORK_CODEX_CLI}:"):
        tail = runtime_id[len(FRAMEWORK_CODEX_CLI) + 1:]
        if ":" in tail:
            provider, model_id = tail.split(":", 1)
            return provider, model_id
        # codex-cli:<model> — legacy, assume openai.
        return "openai", tail
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
    if is_subscription_cli_model(runtime_id):
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
          {"id": 1, "name": "openai", "framework": "api-based",
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
                entry_framework = cfg.get("framework") or FRAMEWORK_API_BASED
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
        provider_framework = entry.get("framework") or FRAMEWORK_API_BASED
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
                if provider_framework in SUBSCRIPTION_CLI_FRAMEWORKS
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

    When a provider is registered under both frameworks (anthropic+api-based
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

    # 1. Subscription-CLI frameworks (claude-cli / codex-cli) dispatch
    # against the user's Pro/Max or ChatGPT Plus/Pro subscription —
    # no per-token billing, ever. Short-circuit before any lookup so we
    # don't accidentally attribute API pricing to a subscription session.
    if is_claude_cli_model(runtime_id):
        _log_pricing(model_ref, runtime_id, "claude_cli_subscription", 0.0, 0.0)
        return {"input_cost_per_million": 0.0, "output_cost_per_million": 0.0}
    if is_codex_cli_model(runtime_id):
        _log_pricing(model_ref, runtime_id, "codex_cli_subscription", 0.0, 0.0)
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

        from src.models import discovery
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


# Cached reverse map of OpenAgent provider name → OpenRouter vendor prefix.
# Built lazily on first pricing lookup and reused thereafter — used to
# avoid scanning ``_OPENROUTER_VENDOR_MAP`` per lookup.
_REVERSE_VENDOR_MAP_CACHE: dict[str, str] | None = None

# Cached id → entry index over the OpenRouter catalog, keyed by the
# cache timestamp so a refresh invalidates it automatically. Replaces
# the previous linear scan that walked every entry per pricing lookup.
_OPENROUTER_INDEX: tuple[float, dict[str, dict[str, Any]]] | None = None


def _get_reverse_vendor_map(discovery_module: Any) -> dict[str, str]:
    global _REVERSE_VENDOR_MAP_CACHE
    if _REVERSE_VENDOR_MAP_CACHE is None:
        _REVERSE_VENDOR_MAP_CACHE = {
            our_name: vendor
            for vendor, our_name in discovery_module._OPENROUTER_VENDOR_MAP.items()
        }
    return _REVERSE_VENDOR_MAP_CACHE


def _get_openrouter_index(
    cache: tuple[float, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Return ``{entry['id']: entry}`` over the OpenRouter catalog,
    rebuilt whenever the cache timestamp changes.
    """
    global _OPENROUTER_INDEX
    cache_ts, entries = cache
    if _OPENROUTER_INDEX is not None and _OPENROUTER_INDEX[0] == cache_ts:
        return _OPENROUTER_INDEX[1]
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id") or "")
        if entry_id:
            index[entry_id] = entry
    _OPENROUTER_INDEX = (cache_ts, index)
    return index


def _openrouter_pricing_lookup(runtime_id: str) -> dict[str, float] | None:
    """Look up pricing for ``runtime_id`` in the OpenRouter cache.

    Reads ``discovery._OPENROUTER_CACHE`` without triggering a fetch —
    this is a hot path and must not block on network. Uses an
    indexed-by-id map (rebuilt on cache refresh) so the lookup is
    O(1) instead of a linear scan over ~300 catalog entries per call.
    Returns ``None`` when the cache is empty or the model isn't in
    OpenRouter's catalog.
    """
    try:
        from src.models import discovery
    except ImportError:
        return None
    cache = getattr(discovery, "_OPENROUTER_CACHE", None)
    if not cache or ":" not in runtime_id:
        return None
    provider, bare = runtime_id.split(":", 1)
    reverse_map = _get_reverse_vendor_map(discovery)
    want_prefix = reverse_map.get(provider)
    if not want_prefix:
        return None
    target = f"{want_prefix}/{bare}"
    entry = _get_openrouter_index(cache).get(target)
    if entry is None:
        return None
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
