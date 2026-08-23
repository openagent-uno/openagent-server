"""Catalog helpers for configured providers, models, and pricing.

This module deliberately keeps product-facing provider/model metadata under
OpenAgent control instead of delegating it to the inlined runtime. The runtime is the
execution engine for the ``api-based`` LLM path (native ``Agent``) — while
OpenAgent remains the source of truth for:

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

# ── Session continuity (vision §16) ───────────────────────────────────
#
# The runtime persists every conversation's transcript in the shared
# ``sessions`` SQLite table, keyed by ``session_id``. Two constants make
# that durable end-to-end:
#
# ``RUNTIME_SESSION_USER_ID`` — the single, stable owner the runtime
#   stamps on (and reads back from) every ``sessions`` row. The runtime
#   store gates BOTH its history read and its runs write on
#   ``user_id == <this> OR user_id IS NULL`` (see
#   ``src/memory/store/sqlite/sqlite.py``). Using one constant on every
#   path — live chat, Telegram/Discord/WhatsApp bridges, scheduler,
#   workflows, sub-agents — keeps the row readable/writable no matter
#   which surface the turn arrived on. Per-user separation is carried by
#   the ``session_id`` itself (e.g. ``tg:<uid>``), never by this column,
#   so one owner is correct for the single-tenant agent (vision §17).
#   The gateway is metadata-only on this table and must NEVER write a
#   different value here — doing so silently blocks the runtime's
#   read/write and the agent "forgets" the conversation every turn.
#   (See ``MemoryDB.upsert_session`` and ``_migrate_reclaim_session_owners``.)
#
# ``FULL_SESSION_HISTORY_RUNS`` — the ``num_history_runs`` handed to the
#   runtime Agent/Team so ``add_history_to_context=True`` replays the
#   ENTIRE stored transcript into context, not a trailing N-turn window.
#   The runtime slices ``runs[-N:]``; this (effectively unbounded) N
#   returns everything. In-place compaction (``src/core/compaction.py``,
#   vision §2) keeps the actual token footprint under the model's context
#   limit by folding the oldest runs into a recap row when the threshold
#   is crossed — so "load the whole session" stays safe.
RUNTIME_SESSION_USER_ID = "openagent"
FULL_SESSION_HISTORY_RUNS = 10_000_000

# OpenAgent vocabulary:
#   - **provider**  : the model's vendor / owner (anthropic, openai, google, …).
#   - **framework** : the adapter OpenAgent uses to instantiate the runtime
#                     agent. ``api-based`` (the native runtime ``Agent`` with
#                     the provider's API key) is the only shipped framework;
#                     the ``framework`` column remains the seam for adding
#                     more later.
#   - **kind**      : ``llm`` / ``tts`` / ``stt``. The runtime dispatches by
#                     kind first; framework only matters for ``kind='llm'``.
#                     TTS/STT rows always have framework='api-based' and
#                     route through ``litellm.aspeech`` / ``atranscription``.
#   - **model**     : the bare model id (``gpt-4o-mini``, ``claude-sonnet-4-6``).
#
# ``runtime_id`` encodes provider+model as ``<provider>:<model>``.
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

# Framework values written to ``providers.framework``:
#   - ``api-based``  → runtime ``Agent`` for LLM, ``litellm.aspeech`` /
#                      ``litellm.atranscription`` for TTS/STT.
# This is currently the only shipped framework; the ``framework`` column
# remains the seam for adding more (e.g. a subscription-CLI adapter) later.
FRAMEWORK_API_BASED = "api-based"

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
LLM_FRAMEWORKS = (FRAMEWORK_API_BASED,)
SUPPORTED_FRAMEWORKS = LLM_FRAMEWORKS


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
    # responses.
    #
    # Despite the name, this is NOT a "pick a model per turn" classifier
    # flag: the classifier-LLM router (``SmartRouter``) was retired in
    # v0.14 (``e8f5d68``) in favour of the runtime's Team-based routing,
    # and no per-turn classifier call survives it.
    #
    # It IS consumed, though — do not believe otherwise. It is the
    # user's persistent *default team leader* hint, read by
    # ``ModelDispatcher._resolve_entry_model`` as the second step of
    # ``per-session pin -> is_classifier -> first enabled``. The flagged
    # model becomes the entry model, i.e. the Team leader that routes to
    # the other enabled models as members. Flag a cheap model here and it
    # answers trivia itself while delegating the hard turns — which is
    # what the retired classifier router was for, minus the extra
    # round-trip.
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
    """Canonical runtime_id for a (provider, model) pair: ``<provider>:<model>``.

    ``framework`` is accepted for signature stability — ``api-based`` is the
    only shipped framework and every entry uses this 2-part form.
    """
    del framework  # only api-based ships; the runtime_id is framework-agnostic
    raw = str(model_id or "").strip()
    if not raw:
        return raw
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
    if ":" in raw:
        return raw
    configured_names = _configured_provider_names(providers_config)
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
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

    Accepts the v0.12 flat list, the pre-v0.12 name-keyed dict, or ``None``.
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


def framework_of(model_ref: str | None) -> str:
    """Return the framework name for ``model_ref``.

    ``api-based`` is the only shipped framework, so this is always
    ``api-based``. Kept as the seam that resolves a runtime_id's framework.
    """
    del model_ref
    return FRAMEWORK_API_BASED


def split_runtime_id(runtime_id: str) -> tuple[str, str]:
    """Split a runtime id into ``(provider, model_id)`` for billing / display.

    Forms:
      - ``<provider>:<model>``  → (provider, model)
      - ``<provider>/<model>``  → (provider, model)
      - bare ``<id>``           → (id, id)
    """
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
    # api-based — the only shipped framework — is platform-managed history.
    del model_ref, providers_config
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
          {"id": 2, "name": "anthropic", "framework": "api-based",
           "api_key": "sk-…", "models": [{"id": 7, "model": "claude-opus-4-7"}]},
        ]

    Legacy shape (accepted for back-compat with yaml seed / old tests) —
    a ``dict`` keyed by provider name.
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
            mode = "platform"
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

    When a provider is registered under more than one framework, pass
    ``framework=`` to disambiguate.
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
      1. OpenRouter in-process cache — primed lazily on first miss so the
         next call hits warm cache.
      2. Zero pricing (logged as "missing") if OpenRouter is unreachable.

    Always returns a dict; never raises. ``providers_config`` is accepted
    for backward compat with callers that still pass it but is no longer
    consulted for pricing — the DB / yaml never carry authoritative cost
    anymore. Emits ``catalog.pricing_resolved`` so zero-cost events can
    be alerted on.
    """
    runtime_id = normalize_runtime_model_id(model_ref, providers_config)

    # Online catalog (OpenRouter). Resolved from a process-wide cache
    # populated by discovery.py; never blocks — returns None on cache miss.
    online = _openrouter_pricing_lookup(runtime_id)
    if online is not None:
        _log_pricing(
            model_ref, runtime_id, "openrouter",
            online["input_cost_per_million"], online["output_cost_per_million"],
        )
        return online

    # 1b. Cache miss — fire-and-forget a prime so the next lookup hits.
    # Doesn't block the current turn; we still return zero this time.
    _maybe_prime_openrouter_cache()

    # 2. Nothing — log so ops can alert on persistently-zero entries.
    _log_pricing(model_ref, runtime_id, "missing", 0.0, 0.0)
    return {
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
        "input_cache_read_per_million": 0.0,
    }


def openrouter_pricing_ready() -> bool:
    """True when OpenRouter's pricing catalog is loaded (non-empty, unexpired).

    :func:`get_model_pricing` returns ``$0`` on a COLD cache exactly as it does
    for a model genuinely absent from OpenRouter (the $0 claude-sub-proxy). A
    caller that treats "$0 == unpriced" — the budget guard's cost-metric
    warning — must first confirm the catalog actually loaded, otherwise a
    boot-time miss reads like a free model and fires a false warning. Never
    raises; a cold/missing cache simply reports "not ready".
    """
    try:
        from src.models import discovery
    except ImportError:
        return False
    cache = getattr(discovery, "_OPENROUTER_CACHE", None)
    if not cache:
        return False
    try:
        import time as _time

        ts, entries = cache[0], cache[1]
        return bool(entries) and (_time.time() - ts) < discovery._CACHE_TTL_SECONDS
    except Exception:  # noqa: BLE001 — a malformed cache tuple → "not ready"
        return False


def cheapest_enabled_model(providers_config: Any) -> "CatalogModel | None":
    """The cheapest enabled ``api-based`` model in ``providers_config``.

    "Cheapest" = lowest ``input + output`` cost-per-million per
    :func:`get_model_pricing` (OpenRouter-live). A model absent from OpenRouter
    — notably the $0 claude-sub-proxy — resolves to 0 and therefore wins as
    free. Ties resolve to configuration order (first wins). Returns ``None``
    when no enabled api-based row exists.

    Used by background jobs (compaction summariser, quality judge) that want a
    throwaway single-model call at the cheapest enabled row instead of routing
    the whole ~150k-token fold through the full Team leader.
    """
    from src.core.execution_profile import _is_cloud_model_id

    best: CatalogModel | None = None
    best_key: tuple[float, int] | None = None
    for entry in iter_configured_models(providers_config):
        if entry.disabled or entry.framework != FRAMEWORK_API_BASED:
            continue
        pricing = get_model_pricing(entry.runtime_id, providers_config)
        cost = (
            float(pricing.get("input_cost_per_million", 0.0) or 0.0)
            + float(pricing.get("output_cost_per_million", 0.0) or 0.0)
        )
        # Tie-break towards a self-hosted row. Both a subscription proxy and
        # our own GPU report $0 here, so cost alone left the choice to
        # configuration order - and a background job silently landed on cloud
        # Claude. A self-hosted row is the only one that is genuinely free.
        key = (cost, 1 if _is_cloud_model_id(entry.runtime_id) else 0)
        if best_key is None or key < best_key:
            best, best_key = entry, key
    return best


# Fallback context window used when a model is absent from OpenRouter's
# catalog (local / sub-proxy / self-hosted ids). Kept identical to
# ``compaction._FALLBACK_MAX_CONTEXT`` so the /context gauge and the
# compaction trigger agree on the denominator for unknown models.
_FALLBACK_CONTEXT_WINDOW = 200_000

# Context windows published by the vendors themselves, matched on a substring
# of the runtime model id, most specific first.
#
# Why this table has to exist: the OpenRouter lookup only answers for ids
# OpenRouter serves. A model reached through a local or subscription proxy has
# an id of OUR making (``local:claude-haiku-4-5``), so the lookup always missed
# and every such model reported the generic 200k fallback. Anything that reads
# this number was therefore unable to tell two models apart — including
# compaction, which sizes the transcript it hands the summariser with it. A
# 1M-window model was under-used, and a small-window one was handed folds it
# could never accept.
#
# An unknown model still falls back to 200k, which is the conservative choice:
# we would rather compact a long-context model too eagerly than hand a short
# one a prompt it must reject.
_STATIC_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("claude-opus-5", 1_000_000),
    ("claude-opus-4", 200_000),
    ("claude-sonnet-5", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3-5", 200_000),
    ("deepseek-v4", 104_856),
    ("gpt-4o", 128_000),
)


def _static_context_length_lookup(runtime_id: str) -> int | None:
    """Vendor-published window for *runtime_id*, or ``None`` if we don't know.

    Substring match, in table order, against the id lowercased — so it works
    whether the caller passes ``claude-opus-5``, ``local:claude-opus-5`` or
    ``us.anthropic.claude-opus-5-v1``.
    """
    if not runtime_id:
        return None
    needle = runtime_id.lower()
    for marker, window in _STATIC_CONTEXT_WINDOWS:
        if marker in needle:
            return window
    return None


def get_model_context_window(
    model_ref: str, providers_config: dict | None = None
) -> tuple[int, str]:
    """Return ``(context_window_tokens, source)`` for a model.

    ``source`` is ``"openrouter"`` when the size came from OpenRouter's
    live catalog (which already carries a ``context_length`` per model,
    see :func:`_openrouter_context_length_lookup`), ``"static"`` when it came
    from the vendor-published table above (the normal answer for a
    proxy-served id), or ``"fallback"`` when we know nothing about the model
    and assume ``_FALLBACK_CONTEXT_WINDOW``. Never raises — mirrors
    :func:`get_model_pricing` so any configured provider resolves.
    """
    runtime_id = normalize_runtime_model_id(model_ref, providers_config)
    online = _openrouter_context_length_lookup(runtime_id)
    if online:
        return online, "openrouter"
    static = _static_context_length_lookup(runtime_id)
    if static:
        # Prime the cache anyway: a live answer outranks the table next time.
        _maybe_prime_openrouter_cache()
        return static, "static"
    # Cache miss — prime for next time, same as the pricing path.
    _maybe_prime_openrouter_cache()
    return _FALLBACK_CONTEXT_WINDOW, "fallback"


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


async def warm_pricing_cache() -> bool:
    """Await the OpenRouter catalog fetch so pricing is hot BEFORE the first
    turn — closes the cold-start blind spot where the first billed call of each
    boot logs ``$0`` (``get_model_pricing`` returns zero on a cache miss and only
    fires a background prime). Called from the server boot warm path next to the
    budget-guard warm, so a spend cap counts from call #1 instead of call #2.

    Idempotent (``_fetch_openrouter_catalog`` no-ops when the cache is fresh) and
    never fatal: on any network/HTTP error it logs and returns False, leaving the
    existing lazy-prime fallback intact. Returns True when the cache is warm.
    """
    try:
        from src.models import discovery
    except ImportError:
        return False
    try:
        entries = await discovery._fetch_openrouter_catalog()
        return bool(entries)
    except Exception as e:  # noqa: BLE001 — pricing warm must never block boot
        try:
            elog("catalog.pricing_warm_error", level="warning", error=str(e))
        except Exception:
            pass
        return False


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
        # Cached-prefix read price. DeepSeek (and others) prefix-cache
        # server-side and bill a re-sent prefix at a fraction of ``prompt``
        # (deepseek-v4-pro: $0.0036/M cache-read vs $0.435/M miss — ~120x).
        # Absent/None → 0.0, and compute_cost then falls back to the full
        # input price, so a model without a cache-read price is unchanged.
        cache_read_cost = float(pricing.get("input_cache_read") or 0.0) * 1_000_000
    except (TypeError, ValueError):
        return None
    if input_cost <= 0 and output_cost <= 0:
        return None
    return {
        "input_cost_per_million": input_cost,
        "output_cost_per_million": output_cost,
        "input_cache_read_per_million": cache_read_cost,
    }


def _openrouter_context_length_lookup(runtime_id: str) -> int | None:
    """Look up the context-window size for ``runtime_id`` in OpenRouter's cache.

    Mirrors :func:`_openrouter_pricing_lookup` exactly — same index, same
    reverse vendor map, no network — but reads the ``context_length``
    field (falling back to ``top_provider.context_length``) that the
    catalog fetch already carries and that pricing ignores. Returns
    ``None`` on cache miss or when the model isn't in OpenRouter.
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
    candidates = [
        entry.get("context_length"),
        (entry.get("top_provider") or {}).get("context_length")
        if isinstance(entry.get("top_provider"), dict)
        else None,
    ]
    for candidate in candidates:
        try:
            val = int(candidate)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
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


def compute_cost(
    model_ref: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Cost in USD for one call.

    ``input_tokens`` is the FULL prompt count (it includes ``cache_read_tokens``
    — that is how DeepSeek/OpenAI report ``prompt_tokens``). When a provider
    prefix-caches server-side, the re-sent prefix is billed at
    ``input_cache_read_per_million`` (a fraction of the miss price), so we split
    the prompt into miss + cache-read and price each. ``cache_read_tokens=0``
    (the default, and any model without a cache-read price) reproduces the old
    flat ``input × price + output × price`` exactly — a zero cache-read price
    falls back to the full input price, so a non-caching model is unchanged.
    """
    pricing = get_model_pricing(model_ref)
    in_tok = max(0, input_tokens)
    cache_read = min(max(0, cache_read_tokens), in_tok)  # can't exceed the prompt
    miss_tok = in_tok - cache_read
    in_price = pricing["input_cost_per_million"]
    # A zero/absent cache-read price means "unknown" → bill cache reads at the
    # full input price (conservative: over-, never under-count the brake).
    cache_price = pricing.get("input_cache_read_per_million") or in_price
    return (
        (in_price * miss_tok) / 1_000_000
        + (cache_price * cache_read) / 1_000_000
        + (pricing["output_cost_per_million"] * max(0, output_tokens)) / 1_000_000
    )
