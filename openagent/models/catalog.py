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
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    metadata: dict[str, Any] | None = None


def _coerce_cost(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _entry_model_id(entry: Any) -> str:
    if isinstance(entry, dict):
        for key in ("id", "model_id", "model"):
            value = entry.get(key)
            if value:
                return str(value).strip()
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


def normalize_runtime_model_id(model_ref: str, providers_config: dict | None = None) -> str:
    raw = str(model_ref or "").strip()
    if not raw:
        return raw
    if is_claude_cli_model(raw):
        return raw
    if ":" in raw:
        return raw
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        if prefix == FRAMEWORK_CLAUDE_CLI:
            return raw
        if prefix in SUPPORTED_PROVIDERS or prefix in (providers_config or {}):
            return f"{prefix}:{rest}"
        return raw
    for provider_name, cfg in (providers_config or {}).items():
        for entry in cfg.get("models", []):
            if _entry_model_id(entry) == raw:
                return build_runtime_model_id(provider_name, raw)
    return raw


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


def model_history_mode(model_ref: str, providers_config: dict | None = None) -> str:
    runtime_id = normalize_runtime_model_id(model_ref, providers_config)
    if is_claude_cli_model(runtime_id):
        return "provider"
    return "platform"


def iter_configured_models(
    providers_config: dict | None,
    *,
    include_disabled: bool = False,
    history_mode: str | None = None,
) -> list[CatalogModel]:
    """Flatten yaml ``providers.X.models`` into CatalogModel records.

    Legacy quirk: if yaml still declares ``providers.claude-cli.models``
    (pre-v0.10), we treat those rows as framework=claude-cli models on
    the ``anthropic`` provider — that's the only thing claude-cli can
    actually dispatch.
    """
    results: list[CatalogModel] = []
    seen: set[str] = set()

    for provider_name, cfg in (providers_config or {}).items():
        disabled = {str(item).strip() for item in cfg.get("disabled_models", [])}
        # Legacy yaml: ``providers.claude-cli.models`` declares claude-cli
        # models in a v0.9-vocabulary config. Map onto the new shape.
        if provider_name == FRAMEWORK_CLAUDE_CLI:
            row_provider = "anthropic"
            row_framework = FRAMEWORK_CLAUDE_CLI
        else:
            row_provider = provider_name
            row_framework = FRAMEWORK_AGNO

        for entry in cfg.get("models", []):
            model_id = _entry_model_id(entry)
            if not model_id:
                continue
            is_disabled = model_id in disabled
            if is_disabled and not include_disabled:
                continue

            runtime_id = build_runtime_model_id(row_provider, model_id, row_framework)
            mode = model_history_mode(runtime_id, providers_config)
            if history_mode and mode != history_mode:
                continue
            if runtime_id in seen:
                continue
            seen.add(runtime_id)

            metadata = _entry_metadata(entry)
            results.append(
                CatalogModel(
                    provider=row_provider,
                    model_id=model_id,
                    runtime_id=runtime_id,
                    history_mode=mode,
                    framework=row_framework,
                    disabled=is_disabled,
                    input_cost_per_million=_coerce_cost(metadata.get("input_cost_per_million")),
                    output_cost_per_million=_coerce_cost(metadata.get("output_cost_per_million")),
                    metadata=metadata or None,
                )
            )
    return results


def supported_providers(configured: dict | None = None) -> list[str]:
    provider_set = set(SUPPORTED_PROVIDERS)
    provider_set.update((configured or {}).keys())
    return sorted(provider_set)


def get_default_model_for_provider(provider_name: str, providers_config: dict | None = None) -> str | None:
    for entry in iter_configured_models(providers_config):
        if entry.provider == provider_name:
            return entry.runtime_id
    return None


def get_model_pricing(model_ref: str, providers_config: dict | None = None) -> dict[str, float]:
    """Return ``{input_cost_per_million, output_cost_per_million}`` for a model.

    Lookup order:
      1. claude-cli models → zero (billed via Claude Pro/Max subscription,
         not per token)
      2. User-provided pricing in ``providers_config`` (per-model metadata)
      3. Online catalog (OpenRouter) — see ``_openrouter_pricing_lookup``
      4. Zero pricing (logged as "missing")

    Always returns a dict; never raises. Emits ``catalog.pricing_resolved`` event
    on every call with the lookup ``source`` so cost issues can be diagnosed
    from the event log.
    """
    runtime_id = normalize_runtime_model_id(model_ref, providers_config)

    # 1. claude-cli is the local subprocess wrapping the user's Claude
    # Pro/Max subscription — no per-token billing, ever. Short-circuit
    # before any lookup so we don't accidentally attribute Anthropic API
    # pricing to a claude-cli session.
    if is_claude_cli_model(runtime_id):
        _log_pricing(model_ref, runtime_id, "claude_cli_subscription", 0.0, 0.0)
        return {"input_cost_per_million": 0.0, "output_cost_per_million": 0.0}

    bare_id = model_id_from_runtime(runtime_id)

    # 2. Per-model metadata in user config wins.
    for entry in iter_configured_models(providers_config, include_disabled=True):
        if runtime_id in {entry.runtime_id, entry.model_id} or bare_id in {entry.runtime_id, entry.model_id}:
            input_cost = float(entry.input_cost_per_million or 0.0)
            output_cost = float(entry.output_cost_per_million or 0.0)
            if input_cost > 0 or output_cost > 0:
                _log_pricing(model_ref, runtime_id, "config", input_cost, output_cost)
                return {"input_cost_per_million": input_cost, "output_cost_per_million": output_cost}
            break

    # 3. Online catalog (OpenRouter). Resolved from a process-wide cache
    # populated by discovery.py; never blocks — returns None on cache miss.
    online = _openrouter_pricing_lookup(runtime_id)
    if online is not None:
        _log_pricing(
            model_ref, runtime_id, "openrouter",
            online["input_cost_per_million"], online["output_cost_per_million"],
        )
        return online

    # 4. Nothing — log a warning so the user can fix their config.
    _log_pricing(model_ref, runtime_id, "missing", 0.0, 0.0)
    return {"input_cost_per_million": 0.0, "output_cost_per_million": 0.0}


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


def compute_cost(model_ref: str, input_tokens: int, output_tokens: int, providers_config: dict | None = None) -> float:
    pricing = get_model_pricing(model_ref, providers_config)
    return (
        (pricing["input_cost_per_million"] * max(0, input_tokens)) / 1_000_000
        + (pricing["output_cost_per_million"] * max(0, output_tokens)) / 1_000_000
    )
