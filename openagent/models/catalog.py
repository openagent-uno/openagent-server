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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openagent.core.logging import elog

_DEFAULT_PRICING_PATH = Path(__file__).with_name("default_pricing.json")
_DEFAULT_PRICING_CACHE: dict[str, dict[str, float]] | None = None
# Dedup `catalog.pricing_resolved` events so each (runtime_id, source) pair only
# logs once per process. Without this every call to compute_cost emits a row;
# at 3+ lookups per chat turn that drowns the event log.
_LOGGED_PRICING: set[tuple[str, str]] = set()

# Provider-specific defaults. Keep here so they have a single home alongside
# the rest of the provider/model catalog.
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"


def _load_default_pricing() -> dict[str, dict[str, float]]:
    """Load the bundled default pricing table, cached for the process lifetime."""
    global _DEFAULT_PRICING_CACHE
    if _DEFAULT_PRICING_CACHE is not None:
        return _DEFAULT_PRICING_CACHE
    try:
        raw = json.loads(_DEFAULT_PRICING_PATH.read_text())
    except (OSError, ValueError):
        _DEFAULT_PRICING_CACHE = {}
        return _DEFAULT_PRICING_CACHE
    cleaned: dict[str, dict[str, float]] = {}
    for key, value in raw.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        cleaned[key] = {
            "input_cost_per_million": float(value.get("input_cost_per_million", 0.0) or 0.0),
            "output_cost_per_million": float(value.get("output_cost_per_million", 0.0) or 0.0),
        }
    _DEFAULT_PRICING_CACHE = cleaned
    return cleaned

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
]

CLAUDE_CLI_PROVIDER = "claude-cli"


@dataclass(frozen=True)
class CatalogModel:
    provider: str
    model_id: str
    runtime_id: str
    history_mode: str
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


def build_runtime_model_id(provider_name: str, model_id: str) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return raw
    if is_claude_cli_model(raw):
        return raw
    if ":" in raw:
        return raw
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        if prefix == CLAUDE_CLI_PROVIDER:
            return raw
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
        if prefix == CLAUDE_CLI_PROVIDER:
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
    raw = str(model_ref or "").strip()
    return raw == CLAUDE_CLI_PROVIDER or raw.startswith(f"{CLAUDE_CLI_PROVIDER}/")


def claude_cli_model_spec(model_id: str | None = None) -> str:
    raw = str(model_id or "").strip()
    return f"{CLAUDE_CLI_PROVIDER}/{raw}" if raw else CLAUDE_CLI_PROVIDER


def split_runtime_id(runtime_id: str) -> tuple[str, str]:
    """Split a runtime model id into ``(provider, model_id)``.

    Most providers use ``:`` as the separator (``openai:gpt-4o-mini``).
    ``claude-cli`` uses ``/`` (``claude-cli/claude-sonnet-4-6``). When neither
    separator is present, the input is returned twice so callers can treat it
    as both provider and model id.
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
    results: list[CatalogModel] = []
    seen: set[str] = set()

    for provider_name, cfg in (providers_config or {}).items():
        disabled = {str(item).strip() for item in cfg.get("disabled_models", [])}
        for entry in cfg.get("models", []):
            model_id = _entry_model_id(entry)
            if not model_id:
                continue
            is_disabled = model_id in disabled
            if is_disabled and not include_disabled:
                continue

            runtime_id = build_runtime_model_id(provider_name, model_id)
            mode = model_history_mode(runtime_id, providers_config)
            if history_mode and mode != history_mode:
                continue
            if runtime_id in seen:
                continue
            seen.add(runtime_id)

            metadata = _entry_metadata(entry)
            results.append(
                CatalogModel(
                    provider=provider_name,
                    model_id=model_id,
                    runtime_id=runtime_id,
                    history_mode=mode,
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
      1. User-provided pricing in ``providers_config`` (per-model metadata)
      2. Bundled defaults in ``models/default_pricing.json``
      3. Zero pricing (logged as a warning)

    Always returns a dict; never raises. Emits ``catalog.pricing_resolved`` event
    on every call with the lookup ``source`` so cost issues can be diagnosed
    from the event log.
    """
    runtime_id = normalize_runtime_model_id(model_ref, providers_config)
    bare_id = model_id_from_runtime(runtime_id)

    # 1. Per-model metadata in user config wins.
    for entry in iter_configured_models(providers_config, include_disabled=True):
        if runtime_id in {entry.runtime_id, entry.model_id} or bare_id in {entry.runtime_id, entry.model_id}:
            input_cost = float(entry.input_cost_per_million or 0.0)
            output_cost = float(entry.output_cost_per_million or 0.0)
            if input_cost > 0 or output_cost > 0:
                _log_pricing(model_ref, runtime_id, "config", input_cost, output_cost)
                return {"input_cost_per_million": input_cost, "output_cost_per_million": output_cost}
            # Config entry exists but has no pricing → fall through to defaults.
            break

    # 2. Bundled default pricing table.
    defaults = _load_default_pricing()
    pricing = defaults.get(runtime_id) or defaults.get(bare_id)
    # claude-cli is a transport for Anthropic models — fall back to anthropic
    # pricing when no claude-cli-specific entry exists (claude-cli:<X> → anthropic:<X>).
    if pricing is None and is_claude_cli_model(runtime_id) and bare_id and bare_id != CLAUDE_CLI_PROVIDER:
        pricing = defaults.get(f"anthropic:{bare_id}")
        if pricing:
            _log_pricing(
                model_ref, runtime_id, "default_via_anthropic",
                pricing["input_cost_per_million"], pricing["output_cost_per_million"],
            )
            return dict(pricing)
    if pricing:
        _log_pricing(
            model_ref, runtime_id, "default",
            pricing["input_cost_per_million"], pricing["output_cost_per_million"],
        )
        return dict(pricing)

    # 3. Nothing — log a warning so the user can fix their config.
    _log_pricing(model_ref, runtime_id, "missing", 0.0, 0.0)
    return {"input_cost_per_million": 0.0, "output_cost_per_million": 0.0}


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
