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
    runtime_id = normalize_runtime_model_id(model_ref, providers_config)
    bare_id = runtime_id.split(":", 1)[1] if ":" in runtime_id else runtime_id

    for entry in iter_configured_models(providers_config, include_disabled=True):
        if runtime_id in {entry.runtime_id, entry.model_id} or bare_id in {entry.runtime_id, entry.model_id}:
            return {
                "input_cost_per_million": float(entry.input_cost_per_million or 0.0),
                "output_cost_per_million": float(entry.output_cost_per_million or 0.0),
            }
    return {
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
    }


def compute_cost(model_ref: str, input_tokens: int, output_tokens: int, providers_config: dict | None = None) -> float:
    pricing = get_model_pricing(model_ref, providers_config)
    return (
        (pricing["input_cost_per_million"] * max(0, input_tokens)) / 1_000_000
        + (pricing["output_cost_per_million"] * max(0, output_tokens)) / 1_000_000
    )
