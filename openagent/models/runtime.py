"""Shared runtime helpers for building and testing chat models."""

from __future__ import annotations

from typing import Any

from openagent.models.base import BaseModel, ModelResponse
from openagent.models.catalog import (
    DEFAULT_ZAI_BASE_URL,
    FRAMEWORK_AGNO,
    FRAMEWORK_CLAUDE_CLI,
    _iter_provider_entries,
    claude_cli_model_spec,
    framework_of,
    get_default_model_for_provider,
    is_claude_cli_model,
    model_id_from_runtime,
    normalize_runtime_model_id,
)

DEFAULT_API_MODEL = "anthropic:claude-sonnet-4-20250514"
LEGACY_PROVIDER_ALIASES = {
    "claude-api": "anthropic",
    "litellm": "agno",
    "zhipu": "zai",
}


def _resolved_claude_permission_mode(
    providers_config: Any,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    # Find the claude-cli anthropic provider row (if any) and look for a
    # permission_mode override in its metadata. Falls back to bypass.
    for entry in _iter_provider_entries(providers_config):
        if entry.get("name") != "anthropic":
            continue
        if entry.get("framework") != FRAMEWORK_CLAUDE_CLI:
            continue
        metadata = entry.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("permission_mode"):
            return str(metadata["permission_mode"])
    return "bypass"


def _canonical_provider_name(provider: str | None) -> str:
    raw = str(provider or "agno").strip() or "agno"
    return LEGACY_PROVIDER_ALIASES.get(raw, raw)


def _runtime_spec_from_config(
    model_cfg: dict,
    providers_config: Any,
) -> tuple[str, str | None]:
    provider = _canonical_provider_name(model_cfg.get("provider"))
    model_id = str(model_cfg.get("model_id") or "").strip()
    base_url = model_cfg.get("base_url")

    if provider == "claude-cli":
        return claude_cli_model_spec(model_id or None), base_url
    if provider == "anthropic":
        return normalize_runtime_model_id(f"anthropic:{model_id or 'claude-sonnet-4-6'}", providers_config), base_url
    if provider == "zai":
        spec = normalize_runtime_model_id(f"zai:{model_id or 'glm-5'}", providers_config)
        return spec, base_url or DEFAULT_ZAI_BASE_URL
    if provider == "agno":
        return normalize_runtime_model_id(model_id or DEFAULT_API_MODEL, providers_config), base_url
    raise ValueError(f"Unknown model provider: {provider}")


def wire_model_runtime(
    model: BaseModel,
    *,
    db: Any = None,
    mcp_pool: Any = None,
) -> BaseModel:
    """Attach runtime dependencies to a model when it supports them.

    Both providers consume from a single ``MCPPool`` that owns MCP
    lifecycle for the process. AgnoProvider gets pre-connected Agno
    ``MCPTools`` instances; ClaudeCLI gets the raw stdio config dict
    that the Claude Agent SDK accepts as its ``mcp_servers`` parameter.
    """
    if db is not None:
        set_db = getattr(model, "set_db", None)
        if callable(set_db):
            set_db(db)
    if mcp_pool is not None:
        # AgnoProvider / SmartRouter: pre-connected Agno MCPTools instances.
        set_mcp_toolkits = getattr(model, "set_mcp_toolkits", None)
        if callable(set_mcp_toolkits):
            set_mcp_toolkits(mcp_pool.agno_toolkits)
        # ClaudeCLI: raw stdio config for the Claude Agent SDK.
        set_mcp_servers = getattr(model, "set_mcp_servers", None)
        if callable(set_mcp_servers):
            set_mcp_servers(mcp_pool.claude_sdk_servers())
        # SmartRouter holds the pool itself so it can re-wire newly created
        # tier providers as they're lazily instantiated.
        set_mcp_pool = getattr(model, "set_mcp_pool", None)
        if callable(set_mcp_pool):
            set_mcp_pool(mcp_pool)
    return model


def create_model_from_spec(
    spec: str,
    *,
    providers_config: Any = None,
    api_key: str | None = None,
    base_url: str | None = None,
    monthly_budget: float = 0.0,
    routing: dict[str, str] | None = None,
    classifier_model: str | None = None,
    claude_permission_mode: str | None = None,
    claude_idle_timeout_seconds: int | None = None,
    claude_hard_timeout_seconds: int | None = None,
    claude_idle_ttl_seconds: int | None = None,
    db: Any = None,
    mcp_pool: Any = None,
) -> BaseModel:
    """Create a model instance from a compact OpenAgent runtime spec."""
    if providers_config is None:
        providers_config = []
    permission_mode = _resolved_claude_permission_mode(providers_config, claude_permission_mode)

    if spec == "smart":
        from openagent.models.smart_router import SmartRouter

        model: BaseModel = SmartRouter(
            routing=routing,
            providers_config=providers_config,
            api_key=api_key,
            monthly_budget=monthly_budget,
            classifier_model=classifier_model,
            claude_permission_mode=permission_mode,
        )
    elif is_claude_cli_model(spec):
        from openagent.models.claude_cli import ClaudeCLIRegistry

        bare = model_id_from_runtime(spec)
        default_model = bare if bare and bare != spec else None
        # The registry hosts one ClaudeCLI per session; the model the
        # live subprocess is pinned to can change mid-session via
        # ClaudeSDKClient.set_model(), so multiple claude-cli entries
        # in the ``models`` table can coexist without duplicating
        # subprocesses per model.
        model = ClaudeCLIRegistry(
            default_model=default_model,
            permission_mode=permission_mode,
            providers_config=providers_config,
            idle_timeout_seconds=claude_idle_timeout_seconds,
            hard_timeout_seconds=claude_hard_timeout_seconds,
            idle_ttl_seconds=claude_idle_ttl_seconds,
        )
    else:
        from openagent.models.agno_provider import AgnoProvider

        model = AgnoProvider(
            model=spec,
            api_key=api_key,
            base_url=base_url,
            providers_config=providers_config,
            db_path=getattr(db, "db_path", None),
        )

    return wire_model_runtime(model, db=db, mcp_pool=mcp_pool)


def create_model_from_config(config: dict) -> BaseModel:
    """Instantiate the active model from the resolved OpenAgent config.

    Always returns a SmartRouter — SmartRouter is the single top-level
    runtime and dispatches each session to either Agno or the Claude CLI
    registry internally (see ``openagent.models.smart_router``).

    Legacy ``model.provider`` values (``claude-cli``, ``anthropic``,
    ``agno``, etc.) are honored as a *routing hint*: the specified
    ``model_id`` is slotted into the routing tiers so existing yaml
    configs keep working. The ``models`` DB table is still the source
    of truth — SmartRouter rebuilds its routing from it on every
    hot-reload tick.
    """
    model_cfg = config.get("model", {})
    providers_config = config.get("providers") or []
    permission_mode = model_cfg.get("permission_mode", "bypass")
    api_key = model_cfg.get("api_key")
    provider = _canonical_provider_name(model_cfg.get("provider"))

    routing = dict(model_cfg.get("routing") or {})

    # Legacy compat: a yaml like ``model: { provider: claude-cli,
    # model_id: claude-sonnet-4-6 }`` becomes a SmartRouter whose every
    # tier points at the single claude-cli model. Users who want the
    # real auto-routing just drop ``model.provider`` or set it to
    # ``smart``.
    if provider not in ("smart", "agno") and not routing:
        try:
            legacy_spec, _ = _runtime_spec_from_config(model_cfg, providers_config)
        except ValueError:
            legacy_spec = ""
        if legacy_spec:
            for tier in ("simple", "medium", "hard", "fallback"):
                routing.setdefault(tier, legacy_spec)

    return create_model_from_spec(
        "smart",
        providers_config=providers_config,
        api_key=api_key,
        monthly_budget=float(model_cfg.get("monthly_budget", 0)),
        routing=routing or None,
        classifier_model=model_cfg.get("classifier_model"),
        claude_permission_mode=permission_mode,
    )


async def run_provider_smoke_test(
    provider_name: str,
    providers_config: Any,
    *,
    model_id: str | None = None,
    framework: str | None = None,
    session_id: str = "provider-test",
    prompt: str = "Say 'ok' and nothing else.",
) -> tuple[str, ModelResponse]:
    """Run a minimal prompt through the configured runtime for one provider.

    When the same vendor is registered under both frameworks
    (anthropic+agno AND anthropic+claude-cli), pass ``framework=`` to
    disambiguate — otherwise the first matching entry wins.
    """
    # Resolve the provider row by (name, framework) pair. Fall back to
    # the first entry that matches by name when framework is unspecified.
    cfg: dict[str, Any] | None = None
    for entry in _iter_provider_entries(providers_config):
        if str(entry.get("name") or "").strip() != provider_name:
            continue
        if framework and entry.get("framework") != framework:
            continue
        cfg = dict(entry)
        break
    if cfg is None:
        raise ValueError(f"Provider '{provider_name}' not configured")

    # If caller supplied a model_id that already encodes a framework
    # (e.g. ``claude-cli:anthropic:claude-opus-4-7``), honour it as-is.
    # Otherwise resolve a default scoped to the provider row's framework.
    if model_id and framework_of(model_id) == FRAMEWORK_CLAUDE_CLI:
        runtime_model = model_id
    else:
        runtime_model = model_id or get_default_model_for_provider(
            provider_name,
            providers_config,
            framework=cfg.get("framework"),
        )
        if not runtime_model:
            raise ValueError(f"No models configured for provider '{provider_name}'")
        runtime_model = normalize_runtime_model_id(runtime_model, providers_config)

    provider = create_model_from_spec(
        runtime_model,
        providers_config=providers_config,
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
    )
    resp = await provider.generate(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )
    return runtime_model, resp
