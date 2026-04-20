"""Shared runtime helpers for building and testing chat models."""

from __future__ import annotations

from typing import Any

from openagent.models.base import BaseModel, ModelResponse
from openagent.models.catalog import (
    FRAMEWORK_CLAUDE_CLI,
    _iter_provider_entries,
    framework_of,
    get_default_model_for_provider,
    is_claude_cli_model,
    model_id_from_runtime,
    normalize_runtime_model_id,
)


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
    db: Any = None,
    mcp_pool: Any = None,
) -> BaseModel:
    """Create a model instance from a compact OpenAgent runtime spec."""
    if providers_config is None:
        providers_config = []

    if spec == "smart":
        from openagent.models.smart_router import SmartRouter

        model: BaseModel = SmartRouter(providers_config=providers_config)
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
            providers_config=providers_config,
        )
    else:
        from openagent.models.agno_provider import AgnoProvider

        model = AgnoProvider(
            model=spec,
            providers_config=providers_config,
            db_path=getattr(db, "db_path", None),
        )

    return wire_model_runtime(model, db=db, mcp_pool=mcp_pool)


def create_model_from_config(config: dict) -> BaseModel:
    """Instantiate the active model from the resolved OpenAgent config.

    Always returns a SmartRouter — SmartRouter is the single top-level
    runtime and dispatches each session to either Agno or the Claude CLI
    registry internally (see ``openagent.models.smart_router``). The
    ``providers`` / ``models`` SQLite tables are the sole source of
    truth for the catalog; SmartRouter starts empty and gets its routing
    populated by ``Agent.initialize`` (and every hot-reload tick) via
    ``rebuild_routing``. The yaml is never consulted for provider or
    model state.
    """
    del config  # catalog comes from the DB, not yaml
    return create_model_from_spec("smart", providers_config=[])


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
    )
    resp = await provider.generate(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )
    return runtime_model, resp
