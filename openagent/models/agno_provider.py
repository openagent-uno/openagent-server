"""Agno-backed provider for API models with session-managed history.

OpenAgent continues to own:
- the product gateway
- provider/model catalog
- pricing/reporting
- Obsidian/wiki memory and MCP topology

Agno is used here strictly as the execution engine for API-backed models plus
session history persistence. No Agno memory or knowledge stores are configured.
"""

from __future__ import annotations

import importlib
import inspect
import keyword
import logging
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from openagent.core.logging import elog
from openagent.models.base import BaseModel, ModelResponse
from openagent.models.catalog import normalize_runtime_model_id

logger = logging.getLogger(__name__)
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "ZAI_API_KEY",
    "zhipu": "ZAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}
AGNO_PROVIDER_CLASSES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "anthropic": ("agno.models.anthropic", "Claude", {}),
    "openai": ("agno.models.openai", "OpenAIChat", {}),
    "google": ("agno.models.google", "Gemini", {}),
    "openrouter": ("agno.models.openrouter", "OpenRouter", {}),
    "groq": ("agno.models.groq", "Groq", {}),
    "mistral": ("agno.models.mistral", "MistralChat", {}),
    "xai": ("agno.models.xai", "xAI", {}),
    "deepseek": ("agno.models.deepseek", "DeepSeek", {}),
    "cerebras": ("agno.models.cerebras", "Cerebras", {}),
    "zai": ("agno.models.openai.like", "OpenAILike", {"name": "ZAI"}),
}


class AgnoProvider(BaseModel):
    """API model provider backed by Agno sessions and tool orchestration."""

    history_mode = "platform"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        providers_config: dict | None = None,
        db_path: str | None = None,
        history_runs: int = 6,
    ):
        self._providers_config = providers_config or {}
        self.model = normalize_runtime_model_id(model, self._providers_config)
        self._api_key = api_key
        self._base_url = base_url
        self._db_path = db_path
        self._history_runs = history_runs
        self._mcp_registry = None
        self._agno_agents: dict[tuple[str, ...], Any] = {}
        self._tool_cache: dict[str, Callable[..., Awaitable[str]]] | None = None

        self._inject_provider_keys()

    def set_db(self, db) -> None:
        self._db_path = getattr(db, "db_path", self._db_path)
        self._agno_agents.clear()

    def set_mcp_registry(self, registry) -> None:
        self._mcp_registry = registry
        self._tool_cache = None
        self._agno_agents.clear()

    def _provider_name(self) -> str:
        runtime_id = self.model
        return runtime_id.split(":", 1)[0] if ":" in runtime_id else runtime_id.split("/", 1)[0]

    def _inject_provider_keys(self) -> None:
        provider_name = self._provider_name()
        if self._api_key:
            env_var = PROVIDER_ENV_VARS.get(provider_name)
            if env_var and not os.environ.get(env_var):
                os.environ[env_var] = self._api_key
            if provider_name == "google" and not os.environ.get("GEMINI_API_KEY"):
                os.environ["GEMINI_API_KEY"] = self._api_key

        for name, cfg in self._providers_config.items():
            env_var = PROVIDER_ENV_VARS.get(name)
            key = cfg.get("api_key")
            if env_var and key and not os.environ.get(env_var):
                os.environ[env_var] = key
            if name == "google" and key and not os.environ.get("GEMINI_API_KEY"):
                os.environ["GEMINI_API_KEY"] = key

        if self._base_url and provider_name == "openai" and not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = self._base_url

    def _runtime_db_path(self) -> str:
        if self._db_path:
            return str(self._db_path)
        from openagent.core.paths import default_db_path

        return str(default_db_path())

    def _sanitize_param_name(self, value: str) -> str:
        candidate = re.sub(r"\W+", "_", value).strip("_") or "arg"
        if candidate[0].isdigit():
            candidate = f"arg_{candidate}"
        if keyword.iskeyword(candidate):
            candidate = f"{candidate}_arg"
        return candidate

    def _build_tool_function(self, tool_def: dict[str, Any]) -> Callable[..., Awaitable[str]]:
        tool_name = tool_def["name"]
        schema = tool_def.get("input_schema") or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])

        async def _tool_handler(name: str, arguments: dict[str, Any]) -> str:
            if not self._mcp_registry:
                raise RuntimeError("MCP registry is not available for Agno tool execution")
            return await self._mcp_registry.call_tool(name, arguments)

        params: list[str] = []
        assignments: list[str] = ["    arguments = {}"]
        used_names: set[str] = set()
        for original_name in properties.keys():
            param_name = self._sanitize_param_name(original_name)
            while param_name in used_names:
                param_name = f"{param_name}_arg"
            used_names.add(param_name)
            default = "" if original_name in required else " = None"
            params.append(f"{param_name}{default}")
            assignments.append(f"    if {param_name} is not None:")
            assignments.append(f"        arguments[{original_name!r}] = {param_name}")

        signature = ", ".join(params)
        func_name = self._sanitize_param_name(tool_name)
        source_lines = [f"async def {func_name}({signature}):" if signature else f"async def {func_name}():", *assignments]
        source_lines.append(f"    return await _tool_handler({tool_name!r}, arguments)")
        namespace: dict[str, Any] = {}
        exec("\n".join(source_lines), {"_tool_handler": _tool_handler}, namespace)
        func = namespace[func_name]
        func.__name__ = func_name
        func.__doc__ = tool_def.get("description") or f"MCP tool: {tool_name}"
        return func

    def _build_tools(self) -> dict[str, Callable[..., Awaitable[str]]]:
        if self._tool_cache is not None:
            return self._tool_cache
        if not self._mcp_registry:
            self._tool_cache = {}
            return self._tool_cache
        self._tool_cache = {
            tool_def["name"]: self._build_tool_function(tool_def)
            for tool_def in self._mcp_registry.all_tools()
        }
        return self._tool_cache

    def _selected_tools(self, tools: list[dict[str, Any]] | None) -> list[Callable[..., Awaitable[str]]]:
        if not tools:
            return []
        requested_names = {str(tool.get("name", "")).strip() for tool in tools if tool.get("name")}
        available = self._build_tools()
        return [available[name] for name in requested_names if name in available]

    def _runtime_parts(self) -> tuple[str, str]:
        runtime_id = self.model
        if ":" in runtime_id:
            return runtime_id.split(":", 1)
        if "/" in runtime_id:
            return runtime_id.split("/", 1)
        return runtime_id, runtime_id

    def _provider_config(self) -> dict[str, Any]:
        provider_name, _ = self._runtime_parts()
        return self._providers_config.get(provider_name, {})

    def _provider_setting(self, key: str) -> str | None:
        value = self._provider_config().get(key)
        return str(value).strip() if value is not None else None

    def _resolved_api_key(self) -> str | None:
        return self._api_key or self._provider_setting("api_key")

    def _resolved_base_url(self) -> str | None:
        provider_name, _ = self._runtime_parts()
        if self._base_url:
            return self._base_url
        if provider_name == "zai":
            return self._provider_setting("base_url") or DEFAULT_ZAI_BASE_URL
        return self._provider_setting("base_url")

    def _construct_model(self, cls: type, **kwargs: Any) -> Any:
        accepted = inspect.signature(cls).parameters
        filtered = {k: v for k, v in kwargs.items() if v is not None and k in accepted}
        return cls(**filtered)

    def _load_agno_model_class(self, provider_name: str) -> tuple[type | None, dict[str, Any]]:
        spec = AGNO_PROVIDER_CLASSES.get(provider_name)
        if not spec:
            return None, {}
        module_name, class_name, extra_kwargs = spec
        module = importlib.import_module(module_name)
        return getattr(module, class_name), dict(extra_kwargs)

    def _build_agno_model(self) -> Any:
        provider_name, model_id = self._runtime_parts()
        api_key = self._resolved_api_key()
        base_url = self._resolved_base_url()
        model_class, extra_kwargs = self._load_agno_model_class(provider_name)
        if model_class is not None:
            return self._construct_model(
                model_class,
                id=model_id,
                api_key=api_key,
                base_url=base_url,
                **extra_kwargs,
            )

        from agno.models.utils import get_model

        return get_model(self.model)

    def _missing_dependency_hint(self, exc: ImportError) -> str:
        detail = str(exc) or exc.__class__.__name__
        return (
            "Agno runtime dependencies are incomplete. "
            "Install OpenAgent's API-model dependencies (for example "
            "`sqlalchemy`, provider SDKs like `openai`/`anthropic`/`google-genai`) "
            f"and retry. Original import error: {detail}"
        )

    def _ensure_agent(self, tools: list[dict[str, Any]] | None = None):
        selected_tools = self._selected_tools(tools)
        cache_key = tuple(sorted(func.__name__ for func in selected_tools))
        cached = self._agno_agents.get(cache_key)
        if cached is not None:
            return cached
        try:
            from agno.agent import Agent as AgnoAgent
            from agno.db.sqlite import SqliteDb
        except ImportError as exc:
            raise RuntimeError(self._missing_dependency_hint(exc)) from exc

        db_path = Path(self._runtime_db_path())
        db_path.parent.mkdir(parents=True, exist_ok=True)
        agent = AgnoAgent(
            model=self._build_agno_model(),
            db=SqliteDb(db_file=str(db_path)),
            tools=selected_tools,
            add_history_to_context=True,
            num_history_runs=self._history_runs,
            markdown=False,
        )
        self._agno_agents[cache_key] = agent
        return agent

    def _flatten_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> str:
        parts: list[str] = []
        if system:
            parts.append(f"[System Instructions]\n{system.strip()}")
        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", "") or "")
            if role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[Assistant] {content}")
            elif role == "tool":
                parts.append(f"[Tool:{msg.get('name', 'tool')}] {content}")
        return "\n\n".join(part for part in parts if part).strip()

    def _extract_metric(self, metrics: Any, *names: str) -> int:
        if not isinstance(metrics, dict):
            return 0
        for name in names:
            value = metrics.get(name)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return 0

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
    ) -> ModelResponse:
        prompt = self._flatten_messages(messages, system=system)
        sid = session_id or "default"
        agent = self._ensure_agent(tools)
        elog(
            "agno.request",
            model=self.model,
            session_id=sid,
            prompt_len=len(prompt),
            tools=len(self._selected_tools(tools)),
        )

        if on_status:
            try:
                await on_status("Thinking...")
            except Exception:
                pass

        try:
            response = await agent.arun(prompt, session_id=sid)
        except Exception as e:
            elog("agno.error", model=self.model, session_id=sid, error=str(e))
            raise

        content = getattr(response, "content", None) or str(response)
        metrics = getattr(response, "metrics", {}) or {}
        input_tokens = self._extract_metric(metrics, "input_tokens", "prompt_tokens", "input")
        output_tokens = self._extract_metric(metrics, "output_tokens", "completion_tokens", "output")
        stop_reason = metrics.get("stop_reason") if isinstance(metrics, dict) else None

        elog(
            "agno.generate",
            model=self.model,
            session_id=sid,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason or "stop",
        )
        return ModelResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason or "stop",
        )
