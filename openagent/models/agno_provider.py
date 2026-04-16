"""Agno-backed provider for API models.

OpenAgent owns the *product* layer:
- provider/model catalog
- pricing / budget reporting
- gateway, channels, memory vault

Agno owns the *runtime* layer:
- API call execution
- session history persistence
- MCP tool orchestration (via ``agno.tools.mcp.MCPTools`` instances supplied
  by the OpenAgent ``MCPPool`` — see ``openagent.mcp.pool``)

Tool wiring: this provider does NOT wrap MCP tools manually. It receives a
list of pre-connected Agno ``MCPTools`` instances from the pool and passes
them straight to the Agno ``Agent``. Agno handles the tool loop, content-type
serialisation (image artifacts, embedded resources, etc.), and per-call
scheduling. We only need to compute and mirror cost back into the metrics so
``agno_sessions.runs[*].metrics.cost`` stays queryable.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from openagent.core.logging import elog
from openagent.models.base import BaseModel, ModelResponse
from openagent.models.catalog import (
    DEFAULT_ZAI_BASE_URL,
    compute_cost,
    model_id_from_runtime,
    normalize_runtime_model_id,
    split_runtime_id,
)

logger = logging.getLogger(__name__)
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
        history_runs: int = 20,
    ):
        self._providers_config = providers_config or {}
        self.model = normalize_runtime_model_id(model, self._providers_config)
        self._api_key = api_key
        self._base_url = base_url
        self._db_path = db_path
        self._history_runs = history_runs
        # Pre-connected Agno MCPTools instances supplied by MCPPool. Shared
        # across all AgnoProvider instances under the same SmartRouter so we
        # don't spawn duplicate MCP server processes per tier.
        self._mcp_toolkits: list[Any] = []
        # One Agno Agent per (system_message) so the framework prompt is sent
        # as a real ``system`` role message — not buried inside the user
        # prompt. Without this, gpt-4o-mini ignores procedural instructions
        # like "introspect your tool list and group by prefix" and just
        # confabulates ("vault, functions"). Cache stays small in practice:
        # the classifier uses ``""`` (no system); the main call uses the
        # static framework+user prompt; that's two entries.
        self._agno_agents: dict[str, Any] = {}
        # Parallel cache for Team-mode runners. One Team per framework
        # system prompt, built only when ≥2 MCP tool families are connected.
        # Classifier calls (empty system) never trigger Team construction.
        self._agno_teams: dict[str, Any] = {}

        self._inject_provider_keys()

    def set_db(self, db) -> None:
        self._db_path = getattr(db, "db_path", self._db_path)
        # Force agent rebuild so the new SqliteDb path takes effect.
        self._agno_agents.clear()
        self._agno_teams.clear()

    def set_mcp_toolkits(self, toolkits: list[Any]) -> None:
        """Receive the pool's pre-connected Agno ``MCPTools`` instances.

        Called by ``wire_model_runtime``. The pool owns lifecycle (entered
        once at agent startup, exited at shutdown); we just hold references.
        """
        self._mcp_toolkits = list(toolkits)
        # Force agent/team rebuild so the new tool list is picked up.
        self._agno_agents.clear()
        self._agno_teams.clear()

    async def close_session(self, session_id: str) -> None:
        """Agno persists session history in DB but keeps no per-session subprocess."""
        return None

    def _provider_name(self) -> str:
        return split_runtime_id(self.model)[0]

    def _inject_provider_keys(self) -> None:
        # NOTE: this mutates ``os.environ`` from a constructor — surprising but
        # intentional. Agno's provider classes (``Claude``, ``OpenAIChat``,
        # ``Gemini``, …) read API keys from process env vars, not from
        # constructor args we pass through. We export keys here so Agno can find
        # them. Two ``AgnoProvider`` instances with different keys for the same
        # provider will race; in practice OpenAgent uses one key per provider so
        # it's fine. Keys already in the env are not overwritten.
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

    def _runtime_parts(self) -> tuple[str, str]:
        return split_runtime_id(self.model)

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

    def _build_list_mcps_tool(self) -> Callable[..., str]:
        """Return a callable the LLM can invoke to discover MCP servers.

        Weak models (e.g. gpt-4o-mini) reliably fail to introspect 75+
        function definitions and group them by their ``<server>_`` prefix —
        empirically they confabulate ("vault, functions") even with the
        full tool list and proper system role. Giving them a callable that
        returns the authoritative server inventory means they don't need
        to introspect at all: they call this tool, then relay the result.

        The callable's docstring is its description for the LLM, which IS
        the appropriate place to describe a tool (this isn't injecting MCP
        names into the system prompt — server names are computed from the
        live toolkit list at call time, not baked into any string).
        """
        toolkits = self._mcp_toolkits

        def list_mcp_servers() -> str:
            """List every MCP server wired to this agent and how many tools each provides.

            Call this when the user asks ANY of: which MCPs do you have, what MCPs
            are available, what can you do, what tools do you have, list your
            capabilities, or anything semantically similar — in any language.
            Returns a JSON array of ``{"server": str, "tools": int}`` items. Each
            server's tools appear in your function list as ``<server>_<tool>``.
            Use the returned data verbatim; do not invent or omit servers.
            """
            import json
            result = []
            for tk in toolkits:
                prefix = (
                    getattr(tk, "tool_name_prefix", None)
                    or getattr(tk, "name", None)
                    or "?"
                )
                count = len(getattr(tk, "functions", {}) or {})
                result.append({"server": prefix, "tools": count})
            return json.dumps(result, indent=2)

        return list_mcp_servers

    def _ensure_agent(self, system: str | None = None):
        """Lazily construct one Agno Agent per unique system prompt.

        ``system_message`` is set at construction time so OpenAgent's framework
        prompt reaches OpenAI as a real ``system`` role message, not as user
        text. Agents are cached by the system string so we don't rebuild on
        every call. ``set_db`` and ``set_mcp_toolkits`` flush the cache.

        Tools passed to AgnoAgent: every connected MCPTools toolkit plus a
        ``list_mcp_servers`` meta-callable for inventory questions (see
        :meth:`_build_list_mcps_tool`).
        """
        sys_key = (system or "").strip()
        cached = self._agno_agents.get(sys_key)
        if cached is not None:
            return cached
        try:
            from agno.agent import Agent as AgnoAgent
            from agno.db.sqlite import SqliteDb
        except ImportError as exc:
            raise RuntimeError(self._missing_dependency_hint(exc)) from exc

        db_path = Path(self._runtime_db_path())
        db_path.parent.mkdir(parents=True, exist_ok=True)
        agent_tools: list[Any] = list(self._mcp_toolkits)
        agent_tools.append(self._build_list_mcps_tool())
        agent = AgnoAgent(
            model=self._build_agno_model(),
            db=SqliteDb(db_file=str(db_path)),
            tools=agent_tools,
            system_message=sys_key or None,
            add_history_to_context=True,
            num_history_runs=self._history_runs,
            # Agno maintains a rolling summary of older turns in the same
            # SqliteDb and injects it into context on each call. Combined
            # with ``num_history_runs=20`` this gives us long-horizon
            # recall without blowing the token budget on raw transcript.
            enable_session_summaries=True,
            add_session_summary_to_context=True,
            # Agentic memory lets the model persist user-scoped facts
            # across sessions (e.g. preferences the user restated). Uses
            # the agent's own model as the default memory manager.
            enable_agentic_memory=True,
            markdown=False,
        )
        self._agno_agents[sys_key] = agent
        return agent

    def _tool_families(self) -> dict[str, list[Any]]:
        """Group connected MCP toolkits by their server prefix.

        Each ``MCPTools`` instance wraps exactly one MCP server, so the
        prefix doubles as a natural "tool family" identifier for Team-mode
        routing. Uses the same resolution order as ``list_mcp_servers``
        (``tool_name_prefix`` → ``name`` → ``"default"``). Empty dict when
        no toolkits are connected.
        """
        families: dict[str, list[Any]] = {}
        for tk in self._mcp_toolkits:
            family = (
                getattr(tk, "tool_name_prefix", None)
                or getattr(tk, "name", None)
                or "default"
            )
            families.setdefault(str(family), []).append(tk)
        return families

    def _ensure_team(self, system: str):
        """Lazily construct an Agno Team in route mode for the main agent.

        Returns ``None`` when the team path is not applicable:
        - empty system prompt (classifier / no-framework-prompt calls)
        - fewer than 2 connected MCP tool families (nothing to route between)
        - Team import failure (older Agno or missing extra)

        The team leader carries the framework prompt and the
        ``list_mcp_servers`` meta-tool so inventory questions still work.
        Members are thin specialists: each owns one family's toolkit and
        a short role blurb the router uses to pick between them. Members
        inherit the framework prompt too so when the leader synthesises
        their output the persona is preserved.
        """
        sys_key = (system or "").strip()
        if not sys_key:
            return None
        cached = self._agno_teams.get(sys_key)
        if cached is not None:
            return cached

        families = self._tool_families()
        if len(families) < 2:
            return None

        try:
            from agno.agent import Agent as AgnoAgent
            from agno.db.sqlite import SqliteDb
            from agno.team import Team, TeamMode
        except ImportError as exc:
            # Older Agno builds without the team module — fall back to
            # single-agent transparently instead of crashing the session.
            elog("agno.team.unavailable", model=self.model, error=str(exc))
            return None

        db_path = Path(self._runtime_db_path())
        db_path.parent.mkdir(parents=True, exist_ok=True)

        members: list[Any] = []
        for family, toolkits in families.items():
            member_role = (
                f"Specialist for the {family} MCP server. "
                f"Handles requests best solved with {family} tools. "
                f"If a request falls outside that area, say so briefly."
            )
            member_system = (
                f"{sys_key}\n\n--- Role ---\nYou are the {family} "
                f"specialist. Prefer {family} tools; defer to the team "
                f"leader if the request is outside your area."
            )
            member = AgnoAgent(
                model=self._build_agno_model(),
                tools=list(toolkits),
                system_message=member_system,
                name=f"{family}_specialist",
                role=member_role,
                markdown=False,
            )
            members.append(member)

        # Leader gets the full toolkit list as a safety net so if gpt-4o-mini
        # (or any routing-weak model) decides to answer directly instead of
        # delegating, it still has the tools it needs. In normal route-mode
        # operation the leader delegates to one specialist and never calls
        # these itself — so the fallback costs nothing on the happy path.
        leader_tools: list[Any] = list(self._mcp_toolkits)
        leader_tools.append(self._build_list_mcps_tool())

        try:
            team = Team(
                members=members,
                mode=TeamMode.route,
                model=self._build_agno_model(),
                db=SqliteDb(db_file=str(db_path)),
                tools=leader_tools,
                system_message=sys_key,
                # Surface each member's tool list in the leader's context so
                # routing decisions see capabilities, not just the member's
                # short role blurb. Without this, weak routing models tend
                # to answer directly ("I cannot access files") when the
                # member name alone doesn't make the match obvious.
                add_member_tools_to_context=True,
                add_history_to_context=True,
                num_history_runs=self._history_runs,
                enable_session_summaries=True,
                add_session_summary_to_context=True,
                enable_agentic_memory=True,
                markdown=False,
            )
        except Exception as exc:
            # If Team construction fails for any reason (signature drift
            # between Agno versions, model incompatibility, …), log and
            # fall back rather than breaking the main generate path.
            elog(
                "agno.team.build_failed",
                model=self.model,
                error_type=type(exc).__name__,
                error=str(exc),
                families=sorted(families.keys()),
            )
            return None

        elog(
            "agno.team.built",
            model=self.model,
            families=sorted(families.keys()),
            member_count=len(members),
        )
        self._agno_teams[sys_key] = team
        return team

    def _flatten_messages(self, messages: list[dict[str, Any]]) -> str:
        """Render conversation turns as a single user-side prompt for ``arun``.

        The system prompt is NOT included here — it's set on the AgnoAgent via
        ``system_message`` (see ``_ensure_agent``) so OpenAI receives it as a
        real ``system`` role message. Including it here would duplicate it as
        user text, undoing the fix that makes procedural instructions
        authoritative for weak models.
        """
        parts: list[str] = []
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

    def _metrics_to_dict(self, metrics: Any) -> dict[str, Any]:
        """Coerce Agno's metrics (dataclass / Pydantic / dict / object) into a dict.

        Agno 2.x returns ``response.metrics`` as a ``RunMetrics`` dataclass;
        older versions returned a dict or a Pydantic model. Normalise so token
        and cost extraction works across versions.
        """
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            return metrics
        if hasattr(metrics, "model_dump"):
            try:
                return metrics.model_dump()
            except Exception:
                pass
        if hasattr(metrics, "__dataclass_fields__"):
            from dataclasses import asdict
            try:
                return asdict(metrics)
            except Exception:
                pass
        if hasattr(metrics, "__dict__"):
            return {k: v for k, v in vars(metrics).items() if not k.startswith("_")}
        return {}

    def _extract_metric(self, metrics: Any, *names: str) -> int:
        data = metrics if isinstance(metrics, dict) else self._metrics_to_dict(metrics)
        if not data:
            return 0
        for name in names:
            value = data.get(name)
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
        """Run a single turn through Agno.

        Note: ``tools`` is accepted for ``BaseModel`` compatibility but
        ignored — Agno's Agent already holds the configured ``MCPTools``
        instances via ``_mcp_toolkits`` and runs the full tool loop
        internally (including image-artifact extraction so screenshots no
        longer blow up the context).
        """
        prompt = self._flatten_messages(messages)
        sid = session_id or "default"
        # Prefer a Team when the caller supplied a system prompt AND we
        # have ≥2 MCP tool families to route between. Classifier/no-system
        # calls stay on single Agent so they don't pay the leader-routing
        # round trip. _ensure_team() returns None whenever the team path
        # is not applicable, so this collapses to the single-agent path
        # in minimal deployments.
        runner = self._ensure_team(system=system or "")
        runner_kind = "team"
        if runner is None:
            runner = self._ensure_agent(system=system)
            runner_kind = "agent"
        elog(
            "agno.request",
            model=self.model,
            session_id=sid,
            prompt_len=len(prompt),
            mcp_toolkits=len(self._mcp_toolkits),
            runner=runner_kind,
        )

        if on_status:
            try:
                await on_status("Thinking...")
            except Exception:
                pass

        try:
            # ``user_id`` is required by ``enable_agentic_memory``; we use a
            # stable constant so memory accumulates for the single-tenant
            # OpenAgent deployment. Multi-tenant deployments can wire a
            # real user_id through BaseModel.generate() in a follow-up.
            response = await runner.arun(
                prompt, session_id=sid, user_id="openagent"
            )
        except Exception as e:
            elog(
                "agno.error",
                model=self.model,
                session_id=sid,
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
            raise

        # Agno occasionally returns a response whose ``.content`` is None or
        # empty string (tool-only turn, provider returned an empty choice,
        # length cap hit mid-text). Surface it as a structured event and
        # substitute a non-empty placeholder so bridges don't forward zero
        # bytes to the user. Symmetrical with claude_cli.py's empty-result
        # handling.
        raw_content = getattr(response, "content", None)
        if raw_content:
            content = raw_content
        else:
            logger.warning(
                "Agno returned empty content (model=%s, session=%s, response=%r)",
                self.model, sid, response,
            )
            elog(
                "agno.empty_result",
                model=self.model,
                session_id=sid,
                response_type=type(response).__name__,
            )
            content = "(Done — no final message was returned.)"
        metrics_obj = getattr(response, "metrics", None)
        metrics_dict = self._metrics_to_dict(metrics_obj)

        # Trace event so we can debug if Agno changes the metrics shape again.
        elog(
            "agno.metrics.shape",
            model=self.model,
            session_id=sid,
            type=type(metrics_obj).__name__ if metrics_obj is not None else "None",
            keys=sorted(metrics_dict.keys())[:12] if metrics_dict else [],
        )

        input_tokens = self._extract_metric(metrics_dict, "input_tokens", "prompt_tokens", "input")
        output_tokens = self._extract_metric(metrics_dict, "output_tokens", "completion_tokens", "output")
        stop_reason = metrics_dict.get("stop_reason") if isinstance(metrics_dict, dict) else None

        # Compute cost from OpenAgent's catalog and mirror it back into Agno's
        # metrics so SessionMetrics.cost aggregation works for free.
        cost = self._compute_and_mirror_cost(
            metrics_obj=metrics_obj,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            session_id=sid,
        )

        elog(
            "agno.generate",
            model=self.model,
            session_id=sid,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            stop_reason=stop_reason or "stop",
        )
        return ModelResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason or "stop",
            model=self.model,
        )

    def _compute_and_mirror_cost(
        self,
        *,
        metrics_obj: Any,
        input_tokens: int,
        output_tokens: int,
        session_id: str,
    ) -> float:
        """Compute cost from OpenAgent's catalog and write it onto Agno's metrics.

        Agno propagates the ``cost`` field through ``MessageMetrics → RunMetrics
        → SessionMetrics``, but never populates it (provider SDKs don't return
        cost). By mutating ``metrics_obj.cost`` (and the per-(provider, id)
        entries in ``metrics.details``) we make Agno's session-level cost
        aggregation work — so ``agno_sessions.runs[*].metrics.cost`` becomes
        directly queryable and ``SessionMetrics`` sums correctly across runs.

        The canonical cost record still lives in OpenAgent's ``usage_log``
        (written by ``SmartRouter``); this is the queryable mirror.
        """
        cost = compute_cost(
            model_ref=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            providers_config=self._providers_config,
        )

        if input_tokens == 0 and output_tokens == 0:
            elog(
                "agno.cost_skipped",
                model=self.model,
                session_id=session_id,
                reason="zero_tokens",
            )
            return cost

        if metrics_obj is None:
            elog(
                "agno.cost_skipped",
                model=self.model,
                session_id=session_id,
                reason="no_metrics_object",
                cost_usd=cost,
            )
            return cost

        bare_id = model_id_from_runtime(self.model)
        mirrored_targets: list[str] = []

        # 1. Top-level RunMetrics.cost
        try:
            setattr(metrics_obj, "cost", cost)
            mirrored_targets.append("run")
        except (AttributeError, TypeError):
            pass

        # 2. Per-(provider, id) ModelMetrics.cost in details["model"], details["output_model"], …
        details = getattr(metrics_obj, "details", None)
        if isinstance(details, dict):
            for model_type, entries in details.items():
                if not entries:
                    continue
                for entry in entries:
                    entry_id = getattr(entry, "id", None)
                    if entry_id and (entry_id == bare_id or entry_id == self.model):
                        try:
                            setattr(entry, "cost", cost)
                            mirrored_targets.append(f"details.{model_type}[{entry_id}]")
                        except (AttributeError, TypeError):
                            pass

        elog(
            "agno.cost_mirrored",
            model=self.model,
            session_id=session_id,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            targets=mirrored_targets,
        )
        return cost
