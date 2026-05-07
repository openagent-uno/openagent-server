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
from typing import Any, AsyncIterator, Awaitable, Callable

from openagent.core.logging import elog
from openagent.models.base import BaseModel, ModelResponse
from openagent.models.catalog import (
    DEFAULT_ZAI_BASE_URL,
    FRAMEWORK_AGNO,
    _iter_provider_entries,
    compute_cost,
    model_id_from_runtime,
    normalize_runtime_model_id,
    split_runtime_id,
)

logger = logging.getLogger(__name__)
_INCOMPATIBLE_TOOL_FAMILIES_BY_PROVIDER: dict[str, frozenset[str]] = {
    # DeepSeek v4 flash/pro chat completions are text-only on the official
    # API, so Agno computer-control screenshot artifacts (image parts) fail
    # with "unknown variant image_url, expected text". Filter that toolkit
    # out up front instead of letting sessions crash mid-turn.
    "deepseek": frozenset({"computer_control"}),
}


class AgnoProviderError(RuntimeError):
    """Raised when Agno's underlying provider failed (e.g. OpenAI 403,
    rate-limit, model-not-allowed) but Agno swallowed the exception and
    returned an empty response. We capture the ERROR log record(s) and
    re-raise with a user-readable message so the chat UI can show what
    actually went wrong instead of a silent placeholder."""


def _extract_tool_names_from_agno_response(response: Any) -> list[str]:
    """Best-effort extraction of executed tool names from an Agno RunResponse.

    Agno surfaces executed tools in different shapes across versions:
    ``response.tools`` (list of ``ToolExecution`` with ``.tool_name``),
    ``response.tool_executions``, or nested under ``response.run_response``.
    We probe each candidate and return the first non-empty list of names.
    Returns ``[]`` on miss — telemetry is best-effort, not load-bearing.
    """
    candidates = [
        getattr(response, "tools", None),
        getattr(response, "tool_executions", None),
        getattr(getattr(response, "run_response", None), "tools", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        names: list[str] = []
        for entry in candidate:
            name = None
            if isinstance(entry, dict):
                name = entry.get("tool_name") or entry.get("name")
            else:
                name = getattr(entry, "tool_name", None) or getattr(entry, "name", None)
            if name:
                names.append(str(name))
        if names:
            return names
    return []


def _summarize_provider_errors(errs: list[str]) -> str:
    """Pick the most useful line from a list of captured ERROR log
    messages. Agno emits a small flurry of three lines for one provider
    failure ("API status error", "Non-retryable model provider error",
    "Error in Team run") — the second one is the cleanest and shortest,
    so we prefer that pattern, then fall back to the last record.

    Falls back to joining all of them when nothing matches the known
    Agno phrasing.
    """
    if not errs:
        return "Provider returned no content"
    # Prefer the "Non-retryable model provider error: <reason>" line
    # since that's the cleanest provider-side message.
    for line in errs:
        if "Non-retryable model provider error:" in line:
            return line.split("Non-retryable model provider error:", 1)[1].strip() or line
    # Otherwise prefer the API status error which carries the HTTP code.
    for line in errs:
        if "API status error" in line or "status_code" in line.lower():
            return line.strip()
    # Fall back to the last error message.
    return errs[-1].strip()


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

    async def forget_session(self, session_id: str) -> None:
        """Erase Agno's stored history for ``session_id`` so the next
        call on that session id starts empty.

        Without this, ``add_history_to_context=True`` (see
        :meth:`_ensure_agent`) reloads the full prior transcript AND
        any rolling session summary on the next generate() — so the
        gateway's ``/clear`` and the scheduler's per-fire forget both
        silently break for agno-backed models.

        Strategy: call Agno's native ``SqliteDb.delete_session`` (which
        drops the ``agno_sessions`` row carrying both the transcript
        and the summary). Fall back to raw SQL if the API ever changes.
        The ``_agno_agents`` / ``_agno_teams`` caches don't need to
        be invalidated — the live ``Agent`` re-reads history from the
        DB on every ``arun()``, so deletion takes effect immediately.
        Agentic memory rows (``agno_memories``) are user-scoped by
        design; wiping them per-session would contradict the product
        model (user-level preferences should survive /clear).
        """
        if not session_id:
            return
        try:
            from agno.db.sqlite import SqliteDb
        except ImportError:
            return

        db_path = self._runtime_db_path()
        try:
            db = SqliteDb(db_file=db_path)
        except Exception as e:
            logger.debug("agno forget_session: SqliteDb init failed: %s", e)
            await self._agno_fallback_sql_forget(db_path, session_id)
            elog("agno.session_forget", session_id=session_id, db=db_path)
            return

        delete_fn = getattr(db, "delete_session", None)
        deleted_via_api = False
        if callable(delete_fn):
            try:
                result = delete_fn(session_id=session_id)
                if inspect.isawaitable(result):
                    await result
                deleted_via_api = True
            except Exception as e:
                logger.debug(
                    "agno delete_session failed for %s: %s", session_id, e,
                )
        if not deleted_via_api:
            await self._agno_fallback_sql_forget(db_path, session_id)
        elog("agno.session_forget", session_id=session_id, db=db_path)

    async def _agno_fallback_sql_forget(
        self, db_path: str, session_id: str
    ) -> None:
        """Raw-SQL delete for when Agno's ``SqliteDb.delete_session``
        is unavailable or errors out.

        Agno 2.x keeps session history and summary in ``agno_sessions``;
        earlier/alternate schemas may call it ``sessions``. Drop the
        row in whichever of those tables exists — a missing table is
        a no-op, not an error. Read-only when no relevant table is
        present (brand-new DB before Agno has created its schema).
        """
        import sqlite3
        candidate_tables = ("agno_sessions", "sessions")
        try:
            conn = sqlite3.connect(db_path)
        except Exception as e:
            logger.debug("agno fallback sql: connect %s failed: %s", db_path, e)
            return
        try:
            cur = conn.cursor()
            present = {
                row[0]
                for row in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            for table in candidate_tables:
                if table not in present:
                    continue
                try:
                    cur.execute(
                        f"DELETE FROM {table} WHERE session_id = ?",
                        (session_id,),
                    )
                except sqlite3.OperationalError as e:
                    logger.debug(
                        "agno fallback delete from %s for %s failed: %s",
                        table, session_id, e,
                    )
            conn.commit()
        finally:
            conn.close()

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

        # Only agno-framework provider rows carry api_keys worth exporting.
        # claude-cli rows have api_key=NULL by schema, but be defensive
        # against legacy dict-shaped configs that might still be in play.
        for entry in _iter_provider_entries(self._providers_config):
            if entry.get("framework", FRAMEWORK_AGNO) != FRAMEWORK_AGNO:
                continue
            name = str(entry.get("name") or "").strip()
            key = entry.get("api_key")
            if not name or not key:
                continue
            env_var = PROVIDER_ENV_VARS.get(name)
            if env_var and not os.environ.get(env_var):
                os.environ[env_var] = key
            if name == "google" and not os.environ.get("GEMINI_API_KEY"):
                os.environ["GEMINI_API_KEY"] = key

        if self._base_url and provider_name == "openai" and not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = self._base_url

    def _runtime_db_path(self) -> str:
        if self._db_path:
            return str(self._db_path)
        from openagent.core.paths import default_db_path

        return str(default_db_path())

    async def commit_partial_assistant(self, session_id: str, text: str) -> None:
        """Append a synthetic assistant run to Agno's session row.

        After barge-in, the in-flight Agno run is cancelled mid-stream and
        nothing has been committed to ``agno_sessions.runs``. We append a
        synthetic ``RunOutput`` with ``status=cancelled`` carrying the
        partial assistant text, so the next turn (which reads the session
        via ``add_history_to_context=True``) sees ``user → assistant
        (interrupted) → user`` rather than two adjacent user turns.

        Best-effort: missing session row, schema mismatch, or import
        errors all degrade silently — barge-in must succeed even when
        history persistence fails.
        """
        if not session_id or not text:
            return
        try:
            from agno.db.sqlite import SqliteDb
            from agno.db.base import SessionType
            from agno.session.agent import AgentSession
            from agno.run.agent import RunOutput
            from agno.run.base import RunStatus
            from agno.models.message import Message
        except ImportError as e:
            logger.debug("agno commit_partial_assistant: import failed: %s", e)
            return

        db_path = self._runtime_db_path()
        try:
            db = SqliteDb(db_file=db_path)
        except Exception as e:  # noqa: BLE001
            logger.debug("agno commit_partial_assistant: SqliteDb init failed: %s", e)
            return

        try:
            session = db.get_session(
                session_id=session_id,
                session_type=SessionType.AGENT,
                deserialize=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("agno commit_partial_assistant: get_session %s: %s", session_id, e)
            return

        if not isinstance(session, AgentSession):
            return

        import time as _time
        # ``agent_id`` is REQUIRED on the run dict — Agno's
        # ``AgentSession.from_dict`` filters out any run that lacks both
        # ``agent_id`` and ``team_id`` when deserialising. Without it,
        # our synth run round-trips through SqliteDb and silently
        # disappears on the next read.
        run = RunOutput(
            run_id=f"interrupted-{int(_time.time() * 1000)}",
            agent_id=session.agent_id,
            session_id=session_id,
            user_id=session.user_id,
            content=text,
            messages=[Message(role="assistant", content=text)],
            status=RunStatus.cancelled,
        )
        try:
            session.upsert_run(run)
        except Exception as e:  # noqa: BLE001
            logger.debug("agno commit_partial_assistant: upsert_run failed: %s", e)
            return
        try:
            db.upsert_session(session, deserialize=False)
            elog("agno.barge_in_commit", session_id=session_id, chars=len(text))
        except Exception as e:  # noqa: BLE001
            logger.debug("agno commit_partial_assistant: upsert_session failed: %s", e)

    def _runtime_parts(self) -> tuple[str, str]:
        return split_runtime_id(self.model)

    @staticmethod
    def _toolkit_family_name(toolkit: Any) -> str:
        return str(
            getattr(toolkit, "tool_name_prefix", None)
            or getattr(toolkit, "name", None)
            or "default"
        )

    def _compatible_mcp_toolkits(self) -> tuple[list[Any], list[str]]:
        provider_name, _model_id = self._runtime_parts()
        blocked = _INCOMPATIBLE_TOOL_FAMILIES_BY_PROVIDER.get(provider_name, frozenset())
        if not blocked:
            return list(self._mcp_toolkits), []
        allowed: list[Any] = []
        filtered: set[str] = set()
        for toolkit in self._mcp_toolkits:
            family = self._toolkit_family_name(toolkit)
            if family in blocked:
                filtered.add(family)
                continue
            allowed.append(toolkit)
        return allowed, sorted(filtered)

    def _rewrite_provider_error_detail(self, detail: str) -> str:
        provider_name, _model_id = self._runtime_parts()
        lowered = (detail or "").lower()
        if provider_name == "deepseek" and "unknown variant image_url" in lowered:
            return (
                "DeepSeek's official chat API only accepts text message content, "
                "so Agno computer-control screenshots are incompatible with this "
                "model. Use a non-DeepSeek model for computer-control / GUI tasks."
            )
        return detail

    def _provider_config(self) -> dict[str, Any]:
        """Return the agno-framework provider entry matching this model's vendor.

        v0.12 stores providers as a flat list where the same vendor can
        appear twice (agno + claude-cli). AgnoProvider only cares about
        the agno row — the claude-cli row lives in its own registry.
        Falls back to the legacy dict-shape for early-boot / tests.
        """
        provider_name, _ = self._runtime_parts()
        for entry in _iter_provider_entries(self._providers_config):
            if str(entry.get("name") or "").strip() != provider_name:
                continue
            if entry.get("framework", FRAMEWORK_AGNO) != FRAMEWORK_AGNO:
                continue
            return dict(entry)
        return {}

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
        toolkits, _filtered = self._compatible_mcp_toolkits()

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
        compatible_toolkits, filtered_families = self._compatible_mcp_toolkits()
        if filtered_families:
            elog(
                "agno.toolkits_filtered",
                model=self.model,
                filtered_families=filtered_families,
                runner="agent",
            )
        agent_tools: list[Any] = list(compatible_toolkits)
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
        compatible_toolkits, _filtered = self._compatible_mcp_toolkits()
        families: dict[str, list[Any]] = {}
        for tk in compatible_toolkits:
            family = self._toolkit_family_name(tk)
            families.setdefault(family, []).append(tk)
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

        compatible_toolkits, filtered_families = self._compatible_mcp_toolkits()
        families = self._tool_families()
        if len(families) < 2:
            return None
        if filtered_families:
            elog(
                "agno.toolkits_filtered",
                model=self.model,
                filtered_families=filtered_families,
                runner="team",
            )

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
        leader_tools: list[Any] = list(compatible_toolkits)
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
        compatible_toolkits, filtered_families = self._compatible_mcp_toolkits()
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
            mcp_toolkits=len(compatible_toolkits),
            filtered_families=filtered_families or None,
            runner=runner_kind,
        )

        if on_status:
            try:
                await on_status("Thinking...")
            except Exception:
                pass

        # Capture ERROR-level log records from any logger during the run so
        # that when Agno swallows a provider failure (e.g. OpenAI 403,
        # rate-limit, model-not-allowed), we can surface the underlying
        # message to the user instead of returning the silent empty-result
        # placeholder. Agno logs the failure with strings like
        # "API status error from OpenAI API: Error code: 403 - ..." and
        # "Non-retryable model provider error: ...", but does not re-raise.
        captured_errors: list[str] = []

        class _ErrorCapture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.levelno < logging.ERROR:
                    return
                try:
                    msg = record.getMessage()
                except Exception:
                    msg = str(record.msg)
                if msg:
                    captured_errors.append(msg)

        capture = _ErrorCapture(level=logging.ERROR)
        root_logger = logging.getLogger()
        root_logger.addHandler(capture)
        try:
            # ``user_id`` is required by ``enable_agentic_memory``; we use a
            # stable constant so memory accumulates for the single-tenant
            # OpenAgent deployment. Multi-tenant deployments can wire a
            # real user_id through BaseModel.generate() in a follow-up.
            try:
                response = await runner.arun(
                    prompt, session_id=sid, user_id="openagent"
                )
            except Exception as e:
                raw_error = str(e) or repr(e)
                rewritten_error = self._rewrite_provider_error_detail(raw_error)
                elog(
                    "agno.error",
                    model=self.model,
                    session_id=sid,
                    error_type=type(e).__name__,
                    error=rewritten_error,
                )
                if rewritten_error != raw_error:
                    raise AgnoProviderError(rewritten_error) from e
                raise
        finally:
            root_logger.removeHandler(capture)

        # Agno occasionally returns a response whose ``.content`` is None or
        # empty string. Two cases to distinguish:
        #   1. A *legitimate* tool-only turn — no error logs captured. Fall
        #      back to the placeholder so bridges don't ship zero bytes.
        #   2. A *swallowed provider failure* — Agno logged ERRORs but did
        #      not re-raise (this is the OpenAI 403 / rate-limit case the
        #      user reported). Raise a clean exception with the captured
        #      message so ``agent.run()`` formats it for the chat UI.
        raw_content = getattr(response, "content", None)
        if raw_content:
            content = raw_content
        else:
            if captured_errors:
                detail = self._rewrite_provider_error_detail(
                    _summarize_provider_errors(captured_errors)
                )
                elog(
                    "agno.provider_error",
                    level="error",
                    model=self.model,
                    session_id=sid,
                    detail=detail[:300],
                )
                raise AgnoProviderError(detail)
            elog(
                "agno.empty_result",
                level="warning",
                model=self.model,
                session_id=sid,
                response_type=type(response).__name__,
                response=repr(response)[:200],
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
        tool_names_called = _extract_tool_names_from_agno_response(response)
        return ModelResponse(
            content=content,
            tool_names_called=tool_names_called,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason or "stop",
            model=self.model,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream content deltas via Agno's native ``stream=True`` path.

        Yields raw text strings as they arrive from the LLM. Tool-call
        events and other meta events are dropped — voice-mode UX only
        needs the speakable text. On any failure, falls back to
        :meth:`generate` and yields the full content as one chunk so the
        caller still gets a reply (just without time-to-first-audio).

        ``session_id`` is forwarded into Agno's ``arun(session_id=...)``
        so each chat tab keeps its own RAM history. The previous
        hardcoded ``"default"`` collided every concurrent stream — two
        browser tabs would stomp on each other's history mid-turn.

        ``on_status`` is accepted for parity with
        :class:`ClaudeCLIRegistry.stream` (so ``Agent._run_inner_stream``
        can introspect-and-forward without provider-specific branching);
        Agno doesn't surface tool-running statuses through its content
        stream so this is currently a no-op.
        """
        prompt = self._flatten_messages(messages)
        sid = session_id or "default"
        compatible_toolkits, filtered_families = self._compatible_mcp_toolkits()
        runner = self._ensure_team(system=system or "")
        if runner is None:
            runner = self._ensure_agent(system=system)
        elog(
            "agno.stream.start",
            model=self.model,
            prompt_len=len(prompt),
            mcp_toolkits=len(compatible_toolkits),
            filtered_families=filtered_families or None,
        )
        try:
            # Lazy import — Agno event types are only needed on the streaming
            # path, and we don't want to require them on every import.
            from agno.run.agent import RunContentEvent

            stream = runner.arun(
                prompt, session_id=sid, user_id="openagent", stream=True,
            )
            try:
                emitted = 0
                async for event in stream:
                    # Only RunContentEvent carries text deltas; other events
                    # (tool starts, reasoning, run lifecycle) are noise from
                    # the TTS pipeline's perspective.
                    if isinstance(event, RunContentEvent):
                        text = getattr(event, "content", None) or ""
                        if text:
                            emitted += len(text)
                            yield text
            finally:
                # Agno's iterator may need explicit closing — best-effort.
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
            elog("agno.stream.done", model=self.model, chars=emitted)
        except Exception as e:
            elog(
                "agno.stream.fallback",
                level="warning",
                model=self.model,
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
            response = await self.generate(
                messages,
                system=system,
                tools=tools,
                on_status=on_status,
                session_id=session_id,
            )
            if response.content:
                yield response.content

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
