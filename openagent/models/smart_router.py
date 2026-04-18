"""Smart model router — the single top-level runtime.

SmartRouter is the ONLY active-model class wired into the Agent. It
dispatches each session to either the Agno stack ("agno") or the Claude
CLI registry ("claude-cli") based on:

  1. **Session binding** (``sdk_sessions`` + ``session_bindings`` tables)
     — once a session has been served by one side its conversation
     state lives there (Agno's SqliteDb vs Claude's own session store),
     so the router must keep subsequent turns on the same side.
  2. **Classifier** — for fresh sessions, a small LLM call (using the
     cheapest configured model) tags the user turn as simple / medium /
     hard and we look up the corresponding routing tier.
  3. **Budget** — when the monthly budget runs low we degrade to the
     fallback tier instead of the tier the classifier asked for.

Claude-cli and agno sessions are strictly isolated: once bound, a
session can't cross. If the bound side has no enabled models the router
fails cleanly ("No <side> model available for this bound session")
rather than silently falling through to the other side.

``history_mode`` is intentionally ``None`` — the gateway's
``SessionManager.bind_history_mode`` bails out on falsy modes, so
SmartRouter handles binding internally rather than surfacing a
contradictory "platform" declaration that isn't true in practice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from openagent.core.logging import elog
from openagent.models.base import BaseModel, ModelResponse
from openagent.models.budget import BudgetTracker
from openagent.models.catalog import (
    is_claude_cli_model,
    iter_configured_models,
    model_history_mode,
    normalize_runtime_model_id,
)
from openagent.models.runtime import create_model_from_spec, wire_model_runtime

logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """\
Classify this task as simple, medium, or hard.
- simple: greetings, short factual questions, text formatting, status checks, translations
- medium: code review, document summarization, multi-step reasoning, data analysis
- hard: complex architecture design, multi-file code generation, debugging across systems, research synthesis
Reply with ONLY one word: simple, medium, or hard."""

TIERS = ("simple", "medium", "hard")

DEFAULT_AUTO_ROUTING: dict[str, str] = {
    "simple": "openai:gpt-4o-mini",
    "medium": "openai:gpt-4.1-mini",
    "hard": "openai:gpt-4.1",
    "fallback": "openai:gpt-4o-mini",
}
DEFAULT_CLASSIFIER_MODEL = "openai:gpt-4o-mini"

HARD_HINTS = (
    "powerful model",
    "strong model",
    "stronger model",
    "best model",
    "top model",
    "most capable model",
    "use a powerful model",
    "use the best model",
    "use the strongest model",
    "modello potente",
    "modello forte",
    "modello migliore",
    "modello più potente",
)

# Canonical names used in DB ``session_bindings.provider`` and
# ``sdk_sessions.provider``. Exported for tests.
SIDE_AGNO = "agno"
SIDE_CLAUDE_CLI = "claude-cli"


@dataclass(frozen=True)
class RoutingDecision:
    requested_tier: str
    effective_tier: str
    reason: str
    primary_model: str
    candidates: list[str]
    bound_side: str | None = None


class SmartRouter(BaseModel):
    """Hybrid dispatcher covering both Agno and Claude CLI runtimes."""

    # Intentionally None: the gateway's SessionManager.bind_history_mode
    # bails when history_mode is falsy, so we opt out of the pre-bind
    # check and handle binding ourselves (see ``_session_side``).
    history_mode = None

    def __init__(
        self,
        routing: dict[str, str] | None = None,
        api_key: str | None = None,
        monthly_budget: float = 0.0,
        classifier_model: str | None = None,
        providers_config: dict | None = None,
        claude_permission_mode: str = "bypass",
    ):
        self._providers_config = providers_config or {}
        self._api_key = api_key
        self._monthly_budget = monthly_budget
        self._claude_permission_mode = claude_permission_mode

        self._budget: BudgetTracker | None = None
        self._db: Any = None
        self._mcp_pool: Any = None

        # Agno tier providers — keyed by runtime_id. Lazily created.
        self._agno_providers: dict[str, BaseModel] = {}
        # Single ClaudeCLIRegistry serving every claude-cli runtime_id,
        # lazily created on first claude-cli dispatch so pure-Agno
        # deployments don't pay for the import.
        self._claude_registry: BaseModel | None = None

        self._last_tier_by_session: dict[str, str] = {}
        # In-process mirror of session_bindings / sdk_sessions so the
        # routing decision doesn't re-hit the DB on every turn. Written
        # after first successful dispatch and kept in sync with the
        # ``close_session`` / ``forget_session`` wipes.
        self._session_side: dict[str, str] = {}

        self._explicit_routing = dict(routing) if routing else None
        self._routing = self._normalise_routing(routing) if routing else self._build_auto_routing()

        self._classifier_model = normalize_runtime_model_id(
            classifier_model or self._routing.get("simple") or DEFAULT_CLASSIFIER_MODEL,
            self._providers_config,
        )
        elog(
            "router.config",
            routing=self._routing,
            classifier_model=self._classifier_model,
            monthly_budget=self._monthly_budget,
        )

    # ── routing table ────────────────────────────────────────────────

    def _normalise_routing(self, routing: dict[str, str]) -> dict[str, str]:
        normalised: dict[str, str] = {}
        for tier, model_id in routing.items():
            runtime_id = normalize_runtime_model_id(model_id, self._providers_config)
            if runtime_id:
                normalised[tier] = runtime_id
        return normalised or self._build_auto_routing()

    def _build_auto_routing(self) -> dict[str, str]:
        """Build routing from every configured model (agno + claude-cli).

        Sorted by output cost so "simple" maps to cheapest, "hard" to
        most expensive. Ties are broken by insertion order (which for
        DB-sourced entries is provider/model_id).
        """
        entries = list(iter_configured_models(self._providers_config))
        models_with_price: list[tuple[str, float]] = [
            (e.runtime_id, float(e.output_cost_per_million or 0.0)) for e in entries
        ]
        if not models_with_price:
            routing = dict(DEFAULT_AUTO_ROUTING)
            elog("router.auto_routing_default", level="warning", routing=routing)
            return routing

        models_with_price.sort(key=lambda item: item[1])
        n = len(models_with_price)
        routing = {
            "simple": models_with_price[0][0],
            "medium": models_with_price[n // 2][0],
            "hard": models_with_price[-1][0],
            "fallback": models_with_price[0][0],
        }
        elog("router.auto_routing", routing=routing, candidates=n)
        return routing

    def rebuild_routing(self, providers_config: dict | None = None) -> None:
        """Called by the hot-reload loop when the ``models`` DB table changes."""
        if providers_config is not None:
            self._providers_config = providers_config or {}
        self._routing = (
            self._normalise_routing(self._explicit_routing)
            if self._explicit_routing
            else self._build_auto_routing()
        )
        self._classifier_model = normalize_runtime_model_id(
            self._classifier_model or self._routing.get("simple") or DEFAULT_CLASSIFIER_MODEL,
            self._providers_config,
        )
        elog("router.rebuilt", routing=self._routing)

    # ── runtime wiring ───────────────────────────────────────────────

    def set_db(self, db: Any) -> None:
        self._db = db
        self._budget = BudgetTracker(db, self._monthly_budget)
        for model in self._agno_providers.values():
            wire_model_runtime(model, db=db)
        if self._claude_registry is not None:
            wire_model_runtime(self._claude_registry, db=db)

    def set_mcp_pool(self, pool: Any) -> None:
        self._mcp_pool = pool
        for model in self._agno_providers.values():
            wire_model_runtime(model, mcp_pool=pool)
        if self._claude_registry is not None:
            wire_model_runtime(self._claude_registry, mcp_pool=pool)

    async def cleanup_idle(self) -> None:
        for model in self._agno_providers.values():
            fn = getattr(model, "cleanup_idle", None)
            if callable(fn):
                await fn()
        if self._claude_registry is not None:
            fn = getattr(self._claude_registry, "cleanup_idle", None)
            if callable(fn):
                await fn()

    async def shutdown(self) -> None:
        for model in self._agno_providers.values():
            fn = getattr(model, "shutdown", None)
            if callable(fn):
                await fn()
        if self._claude_registry is not None:
            fn = getattr(self._claude_registry, "shutdown", None)
            if callable(fn):
                await fn()

    async def close_session(self, session_id: str) -> None:
        if not session_id:
            return
        self._last_tier_by_session.pop(session_id, None)
        self._session_side.pop(session_id, None)
        # Wipe the agno-side binding too; claude-cli's own close_session
        # below keeps its sdk_sessions entry alive so `/clear` can
        # distinguish "release subprocess" from "forget conversation".
        if self._db is not None:
            try:
                await self._db.delete_session_binding(session_id)
            except Exception as e:  # noqa: BLE001 — best effort
                logger.debug("delete_session_binding %s: %s", session_id, e)
        for model in self._agno_providers.values():
            fn = getattr(model, "close_session", None)
            if callable(fn):
                await fn(session_id)
        if self._claude_registry is not None:
            fn = getattr(self._claude_registry, "close_session", None)
            if callable(fn):
                await fn(session_id)

    # ── dispatch plumbing ───────────────────────────────────────────

    def _get_agno_provider(self, runtime_id: str) -> BaseModel:
        if runtime_id not in self._agno_providers:
            self._agno_providers[runtime_id] = create_model_from_spec(
                runtime_id,
                providers_config=self._providers_config,
                api_key=self._api_key,
                claude_permission_mode=self._claude_permission_mode,
                db=self._db,
                mcp_pool=self._mcp_pool,
            )
        return self._agno_providers[runtime_id]

    def _get_claude_registry(self) -> BaseModel:
        if self._claude_registry is None:
            from openagent.models.claude_cli import ClaudeCLIRegistry

            self._claude_registry = ClaudeCLIRegistry(
                default_model=None,
                permission_mode=self._claude_permission_mode,
                providers_config=self._providers_config,
            )
            if self._db is not None:
                wire_model_runtime(self._claude_registry, db=self._db)
            if self._mcp_pool is not None:
                wire_model_runtime(self._claude_registry, mcp_pool=self._mcp_pool)
        return self._claude_registry

    @staticmethod
    def _side_for_model(runtime_id: str) -> str:
        return SIDE_CLAUDE_CLI if is_claude_cli_model(runtime_id) else SIDE_AGNO

    async def _hydrate_bound_side(self, session_id: str) -> str | None:
        """Populate the in-memory side cache from the DB once per session."""
        if session_id in self._session_side:
            return self._session_side[session_id]
        if self._db is None:
            return None
        try:
            side = await self._db.get_session_binding(session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("get_session_binding %s: %s", session_id, e)
            return None
        if side:
            self._session_side[session_id] = side
        return side

    async def _persist_bound_side(self, session_id: str, side: str) -> None:
        self._session_side[session_id] = side
        # Claude-cli bindings also land in ``sdk_sessions`` via the
        # registry's own write path (ClaudeCLI._persist_sdk_session);
        # the session_bindings row is useful but redundant, so skip it.
        if side == SIDE_CLAUDE_CLI or self._db is None:
            return
        try:
            await self._db.set_session_binding(session_id, side)
        except Exception as e:  # noqa: BLE001
            logger.debug("set_session_binding %s: %s", session_id, e)

    # ── classifier + routing ─────────────────────────────────────────

    async def _classify(self, messages: list[dict[str, Any]], session_id: str | None = None) -> str:
        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = str(msg.get("content", ""))[:500]
                break
        if not user_msg:
            elog("router.classify_default", session_id=session_id, tier="medium", reason="empty_user_message")
            return "medium"

        lowered = user_msg.lower()
        if any(hint in lowered for hint in HARD_HINTS):
            elog("router.classify_hint", session_id=session_id, tier="hard", reason="explicit_capability_request")
            return "hard"

        try:
            classifier_session_id = f"{session_id}:classifier" if session_id else "router-classifier"
            classifier_input = (
                f"{CLASSIFIER_PROMPT}\n\n"
                f"Task to classify:\n{user_msg}\n\n"
                "Answer:"
            )
            elog(
                "router.classify_start",
                session_id=session_id,
                classifier_model=self._classifier_model,
                prompt_len=len(user_msg),
            )
            # The classifier always runs against an Agno model — it's a
            # cheap throwaway probe and claude-cli can't serve a
            # sub-session without polluting the parent's history.
            provider = self._get_agno_provider(self._classifier_model)
            resp = await provider.generate(
                messages=[{"role": "user", "content": classifier_input}],
                session_id=classifier_session_id,
            )
            text = resp.content.strip().lower()
            for tier in TIERS:
                if tier in text:
                    elog("router.classify_result", session_id=session_id, tier=tier, raw=text[:80])
                    return tier
        except Exception as e:
            elog("router.classify_error", session_id=session_id, error=str(e))

        elog("router.classify_default", session_id=session_id, tier="medium", reason="unrecognized_classifier_output")
        return "medium"

    def _pick_model(self, tier: str, budget_ratio: float) -> tuple[str, str, str]:
        effective_tier = tier
        reason = "tier"
        if budget_ratio <= 0:
            effective_tier, reason = "fallback", "budget_exhausted"
            return self._routing.get("fallback", self._routing.get("simple", "")), effective_tier, reason
        if budget_ratio < 0.05:
            effective_tier, reason = "fallback", "budget_critical"
            return self._routing.get("fallback", self._routing.get("simple", "")), effective_tier, reason
        if budget_ratio < 0.20:
            effective_tier, reason = "simple", "budget_degraded"
        return self._routing.get(effective_tier, self._routing.get("medium", "")), effective_tier, reason

    def _configured_models_for_side(self, side: str | None) -> list[str]:
        result: list[str] = []
        for entry in iter_configured_models(self._providers_config):
            if side and self._side_for_model(entry.runtime_id) != side:
                continue
            result.append(entry.runtime_id)
        return result

    def _candidate_models(
        self,
        requested_tier: str,
        effective_tier: str,
        primary_model: str,
        bound_side: str | None,
    ) -> list[str]:
        """Build the fallback chain, restricted to ``bound_side`` if set."""
        want_side = bound_side or self._side_for_model(primary_model)
        candidates: list[str] = []

        def add(model_id: str | None) -> None:
            if not model_id or model_id in candidates:
                return
            if self._side_for_model(model_id) != want_side:
                return
            candidates.append(model_id)

        add(primary_model)
        add(self._routing.get("fallback"))
        add(self._routing.get(requested_tier))
        add(self._routing.get("medium"))
        add(self._routing.get("simple"))
        add(self._routing.get("hard"))
        for model_id in self._configured_models_for_side(want_side):
            add(model_id)
        return candidates

    async def _budget_ratio(self, session_id: str | None = None) -> float:
        ratio = 1.0
        if self._budget:
            ratio = await self._budget.get_budget_ratio()
            elog("router.budget", session_id=session_id, budget_ratio=round(ratio, 3))
        return ratio

    def _remember_tier(self, session_id: str | None, tier: str) -> None:
        key = session_id or "__default__"
        self._last_tier_by_session[key] = tier

    def _recall_tier(self, session_id: str | None) -> str:
        key = session_id or "__default__"
        return self._last_tier_by_session.get(key, "medium")

    @staticmethod
    def _is_retryable_response(response: ModelResponse) -> bool:
        stop_reason = (response.stop_reason or "").strip().lower()
        return stop_reason in {"error", "timeout", "rate_limit", "provider_error", "service_unavailable"}

    @staticmethod
    def _is_tool_continuation(messages: list[dict[str, Any]]) -> bool:
        return bool(messages and messages[-1].get("role") == "tool")

    async def _resolve_requested_tier(self, messages: list[dict[str, Any]], session_id: str | None) -> str:
        if self._is_tool_continuation(messages):
            tier = self._recall_tier(session_id)
            elog("router.continuation", session_id=session_id, tier=tier)
            return tier
        tier = await self._classify(messages, session_id=session_id)
        self._remember_tier(session_id, tier)
        return tier

    async def _routing_decision(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None,
        budget_ratio: float,
    ) -> RoutingDecision:
        bound_side = (
            await self._hydrate_bound_side(session_id) if session_id else None
        )

        requested_tier = await self._resolve_requested_tier(messages, session_id)
        primary_model, effective_tier, reason = self._pick_model(requested_tier, budget_ratio)

        # If the session is already bound and the classifier's pick lives
        # on the wrong side, substitute the first configured model on
        # the bound side. Failing that, leave ``primary_model`` as-is
        # and let the candidate filter raise downstream.
        if bound_side and primary_model and self._side_for_model(primary_model) != bound_side:
            for alt in self._configured_models_for_side(bound_side):
                primary_model = alt
                reason = f"bound_to_{bound_side}"
                break
            else:
                primary_model = ""

        candidates = self._candidate_models(requested_tier, effective_tier, primary_model, bound_side)
        return RoutingDecision(
            requested_tier=requested_tier,
            effective_tier=effective_tier,
            reason=reason,
            primary_model=primary_model,
            candidates=candidates,
            bound_side=bound_side,
        )

    # ── provider dispatch ────────────────────────────────────────────

    async def _dispatch(
        self,
        runtime_id: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]] | None,
        on_status: Callable[[str], Awaitable[None]] | None,
        session_id: str | None,
    ) -> ModelResponse:
        if is_claude_cli_model(runtime_id):
            registry = self._get_claude_registry()
            return await registry.generate(
                messages,
                system=system,
                tools=tools,
                on_status=on_status,
                session_id=session_id,
                model_override=runtime_id,
            )
        provider = self._get_agno_provider(runtime_id)
        return await provider.generate(
            messages,
            system=system,
            tools=tools,
            on_status=on_status,
            session_id=session_id,
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
    ) -> ModelResponse:
        budget_ratio = await self._budget_ratio(session_id)

        if budget_ratio <= 0 and self._monthly_budget > 0:
            elog("router.budget_exceeded", session_id=session_id, monthly_budget=self._monthly_budget)
            return ModelResponse(
                content="Monthly budget exhausted. Please increase the budget or wait for the next billing period.",
                stop_reason="budget_exceeded",
            )

        decision = await self._routing_decision(messages, session_id, budget_ratio)
        if not decision.primary_model or not decision.candidates:
            # Bound-side has no enabled model, or routing is empty.
            msg = (
                f"No {decision.bound_side} model available for this session."
                if decision.bound_side
                else "No model configured for this task tier."
            )
            elog(
                "router.error",
                session_id=session_id,
                tier=decision.requested_tier,
                routing=self._routing,
                bound_side=decision.bound_side,
            )
            return ModelResponse(content=msg, stop_reason="error")

        elog(
            "router.route",
            session_id=session_id,
            requested_tier=decision.requested_tier,
            effective_tier=decision.effective_tier,
            reason=decision.reason,
            model=decision.primary_model,
            bound_side=decision.bound_side,
            budget_ratio=round(budget_ratio, 3),
        )
        elog("router.candidates", session_id=session_id, models=decision.candidates)

        resp: ModelResponse | None = None
        active_model_id = decision.primary_model
        last_error: Exception | None = None
        for attempt, candidate_model in enumerate(decision.candidates, start=1):
            if attempt > 1:
                elog(
                    "router.retry",
                    session_id=session_id,
                    attempt=attempt,
                    next_model=candidate_model,
                    previous_error=str(last_error) if last_error else None,
                )
            try:
                resp = await self._dispatch(
                    candidate_model, messages, system, tools, on_status, session_id,
                )
                if self._is_retryable_response(resp):
                    raise RuntimeError(resp.content or resp.stop_reason or "provider returned an error response")
                active_model_id = candidate_model
                break
            except Exception as e:
                last_error = e
                elog(
                    "router.fallback",
                    session_id=session_id,
                    failed_model=candidate_model,
                    error=str(e),
                    next_model=decision.candidates[attempt] if attempt < len(decision.candidates) else None,
                    attempt=attempt,
                )

        if resp is None:
            assert last_error is not None
            raise last_error

        # Persist the binding once we've served at least one turn. The
        # claude-cli path auto-writes to ``sdk_sessions`` via the
        # registry, so we only need to write the agno row here.
        if session_id:
            await self._persist_bound_side(session_id, self._side_for_model(active_model_id))

        if self._budget:
            cost = BudgetTracker.compute_cost(
                active_model_id,
                resp.input_tokens,
                resp.output_tokens,
                providers_config=self._providers_config,
            )
            try:
                await self._budget.record(
                    model=active_model_id,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    cost=cost,
                    session_id=session_id,
                )
                elog(
                    "router.cost_recorded",
                    session_id=session_id,
                    model=active_model_id,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    cost_usd=cost,
                )
            except Exception as e:
                elog("router.cost_record_error", session_id=session_id, model=active_model_id, error=str(e))
        else:
            elog("router.cost_skipped", session_id=session_id, model=active_model_id, reason="no_budget_tracker")

        resp.model = active_model_id
        return resp

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Streaming path — agno-only (claude-cli isn't streamable here).

        Used by the REST smoke-test endpoint; the interactive turn
        surface always goes through ``generate`` which handles both
        sides.
        """
        model_id = self._routing.get("medium") or self._routing.get("simple") or ""
        if is_claude_cli_model(model_id):
            # Fall back to whichever agno model the router knows about.
            for candidate in self._routing.values():
                if not is_claude_cli_model(candidate):
                    model_id = candidate
                    break
        provider = self._get_agno_provider(model_id)
        async for chunk in provider.stream(messages, system=system, tools=tools):
            yield chunk
