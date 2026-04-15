"""Smart model router with OpenAgent-owned pricing and Agno-backed execution."""

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

# Fallback routing used when no platform-managed models are configured. Kept as
# a constant so the next "GPT-X is out, swap defaults" change is one edit.
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


@dataclass(frozen=True)
class RoutingDecision:
    requested_tier: str
    effective_tier: str
    reason: str
    primary_model: str
    candidates: list[str]


class SmartRouter(BaseModel):
    """Cost-aware router for platform-managed API-backed model sessions."""

    history_mode = "platform"

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
        self._budget: BudgetTracker | None = None
        self._db = None
        self._providers: dict[str, BaseModel] = {}
        # Single shared MCPPool — wired to every tier provider as it's lazily
        # created so all tiers reuse the same MCP server processes.
        self._mcp_pool: Any = None
        self._claude_permission_mode = claude_permission_mode
        self._last_tier_by_session: dict[str, str] = {}

        if routing:
            normalised: dict[str, str] = {}
            stripped: list[tuple[str, str]] = []
            for tier, model_id in routing.items():
                runtime_id = normalize_runtime_model_id(model_id, self._providers_config)
                if is_claude_cli_model(runtime_id):
                    # claude-cli is a standalone provider; sessions stay on it for
                    # their lifetime. SmartRouter is contractually platform-mode
                    # (``history_mode = "platform"``), so we strip claude-cli from
                    # any routing tier rather than silently violating the contract.
                    stripped.append((tier, runtime_id))
                    continue
                normalised[tier] = runtime_id
            if stripped:
                elog("router.routing_stripped", stripped=stripped)
                logger.warning(
                    "SmartRouter: stripped claude-cli model(s) from routing %s — "
                    "claude-cli is a standalone provider and cannot be mixed with "
                    "Agno-routed models in the same session.",
                    stripped,
                )
            self._routing = normalised or self._build_auto_routing()
        else:
            self._routing = self._build_auto_routing()

        self._classifier_model = normalize_runtime_model_id(
            classifier_model or self._routing.get("simple", DEFAULT_CLASSIFIER_MODEL),
            self._providers_config,
        )
        elog(
            "router.config",
            routing=self._routing,
            classifier_model=self._classifier_model,
            monthly_budget=self._monthly_budget,
        )

    def _build_auto_routing(self) -> dict[str, str]:
        """Build routing from configured platform-managed models and their costs."""
        models_with_price: list[tuple[str, float]] = []
        for entry in iter_configured_models(self._providers_config, history_mode="platform"):
            price = float(entry.output_cost_per_million or 0.0)
            models_with_price.append((entry.runtime_id, price))

        if not models_with_price:
            logger.warning("SmartRouter: no API-backed models found in providers config, using defaults")
            routing = dict(DEFAULT_AUTO_ROUTING)
            elog("router.auto_routing_default", routing=routing)
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

    def set_db(self, db) -> None:
        self._db = db
        self._budget = BudgetTracker(db, self._monthly_budget)
        for model in self._providers.values():
            wire_model_runtime(model, db=db)

    def set_mcp_pool(self, pool: Any) -> None:
        """Receive the process-wide MCPPool. Re-wires already-created tier providers."""
        self._mcp_pool = pool
        for model in self._providers.values():
            wire_model_runtime(model, mcp_pool=pool)

    async def cleanup_idle(self) -> None:
        for model in self._providers.values():
            cleanup_idle = getattr(model, "cleanup_idle", None)
            if callable(cleanup_idle):
                await cleanup_idle()

    async def shutdown(self) -> None:
        for model in self._providers.values():
            shutdown = getattr(model, "shutdown", None)
            if callable(shutdown):
                await shutdown()

    async def close_session(self, session_id: str) -> None:
        if not session_id:
            return
        self._last_tier_by_session.pop(session_id, None)
        for model in self._providers.values():
            close_session = getattr(model, "close_session", None)
            if callable(close_session):
                await close_session(session_id)

    def _get_provider(self, model: str) -> BaseModel:
        if is_claude_cli_model(model):
            # Defence-in-depth: claude-cli is a standalone provider with its own
            # history_mode ("provider") and its own usage_log writer. The routing
            # input is already filtered (see __init__ stripping), so reaching
            # here implies a programming bug — fail loudly rather than violate
            # the SmartRouter ↔ ClaudeCLI isolation invariant.
            raise RuntimeError(
                f"SmartRouter cannot dispatch to claude-cli model '{model}'. "
                "Switch the agent's model.provider to 'claude-cli' instead."
            )
        if model not in self._providers:
            self._providers[model] = create_model_from_spec(
                model,
                providers_config=self._providers_config,
                api_key=self._api_key,
                claude_permission_mode=self._claude_permission_mode,
                db=self._db,
                mcp_pool=self._mcp_pool,
            )
        return self._providers[model]

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
            elog(
                "router.classify_start",
                session_id=session_id,
                classifier_model=self._classifier_model,
                prompt_len=len(user_msg),
            )
            provider = self._get_provider(self._classifier_model)
            classifier_session_id = f"{session_id}:classifier" if session_id else "router-classifier"
            classifier_input = (
                f"{CLASSIFIER_PROMPT}\n\n"
                f"Task to classify:\n{user_msg}\n\n"
                "Answer:"
            )
            resp = await provider.generate(
                messages=[{"role": "user", "content": classifier_input}],
                session_id=classifier_session_id,
            )
            text = resp.content.strip().lower()
            for tier in TIERS:
                if tier in text:
                    elog(
                        "router.classify_result",
                        session_id=session_id,
                        classifier_model=self._classifier_model,
                        tier=tier,
                        raw=text[:80],
                    )
                    return tier
        except Exception as e:
            logger.debug("Classification failed, defaulting to medium: %s", e)
            elog(
                "router.classify_error",
                session_id=session_id,
                classifier_model=self._classifier_model,
                error=str(e),
            )

        elog("router.classify_default", session_id=session_id, tier="medium", reason="unrecognized_classifier_output")
        return "medium"

    def _pick_model(self, tier: str, budget_ratio: float) -> tuple[str, str, str]:
        effective_tier = tier
        reason = "tier"

        if budget_ratio <= 0:
            effective_tier = "fallback"
            reason = "budget_exhausted"
            return self._routing.get("fallback", self._routing.get("simple", "")), effective_tier, reason
        if budget_ratio < 0.05:
            effective_tier = "fallback"
            reason = "budget_critical"
            return self._routing.get("fallback", self._routing.get("simple", "")), effective_tier, reason
        if budget_ratio < 0.20:
            effective_tier = "simple"
            reason = "budget_degraded"

        return self._routing.get(effective_tier, self._routing.get("medium", "")), effective_tier, reason

    def _configured_models(self, history_mode: str | None = None) -> list[str]:
        return [
            entry.runtime_id
            for entry in iter_configured_models(self._providers_config, history_mode=history_mode)
        ]

    def _candidate_models(self, requested_tier: str, effective_tier: str, primary_model: str) -> list[str]:
        candidates: list[str] = []

        def add(model_id: str | None) -> None:
            if model_id and model_id not in candidates:
                candidates.append(model_id)

        add(primary_model)
        add(self._routing.get("fallback"))
        add(self._routing.get(requested_tier))
        add(self._routing.get("medium"))
        add(self._routing.get("simple"))
        add(self._routing.get("hard"))

        primary_mode = model_history_mode(primary_model, self._providers_config)
        for model_id in self._configured_models(history_mode=primary_mode):
            add(model_id)

        return [model_id for model_id in candidates if model_history_mode(model_id, self._providers_config) == primary_mode]

    async def _budget_ratio(self, session_id: str | None = None) -> float:
        ratio = 1.0
        if self._budget:
            ratio = await self._budget.get_budget_ratio()
            elog("router.budget", session_id=session_id, budget_ratio=round(ratio, 3))
        return ratio

    def _remember_tier(self, session_id: str | None, tier: str) -> None:
        if session_id:
            self._last_tier_by_session[session_id] = tier
        else:
            self._last_tier_by_session["__default__"] = tier

    def _recall_tier(self, session_id: str | None) -> str:
        if session_id:
            return self._last_tier_by_session.get(session_id, "medium")
        return self._last_tier_by_session.get("__default__", "medium")

    def _is_retryable_response(self, response: ModelResponse) -> bool:
        stop_reason = (response.stop_reason or "").strip().lower()
        return stop_reason in {"error", "timeout", "rate_limit", "provider_error", "service_unavailable"}

    def _is_tool_continuation(self, messages: list[dict[str, Any]]) -> bool:
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
        requested_tier = await self._resolve_requested_tier(messages, session_id)
        primary_model, effective_tier, reason = self._pick_model(requested_tier, budget_ratio)
        candidates = self._candidate_models(requested_tier, effective_tier, primary_model)
        return RoutingDecision(
            requested_tier=requested_tier,
            effective_tier=effective_tier,
            reason=reason,
            primary_model=primary_model,
            candidates=candidates,
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
        if not decision.primary_model:
            elog("router.error", session_id=session_id, tier=decision.requested_tier, routing=self._routing)
            return ModelResponse(content="No model configured for this task tier.", stop_reason="error")

        elog(
            "router.route",
            session_id=session_id,
            requested_tier=decision.requested_tier,
            effective_tier=decision.effective_tier,
            reason=decision.reason,
            model=decision.primary_model,
            budget_ratio=round(budget_ratio, 3),
        )
        elog("router.candidates", session_id=session_id, models=decision.candidates)

        resp = None
        active_model_id = decision.primary_model
        last_error: Exception | None = None
        for attempt, candidate_model in enumerate(decision.candidates, start=1):
            provider = self._get_provider(candidate_model)
            if attempt > 1:
                elog(
                    "router.retry",
                    session_id=session_id,
                    attempt=attempt,
                    next_model=candidate_model,
                    previous_error=str(last_error) if last_error else None,
                )
            try:
                resp = await provider.generate(
                    messages,
                    system=system,
                    tools=tools,
                    on_status=on_status,
                    session_id=session_id,
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
                elog(
                    "router.cost_record_error",
                    session_id=session_id,
                    model=active_model_id,
                    error=str(e),
                )
        else:
            elog(
                "router.cost_skipped",
                session_id=session_id,
                model=active_model_id,
                reason="no_budget_tracker",
            )

        resp.model = active_model_id
        return resp

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        model_id = self._routing.get("medium", self._routing.get("simple", ""))
        provider = self._get_provider(model_id)
        async for chunk in provider.stream(messages, system=system, tools=tools):
            yield chunk
