"""Smart model router — classifies tasks and picks the optimal model.

Routes requests to cheap models for simple tasks and expensive models
for hard ones, tracking spend against a monthly budget.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from openagent.models.base import BaseModel, ModelResponse
from openagent.models.budget import BudgetTracker
from openagent.models.litellm_provider import LiteLLMProvider

logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """\
Classify this task as simple, medium, or hard.
- simple: greetings, short factual questions, text formatting, status checks, translations
- medium: code review, document summarization, multi-step reasoning, data analysis
- hard: complex architecture design, multi-file code generation, debugging across systems, research synthesis
Reply with ONLY one word: simple, medium, or hard."""

TIERS = ("simple", "medium", "hard")


class SmartRouter(BaseModel):
    """Routes requests to the best model based on task difficulty and budget.

    Supports two modes:
    1. Manual routing: explicit routing dict mapping tiers to models
    2. Auto-pricing: builds routing from all models in providers_config,
       sorted by price (cheapest for simple, mid for medium, most expensive for hard)
    """

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
        self._providers: dict[str, BaseModel] = {}
        self._mcp_servers: dict[str, dict] = {}
        self._claude_permission_mode = claude_permission_mode
        self._last_tier_by_session: dict[str, str] = {}

        # Build routing: use explicit dict if provided, else auto-price from providers
        if routing:
            self._routing = routing
        else:
            self._routing = self._build_auto_routing()

        self._classifier_model = classifier_model or self._routing.get("simple", "anthropic/claude-haiku-4-5")
        from openagent.core.logging import elog

        elog(
            "router.config",
            routing=self._routing,
            classifier_model=self._classifier_model,
            monthly_budget=self._monthly_budget,
        )

    def _normalize_model_id(self, provider_name: str, model_id: str, model_cost: dict[str, Any]) -> str:
        """Normalize a configured provider/model entry into a LiteLLM model id.

        Accepts either:
        - bare ids like ``gpt-4o-mini`` or ``glm-5``
        - fully-qualified ids already in LiteLLM form like ``google.gemma-3-4b-it``
          or ``zai/glm-5``
        """
        raw = str(model_id or "").strip()
        if not raw:
            return raw
        if raw == "claude-cli" or raw.startswith("claude-cli/"):
            return raw

        candidates: list[str] = [raw]
        if not raw.startswith(f"{provider_name}/"):
            candidates.append(f"{provider_name}/{raw}")
        if not raw.startswith(f"{provider_name}."):
            candidates.append(f"{provider_name}.{raw}")

        for candidate in candidates:
            if candidate in model_cost:
                return candidate

        # Fall back to the raw value when it already looks fully qualified.
        if "/" in raw or "." in raw:
            return raw

        # Default to slash form for provider-prefixed bare ids.
        return f"{provider_name}/{raw}"

    def _build_auto_routing(self) -> dict[str, str]:
        """Build routing dict automatically from providers config, sorted by price."""
        from openagent.core.logging import elog

        models_with_price = []
        try:
            from litellm import model_cost
        except ImportError:
            model_cost = {}

        for provider_name, cfg in self._providers_config.items():
            disabled = set(cfg.get("disabled_models", []))
            for model_id in cfg.get("models", []):
                if model_id in disabled:
                    continue
                full_id = self._normalize_model_id(provider_name, model_id, model_cost)
                info = model_cost.get(full_id, {})
                # Use output cost as the price signal (higher = more capable)
                price = (info.get("output_cost_per_token", 0) or 0)
                models_with_price.append((full_id, price))

        if not models_with_price:
            logger.warning("SmartRouter: no models found in providers config, using defaults")
            routing = {
                "simple": "anthropic/claude-haiku-4-5",
                "medium": "anthropic/claude-sonnet-4-6",
                "hard": "anthropic/claude-opus-4-6",
                "fallback": "anthropic/claude-haiku-4-5",
            }
            elog("router.auto_routing_default", routing=routing)
            return routing

        models_with_price.sort(key=lambda x: x[1])
        n = len(models_with_price)

        routing = {
            "simple": models_with_price[0][0],
            "medium": models_with_price[n // 2][0],
            "hard": models_with_price[-1][0],
            "fallback": models_with_price[0][0],
        }
        logger.info("SmartRouter auto-routing: %s", routing)
        elog("router.auto_routing", routing=routing, candidates=n)
        return routing

    def set_db(self, db) -> None:
        """Wire up budget tracking after DB is available."""
        self._budget = BudgetTracker(db, self._monthly_budget)

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        """Pass MCP server config through to routed Claude CLI instances."""
        self._mcp_servers = servers
        for model in self._providers.values():
            setter = getattr(model, "set_mcp_servers", None)
            if callable(setter):
                setter(servers)

    async def cleanup_idle(self) -> None:
        """Run idle cleanup on any routed provider that supports it."""
        for model in self._providers.values():
            cleanup_idle = getattr(model, "cleanup_idle", None)
            if callable(cleanup_idle):
                await cleanup_idle()

    async def shutdown(self) -> None:
        """Shut down any cached routed providers that need cleanup."""
        for model in self._providers.values():
            shutdown = getattr(model, "shutdown", None)
            if callable(shutdown):
                await shutdown()

    def _get_provider(self, model: str) -> BaseModel:
        if model not in self._providers:
            if model == "claude-cli" or model.startswith("claude-cli/"):
                from openagent.models.claude_cli import ClaudeCLI

                claude_model = model.split("/", 1)[1] if "/" in model else None
                permission_mode = (
                    self._providers_config.get("anthropic", {}).get("permission_mode")
                    or self._claude_permission_mode
                )
                routed_model: BaseModel = ClaudeCLI(
                    model=claude_model,
                    permission_mode=permission_mode,
                    mcp_servers=self._mcp_servers or None,
                )
            else:
                routed_model = LiteLLMProvider(
                    model=model,
                    api_key=self._api_key,
                    providers_config=self._providers_config,
                )
            self._providers[model] = routed_model
        return self._providers[model]

    async def _classify(self, messages: list[dict[str, Any]], session_id: str | None = None) -> str:
        """Classify the latest user message as simple/medium/hard."""
        from openagent.core.logging import elog

        # Extract latest user message
        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = str(msg.get("content", ""))[:500]
                break

        if not user_msg:
            elog("router.classify_default", session_id=session_id, tier="medium", reason="empty_user_message")
            return "medium"

        try:
            elog(
                "router.classify_start",
                session_id=session_id,
                classifier_model=self._classifier_model,
                prompt_len=len(user_msg),
            )
            provider = self._get_provider(self._classifier_model)
            classifier_session_id = f"{session_id}:classifier" if session_id else "router-classifier"
            resp = await provider.generate(
                messages=[{"role": "user", "content": user_msg}],
                system=CLASSIFIER_PROMPT,
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
        """Select model based on tier and remaining budget."""
        effective_tier = tier
        reason = "tier"

        # Budget degradation
        if budget_ratio <= 0:
            effective_tier = "fallback"
            reason = "budget_exhausted"
            return (
                self._routing.get("fallback", self._routing.get("simple", "")),
                effective_tier,
                reason,
            )
        if budget_ratio < 0.05:
            effective_tier = "fallback"
            reason = "budget_critical"
            return (
                self._routing.get("fallback", self._routing.get("simple", "")),
                effective_tier,
                reason,
            )
        if budget_ratio < 0.20:
            effective_tier = "simple"
            reason = "budget_degraded"

        return self._routing.get(effective_tier, self._routing.get("medium", "")), effective_tier, reason

    def _configured_models(self) -> list[str]:
        """Return all enabled configured API-callable models, normalized for LiteLLM."""
        try:
            from litellm import model_cost
        except ImportError:
            model_cost = {}

        models: list[str] = []
        for provider_name, cfg in self._providers_config.items():
            disabled = set(cfg.get("disabled_models", []))
            for model_id in cfg.get("models", []):
                if model_id in disabled:
                    continue
                normalized = self._normalize_model_id(provider_name, model_id, model_cost)
                if normalized and normalized not in models:
                    models.append(normalized)
        return models

    def _candidate_models(self, requested_tier: str, effective_tier: str, primary_model: str) -> list[str]:
        """Build a failover chain for the current request."""
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

        for model_id in self._configured_models():
            add(model_id)

        return candidates

    def _remember_tier(self, session_id: str | None, tier: str) -> None:
        """Persist the chosen tier for later tool-continuation turns."""
        if session_id:
            self._last_tier_by_session[session_id] = tier
        else:
            self._last_tier_by_session["__default__"] = tier

    def _recall_tier(self, session_id: str | None) -> str:
        """Look up the last tier used by this session."""
        if session_id:
            return self._last_tier_by_session.get(session_id, "medium")
        return self._last_tier_by_session.get("__default__", "medium")

    def _is_retryable_response(self, response: ModelResponse) -> bool:
        """Treat provider-declared error stop reasons as retryable router failures."""
        stop_reason = (response.stop_reason or "").strip().lower()
        return stop_reason in {
            "error",
            "timeout",
            "rate_limit",
            "provider_error",
            "service_unavailable",
        }

    def _is_tool_continuation(self, messages: list[dict[str, Any]]) -> bool:
        """Check if this is a tool-result continuation (skip classification)."""
        if messages and messages[-1].get("role") == "tool":
            return True
        return False

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Optional[Callable[[str], Awaitable[None]]] = None,
        session_id: str | None = None,
    ) -> ModelResponse:
        from openagent.core.logging import elog

        # Determine budget ratio
        budget_ratio = 1.0
        if self._budget:
            budget_ratio = await self._budget.get_budget_ratio()
            logger.debug("SmartRouter: budget_ratio=%.2f (%.1f%% remaining)", budget_ratio, budget_ratio * 100)
            elog("router.budget", session_id=session_id, budget_ratio=round(budget_ratio, 3))

        if budget_ratio <= 0 and self._monthly_budget > 0:
            logger.warning("SmartRouter: budget exhausted")
            elog("router.budget_exceeded", session_id=session_id, monthly_budget=self._monthly_budget)
            return ModelResponse(
                content="Monthly budget exhausted. Please increase the budget or wait for the next billing period.",
                stop_reason="budget_exceeded",
            )

        # Classify task (skip for tool continuations)
        if self._is_tool_continuation(messages):
            tier = self._recall_tier(session_id)
            logger.debug("SmartRouter: tool continuation, reusing tier=%s", tier)
            elog("router.continuation", session_id=session_id, tier=tier)
        else:
            tier = await self._classify(messages, session_id=session_id)
            self._remember_tier(session_id, tier)
            logger.debug("SmartRouter: classified as tier=%s", tier)

        # Pick model
        model_id, effective_tier, route_reason = self._pick_model(tier, budget_ratio)
        if not model_id:
            logger.error("SmartRouter: no model for tier=%s, routing=%s", tier, self._routing)
            elog("router.error", session_id=session_id, tier=tier, routing=self._routing)
            return ModelResponse(content="No model configured for this task tier.", stop_reason="error")

        logger.info("SmartRouter: tier=%s budget=%.0f%% → %s", tier, budget_ratio * 100, model_id)
        elog(
            "router.route",
            session_id=session_id,
            requested_tier=tier,
            effective_tier=effective_tier,
            reason=route_reason,
            model=model_id,
            budget_ratio=round(budget_ratio, 3),
        )
        candidates = self._candidate_models(tier, effective_tier, model_id)
        elog("router.candidates", session_id=session_id, models=candidates)

        # Generate with failover across the routed/configured model set.
        resp = None
        active_model_id = model_id
        last_error: Exception | None = None
        for attempt, candidate_model in enumerate(candidates, start=1):
            provider = self._get_provider(candidate_model)
            if attempt > 1:
                logger.warning(
                    "SmartRouter: retrying with %s after %s",
                    candidate_model,
                    last_error,
                )
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
                    next_model=candidates[attempt] if attempt < len(candidates) else None,
                    attempt=attempt,
                )
                continue

        if resp is None:
            assert last_error is not None
            raise last_error

        # Record usage
        if self._budget:
            cost = BudgetTracker.compute_cost(active_model_id, resp.input_tokens, resp.output_tokens)
            logger.info("SmartRouter: usage model=%s in=%d out=%d cost=$%.6f", active_model_id, resp.input_tokens, resp.output_tokens, cost)
            elog(
                "router.usage",
                session_id=session_id,
                model=active_model_id,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost=cost,
            )
            await self._budget.record(
                model=active_model_id,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost=cost,
                session_id=session_id,
            )

        return resp

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        # Use medium tier for streaming (no classification overhead)
        model_id = self._routing.get("medium", self._routing.get("simple", ""))
        provider = self._get_provider(model_id)
        async for chunk in provider.stream(messages, system=system, tools=tools):
            yield chunk
