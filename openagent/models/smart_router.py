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
    ):
        self._providers_config = providers_config or {}
        self._api_key = api_key
        self._monthly_budget = monthly_budget
        self._budget: BudgetTracker | None = None
        self._providers: dict[str, LiteLLMProvider] = {}

        # Build routing: use explicit dict if provided, else auto-price from providers
        if routing:
            self._routing = routing
        else:
            self._routing = self._build_auto_routing()

        self._classifier_model = classifier_model or self._routing.get("simple", "anthropic/claude-haiku-4-5")

    def _build_auto_routing(self) -> dict[str, str]:
        """Build routing dict automatically from providers config, sorted by price."""
        models_with_price = []
        try:
            from litellm import model_cost
        except ImportError:
            model_cost = {}

        for provider_name, cfg in self._providers_config.items():
            disabled = set(cfg.get("disabled_models", []))
            for model_id in cfg.get("models", []):
                if model_id == "claude-cli":
                    continue  # skip subscription-based; not API-callable via litellm
                if model_id in disabled:
                    continue
                full_id = f"{provider_name}/{model_id}"
                info = model_cost.get(full_id, {})
                # Use output cost as the price signal (higher = more capable)
                price = (info.get("output_cost_per_token", 0) or 0)
                models_with_price.append((full_id, price))

        if not models_with_price:
            logger.warning("SmartRouter: no models found in providers config, using defaults")
            return {
                "simple": "anthropic/claude-haiku-4-5",
                "medium": "anthropic/claude-sonnet-4-6",
                "hard": "anthropic/claude-opus-4-6",
                "fallback": "anthropic/claude-haiku-4-5",
            }

        models_with_price.sort(key=lambda x: x[1])
        n = len(models_with_price)

        routing = {
            "simple": models_with_price[0][0],
            "medium": models_with_price[n // 2][0],
            "hard": models_with_price[-1][0],
            "fallback": models_with_price[0][0],
        }
        logger.info("SmartRouter auto-routing: %s", routing)
        return routing

    def set_db(self, db) -> None:
        """Wire up budget tracking after DB is available."""
        self._budget = BudgetTracker(db, self._monthly_budget)

    def _get_provider(self, model: str) -> LiteLLMProvider:
        if model not in self._providers:
            self._providers[model] = LiteLLMProvider(
                model=model,
                api_key=self._api_key,
                providers_config=self._providers_config,
            )
        return self._providers[model]

    async def _classify(self, messages: list[dict[str, Any]]) -> str:
        """Classify the latest user message as simple/medium/hard."""
        # Extract latest user message
        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = str(msg.get("content", ""))[:500]
                break

        if not user_msg:
            return "medium"

        try:
            provider = self._get_provider(self._classifier_model)
            resp = await provider.generate(
                messages=[{"role": "user", "content": user_msg}],
                system=CLASSIFIER_PROMPT,
            )
            text = resp.content.strip().lower()
            for tier in TIERS:
                if tier in text:
                    return tier
        except Exception as e:
            logger.debug("Classification failed, defaulting to medium: %s", e)

        return "medium"

    def _pick_model(self, tier: str, budget_ratio: float) -> str:
        """Select model based on tier and remaining budget."""
        # Budget degradation
        if budget_ratio <= 0:
            return self._routing.get("fallback", self._routing.get("simple", ""))
        if budget_ratio < 0.05:
            return self._routing.get("fallback", self._routing.get("simple", ""))
        if budget_ratio < 0.20:
            tier = "simple"

        return self._routing.get(tier, self._routing.get("medium", ""))

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
        # Determine budget ratio
        budget_ratio = 1.0
        if self._budget:
            budget_ratio = await self._budget.get_budget_ratio()

        if budget_ratio <= 0 and self._monthly_budget > 0:
            return ModelResponse(
                content="Monthly budget exhausted. Please increase the budget or wait for the next billing period.",
                stop_reason="budget_exceeded",
            )

        # Classify task (skip for tool continuations)
        if self._is_tool_continuation(messages):
            tier = getattr(self, "_last_tier", "medium")
        else:
            tier = await self._classify(messages)
            self._last_tier = tier

        # Pick model
        model_id = self._pick_model(tier, budget_ratio)
        if not model_id:
            return ModelResponse(content="No model configured for this task tier.", stop_reason="error")

        logger.info("SmartRouter: tier=%s budget=%.0f%% → %s", tier, budget_ratio * 100, model_id)

        # Generate
        provider = self._get_provider(model_id)
        try:
            resp = await provider.generate(messages, system=system, tools=tools, on_status=on_status, session_id=session_id)
        except Exception as e:
            # Fallback on error
            fallback = self._routing.get("fallback")
            if fallback and fallback != model_id:
                logger.warning("SmartRouter: %s failed (%s), falling back to %s", model_id, e, fallback)
                provider = self._get_provider(fallback)
                resp = await provider.generate(messages, system=system, tools=tools, on_status=on_status, session_id=session_id)
            else:
                raise

        # Record usage
        if self._budget:
            cost = BudgetTracker.compute_cost(model_id, resp.input_tokens, resp.output_tokens)
            await self._budget.record(
                model=model_id,
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
