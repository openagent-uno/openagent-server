"""Budget tracking for LLM API usage."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Tracks LLM API spend against a monthly budget.

    Wraps MemoryDB usage methods and adds budget-aware logic.
    """

    def __init__(self, db, monthly_budget: float = 0.0):
        self._db = db
        self.monthly_budget = monthly_budget

    async def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        session_id: str | None = None,
    ) -> None:
        try:
            await self._db.record_usage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                session_id=session_id,
            )
        except Exception as e:
            logger.warning("Failed to record usage: %s", e)

    async def get_current_month_spend(self) -> float:
        return await self._db.get_monthly_usage()

    async def get_remaining(self) -> float:
        if self.monthly_budget <= 0:
            return float("inf")
        spent = await self.get_current_month_spend()
        return max(0.0, self.monthly_budget - spent)

    async def get_budget_ratio(self) -> float:
        """Fraction of budget remaining (1.0 = full, 0.0 = exhausted)."""
        if self.monthly_budget <= 0:
            return 1.0
        spent = await self.get_current_month_spend()
        return max(0.0, 1.0 - spent / self.monthly_budget)

    async def get_usage_summary(self) -> dict[str, Any]:
        summary = await self._db.get_usage_summary()
        return {
            "monthly_spend": summary["total"],
            "monthly_budget": self.monthly_budget,
            "remaining": max(0.0, self.monthly_budget - summary["total"]) if self.monthly_budget > 0 else None,
            "by_model": summary["by_model"],
        }

    @staticmethod
    def compute_cost(model: str, input_tokens: int, output_tokens: int, providers_config: dict | None = None) -> float:
        """Compute cost using OpenAgent's configured product catalog."""
        from openagent.models.catalog import compute_cost

        return compute_cost(
            model_ref=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            providers_config=providers_config,
        )
