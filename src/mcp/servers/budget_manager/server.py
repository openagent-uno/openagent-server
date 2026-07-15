"""Budget-manager MCP server.

Exposes OpenAgent's spend caps over MCP so the agent can inspect and adjust its
own budgets at runtime — the same way the scheduler / events-manager MCPs manage
their objects (vision §15: the agent knows and reaches for its own levers). The
whole point of this surface is the agent being able to ask itself "am I about to
blow my DeepSeek budget?" and act — throttle its own delegation, warn the user,
or raise the cap deliberately.

Transport: stdio (launched as a subprocess by MCPPool).
Storage: the same SQLite DB as the main runtime; path from OPENAGENT_DB_PATH
(injected by the Agent — the ``budget-manager`` name MUST be in
``resolve_default_entry``'s inject list, else this points at the wrong file).

Writes go straight to the ``budgets`` table via ``MemoryDB``; the main process's
``BudgetGuard`` reloads all rules on its next refresh (cost-triggered or the
short TTL backstop), so a cap the agent sets here takes effect within seconds
without an IPC hop. The usage view is computed fresh from ``usage_log`` here, so
it never depends on the main process's cached snapshot.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.core.budget_guard import compute_budget_usage, normalize_rule_input
from src.memory.db import MemoryDB

logger = logging.getLogger(__name__)


def _db_path() -> str:
    return os.environ.get("OPENAGENT_DB_PATH") or "openagent.db"


_db: MemoryDB | None = None


async def _get_db() -> MemoryDB:
    global _db
    if _db is None:
        db = MemoryDB(_db_path())
        await db.connect()
        _db = db
        logger.info("budget-manager MCP connected to %s", _db_path())
    return _db


mcp = FastMCP("budget-manager")


@mcp.tool()
async def list_budgets(enabled_only: bool = False) -> list[dict[str, Any]]:
    """List every budget rule. A rule caps spend for a scope over a window and,
    when tripped, makes the router route AROUND that scope (it drops out of the
    enabled catalog) for the rest of the window — the agent keeps working on
    whatever stays enabled, it just stops spending on the capped one."""
    db = await _get_db()
    return await db.list_budgets(enabled_only=enabled_only)


@mcp.tool()
async def get_budget_usage(enabled_only: bool = False) -> list[dict[str, Any]]:
    """Current spend vs limit for every rule — the meter to read before doing
    expensive work. Each entry carries ``spend``, ``limit`` (``amount``),
    ``ratio``, ``over``, ``remaining`` and the window bounds. ``enforced=false``
    marks ``task`` scopes and ``per_run`` windows, which are reported but not yet
    enforced by the router (phase 2)."""
    db = await _get_db()
    return await compute_budget_usage(db, enabled_only=enabled_only)


@mcp.tool()
async def set_budget(
    scope: str,
    amount: float,
    scope_kind: str | None = None,
    metric: str = "cost_usd",
    window: str = "day",
    alert_thresholds: list[float] | None = None,
    webhook_url: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Create or update a budget rule (upsert on scope+metric+window).

    Args:
        scope: what to meter — a provider name ('deepseek'), a model runtime_id
            ('deepseek:deepseek-v4-pro'), or '' / 'global' for ALL spend.
        amount: the cap. Interpreted in dollars when metric='cost_usd', or in
            tokens (input+output) when metric='tokens'.
        scope_kind: 'global' | 'provider' | 'model' | 'task'. Inferred from
            ``scope`` when omitted (a colon → model, empty → global, else
            provider). 'task' is stored/reported but not yet enforced.
        metric: 'cost_usd' (dollars) or 'tokens'. Use 'tokens' when dollar
            pricing is uncertain (token caps are model-agnostic and robust).
        window: 'hour' | 'day' | 'month' (calendar windows in the agent's
            timezone), or 'per_run' (stored, phase-2 only).
        alert_thresholds: fractions in (0,1) at which to alert, default
            [0.5, 0.9]; the 100% cap alert always fires.
        webhook_url: optional URL POSTed on each threshold crossing.
        enabled: whether the rule is active.

    Returns the stored rule. Takes effect on the main process within seconds.
    """
    raw = {
        "scope": scope,
        "scope_kind": scope_kind,
        "amount": amount,
        "metric": metric,
        "window": window,
        "alert_thresholds": alert_thresholds,
        "webhook_url": webhook_url,
        "enabled": enabled,
    }
    try:
        norm = normalize_rule_input(raw)
    except ValueError as e:
        raise ValueError(str(e))
    if norm is None:
        raise ValueError("invalid budget rule")
    if norm["amount"] <= 0:
        raise ValueError("amount must be > 0")

    db = await _get_db()
    # Upsert on the unique identity so "set" is idempotent for the agent.
    existing = next(
        (
            b
            for b in await db.list_budgets()
            if b["scope_kind"] == norm["scope_kind"]
            and (b["scope_value"] or "") == norm["scope_value"]
            and b["metric"] == norm["metric"]
            and b["window"] == norm["window"]
        ),
        None,
    )
    if existing is not None:
        await db.update_budget(existing["id"], **norm)
        return await db.get_budget(existing["id"])
    budget_id = await db.add_budget(source="agent", **norm)
    return await db.get_budget(budget_id)


@mcp.tool()
async def remove_budget(
    id: str | None = None,
    scope: str | None = None,
    scope_kind: str | None = None,
    metric: str = "cost_usd",
    window: str = "day",
) -> dict[str, Any]:
    """Delete a budget rule, by ``id`` (full or 8-char prefix) or by its scope
    identity (scope + metric + window). Returns ``{ok, id}``."""
    db = await _get_db()
    rows = await db.list_budgets()
    target = None
    if id:
        matches = [b for b in rows if b["id"] == id or b["id"].startswith(id)]
        if not matches:
            raise ValueError(f"no budget matching id {id!r}")
        if len(matches) > 1:
            raise ValueError(f"ambiguous id {id!r}: matches multiple budgets")
        target = matches[0]
    elif scope is not None:
        norm = normalize_rule_input(
            {"scope": scope, "scope_kind": scope_kind, "amount": 1,
             "metric": metric, "window": window}
        )
        target = next(
            (
                b
                for b in rows
                if b["scope_kind"] == norm["scope_kind"]
                and (b["scope_value"] or "") == norm["scope_value"]
                and b["metric"] == norm["metric"]
                and b["window"] == norm["window"]
            ),
            None,
        )
        if target is None:
            raise ValueError(
                f"no budget for scope={scope!r} metric={metric!r} window={window!r}"
            )
    else:
        raise ValueError("pass an id or a scope to remove")
    await db.delete_budget(target["id"])
    return {"ok": True, "id": target["id"]}


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("OPENAGENT_BUDGET_MCP_LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
