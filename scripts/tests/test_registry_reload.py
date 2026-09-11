"""``Agent.refresh_registries`` — the hot-reload probe must notice deletions.

``registry_status`` reports ``MAX(updated_at)`` per table. That value is
**not** monotonic: delete the most recently touched row and it drops back
to the previous row's timestamp. The probe compared with ``>``, so a
deletion moved the value in the one direction the comparison could not
see — the MCP's tools stayed in the model's context after it was removed
(§6 promises the opposite), and the router kept a deleted model. The
``models`` branch additionally stored a ``max()`` high-water mark, so once
a delete lowered the table's value the catalogue stayed stale until
restart.

These tests pin both halves: the database fact that the value goes down,
and the probe reacting to it.
"""
from __future__ import annotations

from types import SimpleNamespace

from ._framework import TestContext, test


@test("registry_reload", "deleting the newest row lowers MAX(updated_at)")
async def t_delete_lowers_max_updated(ctx: TestContext) -> None:
    """The premise the comparison depends on. If this ever stopped being
    true the ``!=`` below would be unnecessary — so assert it rather than
    assume it."""
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        pid = await db.upsert_provider(
            name="reload-probe", framework="api-based", api_key="sk",
        )
        first = await db.upsert_model(provider_id=pid, model="model-first")
        second = await db.upsert_model(provider_id=pid, model="model-second")

        _, before, _, _ = await db.registry_status()

        await db.delete_model(second)
        _, after, _, _ = await db.registry_status()

        assert after <= before, (
            "deleting the most recently written model must not raise "
            f"MAX(updated_at) (before={before}, after={after})"
        )

        await db.delete_model(first)
        await db.delete_provider(pid)
    finally:
        await db.close()


def _probe_agent(status, *, mcp_reloads: list[int]):
    """A stand-in carrying only what ``refresh_registries`` touches."""

    async def _registry_status():
        return status["value"]

    async def _reload():
        mcp_reloads.append(1)

    async def _hydrate():
        return None

    return SimpleNamespace(
        _db=SimpleNamespace(registry_status=_registry_status),
        _mcp=SimpleNamespace(reload=_reload),
        _runtime_models=[],
        _providers_config={},
        _hydrate_providers_from_db=_hydrate,
    )


@test("registry_reload", "a deletion that lowers the stamp still reloads")
async def t_reload_on_lowered_stamp(_ctx: TestContext) -> None:
    from src.core.agent import Agent

    reloads: list[int] = []
    status = {"value": (200.0, 200.0, 1, 200.0)}
    agent = _probe_agent(status, mcp_reloads=reloads)

    # First pass: 0.0 → 200.0 establishes the baseline and reloads once.
    await Agent.refresh_registries(agent)
    assert len(reloads) == 1, f"first probe should reload once: {len(reloads)}"

    # Idle: nothing changed, nothing reloads.
    await Agent.refresh_registries(agent)
    assert len(reloads) == 1, f"an unchanged stamp must not reload: {len(reloads)}"

    # A delete removed the newest row, so the stamp went DOWN. With ``>``
    # this reloaded zero times and the removed MCP kept its tools.
    status["value"] = (100.0, 100.0, 1, 100.0)
    await Agent.refresh_registries(agent)
    assert len(reloads) == 2, (
        f"a lowered stamp means rows changed and must reload: {len(reloads)}"
    )


@test("registry_reload", "the models stamp is not pinned to a high-water mark")
async def t_models_stamp_not_high_water(_ctx: TestContext) -> None:
    """``max()`` could only ratchet up, so after one delete the models
    branch never fired again — not even for a later, legitimate edit."""
    from src.core.agent import Agent

    reloads: list[int] = []
    status = {"value": (0.0, 500.0, 1, 0.0)}
    agent = _probe_agent(status, mcp_reloads=reloads)

    await Agent.refresh_registries(agent)
    assert agent._models_last_updated == 500.0, agent._models_last_updated

    # Delete lowers it.
    status["value"] = (0.0, 300.0, 1, 0.0)
    await Agent.refresh_registries(agent)
    assert agent._models_last_updated == 300.0, (
        "the observed value must be recorded, not the maximum seen: "
        f"{agent._models_last_updated}"
    )

    # And an edit after that delete is still noticed.
    status["value"] = (0.0, 400.0, 1, 0.0)
    reloaded, _ = await Agent.refresh_registries(agent)
    assert reloaded, "an edit following a delete must still reload"
