from __future__ import annotations

import os
from pathlib import Path

import click

from ._framework import TestContext, test


@test("local_e2e", "local E2E mode requires a marked temporary fixture")
async def test_local_e2e_guard(ctx: TestContext) -> None:
    from src.cli import _enable_local_e2e

    previous = os.environ.get("OPENAGENT_IROH_DISCOVERY")
    try:
        fixture = ctx.test_dir / "local-e2e-guard"
        fixture.mkdir()
        config_path = fixture / "openagent.yaml"
        config_path.write_text("local_e2e_fixture: true\n", encoding="utf-8")
        config = {
            "local_e2e_fixture": True,
            "_config_path": str(config_path),
        }
        _enable_local_e2e(config, fixture)
        assert config["_local_e2e"] is True
        assert os.environ["OPENAGENT_IROH_DISCOVERY"] == "none"

        try:
            _enable_local_e2e({}, fixture)
        except click.ClickException:
            pass
        else:
            raise AssertionError("unmarked fixture was accepted")

        try:
            _enable_local_e2e(
                {
                    "local_e2e_fixture": True,
                    "_config_path": str(config_path),
                },
                Path.home(),
            )
        except click.ClickException:
            pass
        else:
            raise AssertionError("non-temporary agent directory was accepted")

        escaped_db = fixture / "openagent.db"
        escaped_db.symlink_to(Path.home() / "openagent.db")
        try:
            _enable_local_e2e(
                {
                    "local_e2e_fixture": True,
                    "_config_path": str(config_path),
                },
                fixture,
            )
        except click.ClickException:
            pass
        else:
            raise AssertionError("fixture with a database symlink was accepted")
    finally:
        if previous is None:
            os.environ.pop("OPENAGENT_IROH_DISCOVERY", None)
        else:
            os.environ["OPENAGENT_IROH_DISCOVERY"] = previous


@test("local_e2e", "local E2E agent opens only its database")
async def test_local_e2e_agent_boot_is_hermetic(ctx: TestContext) -> None:
    from src.core.agent import Agent

    class SpyDB:
        connected = False

        async def connect(self) -> None:
            self.connected = True

    class ForbiddenPool:
        async def connect_all(self) -> None:
            raise AssertionError("MCP pool connected in local E2E mode")

    db = SpyDB()
    agent = Agent(
        model=None,
        mcp_pool=ForbiddenPool(),  # type: ignore[arg-type]
        config={"_local_e2e": True},
    )
    agent._db = db  # type: ignore[assignment]

    await agent.initialize()

    assert db.connected is True
    assert agent._initialized is True


@test("local_e2e", "local E2E server omits every background writer")
async def test_local_e2e_server_boot_is_hermetic(ctx: TestContext) -> None:
    from src.core.server import AgentServer

    class FakeAgent:
        name = "fixture"
        model = None
        _db = None

        async def initialize(self) -> None:
            return None

    server = AgentServer(FakeAgent(), {"_local_e2e": True})  # type: ignore[arg-type]

    async def no_network():
        return None

    server._build_network_state = no_network  # type: ignore[method-assign]
    await server.start()

    assert server._scheduler is None
    assert server._bridges == []
    assert server._curator_task is None
