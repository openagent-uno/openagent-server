"""Service installer rendering tests.

Focus on the pure unit-file generation path so operator-facing knobs such as
optional systemd resource limits stay regression-proof.
"""
from __future__ import annotations

from unittest.mock import patch

from ._framework import TestContext, test


@test("setup", "linux unit omits resource limits by default")
async def t_linux_unit_no_limits_by_default(ctx: TestContext) -> None:
    import openagent.setup.installer as installer

    agent_dir = ctx.test_dir / "service-defaults"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "openagent.yaml").write_text("name: test-agent\n")

    with (
        patch.object(installer, "_get_openagent_cmd", return_value=["/tmp/openagent", "serve", str(agent_dir)]),
        patch.object(installer, "_get_env_path", return_value="/usr/bin:/bin"),
    ):
        unit = installer._build_linux_unit(agent_dir)

    assert "MemoryHigh=" not in unit, unit
    assert "MemoryMax=" not in unit, unit
    assert "MemorySwapMax=" not in unit, unit


@test("setup", "linux unit includes configured systemd overrides and omits nulls")
async def t_linux_unit_systemd_overrides(ctx: TestContext) -> None:
    import openagent.setup.installer as installer

    agent_dir = ctx.test_dir / "service-overrides"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "openagent.yaml").write_text(
        "\n".join(
            (
                "name: test-agent",
                "service:",
                "  systemd:",
                "    MemoryHigh: 2500M",
                "    MemoryMax: 3500M",
                "    MemorySwapMax: null",
                "    TasksMax: 1024",
            )
        )
    )

    with (
        patch.object(
            installer,
            "_get_openagent_cmd",
            return_value=["/tmp/openagent binary", "serve", str(agent_dir)],
        ),
        patch.object(installer, "_get_env_path", return_value="/usr/bin:/bin"),
    ):
        unit = installer._build_linux_unit(agent_dir)

    assert "ExecStart='/tmp/openagent binary' serve" in unit, unit
    assert "MemoryHigh=2500M" in unit, unit
    assert "MemoryMax=3500M" in unit, unit
    assert "TasksMax=1024" in unit, unit
    assert "MemorySwapMax=" not in unit, unit
