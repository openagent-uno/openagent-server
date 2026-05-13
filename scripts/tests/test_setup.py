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


@test("setup", "macos plist uses unconditional KeepAlive (parity with Linux Restart=always)")
async def t_macos_plist_keepalive_unconditional(ctx: TestContext) -> None:
    """The launchd plist must use ``<key>KeepAlive</key><true/>`` rather
    than the older conditional ``{SuccessfulExit: false}`` form.

    The conditional form only triggers a relaunch on non-zero exit, so
    any clean shutdown — SIGTERM from the OS during sleep/reboot, an
    internal graceful stop, ``launchctl stop``, OOM-killer with no
    crash report — left the agent dead with no operator notification.
    Linux already uses ``Restart=always``; this asserts the macOS side
    matches so behavior is consistent across platforms.
    """
    import openagent.setup.installer as installer

    agent_dir = ctx.test_dir / "service-macos"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "openagent.yaml").write_text("name: test-agent\n")

    with (
        patch.object(installer, "_get_openagent_cmd", return_value=["/tmp/openagent", "serve", str(agent_dir)]),
        patch.object(installer, "_get_env_path", return_value="/usr/bin:/bin"),
    ):
        plist = installer._build_macos_plist(agent_dir)

    # Unconditional KeepAlive present.
    assert "<key>KeepAlive</key>\n    <true/>" in plist, plist
    # No vestige of the old conditional form.
    assert "SuccessfulExit" not in plist, plist
    # Sanity-check the rest of the plist is well-formed enough.
    assert "<key>RunAtLoad</key>\n    <true/>" in plist, plist
    assert f"<string>com.openagent.{agent_dir.name}</string>" in plist, plist
