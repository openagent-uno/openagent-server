from __future__ import annotations

import iroh

from ._framework import TestContext, test
from src.network.iroh_node import _discovery_config


@test("network", "Iroh discovery keeps its production default")
async def test_iroh_discovery_defaults_to_production(ctx: TestContext) -> None:
    import os

    previous = os.environ.pop("OPENAGENT_IROH_DISCOVERY", None)
    try:
        assert _discovery_config() is iroh.NodeDiscoveryConfig.DEFAULT
    finally:
        if previous is not None:
            os.environ["OPENAGENT_IROH_DISCOVERY"] = previous


@test("network", "Iroh discovery can be disabled for hermetic E2E")
async def test_iroh_discovery_can_be_disabled_for_hermetic_e2e(ctx: TestContext) -> None:
    import os

    previous = os.environ.get("OPENAGENT_IROH_DISCOVERY")
    os.environ["OPENAGENT_IROH_DISCOVERY"] = "none"
    try:
        assert _discovery_config() is iroh.NodeDiscoveryConfig.NONE
    finally:
        if previous is None:
            os.environ.pop("OPENAGENT_IROH_DISCOVERY", None)
        else:
            os.environ["OPENAGENT_IROH_DISCOVERY"] = previous
