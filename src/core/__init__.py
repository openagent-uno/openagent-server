"""Core runtime: agent loop, server lifecycle, scheduler, config, prompts.

The submodules import lazily — eagerly importing ``Agent`` /
``AgentServer`` at package-init pulls in ``src.mcp.pool``, which in turn
needs ``src.core.logging``, which re-enters this ``__init__`` and
deadlocks before ``Agent`` is bound. Subprocess MCPs (scheduler /
workflow-manager) trip the cycle because their import path comes through
``src.memory.db`` rather than the main process's order. Importing
``src.core.config.load_config`` lazily lets every caller use the full
``from src.core.<module> import ...`` form without paying that cost.
"""

from src.core.config import load_config

__all__ = ["load_config"]
