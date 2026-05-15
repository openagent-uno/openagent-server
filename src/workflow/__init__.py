"""Workflow engine for OpenAgent — multi-block pipelines executable via
AI, schedule, or manual trigger. See ``openagent/workflow/blocks.py``
for the block type catalog; ``openagent/workflow/executor.py`` for
the DAG walker.
"""

from src.workflow.blocks import (
    BLOCK_CATALOG,
    BlockSpec,
    get_block_spec,
    iter_block_specs,
)
from src.workflow.templating import resolve_templates
from src.workflow.validate import (
    ValidationError,
    mcp_callability_from_pool,
    mcp_inventory_from_pool,
    validate_graph,
)

__all__ = [
    "BLOCK_CATALOG",
    "BlockSpec",
    "get_block_spec",
    "iter_block_specs",
    "resolve_templates",
    "ValidationError",
    "mcp_callability_from_pool",
    "mcp_inventory_from_pool",
    "validate_graph",
]
