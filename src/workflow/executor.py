from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from src.memory.db import MemoryDB
from src.workflow.blocks import BLOCK_CATALOG, get_block_spec
from src.workflow.templating import (
    TemplateError,
    evaluate_expression,
    resolve_templates,
)
from src.workflow.validate import (
    mcp_callability_from_pool,
    mcp_inventory_from_pool,
    validate_graph,
)

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], Awaitable[None]]


class WorkflowExecutionError(RuntimeError):
    def __init__(self, message: str, run_row: dict | None = None):
        super().__init__(message)
        self.run_row = run_row


@dataclass
class NodeResult:
    output: Any
    taken: frozenset[str] | None = None


@dataclass
class _RunCtx:
    run_id: str
    workflow_id: str
    inputs: dict[str, Any]
    vars: dict[str, Any]
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    graph: dict[str, Any] | None = None
    parent: _RunCtx | None = None

    def to_template_ctx(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs,
            "vars": self.vars,
            "nodes": self.nodes,
            "now": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
        }