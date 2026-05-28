"""Workflow run-event placeholders.

OpenAgent's workflow engine lives in ``src/workflow/`` and emits its own
events through ``src/stream``; the inlined provider layer never sees a
workflow event. The names below exist only so the provider base class
can type-check ``isinstance(event, WorkflowRunOutputEvent)`` checks
without pulling in a workflow-event taxonomy we don't use.
"""

from __future__ import annotations

from typing import Any, Union


class _WorkflowEventBase:
    """Inert base — never instantiated in this runtime."""


class WorkflowStartedEvent(_WorkflowEventBase):
    pass


class WorkflowCompletedEvent(_WorkflowEventBase):
    pass


class WorkflowErrorEvent(_WorkflowEventBase):
    pass


class WorkflowCancelledEvent(_WorkflowEventBase):
    pass


# Union alias used by the provider base for isinstance checks.
WorkflowRunOutputEvent = Union[
    WorkflowStartedEvent,
    WorkflowCompletedEvent,
    WorkflowErrorEvent,
    WorkflowCancelledEvent,
]
