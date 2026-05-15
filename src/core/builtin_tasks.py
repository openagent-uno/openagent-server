"""Built-in scheduled task identifiers.

These rows are seeded into the ``scheduled_tasks`` table by
``AgentServer._sync_*`` at boot and represent OpenAgent's own
maintenance routines (nightly hygiene, weekly self-review, auto-update
poller). They are owned by the framework, not by the user, so the
gateway hides them from ``/api/scheduled-tasks`` and rejects writes —
toggle them via the ``/api/config/{section}`` endpoint and the
matching settings panel instead.

Living in its own tiny module avoids a circular import between
``openagent.core.server`` (which seeds them) and
``openagent.gateway.api.scheduled_tasks`` (which must filter them).
"""

from __future__ import annotations

DREAM_MODE_TASK_NAME = "dream-mode"
MANAGER_REVIEW_TASK_NAME = "manager-review"
AUTO_UPDATE_TASK_NAME = "auto-update"

BUILTIN_TASK_NAMES: frozenset[str] = frozenset(
    {DREAM_MODE_TASK_NAME, MANAGER_REVIEW_TASK_NAME, AUTO_UPDATE_TASK_NAME}
)

CONFIG_SECTION_BY_TASK: dict[str, str] = {
    DREAM_MODE_TASK_NAME: "dream_mode",
    MANAGER_REVIEW_TASK_NAME: "manager_review",
    AUTO_UPDATE_TASK_NAME: "auto_update",
}
