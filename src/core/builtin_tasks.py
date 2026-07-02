"""Built-in scheduled task identifiers.

These rows are seeded into the ``scheduled_tasks`` table by
``AgentServer._sync_*`` at boot and represent OpenAgent's own
maintenance routines (nightly dream-mode maintenance, auto-update
poller). They are owned by the framework, not by the user, so the
gateway hides them from the default ``/api/scheduled-tasks`` list and
rejects writes — toggle them via the ``/api/config/{section}`` endpoint
and the matching settings panel instead. Their run history is still
readable (``?include_builtin=1`` on the list, and the per-task get /
runs endpoints) so a firing surfaces in the app's "Recent" feed like any
other scheduled run.

Living in its own tiny module avoids a circular import between
``openagent.core.server`` (which seeds them) and
``openagent.gateway.api.scheduled_tasks`` (which must filter them).
"""

from __future__ import annotations

DREAM_MODE_TASK_NAME = "dream-mode"
AUTO_UPDATE_TASK_NAME = "auto-update"

BUILTIN_TASK_NAMES: frozenset[str] = frozenset(
    {DREAM_MODE_TASK_NAME, AUTO_UPDATE_TASK_NAME}
)

CONFIG_SECTION_BY_TASK: dict[str, str] = {
    DREAM_MODE_TASK_NAME: "dream_mode",
    AUTO_UPDATE_TASK_NAME: "auto_update",
}
