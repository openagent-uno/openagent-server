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
# The skill-curator: "dream-mode for skills". A scheduled child session that
# consolidates AGENT-authored SKILL.md playbooks. OFF by default and gated on
# BOTH ``skills.enabled`` and ``skills.curator_enabled`` — see
# ``AgentServer._sync_skill_curator``. Its config section is ``skills`` (the
# ``skills.curator_enabled`` toggle lives there), not a section of its own.
SKILL_CURATOR_TASK_NAME = "skill-curator"
# The skill-distiller: the automatic WRITER half of the self-improvement loop.
# A scheduled child session that reviews RECENT SUCCESSFUL sessions and distills
# a novel, recurring resolution into a NEW ``created_by: agent`` skill. OFF by
# default and gated on BOTH ``skills.enabled`` and ``skills.distiller_enabled``
# — see ``AgentServer._sync_skill_distiller``. Cleanly layered against the
# curator: the distiller only CREATES, the curator only CONSOLIDATES. Same
# ``skills`` config section as the curator (its ``skills.distiller_enabled``
# toggle lives there), not a section of its own.
SKILL_DISTILLER_TASK_NAME = "skill-distiller"

BUILTIN_TASK_NAMES: frozenset[str] = frozenset(
    {
        DREAM_MODE_TASK_NAME,
        AUTO_UPDATE_TASK_NAME,
        SKILL_CURATOR_TASK_NAME,
        SKILL_DISTILLER_TASK_NAME,
    }
)

CONFIG_SECTION_BY_TASK: dict[str, str] = {
    DREAM_MODE_TASK_NAME: "dream_mode",
    AUTO_UPDATE_TASK_NAME: "auto_update",
    SKILL_CURATOR_TASK_NAME: "skills",
    SKILL_DISTILLER_TASK_NAME: "skills",
}
