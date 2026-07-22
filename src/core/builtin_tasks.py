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
# The quality-scorer: the INTRINSIC self-improvement pass. Unlike the skill
# builtins (OFF by default), this is ON by default (``self_improvement.enabled``)
# — continuous quality self-critique is a built-in capability of every agent, not
# an opt-in. It reads the agent's OWN recent outputs, grades them against the
# agent's OWN vault rules, and writes a grounded correction for each weak one —
# see ``AgentServer._sync_quality_scorer``. Its config section is
# ``self_improvement``. DEDUP: an agent that already ships a tuned, NON-builtin
# quality-scorer (eSound/Lyra do) keeps it — the builtin defers rather than
# double-running.
QUALITY_SCORER_TASK_NAME = "quality-scorer"
# The quality-digest: the daily synthesis half of the intrinsic loop. It rolls
# the scorer's recent findings into concrete grounded improvements —
# auto-applying only SAFE vault rule/doc fixes, proposing anything risky — and
# sends the operator one short recap. ON by default, same
# ``self_improvement`` section, same DEDUP discipline as the scorer.
QUALITY_DIGEST_TASK_NAME = "quality-digest"
# The cost-observability watcher: the CONSUMPTION arm of the intrinsic
# self-improvement loop. Where the quality builtins grade the agent's OUTPUT,
# this watches its SPEND — but CACHE-AWARE: it pages only on REAL cost and FRESH
# (non-cached) tokens via the engine's own ``router.cost_anomaly`` signal, NEVER
# on the raw summed input-token count (which is ~85-90% cheap cache-reads of the
# fixed prefix re-sent every step, so alerting on it pages on nonsense). ON by
# default (``self_improvement.enabled`` AND ``cost_observability_enabled``), same
# ``self_improvement`` section, same DEDUP discipline as the scorer/digest: an
# agent that already ships a tuned, NON-builtin cost watcher (eSound/Lyra do)
# keeps it. See ``AgentServer._sync_cost_observability``.
COST_OBSERVABILITY_TASK_NAME = "cost-observability"

BUILTIN_TASK_NAMES: frozenset[str] = frozenset(
    {
        DREAM_MODE_TASK_NAME,
        AUTO_UPDATE_TASK_NAME,
        SKILL_CURATOR_TASK_NAME,
        SKILL_DISTILLER_TASK_NAME,
        QUALITY_SCORER_TASK_NAME,
        QUALITY_DIGEST_TASK_NAME,
        COST_OBSERVABILITY_TASK_NAME,
    }
)

CONFIG_SECTION_BY_TASK: dict[str, str] = {
    DREAM_MODE_TASK_NAME: "dream_mode",
    AUTO_UPDATE_TASK_NAME: "auto_update",
    SKILL_CURATOR_TASK_NAME: "skills",
    SKILL_DISTILLER_TASK_NAME: "skills",
    QUALITY_SCORER_TASK_NAME: "self_improvement",
    QUALITY_DIGEST_TASK_NAME: "self_improvement",
    COST_OBSERVABILITY_TASK_NAME: "self_improvement",
}
