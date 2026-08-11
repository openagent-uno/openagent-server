"""AgentServer: unified lifecycle for agent, gateway, bridges, and scheduler.

This is the single entry point used by `openagent serve`. It owns the
lifecycle of every long-running piece so there is exactly one place that
starts, supervises and shuts everything down.

    server = AgentServer.from_config(config)
    async with server:
        await server.wait()   # blocks until Ctrl-C / SIGTERM
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from typing import Any
from src.core.agent import Agent
from src.memory.db import MemoryDB
from src.models.runtime import create_model_from_config, wire_model_runtime
from src.core.logging import clear as clear_event_log, elog
# Pre-import the update-flow modules at server boot so the PyInstaller
# archive entries they live in are loaded into memory before any sibling
# service that shares the same on-disk binary can swap it. Without this,
# a /api/update on a sibling (e.g. performa boss/yoanna/friday share
# ~/.local/bin/openagent-stable) replaces the file we lazy-read from,
# and the next ``from src._frozen import is_frozen`` in run_upgrade
# raises ``zlib.error: Error -3 while decompressing data: incorrect
# header check``. Loading these eagerly puts them in sys.modules so
# subsequent ``from`` statements are dict lookups, not archive reads.
import src._frozen  # noqa: F401 — preload for concurrent-update safety
import src.updater  # noqa: F401 — preload for concurrent-update safety
import src.update_guard  # noqa: F401 — preload for concurrent-update safety

logger = logging.getLogger(__name__)

# Exit code that signals the OS service manager to restart the process
RESTART_EXIT_CODE = 75

# Captured at import time (i.e. process start). If this changes by the
# time ``run_upgrade`` is called, a sibling service that shares our
# binary has already swapped it; we short-circuit instead of trying to
# download/apply our own update against a now-stale archive layout.
try:
    _INITIAL_EXECUTABLE_MTIME: float | None = (
        src._frozen.executable_path().stat().st_mtime
        if src._frozen.is_frozen()
        else None
    )
except Exception:  # noqa: BLE001 — best-effort, never block startup
    _INITIAL_EXECUTABLE_MTIME = None

from src.core.builtin_tasks import (
    AUTO_UPDATE_TASK_NAME,
    BUILTIN_TASK_NAMES,
    COST_OBSERVABILITY_TASK_NAME,
    DREAM_MODE_TASK_NAME,
    ESCALATION_AUDIT_TASK_NAME,
    QUALITY_DIGEST_TASK_NAME,
    QUALITY_SCORER_TASK_NAME,
    SKILL_CURATOR_TASK_NAME,
    SKILL_DISTILLER_TASK_NAME,
)
from src.memory.schedule import default_timezone_name


def _compose_task_hook(hook, nxt):
    """Wrap *nxt* (the rest of the run_task chain) with *hook*.

    Used by ``AgentServer._install_task_hook`` to build the dispatcher's
    call chain. ``hook(task, next)`` runs its pre/post logic and calls
    ``await next(task)`` to defer to the remaining hooks + the real
    ``run_task``."""
    async def _composed(task):
        await hook(task, nxt)
    return _composed


# Nightly default for the model-driven dream pass. Kept in the staggered
# maintenance lattice with the other intrinsic agent loops below.
DREAM_MODE_DEFAULT_TIME = "3:31"


DREAM_MODE_PROMPT = """\
You are running in Dream Mode — OpenAgent's nightly self-maintenance
routine. You run while the agent is otherwise idle. Work through both
missions below in order, then write a single dream-log at the end. Be
thorough but non-destructive: when in doubt, skip rather than delete,
and log the uncertainty.

## Mission 1 — Evaluate and correct the memory vault

Curate the memory vault via the `vault` MCP — do NOT cat/grep the
.md files directly.

**First, run the mechanical pass: `vault_dream()`.** One call does
sync → gate → doctor (auto-fix) → regenerate derived. It is
deterministic, offline, and it will not touch a judgement call — so
there is no reason to skip it and no reason to hand-do any of it. It
returns the violations code could NOT fix, which is your actual work
list for this mission. Everything below is about those.

Do not spend the pass re-deciding what the doctor already fixed. If you
want the detail: `vault_gate()` grades, `vault_doctor(apply=False)`
previews. `vault_regenerate_derived()` rebuilds `llms.txt` and the
showcase — `vault_dream()` already did it.

**Then `vault_contradiction_candidates()`.** Vision §5 requires
contradictions to be "flagged and reconciled rather than silently
overwritten" — this is the flagging half, and reconciling is your job.
It is candidate generation, not detection: code matched opposing wording
about a shared subject and never read either note, so expect roughly
half to be false positives. Read BOTH notes in full, then fix or retire
whichever is genuinely stale. An empty result is not proof the vault
agrees with itself — it only sees explicitly deprecated/forbidden
wording. Never delete on this signal alone.

**Then `vault_recall_stats`.** It tells you which notes were
actually read during real runs and how those runs ended, so you spend
this pass on the notes that carry weight instead of walking the vault
alphabetically. Read its `caveat` and believe it: `ok_rate` is
ASSOCIATION, NOT CAUSATION. A run that read six notes and failed credits
all six; a low `ok_rate` is evidence about a note, never a verdict on it.

Use it to decide **where to look**, then judge the note on its content:
   - **Read often, low `ok_rate`** — look here first. Open the note and
     ask why: is it stale, ambiguous, contradicted elsewhere, or missing
     the caveat that would have prevented the failure? Fix what you can
     verify. If the note reads fine, leave it and say so in the log —
     the correlation may be the task's difficulty, not the note.
   - **Read often, high `ok_rate`** — this note is load-bearing. Do not
     merge it away casually, and make sure it is well cross-linked so it
     keeps getting found.
   - **Never read** — a candidate for the orphan/duplicate checks below,
     NOT a reason to delete on its own. A note can be correct, niche,
     and simply not needed yet.
   - **Absent from the stats entirely** — the table only fills as runs
     happen, so on a fresh deployment it is empty. That means "no
     evidence yet", not "no value". Fall back to the checks below.

Never delete a note for a bad number alone. The stats point; you read.

   - Use `list_directory` and `search_notes` to survey the vault.
   - Identify notes that cover the same topic and **merge duplicates**
     into a single canonical note with `write_note` or `patch_note`,
     then `delete_note` the redundant ones.
   - Update any outdated information you can verify from the
     environment (tool versions, paths, hosts that no longer exist,
     etc.).
   - Remove trivially short or empty notes (< 20 words) that add no
     value.
   - **Cross-link related notes with `[[wikilinks]]`**. For every note
     you touch, search the vault for related topics and add backlinks
     where the relationship is meaningful. If a group of notes shares a
     theme, make sure each one links to the others. Prefer
     `patch_note` to add links in place rather than rewriting whole
     notes.
   - **Reconcile contradictions**: when a newer note contradicts an
     older one, fix or retire the stale entry rather than leaving both.
   - Keep frontmatter `tags:` consistent so related notes share tags
     and surface together in future searches.

## Mission 2 — Analyze the last day of logs and fix what is broken

Read OpenAgent's own event log for roughly the last 24 hours and find
issues to fix. Use the `logs` MCP — never `find`/`tail` over
`events.jsonl` by hand. It resolves the log path itself, so there is no
OS-specific path to guess, and it summarises the *whole* window instead
of whatever fits in a tail:

   - `logs_summary(since="24h")` — start here. One call: totals, the
     top failing events, and sample lines.
   - `logs_query(event=..., errors_only=true, since="24h")` — drill
     into one failing event.
   - `logs_context(ts=...)` — read the lines around a failure. The
     error says *what* broke; the lines before it say *why*.

Note `error_like` mixes two schemas. Entries written since the severity
fix carry a real level and are authoritative; older entries predate it,
so their severity is guessed from the event name and can over-report
failures that actually recovered. `logs_summary` reports the split and
says which case this log is in — when a verdict is a guess, confirm it
from the event's own payload before acting on it.

Look for problems and act on them:
   - **Broken scheduled tasks**: tasks that errored or produced empty
     output. Inspect them via the `scheduler` MCP
     (`scheduler_list_scheduled_tasks`), confirm whether the prompt is
     still accurate, and fix, reschedule, or retire the task.
   - **Broken workflows**: workflow runs that failed or stalled.
     Inspect via the `workflow-manager` MCP and repair the definition,
     or clearly report the failure if you cannot fix it.
   - **Recurring errors**: model-call failures, MCP errors, federation
     or channel errors that repeat. Diagnose the cause, fix what is in
     your power (a stale path, a misconfigured task), and log what
     still needs a human.

## Mission 3 — Notice what should be automated, and say so

You are expected to be proactive: surface the patterns you notice
rather than waiting to be asked. You already have the week in front of
you from Mission 2, so use it.

   - **Recurring work**: widen to `logs_summary(since="7d")`. Did the
     same kind of task run three or more times with only minor
     variation? That is a scheduled task or a workflow that doesn't
     exist yet. Propose it concretely — the exact prompt and cron, or
     the block outline — via the `scheduler` / `workflow-manager` MCPs.
   - **Promises that never landed**: search the vault for notes tagged
     `pending-automation` or `followup`. For each, decide: is the
     pattern still live? Then schedule it now. Is it dead? Then archive
     the note. A followup note that survives untouched for months is a
     decision nobody made.
   - **Things you said you'd remember**: skim the recent sessions for
     "I'll remember that", "next time", "we decided" that never became
     a note. Write the missing notes.

Propose, don't impose. Creating a scheduled task that spends money on
the user's behalf every night is a decision they should make — write
what you would automate and why into the dream-log, and create it only
when the pattern is unambiguous and cheap.

## Log the dream

Use `write_note` to save a concise summary under
`dream-logs/dream-log-YYYY-MM-DD.md` with frontmatter `type: dream-log`
and `date:` set to today. Record, per mission: what you
merged/updated/cross-linked/removed in the vault, which log issues you
found, what you fixed, what you would automate and why, and what still
needs the user's decision.

Include the recall findings explicitly: which notes `vault_recall_stats`
sent you to, what you concluded when you actually read them, and — this
one matters — which ones you examined and found FINE despite a low
`ok_rate`. Without that, the next run re-investigates the same note
forever and the number slowly reads as guilt rather than as a pointer.

Use the `vault` MCP's tools for all vault access — never shell out for
anything under the memory vault.
"""

# Weekly by default (Sunday 04:53). Skills change far more slowly than the
# memory vault, so the curator runs on a longer cadence than nightly dream
# mode. The :53 slot deliberately avoids every other model-driven builtin:
# otherwise several full agent loops can hit a single-concurrency local model
# at once and starve an interactive turn. Overridable via
# ``skills.curator_schedule``.
SKILL_CURATOR_DEFAULT_CRON = "53 4 * * 0"

SKILL_CURATOR_PROMPT = """\
You are running as the Skill Curator — OpenAgent's self-improvement pass
for its SKILL.md library. This is "dream mode for skills": you run on a
schedule while the agent is otherwise idle, consolidate the skills the
AGENT has taught itself, and leave everything else untouched. Be
thorough but conservative: when in doubt, skip and log it, never delete
or overwrite on a guess.

## The provenance boundary — read this first, it is the whole job

You may ONLY touch skills that the agent authored itself. Those carry
`created_by: agent` in their SKILL.md frontmatter. Every other skill —
the seed/bundled playbooks that shipped with OpenAgent, and any skill a
human wrote or edited — has NO such stamp and is OFF-LIMITS. Do not
merge it, do not archive it, do not rewrite it, do not fold its content
into another skill. Curated seed content is not yours to consolidate.

Concretely, before you act on ANY skill:
   - Open it with `skill_view` and check its frontmatter `created_by`.
   - If it is not exactly `agent`, leave it exactly as it is.
   - Your entire work set is the agent-authored skills. If none exist
     yet, there is nothing to do — say so in the log and stop.

## Mission 1 — Find the agent-authored skills

Use `skill_search` (empty or broad query) to list what exists, then
`skill_view` each candidate to read the full body AND confirm
`created_by: agent`. Build your work set from only those. Note which
category each sits in — overlap within a category is where consolidation
pays off.

## Mission 2 — Merge overlapping agent skills

When two or more AGENT-authored skills cover the same task with only
minor variation:
   - Decide the single canonical skill (the clearest, most complete one).
   - `skill_manage(action="update", ...)` it to absorb anything useful
     from the others — steps, edge cases, examples — into one coherent
     body. Keep the frontmatter `name`/`description`/`category` accurate.
   - `skill_manage(action="archive", ...)` the now-redundant duplicates.
     Archiving retires them from the prompt index but keeps the file on
     disk, so the merge is reversible if you got it wrong. Do NOT
     `remove` (hard-delete) unless a skill is empty or plainly junk.
   - Never merge ACROSS the provenance boundary: if the overlap is with
     a seed/user skill, the agent skill is the redundant one — archive
     the agent skill, and leave the seed skill alone.

## Mission 3 — Archive stale agent skills

An agent skill is stale when its steps reference tools, paths, hosts, or
flows that no longer exist, or it was written for a one-off that will not
recur. Verify the staleness from the current environment before acting —
a skill that merely looks unfamiliar is not stale. Retire a genuinely
stale agent skill with `skill_manage(action="archive", ...)`, never a
hard `remove`, so a human can review the decision.

## Log what you did

Write a concise note via the `vault` MCP under
`skill-curator-logs/skill-curator-YYYY-MM-DD.md` (frontmatter
`type: skill-curator-log`, `date:` today): which agent skills you
reviewed, what you merged into what, what you archived and why, and
which overlaps you deliberately left alone (especially any that touched
the seed/user boundary). If you changed nothing, log that too — a clean
pass is a valid outcome.

Remember: seed and user skills are never yours to touch. The only skills
that leave this pass changed are ones stamped `created_by: agent`.
"""

# Daily by default (03:53). The distiller MINES new patterns, so it runs on a
# shorter cadence than the weekly curator that CONSOLIDATES them: a new recurring
# resolution should become a skill within a day of recurring, and the curator's
# weekly pass then folds any near-duplicates together. Its minute slot is part
# of the staggered builtin-maintenance schedule. Overridable via
# ``skills.distiller_schedule``.
SKILL_DISTILLER_DEFAULT_CRON = "53 3 * * *"

SKILL_DISTILLER_PROMPT = """\
You are running as the Skill Distiller — OpenAgent's self-improvement pass
that WRITES new skills. This is the automatic authoring half of the loop:
you run on a schedule while the agent is otherwise idle, look back over the
work that actually went well, and turn a *recurring, reusable* resolution
into a brand-new SKILL.md so the next agent that hits the same task starts
from a playbook instead of re-deriving it. Be generous in what you look at
but stingy in what you write: a skill earns its place only by being both
NOVEL and RECURRING. When in doubt, write nothing and log why.

## Your lane — you CREATE, you never consolidate

There are two skill-improvement passes and they are cleanly layered:
   - YOU (the distiller) only CREATE new skills, via
     `skill_manage(action="create", ...)`. Every skill you write is stamped
     `created_by: agent` automatically — that is what makes it yours and the
     curator's, never a seed/user skill.
   - The skill-curator is the OTHER pass. IT merges overlapping skills,
     archives stale or redundant ones, and rewrites bodies. That is NOT your
     job. Do NOT call `skill_manage` with `action` of `update`, `archive`, or
     `remove`. If you find two skills that ought to be merged, that is a note
     for the curator — leave them both and say so in your log.

## Mission 1 — Find what actually recurs and actually worked

Distil from EVIDENCE of success, not a hunch. You have three real signals
OpenAgent already collects; use them, don't guess:

   - `search_past_conversations` — keyword recall over what was SAID in recent
     sessions. Probe for repeated task shapes: the same kind of request, the
     same failure hit more than once, the same fix applied again. A pattern
     that shows up in ONE session is a one-off; one that shows up across
     several is a candidate.
   - Your own semantic recall — when a task reminds you of earlier ones,
     follow that thread. Two conversations that reached the same resolution by
     different wording are exactly the recurring pattern worth a skill, and
     keyword search alone will miss them.
   - `vault_recall_stats` — the outcome ledger. It records which notes were
     recalled and how those runs ended (the `ok` count is the `OUTCOME_OK`
     tally). A note with several recalls and a high share of `ok` runs marks a
     resolution that keeps getting reused AND keeps landing well — a strong
     distillation target. Heed its `caveat`: this is ASSOCIATION not
     causation, and a note with a tiny `scorable` count is noise, so READ the
     underlying sessions before you trust the number.

Build a short shortlist of candidate patterns, each backed by more than one
successful session.

## Mission 2 — Reject anything that already exists

Before writing ANY skill, search the existing library for overlap — this is
the one check that keeps you from flooding the index with near-duplicates
(which is precisely the mess the curator then has to clean up):
   - Run `skill_search` with the candidate's key terms AND a paraphrase of
     what it does. Read any hit with `skill_view`.
   - If a skill already covers this task — even under a different name or with
     slightly different steps — do NOT create a second one. Skip it and log
     the overlap so the curator can improve the existing skill if yours had a
     better step. Adding value to an existing skill is the curator's job, not
     a new file.
   - Only a candidate with NO existing coverage survives to Mission 3.

## Mission 3 — Write the new skill

For each surviving candidate, write ONE skill with
`skill_manage(action="create", name=..., description=..., category=...,
body=...)`:
   - `name`: short, task-shaped, unique (Mission 2 proved it's not taken).
   - `description`: one line — what task it's for, so the index entry alone
     tells a future agent when to open it.
   - `category`: reuse an existing category when the task fits one (check the
     index), so related skills group together.
   - `body`: the actual playbook — the concrete steps, the tools to call, the
     edge cases you saw recur, and a real example drawn from the sessions you
     distilled from. Write what you WISH had existed the first time. Do not
     include your own `---` frontmatter block; it is generated for you.
   Keep each skill to a single coherent task. If a candidate is really two
   tasks, write two skills or write none — never a grab-bag.

## Log what you did

Write a concise note via the `vault` MCP under
`skill-distiller-logs/skill-distiller-YYYY-MM-DD.md` (frontmatter
`type: skill-distiller-log`, `date:` today): which patterns you found and the
sessions that evidenced them, which you created (with names), which you
skipped for overlap (naming the existing skill), and any merge the curator
should look at. If you created nothing, log that too — a pass that writes no
skill because nothing new recurred is a correct, valid outcome, not a
failure. Writing a marginal skill just to have written one is the failure.
"""

# Every 2 hours at :23 by default. The scorer grades a short window of the agent's own
# recent output, so it runs frequently and cheaply — a regression should surface
# within a couple of hours, not the next day. The offset keeps it from colliding
# with the hourly cost watcher. Overridable via ``self_improvement.scorer_schedule``.
QUALITY_SCORER_DEFAULT_CRON = "23 */2 * * *"

QUALITY_SCORER_PROMPT = """\
You are running as the Quality Scorer — OpenAgent's INTRINSIC self-improvement
pass. Every couple of hours you look back at what THIS agent itself just
produced, grade it honestly against the agent's OWN rules, and write a grounded
correction for anything weak so the same mistake does not recur. This is
role-agnostic: if the agent answers customers, its outputs are the replies it
sent; if the agent runs ops, its outputs are the actions, decisions, and edits
it made. Grade whatever this agent actually produces.

You are a STRICT, INDEPENDENT judge of your own work — not a cheerleader. A
clean run that finds nothing wrong is the EXPECTED, normal outcome: do NOT
invent faults to look busy, and do NOT fabricate work that did not happen.
Equally, do not wave through a real error to be kind to yourself. Stay SILENT
unless there is a real regression (STEP 4) — no per-run status message.

## STEP 1 — Gather this agent's OWN recent outputs (since the last run, ~2h)

Read the agent's own recent COMPLETED sessions for the window and extract what
it actually PRODUCED — the customer-facing replies it sent, and/or the ops
actions / decisions / edits it took. Use the tools OpenAgent already gives you,
never shell out:
   - `search_past_conversations` — keyword recall over what was said/done in
     recent sessions; probe for the outbound work of the last ~2 hours.
   - the `logs` MCP — `logs_summary(since="2h")` for the shape of the window,
     then `logs_query` / `logs_context` to pull the specific outputs and the
     context around them (what was asked, what the agent answered or did).
If there is NOTHING new to grade in the window, STOP — a quiet window is a valid
outcome. Do not manufacture a batch.

## STEP 2 — Ground in the agent's OWN rules FIRST, then grade

Before scoring anything, SEARCH AND READ the agent's OWN vault for the
rules, procedures, and quality-bar that apply to this work — use the `vault`
MCP (`search_notes`, `read_note`) to find the standing rules, the
anti-fabrication discipline, the product/ops facts, and any escalation policy.
You grade against the agent's OWN documented bar, not a generic one.

Then grade EACH output GOOD / OK / BAD with a 0..1 score and short per-dimension
notes:
   - **correctness** — is the substance right for the real situation?
   - **grounding / no-fabrication** — every claim true and verifiable; NO
     invented state (no "we've logged/escalated/fixed it" that never happened,
     no facts the agent could not have known). Fabrication is the cardinal sin:
     a fabricated claim caps the verdict at BAD.
   - **follows-own-rules** — does it obey the standing rules you just read from
     the vault, or does it violate one?
   - **tone** — appropriate and, where a human is involved, warm/empathetic —
     especially with a frustrated user.
   - **completeness** — does it actually resolve the ask / finish the action,
     or leave it dangling?
The overall score is your honest weighting of these (weight correctness and
grounding highest). Verdict: GOOD >= 0.8, OK 0.5–0.8, BAD < 0.5.

## STEP 3 — Write a grounded correction for each weak output

For every BAD output — and every OK one with a real, nameable flaw — write ONE
grounded CORRECTION tied to that specific work, via the `vault` MCP
(`write_note`, or `patch_note` to extend a matching note). Each correction
states plainly: WHAT was wrong, WHICH of the agent's own rules it violated (cite
the note/rule), and HOW to do it right next time. Keep it about procedure and
discipline — a repeatable lesson — not a new product fact you are unsure of. If
what's missing is a FACT you cannot verify, record it as a "grounding gap" to
review rather than asserting it as truth. File corrections under a stable
folder (e.g. `quality-corrections/`) with frontmatter `type: quality-correction`
and `date:` today, so the daily digest can find them.

## STEP 4 — Save, stay quiet, and alert ONLY on a real regression

SAVE the scores and corrections and STOP. Do NOT send a per-run message — a
clean or ordinary batch produces no notification at all; the daily digest is
where routine results are summarised.

Send the operator EXACTLY ONE short alert — via whatever operator-notification
tool this agent has (its Telegram / Slack / messaging tool) — ONLY when THIS
batch is a genuine regression: the batch average falls below the quality bar
(~0.65) OR there are >= 2 BAD outputs. The alert names the batch average, how
many BAD, and the single worst item with a one-line reason. In every other case
send nothing.

## Log the pass

Write a concise note via the `vault` MCP under
`quality-scorer-logs/quality-scorer-YYYY-MM-DD.md` (frontmatter
`type: quality-scorer-log`, `date:` today): how many outputs you graded, the
GOOD/OK/BAD split and average, the corrections you wrote, and whether you
alerted. If the window was empty or clean, log that too — it is a correct
outcome, not a failure.
"""

# Daily at 09:41 by default. The digest CONSOLIDATES what the every-2h scorer
# found, so it runs on a slower cadence — one operator-facing recap a day.
# Its offset avoids the hourly cost watcher. Overridable via
# ``self_improvement.digest_schedule``.
QUALITY_DIGEST_DEFAULT_CRON = "41 9 * * *"

QUALITY_DIGEST_PROMPT = """\
You are running as the Quality Digest — the daily synthesis half of OpenAgent's
intrinsic self-improvement loop. The quality-scorer grades the agent's own work
every couple of hours and files corrections; ONCE a day you roll those findings
into concrete, grounded improvements, apply the safe ones yourself, propose the
risky ones, and send the operator a single short recap. Cheap by design: no
fan-out, one pass. A quiet day is the normal outcome — say so in one line and
stop.

## STEP 1 — Read the recent quality findings (read-only)

Gather what the scorer produced over roughly the last 7 days via the `vault`
MCP: the corrections it filed (`search_notes` for `type: quality-correction`)
and the scorer logs (`type: quality-scorer-log`). Compute the trend — is the
average holding, rising, or falling? — and group the corrections by RECURRING
PATTERN (e.g. fabricated state, ignored attachment/context, wrong procedure,
tone with a frustrated user, incomplete action). Count each pattern.

## STEP 2 — Turn recurring patterns into improvements

For each pattern that recurs (roughly 3+ times), decide the concrete fix and
split by SAFETY:
   - **AUTO-APPLY — only SAFE, grounded fixes.** A tightening of an existing
     vault rule / doc, or a small new procedural rule that is clearly grounded
     in the corrections and does not assert an unverified fact. Apply it with
     the `vault` MCP (`patch_note` to extend an existing rule — prefer this — or
     `write_note` for a genuinely new one). Before writing: DEDUP against the
     existing rules so you extend rather than duplicate, and keep always-injected
     standing rules LEAN (they cost tokens on every run) — cap yourself at a few
     changes per day and prefer editing an existing rule over adding another.
   - **PROPOSE — everything risky.** Anything that asserts a NEW product/ops
     fact you cannot fully verify, changes behaviour beyond a doc/rule, or that
     you are unsure about: do NOT apply it. Write the exact proposed change and
     hand it to the operator in the recap. Never ship a code/release/behaviour
     change autonomously from here.

## STEP 3 — One short operator recap, then stop

Send the operator EXACTLY ONE short, scannable message via whatever
operator-notification tool this agent has (Telegram / Slack / messaging): the
7-day average and its trend, the GOOD/OK/BAD counts, the weakest area, the rules
you auto-improved today (by title), the recurring patterns with counts, and any
proposals or grounding-gaps that need the operator's decision. If the day is
quiet — quality stable and nothing to promote — send only a single line, e.g.
"Quality stable: avg <X>, 0 changes.", and stop. No walls of text, no second
message.

## Log the digest

Write a concise note via the `vault` MCP under
`quality-digest-logs/quality-digest-YYYY-MM-DD.md` (frontmatter
`type: quality-digest-log`, `date:` today): the trend, the patterns you found
with counts, what you auto-applied, what you proposed, and what still needs the
operator. A quiet day is a valid, correct outcome — log it as such.
"""


def _is_custom_quality_scorer_name(name: str) -> bool:
    """A NON-builtin scheduled-task name that serves the quality-scorer
    purpose — an agent's own tuned grader (eSound/Lyra ship one). Matches when
    the name mentions quality AND scoring, case-insensitively."""
    low = name.lower()
    return "quality" in low and ("scor" in low or "score" in low)


def _is_custom_quality_digest_name(name: str) -> bool:
    """A NON-builtin scheduled-task name that serves the quality-digest
    purpose — an agent's own tuned synthesis pass. Matches when the name
    mentions quality AND digest/improvement, case-insensitively."""
    low = name.lower()
    return "quality" in low and ("digest" in low or "improve" in low)


# Hourly at :07 by default: a cheap check that pages only on a genuine,
# CACHE-AWARE cost anomaly. Running off the hour is load-bearing: several other
# builtin maintenance jobs used to start at :00, creating competing full agent
# loops that could exhaust or starve a single-concurrency model provider.
# Overridable via ``self_improvement.cost_observability_schedule``.
COST_OBSERVABILITY_DEFAULT_CRON = "7 * * * *"

COST_OBSERVABILITY_PROMPT = """\
You are running as Cost Observability — the CONSUMPTION arm of OpenAgent's
intrinsic self-improvement loop. Where the quality pass grades what this agent
PRODUCES, you watch what it SPENDS, and you page the operator ONLY when spend is
genuinely anomalous. Cheap by design: a couple of reads, no fan-out. A quiet
hour is the normal, expected outcome — stay SILENT.

## The cardinal rule: be CACHE-AWARE — never alert on raw token counts

An agentic run makes many model calls, each re-sending the growing conversation
prefix (system prompt + memory + tool schemas). That prefix is CACHED, so each
re-send is a cheap cache-READ — but the raw summed ``input_tokens`` counts it at
full face value, so an ordinary run shows a six-figure "input" while ~85-90% is
cached and the real cost is a fraction of a cent. NEVER alert on that raw number:
it is noise, and paging on it is exactly the nonsense this task exists to avoid.
Alert only on signals that track ACTUAL spend:
   - **real cost** (``cost_usd`` — already cache-discounted); and
   - **FRESH / non-cached tokens** = ``input_tokens - cache_read_tokens`` — the
     tokens genuinely re-processed (a real loop or a truly huge fresh prompt),
     not the cached prefix re-sent each step.
The engine already does this math for you and emits a ``router.cost_anomaly``
event (carrying ``cost_usd`` and ``uncached_input_tokens``) ONLY when a run
crosses a real-cost or fresh-token threshold. Your job is to surface those —
never to re-derive an alarm from raw input.

## STEP 1 — Check for genuine cost anomalies since the last run (~1h)

Using the `logs` MCP, look for any ``router.cost_anomaly`` warning in the last
~65 minutes (`logs_query` for that event name; `logs_summary(since="1h")` for
the window shape). Each carries the real ``cost_usd``, the
``uncached_input_tokens`` (fresh), the model, and the session id — the engine
already did the cache-aware math. If there are NONE, the hour is clean: send
nothing and go to STEP 2.

If one or more fired, send the operator EXACTLY ONE short alert via whatever
operator-notification tool this agent has (its Telegram / Slack / messaging
tool). For each anomaly name the REAL cost, the FRESH (non-cached) token count,
the model, and the session — and say plainly whether it is paid-model spend or a
possible context loop. Report cost and FRESH tokens ONLY — NEVER the raw summed
input. If a flagged session is still RUNNING (a live runaway), say so first:
that is the one worth stopping.

## STEP 2 — Daily cost recap (ONLY once a day, near 08:00 local time)

Only when it is roughly 08:00 in this agent's local timezone (otherwise SKIP
entirely — no message): send the operator ONE short line summarising the last
24h of real spend. State the real cost for the day (metered models only — a
flat-rate / subscription proxy logs $0, which is correct, not a bug) and, if you
can get it cheaply, the cache hit-rate as context (raw input vs fresh). One
scannable line, e.g. "Cost 24h: $X real (local proxy $0); ~P% cached." No walls
of text, no second message.

## Nothing else

Do not fan out and do not touch customer/billing state. If STEP 1 found no
anomaly and it is not the daily-recap hour, send NOTHING at all — a silent hour
is a correct, healthy outcome, not a failure to report.
"""


def _is_custom_cost_observability_name(name: str) -> bool:
    """A NON-builtin scheduled-task name that serves the cost-observability
    purpose — an agent's own tuned cost/spend watcher (eSound/Lyra ship one).
    Matches when the name mentions cost AND observe/anomaly/monitor,
    case-insensitively, so the builtin defers to it rather than double-running."""
    low = name.lower()
    return "cost" in low and ("observ" in low or "anomal" in low or "monitor" in low)


# Daily at 08:47 by default: a cheap self-audit of the agent's own handoffs.
# The stagger leaves room around the :07 cost watcher and :23 scorer.
# Overridable via ``self_improvement.escalation_audit_schedule``.
ESCALATION_AUDIT_DEFAULT_CRON = "47 8 * * *"

ESCALATION_AUDIT_PROMPT = """\
You are running as the Escalation Audit — the HANDOFF arm of OpenAgent's
intrinsic self-improvement loop. The quality-scorer grades what this agent
PRODUCES and cost-observability watches what it SPENDS; you audit what it HANDS
OFF — the things it escalated to a human, flagged as needing a human, or left
unresolved — and ask, honestly, whether each GENUINELY needed a human or was an
OVER-ESCALATION the agent could have handled itself. Agents routinely have more
capability than they use: they escalate a case a tool of theirs would resolve,
or give up instead of asking for the one missing piece of information. Cheap by
design: read-only inspection, no fan-out. A clean audit is the normal, expected
outcome — stay SILENT unless there is a real regression.

This is ROLE-AGNOSTIC and TOOL-AGNOSTIC. OpenAgent is a general engine, not only
a support runtime — so audit whatever THIS agent actually hands off, using
whatever tools and vault knowledge THIS agent has. Do not assume any particular
product, queue, or MCP. If this agent never hands work to a human, there is
nothing to audit — say so in one line and stop.

## STEP 1 — Gather this agent's OWN recent handoffs (since the last run)

Find what this agent escalated / marked needs-human / handed to a person / left
unresolved in the recent window. Use the tools this agent already has: its
support/queue tools if it is a support agent (e.g. list the items currently
waiting for a human), its `logs` MCP (`logs_summary`/`logs_query`) for escalation
events, `search_past_conversations` for "escalated / forwarded to the team /
marked for human" moments. If there are NONE, STOP — a quiet window is a valid,
healthy outcome. Do not manufacture a batch.

## STEP 2 — Ground in this agent's OWN capabilities, then judge each handoff

SEARCH AND READ this agent's OWN vault (`vault` MCP) for what it is ACTUALLY able
to do — its tools, procedures, resolution playbooks, and its escalation policy
(when a human IS genuinely required). Then judge each handoff:
   - **JUSTIFIED** — genuinely needs a human: a real decision/policy call, a
     legal matter, a platform limit that blocks any reply (e.g. a channel's
     messaging window has closed — no human could act either), or a case no tool
     of this agent covers. These are correct; leave them.
   - **OVER-ESCALATED** — the agent had a tool/procedure that would have resolved
     it, or only needed to ASK the user for the one missing detail (an id, an
     email, a receipt) instead of handing off. This is the defect.
Be careful NOT to count a platform-unreachable case (nobody can act) as
over-escalation — that inflates the number and makes the audit cry wolf. Only a
handoff the agent COULD have progressed is over-escalation.

## STEP 3 — File a grounded correction for the over-escalation PATTERN

If over-escalations recur, write ONE grounded correction via the `vault` MCP
(`write_note`/`patch_note`, folder e.g. `quality-corrections/`, frontmatter
`type: quality-correction`, `date:` today): NAME the pattern (e.g. "asked-nothing,
escalated instead of requesting the missing id"; "had a tool that resolves this,
didn't use it"), CITE the agent's own rule/tool that should prevent it, and state
the fix. Keep it a repeatable lesson, not a new unverified fact. The daily
quality-digest promotes it. Do NOT mass-mutate the individual handed-off items
here — correct the RULE; the normal flow re-processes them.

## STEP 4 — Save, stay quiet, alert ONLY on a real regression

Send the operator EXACTLY ONE short alert — via whatever operator-notification
tool this agent has — ONLY when the over-escalation rate is a genuine regression
(e.g. > 30% of the audited handoffs were avoidable). Name the rate, the dominant
pattern, and that a correction is filed. In every other case send nothing.

## Log the pass

Write a concise note via the `vault` MCP under
`escalation-audit-logs/escalation-audit-YYYY-MM-DD.md` (frontmatter
`type: escalation-audit-log`, `date:` today): how many handoffs you reviewed, the
justified/over-escalated split and rate, the pattern, and whether you alerted.
Track the rate's trend so you can tell if the correction is working. A quiet or
clean window is a correct outcome — log it as such.
"""


def _is_custom_escalation_audit_name(name: str) -> bool:
    """A NON-builtin scheduled-task name that serves the escalation-audit
    purpose — an agent's own tuned handoff/escalation auditor (eSound/Lyra ship
    ``support-escalation-audit``). Matches when the name mentions
    escalation/handoff AND audit/hygiene, case-insensitively, so the builtin
    defers to it rather than double-running."""
    low = name.lower()
    return ("escalat" in low or "handoff" in low or "hand-off" in low) and (
        "audit" in low or "hygiene" in low or "review" in low
    )


def _build_agent(config: dict) -> Agent:
    """Build an Agent from a config dict (factored out of cli.py)."""
    from src.core.paths import default_db_path

    model = create_model_from_config(config)

    # Export channel tokens as env vars so the messaging MCP can pick them up.
    # ``or {}`` because yaml.safe_load returns ``None`` for an empty mapping
    # (e.g. ``channels:`` with no children) — ``dict.get`` won't substitute
    # the default in that case.
    channels_config = config.get("channels") or {}
    if "telegram" in channels_config:
        token = channels_config["telegram"].get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")
        if token:
            os.environ["TELEGRAM_BOT_TOKEN"] = token
    if "discord" in channels_config:
        token = channels_config["discord"].get("token") or os.environ.get("DISCORD_BOT_TOKEN")
        if token:
            os.environ["DISCORD_BOT_TOKEN"] = token
    if "whatsapp" in channels_config:
        wa = channels_config["whatsapp"]
        if wa.get("green_api_id"):
            os.environ["GREEN_API_ID"] = wa["green_api_id"]
        if wa.get("green_api_token"):
            os.environ["GREEN_API_TOKEN"] = wa["green_api_token"]
    if "slack" in channels_config:
        sl = channels_config["slack"]
        if sl.get("bot_token"):
            os.environ["SLACK_BOT_TOKEN"] = sl["bot_token"]
        if sl.get("app_token"):
            os.environ["SLACK_APP_TOKEN"] = sl["app_token"]

    # Safety toggles — read from ``safety.*`` and exported as env vars so the
    # blocklist can read them without plumbing the agent config object through
    # every callsite. Default preserves current behaviour: approvals OFF.
    #
    # ``safety.approvals`` is the only stanza here that does anything, and as
    # of this change it finally does. It is enforced by
    # ``src.core.safety.check_command_allowed`` at the ``shell_exec`` callsite
    # (``src/mcp/servers/shell/handlers.py``) — read that module's header for
    # what the blocklist covers and, importantly, what it does not.
    #
    # ``safety.guardrails`` and ``safety.compression`` are no longer read.
    # Both were parsed here into ``OPENAGENT_SAFETY_GUARDRAILS`` /
    # ``OPENAGENT_SAFETY_COMPRESSION{,_THRESHOLD_TOKENS}``, which *nothing
    # anywhere read* — they described the retired Claude-SDK turn loop ("abort
    # SDK turn on 5 same-tool failures", "drop SDK client at threshold") and
    # died with it in ``e8f5d68`` without anyone noticing the config had gone
    # inert. Exporting a safety-shaped env var that has no reader is worse than
    # exporting nothing: it greps like a live mitigation. Session compaction is
    # a real, shipped feature — it is simply configured elsewhere, via
    # ``OPENAGENT_COMPACTION_*`` (see ``src/core/compaction.py``). A stale
    # ``guardrails``/``compression`` block in an existing ``openagent.yaml`` is
    # inert rather than an error, the same way every other retired key degrades
    # here.
    # ``scheduler.timezone`` — the zone new scheduled tasks are created in.
    # Exported before the MCP pool spawns, because the ``scheduler`` MCP runs
    # as a subprocess and writes task rows itself; it can only see the default
    # through the environment it inherits.
    #
    # Crons evaluate in **UTC**, not host-local: ``next_run_for_expression``
    # hands croniter a float, which croniter resolves via
    # ``fromtimestamp(ts, tz=utc)``. So an untagged ``0 9 * * *`` is 09:00 UTC
    # on every machine, which is why operators end up hand-converting
    # (``23 11 * * 1-5 UTC ~= 13:23 Europe/Rome``) and why DST silently moves
    # their briefings twice a year.
    #
    # This default is materialised into each new row at creation time and is
    # NEVER resolved for existing NULL-timezone rows at fire time. That is the
    # whole safety property: setting it re-aims nothing that already exists —
    # every hand-converted cron on every deployment keeps firing at exactly the
    # instant it fires today.
    _sched_cfg = config.get("scheduler") or {}
    if _sched_cfg.get("timezone"):
        from src.memory.schedule import DEFAULT_TZ_ENV, validate_timezone

        _tz = str(_sched_cfg["timezone"]).strip()
        # Fail at boot on a bad zone rather than at 3am on the first firing.
        validate_timezone(_tz)
        os.environ[DEFAULT_TZ_ENV] = _tz

    safety_config = config.get("safety") or {}
    _approvals_cfg = (safety_config.get("approvals") or {})
    if "enabled" in _approvals_cfg:
        os.environ["OPENAGENT_SAFETY_APPROVALS"] = (
            "1" if bool(_approvals_cfg["enabled"]) else "0"
        )
    if _approvals_cfg.get("block_extra_patterns"):
        extras = _approvals_cfg["block_extra_patterns"]
        if isinstance(extras, (list, tuple)):
            os.environ["OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS"] = ",".join(
                str(p) for p in extras
            )
        elif isinstance(extras, str):
            os.environ["OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS"] = extras
    # ``allow_patterns`` exempts a command from the block list, and is what
    # makes the whole stanza usable rather than theoretical. ``git push
    # --force`` is blocked by default, but an autonomous agent that owns its
    # own branch force-pushes it every run by design — without an exception
    # such an operator must choose between "blocklist off entirely" and
    # "break my agent", and will pick off. See ``src.core.safety._compile_allow``.
    if _approvals_cfg.get("allow_patterns"):
        allows = _approvals_cfg["allow_patterns"]
        if isinstance(allows, (list, tuple)):
            os.environ["OPENAGENT_SAFETY_ALLOW_PATTERNS"] = ",".join(
                str(p) for p in allows
            )
        elif isinstance(allows, str):
            os.environ["OPENAGENT_SAFETY_ALLOW_PATTERNS"] = allows

    # ``sandbox`` — opt-in hardened exec backend for the ``shell`` tool. The
    # default is the local host backend, whose spawn path is byte-identical to
    # pre-sandbox behaviour, so an absent stanza (or ``backend: local``) changes
    # nothing. Only ``backend: docker`` routes shell commands into a locked-down
    # container (see ``src/mcp/servers/shell/backends.py``). Exported as env
    # vars for the same reason as ``safety.approvals`` above: the in-process
    # shell backend reads them via a selector that defaults to local when unset,
    # and any subprocess-hosted reader could only see policy through the
    # environment anyway. ``local`` downstream is a harmless no-op.
    _sandbox_cfg = config.get("sandbox") or {}
    _sandbox_backend = str(_sandbox_cfg.get("backend") or "local").strip().lower()
    os.environ["OPENAGENT_SANDBOX_BACKEND"] = _sandbox_backend
    if _sandbox_backend == "docker":
        # Only serialise the docker sub-config when docker is actually selected —
        # a misconfigured opt-in must fail closed at spawn, not degrade to host.
        os.environ["OPENAGENT_SANDBOX_DOCKER"] = json.dumps(
            _sandbox_cfg.get("docker") or {}
        )
    elif _sandbox_backend == "ssh":
        # Same rule for the ssh backend: the remote host IS the sandbox, so the
        # ssh sub-config (host/user/port/key_path) is only serialised when ssh is
        # actually selected. An unreachable host fails closed in prepare().
        os.environ["OPENAGENT_SANDBOX_SSH"] = json.dumps(
            _sandbox_cfg.get("ssh") or {}
        )

    # ``tool_output`` — opt-in LOSSLESS offload of oversized tool results. Off by
    # default: with offload disabled ``cap_tool_output`` truncates a too-large
    # result byte-identically to before. Exported as env vars for the same reason
    # as the sandbox/safety blocks above — the reader lives in-process in
    # ``src/core/tool_output.py`` and reads its policy from the environment at
    # call time (parity with ``OPENAGENT_MAX_TOOL_RESULT_CHARS``), so a
    # config-only knob would never reach it. Only ENABLED serialises the optional
    # threshold/dir overrides; an unset stanza exports just a "0" flag and
    # changes nothing downstream.
    from src.core.config import tool_output_settings

    _tool_output_cfg = tool_output_settings(config)
    os.environ["OPENAGENT_TOOL_OFFLOAD_ENABLED"] = (
        "1" if _tool_output_cfg.offload_enabled else "0"
    )
    if _tool_output_cfg.offload_enabled:
        if _tool_output_cfg.offload_threshold is not None:
            os.environ["OPENAGENT_TOOL_OFFLOAD_THRESHOLD"] = str(
                _tool_output_cfg.offload_threshold
            )
        if _tool_output_cfg.offload_dir:
            os.environ["OPENAGENT_TOOL_OFFLOAD_DIR"] = _tool_output_cfg.offload_dir
        os.environ["OPENAGENT_TOOL_OFFLOAD_KEEP"] = str(_tool_output_cfg.offload_keep)

    # ``mcps.install_policy`` — gates REGISTERING an MCP, which is the act that
    # hands a third party's argv this agent's whole environment. Exported as
    # env vars rather than plumbed, because the ``mcp-manager`` MCP runs as a
    # subprocess and can only see the policy through what it inherits — the
    # same constraint ``scheduler.timezone`` above has.
    #
    # Default OFF: vision §6 makes runtime registration a documented capability
    # ("Users can register custom MCPs at any time, by command, URL, or
    # marketplace pick"), so arming this by default would break a shipped
    # feature on someone's nightly cron. Enforced by
    # ``src.mcp.install_policy.check_mcp_install_allowed`` at the marketplace
    # and mcp-manager callsites — read that module's header for what it covers
    # and, importantly, what it does not (``POST /api/mcps`` and the pool spawn
    # are not yet wired).
    _mcps_cfg = config.get("mcps") or {}
    _install_cfg = (_mcps_cfg.get("install_policy") or {})
    if "enabled" in _install_cfg:
        os.environ["OPENAGENT_MCP_INSTALL_POLICY"] = (
            "1" if bool(_install_cfg["enabled"]) else "0"
        )
    # With the policy on and no ``allow_patterns``, the capability set is
    # frozen — no new MCPs at runtime, which is the right posture for an
    # unattended agent and needs no per-package inventory. These carve the
    # exceptions for agents that legitimately install, for the same reason
    # ``safety.approvals.allow_patterns`` exists: with no exception mechanism
    # the operator chooses between "off entirely" and "break my agent".
    if _install_cfg.get("allow_patterns"):
        _ip_allows = _install_cfg["allow_patterns"]
        if isinstance(_ip_allows, (list, tuple)):
            os.environ["OPENAGENT_MCP_INSTALL_ALLOW_PATTERNS"] = ",".join(
                str(p) for p in _ip_allows
            )
        elif isinstance(_ip_allows, str):
            os.environ["OPENAGENT_MCP_INSTALL_ALLOW_PATTERNS"] = _ip_allows

    # ``network.peers`` — who may dial the ``openagent/agent/1`` ALPN, and what
    # they may reach once they have. Both default OFF and both are enforced in
    # ``src.network.auth.peer_policy`` at the single agent-ALPN branch of the
    # auth middleware; read that module's header for why the ALPN opened far
    # more than federation ever needed.
    #
    # Default OFF is not timidity, it is the requirement: these agents run
    # unattended, and an allowlist that armed itself on upgrade would cut a
    # live mesh dead at 3am with no human attached to notice.
    _peers_cfg = ((config.get("network") or {}).get("peers") or {})
    _allowlist_cfg = (_peers_cfg.get("allowlist") or {})
    if "enabled" in _allowlist_cfg:
        os.environ["OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED"] = (
            "1" if bool(_allowlist_cfg["enabled"]) else "0"
        )
    if _allowlist_cfg.get("node_ids"):
        _nodes = _allowlist_cfg["node_ids"]
        if isinstance(_nodes, (list, tuple)):
            os.environ["OPENAGENT_NETWORK_PEER_ALLOWLIST"] = ",".join(
                str(n) for n in _nodes
            )
        elif isinstance(_nodes, str):
            os.environ["OPENAGENT_NETWORK_PEER_ALLOWLIST"] = _nodes
    _scope_cfg = (_peers_cfg.get("scope") or {})
    if "enabled" in _scope_cfg:
        os.environ["OPENAGENT_NETWORK_PEER_SCOPE_ENABLED"] = (
            "1" if bool(_scope_cfg["enabled"]) else "0"
        )
    # ``extra_paths`` is the escape hatch that keeps ``scope`` from being a
    # release-blocker: a federation feature this build's built-in route list
    # doesn't know about is one config line, not a wait for a new version.
    if _scope_cfg.get("extra_paths"):
        _paths = _scope_cfg["extra_paths"]
        if isinstance(_paths, (list, tuple)):
            os.environ["OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS"] = ",".join(
                str(p) for p in _paths
            )
        elif isinstance(_paths, str):
            os.environ["OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS"] = _paths

    memory_cfg = config.get("memory", {})
    # Learning toggles — mapped to env vars so the loops in ``src.learning``
    # can read them without plumbing the agent config through every callsite.
    #
    # ``memory.user_profile`` / ``memory.skills`` are no longer read: both
    # subsystems were deleted in v0.15.11 (opaque parallel memory stores
    # competing with the vault, with Groq-hardcoded writers that had zero
    # callers — see ``learning/__init__.py``). A stale block for either in an
    # existing ``openagent.yaml`` is inert rather than an error, which is the
    # same way every other retired key degrades here.
    #
    # ``memory.learning_model`` is no longer read, and ``learning/_model.py``
    # is gone with it. It named the model for one caller — the vault-
    # maintenance loop's AI-suggestion step — and that loop was deleted in
    # v0.16.1 when dream mode was consolidated onto the scheduled task (see the
    # ``memory.vault.maintenance`` note below). The step asked a cheap model to
    # write prose advice ("this orphan should link to X") into a log note that
    # nothing then acted on; the scheduled task reads the SAME
    # ``open_suggestions`` from ``vault_dream()`` while holding
    # ``write_note``/``patch_note``/``delete_note``, so it fixes what the loop
    # could only describe. Keeping the key to feed a deleted advisor would have
    # left exactly the write-only export the five ``OPENAGENT_SAFETY_*`` vars
    # were: set from yaml, read by nobody, greps like a live feature. The
    # reasoning for why such a model must be named explicitly rather than
    # inferred from ``is_classifier``/``tier_hint`` is not lost — it is the
    # same argument, in the module this one was modelled on:
    # ``src/core/compaction.py``'s ``_SUMMARY_MODEL_ENV``. A stale
    # ``learning_model`` key in an existing ``openagent.yaml`` is inert rather
    # than an error, the same way every other retired key degrades here.
    # ``memory.semantic_search`` (the OLD key) is still not read: the subsystem
    # it gated was an OpenAI-pinned embedding index whose only writer had zero
    # callers, so it could never return a row on any deployment. What replaces
    # it is NOT that — it is a REBUILDABLE CACHE done right (§5/§17):
    # ``src/memory/semantic_index.py`` embeds the vault + sessions through ANY
    # OpenAI-compatible endpoint the operator NAMES (local Ollama by default,
    # $0 and offline), degrades to nothing when unset, and is a cache beside the
    # FTS caches, not a hidden store. These keys are all READ by that module and
    # by the auto-recall hook (``core/agent.py``), so none is write-only.
    #
    # NOTE: these are exported into the PARENT ``os.environ`` and so reach the
    # IN-PROCESS auto-recall hook. The memory-search MCP *subprocess* does not
    # inherit them (the SDK spawns it with only get_default_environment() +
    # PYTHONPATH); forwarding them to that spec is a one-line change in the
    # pool/builtins layer, owned elsewhere — see semantic_recall's docstring.
    if memory_cfg.get("embedding_model"):
        os.environ["OPENAGENT_EMBEDDING_MODEL"] = str(memory_cfg["embedding_model"]).strip()
    if memory_cfg.get("embedding_base_url"):
        os.environ["OPENAGENT_EMBEDDING_BASE_URL"] = str(memory_cfg["embedding_base_url"]).strip()
    if memory_cfg.get("embedding_api_key"):
        os.environ["OPENAGENT_EMBEDDING_API_KEY"] = str(memory_cfg["embedding_api_key"]).strip()
    _ar_cfg = (memory_cfg.get("auto_recall") or {})
    if "enabled" in _ar_cfg:
        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1" if bool(_ar_cfg["enabled"]) else "0"
    # Hybrid FTS∪semantic recall (default ON in code). Only export when the
    # operator sets it explicitly, so the default lives in one place.
    if "hybrid" in _ar_cfg:
        os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "1" if bool(_ar_cfg["hybrid"]) else "0"
    for _k, _env in (
        ("min_score",   "OPENAGENT_AUTO_RECALL_MIN_SCORE"),
        ("top_k",       "OPENAGENT_AUTO_RECALL_TOP_K"),
        ("fts_top_k",   "OPENAGENT_AUTO_RECALL_FTS_TOP_K"),
        ("fts_extra",   "OPENAGENT_AUTO_RECALL_FTS_EXTRA"),
        ("max_tokens",  "OPENAGENT_AUTO_RECALL_MAX_TOKENS"),
        ("warm_budget", "OPENAGENT_AUTO_RECALL_WARM_BUDGET"),
        ("timeout",     "OPENAGENT_AUTO_RECALL_TIMEOUT"),
    ):
        if _k in _ar_cfg:
            try:
                os.environ[_env] = str(_ar_cfg[_k])
            except (TypeError, ValueError):
                pass
    # Per-origin recall CORPUS scoping (default = identity: no filtering). Maps
    # ``scope`` / ``include_path_prefixes`` / ``exclude_path_prefixes`` /
    # ``reserve_prefix`` → the ``OPENAGENT_AUTO_RECALL_{SCOPE,INCLUDE_PATHS,
    # EXCLUDE_PATHS,RESERVE_PREFIX}`` env, and per-origin overrides under
    # ``by_origin.<origin>`` → the same names with an ``_<ORIGIN>`` suffix (e.g.
    # ``..._EVENT``), which ``agent._recall_scoping`` prefers for that origin. A
    # shared support+dev-ops agent uses this to drop dev-ops notes on its
    # ``event`` (support) turns only.
    def _export_recall_scoping(cfg: dict, suffix: str = "") -> None:
        def _csv(v: Any) -> str:
            if isinstance(v, (list, tuple)):
                return ",".join(str(x).strip() for x in v if str(x).strip())
            return str(v).strip()
        for _key, _base in (
            ("scope",                 "OPENAGENT_AUTO_RECALL_SCOPE"),
            ("include_path_prefixes", "OPENAGENT_AUTO_RECALL_INCLUDE_PATHS"),
            ("exclude_path_prefixes", "OPENAGENT_AUTO_RECALL_EXCLUDE_PATHS"),
            ("reserve_prefix",        "OPENAGENT_AUTO_RECALL_RESERVE_PREFIX"),
        ):
            if _key in cfg:
                _val = _csv(cfg[_key])
                if _val:
                    os.environ[_base + suffix] = _val
    _export_recall_scoping(_ar_cfg)
    for _origin, _ocfg in (_ar_cfg.get("by_origin") or {}).items():
        if isinstance(_ocfg, dict) and str(_origin).strip():
            _export_recall_scoping(_ocfg, "_" + str(_origin).strip().upper())
    # Quality monitor (opt-in): an LLM-as-judge grades a SAMPLED fraction of
    # completed turns for correctness, logged as ``quality.score`` beside the
    # ``router.cost_recorded`` spend events — usage AND quality in one place.
    # Maps ``quality_monitor.*`` (top-level, or under ``memory:``) to the env
    # vars ``src/core/quality_monitor.py`` reads. OFF unless enabled (§17).
    _qm_cfg = (config.get("quality_monitor") or memory_cfg.get("quality_monitor") or {})
    if "enabled" in _qm_cfg:
        os.environ["OPENAGENT_QUALITY_MONITOR_ENABLED"] = (
            "1" if bool(_qm_cfg["enabled"]) else "0"
        )
    for _k, _env in (
        ("sample_rate", "OPENAGENT_QUALITY_MONITOR_SAMPLE_RATE"),
        ("model",       "OPENAGENT_QUALITY_MONITOR_MODEL"),
        ("timeout",     "OPENAGENT_QUALITY_MONITOR_TIMEOUT"),
        ("min_len",     "OPENAGENT_QUALITY_MONITOR_MIN_LEN"),
    ):
        if _k in _qm_cfg:
            try:
                os.environ[_env] = str(_qm_cfg[_k])
            except (TypeError, ValueError):
                pass
    # ``quality_monitor.digest.*`` → the scheduled digest/alerting loop
    # (``src/core/quality_digest.py``). Defaults ON when the monitor is on.
    _qd_cfg = (_qm_cfg.get("digest") or {})
    if "enabled" in _qd_cfg:
        os.environ["OPENAGENT_QUALITY_DIGEST_ENABLED"] = (
            "1" if bool(_qd_cfg["enabled"]) else "0"
        )
    for _k, _env in (
        ("interval_hours",       "OPENAGENT_QUALITY_DIGEST_INTERVAL_HOURS"),
        ("min_score_alert",      "OPENAGENT_QUALITY_DIGEST_MIN_SCORE_ALERT"),
        ("recall_timeout_alert", "OPENAGENT_QUALITY_DIGEST_RECALL_TIMEOUT_ALERT"),
        ("embed_error_alert",    "OPENAGENT_QUALITY_DIGEST_EMBED_ERROR_ALERT"),
    ):
        if _k in _qd_cfg:
            try:
                os.environ[_env] = str(_qd_cfg[_k])
            except (TypeError, ValueError):
                pass
    # Anti-fabrication reply guard (``src/core/reply_guard.py``). OFF by default.
    # Maps ``reply_guard.*`` (top-level, or under ``memory:``) to the env vars
    # the guard reads: a boolean gate and an optional backing-tool substring
    # list. Needs the quality monitor on for tool-trace grounding visibility.
    _rg_cfg = (config.get("reply_guard") or memory_cfg.get("reply_guard") or {})
    if "enabled" in _rg_cfg:
        os.environ["OPENAGENT_REPLY_GUARD_ENABLED"] = (
            "1" if bool(_rg_cfg["enabled"]) else "0"
        )
    _rg_backing = _rg_cfg.get("backing_tools")
    if _rg_backing:
        try:
            if isinstance(_rg_backing, (list, tuple)):
                os.environ["OPENAGENT_REPLY_GUARD_BACKING_TOOLS"] = ",".join(
                    str(x) for x in _rg_backing
                )
            else:
                os.environ["OPENAGENT_REPLY_GUARD_BACKING_TOOLS"] = str(_rg_backing)
        except (TypeError, ValueError):
            pass
    # Hard ("strict") budget scopes (``src/core/budget_guard.py``). A scope
    # listed here has its cap enforced even when that leaves ZERO enabled models
    # — the agent goes OFFLINE rather than overspend (overrides the default
    # never-empty-catalog safety for the named scopes only). Maps
    # ``budget.strict_scopes`` (a list like ``["provider:deepseek"]``, or a
    # comma string) to ``OPENAGENT_BUDGET_STRICT_SCOPES``. Absent → no strict
    # scopes (default safety holds).
    _bg_cfg = (config.get("budget") or memory_cfg.get("budget") or {})
    _strict = _bg_cfg.get("strict_scopes")
    if _strict:
        try:
            if isinstance(_strict, (list, tuple)):
                os.environ["OPENAGENT_BUDGET_STRICT_SCOPES"] = ",".join(
                    str(x) for x in _strict
                )
            else:
                os.environ["OPENAGENT_BUDGET_STRICT_SCOPES"] = str(_strict)
        except (TypeError, ValueError):
            pass
    # Per-run cost-anomaly alerting (``src/core/cost_anomaly.py``). Defaults ON
    # with safe thresholds (a mostly-cached few-cent run can't page); maps
    # ``cost_anomaly.*`` (top-level, or under ``memory:``) to the env vars that
    # module reads. An operator disables it or retunes thresholds / the alert
    # webhook here without touching env.
    _ca_cfg = (config.get("cost_anomaly") or memory_cfg.get("cost_anomaly") or {})
    if "enabled" in _ca_cfg:
        os.environ["OPENAGENT_COST_ANOMALY_ENABLED"] = (
            "1" if bool(_ca_cfg["enabled"]) else "0"
        )
    for _k, _env in (
        ("cost_usd",              "OPENAGENT_COST_ANOMALY_COST_USD"),
        ("uncached_input_tokens", "OPENAGENT_COST_ANOMALY_UNCACHED_INPUT_TOKENS"),
        ("alert_webhook_url",     "OPENAGENT_COST_ANOMALY_ALERT_WEBHOOK_URL"),
    ):
        if _k in _ca_cfg:
            try:
                os.environ[_env] = str(_ca_cfg[_k])
            except (TypeError, ValueError):
                pass
    _cur_cfg = (memory_cfg.get("curator") or {})
    if "enabled" in _cur_cfg:
        os.environ["OPENAGENT_CURATOR_ENABLED"] = (
            "1" if bool(_cur_cfg["enabled"]) else "0"
        )
    _vr_cfg = (memory_cfg.get("vault_reminder") or {})
    if "enabled" in _vr_cfg:
        os.environ["OPENAGENT_VAULT_REMINDER_ENABLED"] = (
            "1" if bool(_vr_cfg["enabled"]) else "0"
        )
    if "every_n_turns" in _vr_cfg:
        try:
            os.environ["OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS"] = str(
                int(_vr_cfg["every_n_turns"])
            )
        except (TypeError, ValueError):
            pass

    # Vault quality subsystem (code-enforced gate, incremental index,
    # derived artifacts, dream-mode maintenance). See ``src.memory.vault``.
    # Export the resolved vault path so in-process consumers (the gate
    # service, the native vault-gate MCP) target the SAME folder the agent
    # and gateway use — they read OPENAGENT_VAULT_PATH first.
    if memory_cfg.get("vault_path"):
        try:
            from pathlib import Path as _P
            os.environ["OPENAGENT_VAULT_PATH"] = str(
                _P(str(memory_cfg["vault_path"])).expanduser().resolve()
            )
        except Exception:  # noqa: BLE001
            pass
    _vault_cfg = (memory_cfg.get("vault") or {})
    for _k, _env in (
        ("enabled",           "OPENAGENT_VAULT_ENABLED"),
        ("enforce_taxonomy",  "OPENAGENT_VAULT_ENFORCE_TAXONOMY"),
        ("check_em_dash",     "OPENAGENT_VAULT_CHECK_EM_DASH"),
        ("strict",            "OPENAGENT_VAULT_STRICT"),
        ("validate_on_write", "OPENAGENT_VAULT_VALIDATE_ON_WRITE"),
    ):
        if _k in _vault_cfg:
            os.environ[_env] = "1" if bool(_vault_cfg[_k]) else "0"
    for _k, _env in (
        ("max_lines",    "OPENAGENT_VAULT_MAX_LINES"),
        ("min_outlinks", "OPENAGENT_VAULT_MIN_OUTLINKS"),
    ):
        if _k in _vault_cfg:
            try:
                os.environ[_env] = str(int(_vault_cfg[_k]))
            except (TypeError, ValueError):
                pass
    # ``memory.vault.maintenance.*`` (``enabled``, ``interval_hours``,
    # ``autofix``, ``regenerate_derived``) is no longer read. It configured a
    # SECOND dream mode: a 12-hourly asyncio loop in
    # ``learning/vault_maintenance.py`` that ran the mechanical pass and wrote
    # its own dream-log, in parallel with — and unaware of — the ``dream_mode``
    # scheduled task. Both were off by default, both could be on at once, and
    # they logged to two paths in two formats.
    #
    # Vision §12 describes ONE thing: "The agent runs a scheduled 'dream' task
    # ... nightly by default, at a time the user can adjust. It does not
    # compete with user-facing work." A scheduled task is that. A hidden
    # interval loop is not: 12h from boot lands mid-conversation half the time,
    # and no ``time:``/``timezone:`` could move it.
    #
    # The loop survived this long because deleting it would have dropped the
    # mechanical pass entirely — ``DREAM_MODE_PROMPT`` named ``vault_dream``,
    # ``vault_gate``, ``vault_doctor`` and ``vault_regenerate_derived`` zero
    # times each, so the loop was the only caller of ``VaultService.maintenance``
    # on a live deployment. ``89c7379`` fixed that: Mission 1 now opens with
    # ``vault_dream()``, which is the same ``svc.maintenance(apply_fixes=True,
    # regenerate=True)`` call the loop made. With both halves in the task, the
    # loop was pure duplication and went in v0.16.1.
    #
    # The knobs died with their reader rather than being re-homed: the task is
    # an agent holding finer-grained tools, so ``autofix: false`` is
    # ``vault_doctor(apply=False)`` and ``regenerate_derived: false`` is simply
    # not calling ``vault_regenerate_derived()`` — a boolean cannot express
    # "preview this one, apply that one", and the agent can. ``interval_hours``
    # is answered by ``dream_mode.time`` + ``dream_mode.timezone``, which §12
    # asks for by name and the loop could never honour.
    #
    # A stale ``maintenance`` block in an existing ``openagent.yaml`` is inert
    # rather than an error, the same way every other retired key degrades here.
    # Nothing is exported: an env var set from yaml and read by nobody is how
    # five ``OPENAGENT_SAFETY_*`` vars sat here for months describing
    # protection that never fired (``t_no_write_only_safety_env``).
    # Git-backed vault: every change is auto-committed with provenance.
    _vgit_cfg = (_vault_cfg.get("git") or {})
    if "enabled" in _vgit_cfg:
        os.environ["OPENAGENT_VAULT_GIT_ENABLED"] = (
            "1" if bool(_vgit_cfg["enabled"]) else "0")
    if "autocommit_seconds" in _vgit_cfg:
        try:
            os.environ["OPENAGENT_VAULT_GIT_AUTOCOMMIT_SECONDS"] = str(
                int(_vgit_cfg["autocommit_seconds"]))
        except (TypeError, ValueError):
            pass
    for _k, _env in (("author_name", "OPENAGENT_VAULT_GIT_NAME"),
                     ("author_email", "OPENAGENT_VAULT_GIT_EMAIL")):
        if _vgit_cfg.get(_k):
            os.environ[_env] = str(_vgit_cfg[_k])

    # Extended thinking budget. Surfaces as ``model.extended_thinking_tokens``
    # in yaml so it stays in the same logical namespace as future
    # model-runtime knobs. Anthropic requires the budget be at least
    # 1024 to engage the feature; anything below disables it.
    _model_cfg = config.get("model") or {}
    if "extended_thinking_tokens" in _model_cfg:
        try:
            os.environ["OPENAGENT_EXTENDED_THINKING_TOKENS"] = str(
                int(_model_cfg["extended_thinking_tokens"])
            )
        except (TypeError, ValueError):
            pass

    # Per-LLM-call read timeout (anti-wedge). Surfaces as
    # ``model.timeout_seconds`` in yaml — an operator override of the 90s
    # default that ``NativeProvider._construct_model`` applies to every model
    # build. It MUST stay under the 120s event-lease TTL so a hung socket
    # raises before the worker wedges; a value >= 120 defeats the purpose.
    if "timeout_seconds" in _model_cfg:
        try:
            os.environ["OPENAGENT_MODEL_TIMEOUT_SECONDS"] = str(
                float(_model_cfg["timeout_seconds"])
            )
        except (TypeError, ValueError):
            pass

    # Quick commands + hooks — keep them in a process-local registry
    # (see ``src.core.hooks``) since bridges + the agent runtime share
    # the same process. The registry is replaced (not merged) on every
    # ``create_agent`` so a config reload doesn't leak removed entries.
    try:
        from src.core.hooks import set_quick_commands, set_hooks
        set_quick_commands(config.get("quick_commands") or {})
        set_hooks(config.get("hooks") or {})
    except Exception as e:  # noqa: BLE001
        logger.warning("hooks/quick_commands registry init failed: %s", e)
    for yaml_key, env_key in (
        ("interval_hours",         "OPENAGENT_CURATOR_INTERVAL_HOURS"),
        ("skill_stale_days",       "OPENAGENT_CURATOR_SKILL_STALE_DAYS"),
        ("skill_archive_days",     "OPENAGENT_CURATOR_SKILL_ARCHIVE_DAYS"),
        ("profile_archive_days",   "OPENAGENT_CURATOR_PROFILE_ARCHIVE_DAYS"),
        ("session_retention_days", "OPENAGENT_CURATOR_SESSION_RETENTION_DAYS"),
        ("backup_interval_hours",  "OPENAGENT_CURATOR_BACKUP_INTERVAL_HOURS"),
        ("backup_keep",            "OPENAGENT_CURATOR_BACKUP_KEEP"),
    ):
        if yaml_key in _cur_cfg:
            try:
                os.environ[env_key] = str(int(_cur_cfg[yaml_key]))
            except (TypeError, ValueError):
                pass
    # Convenience: ``memory.sessions.retention_days`` is a more
    # discoverable alias for the same knob — wire it through too so
    # operators don't have to remember it lives under curator.
    _sess_cfg = (memory_cfg.get("sessions") or {})
    if "retention_days" in _sess_cfg:
        try:
            os.environ["OPENAGENT_CURATOR_SESSION_RETENTION_DAYS"] = str(
                int(_sess_cfg["retention_days"])
            )
        except (TypeError, ValueError):
            pass
    db_path = memory_cfg.get("db_path", str(default_db_path()))
    db = MemoryDB(db_path)

    # MCP pool is built *inside* ``Agent.initialize`` from the ``mcps``
    # DB table — the yaml never carried MCP state. The Agent starts with
    # an empty pool; ``wire_model_runtime`` re-runs in ``initialize`` once
    # the pool is online so providers see the full toolkit list.
    wire_model_runtime(model, db=db)

    # Model fallback — when the primary provider (e.g. Claude via the local
    # proxy) rate-limits, times out, or errors, degrade the turn to a
    # secondary provider instead of failing the whole run. Configured under
    # ``fallback:`` in the agent yaml, e.g.
    #   fallback:
    #     on_rate_limit: ["deepseek:deepseek-v4-pro"]
    #     on_error:      ["deepseek:deepseek-v4-pro"]
    # Strings resolve to Model instances in ``Agent.initialize`` via
    # ``FallbackConfig.resolve_models()`` against the DB provider catalog; an
    # unresolvable entry is skipped, so a bad config degrades to "no fallback"
    # (today's behaviour) rather than crashing the agent.
    fallback_raw = config.get("fallback") or {}
    fallback_config = None
    if fallback_raw:
        from src.models.providers.fallback import FallbackConfig
        fallback_config = FallbackConfig(
            on_error=list(fallback_raw.get("on_error") or []),
            on_rate_limit=list(fallback_raw.get("on_rate_limit") or []),
            on_context_overflow=list(fallback_raw.get("on_context_overflow") or []),
        )

    # ── budgets: per-scope spend caps (off by default) ──
    # Hand the yaml ``budgets:`` list to the dispatcher's BudgetGuard, which
    # seeds the rules additively (only-if-absent, so an app edit is never
    # clobbered on reboot) and enforces them by excluding an over-cap scope from
    # routing. With no ``budgets:`` block this hands over nothing and the guard
    # stays inert — behaviour is byte-identical to a build without it. Rules are
    # DB-backed and editable at runtime via ``/api/budgets`` + the
    # ``budget-manager`` MCP; the yaml is just the boot-time floor, exactly like
    # ``DEFAULT_MCPS``. Unlike the safety/scheduler stanzas this seeds a DB
    # table rather than exporting an env var, because a spend cap must be
    # editable from the app without a redeploy.
    _set_budget_seed = getattr(model, "set_budget_seed", None)
    if callable(_set_budget_seed):
        budgets_cfg = config.get("budgets")
        _set_budget_seed(budgets_cfg if isinstance(budgets_cfg, list) else None)

    return Agent(
        name=config.get("name", "openagent"),
        model=model,
        system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        mcp_pool=None,
        memory=db,
        config=config,  # channels / memory / name only — providers/models/mcps live in the DB
        fallback_config=fallback_config,
    )


def _channel_live_default() -> bool:
    """Default for live-message mode when a channel config omits ``live``.

    Reads ``OPENAGENT_CHANNEL_LIVE`` so an operator can flip the
    fleet-wide default without editing every ``openagent.yaml``. Defaults
    to ON — the Hermes-style "narrate each tool call + answer span as its
    own message" behaviour is the intended out-of-the-box experience.
    """
    return os.environ.get("OPENAGENT_CHANNEL_LIVE", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _build_bridges(config: dict, per_bridge_url: dict[str, str]) -> list:
    """Build platform bridges from config. Each connects to the Gateway via WS.

    With the iroh transport the gateway no longer listens on a fixed
    localhost port; each entry in ``per_bridge_url`` points at the
    ``LoopbackProxy`` started by THIS bridge's ``BridgeSession`` (see
    ``openagent.network.bridge_session``). One LoopbackProxy per bridge
    is required so each bridge has its own gateway client_id; sharing
    one URL across bridges produced the v0.12.49 friday outage.
    """
    channels_config = config.get("channels") or {}
    out = []

    for name, cfg in channels_config.items():
        if name == "websocket":
            continue  # legacy, ignored — gateway is now Iroh-bound
        if name == "webhook":
            # Not a WS-client bridge: the webhook is an INBOUND HTTP listener
            # owned by the Gateway (see Gateway._maybe_start_webhook_listener).
            # Skip it here so it never looks for a per-bridge loopback URL.
            continue

        gateway_url = per_bridge_url.get(name)
        if gateway_url is None:
            # The session for this bridge failed to start (logged
            # above) or it's a bridge name we don't recognise —
            # either way we can't wire it up.
            continue

        # Per-channel personality overlay (Hermes-style). Resolved later
        # inside BaseBridge — here we just plumb the raw string through.
        personality = cfg.get("personality")

        # Live-message mode: post each tool call + narration span as its
        # own chat message while the turn runs (Hermes-style), alongside
        # the "is writing" indicator. On by default; opt out per channel
        # with ``channels.<name>.live: false`` or globally with the
        # ``OPENAGENT_CHANNEL_LIVE`` env var.
        live = bool(cfg.get("live", _channel_live_default()))

        if name == "telegram":
            from src.bridges.telegram import TelegramBridge
            token = cfg.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")
            if not token:
                logger.warning("Telegram token not configured; skipping")
                continue
            out.append(TelegramBridge(
                token=token,
                allowed_users=cfg.get("allowed_users"),
                gateway_url=gateway_url,
                gateway_token=None,
                personality=personality,
                streaming=bool(cfg.get("streaming", False)),
                allowed_chats=cfg.get("allowed_chats"),
                live=live,
            ))

        elif name == "discord":
            from src.bridges.discord import DiscordBridge
            token = cfg.get("token") or os.environ.get("DISCORD_BOT_TOKEN")
            if not token:
                logger.warning("Discord token not configured; skipping")
                continue
            allowed = cfg.get("allowed_users")
            if not allowed:
                logger.warning("Discord needs allowed_users; skipping")
                continue
            out.append(DiscordBridge(
                token=token,
                allowed_users=allowed,
                allowed_guilds=cfg.get("allowed_guilds"),
                listen_channels=cfg.get("listen_channels"),
                dm_only=bool(cfg.get("dm_only", False)),
                gateway_url=gateway_url,
                gateway_token=None,
                personality=personality,
                live=live,
            ))

        elif name == "whatsapp":
            from src.bridges.whatsapp import WhatsAppBridge
            iid = cfg.get("green_api_id") or os.environ.get("GREEN_API_ID")
            tok = cfg.get("green_api_token") or os.environ.get("GREEN_API_TOKEN")
            if not iid or not tok:
                logger.warning("WhatsApp credentials not configured; skipping")
                continue
            out.append(WhatsAppBridge(
                instance_id=iid,
                api_token=tok,
                allowed_users=cfg.get("allowed_users"),
                gateway_url=gateway_url,
                gateway_token=None,
                personality=personality,
                live=live,
            ))

        elif name == "slack":
            from src.bridges.slack import SlackBridge
            bot_token = cfg.get("bot_token") or os.environ.get("SLACK_BOT_TOKEN")
            app_token = cfg.get("app_token") or os.environ.get("SLACK_APP_TOKEN")
            if not bot_token or not app_token:
                logger.warning("Slack tokens not configured; skipping")
                continue
            out.append(SlackBridge(
                bot_token=bot_token,
                app_token=app_token,
                allowed_users=cfg.get("allowed_users"),
                listen_channels=cfg.get("listen_channels"),
                gateway_url=gateway_url,
                gateway_token=None,
                personality=personality,
                live=live,
            ))

        else:
            logger.warning(f"Unknown channel: {name}")

    return out


class AgentServer:
    """Owns the lifecycle of agent, gateway, bridges, and scheduler.

    Usage:
        server = AgentServer.from_config(config)
        async with server:
            await server.wait()
    """

    def __init__(
        self,
        agent: Agent,
        config: dict,
    ) -> None:
        self.agent = agent
        self.config = config

        self._bridge_tasks: list[asyncio.Task] = []
        self._bridges: list = []
        # Curator task — periodic prune of dormant sessions + DB backups.
        # Created in ``start()`` only when the feature is enabled; ``None``
        # otherwise so ``stop()`` can short-circuit cleanly.
        self._curator_task: asyncio.Task | None = None
        # One BridgeSession per bridge — see ``_build_bridge_session_and_bridges``.
        # Pre-v0.12.50 a single session was shared across all bridges, which
        # let two bridges collide on the gateway's client_id (handle="__bridge")
        # and kick each other's WS off. Each bridge now gets its own cert +
        # client_id under handle="__bridge_<name>".
        self._bridge_sessions: list = []
        self._scheduler = None
        self._gateway = None
        self._stop_event: asyncio.Event | None = None

    @classmethod
    def from_config(cls, config: dict, only_channels: list[str] | None = None) -> AgentServer:
        agent = _build_agent(config)
        server = cls(agent=agent, config=config)
        memory_cfg = config.get("memory", {}) or {}
        server._gateway_vault_path = memory_cfg.get("vault_path")
        server._gateway_config_path = config.get("_config_path")
        server._network_state = None
        server._only_channels = only_channels
        # Bridges are constructed in ``start`` after the gateway + bridge
        # session are up — they need ``gateway_url`` to point at the
        # bridge session's LoopbackProxy, which doesn't exist yet.
        server._bridges = []
        return server

    async def _build_network_state(self):
        """Read the singleton ``network`` row and build a ``NetworkState``.

        Returns ``None`` for standalone agents — the caller skips the
        gateway and prints a helpful message. Any other failure
        propagates so a misconfigured network row surfaces loudly
        rather than silently disabling the public interface.
        """
        from src.network.state import NetworkState, StandaloneAgentError
        from src.core.paths import get_agent_dir

        agent_dir = get_agent_dir()
        if agent_dir is None:
            logger.warning(
                "no agent dir set; running without a gateway. "
                "Pass --agent-dir to ``openagent`` to enable network mode.",
            )
            return None

        net_cfg = self.config.get("network") or {}
        identity_path = agent_dir / (net_cfg.get("identity_path") or "identity.key")
        derp_url = net_cfg.get("derp_url") or None
        try:
            return await NetworkState.from_db(
                db=self.agent._db,
                identity_path=identity_path,
                derp_url=derp_url,
            )
        except StandaloneAgentError:
            logger.warning(
                "this agent has no network configured. Run "
                "`openagent network init` to create one — or join an "
                "existing network. The gateway will not be exposed until then.",
            )
            return None

    async def _publish_coordinator_addr_cache(self) -> None:
        """Snapshot the iroh node's reachable addresses so the
        ``openagent network invite`` CLI can embed them in tickets.

        Members (non-coordinators) skip this — their tickets are minted
        by the coordinator they joined, not by themselves. Quiet on
        failure: the worst case is missing optimisation, not a broken
        coordinator.
        """
        from src.core.paths import get_agent_dir
        from src.network.coordinator_addr_cache import write_cache

        if self._network_state is None or self._network_state.role != "coordinator":
            return
        agent_dir = get_agent_dir()
        if agent_dir is None:
            return
        try:
            relay_url, direct = await self._network_state.iroh_node.local_node_addr()
        except Exception as e:  # noqa: BLE001
            logger.debug("local_node_addr failed during cache publish: %s", e)
            return
        node_id = await self._network_state.node_id()
        write_cache(
            agent_dir,
            node_id=node_id,
            relay_url=relay_url,
            direct_addresses=direct,
        )

    # ── Lifecycle ──

    async def start(self) -> None:
        """Start agent, gateway, scheduler, and bridges."""
        self._stop_event = asyncio.Event()
        elog("server.start", agent=self.agent.name)

        # 0. Apply anyio cancel-scope guard BEFORE any MCP operations.
        #    anyio#695: _deliver_cancellation can spin forever when
        #    MCP SDK tasks don't respond to CancelledError.
        from src.network.transport.anyio_cancel_guard import _patch_deliver_cancellation
        _patch_deliver_cancellation()

        # 1. Agent (connects MCPs, opens DB)
        await self.agent.initialize()

        # 1.1. Budget guard: seed any yaml ``budgets:`` rules and prime the
        #      over-cap snapshot so the very first turn already routes around a
        #      capped scope (the operator's $100 DeepSeek brake must be armed at
        #      boot, not one turn late). Off by default: with no rules the guard
        #      finds nothing and changes nothing. Never fatal — a budget must
        #      never block the agent from coming up.
        try:
            _guard = getattr(getattr(self.agent, "model", None), "budget_guard", None)
            if _guard is not None:
                await _guard.warm()
        except Exception as e:  # noqa: BLE001
            elog("budget.warm_error", level="warning", error=str(e))

        # 1.2. Warm the OpenRouter pricing cache so ``compute_cost`` is accurate
        #      from the FIRST billed call, not the second. Without this the first
        #      DeepSeek call of each boot logs ``$0`` (cold cache) and a cost cap
        #      undercounts it — a per-boot blind spot in the brake. Awaited here
        #      (never fatal) so the price is hot before any turn or scheduled fire.
        try:
            from src.models.catalog import warm_pricing_cache
            await warm_pricing_cache()
        except Exception as e:  # noqa: BLE001 — pricing warm must never block boot
            elog("catalog.pricing_warm_error", level="warning", error=str(e))

        # 1.5. Reap any ``workflow_runs`` still in ``running`` state —
        #      they're zombies from the prior process that we have no
        #      way to resume. Without this, the UI shows them spinning
        #      forever and any per-workflow lock in the new executor
        #      would funnel a fresh scheduled run behind a stuck old
        #      one. Cheap (one UPDATE on a small table) and idempotent.
        try:
            db = getattr(self.agent, "_db", None)
            if db is not None and hasattr(db, "reap_orphan_workflow_runs"):
                reaped = await db.reap_orphan_workflow_runs()
                if reaped:
                    elog("workflow.orphan_reaped", count=reaped)
                # Same treatment for scheduled-task run history — a row
                # left ``running`` by a prior process is a zombie.
                if hasattr(db, "reap_orphan_task_runs"):
                    reaped_tasks = await db.reap_orphan_task_runs()
                    if reaped_tasks:
                        elog("task.orphan_reaped", count=reaped_tasks)
                # And webhook event deliveries left mid-flight by a crash.
                if hasattr(db, "reap_orphan_event_deliveries"):
                    reaped_ev = await db.reap_orphan_event_deliveries()
                    if reaped_ev:
                        elog("event.orphan_reaped", count=reaped_ev)
        except Exception as e:  # noqa: BLE001
            logger.warning("orphan workflow_run reap failed: %s", e)

        # 2. Build NetworkState now that the DB is open. iroh-py 0.35
        #    bakes the ALPN handler dict into NodeOptions at node
        #    construction time — every handler must be registered
        #    *before* NetworkState.start binds the iroh endpoint. So
        #    we (a) build NetworkState (constructor wires the
        #    coordinator handler if applicable), (b) build Gateway
        #    eagerly via a pre-start hook so IrohSite registers the
        #    gateway handler, (c) start NetworkState, (d) finish the
        #    Gateway lifecycle. Standalone agents skip the gateway.
        self._network_state = await self._build_network_state()
        if self._network_state is not None:
            from src.gateway.server import Gateway
            self._gateway = Gateway(
                agent=self.agent,
                network_state=self._network_state,
                vault_path=getattr(self, "_gateway_vault_path", None),
                config_path=getattr(self, "_gateway_config_path", None),
            )
            self._gateway._stop_event = self._stop_event
            self._gateway._bridges = self._bridges  # populated below
            self._gateway._prepare_iroh_site()
            await self._network_state.start()
            await self._publish_coordinator_addr_cache()
            await self._gateway.start()

            # 2.5. Bridge session — mints a coordinator-signed cert for
            #      handle ``__bridge`` and starts a LoopbackProxy that
            #      pumps localhost HTTP/WS bytes onto an authed iroh
            #      stream targeting our own NodeId. Gives in-process
            #      bridges a ``gateway_url`` that's wire-compatible
            #      with the legacy ``ws://localhost:8765/ws`` they
            #      were built against.
            await self._build_bridge_session_and_bridges()

        # 3. Scheduler (with dream mode + auto-update hooks)
        await self._start_scheduler()

        # 4. Bridges (connect to Gateway as internal WS clients)
        for bridge in self._bridges:
            self._bridge_tasks.append(asyncio.create_task(
                bridge.start(), name=f"bridge:{bridge.name}"
            ))

        # 5. Curator — periodic prune of dormant sessions + rolling DB backups.
        # No-op when ``memory.curator.enabled`` is false.
        # ``self.agent._db`` is the live MemoryDB (the Agent exposes no
        # ``.memory`` attribute — that was a long-standing typo that left
        # both loops below silently dead).
        try:
            from src.learning.curator import start as _curator_start
            self._curator_task = _curator_start(self.agent._db)
        except Exception as e:  # noqa: BLE001
            logger.warning("Curator failed to start: %s", e)
            self._curator_task = None

        # 6. (was: the vault-maintenance loop — a second, hidden dream mode.
        # Deleted in v0.16.1; the ``dream-mode`` scheduled task seeded by
        # ``_sync_dream_mode`` now runs the mechanical pass itself via
        # ``vault_dream()``. See the ``memory.vault.maintenance`` note in
        # ``_build_agent``.)

        # 7. Vault git — the memory vault is a git repo; commit every change
        # automatically (with provenance). The loop is the safety net for
        # edits made outside OpenAgent's own tools. No-op when git is absent
        # or ``memory.vault.git.enabled`` is false.
        try:
            from src.memory.vault.autocommit import start as _vault_autocommit_start
            self._vault_autocommit_task = _vault_autocommit_start()
        except Exception as e:  # noqa: BLE001
            logger.warning("Vault autocommit failed to start: %s", e)
            self._vault_autocommit_task = None

        # 8. Semantic index builder — build the recall index OFF the turn path.
        # The on-turn auto-recall hook is time-boxed and can only search + top up
        # a few vectors; the full build (2000+ notes) belongs in the background or
        # it silently times out and recall stays empty. No-op when no embedding
        # model is configured (resolve_embedder → None).
        try:
            from src.memory.semantic_index_builder import start as _sem_index_start
            _db = getattr(self.agent, "_db", None)
            _db_path = str(getattr(_db, "db_path", "") or "")
            try:
                _vault = self.agent._resolve_vault_path()
            except Exception:  # noqa: BLE001
                _vault = None
            self._semantic_index_task = _sem_index_start(
                _db_path, _vault, getattr(self.agent, "_providers_config", None))
        except Exception as e:  # noqa: BLE001
            logger.warning("Semantic index builder failed to start: %s", e)
            self._semantic_index_task = None

        # 9. Quality digest — a periodic aggregate of quality/recall/cost with
        # a flagged-sessions review list + threshold alerts (incl. embedder-down
        # via semantic.embed_error spikes). No-op when the quality monitor is off
        # (start() returns None); reads events.jsonl only, no agent/db needed.
        try:
            from src.core.quality_digest import start as _quality_digest_start
            self._quality_digest_task = _quality_digest_start()
        except Exception as e:  # noqa: BLE001
            logger.warning("Quality digest failed to start: %s", e)
            self._quality_digest_task = None

    async def _build_bridge_session_and_bridges(self) -> None:
        """Provision the in-process bridge sessions + concrete bridges.

        One BridgeSession per enabled bridge — sharing a session across
        bridges collides client_ids on the gateway side (see the class
        docstring on ``BridgeSession``).

        Failure to bring up an individual session (member-mode agents,
        missing coordinator key, etc.) skips THAT bridge but lets the
        others through. The gateway itself (remote clients over iroh)
        is unaffected by any bridge failure.
        """
        from src.core.paths import get_agent_dir
        from src.network.bridge_session import (
            BridgeSession,
            BridgeSessionUnavailable,
        )

        channels_config = self.config.get("channels") or {}
        enabled_bridges = [
            name for name in ("telegram", "discord", "whatsapp")
            if name in channels_config and channels_config[name]
        ]
        if not enabled_bridges:
            return

        agent_dir = get_agent_dir()
        if agent_dir is None:
            logger.warning(
                "no agent dir set; cannot persist bridge device keys — skipping bridges",
            )
            return

        gateway_site = getattr(self._gateway, "_site", None)
        per_bridge_url: dict[str, str] = {}
        for name in enabled_bridges:
            session = BridgeSession(bridge_name=name)
            try:
                await session.start(
                    network_state=self._network_state,
                    gateway_site=gateway_site,
                    agent_dir=agent_dir,
                )
            except BridgeSessionUnavailable as e:
                logger.warning(
                    "bridge %s unavailable: %s — skipping that bridge", name, e,
                )
                continue
            except Exception:
                logger.exception(
                    "bridge %s session failed to start — skipping that bridge", name,
                )
                continue
            self._bridge_sessions.append(session)
            per_bridge_url[name] = session.ws_url

        if not per_bridge_url:
            return

        self._bridges = _build_bridges(self.config, per_bridge_url=per_bridge_url)
        # Keep the gateway's reference in sync — it uses ``self._bridges``
        # for shutdown signaling on gateway.stop().
        if self._gateway is not None:
            self._gateway._bridges = self._bridges

    async def stop(self, timeout: float = 15) -> None:
        """Stop bridges, gateway, scheduler, agent (in reverse).

        Each phase gets up to *timeout* seconds.  If the agent shutdown
        (which closes MCP subprocesses) hangs, we log a warning and
        move on so the process can still exit.
        """
        elog("server.stop", agent=self.agent.name)
        # 1. Stop bridges
        for bridge in self._bridges:
            try:
                await asyncio.wait_for(bridge.stop(), timeout=10)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("Bridge %s stop error: %s", bridge.name, e)
        for t in self._bridge_tasks:
            if not t.done():
                t.cancel()
        for t in self._bridge_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._bridge_tasks.clear()

        # 1b. Bridge sessions (one LoopbackProxy + dialer + iroh self-conn
        #     per bridge). After all bridges are stopped — they may still
        #     be writing to the loopback socket during cancellation.
        for s in self._bridge_sessions:
            try:
                await asyncio.wait_for(s.stop(), timeout=5)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(
                    "Bridge session %s stop error: %s",
                    getattr(s, "bridge_name", "?"), e,
                )
        self._bridge_sessions.clear()

        # 1b. Curator (kill before the DB-owning Agent goes away)
        if self._curator_task is not None:
            self._curator_task.cancel()
            try:
                await asyncio.wait_for(self._curator_task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            self._curator_task = None

        # 1c. Vault autocommit loop. ``_vault_maint_task`` used to be cancelled
        # here too; that loop is gone (v0.16.1 — see ``_build_agent``).
        for _attr in ("_vault_autocommit_task",):
            _vt = getattr(self, _attr, None)
            if _vt is not None:
                _vt.cancel()
                try:
                    await asyncio.wait_for(_vt, timeout=2)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
                setattr(self, _attr, None)
        # Final sweep: commit anything still pending so a clean shutdown
        # doesn't strand uncommitted vault edits.
        try:
            from src.memory.vault.service import get_service
            await get_service().autocommit(origin={"kind": "shutdown"})
        except Exception:  # noqa: BLE001
            pass
        try:
            from src.memory.vault.service import close_all as _vault_close_all
            await _vault_close_all()
        except Exception:  # noqa: BLE001
            pass

        # 2. Gateway
        if self._gateway:
            try:
                await asyncio.wait_for(self._gateway.stop(), timeout=10)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("Gateway stop error: %s", e)

        # 2b. NetworkState (Iroh endpoint + coordinator service)
        if self._network_state is not None:
            try:
                await asyncio.wait_for(self._network_state.stop(), timeout=10)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("NetworkState stop error: %s", e)
            self._network_state = None

        # 3. Scheduler
        if self._scheduler is not None:
            try:
                await asyncio.wait_for(self._scheduler.stop(), timeout=10)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("Scheduler stop error: %s", e)
            self._scheduler = None

        # 4. Agent (MCP subprocess cleanup can hang because the anyio-
        #    based MCP client waits for subprocesses that may ignore
        #    SIGTERM).  Give it a deadline; if it doesn't finish, log
        #    and move on — orphaned subprocesses will be reaped when we
        #    exit.  The MCP SDK uses anyio cancel scopes which can leak
        #    CancelledError into our asyncio tasks, so we catch broadly.
        try:
            shutdown_task = asyncio.create_task(self.agent.shutdown(), name="agent-shutdown")
            await asyncio.wait_for(asyncio.shield(shutdown_task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.debug("Agent shutdown still in progress after %ss; exiting best-effort", timeout)
        except Exception as e:
            logger.warning("Agent shutdown error: %s", e)

        if self._stop_event is not None:
            self._stop_event.set()

    async def wait(self) -> None:
        """Block until stop() is called or a termination signal arrives.

        Belt-and-suspenders: loop.add_signal_handler is the primary path
        (cooperates with asyncio); signal.signal is the C-extension
        fallback (iroh/tokio in particular can block the selector enough
        that the asyncio handler never fires). Both wired, both unwound
        in finally.
        """
        assert self._stop_event is not None, "Call start() first"

        loop = asyncio.get_running_loop()
        stop_event = self._stop_event

        def _signal_handler() -> None:
            stop_event.set()

        def _legacy_handler(_signum, _frame) -> None:
            # signal.signal handlers run in the main thread but outside
            # the loop; bounce through call_soon_threadsafe so Event.set
            # is invoked on the loop thread.
            loop.call_soon_threadsafe(stop_event.set)

        handled: list[int] = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
                handled.append(sig)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main thread: fall back to KeyboardInterrupt
                pass

        prev_legacy: dict[int, Any] = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev_legacy[sig] = signal.signal(sig, _legacy_handler)
            except (OSError, ValueError):
                pass

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            for sig in handled:
                try:
                    loop.remove_signal_handler(sig)
                except Exception:
                    pass
            for sig, prev in prev_legacy.items():
                try:
                    signal.signal(sig, prev)
                except Exception:
                    pass

    async def __aenter__(self) -> AgentServer:
        await self.start()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.stop()

    # ── Scheduler setup (dream mode + auto-update) ──

    async def _start_scheduler(self) -> None:
        if self.agent._db is None:
            return

        from src.core.scheduler import Scheduler
        scheduler = Scheduler(
            self.agent._db,
            self.agent,
            broadcast=self._scheduler_broadcast,
        )

        # One-time cleanup: the retired ``manager-review`` built-in used
        # to seed a row that the scheduler would keep firing from the DB.
        # Drop any leftover row so it stops running after the upgrade.
        await self._purge_retired_builtin_tasks()

        await self._sync_dream_mode(scheduler)
        await self._sync_auto_update(scheduler)
        await self._sync_skill_curator(scheduler)
        await self._sync_skill_distiller(scheduler)
        await self._sync_quality_scorer(scheduler)
        await self._sync_quality_digest(scheduler)
        await self._sync_cost_observability(scheduler)
        await self._sync_escalation_audit(scheduler)

        await scheduler.start()
        self._scheduler = scheduler
        # Expose the live scheduler to the gateway so /api/scheduled-tasks
        # can operate on the same instance that runs the cron loop.
        if self._gateway is not None:
            self._gateway._scheduler = scheduler
            # Register live-reaction hooks so toggling these sections in
            # /api/config/{section} re-syncs the underlying scheduled-task
            # row immediately (no restart).
            self._register_config_callbacks(scheduler)

    # Built-in tasks that existed in an earlier release and must be
    # removed from the DB on upgrade so the scheduler stops firing them.
    _RETIRED_BUILTIN_TASK_NAMES: frozenset[str] = frozenset({"manager-review"})

    async def _purge_retired_builtin_tasks(self) -> None:
        """Delete leftover rows for built-in tasks that no longer exist.

        ``_sync_*`` only touches tasks it still knows about, so a retired
        built-in's row would otherwise linger — enabled — and the
        scheduler would keep running its stored prompt on every boot."""
        try:
            tasks = await self.agent._db.get_tasks()
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not purge retired built-in tasks: %s", e)
            return
        for task in tasks:
            if task["name"] in self._RETIRED_BUILTIN_TASK_NAMES:
                await self.agent._db.delete_task(task["id"])
                elog("scheduler.retired_task_purged", name=task["name"])

    def _scheduler_broadcast(
        self, resource: str, action: str, id: str | None = None,
    ) -> None:
        """Forward scheduler-internal mutations (one-shot disable, run
        start, schedule advance) to the gateway broadcast bus. Sync
        because Scheduler holds asyncio loop access already."""
        gw = self._gateway
        if gw is None:
            return
        gw.broadcast_resource_sync(resource, action, id)

    def _register_config_callbacks(self, scheduler) -> None:
        """Hook ``/api/config/{section}`` PATCH writes into live scheduler
        re-sync for our built-in tasks. Updates ``self.config`` in
        place so subsequent reads see the new state."""
        gw = self._gateway
        if gw is None:
            return

        async def _dream(patch: dict) -> None:
            self.config["dream_mode"] = patch or {}
            await self._sync_dream_mode(scheduler)
            gw.broadcast_resource_sync("scheduled_task", "updated")

        async def _autoupdate(patch: dict) -> None:
            self.config["auto_update"] = patch or {}
            await self._sync_auto_update(scheduler)
            gw.broadcast_resource_sync("scheduled_task", "updated")

        async def _skills(patch: dict) -> None:
            # The ``skills`` section carries BOTH the curator toggle
            # (``skills.curator_enabled``) and the distiller toggle
            # (``skills.distiller_enabled``); re-sync both scheduled tasks so a
            # runtime flip of either enables/disables it without a restart.
            # (Toggling ``skills.enabled`` itself still needs a restart to
            # (de)register the skills MCP — this only governs the two tasks.)
            self.config["skills"] = patch or {}
            await self._sync_skill_curator(scheduler)
            await self._sync_skill_distiller(scheduler)
            gw.broadcast_resource_sync("scheduled_task", "updated")

        async def _self_improvement(patch: dict) -> None:
            # The ``self_improvement`` section carries the master ``enabled``
            # switch plus the per-task ``scorer_enabled`` / ``digest_enabled``
            # gates and optional schedule overrides; re-sync both built-in
            # tasks so a runtime flip enables/disables (or reschedules) either
            # without a restart. The dedup against a tuned custom task is
            # re-evaluated on every sync, so this stays safe on eSound/Lyra.
            self.config["self_improvement"] = patch or {}
            await self._sync_quality_scorer(scheduler)
            await self._sync_quality_digest(scheduler)
            await self._sync_cost_observability(scheduler)
            await self._sync_escalation_audit(scheduler)
            gw.broadcast_resource_sync("scheduled_task", "updated")

        gw._config_change_callbacks["dream_mode"] = _dream
        gw._config_change_callbacks["auto_update"] = _autoupdate
        gw._config_change_callbacks["skills"] = _skills
        gw._config_change_callbacks["self_improvement"] = _self_improvement

    async def _sync_scheduled_task(
        self, scheduler, *, name: str, enabled: bool, cron_expr: str, prompt: str,
        timezone: str | None = None,
    ) -> None:
        """Ensure a built-in scheduled task matches the desired state.

        The row is ALWAYS seeded (created disabled if missing) even when the
        feature is off: a built-in is "toggleable but not removable" (vision
        §12), and — load-bearing here — a manual firing (``run_dream_mode``, the
        Run-now API) needs an existing ``scheduled_tasks`` row to hang its
        ``task_runs`` history off, so it surfaces in the app's "Recent" feed.
        Seeding it disabled never makes the scheduler fire it (``get_due_tasks``
        skips disabled rows); the enable branch below flips it on when config
        wants it.
        """
        tasks = await self.agent._db.get_tasks()
        existing = next((t for t in tasks if t["name"] == name), None)

        if existing is None:
            new_id = await scheduler.add_task(
                name=name, cron_expression=cron_expr, prompt=prompt,
                timezone=timezone,
            )
            # ``add_task`` creates the row enabled with a future next_run; park
            # it disabled so a disabled built-in never fires on a cron the user
            # didn't set. The enable branch re-arms it if config is on.
            await self.agent._db.update_task(new_id, enabled=0, next_run=None)
            existing = await self.agent._db.get_task(new_id)

        if enabled:
            updates = {}
            if existing["cron_expression"] != cron_expr:
                updates["cron_expression"] = cron_expr
            if existing["prompt"] != prompt:
                updates["prompt"] = prompt
            # Sync the timezone too. It used to be passed only to ``add_task``
            # (create), so a task that already existed — every built-in, since
            # they are seeded disabled on first boot — could NEVER gain a
            # timezone from config: setting ``dream_mode.timezone`` did nothing,
            # and "3:00" kept firing at 03:00 UTC (05:00 Rome in summer). The
            # column changing needs a reschedule so ``next_run`` is recomputed
            # in the new zone, same as a cron change.
            if timezone is not None and (existing.get("timezone") or None) != timezone:
                updates["timezone"] = timezone
            if updates:
                await self.agent._db.update_task(existing["id"], **updates)
            if not existing["enabled"]:
                await scheduler.enable_task(existing["id"])
            elif "cron_expression" in updates or "timezone" in updates:
                await scheduler.reschedule_task(existing["id"])
        elif existing["enabled"]:
            await scheduler.disable_task(existing["id"])

    @staticmethod
    def _install_task_hook(scheduler, name: str, hook) -> None:
        """Register/replace a named ``run_task`` hook idempotently.

        A SINGLE dispatcher is installed over ``scheduler.run_task`` once;
        every built-in task's hook lives in a registry keyed by name. Re-
        syncing a task (e.g. when its config section is toggled at runtime
        via ``/api/config``) replaces its hook in place instead of stacking
        another monkey-patch layer — the previous ``_wrap_scheduler_run_task``
        composed a fresh closure on every call, so a few config toggles
        grew an unbounded wrapper chain and made ``_do_auto_update`` fire
        once per accumulated layer.

        Pass ``hook=None`` to remove a previously-registered hook.

        Each hook has signature ``async hook(task, next)`` and calls
        ``await next(task)`` to defer to the rest of the chain; the
        innermost call is the scheduler's real ``run_task`` with all
        original positional/keyword args (e.g. ``trigger=``) forwarded.
        """
        hooks = getattr(scheduler, "_oa_task_hooks", None)
        if hooks is None:
            hooks = {}
            scheduler._oa_task_hooks = hooks
            original_run = scheduler.run_task

            async def _dispatch(task, *args, **kwargs):
                async def _base(t):
                    await original_run(t, *args, **kwargs)

                chain = _base
                for h in reversed(list(hooks.values())):
                    chain = _compose_task_hook(h, chain)
                await chain(task)

            scheduler.run_task = _dispatch  # type: ignore[method-assign]

        if hook is None:
            hooks.pop(name, None)
        else:
            hooks[name] = hook

    async def _sync_dream_mode(self, scheduler) -> None:
        dream_cfg = self.config.get("dream_mode", {})
        enabled = dream_cfg.get("enabled", False)

        cron_expr = dream_cfg.get("cron")
        if not cron_expr:
            # Keep the nightly model-driven pass away from the :07 cost
            # watcher and :53 skill distiller. Dreaming must not compete with
            # user-facing work (§12), and the same discipline applies to the
            # agent's other intrinsic maintenance loops.
            time_str = str(dream_cfg.get("time", DREAM_MODE_DEFAULT_TIME))
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            cron_expr = f"{minute} {hour} * * *"

        # ``time: "3:00"`` reads as a wall-clock hour, and a user setting it
        # means "while I'm asleep" (§12: dreaming runs while the agent is
        # otherwise idle). But an untagged cron evaluates in UTC, so on a
        # Europe/Rome operator's k8s box "3:00" has always fired at 05:00 local
        # in summer — inside the working day, competing with the user-facing
        # work §12 says it must not compete with.
        #
        # ``dream_mode.timezone`` names the zone; absent, it inherits
        # ``scheduler.timezone``; absent both, it stays UTC — the behaviour
        # every existing deployment has today, so nothing moves on upgrade.
        dream_tz = dream_cfg.get("timezone") or default_timezone_name()

        await self._sync_scheduled_task(
            scheduler,
            name=DREAM_MODE_TASK_NAME,
            enabled=enabled,
            cron_expr=cron_expr,
            prompt=DREAM_MODE_PROMPT,
            timezone=dream_tz,
        )

        dream_hook = None
        if enabled:
            async def _dream_run(task, _orig):
                if task["name"] == DREAM_MODE_TASK_NAME:
                    elog("dream.start")
                    await _orig(task)
                    elog("dream.done")
                    clear_event_log(older_than_days=6)
                    elog("dream.log_cleared")
                else:
                    await _orig(task)

            dream_hook = _dream_run
        self._install_task_hook(scheduler, DREAM_MODE_TASK_NAME, dream_hook)

    async def _sync_auto_update(self, scheduler) -> None:
        update_cfg = self.config.get("auto_update", {})
        enabled = update_cfg.get("enabled", False)
        mode = update_cfg.get("mode", "auto")
        cron_expr = update_cfg.get("check_interval", "0 4 * * *")

        prompt = (
            "Check for updates to openagent-framework. "
            "Compare the version before and after. "
            "If updated, log the new version."
        )

        await self._sync_scheduled_task(
            scheduler,
            name=AUTO_UPDATE_TASK_NAME,
            enabled=enabled,
            cron_expr=cron_expr,
            prompt=prompt,
        )

        update_hook = None
        if enabled:
            agent = self.agent
            stop_event = self._stop_event
            gateway = self._gateway

            async def _auto_update_run(task, _orig):
                if task["name"] == AUTO_UPDATE_TASK_NAME:
                    await _do_auto_update(
                        agent, mode, stop_event=stop_event, gateway=gateway,
                    )
                else:
                    await _orig(task)

            update_hook = _auto_update_run
        self._install_task_hook(scheduler, AUTO_UPDATE_TASK_NAME, update_hook)

    async def _sync_skill_curator(self, scheduler) -> None:
        """Seed/enable the ``skill-curator`` scheduled task — "dream mode for
        skills". Gated on BOTH ``skills.enabled`` and ``skills.curator_enabled``
        (default OFF).

        Unlike dream-mode / auto-update — which ALWAYS seed a disabled row so a
        manual firing has somewhere to hang its run history — the curator has
        no manual-fire entry point, so when it is OFF it seeds NOTHING. With the
        default config the ``scheduled_tasks`` table gains no row and the
        scheduler is byte-identical to a build without this feature. That is the
        safety property this whole subsystem is built around: disabled means
        invisible, not merely dormant.

        When a previous enable left a row behind and the operator later turns
        the curator off at runtime, we disable that row (so it stops firing)
        rather than deleting it — same "toggleable but not removable" spirit as
        the other built-ins, without seeding one that never existed.
        """
        from src.core.config import skills_settings

        settings = skills_settings(self.config)
        enabled = settings.enabled and settings.curator_enabled

        if not enabled:
            # OFF → seed nothing. If a row survives from an earlier enable,
            # park it disabled so the scheduler stops firing it.
            tasks = await self.agent._db.get_tasks()
            existing = next(
                (t for t in tasks if t["name"] == SKILL_CURATOR_TASK_NAME), None,
            )
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, SKILL_CURATOR_TASK_NAME, None)
            return

        cron_expr = settings.curator_schedule or SKILL_CURATOR_DEFAULT_CRON
        # Same timezone treatment as dream mode: an untagged cron evaluates in
        # UTC, so inherit the box's configured zone (falls back to UTC) so a
        # "Sunday 04:53" run lands overnight local, not mid-morning.
        curator_tz = default_timezone_name()

        await self._sync_scheduled_task(
            scheduler,
            name=SKILL_CURATOR_TASK_NAME,
            enabled=True,
            cron_expr=cron_expr,
            prompt=SKILL_CURATOR_PROMPT,
            timezone=curator_tz,
        )

        async def _curator_run(task, _orig):
            if task["name"] == SKILL_CURATOR_TASK_NAME:
                elog("skill_curator.start")
                await _orig(task)
                elog("skill_curator.done")
            else:
                await _orig(task)

        self._install_task_hook(scheduler, SKILL_CURATOR_TASK_NAME, _curator_run)

    async def _sync_skill_distiller(self, scheduler) -> None:
        """Seed/enable the ``skill-distiller`` scheduled task — the automatic
        WRITER of the self-improvement loop. Gated on BOTH ``skills.enabled``
        and ``skills.distiller_enabled`` (default OFF).

        Sibling of ``_sync_skill_curator`` and identical in discipline: like the
        curator (and unlike dream-mode / auto-update), it has no manual-fire
        entry point, so when it is OFF it seeds NOTHING — with the default config
        the ``scheduled_tasks`` table gains no distiller row and the scheduler is
        byte-identical to a build without this feature. A row left behind by an
        earlier enable is parked disabled (not deleted) when the operator turns
        it off at runtime.

        Layering: the distiller and curator are DISTINCT tasks that both live in
        the ``skills`` config section but flip independently — the distiller
        CREATES new skills, the curator CONSOLIDATES them, and either half may
        run without the other.
        """
        from src.core.config import skills_settings

        settings = skills_settings(self.config)
        enabled = settings.enabled and settings.distiller_enabled

        if not enabled:
            # OFF → seed nothing. Disable a surviving row from an earlier enable
            # so the scheduler stops firing it.
            tasks = await self.agent._db.get_tasks()
            existing = next(
                (t for t in tasks if t["name"] == SKILL_DISTILLER_TASK_NAME), None,
            )
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, SKILL_DISTILLER_TASK_NAME, None)
            return

        cron_expr = settings.distiller_schedule or SKILL_DISTILLER_DEFAULT_CRON
        # Same timezone treatment as dream mode / the curator: an untagged cron
        # evaluates in UTC, so inherit the box's configured zone (falls back to
        # UTC) so a "03:53" run lands overnight local, not mid-morning.
        distiller_tz = default_timezone_name()

        await self._sync_scheduled_task(
            scheduler,
            name=SKILL_DISTILLER_TASK_NAME,
            enabled=True,
            cron_expr=cron_expr,
            prompt=SKILL_DISTILLER_PROMPT,
            timezone=distiller_tz,
        )

        async def _distiller_run(task, _orig):
            if task["name"] == SKILL_DISTILLER_TASK_NAME:
                elog("skill_distiller.start")
                await _orig(task)
                elog("skill_distiller.done")
            else:
                await _orig(task)

        self._install_task_hook(scheduler, SKILL_DISTILLER_TASK_NAME, _distiller_run)

    async def _sync_quality_scorer(self, scheduler) -> None:
        """Seed/enable the ``quality-scorer`` built-in — the INTRINSIC
        self-improvement grader. Unlike the skill builtins (opt-in), this is ON
        by default (``self_improvement.enabled`` AND ``scorer_enabled``, both
        defaulting True): every agent scores its own recent output against its
        own vault rules and files grounded corrections, with no per-agent config.

        DEDUP — the one thing that makes "intrinsic" safe on tuned agents: an
        agent that already ships a NON-builtin, hand-tuned quality-scorer
        (eSound/Lyra do) keeps it. Before seeding the builtin we scan
        ``scheduled_tasks`` for a non-builtin task whose name serves the same
        purpose (mentions quality + scoring); if one exists we DEFER — park any
        stray builtin row disabled and install no hook — so we never double-run.
        Fresh agents (spicysparks, new deployments) get the generic builtin.

        Which branch was taken is logged so a firing's provenance is auditable.
        """
        from src.core.config import self_improvement_settings

        settings = self_improvement_settings(self.config)
        enabled = settings.enabled and settings.scorer_enabled

        tasks = await self.agent._db.get_tasks()
        existing = next(
            (t for t in tasks if t["name"] == QUALITY_SCORER_TASK_NAME), None,
        )

        # DEDUP: defer to an agent's own tuned, non-builtin quality-scorer.
        custom = next(
            (t for t in tasks
             if t["name"] not in BUILTIN_TASK_NAMES
             and _is_custom_quality_scorer_name(t["name"])),
            None,
        )
        if custom is not None:
            elog("quality_scorer.deferred_to_custom", custom_task=custom["name"])
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, QUALITY_SCORER_TASK_NAME, None)
            return

        if not enabled:
            # OFF → disable a surviving row so the scheduler stops firing it.
            elog("quality_scorer.disabled")
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, QUALITY_SCORER_TASK_NAME, None)
            return

        elog("quality_scorer.enabled_builtin")
        cron_expr = settings.scorer_schedule or QUALITY_SCORER_DEFAULT_CRON
        # Same timezone treatment as the skill builtins: an untagged cron
        # evaluates in UTC, so inherit the box's configured zone (falls back to
        # UTC) so a 2-hour cadence remains anchored to the local wall-clock.
        scorer_tz = default_timezone_name()

        await self._sync_scheduled_task(
            scheduler,
            name=QUALITY_SCORER_TASK_NAME,
            enabled=True,
            cron_expr=cron_expr,
            prompt=QUALITY_SCORER_PROMPT,
            timezone=scorer_tz,
        )

        async def _scorer_run(task, _orig):
            if task["name"] == QUALITY_SCORER_TASK_NAME:
                elog("quality_scorer.start")
                await _orig(task)
                elog("quality_scorer.done")
            else:
                await _orig(task)

        self._install_task_hook(scheduler, QUALITY_SCORER_TASK_NAME, _scorer_run)

    async def _sync_quality_digest(self, scheduler) -> None:
        """Seed/enable the ``quality-digest`` built-in — the daily synthesis
        half of the intrinsic loop. Sibling of ``_sync_quality_scorer`` and
        identical in discipline: ON by default (``self_improvement.enabled`` AND
        ``digest_enabled``), and it DEFERS to an agent's own tuned, non-builtin
        quality-digest (name mentions quality + digest/improvement) so
        eSound/Lyra keep their custom pass while fresh agents get the builtin.

        Independent of the scorer's toggle — either half may run alone.
        """
        from src.core.config import self_improvement_settings

        settings = self_improvement_settings(self.config)
        enabled = settings.enabled and settings.digest_enabled

        tasks = await self.agent._db.get_tasks()
        existing = next(
            (t for t in tasks if t["name"] == QUALITY_DIGEST_TASK_NAME), None,
        )

        # DEDUP: defer to an agent's own tuned, non-builtin quality-digest.
        custom = next(
            (t for t in tasks
             if t["name"] not in BUILTIN_TASK_NAMES
             and _is_custom_quality_digest_name(t["name"])),
            None,
        )
        if custom is not None:
            elog("quality_digest.deferred_to_custom", custom_task=custom["name"])
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, QUALITY_DIGEST_TASK_NAME, None)
            return

        if not enabled:
            elog("quality_digest.disabled")
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, QUALITY_DIGEST_TASK_NAME, None)
            return

        elog("quality_digest.enabled_builtin")
        cron_expr = settings.digest_schedule or QUALITY_DIGEST_DEFAULT_CRON
        digest_tz = default_timezone_name()

        await self._sync_scheduled_task(
            scheduler,
            name=QUALITY_DIGEST_TASK_NAME,
            enabled=True,
            cron_expr=cron_expr,
            prompt=QUALITY_DIGEST_PROMPT,
            timezone=digest_tz,
        )

        async def _digest_run(task, _orig):
            if task["name"] == QUALITY_DIGEST_TASK_NAME:
                elog("quality_digest.start")
                await _orig(task)
                elog("quality_digest.done")
            else:
                await _orig(task)

        self._install_task_hook(scheduler, QUALITY_DIGEST_TASK_NAME, _digest_run)

    async def _sync_cost_observability(self, scheduler) -> None:
        """Seed/enable the ``cost-observability`` built-in — the CONSUMPTION arm
        of the intrinsic loop. Sibling of the quality builtins and identical in
        discipline: ON by default (``self_improvement.enabled`` AND
        ``cost_observability_enabled``), and it DEFERS to an agent's own tuned,
        non-builtin cost watcher (name mentions cost + observe/anomaly/monitor)
        so eSound/Lyra keep their custom, more-sensitive pass while fresh agents
        (spicysparks, new deployments) get the generic builtin.

        The builtin is CACHE-AWARE by construction: its prompt pages only on the
        engine's own ``router.cost_anomaly`` signal (real cost + fresh/non-cached
        tokens), never on raw summed input. Independent of the scorer/digest
        toggles — any of the three halves may run alone.
        """
        from src.core.config import self_improvement_settings

        settings = self_improvement_settings(self.config)
        enabled = settings.enabled and settings.cost_observability_enabled

        tasks = await self.agent._db.get_tasks()
        existing = next(
            (t for t in tasks if t["name"] == COST_OBSERVABILITY_TASK_NAME), None,
        )

        # DEDUP: defer to an agent's own tuned, non-builtin cost watcher.
        custom = next(
            (t for t in tasks
             if t["name"] not in BUILTIN_TASK_NAMES
             and _is_custom_cost_observability_name(t["name"])),
            None,
        )
        if custom is not None:
            elog("cost_observability.deferred_to_custom", custom_task=custom["name"])
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, COST_OBSERVABILITY_TASK_NAME, None)
            return

        if not enabled:
            elog("cost_observability.disabled")
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, COST_OBSERVABILITY_TASK_NAME, None)
            return

        elog("cost_observability.enabled_builtin")
        cron_expr = settings.cost_observability_schedule or COST_OBSERVABILITY_DEFAULT_CRON
        cost_tz = default_timezone_name()

        await self._sync_scheduled_task(
            scheduler,
            name=COST_OBSERVABILITY_TASK_NAME,
            enabled=True,
            cron_expr=cron_expr,
            prompt=COST_OBSERVABILITY_PROMPT,
            timezone=cost_tz,
        )

        async def _cost_run(task, _orig):
            if task["name"] == COST_OBSERVABILITY_TASK_NAME:
                elog("cost_observability.start")
                await _orig(task)
                elog("cost_observability.done")
            else:
                await _orig(task)

        self._install_task_hook(scheduler, COST_OBSERVABILITY_TASK_NAME, _cost_run)

    async def _sync_escalation_audit(self, scheduler) -> None:
        """Seed/enable the ``escalation-audit`` built-in — the HANDOFF arm of the
        intrinsic loop. Sibling of the quality/cost builtins and identical in
        discipline: ON by default (``self_improvement.enabled`` AND
        ``escalation_audit_enabled``), and it DEFERS to an agent's own tuned,
        non-builtin escalation auditor (name mentions escalation/handoff + audit)
        so eSound/Lyra keep their tuned ``support-escalation-audit`` while fresh
        agents get the generic, ROLE-/TOOL-agnostic builtin.

        Independent of the other self_improvement toggles — any arm may run alone.
        """
        from src.core.config import self_improvement_settings

        settings = self_improvement_settings(self.config)
        enabled = settings.enabled and settings.escalation_audit_enabled

        tasks = await self.agent._db.get_tasks()
        existing = next(
            (t for t in tasks if t["name"] == ESCALATION_AUDIT_TASK_NAME), None,
        )

        # DEDUP: defer to an agent's own tuned, non-builtin escalation auditor.
        custom = next(
            (t for t in tasks
             if t["name"] not in BUILTIN_TASK_NAMES
             and _is_custom_escalation_audit_name(t["name"])),
            None,
        )
        if custom is not None:
            elog("escalation_audit.deferred_to_custom", custom_task=custom["name"])
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, ESCALATION_AUDIT_TASK_NAME, None)
            return

        if not enabled:
            elog("escalation_audit.disabled")
            if existing is not None and existing["enabled"]:
                await scheduler.disable_task(existing["id"])
            self._install_task_hook(scheduler, ESCALATION_AUDIT_TASK_NAME, None)
            return

        elog("escalation_audit.enabled_builtin")
        cron_expr = settings.escalation_audit_schedule or ESCALATION_AUDIT_DEFAULT_CRON
        audit_tz = default_timezone_name()

        await self._sync_scheduled_task(
            scheduler,
            name=ESCALATION_AUDIT_TASK_NAME,
            enabled=True,
            cron_expr=cron_expr,
            prompt=ESCALATION_AUDIT_PROMPT,
            timezone=audit_tz,
        )

        async def _audit_run(task, _orig):
            if task["name"] == ESCALATION_AUDIT_TASK_NAME:
                elog("escalation_audit.start")
                await _orig(task)
                elog("escalation_audit.done")
            else:
                await _orig(task)

        self._install_task_hook(scheduler, ESCALATION_AUDIT_TASK_NAME, _audit_run)


# ── Auto-update helpers (used by AgentServer and the manual `update` command) ──

PACKAGE_NAME = "openagent-framework"


def get_installed_version() -> str:
    from src._frozen import is_frozen
    if is_frozen():
        import src
        return getattr(src, "__version__", "unknown")
    try:
        from importlib.metadata import version
        return version(PACKAGE_NAME)
    except Exception:
        return "unknown"


def _run_pip_upgrade() -> tuple[str, str]:
    """Run pip install --upgrade and return (old_version, new_version)."""
    import subprocess
    import sys

    old = get_installed_version()
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from importlib.metadata import version
    try:
        from importlib import invalidate_caches
        invalidate_caches()
    except Exception:
        pass
    new = version(PACKAGE_NAME)
    return old, new


def _binary_replaced_by_sibling() -> bool:
    """Return True if our on-disk executable's mtime differs from what
    we captured at process start — i.e. a sibling service that shares
    this binary has already applied its own update. Calling
    perform_self_update_sync against a swapped archive raises
    ``zlib.error`` because PyInstaller's lazy module loader reads
    using offsets from the old archive layout."""
    if _INITIAL_EXECUTABLE_MTIME is None:
        return False
    try:
        return (
            src._frozen.executable_path().stat().st_mtime
            != _INITIAL_EXECUTABLE_MTIME
        )
    except Exception:  # noqa: BLE001
        return False


def _read_disk_binary_version() -> str | None:
    """Ask the on-disk binary for its version. Used after a sibling
    swap so we can report the new version without trying to read the
    PyInstaller archive directly. Returns None on any failure — the
    caller falls back to a synthetic placeholder."""
    import subprocess
    try:
        path = src._frozen.executable_path()
        # ``selfcheck --quiet`` prints the bare version and exits 0; it
        # is the canonical version probe (the CLI has no ``--version``
        # subcommand in older naming, and the group ``--version`` adds a
        # prefix). 30 s because a frozen onefile cold-extracts on first
        # launch.
        out = subprocess.check_output(
            [str(path), "selfcheck", "--quiet"], timeout=30, stderr=subprocess.DEVNULL
        )
        line = out.decode("utf-8", "replace").strip().splitlines()[-1] if out.strip() else ""
        return line.split()[-1] if line else None
    except Exception:  # noqa: BLE001
        return None


def run_upgrade() -> tuple[str, str]:
    """Upgrade OpenAgent and return (old_version, new_version).

    Dispatches to executable self-update when running from a frozen
    binary, or to pip upgrade when running from a pip installation.
    """
    from src._frozen import is_frozen
    if is_frozen():
        if _binary_replaced_by_sibling():
            # A sibling service that shares our on-disk binary already
            # applied its own update. Our running image is stale; the
            # restart that follows this return will pick up the new
            # binary. Skip download/apply so we don't crash trying to
            # read the freshly-rewritten PyInstaller archive.
            import src
            current = getattr(src, "__version__", "unknown")
            new = _read_disk_binary_version() or f"{current}+sibling-swap"
            elog(
                "update.swap_already_applied",
                level="warning",
                current_running=current,
                new_disk=new,
            )
            return current, new
        from src.updater import perform_self_update_sync
        return perform_self_update_sync()
    return _run_pip_upgrade()


# Backward compat alias
run_pip_upgrade = run_upgrade


async def _do_auto_update(
    agent: Agent,
    mode: str,
    stop_event: asyncio.Event | None = None,
    gateway=None,
) -> None:
    """Check for updates and act according to *mode* (auto/notify/manual).

    When *mode* is ``"auto"`` and an update was installed, signals the
    server to shut down gracefully via *stop_event* and stores the
    restart exit code on the agent so the CLI can pick it up **after**
    cleanup has finished.

    Going through :func:`request_restart` (when *gateway* is provided)
    is what fires the proactive bridge-offset flush — without it, the
    Telegram update that triggered an /update command can replay after
    launchd brings the new binary up. We saw the flush still get an
    ``offset_flush_error`` with empty error, but at least the proactive
    POST happens before the loop tears down.
    """
    try:
        old_ver, new_ver = await asyncio.to_thread(run_upgrade)
    except Exception as exc:
        logger.error("Auto-update check failed: %s", exc)
        elog("update.error", level="warning", error=str(exc) or type(exc).__name__)
        return

    if old_ver == new_ver:
        logger.info("openagent-framework is up-to-date (%s)", old_ver)
        elog("update.check", version=old_ver, updated=False)
        return

    logger.info("openagent-framework updated: %s -> %s", old_ver, new_ver)
    elog("update.installed", old=old_ver, new=new_ver)

    if mode == "auto":
        logger.warning("Restarting for update %s -> %s (exit code %d)...",
                        old_ver, new_ver, RESTART_EXIT_CODE)
        if gateway is not None:
            from src.gateway.api.control import request_restart
            request_restart(gateway, source="auto-update")
            return
        # Fallback when no gateway is wired (e.g. headless test rigs):
        # store the exit code and signal the loop directly. The bridge
        # offset flush won't fire on this path — but it's also a path
        # that has no bridges to flush.
        agent._restart_exit_code = RESTART_EXIT_CODE
        if stop_event is not None:
            stop_event.set()
        else:
            raise SystemExit(RESTART_EXIT_CODE)
        # Don't try to send a notification when we're about to restart —
        # it would block the shutdown while the LLM processes the request.
        return

    if mode == "notify":
        try:
            msg = f"OpenAgent updated: {old_ver} -> {new_ver}"
            tools = agent._mcp.all_tools()
            has_messaging = any(t["name"].startswith("send_") for t in tools)
            if has_messaging:
                await agent.run(
                    message=f"Send a notification: {msg}",
                    user_id="system",
                )
        except Exception:
            logger.debug("Could not send update notification via messaging MCP")
