"""Background loops that keep the agent's memory healthy.

Submodules:
  ``vault_reminder``    — per-session memory-checkpoint nudge prepended to the
                          user turn, enforcing the §5 discipline ("after any
                          meaningful learning, the agent writes to the vault").
                          Pure prompt text: no model call, no provider.
  ``curator``           — housekeeping: prune dormant sessions, snapshot the DB.

Neither calls a model, so this package no longer holds a provider seam at all
— see below.

THE DREAM LOOP IS GONE, AND DREAM MODE IS NOT (v0.16.1)
-------------------------------------------------------
``vault_maintenance`` (a 12-hourly asyncio loop) and ``_model`` (the
provider-agnostic completion it called) were deleted. This is a
consolidation, not a removal: §12's dream mode is the ``dream-mode``
SCHEDULED TASK (``core/server._sync_dream_mode`` + ``DREAM_MODE_PROMPT``),
which is the only one of the two that matches what §12 actually describes —
"nightly by default, at a time the user can adjust ... does not compete with
user-facing work". An interval loop counting 12h from boot cannot be aimed at
a time, and fires mid-conversation by construction.

Until ``89c7379`` the loop was load-bearing despite that, because the prompt
named ``vault_dream``/``vault_gate``/``vault_doctor``/
``vault_regenerate_derived`` zero times: the loop was the only thing calling
``VaultService.maintenance()``, so deleting it would have silently dropped the
mechanical pass. Mission 1 now opens with ``vault_dream()`` — the identical
``svc.maintenance(apply_fixes=True, regenerate=True)`` call — so the task does
both halves and the loop was duplication.

``_model`` went with it because the loop's AI-suggestion step was its ONLY
caller, and that step is worse than what replaced it. It asked a cheap model
to write one-line advice about the ``open_suggestions`` code could not fix,
then wrote the advice into a log note that nothing ever read back. The
scheduled task receives the same ``open_suggestions`` from ``vault_dream()``
while holding ``write_note``/``patch_note``/``delete_note``/
``vault_rename_note``: it merges the duplicate instead of noting that someone
should. Porting the advisor into the task would have meant paying a second
model call to tell the fixer what it is already looking at. The one thing
``_model`` knew that is worth keeping — why such a model must be named by the
operator rather than inferred from ``is_classifier`` (which means "team
leader", i.e. the user's most expensive row) or ``tier_hint`` (free-form prose;
grepping it for /cheap|fast/ is the switch statement §3 forbids) — survives
verbatim in ``core/compaction.py``'s ``_SUMMARY_MODEL_ENV``, the module
``_model`` was modelled on.

WHAT WAS HERE AND WHY IT IS NOT (v0.15.11)
------------------------------------------
This package used to also hold ``user_profile`` (a Groq-summarised JSON blob
of the user's preferences/projects/style) and ``skills`` (Groq-detected
markdown how-tos), plus the ``_groq`` client they shared. All of it was
deleted rather than wired, for two reasons that are worth not re-litigating:

1. They were a second memory system competing with the vault. A "preference",
   a "how-to", an "ongoing project" is a NOTE — §5 is explicit that long-term
   memory is "human-inspectable Markdown that the user can read, edit, and
   reorganize directly", and §18 that anything a user might want to read lives
   in a format they can read. These stored the same facts in opaque SQLite
   rows the user could not see, Obsidian could not render, dream mode could
   not consolidate, and the curator silently deleted after 90 days. The vault
   + ``vault_reminder`` cover the same ground in the shape the vision asks for.
2. Their write halves were hardcoded to Groq, which is why they were never
   wired in the first place: doing so would have made one vendor a structural
   requirement for the agent to learn (§17). Deleting them removed the
   hardcoding more honestly than a rewrite would have — there was no
   provider-agnostic version of a subsystem that should not exist.

The ``skills``/``user_profiles`` tables still exist in ``memory/db.py`` and
have no writer; ``bridges/telegram.py``'s ``/export`` still reads them and
will keep returning empty payloads, as it always has (both tables have been
unwritable since they were introduced — the detector and flush hooks had zero
callers). Dropping the tables and the export needs an owner for those files.

The one thing genuinely lost is *automatic* injection of relevant memory: the
skills matcher pushed how-tos into the turn without being asked, whereas the
vault must be pulled via ``vault_search_notes``. If that push is wanted back,
it should read the vault — not a parallel store — and it pairs naturally with
outcome scoring (inject note → score the run → prefer notes that led to good
runs). See the seam noted in ``core/agent.py:_emit_tool_call_summary``.
"""
