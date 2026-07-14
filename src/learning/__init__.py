"""Background loops that keep the agent's memory healthy.

Submodules:
  ``vault_reminder``    — per-session memory-checkpoint nudge prepended to the
                          user turn, enforcing the §5 discipline ("after any
                          meaningful learning, the agent writes to the vault").
                          Pure prompt text: no model call, no provider.
  ``vault_maintenance`` — the dream loop (§12): gate, mechanically fix,
                          regenerate derived artifacts, write a dream-log.
  ``curator``           — housekeeping: prune dormant sessions, snapshot the DB.
  ``_model``            — provider-agnostic single-shot completion for the
                          above, on an operator-named model.

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
