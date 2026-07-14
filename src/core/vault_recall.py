"""Outcome attribution for vault recall — which notes preceded which run outcomes.

Vision §5 makes consulting the vault a discipline ("before acting on any
non-trivial question, the agent consults the vault"), and §12 has dream mode
prune it. Neither has any evidence to work from: the vault records that a note
EXISTS, never whether recalling it preceded a run that went anywhere. Recall is
therefore un-ranked — every note is equally worth reading — and dream mode
prunes on structure (orphans, broken links) rather than on use.

This module records the join that was missing: ``note → run → outcome``. It is
NOT a second memory store (§5 rules that out explicitly, and ``learning/`` was
gutted for being one). No note text is stored here — only vault-relative paths,
which are already public in the vault's own filenames. The notes stay Markdown;
this is a side-table of counters ABOUT them, the way ``usage_log`` is a
side-table about model calls.

WHY A CONTEXTVAR SINK, AND WHY HERE
-----------------------------------
Mirrors ``src/models/stream_usage.py`` deliberately — same problem, same shape,
same reason. The pieces that SEE a tool execution (``NativeProvider.stream``,
``_arun_runtime_stream``) cannot import the dispatcher without a cycle, so the
sink lives in a leaf module everybody imports. The dispatcher opens the sink
per call; the inner generators only MUTATE the dict, never rebind the
ContextVar, so nothing depends on context propagating out of an async
generator.

WHY NOT ``agent.turn.tool_calls``
---------------------------------
``_emit_tool_call_summary`` (core/agent.py) was the obvious source: it already
counts vault reads per iteration. It has never once fired in production. The
real log — 11,360 events between 2026-05-18 and 2026-07-14 — contains ZERO
``agent.turn.tool_calls`` entries, because that hook reads
``ModelResponse.tool_names_called``, which only the NON-streaming path
populates. Over that same window ``runtime.generate`` fired 11 times against
697 streamed turns. Counting vault reads there measures a path production does
not take. This is the same defect ``stream_usage`` was written to fix ("the
streaming path used to record NOTHING… every surface that matters streams"),
and it is why recording happens at the tool-execution sites below instead.
"""
from __future__ import annotations

import contextvars
from typing import Any, Optional

# Terminal state of the run a recall happened inside. Deliberately three
# values, and deliberately NOT named "success"/"failure".
#
#   ok        — the run finished without raising. It does NOT mean the answer
#               was correct, useful, or that any recalled note helped. It means
#               the turn completed. That is the whole claim.
#   errored   — an exception propagated out of the run.
#   cancelled — the user barged in (§2: "interrupt and barge-in are first-class
#               behaviors, not afterthoughts"). NOT a failure. Recorded so the
#               denominator stays honest, but never scored — see SCORABLE.
OUTCOME_OK = "ok"
OUTCOME_ERRORED = "errored"
OUTCOME_CANCELLED = "cancelled"

# The outcomes a rate may be computed over. ``cancelled`` is excluded by
# construction rather than by a filter someone can forget to apply.
#
# This is not a hypothetical hazard. On the production log every single one of
# the 294 entries carrying ``errored=True`` ALSO carried ``cancelled=True``
# (and there were 295 ``stream.barge_in`` events to match). A scorer that
# treats errored-or-cancelled as failure therefore learns, from a perfect
# 294/294 correlation, that a user interrupting the agent is a defect to
# minimise — i.e. it learns to discourage a behaviour the vision calls
# first-class. Cancel is checked BEFORE error for the same reason: a barge-in
# that surfaces as CancelledError must never be booked as a failure.
SCORABLE = frozenset({OUTCOME_OK, OUTCOME_ERRORED})

# Vault tools that RECALL a note — i.e. put THAT NOTE'S OWN TEXT in front of
# the model — mapped to the argument holding the path(s). Every key here must
# be a live registration; ``test_vault_recall`` asserts it against the real
# ones, because a hand-kept tool list is exactly what rotted three times this
# session (the framework prompt, the channel labels, and ``_VAULT_READ_TOOLS``
# in core/agent.py, which has silently matched no browse tool since it
# shipped). Note the two spellings: ``vault`` is a Node subprocess so its keys
# are prefixed, while ``vault-gate`` is in-process so its keys are the bare
# function names.
#
# The line is "did this note's CONTENT enter the model", because that is the
# only thing that can help or mislead a run — and therefore the only thing
# worth scoring. That rules out three near-misses:
#
#   ``vault_search_notes`` / ``vault_list_directory`` — the paths live in the
#     RESULT, not the arguments, and surfacing a note is not reading it: the
#     model may ignore every hit. Booking hits would inflate every note a broad
#     query happened to match and wreck the denominator. A search that mattered
#     is followed by a read, and the read is what gets counted.
#   ``vault_backlinks`` — returns the names of OTHER notes pointing at this
#     one. A graph fact, not the note's claims; the note's text never arrives.
#     (It is a real tool and a real vault READ — ``_VAULT_READ_TOOLS`` counts
#     it — it is just not a RECALL. Do not "fix" this by adding it.)
#   ``vault_write_note`` et al — a write says what the agent learned, not what
#     it consulted; crediting it would score a note for the run that wrote it.
_RECALL_TOOL_ARGS: dict[str, str] = {
    "vault_read_note": "path",
    "vault_read_multiple_notes": "paths",
    # Frontmatter is the note's OWN text (title/summary/tags) and the agent
    # named the note to get it — content arrived, so it counts.
    "vault_get_frontmatter": "path",
}

# A vault path is short by construction. Anything longer is not a path — it is
# a note BODY that arrived in the wrong field, and carrying it would put
# unbounded model-authored text into a ContextVar and then a DB row. The whole
# reason this module extracts paths rather than raw ``tool_args`` is that
# ``vault_write_note``'s ``content`` can be tens of kilobytes.
_MAX_PATH_LEN = 512
# Per-run cap on distinct notes. A runaway loop re-reading the vault must cost
# a bounded number of rows, not an unbounded write amplification on a turn that
# is already going wrong.
_MAX_PATHS_PER_RUN = 64

_SINK: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "openagent_vault_recall_sink", default=None
)


def open_sink() -> tuple[dict, contextvars.Token]:
    """Start collecting vault recalls for one call."""
    sink: dict[str, Any] = {"paths": {}}
    return sink, _SINK.set(sink)


def close_sink(token: contextvars.Token) -> None:
    _SINK.reset(token)


def _clean_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.strip()
    if not path or len(path) > _MAX_PATH_LEN:
        return None
    return path


def note_paths_from_tool(tool_name: Any, tool_args: Any) -> list[str]:
    """Extract the vault note path(s) a tool call recalled.

    Returns ``[]`` for anything that is not a vault read — including every
    non-vault tool, which is the overwhelming majority of calls. Takes the
    already-split name/args rather than a runtime ``ToolExecution`` so the
    four call sites can pass whatever shape they have (the runtime has
    changed that object before) and so this stays unit-testable without
    constructing a runtime object.
    """
    if not isinstance(tool_name, str):
        return []
    arg_key = _RECALL_TOOL_ARGS.get(tool_name)
    if arg_key is None:
        return []
    if not isinstance(tool_args, dict):
        return []
    raw = tool_args.get(arg_key)
    if isinstance(raw, str):
        path = _clean_path(raw)
        return [path] if path else []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            path = _clean_path(item)
            if path:
                out.append(path)
        return out
    return []


def record_tool(tool_name: Any, tool_args: Any) -> None:
    """Note one executed tool call, if it recalled a vault note.

    A no-op when no sink is open — a provider streaming outside the dispatcher
    (a test, a direct call) must not blow up on telemetry. Never raises: a
    bookkeeping miss must cost a counter, never a turn.
    """
    sink = _SINK.get()
    if sink is None:
        return
    try:
        paths = note_paths_from_tool(tool_name, tool_args)
    except Exception:  # noqa: BLE001
        return
    if not paths:
        return
    seen: dict[str, str] = sink["paths"]
    for path in paths:
        if path in seen:
            continue
        if len(seen) >= _MAX_PATHS_PER_RUN:
            return
        # First tool to surface a note owns the attribution. Re-reading the
        # same note twice in a turn is one recall, not two — otherwise a
        # retry loop would weight a note purely by how badly the turn went.
        seen[path] = str(tool_name)


def recorded_paths(sink: dict | None) -> dict[str, str]:
    """The ``{note_path: tool_name}`` map collected in a sink."""
    if not sink:
        return {}
    paths = sink.get("paths")
    return dict(paths) if isinstance(paths, dict) else {}


def outcome_for_exception(exc: BaseException | None) -> str:
    """Classify how a run ended.

    ``asyncio.CancelledError``/``GeneratorExit`` mean the consumer stopped
    reading — a barge-in, or a client that hung up. Checked FIRST and mapped
    to ``cancelled``, never ``errored``: see SCORABLE for the 294/294 reason.
    """
    import asyncio

    if exc is None:
        return OUTCOME_OK
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return OUTCOME_CANCELLED
    return OUTCOME_ERRORED


async def flush(
    db: Any,
    *,
    sink: dict | None,
    session_id: str | None,
    outcome: str,
    model: str | None,
    cost: float = 0.0,
) -> int:
    """Persist one row per (run, note) recalled. Returns rows written.

    Best-effort by contract. This runs in a ``finally`` that may already be
    unwinding a barge-in, so it must not raise, and must not turn a cancelled
    turn into a crashed one.

    ``cost`` is passed in rather than joined from ``usage_log`` later. The
    ledger is keyed by ``session_id``, not by run, and a session has many
    turns — so a later join would smear a whole session's spend across every
    note any turn in it ever read. The caller already computed this run's cost
    with ``BudgetTracker.compute_cost``, the same function ``usage_log`` uses,
    so the number has the same provenance at per-run resolution.
    """
    paths = recorded_paths(sink)
    if not paths or db is None:
        return 0
    recorder = getattr(db, "record_vault_recall", None)
    if recorder is None:
        return 0
    written = 0
    for path, tool in paths.items():
        try:
            await recorder(
                session_id=session_id,
                note_path=path,
                tool=tool,
                outcome=outcome,
                model=model,
                cost=cost,
            )
            written += 1
        except Exception as e:  # noqa: BLE001
            from src.core.logging import elog

            elog(
                "vault.recall_record_error",
                level="warning",
                session_id=session_id,
                note_path=path,
                error=str(e),
            )
            break
    return written
