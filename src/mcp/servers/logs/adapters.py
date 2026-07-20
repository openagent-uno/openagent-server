"""Provider adapter for the in-process ``logs`` MCP.

Follows the runtime ``Toolkit`` pattern (see ``vault_gate/adapters.py``):
plain async callables with type hints + docstrings, wrapped once. The runtime
turns the docstrings/signatures into the tool schema the model sees.

Tool names carry the ``logs_`` prefix explicitly. In-process Toolkits are not
passed through the pool's ``_safe_prefix`` namespacing (no
``tool_name_prefix`` constructor arg — see ``tool_search/adapters.py``), so a
bare ``query`` would land in the model's flat tool namespace as ``query`` and
collide with the next MCP that has the same idea.

THE SURFACE, AND WHY IT IS THREE TOOLS
--------------------------------------
It is sized to §14's three questions, and nothing else:

* *"what went wrong yesterday?"*  → ``logs_summary(since="2d")`` — one call,
  bounded output, returns the shape of the damage (counts, top failing
  events, error types, samples) rather than a haystack to read.
* *"why did this scheduled task fail?"* → ``logs_query(event="task",
  errors_only=true)`` to find it, then ``logs_context(ts=…)`` for what
  happened immediately around it. A failure is almost never explained by its
  own line — it is explained by the three lines before it.
* *"which MCP call is slowing me down?"* → partially. ``logs_summary`` ranks
  failure/timeout hot spots per event, which is the honest half. The log
  records no per-call duration (1 of 424 ``elog`` sites passes ``duration_ms``,
  and no MCP path is among them), so a latency ranking would be fabricated.
  Documented in the tool text rather than faked.

Rejected alternatives:

* **One mega-tool** (``logs(mode=…)``) — collapses three schemas into one
  fuzzy one; the model then guesses at mutually-exclusive params.
* **A ``logs_tail`` tool** — ``logs_query()`` with no arguments already is
  the tail, and it is the strictly better default (capped, structured).
* **A dedicated ``logs_trace(session_id)``** — real pull (``session_id`` is
  the log's dominant correlation key: 112 ``elog`` sites carry it, vs 8 for
  ``run_id``), but it is exactly ``logs_query(session_id=…)`` for the timeline
  plus ``logs_summary(session_id=…)`` for the rollup. Both already exist, and
  a fourth tool that is two existing calls in a trench coat costs schema
  tokens on every discovery for no new capability.
* **``logs_clear`` / prune** — deliberately absent; see ``handlers`` for why.
"""
from __future__ import annotations

from typing import Any

from src.mcp.servers.logs import handlers


def build_runtime_toolkit() -> Any:
    from src.mcp._runtime import Toolkit

    async def logs_query(
        event: str | None = None,
        contains: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        errors_only: bool = False,
        level: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 50,
        offset: int = 0,
        time_window_minutes: int | None = None,
    ) -> dict:
        """Search YOUR OWN event log (events.jsonl) — every user turn, model
        call, MCP invocation, sub-agent delegation, scheduled-task fire,
        workflow step, and error this agent has produced. Use it to diagnose
        your own behaviour instead of shelling out to find/tail the file.

        Filters AND together; all are optional. With no arguments you get the
        most recent 50 events.

        - ``event``: match the event name by prefix or substring, e.g.
          ``"scheduler."``, ``"mcp.error"``, ``"cost"``.
        - ``contains``: free-text search over the whole entry — a task name,
          an error message, a model id.
        - ``session_id`` / ``run_id``: correlate one conversation or run.
          ``session_id`` is the reliable one; few events carry ``run_id``.
        - ``errors_only``: keep only entries that failed. Works on ALL
          entries — prefer it over ``level`` when hunting for problems.
        - ``level``: exact severity — ``"error"``, ``"warning"``, ``"info"``,
          ``"debug"``. Only entries written since the severity fix carry a
          level; older ones have none and CANNOT match, so they are excluded
          and counted in ``entries_without_level_skipped``. If that number is
          large, use ``errors_only`` instead — it spans both.
        - ``since`` / ``until``: ``"24h"``, ``"90m"``, ``"7d"`` (that long
          ago), an ISO date ``"2026-07-13"``, or epoch seconds.
        - ``time_window_minutes``: convenience alias — the last N minutes,
          equivalent to ``since="Nm"`` (ignored if ``since`` is set).
        - ``limit`` (max 200) / ``offset``: page through matches.

        Returns entries oldest→newest with ``matched_in_scan`` (how many
        matched vs. were returned) and a ``scan`` block (how much of the log
        was actually read). Output is capped — if you need more, narrow the
        filter rather than raising ``limit``.

        Start with ``logs_summary`` for an overview; use this to drill in.
        """
        if time_window_minutes is not None and since is None:
            since = f"{int(time_window_minutes)}m"
        return await handlers.logs_query(
            event=event, contains=contains, session_id=session_id,
            run_id=run_id, errors_only=errors_only, since=since, until=until,
            limit=limit, offset=offset,
        )

    async def logs_summary(
        since: str | None = None,
        until: str | None = None,
        session_id: str | None = None,
        event: str | None = None,
        top: int = 15,
        time_window_minutes: int | None = None,
    ) -> dict:
        """Aggregate YOUR OWN event log into a small report — the first call
        for "what went wrong yesterday?", "how did this run end?", or "what
        did this cost?". Counts everything, returns only totals plus the top
        offenders, so it stays cheap over a log of any size.

        - ``since`` / ``until``: ``"24h"``, ``"2d"``, ISO date, or epoch.
          Default: the whole scanned tail.
        - ``session_id``: restrict to one conversation/run — this is how you
          score how a specific run ended.
        - ``event``: restrict to one event family, e.g. ``"mcp."``.
        - ``top`` (max 50): how many entries per ranked list.

        Returns: total events, time span, ``error_like`` (count, rate, top
        failing events, top error types, and a few real samples), a ``levels``
        breakdown, the busiest events and sessions, and mirrored ``cost``
        (USD + tokens).

        Read ``error_like.from_level`` vs ``error_like.inferred`` before you
        trust the count. Entries written since the severity fix carry a real
        level (``from_level``, authoritative, counts error+warning). Older
        entries carry none, so their severity is guessed from the event name
        (``inferred``) and can over-report recovered failures. ``notes`` spells
        out which case this log is in.

        ``cost`` is mirrored from ``runtime.cost_mirrored`` events only — the
        canonical spend record is the usage_log DB table, so treat it as a
        lower bound.

        Note: the log records no per-call latency, so this cannot rank MCP
        calls by duration — it ranks them by failure and timeout counts.
        """
        if time_window_minutes is not None and since is None:
            since = f"{int(time_window_minutes)}m"
        return await handlers.logs_summary(
            since=since, until=until, session_id=session_id, event=event,
            top=top,
        )

    async def logs_context(ts: float, before: int = 8, after: int = 8) -> dict:
        """Read the events surrounding one specific log entry — what happened
        immediately before and after it. This is how you explain a failure:
        an error line rarely says why it happened, the lines before it do.

        - ``ts``: the numeric ``ts`` of an entry, copied from a ``logs_query``
          or ``logs_summary`` result (e.g. ``1783960770.0519218``). Matches the
          nearest entry at or before that instant.
        - ``before`` / ``after``: how many events either side (each max 25).

        Returns the ``anchor`` entry plus ``entries`` in chronological order.

        Typical flow: ``logs_query(errors_only=true, since="2d")`` → pick the
        failure → ``logs_context(ts=<its ts>)`` → read what led to it.
        """
        return await handlers.logs_context(ts=ts, before=before, after=after)

    return Toolkit(
        name="logs",
        tools=[logs_query, logs_summary, logs_context],
    )
