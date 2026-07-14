"""In-process ``logs`` MCP — the agent's structured query over events.jsonl.

Vision §14: "The agent itself reads [the log] as a tool to diagnose its own
behavior". Before this MCP the only path was the one DREAM_MODE_PROMPT spells
out in prose — shell out, ``find ~ -name events.jsonl``, ``tail -n 2000``.

These tests are pure-unit: each builds a synthetic ``events.jsonl`` in a temp
agent dir (via ``set_agent_dir``, the same knob ``--agent-dir`` turns) and
drives the handlers directly. No pool, no gateway, no LLM.

The synthetic log deliberately mirrors the REAL schema as it exists on disk,
verified against a 5365-entry production log:
  * ``ts`` + ``event`` on every line, and NO ``level`` key on any line —
    ``_JsonlFormatter`` drops ``record.levelname``, so severity is inferred.
  * ``session_id`` as the dominant correlation key.
  * ``error`` / ``error_type`` / ``traceback`` as the failure evidence.
If those assumptions ever break, ``t_real_schema_assumptions`` fails loudly
rather than letting the heuristics quietly rot.
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

from ._framework import TestContext, test


# ── Fixtures ─────────────────────────────────────────────────────────


@contextlib.contextmanager
def _agent_dir(ctx: TestContext, name: str) -> Iterator[Path]:
    """Point every OpenAgent path (and thus ``log_dir()``) at a temp dir.

    Restores the previous agent dir unconditionally — the suite shares one
    process and a leaked global would silently redirect every later test's
    vault/db/logs.
    """
    from src.core.paths import get_agent_dir, log_dir, set_agent_dir

    previous = get_agent_dir()
    target = ctx.test_dir / f"logs-mcp-{name}"
    target.mkdir(parents=True, exist_ok=True)
    set_agent_dir(target)
    try:
        yield log_dir()
    finally:
        set_agent_dir(previous)


def _write_log(dir_: Path, entries: list[dict[str, Any]] | str) -> Path:
    """Write a synthetic events.jsonl. Accepts raw text for corruption tests."""
    path = dir_ / "events.jsonl"
    if isinstance(entries, str):
        path.write_text(entries, encoding="utf-8")
    else:
        path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8",
        )
    return path


def _sample_entries(now: float | None = None) -> list[dict[str, Any]]:
    """A small, realistic log: two sessions, a failed scheduled task, a
    recent error, cost mirroring, and plain lifecycle noise. Oldest first,
    exactly like the append-only file on disk."""
    now = now or time.time()
    h = 3600.0
    return [
        # ~3 days ago — outside a "2d" window.
        {"ts": now - 72 * h, "event": "mcp.error", "name": "ancient", "error": "old news"},
        # ~30h ago — the scheduled task that failed.
        {"ts": now - 30 * h, "event": "scheduler.start"},
        {"ts": now - 30 * h + 1, "event": "task.fire", "name": "nightly-report", "session_id": "sess-task"},
        {"ts": now - 30 * h + 2, "event": "mcp.connect", "name": "vault", "session_id": "sess-task"},
        {"ts": now - 30 * h + 3, "event": "task.error", "name": "nightly-report",
         "session_id": "sess-task", "error_type": "TimeoutError",
         "error": "vault query exceeded 30s", "traceback": "Traceback...\nTimeoutError: vault query exceeded 30s"},
        # ~2h ago — a live chat session that cost money.
        {"ts": now - 2 * h, "event": "stream.session.start", "session_id": "sess-chat"},
        {"ts": now - 2 * h + 5, "event": "runtime.cost_mirrored", "session_id": "sess-chat",
         "model": "openai:gpt-5", "cost_usd": 0.0125, "input_tokens": 1000, "output_tokens": 250},
        {"ts": now - 2 * h + 6, "event": "runtime.cost_mirrored", "session_id": "sess-chat",
         "model": "openai:gpt-5", "cost_usd": 0.0075, "input_tokens": 500, "output_tokens": 125},
        # cost_skipped carries a cost_usd but was NOT accounted — must not be summed.
        {"ts": now - 2 * h + 7, "event": "runtime.cost_skipped", "session_id": "sess-chat",
         "model": "openai:gpt-5", "reason": "no_metrics_object", "cost_usd": 99.0},
        {"ts": now - 2 * h + 8, "event": "stream.turn.end", "session_id": "sess-chat", "errored": False},
        # ~10m ago — a fresh MCP failure.
        {"ts": now - 600, "event": "mcp.error", "name": "web-search",
         "session_id": "sess-chat", "error_type": "ConnectionError", "error": "refused"},
        {"ts": now - 590, "event": "stream.session.close", "session_id": "sess-chat"},
    ]


# ── Registration ─────────────────────────────────────────────────────


@test("logs_mcp", "logs MCP is registered as an in-process builtin")
async def t_registered(_ctx: TestContext) -> None:
    from src.mcp.builtins import BUILTIN_MCP_SPECS, resolve_builtin_entry

    assert "logs" in BUILTIN_MCP_SPECS, "logs missing from BUILTIN_MCP_SPECS"
    spec = BUILTIN_MCP_SPECS["logs"]
    assert spec.get("in_process") is True, "logs must be in-process (log_dir resolves per-agent)"
    assert spec["adapter_module"] == "src.mcp.servers.logs.adapters"
    assert spec.get("description"), "logs needs a description — tool-search shows it"

    resolved = resolve_builtin_entry("logs")
    assert resolved["name"] == "logs"
    assert resolved["in_process"] is True
    assert resolved["runtime_toolkit_factory"] == "build_runtime_toolkit"
    # In-process specs must NOT try to resolve a directory/subprocess.
    assert "command" not in resolved, f"logs must not spawn a subprocess: {resolved}"


@test("logs_mcp", "logs MCP is enabled by default")
async def t_default_on(_ctx: TestContext) -> None:
    from src.mcp.builtins import DEFAULT_MCPS

    names = [e.get("builtin") for e in DEFAULT_MCPS]
    assert "logs" in names, f"logs not in DEFAULT_MCPS: {names}"


@test("logs_mcp", "toolkit builds and exposes exactly the three logs_* tools")
async def t_toolkit_shape(_ctx: TestContext) -> None:
    from src.mcp.servers.logs.adapters import build_runtime_toolkit

    tk = build_runtime_toolkit()
    assert tk.name == "logs"
    fns = {
        **(getattr(tk, "functions", {}) or {}),
        **(getattr(tk, "async_functions", {}) or {}),
    }
    assert set(fns) == {"logs_query", "logs_summary", "logs_context"}, (
        f"unexpected tool surface: {sorted(fns)}"
    )
    # Names must be self-prefixed: in-process toolkits skip the pool's
    # _safe_prefix namespacing, so a bare `query` would collide globally.
    for name in fns:
        assert name.startswith("logs_"), f"{name} lacks the logs_ prefix"
    # A destructive prune is deliberately absent (logging.clear() exists).
    assert not any("clear" in n or "delete" in n or "prune" in n for n in fns), (
        f"logs MCP must stay read-only: {sorted(fns)}"
    )


@test("logs_mcp", "every logs_* tool yields a description + schema for tool-search")
async def t_tool_docs(_ctx: TestContext) -> None:
    """Since the v0.14 defer-all rewrite, ONLY tool-search is in the model's
    upfront tool list — these tools are reached exclusively through
    ``describe_tool``, which reads ``.description`` / ``.parameters``. A tool
    whose docstring doesn't survive that pipeline is undiscoverable.

    ``.description`` is None until ``process_entrypoint()`` runs (true of
    every in-process toolkit, vault-gate included) — the agent runtime calls
    it at registration, so we do the same here rather than asserting against
    an un-processed Function.
    """
    from src.mcp.servers.logs.adapters import build_runtime_toolkit

    tk = build_runtime_toolkit()
    fns = {
        **(getattr(tk, "functions", {}) or {}),
        **(getattr(tk, "async_functions", {}) or {}),
    }
    for name, fn in fns.items():
        fn.process_entrypoint()
        desc = getattr(fn, "description", "") or ""
        assert len(desc) > 80, f"{name} description too thin for tool-search: {desc!r}"

        props = (getattr(fn, "parameters", {}) or {}).get("properties", {})
        if name == "logs_query":
            # The filters are the tool. If one stops appearing in the schema
            # the model simply cannot use it.
            assert {"event", "session_id", "errors_only", "since", "limit"} <= set(props), (
                f"logs_query lost filters from its schema: {sorted(props)}"
            )
        if name == "logs_context":
            assert "ts" in props, f"logs_context lost its anchor param: {sorted(props)}"


# ── Schema assumptions (the heuristics rest on these) ────────────────


@test("logs_mcp", "real log schema: _JsonlFormatter persists the severity level")
async def t_real_schema_assumptions(_ctx: TestContext) -> None:
    """The tripwire, re-pointed at the new truth.

    This used to assert the OPPOSITE — that severity was dropped (0 of 5365
    real entries carried a level). That was the bug, not the design: 144 elog
    sites passed a level and the formatter discarded every one. It is fixed at
    the source now, and this guards the fix from regressing.

    If it fails because `level` vanished again, the reader's inference path
    still works (it must — every pre-fix line on disk needs it), but every
    NEW entry silently drops to guessed severity.
    """
    import logging as stdlib_logging

    from src.core.logging import _JsonlFormatter

    def emit(level: int, **data: Any) -> dict:
        record = stdlib_logging.LogRecord(
            name="openagent.events", level=level, pathname=__file__,
            lineno=1, msg="some.event", args=(), exc_info=None,
        )
        record.event_data = data  # type: ignore[attr-defined]
        return json.loads(_JsonlFormatter().format(record))

    emitted = emit(stdlib_logging.ERROR, error="boom")
    assert emitted.get("level") == "error", (
        f"events.jsonl must persist a lowercased severity level: {emitted}"
    )
    assert emitted["event"] == "some.event" and "ts" in emitted
    assert emitted["error"] == "boom", "event_data must still be splatted in"

    # Lowercased across the whole elog range — the reader compares lowercase.
    assert emit(stdlib_logging.WARNING)["level"] == "warning"
    assert emit(stdlib_logging.INFO)["level"] == "info"
    assert emit(stdlib_logging.DEBUG)["level"] == "debug"

    # Key order is {when, what, how-bad, details} — level before the splat.
    assert list(emit(stdlib_logging.INFO, a=1))[:4] == ["ts", "event", "level", "a"]


@test("logs_mcp", "elog's `level` can never be shadowed by event data")
async def t_level_cannot_collide(_ctx: TestContext) -> None:
    """`level` sits BEFORE `**event_data` in the JSONL entry, so a `level`
    key inside event_data would silently overwrite the real severity. It
    cannot: `elog` declares `level` as a named parameter, so it is captured
    by the signature and `**data` never sees it. This pins that reasoning —
    if someone reorders elog's signature, the formatter's key order becomes
    a real bug rather than a style choice.
    """
    import logging as stdlib_logging
    from unittest.mock import patch

    from src.core.logging import EVENT_LOGGER, elog

    captured: list[stdlib_logging.LogRecord] = []

    class _Capture(stdlib_logging.Handler):
        def emit(self, record: stdlib_logging.LogRecord) -> None:
            captured.append(record)

    logger = stdlib_logging.getLogger(EVENT_LOGGER)
    handler = _Capture()
    logger.addHandler(handler)
    # Don't let a deliberate ERROR-level probe reach the root console handler
    # and print scary noise in the middle of a passing test run.
    propagate = logger.propagate
    logger.propagate = False
    try:
        with patch("src.core.logging._configured", True):
            elog("t.collide", level="error", note="hi")
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate

    assert captured, "elog did not reach the events logger"
    record = captured[-1]
    data = getattr(record, "event_data", {})
    assert "level" not in data, f"`level` leaked into event_data: {data}"
    assert record.levelname == "ERROR", "level= must set the record's severity"
    assert data == {"note": "hi"}


# ── logs_query ───────────────────────────────────────────────────────


@test("logs_mcp", "logs_query returns the tail as structured, chronological rows")
async def t_query_tail(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "tail") as d:
        _write_log(d, _sample_entries())
        res = await handlers.logs_query()

    assert res["returned"] == 12, f"expected all 12 entries, got {res['returned']}"
    rows = res["entries"]
    ts_list = [r["ts"] for r in rows]
    assert ts_list == sorted(ts_list), "entries must be oldest→newest (read_tail's contract)"
    assert rows[-1]["event"] == "stream.session.close", f"last row wrong: {rows[-1]}"
    # Every row is model-readable: ISO time alongside the raw anchor ts.
    assert all("time" in r and "event" in r for r in rows)
    assert res["scan"]["lines_scanned"] == 12
    assert res["scan"]["corrupt_lines_skipped"] == 0


@test("logs_mcp", "logs_query filters by event, session_id, errors_only, and time window")
async def t_query_filters(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "filters") as d:
        _write_log(d, _sample_entries())

        by_event = await handlers.logs_query(event="mcp.")
        by_session = await handlers.logs_query(session_id="sess-task")
        errors = await handlers.logs_query(errors_only=True)
        recent = await handlers.logs_query(since="2d")
        substring = await handlers.logs_query(event="cost")
        text = await handlers.logs_query(contains="nightly-report")

    assert {r["event"] for r in by_event["entries"]} == {"mcp.error", "mcp.connect"}
    assert all(r["session_id"] == "sess-task" for r in by_session["entries"])
    assert by_session["returned"] == 3, f"sess-task has 3 events, got {by_session['returned']}"

    err_events = [r["event"] for r in errors["entries"]]
    assert "task.error" in err_events and "mcp.error" in err_events
    assert "mcp.connect" not in err_events, f"clean events leaked into errors_only: {err_events}"
    assert "stream.session.start" not in err_events

    # since="2d" must exclude the 72h-old entry.
    assert not any(r.get("name") == "ancient" for r in recent["entries"]), \
        "since=2d leaked a 3-day-old entry"
    assert recent["window"]["since"], "window.since should echo the resolved bound"

    # Substring matching: a model asks for "cost", the events are
    # runtime.cost_*. Prefix-only (read_tail / GET /api/logs) finds nothing.
    assert {r["event"] for r in substring["entries"]} == {
        "runtime.cost_mirrored", "runtime.cost_skipped",
    }
    assert text["returned"] == 2, f"contains= should find both nightly-report rows: {text}"


@test("logs_mcp", "logs_query pages with limit/offset and reports what it withheld")
async def t_query_paging(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "paging") as d:
        _write_log(d, _sample_entries())
        page1 = await handlers.logs_query(session_id="sess-chat", limit=2)
        page2 = await handlers.logs_query(session_id="sess-chat", limit=2, offset=2)

    assert page1["returned"] == 2 and page2["returned"] == 2
    assert page1["matched_in_scan"] == 7, f"sess-chat has 7 events: {page1['matched_in_scan']}"
    assert "hint" in page1, "a partial page must tell the model more exists"
    # offset walks backwards in time from the newest match — no overlap.
    assert not ({r["ts"] for r in page1["entries"]} & {r["ts"] for r in page2["entries"]}), \
        "paged results overlap"


# ── Token discipline ─────────────────────────────────────────────────


@test("logs_mcp", "logs_query clamps limit to the hard cap")
async def t_limit_capped(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers
    from src.mcp.servers.logs.handlers import _MAX_LIMIT

    now = time.time()
    many = [
        {"ts": now - i, "event": "noise.tick", "i": i}
        for i in range(_MAX_LIMIT * 3, 0, -1)
    ]
    with _agent_dir(ctx, "cap") as d:
        _write_log(d, many)
        res = await handlers.logs_query(limit=100_000)
        junk = await handlers.logs_query(limit="all")  # type: ignore[arg-type]

    assert res["returned"] == _MAX_LIMIT, (
        f"limit must clamp to {_MAX_LIMIT}, got {res['returned']}"
    )
    assert junk["returned"] == 50, f"garbage limit should fall back to default: {junk['returned']}"


@test("logs_mcp", "a single fat field cannot dominate a result")
async def t_value_truncation(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers
    from src.mcp.servers.logs.reader import _MAX_VALUE_CHARS

    now = time.time()
    with _agent_dir(ctx, "fatfield") as d:
        _write_log(d, [{
            "ts": now, "event": "mcp.error", "error": "E" * 50_000,
            "traceback": "boring\n" * 5_000 + "ValueError: THE ACTUAL CAUSE",
            "payload": {"blob": ["x" * 2_000 for _ in range(50)]},
            "cost_usd": 0.25, "input_tokens": 10,
        }])
        res = await handlers.logs_query()

    row = res["entries"][0]
    assert len(row["error"]) < _MAX_VALUE_CHARS + 100, f"error field not clamped: {len(row['error'])}"
    assert "trimmed" in row["error"]
    # A traceback keeps its TAIL — the exception line is the useful part.
    assert row["traceback"].endswith("ValueError: THE ACTUAL CAUSE"), \
        f"traceback must keep its tail: {row['traceback'][-80:]!r}"
    # Container values must be clamped by weight, not by len() of their keys.
    assert len(json.dumps(row["payload"])) < _MAX_VALUE_CHARS + 100, \
        "dict/list field escaped the cap"
    # Scalars stay scalars so the model can do arithmetic on them.
    assert row["cost_usd"] == 0.25 and row["input_tokens"] == 10


@test("logs_mcp", "whole-payload budget holds even when every row is reasonable")
async def t_payload_budget(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers
    from src.mcp.servers.logs.handlers import _MAX_RESULT_CHARS

    now = time.time()
    # 200 rows that each pass the per-value cap but together blow the budget.
    fat = [
        {"ts": now - i, "event": "mcp.error", "session_id": f"s{i}",
         "error": "E" * 280, "detail": "D" * 280, "extra": "X" * 280}
        for i in range(200, 0, -1)
    ]
    with _agent_dir(ctx, "budget") as d:
        _write_log(d, fat)
        res = await handlers.logs_query(limit=200)

    size = len(json.dumps(res, default=str))
    assert size <= _MAX_RESULT_CHARS + 2_000, (
        f"payload {size} chars blew the {_MAX_RESULT_CHARS} budget. The global "
        f"cap_tool_output backstop would still bound this at 50k chars (~12.5k "
        f"tokens, re-sent every step) by truncating the repr mid-structure — "
        f"that is a crash guard, not a token budget"
    )
    assert res.get("result_truncated") is True
    assert "hint" in res
    # What survives a squeeze must be the NEWEST slice — a diagnosis reads
    # backwards from now.
    assert res["entries"][-1]["ts"] == fat[-1]["ts"], "budget trim kept the wrong end"


@test("logs_mcp", "reader stops at the byte cap and says so")
async def t_scan_cap(ctx: TestContext) -> None:
    from src.mcp.servers.logs import reader

    now = time.time()
    with _agent_dir(ctx, "scancap") as d:
        path = _write_log(d, [
            {"ts": now - i, "event": "noise.tick", "pad": "p" * 200, "i": i}
            for i in range(5_000, 0, -1)
        ])
        assert path.stat().st_size > 300_000, "fixture too small to prove the cap"

        stats = reader.ScanStats()
        seen = list(reader.iter_entries_reverse(
            stats=stats, max_bytes=50_000, path=path,
        ))

    assert stats.bytes <= 50_000, f"scan overran its cap: {stats.bytes}"
    assert stats.hit_scan_cap is True, "hit_scan_cap must flag a partial view"
    assert 0 < len(seen) < 5_000, f"expected a partial read, got {len(seen)}"
    # Never manufacture parse errors: the line straddling the cap boundary is
    # dropped as truncated, not reported as corrupt.
    assert stats.corrupt == 0, f"scan boundary produced fake corruption: {stats.corrupt}"


# ── Reverse reader correctness ───────────────────────────────────────


@test("logs_mcp", "reverse block reader is exact across block boundaries")
async def t_reverse_reader_exact(ctx: TestContext) -> None:
    """The reader stitches 64 KB blocks backwards; a partial line at every
    boundary is the thing most likely to be silently dropped or duplicated.
    Only a file spanning many blocks exercises it.
    """
    from src.mcp.servers.logs import reader

    now = time.time()
    entries = [
        {"ts": now - (10_000 - i), "event": f"e.{i}", "pad": "z" * (i % 97), "i": i}
        for i in range(10_000)
    ]
    with _agent_dir(ctx, "blocks") as d:
        path = _write_log(d, entries)
        assert path.stat().st_size > reader._BLOCK_BYTES * 3, "fixture must span blocks"
        stats = reader.ScanStats()
        got = list(reader.iter_entries_reverse(stats=stats, path=path, max_bytes=50_000_000))

    assert len(got) == len(entries), f"lost/duplicated lines: {len(got)} vs {len(entries)}"
    assert stats.corrupt == 0, f"block stitching corrupted lines: {stats.corrupt}"
    assert [e["i"] for e in got] == list(range(9_999, -1, -1)), "reverse order broken"


@test("logs_mcp", "since= stops the scan early instead of reading the whole log")
async def t_since_short_circuits(ctx: TestContext) -> None:
    """The append-only log is ts-ordered, so the first out-of-window entry
    means every remaining byte is older. 'What broke in the last hour?' must
    not touch six days of history.
    """
    from src.mcp.servers.logs import handlers

    now = time.time()
    entries = [{"ts": now - 86_400 * 5 + i, "event": "old.noise", "pad": "q" * 300}
               for i in range(4_000)]
    entries.append({"ts": now - 60, "event": "mcp.error", "error": "recent"})
    with _agent_dir(ctx, "shortcircuit") as d:
        path = _write_log(d, entries)
        full_size = path.stat().st_size
        res = await handlers.logs_query(since="1h")

    assert res["returned"] == 1 and res["entries"][0]["error"] == "recent"
    assert res["scan"]["lines_scanned"] <= 3, (
        f"since=1h scanned {res['scan']['lines_scanned']} lines — it must stop "
        f"at the first entry older than the window"
    )
    assert res["scan"]["bytes_scanned"] < full_size / 10, (
        f"read {res['scan']['bytes_scanned']} of {full_size} bytes despite a 1h window"
    )


@test("logs_mcp", "parse_time accepts relative ages, ISO dates, and epochs")
async def t_parse_time(_ctx: TestContext) -> None:
    from src.mcp.servers.logs.reader import parse_time

    now = 1_800_000_000.0
    assert parse_time("24h", now=now) == now - 86_400
    assert parse_time("90m", now=now) == now - 5_400
    assert parse_time("7d", now=now) == now - 604_800
    assert parse_time("30s", now=now) == now - 30
    assert parse_time("2w", now=now) == now - 1_209_600
    assert parse_time(None) is None and parse_time("") is None
    assert parse_time(1_783_960_770.05) == 1_783_960_770.05
    assert parse_time("1783960770.05") == 1_783_960_770.05
    assert parse_time("2026-07-13T00:00:00+00:00") == 1_783_900_800.0

    # A silently-ignored bound would answer confidently for the wrong window.
    for bad in ("yesterday", "last tuesday", "24 hours"):
        try:
            parse_time(bad)
        except ValueError as e:
            assert "24h" in str(e), f"error must show accepted forms: {e}"
        else:
            raise AssertionError(f"parse_time({bad!r}) should raise, not guess")


# ── logs_summary ─────────────────────────────────────────────────────


@test("logs_mcp", "logs_summary answers 'what went wrong yesterday?' in one call")
async def t_summary(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "summary") as d:
        _write_log(d, _sample_entries())
        res = await handlers.logs_summary(since="2d")

    assert res["total_events"] == 11, f"expected 11 in-window events: {res['total_events']}"
    errors = res["error_like"]
    assert errors["count"] == 2, f"task.error + mcp.error expected: {errors}"
    assert 0 < errors["rate"] < 1
    top_err = {e["name"] for e in errors["by_event"]}
    assert top_err == {"task.error", "mcp.error"}, f"top failing events wrong: {top_err}"
    assert {e["name"] for e in errors["by_error_type"]} == {"TimeoutError", "ConnectionError"}
    assert errors["samples"], "summary must carry real samples, not just counts"
    assert len(errors["samples"]) <= 5, "samples must stay bounded"

    assert res["sessions"]["distinct"] == 2
    assert res["window"]["span_hours"] and res["window"]["span_hours"] > 24
    assert res["distinct_events"] >= 6
    # This fixture is pre-fix (no level anywhere), so every verdict is a
    # guess and the caveat must say exactly that, in-band.
    assert errors["inferred"] == 2 and errors["from_level"] == 0
    assert any("INFERRED for every entry" in n for n in res["notes"]), res["notes"]
    assert res["levels"]["entries_without_level"] == 11


@test("logs_mcp", "logs_summary aggregates mirrored cost and ignores skipped cost")
async def t_summary_cost(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "cost") as d:
        _write_log(d, _sample_entries())
        res = await handlers.logs_summary(session_id="sess-chat")

    cost = res["cost"]
    assert abs(cost["total_usd"] - 0.02) < 1e-9, (
        f"expected 0.0125+0.0075=0.02, got {cost['total_usd']} — a "
        f"runtime.cost_skipped entry (99.0) must NOT be summed"
    )
    assert cost["accounted_calls"] == 2
    assert cost["input_tokens"] == 1_500 and cost["output_tokens"] == 375
    assert "usage_log" in cost["note"], "must name the canonical spend record"
    # session_id scoping is what makes 'how did run X end / cost?' answerable.
    assert res["total_events"] == 7, f"sess-chat has 7 events: {res['total_events']}"


@test("logs_mcp", "logs_summary output stays small over a large log")
async def t_summary_bounded(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers
    from src.mcp.servers.logs.handlers import _MAX_RESULT_CHARS

    now = time.time()
    # 6k events across 3k distinct names and 3k sessions — every ranked list
    # in the summary has a long tail to blow up on.
    entries = [
        {"ts": now - (6_000 - i), "event": f"evt.{i % 3_000}",
         "session_id": f"sess-{i % 3_000}", "error": "boom" if i % 2 else None}
        for i in range(6_000)
    ]
    with _agent_dir(ctx, "bigsummary") as d:
        _write_log(d, entries)
        res = await handlers.logs_summary(top=10_000)

    size = len(json.dumps(res, default=str))
    assert size < _MAX_RESULT_CHARS, f"summary grew to {size} chars over a big log"
    assert len(res["by_event"]) <= 50, f"top must clamp: {len(res['by_event'])}"
    assert len(res["sessions"]["busiest"]) <= 50
    # Counts still reflect everything scanned, not just what was listed.
    assert res["total_events"] == 6_000
    assert res["distinct_events"] == 3_000
    assert res["sessions"]["distinct"] == 3_000


# ── Mixed schema: the level fix applies forward only ─────────────────


def _new_schema_entries(now: float | None = None) -> list[dict[str, Any]]:
    """Entries as written AFTER the formatter fix — every line has a level."""
    now = now or time.time()
    return [
        {"ts": now - 300, "event": "stream.session.start", "level": "info",
         "session_id": "sess-new"},
        # A real failure, declared by the call site.
        {"ts": now - 240, "event": "mcp.error", "level": "warning",
         "session_id": "sess-new", "error_type": "ConnectionError", "error": "refused"},
        {"ts": now - 230, "event": "task.error", "level": "error",
         "session_id": "sess-new", "error_type": "TimeoutError", "error": "too slow"},
        # An `error` field at INFO. The old heuristic called this a failure;
        # the persisted level says it is not. This is the fix working.
        {"ts": now - 220, "event": "agent.media.read_skip", "level": "info",
         "session_id": "sess-new", "error": "unreadable frame, skipped"},
        # A barge-in: errored=True AND cancelled=True. 230/230 of these in a
        # real log were interrupts, not faults (§2 makes barge-in first-class).
        {"ts": now - 210, "event": "stream.turn.end", "level": "info",
         "session_id": "sess-new", "errored": True, "cancelled": True},
        # A genuinely errored turn (NOT cancelled) logged at info: structured
        # evidence must outrank the permissive level.
        {"ts": now - 200, "event": "stream.turn.end", "level": "info",
         "session_id": "sess-new", "errored": True, "cancelled": False},
    ]


@test("logs_mcp", "post-fix entries get authoritative severity, not a guess")
async def t_authoritative_severity(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "authoritative") as d:
        _write_log(d, _new_schema_entries())
        res = await handlers.logs_summary()
        errs = await handlers.logs_query(errors_only=True)

    e = res["error_like"]
    assert e["from_level"] == 3 and e["inferred"] == 0, (
        f"every entry has a level, so nothing may be inferred: {e}"
    )
    # mcp.error(warning) + task.error(error) + the errored non-cancelled turn.
    assert e["count"] == 3, f"expected 3 failures: {[r['event'] for r in errs['entries']]}"
    assert any("authoritative" in n for n in res["notes"]), res["notes"]
    assert res["levels"]["entries_without_level"] == 0
    assert {x["name"]: x["count"] for x in res["levels"]["by_level"]} == {
        "info": 4, "warning": 1, "error": 1,
    }

    events = [r["event"] for r in errs["entries"]]
    # The fix in action: an `error` field at info level is NOT a failure.
    assert "agent.media.read_skip" not in events, (
        "a declared info-level entry must not be reported as a failure"
    )
    # And a barge-in is not a failure either — but a real errored turn is.
    assert events.count("stream.turn.end") == 1, f"barge-in leaked in: {events}"
    assert "mcp.error" in events and "task.error" in events


@test("logs_mcp", "a barge-in is not a failure, in either schema")
async def t_barge_in_not_a_failure(_ctx: TestContext) -> None:
    """Measured: 230 of 230 `errored=True` entries in a real log were also
    `cancelled=True` — every one a user interrupting the agent mid-sentence,
    which §2 calls first-class behaviour. Counting them made barge-ins the
    single largest contributor to "what went wrong yesterday".
    """
    from src.mcp.servers.logs.reader import classify

    barge_in = {"event": "stream.turn.end", "errored": True, "cancelled": True}
    real_fail = {"event": "stream.turn.end", "errored": True, "cancelled": False}

    # Pre-fix (no level): the heuristic must exclude the barge-in.
    assert classify(barge_in) == (False, True)
    assert classify(real_fail) == (True, True)
    # Post-fix, at a permissive level: same verdicts, now authoritative.
    assert classify({**barge_in, "level": "info"}) == (False, False)
    assert classify({**real_fail, "level": "info"}) == (True, False)


@test("logs_mcp", "a mixed-schema log reports the authoritative/inferred split")
async def t_mixed_schema(ctx: TestContext) -> None:
    """The real transition state: dream mode keeps ~6 days, so a log spans
    the fix. The count must never blend the two silently.
    """
    from src.mcp.servers.logs import handlers

    now = time.time()
    old = [
        {"ts": now - 7200, "event": "mcp.error", "error": "old failure"},
        {"ts": now - 7100, "event": "stream.session.start", "session_id": "s-old"},
    ]
    with _agent_dir(ctx, "mixed") as d:
        _write_log(d, old + _new_schema_entries(now))
        res = await handlers.logs_summary()

    e = res["error_like"]
    assert e["from_level"] == 3, f"3 post-fix failures: {e}"
    assert e["inferred"] == 1, f"1 pre-fix failure (the old mcp.error): {e}"
    assert e["count"] == 4
    assert res["levels"]["entries_with_level"] == 6
    assert res["levels"]["entries_without_level"] == 2

    note = " ".join(res["notes"])
    assert "mixes two schemas" in note, note
    # "some of this is a guess" is useless without "how much".
    assert "3" in note and "1" in note, f"note must name the split: {note}"


@test("logs_mcp", "level filter matches real levels and never hides what it can't judge")
async def t_level_filter(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    now = time.time()
    old = [{"ts": now - 7200, "event": "mcp.error", "error": "old, level-less"}]
    with _agent_dir(ctx, "levelfilter") as d:
        _write_log(d, old + _new_schema_entries(now))
        warn = await handlers.logs_query(level="warning")
        err = await handlers.logs_query(level="error")
        info = await handlers.logs_query(level="INFO")  # case-insensitive
        clean = await handlers.logs_query(errors_only=True)

    assert [r["event"] for r in warn["entries"]] == ["mcp.error"]
    assert [r["event"] for r in err["entries"]] == ["task.error"]
    assert info["returned"] == 4

    # The blind spot must be LOUD: the pre-fix entry can never match a level
    # query, and answering "no warnings" without saying so would be a lie.
    assert warn["entries_without_level_skipped"] == 1, warn
    assert "predate the severity fix" in warn["level_filter_note"]
    assert "errors_only" in warn["level_filter_note"], "must point at the way out"

    # errors_only spans both schemas — that's the escape hatch the note names.
    assert clean.get("entries_without_level_skipped") is None
    assert "old, level-less" in json.dumps(clean["entries"]), \
        "errors_only must still catch pre-fix failures"


@test("logs_mcp", "level filter rejects a level elog cannot emit")
async def t_level_filter_validation(ctx: TestContext) -> None:
    """Answering '0 results' to a typo is how a model concludes nothing went
    wrong. `critical` is the trap: a real logging level, but elog's _LEVELS
    map cannot emit it, so it would match nothing forever.
    """
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "levelbad") as d:
        _write_log(d, _new_schema_entries())
        for bad in ("critical", "err", "ERRORS", "fatal"):
            try:
                await handlers.logs_query(level=bad)
            except ValueError as e:
                assert "error" in str(e) and "errors_only" in str(e), (
                    f"error must list valid levels and the fallback: {e}"
                )
            else:
                raise AssertionError(f"logs_query(level={bad!r}) should raise")


# ── logs_context ─────────────────────────────────────────────────────


@test("logs_mcp", "logs_context explains a failure with its surrounding events")
async def t_context(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    entries = _sample_entries()
    failure_ts = next(e["ts"] for e in entries if e["event"] == "task.error")

    with _agent_dir(ctx, "context") as d:
        _write_log(d, entries)
        res = await handlers.logs_context(ts=failure_ts, before=2, after=2)

    assert res["found"] is True
    assert res["anchor"]["event"] == "task.error"
    names = [r["event"] for r in res["entries"]]
    # This is the whole point: the error line says "TimeoutError", the lines
    # before it say the task fired and which MCP it was talking to.
    assert names == [
        "task.fire", "mcp.connect", "task.error",
        "stream.session.start", "runtime.cost_mirrored",
    ], f"context window wrong: {names}"
    ts_list = [r["ts"] for r in res["entries"]]
    assert ts_list == sorted(ts_list), "context must read chronologically"


@test("logs_mcp", "logs_context clamps its window and handles a missing anchor")
async def t_context_edges(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers
    from src.mcp.servers.logs.handlers import _MAX_CONTEXT

    entries = _sample_entries()
    now = time.time()
    with _agent_dir(ctx, "ctxedge") as d:
        _write_log(d, entries)
        wide = await handlers.logs_context(ts=entries[-1]["ts"], before=10_000, after=10_000)
        future = await handlers.logs_context(ts=now + 86_400)
        past = await handlers.logs_context(ts=1.0)
        none_side = await handlers.logs_context(ts=entries[4]["ts"], before=0, after=0)

    assert len(wide["entries"]) <= _MAX_CONTEXT * 2 + 1, "context window must clamp"
    # An anchor in the future resolves to the newest entry at-or-before it —
    # nearest-older is the useful reading, not an error.
    assert future["found"] is True and future["anchor"]["event"] == "stream.session.close"
    # Nothing at or before ts=1 → an explicit miss, not a crash or a wrong row.
    assert past["found"] is False and "error" in past
    # before=0/after=0 must yield the anchor alone (a -0 slice bug would
    # silently return everything).
    assert len(none_side["entries"]) == 1
    assert none_side["entries"][0]["event"] == none_side["anchor"]["event"]


@test("logs_mcp", "logs_context rejects a non-numeric anchor with an actionable error")
async def t_context_bad_anchor(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "ctxbad") as d:
        _write_log(d, _sample_entries())
        for bad in ("yesterday", None, "not-a-ts"):
            try:
                await handlers.logs_context(ts=bad)  # type: ignore[arg-type]
            except ValueError as e:
                assert "logs_query" in str(e), f"error must point at the source of ts: {e}"
            else:
                raise AssertionError(f"logs_context(ts={bad!r}) should raise")


# ── Degradation ──────────────────────────────────────────────────────


@test("logs_mcp", "every tool degrades cleanly when the log is missing")
async def t_missing_log(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "missing") as d:
        assert not (d / "events.jsonl").exists()
        q = await handlers.logs_query()
        s = await handlers.logs_summary()
        c = await handlers.logs_context(ts=time.time())

    # A fresh agent has no log yet. Empty is the truth; an exception here
    # would be the agent unable to answer "anything wrong?" on day one.
    assert q["entries"] == [] and q["returned"] == 0
    assert s["total_events"] == 0 and s["error_like"]["count"] == 0
    assert s["error_like"]["rate"] == 0.0, "rate must not divide by zero"
    assert s["cost"]["total_usd"] == 0.0
    assert c["found"] is False


@test("logs_mcp", "every tool degrades cleanly on an empty log")
async def t_empty_log(ctx: TestContext) -> None:
    from src.mcp.servers.logs import handlers

    with _agent_dir(ctx, "empty") as d:
        _write_log(d, "")
        q = await handlers.logs_query()
        s = await handlers.logs_summary()

    assert q["returned"] == 0 and s["total_events"] == 0


@test("logs_mcp", "corrupt lines are skipped, counted, and never break a query")
async def t_corrupt_log(ctx: TestContext) -> None:
    """A half-written tail line is NORMAL for an append-only log killed
    mid-write — and right after a crash is exactly when you read the log.
    """
    from src.mcp.servers.logs import handlers

    now = time.time()
    raw = "\n".join([
        json.dumps({"ts": now - 30, "event": "good.one", "session_id": "s1"}),
        "{not json at all",
        "",
        "[1, 2, 3]",                      # valid JSON, wrong shape
        "null",
        json.dumps({"ts": now - 20, "event": "mcp.error", "error": "real failure"}),
        '{"ts": 123, "event": "half-writ',  # truncated tail, no newline
    ])
    with _agent_dir(ctx, "corrupt") as d:
        _write_log(d, raw)
        q = await handlers.logs_query()
        s = await handlers.logs_summary()

    events = [r["event"] for r in q["entries"]]
    assert events == ["good.one", "mcp.error"], f"good rows lost: {events}"
    assert q["scan"]["corrupt_lines_skipped"] == 4, (
        f"expected 4 unusable lines counted: {q['scan']}"
    )
    assert s["total_events"] == 2 and s["error_like"]["count"] == 1


@test("logs_mcp", "entries without a ts are tolerated")
async def t_missing_ts(ctx: TestContext) -> None:
    """`_JsonlFormatter` always writes ts, but the file is plain text a human
    may have edited, and a ts-less line must not poison the window logic.
    """
    from src.mcp.servers.logs import handlers

    now = time.time()
    with _agent_dir(ctx, "nots") as d:
        _write_log(d, [
            {"event": "no.ts.here", "note": "hand-edited"},
            {"ts": "not-a-number", "event": "bad.ts"},
            {"ts": now - 10, "event": "mcp.error", "error": "boom"},
        ])
        q = await handlers.logs_query()
        s = await handlers.logs_summary(since="1h")

    assert q["returned"] == 3, f"ts-less rows must still be listed: {q['returned']}"
    assert s["total_events"] >= 1 and s["error_like"]["count"] == 1


# ── media-gen description (same-file fix) ────────────────────────────


@test("logs_mcp", "media-gen description matches the tools it actually implements")
async def t_media_gen_description(_ctx: TestContext) -> None:
    """The spec claimed "images, audio, or video"; the server only ever
    registered generate_image + generate_video. tool-search surfaces this
    text as the MCP's pitch, so a phantom audio tool costs the model a turn.

    Asserted against the SERVER MODULE, not a hardcoded list, so the
    description and the implementation cannot drift apart again.
    """
    import inspect

    from src.mcp.builtins import BUILTIN_MCP_SPECS
    from src.mcp.servers.media_gen import server

    tools = {
        name for name, _ in inspect.getmembers(server, inspect.isfunction)
        if name.startswith("generate_")
    }
    assert tools == {"generate_image", "generate_video"}, (
        f"media-gen tool surface changed to {sorted(tools)} — update the "
        f"description in builtins.py to match"
    )

    desc = BUILTIN_MCP_SPECS["media-gen"]["description"]
    assert "audio" not in desc.lower(), f"description still promises audio: {desc!r}"
    assert "image" in desc.lower() and "video" in desc.lower(), f"description lost a real capability: {desc!r}"
