"""``read_tail`` on the shared reverse reader + ``GET /api/logs`` off the loop.

Two defects, one file — they share a cause:

  1. ``read_tail`` did ``read_text().splitlines()``: the *entire* events.jsonl
     into memory, plus a list of every line, before filtering anything. Its
     only REST caller (``GET /api/logs``) is an ``async`` handler, so on a real
     log (measured 728 KB / 5365 entries on a light install; dream mode trims
     the file by age, not size) that ran on the gateway's event loop — the same
     loop carrying live WebSocket streams and voice audio.
  2. The ``logs`` MCP had already written a bounded reverse reader for exactly
     this file, in its own module, because ``logging.py`` offered nothing
     usable. Two readers of one format is a drift we pay for elsewhere.

So the reader moved into ``src.core.logging`` (which owns the format) and
``read_tail`` was reimplemented on it. The whole risk of that move is a
*silent* change to ``read_tail``'s observable contract, so most of this file is
an equivalence proof against the implementation it replaced — including the
one case the MCP's reader would have broken if its 2 MB scan cap had come
along as a default: a prefix filter whose only matches are ancient.

Pure-unit: synthetic logs in a temp agent dir, handlers driven directly. No
gateway, no pool, no LLM.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from aiohttp.test_utils import make_mocked_request

from ._framework import TestContext, test


# ── fixtures ─────────────────────────────────────────────────────────


@contextlib.contextmanager
def _agent_dir(ctx: TestContext, name: str) -> Iterator[Path]:
    """Point every OpenAgent path (and thus ``log_dir()``) at a temp dir.

    Restores the previous agent dir unconditionally — the suite shares one
    process and a leaked global would silently redirect every later test's
    vault/db/logs.
    """
    from src.core.paths import get_agent_dir, log_dir, set_agent_dir

    previous = get_agent_dir()
    target = ctx.test_dir / f"read-tail-{name}"
    target.mkdir(parents=True, exist_ok=True)
    set_agent_dir(target)
    try:
        yield log_dir()
    finally:
        set_agent_dir(previous)


def _read_tail_before(path: Path, lines: int = 100,
                      event_filter: str | None = None) -> list[dict[str, Any]]:
    """The pre-change ``read_tail``, verbatim, as an oracle.

    Copied deliberately rather than imported: it is the *old* behaviour we are
    pinning, so it must not follow the implementation it is checking. If a
    future change to ``read_tail`` is intentional, this fails and someone has
    to say so out loud.
    """
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    out: list[dict[str, Any]] = []
    for line in reversed(raw):
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if event_filter and not entry.get("event", "").startswith(event_filter):
            continue
        out.append(entry)
        if len(out) >= lines:
            break
    out.reverse()
    return out


def _corpus(now: float) -> list[dict[str, Any]]:
    """A log spanning both schemas, oldest first — like the file on disk.

    Mixed ``level`` presence is the realistic case: ``_JsonlFormatter`` only
    started persisting ``record.levelname`` recently (0 of 5365 entries on a
    real log had one), and dream mode keeps ~6 days of the old shape, so any
    reader sees both within a single window.
    """
    h = 3600.0
    return [
        # Pre-fix schema: no `level` key at all.
        {"ts": now - 90 * h, "event": "mcp.error", "name": "ancient", "error": "old news"},
        {"ts": now - 80 * h, "event": "scheduler.start"},
        {"ts": now - 70 * h, "event": "task.fire", "name": "nightly", "session_id": "s1"},
        # An entry with no `event` key — `.get("event", "")` must not match a filter.
        {"ts": now - 60 * h, "session_id": "s1", "note": "eventless"},
        # Post-fix schema: `level` persisted.
        {"ts": now - 2 * h, "event": "stream.session.start", "level": "info", "session_id": "s2"},
        {"ts": now - 1 * h, "event": "mcp.error", "level": "warning",
         "session_id": "s2", "error_type": "ConnectionError", "error": "refused"},
        {"ts": now - 600, "event": "stream.turn.end", "level": "info",
         "session_id": "s2", "errored": False},
        {"ts": now - 60, "event": "task.error", "level": "error", "name": "nightly",
         "traceback": "Traceback...\nTimeoutError: boom"},
    ]


def _write(dir_: Path, entries: list[dict[str, Any]] | str) -> Path:
    path = dir_ / "events.jsonl"
    if isinstance(entries, str):
        path.write_text(entries, encoding="utf-8")
    else:
        path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8",
        )
    return path


# ── the contract: identical to the slurp it replaced ─────────────────


@test("logs_read_tail", "read_tail matches the slurping implementation it replaced")
async def t_contract_equivalence(ctx: TestContext) -> None:
    from src.core.logging import read_tail

    now = time.time()
    with _agent_dir(ctx, "equiv") as log_dir_:
        path = _write(log_dir_, _corpus(now))

        # Every axis the contract has: how many, and which prefix. `lines=0`
        # is in on purpose — the old loop appended before it checked, so it
        # returned ONE entry, and "obviously 0" would be a silent change.
        cases: list[tuple[int, str | None]] = [
            (100, None), (3, None), (1, None), (0, None), (999, None),
            (100, "mcp."), (100, "mcp.error"), (1, "mcp."),
            (100, "task."), (100, "stream."),
            (100, "nope."),          # matches nothing
            (100, ""),               # falsy filter → same as no filter
            (100, "mcp.error.deep"), # longer than any event name
        ]
        for lines, filt in cases:
            got = read_tail(lines=lines, event_filter=filt)
            want = _read_tail_before(path, lines=lines, event_filter=filt)
            assert got == want, (
                f"read_tail(lines={lines}, event_filter={filt!r}) diverged from "
                f"the old implementation:\n  new={got}\n  old={want}"
            )

        # Spot-check the shape itself, so an oracle that was wrong in the same
        # way as the code cannot make this test vacuous.
        entries = read_tail(lines=100)
        assert len(entries) == 8, entries
        ts_list = [e["ts"] for e in entries]
        assert ts_list == sorted(ts_list), "entries must be oldest→newest"
        assert entries[-1]["event"] == "task.error", entries[-1]
        # Mixed schema passes through verbatim — no invented levels.
        assert "level" not in entries[0], entries[0]
        assert entries[-1]["level"] == "error", entries[-1]


@test("logs_read_tail", "read_tail's edge cases are unchanged: corrupt, blank, missing, empty")
async def t_contract_edges(ctx: TestContext) -> None:
    from src.core.logging import read_tail

    # Missing file → [] (not an exception).
    with _agent_dir(ctx, "missing") as log_dir_:
        assert not (log_dir_ / "events.jsonl").exists()
        assert read_tail(lines=10) == []
        assert read_tail(lines=10) == _read_tail_before(log_dir_ / "events.jsonl")

    # Empty file → [].
    with _agent_dir(ctx, "empty") as log_dir_:
        path = _write(log_dir_, "")
        assert read_tail(lines=10) == []
        assert read_tail(lines=10) == _read_tail_before(path)

    # Corrupt + blank lines + a half-written tail (killed mid-write) + no
    # trailing newline. A crash is exactly when you most want to read the log.
    with _agent_dir(ctx, "corrupt") as log_dir_:
        raw = (
            '{"ts": 1.0, "event": "a.first"}\n'
            "\n"
            "not json at all\n"
            '{"ts": 2.0, "event": "a.second"}\n'
            "   \n"
            '{"ts": 3.0, "event": "a.third"}\n'
            '{"ts": 4.0, "event": "a.trunca'   # half-written, no newline
        )
        path = _write(log_dir_, raw)
        got = read_tail(lines=10)
        assert got == _read_tail_before(path, lines=10), (
            f"corrupt-log handling diverged:\n  new={got}\n"
            f"  old={_read_tail_before(path, lines=10)}"
        )
        assert [e["event"] for e in got] == ["a.first", "a.second", "a.third"], got

    # A trailing newline is optional — the last line must still be returned.
    with _agent_dir(ctx, "nonewline") as log_dir_:
        path = _write(log_dir_, '{"ts": 1.0, "event": "only.one"}')
        got = read_tail(lines=10)
        assert got == _read_tail_before(path, lines=10), got
        assert len(got) == 1 and got[0]["event"] == "only.one", got


@test("logs_read_tail", "read_tail spans block boundaries exactly (no dropped/merged lines)")
async def t_block_boundaries(ctx: TestContext) -> None:
    """The reverse reader stitches 64 KB blocks back together; an off-by-one at
    a boundary would silently drop or merge an entry mid-file, which no small
    fixture would ever notice."""
    from src.core.logging import _BLOCK_BYTES, read_tail

    with _agent_dir(ctx, "blocks") as log_dir_:
        # ~250 KB — several blocks, with lines of varying length so boundaries
        # land in different places within an entry.
        entries = [
            {"ts": float(i), "event": f"tick.{i}", "pad": "x" * (i % 97)}
            for i in range(3000)
        ]
        path = _write(log_dir_, entries)
        assert path.stat().st_size > _BLOCK_BYTES * 3, "fixture must span blocks"

        got = read_tail(lines=10_000)
        assert got == _read_tail_before(path, lines=10_000), \
            "multi-block read diverged from the slurp"
        assert len(got) == 3000, len(got)
        assert [e["ts"] for e in got] == [float(i) for i in range(3000)], \
            "block stitching dropped or reordered entries"


@test("logs_read_tail", "a prefix matching only ancient entries still returns them")
async def t_ancient_prefix_survives_unbounded(ctx: TestContext) -> None:
    """The sharp edge of moving onto the MCP's reader.

    ``read_tail`` must stay unbounded. The MCP caps its scans at 2 MB so a
    model-driven query with a rare filter can't walk the whole log on the event
    loop — but if that cap had ridden along as the primitive's default, this
    query (whose only matches are older than the cap) would have quietly
    returned nothing instead of three rows. Silent, and shaped exactly like an
    empty log.

    Both halves are asserted here: read_tail sees the ancient rows, and the
    MCP's reader — same file, same primitive, its own default — does not, which
    is what proves the cap is still opt-in rather than merely absent.
    """
    from src.core.logging import read_tail
    from src.mcp.servers.logs import reader

    with _agent_dir(ctx, "ancient") as log_dir_:
        # Three ancient markers, then >2 MB of newer noise burying them.
        entries: list[dict[str, Any]] = [
            {"ts": 1000.0 + i, "event": "ancient.marker", "seq": i} for i in range(3)
        ]
        entries += [
            {"ts": 2000.0 + i, "event": "noise.tick", "pad": "y" * 400}
            for i in range(6000)
        ]
        path = _write(log_dir_, entries)
        size = path.stat().st_size
        assert size > reader._MAX_SCAN_BYTES, (
            f"fixture must bury the markers beyond the MCP's {reader._MAX_SCAN_BYTES}-byte "
            f"cap; got {size}"
        )

        # read_tail: unbounded, so the ancient prefix is still reachable.
        got = read_tail(lines=10, event_filter="ancient.")
        assert [e["seq"] for e in got] == [0, 1, 2], (
            f"read_tail lost an ancient prefix match — the scan cap leaked into "
            f"the unbounded path. got={got}"
        )
        assert got == _read_tail_before(path, lines=10, event_filter="ancient."), \
            "ancient-prefix result diverged from the slurp"

        # The MCP's reader: capped by default, and it says so rather than
        # implying the log had nothing older.
        stats = reader.ScanStats()
        seen = [
            e for e in reader.iter_entries_reverse(stats=stats)
            if str(e.get("event", "")).startswith("ancient.")
        ]
        assert seen == [], (
            "the MCP's 2 MB cap stopped applying — its default must stay capped"
        )
        assert stats.hit_scan_cap is True, (
            "capped scan must report hit_scan_cap so a caller can tell "
            "'nothing older' from 'I stopped looking'"
        )

        # Same reader, cap lifted explicitly → the markers are there. Proves the
        # emptiness above is the cap, not a broken scan.
        uncapped = [
            e for e in reader.iter_entries_reverse(max_bytes=None)
            if str(e.get("event", "")).startswith("ancient.")
        ]
        assert len(uncapped) == 3, uncapped


@test("logs_read_tail", "core and the logs MCP read the same file through one primitive")
async def t_single_reader(ctx: TestContext) -> None:
    """Guards the de-duplication itself: if someone re-adds a private reader to
    the MCP, these stop being the same object and the two can drift again."""
    from src.core import logging as core_logging
    from src.mcp.servers.logs import reader

    assert reader.iter_lines_reverse is core_logging.iter_lines_reverse
    assert reader.ScanStats is core_logging.ScanStats
    assert reader.events_path is core_logging.events_path
    assert reader.iso is core_logging.iso

    # …and the MCP's wrapper really is core's generator, only pre-capped.
    now = time.time()
    with _agent_dir(ctx, "shared") as log_dir_:
        _write(log_dir_, _corpus(now))
        via_core = list(core_logging.iter_events_reverse())
        via_mcp = list(reader.iter_entries_reverse())
        assert via_core == via_mcp, "the two readers disagree on the same file"


# ── the endpoint: no longer on the event loop ────────────────────────


@test("logs_read_tail", "GET /api/logs reads off the event loop, not on it")
async def t_endpoint_offloads_to_thread(ctx: TestContext) -> None:
    """Proves the handler hands the blocking read to a thread.

    Two independent observations, because either alone is weak: the read runs
    on a *different thread* than the loop, AND a concurrent coroutine keeps
    getting scheduled while the read blocks. The second is the thing users
    feel — a stalled loop is a stalled voice turn — and it fails loudly if the
    ``await asyncio.to_thread`` is ever "simplified" back to a direct call.
    """
    from src.core import logging as core_logging
    from src.gateway.api import logs as logs_api

    loop_thread = threading.get_ident()
    observed: dict[str, Any] = {}

    def _blocking_read(lines=100, event_filter=None):
        # Stands in for a big log: a real, uninterruptible block — the only
        # kind that can pin an event loop.
        observed["thread"] = threading.get_ident()
        observed["args"] = (lines, event_filter)
        time.sleep(0.3)
        return [{"ts": 1.0, "event": "spy.entry"}]

    ticks = 0

    async def _heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    original = core_logging.read_tail
    core_logging.read_tail = _blocking_read
    hb = asyncio.create_task(_heartbeat())
    try:
        await asyncio.sleep(0.02)  # let the heartbeat start ticking
        ticks_before = ticks
        resp = await logs_api.handle_get(
            make_mocked_request("GET", "/api/logs?lines=7&event=spy."),
        )
    finally:
        hb.cancel()
        core_logging.read_tail = original
        try:
            await hb
        except asyncio.CancelledError:
            pass

    assert observed.get("thread") is not None, "read_tail was never called"
    assert observed["thread"] != loop_thread, (
        "read_tail ran on the event-loop thread — the gateway is blocked for "
        "the whole read, which is the bug this endpoint had"
    )
    # The query still reaches read_tail with the same arguments as before.
    assert observed["args"] == (7, "spy."), observed["args"]

    ticked = ticks - ticks_before
    assert ticked >= 5, (
        f"the event loop only advanced {ticked} times during a 0.3s read — it "
        f"was blocked (expected ~30 at a 10ms heartbeat)"
    )

    assert resp.status == 200, resp.status
    assert json.loads(resp.body) == [{"ts": 1.0, "event": "spy.entry"}], resp.body


@test("logs_read_tail", "GET /api/logs returns real entries end-to-end")
async def t_endpoint_end_to_end(ctx: TestContext) -> None:
    """The offload above uses a spy, so this drives the endpoint against a real
    file: same shape (a bare JSON array, oldest→newest), same filter."""
    from src.gateway.api import logs as logs_api

    now = time.time()
    with _agent_dir(ctx, "endpoint") as log_dir_:
        _write(log_dir_, _corpus(now))

        resp = await logs_api.handle_get(make_mocked_request("GET", "/api/logs"))
        assert resp.status == 200, resp.status
        body = json.loads(resp.body)
        assert isinstance(body, list), f"response shape changed: {type(body)}"
        assert len(body) == 8, body
        assert [e["ts"] for e in body] == sorted(e["ts"] for e in body)

        resp = await logs_api.handle_get(
            make_mocked_request("GET", "/api/logs?lines=2&event=mcp."),
        )
        body = json.loads(resp.body)
        assert len(body) == 2, body
        assert all(e["event"].startswith("mcp.") for e in body), body
