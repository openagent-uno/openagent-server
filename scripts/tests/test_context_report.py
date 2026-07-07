"""Context-window composition report (/context) — pure-unit.

Covers the server-side source of truth behind the ``/context`` command,
the ``GET /api/sessions/{id}/context`` endpoint, and the realtime
``context_report`` WS frame: :func:`src.core.context_report.build_context_report`
plus the catalog context-window lookup and the wire round-trip. Uses a
synthetic sessions DB + fake agent, so no network or gateway is needed.
"""
from __future__ import annotations

import json
import sqlite3

from ._framework import TestContext, test


def _make_session_db(path: str, session_id: str, runs, session_data, summary):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, session_data TEXT, "
        "metadata TEXT, runs TEXT, summary TEXT, created_at INTEGER, updated_at INTEGER)"
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
        (session_id, json.dumps(session_data), "{}", json.dumps(runs), summary, 0, 0),
    )
    conn.commit()
    conn.close()


class _FakeMCP:
    def runtime_toolkits_tool_search_only(self):
        return []


class _FakeModel:
    def effective_model_id(self, session_id=None):
        return "anthropic:claude-opus-4-8"


class _FakeAgent:
    def __init__(self, db_path):
        self._db = type("_DB", (), {"db_path": db_path})()
        self.model = _FakeModel()
        self._mcp = _FakeMCP()
        self.system_prompt = "You are a helpful persona."

    def _resolve_vault_path(self):
        return "/tmp/vault"

    def _combined_system_prompt(self, session_id=None):
        return ("FRAMEWORK " * 500) + f"\n<session-id>{session_id}</session-id>"


@test("context_report", "build_context_report produces a sectioned window breakdown")
async def test_build_report(ctx: TestContext) -> None:
    from src.core.context_report import build_context_report

    db_path = str(ctx.test_dir / "context-report.db")
    runs = [
        {"model": "claude-opus-4-8", "model_provider": "anthropic",
         "content": "Assistant reply one.",
         "messages": [{"role": "user", "content": "A user question about tokens and windows."}],
         "metrics": {"input_tokens": 4210, "output_tokens": 120, "cost": 0.01}},
        {"model": "claude-opus-4-8", "model_provider": "anthropic",
         "content": "Assistant reply two, a little longer than the first one here.",
         "messages": [{"role": "user", "content": "Another user turn continuing the thread."}],
         "metrics": {"input_tokens": 6050, "output_tokens": 240, "cost": 0.02}},
    ]
    session_data = {"session_metrics": {
        "input_tokens": 10260, "output_tokens": 360, "cache_read_tokens": 300,
        "reasoning_tokens": 0, "cost": 0.03}}
    _make_session_db(db_path, "sid", runs, session_data, "Rolling summary text.")

    rep = build_context_report(_FakeAgent(db_path), "sid")
    assert rep is not None, "report should not be None for a DB-backed session"
    assert rep["model"] == "anthropic:claude-opus-4-8", rep["model"]
    assert rep["context_window"] == 200_000, rep["context_window"]

    keys = [s["key"] for s in rep["sections"]]
    assert keys == ["system", "tools", "messages", "summary", "free"], keys

    # used = sum of non-free sections; free = window - used; they tile the window.
    non_free = sum(s["tokens"] for s in rep["sections"] if s["key"] != "free")
    free = next(s["tokens"] for s in rep["sections"] if s["key"] == "free")
    assert rep["used_tokens"] == non_free, (rep["used_tokens"], non_free)
    assert rep["free_tokens"] == free == 200_000 - non_free, (free, non_free)
    assert non_free + free == rep["context_window"]

    # System prompt dominates (framework text) and every count is non-negative.
    assert rep["sections"][0]["tokens"] > 0
    assert all(s["tokens"] >= 0 for s in rep["sections"])

    # Cumulative usage lifted from persisted session_metrics. Crucially, the
    # cost is the RECORDED session_metrics.cost verbatim (the queryable mirror
    # of the usage_log ledger behind the app's Settings → Costs screen) — never
    # recomputed here — so /context stays consistent with Settings even if live
    # pricing warms up after a turn was billed.
    assert rep["total_input_tokens"] == 10260
    assert rep["total_output_tokens"] == 360
    assert rep["cost_usd"] == 0.03, "session cost must equal recorded session_metrics.cost (no recompute)"
    assert rep["turns"] == 2
    # Authoritative measured size = last run's provider input_tokens.
    assert rep["measured_input_tokens"] == 6050, rep["measured_input_tokens"]


@test("context_report", "build_context_report returns None without a session id")
async def test_no_session(ctx: TestContext) -> None:
    from src.core.context_report import build_context_report

    assert build_context_report(_FakeAgent("/nonexistent.db"), None) is None


@test("context_report", "empty session (no runs) still yields a valid full-free window")
async def test_empty_session(ctx: TestContext) -> None:
    from src.core.context_report import build_context_report

    db_path = str(ctx.test_dir / "context-empty.db")
    _make_session_db(db_path, "sid", [], {}, "")
    rep = build_context_report(_FakeAgent(db_path), "sid")
    assert rep is not None
    assert rep["turns"] == 0
    messages = next(s for s in rep["sections"] if s["key"] == "messages")
    assert messages["tokens"] == 0, messages
    # System prompt still occupies the window even with no conversation.
    assert rep["used_tokens"] >= rep["sections"][0]["tokens"] > 0
    assert rep["measured_input_tokens"] == 0


@test("context_report", "pre-change sessions (no metrics/model, legacy shapes) work unchanged")
async def test_backward_compat(ctx: TestContext) -> None:
    """No schema change was made — the report only READS existing columns, so a
    session persisted before this feature must still produce a valid report.
    Covers: runs without a ``metrics``/``model`` field, a ``session_data``
    without ``session_metrics``, a legacy bare-dict metrics shape, the runtime's
    double-encoded (str-of-JSON) ``runs`` column, and a NULL ``runs`` column."""
    from src.core.context_report import build_context_report

    db_path = str(ctx.test_dir / "context-legacy.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, session_data TEXT, "
        "metadata TEXT, runs TEXT, summary TEXT, created_at INTEGER, updated_at INTEGER)"
    )
    # Old run dicts: only content/messages — no metrics, no model attribution.
    old_runs = [
        {"content": "Old reply.", "messages": [{"role": "user", "content": "Old user turn."}]},
    ]
    # A: no session_metrics, NULL summary.
    conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                 ("old", "{}", "{}", json.dumps(old_runs), None, 0, 0))
    # B: double-encoded runs (str of JSON) + legacy bare-dict metrics.
    conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                 ("legacy", json.dumps({"session_metrics": {"input_tokens": 999, "cost": 0.007}}),
                  "{}", json.dumps(json.dumps(old_runs)), "s", 0, 0))
    # C: NULL runs + NULL session_data.
    conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                 ("nullruns", None, None, None, None, 0, 0))
    conn.commit()
    conn.close()

    for sid in ("old", "legacy", "nullruns"):
        rep = build_context_report(_FakeAgent(db_path), sid)
        assert rep is not None, f"legacy session {sid} must still report"
        assert rep["context_window"] == 200_000
        # Sections always tile the window; system prompt is always present.
        non_free = sum(s["tokens"] for s in rep["sections"] if s["key"] != "free")
        assert rep["used_tokens"] == non_free
        assert rep["free_tokens"] == 200_000 - non_free
        assert rep["sections"][0]["key"] == "system" and rep["sections"][0]["tokens"] > 0
        # Missing model attribution falls back to a resolvable runtime id.
        assert rep["model"], sid
    # Legacy cost is read verbatim from the bare-dict metrics.
    assert build_context_report(_FakeAgent(db_path), "legacy")["cost_usd"] == 0.007
    # A session with no cumulative metrics reports cost None, not a crash.
    assert build_context_report(_FakeAgent(db_path), "old")["cost_usd"] is None


@test("context_report", "catalog context window falls back to 200k for unknown models")
async def test_catalog_window_fallback(ctx: TestContext) -> None:
    from src.models.catalog import get_model_context_window

    window, source = get_model_context_window("local:some-self-hosted-model")
    assert window == 200_000, window
    assert source == "fallback", source


@test("context_report", "ContextReport event round-trips through the wire codec")
async def test_wire_roundtrip(ctx: TestContext) -> None:
    from src.stream.events import ContextReport
    from src.stream.wire import event_to_wire, wire_to_event

    payload = {"used_tokens": 1234, "context_window": 200_000, "sections": [{"key": "free", "tokens": 198766}]}
    evt = ContextReport(session_id="sX", seq=7, ts_ms=999, report=payload)
    frame = event_to_wire(evt)
    assert frame["type"] == "context_report", frame
    assert frame["report"] == payload
    back = wire_to_event(frame)
    assert isinstance(back, ContextReport)
    assert back.report == payload
    assert back.session_id == "sX" and back.seq == 7 and back.ts_ms == 999


@test("context_report", "format_context_report_text renders a fenced monospace block")
async def test_text_form(ctx: TestContext) -> None:
    from src.core.context_report import format_context_report_text

    report = {
        "model_label": "claude-opus-4-8", "context_window": 200_000, "used_tokens": 42_000,
        "used_pct": 21.0, "window_source": "openrouter", "cost_usd": 0.12,
        "sections": [
            {"key": "system", "label": "System prompt", "tokens": 12_000, "pct": 6.0},
            {"key": "free", "label": "Free space", "tokens": 158_000, "pct": 79.0},
        ],
    }
    text = format_context_report_text(report)
    assert text.startswith("```") and text.rstrip().endswith("```"), text
    assert "System prompt" in text and "session cost: $0.1200" in text
