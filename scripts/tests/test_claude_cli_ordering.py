"""ClaudeCLI persistence — interleaved text + tool blocks land in order.

Pre-fix, ``_persist_turn`` wrote the messages array as
``user -> assistant(full_text) -> tool1 -> tool2 -> ...`` regardless of
the original SDK stream order. Tool-using turns with prose between
calls (the common case for Claude planning out loud) lost everything
except the final ``ResultMessage.result`` text. The fix carries an
ordered ``assistant_blocks`` list straight from ``_run_once`` into
``_persist_turn``, where each block becomes its own message in the
original interleaved order.

Tests stub ``_get_client`` and patch ``claude_agent_sdk.AssistantMessage``
/ ``UserMessage`` / ``ResultMessage`` enough that the production
consumer loop runs unmodified on a fake event script.
"""
from __future__ import annotations

import uuid
from typing import Any

from ._framework import TestContext, test


# ── Fake SDK message types ─────────────────────────────────────────────


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _ToolUseBlock:
    def __init__(self, *, name: str, input: dict, id: str) -> None:
        self.name = name
        self.input = input
        self.id = id
        self.text = None  # block_text attr is checked before tool_name


class _ToolResultBlock:
    def __init__(self, *, tool_use_id: str, content: Any, is_error: bool = False) -> None:
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class _FakeAssistantMessage:
    def __init__(self, content: list[Any], model: str | None = None) -> None:
        self.content = content
        self.model = model


class _FakeUserMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _FakeResultMessage:
    def __init__(self, *, result: str, session_id: str | None = None) -> None:
        self.result = result
        self.session_id = session_id
        self.total_cost_usd = 0.0
        self.usage = {}
        self.model_usage = None
        self.duration_ms = 0
        self.duration_api_ms = 0
        self.num_turns = 1


class _FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self._script = script

    async def query(self, prompt: str, session_id: str) -> None:
        return None

    async def receive_response(self):
        for m in self._script:
            yield m


# ── Test harness ───────────────────────────────────────────────────────


def _patch_sdk_imports(monkey: dict) -> None:
    """``_run_once`` does an ``isinstance(...)`` check against the SDK's
    own classes — patch the names it looks up at import time inside
    the function body so our fakes register as the real thing."""
    import claude_agent_sdk

    monkey["orig"] = {
        "AssistantMessage": claude_agent_sdk.AssistantMessage,
        "UserMessage": claude_agent_sdk.UserMessage,
        "ResultMessage": claude_agent_sdk.ResultMessage,
    }
    claude_agent_sdk.AssistantMessage = _FakeAssistantMessage
    claude_agent_sdk.UserMessage = _FakeUserMessage
    claude_agent_sdk.ResultMessage = _FakeResultMessage


def _unpatch_sdk_imports(monkey: dict) -> None:
    import claude_agent_sdk

    orig = monkey.get("orig") or {}
    for k, v in orig.items():
        setattr(claude_agent_sdk, k, v)


# ── Tests ──────────────────────────────────────────────────────────────


@test("claude_cli_ordering", "assistant_blocks captures interleaved text/tool order")
async def t_interleaved_order(ctx: TestContext) -> None:
    """text -> tool_use -> tool_result -> text -> tool_use -> tool_result
    -> text -> ResultMessage. After ``_run_once`` runs, the
    ``assistant_blocks`` list must mirror that exact order."""
    from src.models.claude_cli import ClaudeCLI

    monkey: dict = {}
    _patch_sdk_imports(monkey)
    try:
        script = [
            _FakeAssistantMessage([_TextBlock("Let me check.")]),
            _FakeAssistantMessage([
                _ToolUseBlock(name="search", input={"q": "x"}, id="tu1"),
            ]),
            _FakeUserMessage([
                _ToolResultBlock(tool_use_id="tu1", content="search results..."),
            ]),
            _FakeAssistantMessage([_TextBlock("Now also reading the file.")]),
            _FakeAssistantMessage([
                _ToolUseBlock(name="read_file", input={"path": "/a"}, id="tu2"),
            ]),
            _FakeUserMessage([
                _ToolResultBlock(tool_use_id="tu2", content="file body..."),
            ]),
            _FakeAssistantMessage([_TextBlock("Done.")]),
            _FakeResultMessage(result="Done.", session_id="sdk-sid"),
        ]

        inst = ClaudeCLI(model="claude-sonnet-4-6")
        blocks: list[dict] = []
        tool_calls: list[dict] = []
        await inst._run_once(
            client=_FakeClient(script),
            prompt="please",
            session_id="sess-int",
            tool_calls_out=tool_calls,
            assistant_blocks_out=blocks,
        )

        types = [b["type"] for b in blocks]
        assert types == [
            "text", "tool_use", "tool_result",
            "text", "tool_use", "tool_result",
            "text",
        ], types
        texts = [b["text"] for b in blocks if b["type"] == "text"]
        assert texts == [
            "Let me check.", "Now also reading the file.", "Done.",
        ], texts
        tool_ids = [b["id"] for b in blocks if b["type"] == "tool_use"]
        assert tool_ids == ["tu1", "tu2"], tool_ids
        tool_names = [b["name"] for b in blocks if b["type"] == "tool_use"]
        assert tool_names == ["search", "read_file"], tool_names
        result_ids = [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"]
        assert result_ids == ["tu1", "tu2"], result_ids
    finally:
        _unpatch_sdk_imports(monkey)


@test("claude_cli_ordering", "_persist_turn writes interleaved messages array")
async def t_persist_turn_order(ctx: TestContext) -> None:
    """The DB-facing messages array must mirror ``assistant_blocks`` —
    text becomes a normal assistant row, tool_use becomes an assistant
    row with a ``tool_calls`` payload, tool_result becomes a tool row
    keyed by ``tool_call_id``."""
    from src.memory.db import MemoryDB
    from src.models.claude_cli import ClaudeCLI

    tmp_db = ctx.db_path.with_name(f"order-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        inst = ClaudeCLI(model="claude-sonnet-4-6")
        inst.set_db(db)

        blocks = [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "x"}},
            {"type": "tool_result", "tool_use_id": "tu1", "content": "res1", "is_error": False},
            {"type": "text", "text": "Now also..."},
            {"type": "tool_use", "id": "tu2", "name": "read_file", "input": {"p": "/a"}},
            {"type": "tool_result", "tool_use_id": "tu2", "content": "res2", "is_error": False},
            {"type": "text", "text": "Done."},
        ]
        sid = f"order-{uuid.uuid4().hex[:8]}"
        inst._persist_turn(
            sid, prompt="please",
            assistant_text="Done.",
            tool_calls=[
                {"name": "search", "id": "tu1", "input": {"q": "x"}, "output": "res1"},
                {"name": "read_file", "id": "tu2", "input": {"p": "/a"}, "output": "res2"},
            ],
            assistant_blocks=blocks,
            model="claude-sonnet-4-6",
        )
        # Drain the fire-and-forget write task before reading back.
        # Snapshot first — the task's done-callback mutates the dict.
        all_tasks: list = []
        for task_set in list(inst._pending_writes.values()):
            all_tasks.extend(task_set)
        for t in all_tasks:
            try:
                await t
            except Exception:
                pass

        runs = await db.list_session_runs(sid, limit=10)
        assert len(runs) == 1, runs
        msgs = runs[0].get("messages", [])
        # 1 user + 3 text-assistant + 2 tool_use-assistant + 2 tool = 8
        assert len(msgs) == 8, [m.get("role") for m in msgs]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant" and msgs[1]["content"] == "Let me check."
        assert msgs[2]["role"] == "assistant" and msgs[2].get("tool_calls")
        assert msgs[2]["tool_calls"][0]["id"] == "tu1"
        assert msgs[2]["tool_calls"][0]["function"]["name"] == "search"
        assert msgs[3]["role"] == "tool" and msgs[3]["tool_call_id"] == "tu1"
        assert msgs[3]["content"] == "res1", msgs[3]
        assert msgs[4]["role"] == "assistant" and msgs[4]["content"] == "Now also..."
        assert msgs[5]["role"] == "assistant" and msgs[5]["tool_calls"][0]["id"] == "tu2"
        assert msgs[6]["role"] == "tool" and msgs[6]["tool_call_id"] == "tu2"
        assert msgs[7]["role"] == "assistant" and msgs[7]["content"] == "Done."
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("claude_cli_ordering", "_persist_turn legacy path still works (no blocks)")
async def t_persist_turn_legacy(ctx: TestContext) -> None:
    """When the caller doesn't thread ``assistant_blocks`` through (older
    tests, partial paths), ``_persist_turn`` falls back to the flat
    user -> assistant -> tool layout. This guards back-compat."""
    from src.memory.db import MemoryDB
    from src.models.claude_cli import ClaudeCLI

    tmp_db = ctx.db_path.with_name(f"legacy-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        inst = ClaudeCLI(model="claude-sonnet-4-6")
        inst.set_db(db)
        sid = f"legacy-{uuid.uuid4().hex[:8]}"
        inst._persist_turn(
            sid, prompt="hi",
            assistant_text="hello",
            tool_calls=[
                {"name": "search", "id": "tu1", "input": {}, "output": "ok"},
            ],
            model="claude-sonnet-4-6",
        )
        all_tasks: list = []
        for task_set in list(inst._pending_writes.values()):
            all_tasks.extend(task_set)
        for t in all_tasks:
            try:
                await t
            except Exception:
                pass
        runs = await db.list_session_runs(sid, limit=10)
        msgs = runs[0].get("messages", [])
        assert [m["role"] for m in msgs] == ["user", "assistant", "tool"], msgs
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
