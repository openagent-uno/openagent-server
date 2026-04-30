"""ClaudeCLI / ClaudeCLIRegistry stream-method coverage.

Covers two related fixes:

* ``ClaudeCLI.stream`` now accepts ``session_id``, ``on_status``, and
  ``model_override``. The legacy implementation hardcoded
  ``sid = "default"`` (bug: every concurrent voice turn collided on one
  shared SDK subprocess) and dropped ``on_status`` (bug: tool-running
  statuses never surfaced to the WS during streamed turns).
* ``ClaudeCLIRegistry`` now overrides ``stream`` to delegate to the
  per-session instance. Without this override, ``BaseModel.stream``
  fell back to a one-shot ``generate`` — the root cause of
  SmartRouter-routed claude-cli replies arriving as one giant chunk.

Tests stub ``_run_once`` and ``_get_client`` so we exercise the
plumbing without spawning a real SDK subprocess (no Anthropic key
needed, no ``claude`` binary on the test box).
"""
from __future__ import annotations

from typing import Any

from ._framework import TestContext, test


# ── Helpers ─────────────────────────────────────────────────────────


def _make_claude_cli(model: str = "claude-sonnet-4-6"):
    """Build a ClaudeCLI instance with stubbed SDK plumbing.

    ``_get_client`` returns a no-op object (the production path expects
    a ClaudeSDKClient, but our stubbed ``_run_once`` never calls into
    it). ``_ensure_session_model`` is similarly a no-op so we don't
    need a control-protocol-capable client. ``_run_once`` just records
    the kwargs it was called with and replays a fixed delta script.
    """
    from openagent.models.claude_cli import ClaudeCLI

    inst = ClaudeCLI(model=model)
    return inst


def _patch_run_once(inst, deltas: list[str], record: list[dict]):
    async def _fake_get_client(sid: str, system: str | None) -> Any:
        return object()

    async def _fake_ensure_session_model(sid: str, client: Any, model: str | None) -> None:
        return None

    async def _fake_run_once(
        client: Any, prompt: str, sid: str,
        on_status: Any = None, on_delta: Any = None, tool_names_out: Any = None,
    ) -> tuple[str, dict]:
        record.append({
            "session_id": sid,
            "on_status": on_status,
            "on_delta": on_delta,
            "prompt": prompt,
        })
        # Replay the script through on_delta — same wiring the real
        # subprocess uses to emit token chunks.
        if on_delta is not None:
            for d in deltas:
                await on_delta(d)
        return ("".join(deltas) or "(empty)", {})

    inst._get_client = _fake_get_client  # type: ignore[assignment]
    inst._ensure_session_model = _fake_ensure_session_model  # type: ignore[assignment]
    inst._run_once = _fake_run_once  # type: ignore[assignment]


# ── Tests ───────────────────────────────────────────────────────────


@test("claude_cli_stream", "ClaudeCLI.stream forwards session_id to _run_once")
async def t_stream_forwards_session_id(_ctx: TestContext) -> None:
    """The hardcoded ``sid = 'default'`` was a per-process collision —
    two concurrent voice tabs both routing through claude-cli would
    fight over one SDK subprocess. The fix accepts session_id and
    threads it through."""
    inst = _make_claude_cli()
    record: list[dict] = []
    _patch_run_once(inst, deltas=["x"], record=record)

    yielded: list[str] = []
    async for chunk in inst.stream(
        [{"role": "user", "content": "hi"}], session_id="sess-A",
    ):
        yielded.append(chunk)

    assert record, "_run_once should have been called"
    assert record[0]["session_id"] == "sess-A", record[0]
    assert yielded == ["x"], yielded


@test("claude_cli_stream", "ClaudeCLI.stream forwards on_status (tool-update bug)")
async def t_stream_forwards_on_status(_ctx: TestContext) -> None:
    """Pre-fix the streaming path passed ``on_status=None``, so the WS
    never received "Using ReadFile…" updates during a streamed turn —
    voice mode looked frozen between sentences."""
    inst = _make_claude_cli()
    record: list[dict] = []
    _patch_run_once(inst, deltas=[], record=record)

    async def on_status(text: str) -> None: ...

    async for _ in inst.stream(
        [{"role": "user", "content": "hi"}],
        session_id="sess-B",
        on_status=on_status,
    ):
        pass

    assert record[0]["on_status"] is on_status, (
        f"on_status must reach _run_once, got {record[0]['on_status']!r}"
    )


@test("claude_cli_stream", "ClaudeCLI.stream honours model_override")
async def t_stream_honours_model_override(_ctx: TestContext) -> None:
    """Mirrors the generate() resolution: an explicit override wins
    over the instance default. SmartRouter passes the routed runtime_id
    here so the SDK subprocess is pinned correctly on first spawn."""
    inst = _make_claude_cli(model="claude-sonnet-4-6")
    record: list[dict] = []
    _patch_run_once(inst, deltas=["ok"], record=record)

    # Override with a different model — instance attribute should flip
    # to the normalised form before _run_once runs.
    async for _ in inst.stream(
        [{"role": "user", "content": "hi"}],
        session_id="sess-override",
        model_override="claude-opus-4-1",
    ):
        pass

    # The instance.model is updated synchronously inside stream() before
    # the driver task fires, so a same-event-loop assertion sees it.
    assert inst.model == "claude-opus-4-1", inst.model


@test("claude_cli_stream", "ClaudeCLIRegistry.stream creates per-session instance")
async def t_registry_stream_per_session(_ctx: TestContext) -> None:
    """Two distinct session_ids must hit two distinct ClaudeCLI
    instances. Pre-fix the registry didn't override stream at all
    (BaseModel default → one-shot generate); even after delegating, a
    bug here would re-introduce the shared 'default' subprocess."""
    from openagent.models.claude_cli import ClaudeCLIRegistry

    registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")

    # Stub the ClaudeCLI that _get_or_create returns so we don't spawn
    # real SDK subprocesses; capture which session each .stream call hit.
    seen_sessions: list[str] = []

    class _StubInst:
        def __init__(self, sid: str):
            self.sid = sid
            self.model = "claude-sonnet-4-6"

        async def stream(
            self, messages, system=None, on_status=None,
            session_id=None, model_override=None,
        ):
            seen_sessions.append(session_id)
            yield f"hello-{session_id}"

    def _fake_get_or_create(sid: str, model_id: str | None):
        return _StubInst(sid)

    registry._get_or_create = _fake_get_or_create  # type: ignore[assignment]

    a_chunks: list[str] = []
    async for c in registry.stream(
        [{"role": "user", "content": "hi"}], session_id="sess-A",
    ):
        a_chunks.append(c)

    b_chunks: list[str] = []
    async for c in registry.stream(
        [{"role": "user", "content": "hi"}], session_id="sess-B",
    ):
        b_chunks.append(c)

    assert a_chunks == ["hello-sess-A"], a_chunks
    assert b_chunks == ["hello-sess-B"], b_chunks
    assert seen_sessions == ["sess-A", "sess-B"], seen_sessions


@test("claude_cli_stream", "ClaudeCLIRegistry.stream forwards model_override")
async def t_registry_stream_forwards_override(_ctx: TestContext) -> None:
    """SmartRouter calls registry.stream(..., model_override=runtime_id)
    so a session bound to claude-opus doesn't get a sonnet subprocess
    by accident."""
    from openagent.models.claude_cli import ClaudeCLIRegistry

    registry = ClaudeCLIRegistry(default_model="claude-sonnet-4-6")
    seen_overrides: list[str | None] = []

    class _StubInst:
        async def stream(
            self, messages, system=None, on_status=None,
            session_id=None, model_override=None,
        ):
            seen_overrides.append(model_override)
            yield "ok"

    registry._get_or_create = lambda sid, m: _StubInst()  # type: ignore[assignment]
    registry._resolve_model = lambda sid, override: override or "claude-sonnet-4-6"  # type: ignore[assignment]

    async for _ in registry.stream(
        [{"role": "user", "content": "hi"}],
        session_id="sess-X",
        model_override="claude-opus-4-1",
    ):
        pass

    assert seen_overrides == ["claude-opus-4-1"], seen_overrides
