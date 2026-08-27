"""Stream protocol — events, wire codec, session, channels.

Pure-unit tests for :mod:`openagent.stream`. Run via the existing test
driver:

    bash scripts/test_openagent.sh --only stream

Coverage:

* Wire round-trip for every event type, including legacy ``MESSAGE`` →
  ``TextFinal`` decoding so older clients keep working.
* :func:`resolve_stt` / :func:`resolve_tts` factory dispatch.
* :class:`StreamSession.run_one_shot` against a fake agent — verifies
  delta + final + (optional) audio chunks land on the outbound queue
  and a ``TurnComplete`` is the last event.
* :class:`BatchedChannel.run_one_shot` collapses the outbound stream
  into one finished reply with concatenated audio.
"""
from __future__ import annotations

import asyncio

from ._framework import TestContext, test


# ── wire round-trip ──────────────────────────────────────────────────


@test("stream", "events round-trip the wire codec verbatim")
async def t_wire_round_trip(ctx: TestContext) -> None:
    from src.stream.events import (
        Attachment, AudioChunk, Interrupt, OutAudioChunk, OutAudioEnd, OutAudioStart,
        OutReasoning, OutTextDelta, OutTextFinal, OutToolStatus, OutVideoFrame,
        SessionOpen, TextDelta, TextFinal, TurnComplete, VideoFrame,
    )
    from src.stream.events import OutError
    from src.stream.wire import event_to_wire, wire_to_event

    cases = [
        OutTextDelta(session_id="s", seq=1, ts_ms=10, text="hi"),
        OutTextFinal(
            session_id="s", seq=2, ts_ms=20, text="done", model="m",
            parts=({"kind": "text", "text": "done"},),
        ),
        OutAudioStart(session_id="s", seq=3, ts_ms=30, format="mp3", mime="audio/mpeg"),
        OutAudioChunk(session_id="s", seq=4, ts_ms=40, data=b"\x00\x01"),
        OutAudioEnd(session_id="s", seq=5, ts_ms=50, total_chunks=1),
        OutToolStatus(session_id="s", seq=6, ts_ms=60, text="Using bash"),
        OutReasoning(session_id="s", seq=16, ts_ms=160, active=True),
        OutReasoning(session_id="s", seq=17, ts_ms=170, active=False),
        OutVideoFrame(session_id="s", seq=7, ts_ms=70, stream="webcam",
                      image_bytes=b"jpgbytes", width=320, height=240),
        TurnComplete(session_id="s", seq=8, ts_ms=80),
        TextDelta(session_id="s", seq=9, ts_ms=90, text="hel", final=False),
        TextFinal(session_id="s", seq=10, ts_ms=100, text="hello", source="stt"),
        AudioChunk(session_id="s", seq=11, ts_ms=110, data=b"raw",
                   end_of_speech=True, sample_rate=16000, encoding="pcm16"),
        VideoFrame(session_id="s", seq=12, ts_ms=120, stream="screen",
                   image_bytes=b"frame", width=1024, height=768, keyframe=True),
        Attachment(
            session_id="s", seq=18, ts_ms=180, kind="image",
            path="/internal/cas", filename="photo.png", mime_type="image/png",
            artifact_id="art_1", artifact_link_id="alink_1", size_bytes=4,
            sha256="a" * 64, url="/api/artifacts/art_1/content",
        ),
        Interrupt(session_id="s", seq=13, ts_ms=130, reason="user_speech"),
        SessionOpen(session_id="s", seq=14, ts_ms=140, profile="realtime",
                    language="en", client_kind="webapp",
                    client_capabilities={"attachments": True, "inline_ui": True}),
        OutError(session_id="s", seq=15, ts_ms=150, text="boom"),
    ]
    for evt in cases:
        wire = event_to_wire(evt)
        back = wire_to_event(wire)
        assert back == evt, f"round-trip mismatch: {evt!r} → {wire!r} → {back!r}"


@test("stream", "legacy MESSAGE frame decodes to TextFinal")
async def t_legacy_message_decodes(ctx: TestContext) -> None:
    from src.stream.events import TextFinal
    from src.stream.wire import wire_to_event

    evt = wire_to_event({"type": "message", "session_id": "s1", "text": "hey"})
    assert isinstance(evt, TextFinal)
    assert evt.text == "hey"
    assert evt.source == "user_typed"


@test("stream", "outbound AttachmentRefs hide local CAS paths except for trusted bridges")
async def t_outbound_attachment_paths_are_internal(ctx: TestContext) -> None:
    from src.stream.events import OutTextFinal
    from src.stream.wire import event_to_wire

    ref = {
        "type": "file",
        "kind": "file",
        "path": "/private/agent/artifacts/sha256/aa/hash",
        "filename": "report.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 4,
        "sha256": "a" * 64,
        "artifact_id": "art_1",
        "url": "/api/artifacts/art_1/content",
    }
    event = OutTextFinal(
        session_id="s",
        text="done",
        attachments=(ref,),
        parts=({"kind": "attachment", "attachment": ref},),
    )
    public = event_to_wire(event)
    assert "path" not in public["attachments"][0]
    assert "path" not in public["parts"][0]["attachment"]
    assert public["attachments"][0]["artifact_id"] == "art_1"
    internal = event_to_wire(event, include_local_attachment_paths=True)
    assert internal["attachments"][0]["path"].startswith("/private/")


@test("stream", "unknown wire types decode to None")
async def t_unknown_wire(ctx: TestContext) -> None:
    from src.stream.wire import wire_to_event

    assert wire_to_event({"type": "auth"}) is None
    assert wire_to_event({"type": "lol_what"}) is None
    assert wire_to_event({}) is None


@test("stream", "session_open emits explicit 0 on the wire (encoder side of the 3-state contract)")
async def t_session_open_emits_explicit_zero(ctx: TestContext) -> None:
    """The decoder distinguishes None / 0 / positive (see
    ``t_session_open_coalesce_default``); pin the encoder side too. A
    future "optimization" that drops 0 from the JSON would silently
    flip explicit opt-out sessions back to the default coalesce."""
    from src.stream.events import SessionOpen
    from src.stream.wire import event_to_wire

    explicit_zero = event_to_wire(SessionOpen(
        session_id="s", seq=1, ts_ms=10, coalesce_window_ms=0,
    ))
    assert explicit_zero["coalesce_window_ms"] == 0, explicit_zero

    explicit_positive = event_to_wire(SessionOpen(
        session_id="s", seq=1, ts_ms=10, coalesce_window_ms=750,
    ))
    assert explicit_positive["coalesce_window_ms"] == 750, explicit_positive

    server_default = event_to_wire(SessionOpen(
        session_id="s", seq=1, ts_ms=10, coalesce_window_ms=None,
    ))
    assert server_default["coalesce_window_ms"] is None, server_default


@test("stream", "session_open without coalesce_window_ms decodes to None (use default)")
async def t_session_open_coalesce_default(ctx: TestContext) -> None:
    """Regression: an absent ``coalesce_window_ms`` on the wire must
    decode to ``None`` so ``StreamSession`` falls back to its built-in
    default. The previous codec coerced missing/null to ``0``, which
    silently disabled coalescence on every webapp-opened session and
    made spam preempt the in-flight turn."""
    from src.stream.events import SessionOpen
    from src.stream.wire import wire_to_event

    # Missing field
    evt = wire_to_event({"type": "session_open", "session_id": "s1"})
    assert isinstance(evt, SessionOpen)
    assert evt.coalesce_window_ms is None

    # Explicit null
    evt = wire_to_event({
        "type": "session_open", "session_id": "s1",
        "coalesce_window_ms": None,
    })
    assert isinstance(evt, SessionOpen)
    assert evt.coalesce_window_ms is None

    # Explicit 0 — opt-out path, must NOT collapse to None.
    evt = wire_to_event({
        "type": "session_open", "session_id": "s1",
        "coalesce_window_ms": 0,
    })
    assert isinstance(evt, SessionOpen)
    assert evt.coalesce_window_ms == 0

    # Explicit positive int
    evt = wire_to_event({
        "type": "session_open", "session_id": "s1",
        "coalesce_window_ms": 750,
    })
    assert isinstance(evt, SessionOpen)
    assert evt.coalesce_window_ms == 750


@test("stream", "StreamSession picks DEFAULT_COALESCE_WINDOW_MS when constructor receives None")
async def t_session_default_coalesce(ctx: TestContext) -> None:
    """The wire→session bridge in the gateway passes ``None`` whenever
    the client didn't carry an explicit value; ``StreamSession`` must
    translate that to its compiled-in default rather than 0."""
    from src.stream.session import StreamSession

    class _Stub: pass

    sess = StreamSession(_Stub(), client_id="c", session_id="s")
    assert sess.coalesce_window_ms == StreamSession.DEFAULT_COALESCE_WINDOW_MS

    sess2 = StreamSession(_Stub(), client_id="c", session_id="s",
                          coalesce_window_ms=None)
    assert sess2.coalesce_window_ms == StreamSession.DEFAULT_COALESCE_WINDOW_MS

    sess3 = StreamSession(_Stub(), client_id="c", session_id="s",
                          coalesce_window_ms=0)
    assert sess3.coalesce_window_ms == 0  # explicit opt-out preserved

    sess4 = StreamSession(_Stub(), client_id="c", session_id="s",
                          coalesce_window_ms=900)
    assert sess4.coalesce_window_ms == 900


# ── factory dispatch ────────────────────────────────────────────────


@test("stream", "resolve_tts returns LocalPiperTTS when no DB row")
async def t_resolve_tts_local(ctx: TestContext) -> None:
    from src.channels.tts_base import LocalPiperTTS, resolve_tts
    from src.channels import tts_local

    if not tts_local.is_available():
        from ._framework import TestSkip
        raise TestSkip("piper not installed")

    tts = await resolve_tts(db=None)
    assert isinstance(tts, LocalPiperTTS), f"got {type(tts).__name__}"


@test("stream", "resolve_tts returns ElevenLabsWSTTS when row opts in")
async def t_resolve_tts_elevenlabs_ws(ctx: TestContext) -> None:
    from src.channels.tts_base import ElevenLabsWSTTS, resolve_tts

    class _StubDB:
        async def latest_audio_model(self, kind: str):
            assert kind == "tts"
            return {
                "provider_name": "elevenlabs",
                "model": "eleven_flash_v2_5",
                "metadata": {"voice_id": "Rachel", "stream_input": True},
                "api_key": "k",
                "base_url": None,
            }

    tts = await resolve_tts(_StubDB())
    assert isinstance(tts, ElevenLabsWSTTS), f"got {type(tts).__name__}"


# ── stream session smoke ────────────────────────────────────────────


class _FakeAgent:
    """Minimal stand-in for ``Agent`` — yields a fixed delta sequence."""

    name = "fake"
    db = None

    def __init__(self, deltas: list[str]):
        self._deltas = deltas

    async def run_stream(self, *, message, user_id, session_id,
                         attachments=None, on_status=None, author=None):
        for d in self._deltas:
            yield {"kind": "delta", "text": d}
        yield {"kind": "done", "text": "".join(self._deltas)}

    def last_response_meta(self, session_id: str) -> dict:
        return {"model": "fake-model"}


@test("stream", "StreamSession.run_one_shot pumps deltas and TurnComplete")
async def t_run_one_shot(ctx: TestContext) -> None:
    from src.stream.events import OutTextDelta, OutTextFinal, TurnComplete
    from src.stream.session import StreamSession

    agent = _FakeAgent(["he", "llo"])
    sess = StreamSession(
        agent, client_id="c", session_id="s", language=None,
    )
    summary = await sess.run_one_shot("hi", speak=False)
    assert summary["text"] == "hello", summary

    out = []
    while not sess.outbound.empty():
        out.append(sess.outbound.get_nowait())

    deltas = [e for e in out if isinstance(e, OutTextDelta)]
    finals = [e for e in out if isinstance(e, OutTextFinal)]
    completes = [e for e in out if isinstance(e, TurnComplete)]

    assert "".join(d.text for d in deltas) == "hello"
    assert finals and finals[-1].text == "hello"
    assert finals[-1].model == "fake-model"
    assert completes, "expected a TurnComplete event"
    assert isinstance(out[-1], TurnComplete), f"TurnComplete must be last; got {out[-1]!r}"


@test("stream", "OA-UI marker is hidden across deltas and emitted as an ordered part")
async def t_ui_marker_stream_parts(ctx: TestContext) -> None:
    import tempfile
    from pathlib import Path

    from src.core.on_behalf_context import OnBehalfIdentity
    from src.custom_views.repository import CustomViewRepository
    from src.memory.db import MemoryDB
    from src.memory.operational.access import AccessContext
    from src.stream.events import OutTextDelta, OutTextFinal
    from src.stream.session import StreamSession

    # A syntactically valid marker is not sufficient: the StreamSession must
    # resolve an authorized, revision-pinned inline View for this exact
    # session before emitting a structured part.
    with tempfile.TemporaryDirectory(prefix="oa-ui-stream-") as raw:
        db = MemoryDB(str(Path(raw) / "agent.db"))
        await db.connect()
        try:
            access = AccessContext(
                tenant_id="tenant-a",
                principal_id="user:alice",
                principal_type="user",
                handle="alice",
                device_id="device-a",
                principal_ids=frozenset({"user:alice", "user:device-a", "device:device-a"}),
                grant_identities=frozenset({("user", "alice"), ("device", "device-a")}),
            )
            view = await CustomViewRepository(db).create(
                access,
                surface="inline",
                title="Status board",
                session_id="s",
                spec={
                    "schemaVersion": 1,
                    "root": {"type": "text", "props": {"text": "Ready"}},
                },
            )
            marker = f"AGENT_UI:{view['id']}@1] now."
            agent = _FakeAgent(["Dashboard ready [OPEN", marker])
            agent.db = db
            sess = StreamSession(
                agent,
                client_id="c",
                session_id="s",
                client_kind="app",
                client_capabilities={"inline_ui": True, "ordered_parts": True},
                on_behalf_identity=OnBehalfIdentity(
                    "tenant-a", "user", "alice", "device-a"
                ),
            )
            summary = await sess.run_one_shot("build it", speak=False)
            assert summary["text"] == "Dashboard ready  now.", summary

            out = await _drain(sess)
            deltas = "".join(e.text for e in out if isinstance(e, OutTextDelta))
            final = next(e for e in out if isinstance(e, OutTextFinal))
            assert "OPENAGENT_UI" not in deltas, deltas
            assert [p["kind"] for p in final.parts] == ["text", "ui_view", "text"]
            assert final.parts[1]["view_id"] == view["id"]
        finally:
            await db.close()


@test("stream", "text-only client strips OA-UI marker without receiving a UI part")
async def t_ui_marker_text_only_client(ctx: TestContext) -> None:
    from src.stream.events import OutTextDelta, OutTextFinal
    from src.stream.session import StreamSession

    sess = StreamSession(
        _FakeAgent(["Created [OPENAGENT_UI:board@1]"]),
        client_id="c",
        session_id="s",
        client_kind="cli",
        client_capabilities={"attachments": True, "inline_ui": False},
    )
    await sess.run_one_shot("build it", speak=False)
    out = await _drain(sess)
    assert all("OPENAGENT_UI" not in e.text for e in out if isinstance(e, OutTextDelta))
    final = next(e for e in out if isinstance(e, OutTextFinal))
    assert all(p["kind"] != "ui_view" for p in final.parts)


@test("stream", "split UI and attachment carriers never reach live text or TTS")
async def t_all_content_markers_hidden_from_stream_and_tts(ctx: TestContext) -> None:
    from src.channels.tts_base import BaseTTS
    from src.stream.events import OutTextDelta
    from src.stream.session import StreamSession

    spoken: list[str] = []

    class _RecordingTTS(BaseTTS):
        @property
        def audio_format(self):
            return "wav", "audio/wav"

        @property
        def voice_id(self):
            return "test"

        async def synthesize_full(self, text, *, language=None):
            return b""

        async def synthesize_stream(self, text_chunks, *, language=None):
            async for piece in text_chunks:
                spoken.append(piece)
            if False:  # keep this an async generator without emitting audio
                yield b""

    async def _tts_factory(_db):
        return _RecordingTTS()

    async def _stt_factory(_db):
        return None

    agent = _FakeAgent([
        "Before [FI",
        "LE:/tmp/report.pdf] [IM",
        "AGE:/tmp/photo.png] [VO",
        "ICE:/tmp/reply.ogg] [VID",
        "EO:/tmp/demo.mp4] [OPEN",
        "AGENT_UI:board@1] after",
    ])
    sess = StreamSession(agent, client_id="c", session_id="s", language=None)
    await sess.start(stt_factory=_stt_factory, tts_factory=_tts_factory)
    try:
        summary = await sess.run_one_shot("show it", speak=True)
    finally:
        await sess.close()

    out = await _drain(sess)
    visible = "".join(e.text for e in out if isinstance(e, OutTextDelta))
    for value in (visible, "".join(spoken), summary["text"]):
        assert "Before" in value and "after" in value, value
        assert "/tmp/" not in value, value
        assert "OPENAGENT_UI" not in value, value
        assert "[FILE:" not in value, value


# ── Reasoning flag (server emits a boolean, never a "Thinking…" string) ─

class _StatusAgent:
    """Fake agent that drives ``on_status`` with a scripted sequence
    (plain thinking strings + tool JSON) then yields deltas, so we can
    pin how ``StreamTurnRunner`` translates statuses into the typed
    reasoning flag vs. tool-status frames."""

    name = "statusy"
    db = None

    def __init__(self, script, deltas):
        self._script = script   # list of on_status strings, in order
        self._deltas = deltas

    async def run_stream(self, *, message, user_id, session_id,
                         attachments=None, on_status=None, author=None):
        for s in self._script:
            if on_status:
                await on_status(s)
        for d in self._deltas:
            yield {"kind": "delta", "text": d}
        yield {"kind": "done", "text": "".join(self._deltas)}

    def last_response_meta(self, session_id: str) -> dict:
        return {"model": "fake-model"}


async def _drain(sess):
    out = []
    while not sess.outbound.empty():
        out.append(sess.outbound.get_nowait())
    return out


@test("stream", "plain 'Thinking…' status becomes OutReasoning(true/false), never an OutToolStatus string")
async def t_reasoning_translation(ctx: TestContext) -> None:
    from src.stream.events import OutReasoning, OutToolStatus, OutTextDelta
    from src.stream.session import StreamSession

    # Loading→Thinking (UI strings) then a tool (data) then the answer.
    agent = _StatusAgent(
        script=[
            "Loading context...",
            "Thinking...",
            '{"tool_name":"bash","tool_call_error":false}',
        ],
        deltas=["the ", "answer"],
    )
    sess = StreamSession(agent, client_id="c", session_id="s")
    await sess.run_one_shot("hi", speak=False)
    out = await _drain(sess)

    reasoning = [e for e in out if isinstance(e, OutReasoning)]
    tools = [e for e in out if isinstance(e, OutToolStatus)]

    # Exactly one true then one false — the two plain strings collapse to a
    # single active=True; the tool start flips it back to active=False.
    assert [e.active for e in reasoning] == [True, False], reasoning
    # The plain UI strings NEVER leak onto the wire as tool-status text.
    assert all(t.text not in ("Thinking...", "Loading context...") for t in tools), tools
    # The tool JSON DID survive as data.
    assert any('"tool_name"' in t.text for t in tools), tools
    # Reasoning(true) precedes the first delta; reasoning(false) precedes it too.
    first_delta_idx = next(i for i, e in enumerate(out) if isinstance(e, OutTextDelta))
    true_idx = out.index(reasoning[0])
    false_idx = out.index(reasoning[1])
    assert true_idx < false_idx < first_delta_idx, (true_idx, false_idx, first_delta_idx)


@test("stream", "tool-free turn: reasoning ends on the first delta")
async def t_reasoning_ends_on_first_delta(ctx: TestContext) -> None:
    from src.stream.events import OutReasoning, OutTextDelta
    from src.stream.session import StreamSession

    agent = _StatusAgent(script=["Thinking..."], deltas=["hello"])
    sess = StreamSession(agent, client_id="c", session_id="s")
    await sess.run_one_shot("hi", speak=False)
    out = await _drain(sess)

    reasoning = [e for e in out if isinstance(e, OutReasoning)]
    assert [e.active for e in reasoning] == [True, False], reasoning
    # active=False lands immediately before the first visible token.
    false_idx = out.index(reasoning[1])
    first_delta_idx = next(i for i, e in enumerate(out) if isinstance(e, OutTextDelta))
    assert false_idx < first_delta_idx, (false_idx, first_delta_idx)


@test("stream", "tool-only / empty turn still terminates reasoning at turn end")
async def t_reasoning_safety_net_on_empty(ctx: TestContext) -> None:
    from src.stream.events import OutReasoning, TurnComplete
    from src.stream.session import StreamSession

    # Thinking fires but the model yields no deltas (tool-only / empty).
    agent = _StatusAgent(script=["Thinking..."], deltas=[])
    sess = StreamSession(agent, client_id="c", session_id="s")
    await sess.run_one_shot("hi", speak=False)
    out = await _drain(sess)

    reasoning = [e for e in out if isinstance(e, OutReasoning)]
    assert [e.active for e in reasoning] == [True, False], reasoning
    # The closing active=False precedes TurnComplete (safety-net path).
    false_idx = out.index(reasoning[-1])
    tc_idx = next(i for i, e in enumerate(out) if isinstance(e, TurnComplete))
    assert false_idx < tc_idx, (false_idx, tc_idx)


@test("stream", "session.compacted envelope becomes typed SessionCompacted frames, not tool JSON")
async def t_compaction_translation(ctx: TestContext) -> None:
    """The turn runner must lift a ``session.compacted`` on_status envelope
    into typed ``SessionCompacted`` frames (running → done) and NEVER let
    the raw JSON leak onto the wire as an ``OutToolStatus`` string — that
    raw-JSON leak was the "displayed poorly" bug this feature fixes.
    """
    import json as _json
    from src.stream.events import SessionCompacted, OutToolStatus
    from src.stream.session import StreamSession

    running = _json.dumps({
        "kind": "session.compacted", "phase": "running",
        "folded_runs": 3, "kept_runs_count": 2, "tokens_before": 900,
    })
    done = _json.dumps({
        "kind": "session.compacted", "phase": "done",
        "folded_runs": 3, "kept_runs_count": 2, "summary_chars": 120,
        "tokens_before": 900, "tokens_after": 60,
    })
    agent = _StatusAgent(script=[running, done], deltas=["ok"])
    sess = StreamSession(agent, client_id="c", session_id="s")
    await sess.run_one_shot("hi", speak=False)
    out = await _drain(sess)

    comp = [e for e in out if isinstance(e, SessionCompacted)]
    assert [c.phase for c in comp] == ["running", "done"], comp
    assert comp[0].folded_runs == 3, comp[0]
    assert comp[0].tokens_before == 900, comp[0]
    assert comp[1].summary_chars == 120, comp[1]
    assert comp[1].tokens_after == 60, comp[1]
    # The raw envelope never ships as tool-status text.
    tools = [e for e in out if isinstance(e, OutToolStatus)]
    assert all("session.compacted" not in t.text for t in tools), tools


@test("stream", "StreamSession.run_one_shot never ships an empty final response")
async def t_run_one_shot_empty_reply_gets_fallback(ctx: TestContext) -> None:
    from src.stream.events import OutTextFinal
    from src.stream.session import StreamSession

    sess = StreamSession(_FakeAgent([]), client_id="c", session_id="s")
    summary = await sess.run_one_shot("hi", speak=False)

    assert "No text response" in summary["text"], summary

    out = []
    while not sess.outbound.empty():
        out.append(sess.outbound.get_nowait())

    finals = [e for e in out if isinstance(e, OutTextFinal)]
    assert finals, out
    assert "No text response" in finals[-1].text, finals[-1]


@test("stream", "StreamSession.run_one_shot finalizes unexpected cancellation")
async def t_run_one_shot_unexpected_cancel_gets_terminal_frame(ctx: TestContext) -> None:
    from src.stream.events import OutTextFinal, TurnComplete
    from src.stream.session import StreamSession

    class _CancelledAgent:
        name = "cancelled"
        db = None

        async def run_stream(self, *, message, user_id, session_id,
                             attachments=None, on_status=None, author=None):
            raise asyncio.CancelledError()
            yield  # pragma: no cover - keeps this an async generator

        def last_response_meta(self, session_id: str) -> dict:
            return {"model": "cancelled-model"}

    sess = StreamSession(_CancelledAgent(), client_id="c", session_id="s")
    summary = await sess.run_one_shot("hi", speak=False)

    assert "interrupted before a response" in summary["text"], summary

    out = []
    while not sess.outbound.empty():
        out.append(sess.outbound.get_nowait())

    finals = [e for e in out if isinstance(e, OutTextFinal)]
    completes = [e for e in out if isinstance(e, TurnComplete)]
    assert finals, out
    assert "interrupted before a response" in finals[-1].text, finals[-1]
    assert completes, out


@test("stream", "BatchedChannel collapses one turn into a finished reply")
async def t_batched_channel(ctx: TestContext) -> None:
    from src.stream.channel import BatchedChannel
    from src.stream.session import StreamSession

    async def _null(_db):
        return None

    agent = _FakeAgent(["foo ", "bar"])
    sess = StreamSession(agent, client_id="c", session_id="s")
    # Start the dispatch loop so ``BatchedChannel.run_one_shot`` drives the
    # turn end-to-end through the real inbound -> dispatch -> outbound path,
    # exactly like a bridge does in production. Drive it through the channel
    # ALONE — do NOT also call ``sess.run_one_shot`` concurrently. The old
    # test raced the channel's startup drain against a manual direct-path
    # turn: if the manual turn published its frames (incl. TurnComplete)
    # before the channel reached its drain-then-consume loop, the drain ate
    # the reply and ``run_one_shot`` blocked until the 5s timeout. Ordering
    # was environment-dependent (green on macOS, red on the ubuntu CI runner).
    await sess.start(stt_factory=_null, tts_factory=_null)
    channel = BatchedChannel(sess)

    reply = await asyncio.wait_for(channel.run_one_shot("ping"), timeout=5.0)

    assert reply.text == "foo bar", reply
    assert reply.audio_bytes is None
    assert reply.model == "fake-model"

    await sess.close()


@test("stream", "wire codec drops binary payloads losslessly via base64")
async def t_wire_binary(ctx: TestContext) -> None:
    from src.stream.events import OutAudioChunk
    from src.stream.wire import event_to_wire, wire_to_event

    payload = bytes(range(256))
    evt = OutAudioChunk(session_id="s", seq=1, ts_ms=1, data=payload)
    wire = event_to_wire(evt)
    back = wire_to_event(wire)
    assert isinstance(back, OutAudioChunk) and back.data == payload


@test("stream", "OutAudioChunk seq starts at 1 per audio span (player invariant)")
async def t_audio_chunk_seq_per_span(ctx: TestContext) -> None:
    """The universal app's ``AudioQueuePlayer`` (audioPlayer.ts) reads
    ``msg.seq`` and waits for ``nextSeq=1`` before playing. If we emit
    audio chunks with the session-wide ``next_seq()`` counter, the
    first audio chunk arrives at seq=N (after text deltas + AudioStart
    bumped the counter), the player never sees seq=1, and the user
    hears nothing. Pin the contract: ``OutAudioChunk.seq`` must count
    1, 2, 3, ... within a single audio span."""
    from src.channels.tts_base import BaseTTS
    from src.stream.events import OutAudioChunk
    from src.stream.session import StreamSession

    class _NoiseTTS(BaseTTS):
        @property
        def audio_format(self):
            return "wav", "audio/wav"

        @property
        def voice_id(self):
            return "test-voice"

        async def synthesize_full(self, text, *, language=None):
            return b"\x00\x01" * 8

        async def synthesize_stream(self, text_chunks, *, language=None):
            # Force three discrete audio chunks regardless of input.
            async for _ in text_chunks:
                pass
            yield b"AAAA"
            yield b"BBBB"
            yield b"CCCC"

    async def _tts_factory(_db):
        return _NoiseTTS()

    async def _stt_factory(_db):
        return None

    sess = StreamSession(
        _FakeAgent(["he", "llo"]),
        client_id="c", session_id="s", language=None,
    )
    await sess.start(stt_factory=_stt_factory, tts_factory=_tts_factory)
    try:
        await sess.run_one_shot("hi", speak=True)
    finally:
        await sess.close()

    chunks = []
    while not sess.outbound.empty():
        evt = sess.outbound.get_nowait()
        if isinstance(evt, OutAudioChunk):
            chunks.append(evt)

    assert len(chunks) == 3, f"expected 3 audio chunks, got {len(chunks)}"
    seqs = [c.seq for c in chunks]
    assert seqs == [1, 2, 3], (
        f"audio chunk seq must be 1,2,3 (audioPlayer.ts contract); got {seqs}"
    )


# ── PCM → WAV wrapping in BaseSTT default stream ───────────────────


@test("stream", "BaseSTT.stream wraps PCM chunks in a valid WAV header")
async def t_basestt_pcm_to_wav(ctx: TestContext) -> None:
    """When the client streams raw 16-bit PCM (the AudioWorklet path),
    the BaseSTT default ``stream`` must concatenate the chunks and
    prepend a RIFF/WAVE header so faster-whisper / litellm can parse
    the resulting tempfile. Verifies the header bytes + that the data
    chunk equals the original PCM input."""
    import io
    import math
    import struct
    import wave

    from src.channels.stt_base import BaseSTT, STTEvent

    # Synthesize 200 ms of a 1 kHz sine at 16 kHz mono.
    sample_rate = 16000
    duration_s = 0.2
    n = int(sample_rate * duration_s)
    pcm_samples = bytearray()
    for i in range(n):
        s = int(0.5 * 32767 * math.sin(2 * math.pi * 1000 * i / sample_rate))
        pcm_samples.extend(struct.pack("<h", s))
    expected_pcm = bytes(pcm_samples)

    captured: dict = {}

    class _RecordingSTT(BaseSTT):
        async def transcribe_file(self, path, *, language=None):
            captured["path"] = path
            with open(path, "rb") as fh:
                captured["bytes"] = fh.read()
            return "ok"

    async def _pcm_iter():
        # Send in 4 sub-chunks to prove concat works.
        chunk_size = max(1, len(expected_pcm) // 4)
        for i in range(0, len(expected_pcm), chunk_size):
            yield expected_pcm[i:i + chunk_size]

    stt = _RecordingSTT()
    events: list[STTEvent] = []
    async for ev in stt.stream(
        _pcm_iter(), language="en", encoding="pcm16", sample_rate=sample_rate,
    ):
        events.append(ev)

    assert events and events[-1].text == "ok", events
    assert captured["path"].endswith(".wav"), captured["path"]
    written = captured["bytes"]
    assert written[:4] == b"RIFF", written[:16]
    assert written[8:12] == b"WAVE", written[8:16]

    with wave.open(io.BytesIO(written), "rb") as wf:
        assert wf.getframerate() == sample_rate
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        data = wf.readframes(wf.getnframes())
    assert data == expected_pcm, "data chunk does not match input PCM"


@test("stream", "BaseSTT.stream container path still writes raw chunks")
async def t_basestt_container_path(ctx: TestContext) -> None:
    """Non-PCM encodings (webm, mp4, ogg) must keep the original
    behaviour: write each chunk verbatim, no WAV header injection."""
    from src.channels.stt_base import BaseSTT

    captured: dict = {}

    class _RecordingSTT(BaseSTT):
        async def transcribe_file(self, path, *, language=None):
            with open(path, "rb") as fh:
                captured["bytes"] = fh.read()
            return "ok"

    payload = b"\x1aE\xdf\xa3FAKE-WEBM-BYTES"

    async def _webm_iter():
        yield payload

    stt = _RecordingSTT()
    out = []
    async for ev in stt.stream(_webm_iter(), encoding="webm"):
        out.append(ev)
    assert captured["bytes"] == payload, "container path must not mangle bytes"


@test("stream", "StreamSession dispatch propagates encoding + sample_rate to STT")
async def t_dispatch_pcm_propagation(ctx: TestContext) -> None:
    """The ``encoding`` + ``sample_rate`` fields on AudioChunk events
    must reach ``stt.stream(...)`` so Deepgram sees ``linear16`` and
    BaseSTT builds the right WAV header."""
    import asyncio as _aio

    from src.channels.stt_base import BaseSTT, STTEvent
    from src.stream.events import AudioChunk, now_ms
    from src.stream.session import StreamSession

    seen: dict = {}

    class _StubSTT(BaseSTT):
        supports_streaming = True

        async def transcribe_file(self, path, *, language=None):
            return None

        async def stream(self, audio_in, *, language=None, encoding="webm",
                         sample_rate=None):
            seen["encoding"] = encoding
            seen["sample_rate"] = sample_rate
            async for _ in audio_in:
                pass
            yield STTEvent(kind="final", text="hi")

    class _FakeAgent:
        name = "fake"
        db = None

        async def run_stream(self, *, message, user_id, session_id,
                             attachments=None, on_status=None, author=None):
            yield {"kind": "done", "text": ""}

        def last_response_meta(self, sid):
            return {"model": "fake"}

    async def _stt_factory(_db):
        return _StubSTT()

    async def _null(_db):
        return None

    sess = StreamSession(_FakeAgent(), client_id="c", session_id="s")
    await sess.start(stt_factory=_stt_factory, tts_factory=_null)
    try:
        await sess.push_in(AudioChunk(
            session_id="s", seq=1, ts_ms=now_ms(),
            data=b"\x00\x00" * 100, encoding="pcm16", sample_rate=16000,
        ))
        await sess.push_in(AudioChunk(
            session_id="s", seq=2, ts_ms=now_ms(),
            data=b"", end_of_speech=True,
        ))
        for _ in range(40):
            await _aio.sleep(0.05)
            if seen:
                break
    finally:
        await sess.close()

    assert seen.get("encoding") == "pcm16", seen
    assert seen.get("sample_rate") == 16000, seen


# ── input coalescence (debounce) ───────────────────────────────────


class _RecordingAgent:
    """Records every ``run_stream`` invocation; can hold the first call open.

    Used by the coalescence tests to put a turn ``in flight`` so the next
    user input lands on the buffer/cancel arm of ``_on_user_turn_complete``.
    Subsequent calls (the merged-burst dispatch) return immediately so the
    test doesn't have to coordinate two release events.

    Yields an empty delta IMMEDIATELY before any blocking — this mirrors
    what real api-based providers signal once the prompt has
    actually been delivered to the SDK, which is the engagement signal
    ``StreamTurnRunner`` uses to flip ``_current_turn_started=True`` and
    take the partial-commit path on cancel rather than salvaging the
    input. Without this the salvage would re-buffer the test's first
    message even though the test agent has already "received" it.
    """

    name = "recording"
    db = None

    def __init__(self, *, block_first: bool = False) -> None:
        self.calls: list[dict] = []
        self.release = asyncio.Event()
        self.block_first = block_first
        self._idx = 0

    async def run_stream(self, *, message, user_id, session_id,
                         attachments=None, on_status=None, author=None):
        self._idx += 1
        self.calls.append({
            "message": message,
            "attachments": list(attachments or []),
        })
        # Engagement signal — see class docstring.
        yield {"kind": "delta", "text": ""}
        if self.block_first and self._idx == 1:
            # Block until the test signals release OR the runner cancels
            # us (the barge-in path). CancelledError must propagate so the
            # runner's finally block runs.
            await self.release.wait()
        yield {"kind": "done", "text": ""}

    def last_response_meta(self, sid):
        return {"model": "recording"}


async def _wait_for(condition, *, timeout: float = 1.0, step: float = 0.01):
    """Poll ``condition()`` until truthy or timeout. Returns the value."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        v = condition()
        if v:
            return v
        await asyncio.sleep(step)
    return condition()


def _make_session(agent, **kwargs):
    from src.stream.session import StreamSession
    return StreamSession(agent, client_id="c", session_id="s", **kwargs)


@test("stream", "coalesce explicitly off preserves preempt-on-each-message")
async def t_coalesce_explicitly_off(ctx: TestContext) -> None:
    """Passing ``coalesce_window_ms=0`` explicitly must keep the legacy
    behaviour: each new TextFinal preempts the previous and dispatches
    as its own turn — no buffering, no merging. (The class default is
    now 500 ms; this test guards the explicit-disable escape hatch.)"""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=0)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1)
        await sess.push_in(TextFinal(
            session_id="s", seq=2, ts_ms=now_ms(), text="B", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 2)
    finally:
        agent.release.set()
        await sess.close()

    msgs = [c["message"] for c in agent.calls]
    assert msgs == ["A", "B"], (
        f"with coalesce off, each push dispatches its own turn; got {msgs}"
    )


@test("stream", "two TextFinals during in-flight turn merge into one turn")
async def t_coalesce_merge_two(ctx: TestContext) -> None:
    """With a 200 ms window, two TextFinals arriving 50 ms apart while
    a turn is in flight must dispatch as a SINGLE merged turn whose
    text is ``"first\\n\\nsecond"`` — one barge-in, one merged dispatch."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=200)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1)
        await sess.push_in(TextFinal(
            session_id="s", seq=2, ts_ms=now_ms(), text="B", source="user_typed",
        ))
        await asyncio.sleep(0.05)
        await sess.push_in(TextFinal(
            session_id="s", seq=3, ts_ms=now_ms(), text="C", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 2, timeout=2.0)
    finally:
        agent.release.set()
        await sess.close()

    msgs = [c["message"] for c in agent.calls]
    assert msgs == ["A", "B\n\nC"], (
        f"merged burst should dispatch one turn with joined text; got {msgs}"
    )


@test("stream", "burst extends while inputs keep arriving (5-message window)")
async def t_coalesce_extends(ctx: TestContext) -> None:
    """Inputs landing within the window keep restarting the timer. Five
    TextFinals at 50 ms intervals (span 200 ms) inside a 200 ms window
    must collapse to ONE merged turn containing all 5."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=200)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1)
        for letter in ("B", "C", "D", "E", "F"):
            await sess.push_in(TextFinal(
                session_id="s", seq=10, ts_ms=now_ms(),
                text=letter, source="user_typed",
            ))
            await asyncio.sleep(0.05)
        await _wait_for(lambda: len(agent.calls) >= 2, timeout=2.0)
    finally:
        agent.release.set()
        await sess.close()

    msgs = [c["message"] for c in agent.calls]
    assert msgs == ["A", "B\n\nC\n\nD\n\nE\n\nF"], (
        f"5-message burst should merge into one turn; got {msgs}"
    )


@test("stream", "isolated typed message dispatches via the debounce window")
async def t_coalesce_isolated_through_window(ctx: TestContext) -> None:
    """All typed messages funnel through the debounce buffer, even when
    no turn is in flight — that's what makes a 3-message burst land as
    ONE merged turn instead of "first dispatched + rest merged" (which
    leaves the first message orphaned in the agent's history). The cost
    is one ``coalesce_window_ms`` of latency on a quiet single send."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=False)
    window_ms = 200
    sess = _make_session(agent, coalesce_window_ms=window_ms)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="solo", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1, timeout=1.5)
        elapsed = loop.time() - t0
    finally:
        await sess.close()

    # Dispatch happens after the window — we accept a generous upper bound
    # because asyncio.sleep + dispatch lock + task scheduling adds jitter.
    assert elapsed >= window_ms / 1000.0 * 0.8, (
        f"isolated typed message should wait the {window_ms}ms window; "
        f"took {elapsed*1000:.0f}ms"
    )
    assert agent.calls and agent.calls[0]["message"] == "solo"
    assert sess._pending_burst == [], "buffer should drain after dispatch"


@test("stream", "first barge-in cancels in-flight; subsequent burst inputs do not re-cancel")
async def t_coalesce_single_cancel(ctx: TestContext) -> None:
    """``_cancel_active_turn`` is the expensive bit (it commits partial
    assistant text, awaits task cleanup). The coalescence path must call
    it exactly once per burst — the first input cancels, all subsequent
    inputs in the same window only extend the buffer."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=200)

    cancel_count = 0
    original_cancel = sess._cancel_active_turn

    async def _counting_cancel(*, reason: str = "manual",
                                suppress_completion: bool = False,
                                salvage_to_burst: bool = False):
        nonlocal cancel_count
        # Only count real cancellations — ``close()`` calls
        # ``_cancel_active_turn`` defensively even when there's nothing
        # to cancel (current_turn is None or already done).
        task = sess._current_turn
        if task is not None and not task.done():
            cancel_count += 1
        await original_cancel(
            reason=reason,
            suppress_completion=suppress_completion,
            salvage_to_burst=salvage_to_burst,
        )

    sess._cancel_active_turn = _counting_cancel  # type: ignore[method-assign]

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1)
        # Three more messages within the window — first one cancels, the
        # next two should only extend the timer.
        for letter in ("B", "C", "D"):
            await sess.push_in(TextFinal(
                session_id="s", seq=10, ts_ms=now_ms(),
                text=letter, source="user_typed",
            ))
            await asyncio.sleep(0.04)
        await _wait_for(lambda: len(agent.calls) >= 2, timeout=2.0)
    finally:
        agent.release.set()
        await sess.close()

    assert cancel_count == 1, (
        f"expected exactly one cancel for the burst; got {cancel_count}"
    )


@test("stream", "Interrupt during burst clears buffer + timer")
async def t_coalesce_interrupt_clears(ctx: TestContext) -> None:
    """An explicit Interrupt is the user saying ``stop``. It must drop
    every buffered message + cancel the pending timer so no merged turn
    ever fires."""
    from src.stream.events import Interrupt, TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=200)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1)
        for letter in ("B", "C", "D"):
            await sess.push_in(TextFinal(
                session_id="s", seq=10, ts_ms=now_ms(),
                text=letter, source="user_typed",
            ))
            await asyncio.sleep(0.03)
        # Buffer should now hold B, C, D; timer scheduled.
        assert len(sess._pending_burst) == 3, sess._pending_burst
        await sess.push_in(Interrupt(
            session_id="s", seq=99, ts_ms=now_ms(), reason="manual",
        ))
        # Give the dispatch loop a tick to handle the Interrupt.
        await asyncio.sleep(0.05)
        # Wait well past the window — no merged dispatch should fire.
        await asyncio.sleep(0.3)
    finally:
        agent.release.set()
        await sess.close()

    assert sess._pending_burst == [], "Interrupt must drop the burst"
    assert sess._burst_timer is None, "Interrupt must cancel the timer"
    msgs = [c["message"] for c in agent.calls]
    assert msgs == ["A"], f"no merged turn should have fired; got {msgs}"


@test("stream", "close() during burst drops buffer cleanly")
async def t_coalesce_close_drops_burst(ctx: TestContext) -> None:
    """Tearing down a session mid-burst must drop the pending merged
    turn — the WS is going away and there's no consumer for the reply."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=200)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)

    await sess.push_in(TextFinal(
        session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
    ))
    await _wait_for(lambda: len(agent.calls) >= 1)
    for letter in ("B", "C", "D"):
        await sess.push_in(TextFinal(
            session_id="s", seq=10, ts_ms=now_ms(),
            text=letter, source="user_typed",
        ))
        await asyncio.sleep(0.03)
    assert len(sess._pending_burst) == 3
    # Release so the cancelled-turn cleanup can finish promptly.
    agent.release.set()
    await sess.close()
    # Give it well past the window — close should have killed the timer.
    await asyncio.sleep(0.3)

    msgs = [c["message"] for c in agent.calls]
    assert msgs == ["A"], (
        f"close() must drop pending burst; got {msgs}"
    )


@test("stream", "attachments union across burst messages")
async def t_coalesce_attachments_union(ctx: TestContext) -> None:
    """Each TextFinal in a burst carries its own attachments. The
    merged dispatch must see all of them concatenated in arrival order."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=200)

    async def _null(_db):
        return None

    a1 = {"type": "image", "path": "/tmp/a.jpg", "filename": "a.jpg"}
    a2 = {"type": "file", "path": "/tmp/b.txt", "filename": "b.txt"}
    a3 = {"type": "image", "path": "/tmp/c.png", "filename": "c.png"}

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1)
        for letter, att in (("B", a1), ("C", a2), ("D", a3)):
            await sess.push_in(TextFinal(
                session_id="s", seq=10, ts_ms=now_ms(),
                text=letter, source="user_typed", attachments=(att,),
            ))
            await asyncio.sleep(0.04)
        await _wait_for(lambda: len(agent.calls) >= 2, timeout=2.0)
    finally:
        agent.release.set()
        await sess.close()

    assert len(agent.calls) == 2, agent.calls
    merged_atts = agent.calls[1]["attachments"]
    # First three entries must be a1, a2, a3 in arrival order. (Trailing
    # entries may include video-frame snapshots — none in this test, but
    # keep the assertion shape forward-compatible.)
    assert merged_atts[:3] == [a1, a2, a3], (
        f"merged attachments should be union in arrival order; got {merged_atts}"
    )


@test("stream", "STT messages bypass the debounce window (instant barge-in)")
async def t_coalesce_stt_bypass(ctx: TestContext) -> None:
    """Voice (``source='stt'``) must dispatch immediately even when the
    debounce window is non-zero. This is what gives voice mode the
    OpenAI-Realtime feel — model stops the instant the user finishes
    speaking, no 500 ms wait."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=500)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        # First turn dispatched normally (typed). Blocks on release.
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        # STT message during in-flight: should bypass coalescence and
        # dispatch the new turn immediately, not after 500 ms.
        await sess.push_in(TextFinal(
            session_id="s", seq=2, ts_ms=now_ms(), text="stop", source="stt",
        ))
        await _wait_for(lambda: len(agent.calls) >= 2, timeout=1.0)
        elapsed = loop.time() - t0
    finally:
        agent.release.set()
        await sess.close()

    assert elapsed < 0.2, (
        f"STT must bypass the {sess.coalesce_window_ms}ms window; took {elapsed*1000:.0f}ms"
    )
    msgs = [c["message"] for c in agent.calls]
    assert msgs == ["A", "stop"], (
        f"STT bypass should preempt without merging; got {msgs}"
    )


@test("stream", "STT folds a buffered typed burst into the same merged turn")
async def t_coalesce_stt_folds_buffer(ctx: TestContext) -> None:
    """Mixed bursts: if the user typed B, C while the assistant was
    talking and THEN spoke ``"and also D"``, the voice command flushes
    the buffer instead of racing it. The merged turn carries
    ``"B\\n\\nC\\n\\nand also D"`` as one user message."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=500)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1)
        # Two typed messages buffer (cancel turn 1, start timer).
        await sess.push_in(TextFinal(
            session_id="s", seq=2, ts_ms=now_ms(), text="B", source="user_typed",
        ))
        await asyncio.sleep(0.04)
        await sess.push_in(TextFinal(
            session_id="s", seq=3, ts_ms=now_ms(), text="C", source="user_typed",
        ))
        await asyncio.sleep(0.04)
        assert sess._pending_burst, "typed burst must be buffered before STT arrives"
        # STT lands → must flush the buffer with itself merged in.
        await sess.push_in(TextFinal(
            session_id="s", seq=4, ts_ms=now_ms(),
            text="and also D", source="stt",
        ))
        await _wait_for(lambda: len(agent.calls) >= 2, timeout=1.0)
    finally:
        agent.release.set()
        await sess.close()

    msgs = [c["message"] for c in agent.calls]
    assert msgs == ["A", "B\n\nC\n\nand also D"], (
        f"STT must fold the typed buffer into the merged turn; got {msgs}"
    )
    assert sess._pending_burst == [], "buffer should have been flushed"


@test("stream", "post_turn_hook receives the resource set tracked from OutToolStatus")
async def t_post_turn_hook_resources(ctx: TestContext) -> None:
    """The gateway's resource-event broadcast pipes through this hook —
    pin the wiring: every ``OutToolStatus`` whose JSON ``tool_name`` field
    matches one of the known MCP prefixes adds to the per-turn set,
    which fires once on TurnComplete."""
    import json as _json

    from src.stream.events import OutToolStatus, TurnComplete, now_ms
    from src.stream.session import StreamSession

    sess = StreamSession(
        _RecordingAgent(), client_id="c", session_id="s", coalesce_window_ms=0,
    )
    seen: list[set[str]] = []

    async def _post(resources: set[str]) -> None:
        seen.append(set(resources))

    sess.post_turn_hook = _post

    # Drive _publish manually to avoid spinning up the full dispatch loop.
    await sess._publish(OutToolStatus(
        session_id="s", seq=1, ts_ms=now_ms(),
        text=_json.dumps({"tool_name": "scheduler_add_task",
                          "tool_call_error": False}),
    ))
    await sess._publish(OutToolStatus(
        session_id="s", seq=2, ts_ms=now_ms(),
        text=_json.dumps({"tool_name": "Bash", "tool_call_error": False}),
    ))
    await sess._publish(OutToolStatus(
        session_id="s", seq=3, ts_ms=now_ms(),
        text=_json.dumps({"tool_name": "mcp__workflow_manager__run",
                          "tool_call_error": False, "result": "ok"}),
    ))
    await sess._publish(TurnComplete(session_id="s", seq=4, ts_ms=now_ms()))

    assert seen == [{"scheduled_task", "workflow"}], (
        f"post_turn_hook should fire once with the union of MCP categories; got {seen}"
    )
    assert sess._turn_resources == set(), "accumulator must reset for the next turn"


@test("stream", "OutError on the wire resolves a bridge collector immediately")
async def t_collector_resolves_on_outerror(ctx: TestContext) -> None:
    """``fold_outbound_event`` returns True on OutError so a session-tagged
    error releases the awaiting bridge / CLI ``send_message`` even when
    the gateway never gets to publish a TurnComplete (turn died early)."""
    from src.stream.collector import StreamCollector, fold_outbound_event
    from src.stream.events import OutError, OutTextFinal, now_ms

    collector = StreamCollector()
    # OutTextFinal latches text but does NOT release.
    done = fold_outbound_event(collector, OutTextFinal(
        session_id="s", seq=1, ts_ms=now_ms(), text="partial"
    ))
    assert done is False
    assert not collector.done.is_set()

    # OutError releases immediately + flips errored.
    done = fold_outbound_event(collector, OutError(
        session_id="s", seq=2, ts_ms=now_ms(), text="boom"
    ))
    assert done is True
    assert collector.errored is True
    assert collector.error_text == "boom"
    reply = collector.to_legacy_reply()
    assert reply["type"] == "error"
    assert reply["text"] == "boom"


# ── barge-in completion suppression + drain race ────────────────────


@test("stream", "barge-in cancel suppresses cancelled-turn OutTextFinal + TurnComplete")
async def t_cancel_suppresses_completion(ctx: TestContext) -> None:
    """Regression for bug: typing during a streaming reply made the
    "Thinking…" indicator vanish for the debounce window because the
    cancelled runner published its own ``OutTextFinal`` + ``TurnComplete``
    before the merged turn dispatched. The session must drop those two
    frames whenever the cancel is followed by a follow-up turn — only
    intermediate frames (deltas, tool status, audio chunks) should
    survive across the cancel boundary."""
    from src.stream.events import (
        OutTextDelta, OutTextFinal, TextFinal, TurnComplete, now_ms,
    )

    class _ChattyAgent:
        name = "chatty"
        db = None

        def __init__(self) -> None:
            self.calls = 0
            self.allow_finish = asyncio.Event()

        async def run_stream(self, *, message, user_id, session_id,
                             attachments=None, on_status=None, author=None):
            self.calls += 1
            if self.calls == 1:
                # Stream a delta then block until the test signals OR
                # the runner cancels us. CancelledError must propagate
                # so the runner's finally block runs.
                yield {"kind": "delta", "text": "Hi"}
                await self.allow_finish.wait()
                yield {"kind": "done", "text": ""}
            else:
                yield {"kind": "delta", "text": "merged"}
                yield {"kind": "done", "text": ""}

        def last_response_meta(self, sid):
            return {"model": "chatty"}

    async def _null(_db):
        return None

    agent = _ChattyAgent()
    sess = _make_session(agent, coalesce_window_ms=200)
    await sess.start(stt_factory=_null, tts_factory=_null)

    seen_kinds: list[str] = []

    async def _drain_outbound() -> None:
        while True:
            evt = await sess.outbound.get()
            seen_kinds.append(type(evt).__name__)

    drain_task = asyncio.create_task(_drain_outbound())
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        # Wait until the first delta lands so we know the cancel will
        # happen mid-stream.
        await _wait_for(lambda: "OutTextDelta" in seen_kinds, timeout=2.0)
        # Barge in. This should cancel turn 1 with suppress=True.
        await sess.push_in(TextFinal(
            session_id="s", seq=2, ts_ms=now_ms(), text="B", source="user_typed",
        ))
        # Let the cancel + buffer settle.
        await asyncio.sleep(0.05)
        # Wait for the merged turn to dispatch and complete.
        await _wait_for(
            lambda: seen_kinds.count("TurnComplete") >= 1, timeout=2.0,
        )
    finally:
        agent.allow_finish.set()
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass
        await sess.close()

    # The cancelled turn's OutTextDelta MUST have made it through
    # (intermediate frames aren't suppressed — useful context).
    assert "OutTextDelta" in seen_kinds, seen_kinds
    # Exactly ONE OutTextFinal + ONE TurnComplete — the merged turn's.
    # The cancelled turn's terminal frames were suppressed.
    assert seen_kinds.count("OutTextFinal") == 1, (
        f"cancelled turn must not publish OutTextFinal; got {seen_kinds}"
    )
    assert seen_kinds.count("TurnComplete") == 1, (
        f"cancelled turn must not publish TurnComplete; got {seen_kinds}"
    )


@test("stream", "3 quick typed messages from quiet state coalesce into ONE turn")
async def t_quick_burst_from_quiet_coalesces(ctx: TestContext) -> None:
    """Regression for the production "responds only to the last
    message" bug: when the user fires three messages back-to-back from
    a quiet state, the agent must see ALL THREE as one merged user
    message — not the first dispatched alone (cancelled mid-stream),
    then the rest merged. The previous design dispatched the first
    message immediately and only buffered the follow-ups, which left
    the first message orphaned and let the LLM "address only the
    follow-ups". Always-debouncing typed text is what makes this work."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=False)
    sess = _make_session(agent, coalesce_window_ms=200)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        # Three messages within the window, no in-flight turn between them.
        for letter in ("hello", "and what's the time", "also weather"):
            await sess.push_in(TextFinal(
                session_id="s", seq=1, ts_ms=now_ms(),
                text=letter, source="user_typed",
            ))
            await asyncio.sleep(0.05)
        # Wait well past the window.
        await _wait_for(lambda: len(agent.calls) >= 1, timeout=2.0)
        await asyncio.sleep(0.3)
    finally:
        await sess.close()

    msgs = [c["message"] for c in agent.calls]
    assert msgs == ["hello\n\nand what's the time\n\nalso weather"], (
        f"3 quick typed messages must merge into ONE turn (the agent must "
        f"see all three as one user message); got {msgs}"
    )


@test("stream", "slow-spawn agent: barge-in during spawn salvages, no message lost")
async def t_slow_spawn_salvage(ctx: TestContext) -> None:
    """🔴 Production regression: a slow provider can take 5–10 s to
    cold-start its subprocess + MCP pool. The runner used to set
    ``_current_turn_started=True`` at the top of ``run()`` (before the
    agent actually had the prompt), so a barge-in arriving during the
    spawn window saw "started" and skipped the salvage path — the
    cancelled turn's user message was lost forever, the next merged
    burst dispatched without it, and the agent only addressed the
    later messages.

    This test simulates the spawn delay with a slow ``run_stream`` that
    awaits before yielding its first event. A second message during the
    spawn must trigger salvage so both messages reach the agent."""
    from src.stream.events import TextFinal, now_ms

    spawn_release = asyncio.Event()
    seen_messages: list[str] = []

    class _SlowSpawnAgent:
        name = "slow-spawn"
        db = None

        def __init__(self) -> None:
            self.calls = 0

        async def run_stream(self, *, message, user_id, session_id,
                             attachments=None, on_status=None, author=None):
            self.calls += 1
            if self.calls == 1:
                # Simulate a slow provider's subprocess spawn — agent has the
                # message in flight but hasn't yielded anything yet.
                # CancelledError from a barge-in propagates here, before
                # any event lands → salvage MUST trigger.
                await spawn_release.wait()
            seen_messages.append(message)
            yield {"kind": "done", "text": ""}

        def last_response_meta(self, sid):
            return {"model": "slow-spawn"}

    agent = _SlowSpawnAgent()
    sess = _make_session(agent, coalesce_window_ms=100)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        # Push msg 1 — drains after 100 ms, runner enters spawn wait.
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(),
            text="hello", source="user_typed",
        ))
        # Wait for the runner to actually be in the spawn wait.
        await _wait_for(lambda: agent.calls >= 1, timeout=2.0)
        # Push msg 2 while spawn is still blocking. This must cancel the
        # in-flight turn AND salvage "hello" back into the burst.
        await sess.push_in(TextFinal(
            session_id="s", seq=2, ts_ms=now_ms(),
            text="and what time is it", source="user_typed",
        ))
        # Release the spawn so any cancelled-turn path can finish.
        spawn_release.set()
        # Wait for the merged dispatch to land.
        await _wait_for(lambda: len(seen_messages) >= 2, timeout=3.0)
    finally:
        spawn_release.set()
        await sess.close()

    # The agent's spawn-blocked first call is preserved in agent.calls
    # but yields no events because it was cancelled. The salvage path
    # then re-buffers "hello" into the burst, and the merged dispatch
    # lands as ONE turn carrying both messages.
    merged_seen = [m for m in seen_messages if "hello" in m and "time" in m]
    assert merged_seen, (
        f"merged turn must contain BOTH 'hello' and 'time' — that's the "
        f"smoking-gun fix. Got seen_messages={seen_messages}"
    )


@test("stream", "Interrupt during cold spawn DROPS the input — no salvage, no merged dispatch")
async def t_interrupt_during_spawn_no_salvage(ctx: TestContext) -> None:
    """Counterpart to ``t_slow_spawn_salvage``: a typed-text barge-in
    SALVAGES the just-dispatched message, but an explicit ``Interrupt``
    DISCARDS it (no salvage, no merged turn). Pin this asymmetry — a
    refactor that flipped Interrupt to ``salvage_to_burst=True`` would
    silently re-feed user content the user was trying to discard."""
    from src.stream.events import Interrupt, TextFinal, now_ms

    spawn_release = asyncio.Event()
    seen_messages: list[str] = []

    class _SlowSpawnAgent:
        name = "slow-spawn"
        db = None

        def __init__(self) -> None:
            self.calls = 0

        async def run_stream(self, *, message, user_id, session_id,
                             attachments=None, on_status=None, author=None):
            self.calls += 1
            if self.calls == 1:
                await spawn_release.wait()
            seen_messages.append(message)
            yield {"kind": "done", "text": ""}

        def last_response_meta(self, sid):
            return {"model": "slow-spawn"}

    agent = _SlowSpawnAgent()
    sess = _make_session(agent, coalesce_window_ms=100)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(),
            text="hello", source="user_typed",
        ))
        await _wait_for(lambda: agent.calls >= 1, timeout=2.0)
        # Interrupt during spawn → drop, do NOT salvage.
        await sess.push_in(Interrupt(
            session_id="s", seq=2, ts_ms=now_ms(), reason="manual",
        ))
        spawn_release.set()
        # Give the dispatch loop time to handle the Interrupt + any
        # ghost dispatch attempts; nothing more should land.
        await asyncio.sleep(0.4)
    finally:
        spawn_release.set()
        await sess.close()

    # The cancelled first call ran (spawn-blocked) but produced no
    # actual yield. ``seen_messages`` only collects what the agent
    # AFTER cancellation processes — Interrupt must not let "hello"
    # come back via a follow-up burst.
    assert "hello" not in seen_messages, (
        f"Interrupt must DISCARD the in-flight typed input — saw "
        f"seen_messages={seen_messages}"
    )
    assert sess._pending_burst == [], (
        f"Interrupt must clear the burst buffer; got {sess._pending_burst}"
    )


@test("stream", "mirror modality: STT input speaks even when speak_enabled=False")
async def t_mirror_modality_stt_speaks_when_typed_silent(ctx: TestContext) -> None:
    """``speak_enabled=False`` silences typed-text replies (chat tab
    default), but voice (``source='stt'``) MUST still speak via the
    mirror-modality rule — without this the OpenAI-Realtime feel
    breaks for voice notes sent into chat-tab sessions."""
    from src.channels.tts_base import BaseTTS
    from src.stream.events import (
        OutAudioChunk, OutTextDelta, TextFinal, TurnComplete, now_ms,
    )

    class _NoiseTTS(BaseTTS):
        @property
        def audio_format(self):
            return "wav", "audio/wav"

        @property
        def voice_id(self):
            return "test-voice"

        async def synthesize_full(self, text, *, language=None):
            return b"\x00\x01" * 8

        async def synthesize_stream(self, text_chunks, *, language=None):
            async for _ in text_chunks:
                pass
            yield b"VOICE-CHUNK"

    async def _tts_factory(_db):
        return _NoiseTTS()

    async def _null(_db):
        return None

    # speak_enabled=False — typed replies stay silent.
    sess = _make_session(_FakeAgent(["he", "llo"]), coalesce_window_ms=0,
                         speak_enabled=False)
    await sess.start(stt_factory=_null, tts_factory=_tts_factory)
    try:
        # Typed message: NO audio.
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(),
            text="typed silently", source="user_typed",
        ))
        # Drain until TurnComplete.
        typed_audio = 0
        while True:
            evt = await asyncio.wait_for(sess.outbound.get(), timeout=2.0)
            if isinstance(evt, OutAudioChunk):
                typed_audio += 1
            if isinstance(evt, TurnComplete):
                break
        assert typed_audio == 0, (
            f"speak_enabled=False must silence typed replies; got {typed_audio} audio chunks"
        )

        # STT message on the SAME session: MUST speak (mirror modality).
        await sess.push_in(TextFinal(
            session_id="s", seq=2, ts_ms=now_ms(),
            text="spoken aloud", source="stt",
        ))
        stt_audio = 0
        while True:
            evt = await asyncio.wait_for(sess.outbound.get(), timeout=2.0)
            if isinstance(evt, OutAudioChunk):
                stt_audio += 1
            if isinstance(evt, TurnComplete):
                break
        assert stt_audio >= 1, (
            f"STT input MUST speak via mirror modality even with "
            f"speak_enabled=False; got {stt_audio} audio chunks"
        )
    finally:
        await sess.close()


@test("stream", "pre_dispatch_hook rejection publishes OutError + TurnComplete, no runner")
async def t_pre_dispatch_hook_rejects(ctx: TestContext) -> None:
    """Gateway uses the hook for budget gating + history-mode binding.
    Returning a non-None error string must publish a clean error frame
    and SKIP the runner entirely — without this, budget-blocked turns
    would still spawn the agent."""
    from src.stream.events import OutError, TextFinal, TurnComplete, now_ms

    agent = _RecordingAgent(block_first=False)
    sess = _make_session(agent, coalesce_window_ms=0)

    async def _reject(_msg):
        return "BUDGET_EXCEEDED"

    sess.pre_dispatch_hook = _reject

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(),
            text="should not reach agent", source="user_typed",
        ))
        # Drain.
        events = []
        while True:
            evt = await asyncio.wait_for(sess.outbound.get(), timeout=2.0)
            events.append(evt)
            if isinstance(evt, TurnComplete):
                break
    finally:
        await sess.close()

    assert agent.calls == [], (
        f"rejected turn must NOT reach the agent; got {agent.calls}"
    )
    errors = [e for e in events if isinstance(e, OutError)]
    assert len(errors) == 1 and errors[0].text == "BUDGET_EXCEEDED", events


@test("stream", "pre_dispatch_hook exception is swallowed, dispatch proceeds")
async def t_pre_dispatch_hook_exception_swallowed(ctx: TestContext) -> None:
    """A buggy hook must not break the session — log, swallow, and
    fall through to the normal dispatch path."""
    from src.stream.events import TextFinal, TurnComplete, now_ms

    agent = _RecordingAgent(block_first=False)
    sess = _make_session(agent, coalesce_window_ms=0)

    async def _crashy(_msg):
        raise RuntimeError("hook crashed")

    sess.pre_dispatch_hook = _crashy

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(),
            text="hi", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1, timeout=1.0)
    finally:
        await sess.close()

    # Hook crashed but the agent still saw the message.
    assert agent.calls and agent.calls[0]["message"] == "hi"


@test("stream", "10-message rapid spam coalesces — every message reaches the agent")
async def t_stress_no_message_lost(ctx: TestContext) -> None:
    """Hard regression for "spamming text messages stuck openagent and
    never responds" + "responding only to the very last message". Push
    10 messages back-to-back as fast as the event loop allows. Every
    single one must end up in some agent.run_stream call — no silent
    drops, no duplicates, no stuck dispatches. The collected calls
    concatenated in order must contain ``msg-0`` … ``msg-9`` exactly
    once each."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=False)
    sess = _make_session(agent, coalesce_window_ms=100)

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        for i in range(10):
            await sess.push_in(TextFinal(
                session_id="s", seq=i, ts_ms=now_ms(),
                text=f"msg-{i}", source="user_typed",
            ))
        # Wait long enough for the whole burst to flush (window + slack).
        await _wait_for(lambda: len(agent.calls) >= 1, timeout=2.0)
        await asyncio.sleep(0.5)
    finally:
        await sess.close()

    joined = "\n\n".join(c["message"] for c in agent.calls)
    for i in range(10):
        assert joined.count(f"msg-{i}") == 1, (
            f"msg-{i} should appear exactly once across all calls; "
            f"got calls={[c['message'] for c in agent.calls]}"
        )


@test("stream", "burst drain races a fresh arrival without dispatching twice")
async def t_drain_race_no_double_dispatch(ctx: TestContext) -> None:
    """Regression: the drain task cleared ``_pending_burst`` and
    ``_burst_timer`` before awaiting ``_dispatch_turn``. A new
    ``TextFinal`` arriving in that gap saw ``has_pending=False,
    in_flight=False`` and dispatched its own turn in parallel — racing
    both onto the same ``_current_turn`` slot. The dispatch lock must
    serialise the two paths so we get exactly two distinct turns
    (the merged one and the new one), not three."""
    from src.stream.events import TextFinal, now_ms

    agent = _RecordingAgent(block_first=True)
    # Tight window so the drain fires quickly. Keep block_first so the
    # first turn stays in flight until we release it after the assertion.
    sess = _make_session(agent, coalesce_window_ms=80)
    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    try:
        # Turn A — blocks the runner.
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="A", source="user_typed",
        ))
        await _wait_for(lambda: len(agent.calls) >= 1)
        # Two messages buffer + arm the timer (cancels turn A).
        await sess.push_in(TextFinal(
            session_id="s", seq=2, ts_ms=now_ms(), text="B", source="user_typed",
        ))
        await sess.push_in(TextFinal(
            session_id="s", seq=3, ts_ms=now_ms(), text="C", source="user_typed",
        ))
        # Wait until the timer is about to fire, then push another
        # message. With the lock, the new message either folds into the
        # merged turn (if it lands first) or waits for the merged turn
        # to dispatch and then schedules its own follow-up burst.
        await asyncio.sleep(0.08)
        await sess.push_in(TextFinal(
            session_id="s", seq=4, ts_ms=now_ms(), text="D", source="user_typed",
        ))
        # Settle long enough for any possible double-dispatch to manifest.
        await _wait_for(lambda: len(agent.calls) >= 2, timeout=2.0)
        await asyncio.sleep(0.3)
    finally:
        agent.release.set()
        await sess.close()

    msgs = [c["message"] for c in agent.calls]
    # First turn: "A". After that, ANY combination of messages B/C/D
    # split across one or more turns is acceptable, as long as we
    # never see a duplicate dispatch of the same merged content. The
    # critical regression check: every B/C/D character appears EXACTLY
    # once across the merged turns.
    after_first = "".join(c["message"] for c in agent.calls[1:])
    assert after_first.count("B") == 1, msgs
    assert after_first.count("C") == 1, msgs
    assert after_first.count("D") == 1, msgs
    assert msgs[0] == "A", msgs


# ── RealtimeChannel reconnect / rebind ──────────────────────────────


@test("stream", "RealtimeChannel.rebind delivers buffered frames to a fresh ws")
async def t_realtime_rebind_recovers_stuck_frames(ctx: TestContext) -> None:
    """Reproduces the 'stuck on Thinking…' bug: the original send target
    rejects every frame (closed transport, returns False); after
    ``rebind`` the same frame lands on the new send target. Without the
    fix the pump dropped the frame on the first False return and the
    UI's ``isProcessing`` flag never cleared."""
    from src.stream.channel import RealtimeChannel
    from src.stream.events import OutTextFinal, TurnComplete
    from src.stream.session import StreamSession

    agent = _FakeAgent(["hello"])
    sess = StreamSession(agent, client_id="c", session_id="s")

    async def dead_send(_payload: dict) -> bool:
        return False

    delivered: list[dict] = []

    async def live_send(payload: dict) -> bool:
        delivered.append(payload)
        return True

    # Tighten the retry interval so the test isn't slow.
    channel = RealtimeChannel(sess, dead_send)
    channel.RETRY_INTERVAL_S = 0.05  # type: ignore[misc]
    channel.UNRECOVERABLE_AFTER_S = 5.0  # type: ignore[misc]
    await channel.start()

    try:
        # Queue a turn — the runner publishes deltas + OutTextFinal +
        # TurnComplete onto outbound. The pump tries to send via
        # dead_send (False on every call) and parks on the retry sleep.
        await sess.run_one_shot("hi", speak=False)
        # Give the pump time to attempt at least one send + retry.
        await asyncio.sleep(0.2)
        assert delivered == [], "dead send must not deliver anything"

        channel.rebind(live_send)
        # Wait for the pump to drain the queued frames onto live_send.
        def _drained() -> bool:
            kinds = {p.get("type") for p in delivered}
            return "response" in kinds and "turn_complete" in kinds
        await _wait_for(_drained, timeout=3.0)

        # OutTextFinal serialises to ``response``; the frontend clears
        # ``isProcessing`` on it.
        types = [p["type"] for p in delivered]
        assert "response" in types, types
        assert types[-1] == "turn_complete", types
    finally:
        await channel.close()


@test("stream", "RealtimeChannel fires on_unrecoverable after the deadline")
async def t_realtime_unrecoverable_callback(ctx: TestContext) -> None:
    """When no rebind ever lands, the pump surrenders the frame after
    the deadline and fires ``on_unrecoverable`` so the gateway can reap
    the orphaned StreamSession instead of leaking the agent resources."""
    from src.stream.channel import RealtimeChannel
    from src.stream.session import StreamSession

    agent = _FakeAgent(["x"])
    sess = StreamSession(agent, client_id="c", session_id="s")

    async def dead_send(_payload: dict) -> bool:
        return False

    fired = asyncio.Event()

    async def on_unrecoverable() -> None:
        fired.set()

    channel = RealtimeChannel(
        sess, dead_send, on_unrecoverable=on_unrecoverable,
    )
    channel.RETRY_INTERVAL_S = 0.05  # type: ignore[misc]
    channel.UNRECOVERABLE_AFTER_S = 0.3  # type: ignore[misc]
    await channel.start()

    try:
        await sess.run_one_shot("hi", speak=False)
        await asyncio.wait_for(fired.wait(), timeout=2.0)
    finally:
        await channel.close()


@test("stream", "Gateway._broadcast_child_frame enqueues (non-blocking) instead of sending inline")
async def t_child_frame_nonblocking(ctx: TestContext) -> None:
    """A detached child run forwards deltas via ``_broadcast_child_frame`` from
    inside the agent loop — it MUST only ``put_nowait`` onto the drain queue, so
    a slow/stuck client can never backpressure (stall) the run. Here a Gateway
    with a stub ``broadcast`` that would explode confirms the frame lands on the
    queue and ``broadcast`` is never called inline."""
    import asyncio
    from src.gateway.server import Gateway

    gw = Gateway.__new__(Gateway)
    gw.agent = type("_Agent", (), {"memory_db": None})()
    gw._child_frame_q = asyncio.Queue(maxsize=16)
    gw._live_replays = {}
    called = []

    async def _boom(payload):  # would be awaited if the path sent inline
        called.append(payload)
        raise AssertionError("broadcast must not run inline from the agent loop")

    gw.broadcast = _boom  # type: ignore[assignment]

    await gw._broadcast_child_frame({"kind": "delta", "session_id": "scheduler:t:r", "text": "hi"})
    await gw._broadcast_child_frame({"kind": "turn_complete", "session_id": "scheduler:t:r"})

    assert called == [], "broadcast was called inline (would stall the run)"
    q = gw._child_frame_q
    assert q.qsize() == 2, q.qsize()
    assert q.get_nowait() == {"session_id": "scheduler:t:r", "type": "delta", "text": "hi"}
    # ``reason`` viaggia insieme al marcatore: un turno finito e un turno morto
    # non possono piu' arrivare come lo stesso frame vuoto. Assente = completato,
    # quindi un client vecchio legge questo esattamente come prima.
    assert q.get_nowait() == {
        "session_id": "scheduler:t:r", "type": "turn_complete", "reason": "completed",
    }


@test("stream", "Gateway._adopt_sessions_to_ws rebinds every channel for the client_id")
async def t_gateway_adopt_sessions_to_ws(ctx: TestContext) -> None:
    """End-to-end check of the reconnect adoption path: sessions
    created with ``ws_old`` should have their channel send-target
    swapped to ``ws_new`` after ``_adopt_sessions_to_ws`` fires."""
    from src.gateway.server import Gateway, _StreamHolder
    from src.stream.channel import RealtimeChannel
    from src.stream.session import StreamSession

    class _StubAgent:
        name = "stub"
        db = None

    from weakref import WeakKeyDictionary
    gw = Gateway.__new__(Gateway)
    gw.clients = {}
    gw._stream_sessions = {}
    gw._live_replays = {}
    # _safe_ws_send_json is now an instance method that serialises sends with a
    # per-socket lock — __new__ skips __init__, so seed the lock map.
    gw._ws_send_locks = WeakKeyDictionary()

    class _FakeWS:
        def __init__(self, name):
            self.name = name
            self.closed = False
            self.sent: list[dict] = []
        async def send_json(self, payload):
            self.sent.append(payload)

    ws_old = _FakeWS("old")
    ws_new = _FakeWS("new")

    sess_a = StreamSession(_StubAgent(), client_id="alice", session_id="sa")
    sess_b = StreamSession(_StubAgent(), client_id="alice", session_id="sb")
    sess_c = StreamSession(_StubAgent(), client_id="bob",   session_id="sc")
    ch_a = RealtimeChannel(
        sess_a,
        lambda p, _ws=ws_old: gw._safe_ws_send_json(_ws, p),
    )
    ch_b = RealtimeChannel(
        sess_b,
        lambda p, _ws=ws_old: gw._safe_ws_send_json(_ws, p),
    )
    ch_c = RealtimeChannel(
        sess_c,
        lambda p, _ws=ws_old: gw._safe_ws_send_json(_ws, p),
    )
    gw._stream_sessions[("alice", "sa")] = _StreamHolder(session=sess_a, channel=ch_a)
    gw._stream_sessions[("alice", "sb")] = _StreamHolder(session=sess_b, channel=ch_b)
    gw._stream_sessions[("bob",   "sc")] = _StreamHolder(session=sess_c, channel=ch_c)

    try:
        # Adopt only Alice's sessions onto ws_new.
        await gw._adopt_sessions_to_ws("alice", ws_new)

        # Alice's channels now hit ws_new; Bob's still hit ws_old.
        await ch_a._send({"type": "ping", "tag": "a"})
        await ch_b._send({"type": "ping", "tag": "b"})
        await ch_c._send({"type": "ping", "tag": "c"})

        assert [p["tag"] for p in ws_new.sent] == ["a", "b"], ws_new.sent
        assert [p["tag"] for p in ws_old.sent] == ["c"], ws_old.sent
    finally:
        await ch_a.close()
        await ch_b.close()
        await ch_c.close()


@test("stream", "Gateway live_state rehydrates an active turn for the same owner")
async def t_gateway_live_state_rehydrates_active_turn(ctx: TestContext) -> None:
    """Closing the app must not make an in-flight turn invisible. The gateway
    keeps a replay tail and sends it back to the same authenticated owner when
    a fresh websocket attaches."""
    from weakref import WeakKeyDictionary
    from src.gateway.server import Gateway

    gw = Gateway.__new__(Gateway)
    gw.agent = type("_Agent", (), {"memory_db": None})()
    gw._live_replays = {}
    gw._stream_sessions = {}
    gw._ws_send_locks = WeakKeyDictionary()

    class _ActiveSession:
        def has_active_turn(self):
            return True

    gw._stream_sessions[("alice", "s1")] = type("_Holder", (), {
        "session": _ActiveSession(),
    })()

    class _FakeWS:
        closed = False
        def __init__(self):
            self.sent: list[dict] = []
        async def send_json(self, payload):
            self.sent.append(payload)

    ws = _FakeWS()
    gw._record_live_input(
        "s1",
        {"type": "text_final", "session_id": "s1", "text": "hello"},
        owner="alice",
    )
    gw._record_live_output(
        "s1",
        {"type": "delta", "session_id": "s1", "text": "he"},
        owner="alice",
    )
    gw._record_live_output(
        "s1",
        {"type": "delta", "session_id": "s1", "text": "llo"},
        owner="alice",
    )

    await gw._send_live_snapshots(ws, client_id="device-1", handle="alice")

    assert len(ws.sent) == 1, ws.sent
    payload = ws.sent[0]
    assert payload["type"] == "live_state"
    assert payload["session_id"] == "s1"
    assert payload["active"] is True
    assert payload["frames"] == [
        {"type": "text_final", "session_id": "s1", "text": "hello"},
        {"type": "delta", "session_id": "s1", "text": "hello"},
    ]


@test("stream", "Gateway live_state is owner-scoped")
async def t_gateway_live_state_is_owner_scoped(ctx: TestContext) -> None:
    """A replay without a resolved owner must not be treated as a broadcast.
    Child runs resolve ownership best-effort; until they do, reconnects should
    skip the snapshot instead of leaking it to another authenticated handle."""
    from weakref import WeakKeyDictionary
    from src.gateway.server import Gateway

    class _DB:
        async def get_session(self, _session_id):
            return None

    gw = Gateway.__new__(Gateway)
    gw.agent = type("_Agent", (), {"memory_db": _DB()})()
    gw._live_replays = {}
    gw._ws_send_locks = WeakKeyDictionary()
    gw._record_live_output(
        "scheduler:t:r",
        {"type": "delta", "session_id": "scheduler:t:r", "text": "secret"},
        owner=None,
    )

    class _FakeWS:
        closed = False
        def __init__(self):
            self.sent: list[dict] = []
        async def send_json(self, payload):
            self.sent.append(payload)

    ws = _FakeWS()
    await gw._send_live_snapshots(ws, client_id="device-2", handle="bob")
    assert ws.sent == []


@test("stream", "Gateway drops stale completed stream-turn live_state")
async def t_gateway_live_state_drops_completed_stream_turn(ctx: TestContext) -> None:
    """A chat turn replay that started with ``text_final`` is only valid while
    its StreamSession still has an active turn. If the transport pump missed the
    terminal frames but the run completed and persisted, reconnect must hydrate
    from the DB rather than resurrect a permanent live/reasoning state."""
    from weakref import WeakKeyDictionary
    from src.gateway.sessions import SessionManager
    from src.gateway.server import Gateway, _StreamHolder

    gw = Gateway.__new__(Gateway)
    gw.agent = type("_Agent", (), {"memory_db": None})()
    gw._live_replays = {}
    gw._stream_sessions = {}
    gw._ws_send_locks = WeakKeyDictionary()
    gw.sessions = SessionManager(agent_name="test")

    class _DoneSession:
        def has_active_turn(self):
            return False

    class _Channel:
        def __init__(self):
            self.closed = False
        async def close(self):
            self.closed = True

    channel = _Channel()
    gw._stream_sessions[("alice", "s1")] = _StreamHolder(
        session=_DoneSession(),
        channel=channel,
    )

    class _FakeWS:
        closed = False
        def __init__(self):
            self.sent: list[dict] = []
        async def send_json(self, payload):
            self.sent.append(payload)

    ws = _FakeWS()
    gw._record_live_input(
        "s1",
        {"type": "text_final", "session_id": "s1", "text": "hello"},
        owner="alice",
    )
    gw._record_live_output(
        "s1",
        {"type": "delta", "session_id": "s1", "text": "hello"},
        owner="alice",
    )

    await gw._send_live_snapshots(ws, client_id="device-1", handle="alice")

    assert ws.sent == []
    assert "s1" not in gw._live_replays
    assert ("alice", "s1") not in gw._stream_sessions
    assert channel.closed is True


@test("stream", "Gateway prefers persisted terminal run over active stale holder")
async def t_gateway_live_state_drops_db_terminal_despite_active_holder(ctx: TestContext) -> None:
    """A websocket can die while ``RealtimeChannel`` is retrying an early
    frame. The agent turn may then finish and persist while the in-memory
    stream holder still looks active. Reconnect must trust the persisted run
    and avoid resurrecting the replay as an endless live/reasoning turn."""
    from weakref import WeakKeyDictionary
    from src.gateway.server import Gateway, _StreamHolder

    class _DB:
        def __init__(self):
            self.runs: list[dict] = []

        async def list_session_runs(self, _session_id, *, limit=20):
            return self.runs[:limit]

    db = _DB()
    gw = Gateway.__new__(Gateway)
    gw.agent = type("_Agent", (), {"memory_db": db})()
    gw._live_replays = {}
    gw._stream_sessions = {}
    gw._ws_send_locks = WeakKeyDictionary()

    class _StillLooksActiveSession:
        def has_active_turn(self):
            return True

    class _Channel:
        def __init__(self):
            self.closed = False
        async def close(self):
            self.closed = True

    channel = _Channel()
    gw._stream_sessions[("alice", "s1")] = _StreamHolder(
        session=_StillLooksActiveSession(),
        channel=channel,
    )

    class _FakeWS:
        closed = False
        def __init__(self):
            self.sent: list[dict] = []
        async def send_json(self, payload):
            self.sent.append(payload)

    ws = _FakeWS()
    gw._record_live_input(
        "s1",
        {"type": "text_final", "session_id": "s1", "text": "hello"},
        owner="alice",
    )
    gw._record_live_output(
        "s1",
        {"type": "reasoning", "session_id": "s1", "active": True},
        owner="alice",
    )
    db.runs = [{
        "status": "COMPLETED",
        "created_at": gw._live_replays["s1"].started_at + 1,
        # Real runtime rows may carry a coalesced/transformed input that does
        # not exactly equal the first replay text_final. The timestamped
        # terminal run is still authoritative for this session turn.
        "input": {"input_content": "hello\n\nsecond burst message"},
        "content": "done",
    }]

    await gw._send_live_snapshots(ws, client_id="device-1", handle="alice")

    assert ws.sent == []
    assert "s1" not in gw._live_replays
    assert ("alice", "s1") not in gw._stream_sessions
    assert channel.closed is True


@test("stream", "Gateway keeps live replay when newer terminal run is unrelated")
async def t_gateway_live_state_ignores_unrelated_terminal_run(ctx: TestContext) -> None:
    """A delayed terminal write from a previous/cancelled turn must not close
    a fresh live replay just because its timestamp lands after the new input."""
    from weakref import WeakKeyDictionary
    from src.gateway.server import Gateway, _StreamHolder

    class _DB:
        def __init__(self):
            self.runs: list[dict] = []

        async def list_session_runs(self, _session_id, *, limit=20):
            return self.runs[:limit]

    db = _DB()
    gw = Gateway.__new__(Gateway)
    gw.agent = type("_Agent", (), {"memory_db": db})()
    gw._live_replays = {}
    gw._stream_sessions = {}
    gw._ws_send_locks = WeakKeyDictionary()

    class _StillActiveSession:
        def has_active_turn(self):
            return True

    class _Channel:
        def __init__(self):
            self.closed = False
        async def close(self):
            self.closed = True

    channel = _Channel()
    gw._stream_sessions[("alice", "s1")] = _StreamHolder(
        session=_StillActiveSession(),
        channel=channel,
    )

    class _FakeWS:
        closed = False
        def __init__(self):
            self.sent: list[dict] = []
        async def send_json(self, payload):
            self.sent.append(payload)

    ws = _FakeWS()
    gw._record_live_input(
        "s1",
        {"type": "text_final", "session_id": "s1", "text": "new live question"},
        owner="alice",
    )
    gw._record_live_output(
        "s1",
        {"type": "reasoning", "session_id": "s1", "active": True},
        owner="alice",
    )
    db.runs = [{
        "status": "COMPLETED",
        "created_at": gw._live_replays["s1"].started_at + 1,
        "input": {"input_content": "old cancelled question"},
        "content": "old answer",
    }]

    await gw._send_live_snapshots(ws, client_id="device-1", handle="alice")

    assert len(ws.sent) == 1, ws.sent
    assert ws.sent[0]["type"] == "live_state"
    assert "s1" in gw._live_replays
    assert ("alice", "s1") in gw._stream_sessions
    assert channel.closed is False


@test("stream", "Gateway terminal follow-up frames do not resurrect live replay")
async def t_gateway_context_report_after_turn_complete_stays_settled(ctx: TestContext) -> None:
    """The runner publishes context_report after turn_complete. That passive
    frame must not create a new active replay, or the next new-chat
    session_open will rehydrate the previous completed chat as live."""
    from weakref import WeakKeyDictionary
    from src.gateway.sessions import SessionManager
    from src.gateway.server import Gateway

    gw = Gateway.__new__(Gateway)
    gw.agent = type("_Agent", (), {"memory_db": None})()
    gw._live_replays = {}
    gw._stream_sessions = {}
    gw._ws_send_locks = WeakKeyDictionary()
    gw.sessions = SessionManager(agent_name="test")

    gw._record_live_input(
        "old",
        {"type": "text_final", "session_id": "old", "text": "old prompt"},
        owner="alice",
    )
    gw._record_live_output(
        "old",
        {"type": "reasoning", "session_id": "old", "active": True},
        owner="alice",
    )
    gw._record_live_output(
        "old",
        {"type": "response", "session_id": "old", "text": "old answer"},
        owner="alice",
    )
    gw._record_live_output(
        "old",
        {"type": "turn_complete", "session_id": "old"},
        owner="alice",
    )
    assert "old" not in gw._live_replays

    gw._record_live_output(
        "old",
        {
            "type": "context_report",
            "session_id": "old",
            "report": {"used_tokens": 10, "max_tokens": 100},
        },
        owner="alice",
    )
    assert "old" not in gw._live_replays

    class _FakeWS:
        closed = False
        def __init__(self):
            self.sent: list[dict] = []
        async def send_json(self, payload):
            self.sent.append(payload)

    ws = _FakeWS()
    await gw._handle_stream_frame(
        ws,
        "device-1",
        {
            "type": "session_open",
            "session_id": "new",
            "profile": "batched",
            "client_kind": "webapp-chat",
            "speak": False,
        },
        handle="alice",
    )

    assert all(
        frame.get("session_id") != "old" for frame in ws.sent
    ), ws.sent
    assert "old" not in gw._live_replays
    await gw._close_stream_session(("alice", "new"))


@test("stream", "Gateway reattach retires stale holder before stale frame flush")
async def t_gateway_adopt_retires_db_terminal_before_rebind_flush(ctx: TestContext) -> None:
    """Regression for close/reopen mid-generation: the channel pump can be
    retrying an old ``reasoning active=true`` frame on the dead websocket while
    the run has already completed in the DB. Reattach must retire the holder
    before ``rebind`` gives that stale frame a fresh transport."""
    from weakref import WeakKeyDictionary
    from src.gateway.sessions import SessionManager
    from src.gateway.server import Gateway, _StreamHolder
    from src.stream.channel import RealtimeChannel
    from src.stream.events import OutReasoning, now_ms
    from src.stream.session import StreamSession

    class _DB:
        def __init__(self):
            self.runs: list[dict] = []

        async def list_session_runs(self, _session_id, *, limit=20):
            return self.runs[:limit]

    db = _DB()
    gw = Gateway.__new__(Gateway)
    gw.agent = type("_Agent", (), {"memory_db": db})()
    gw._live_replays = {}
    gw._stream_sessions = {}
    gw._ws_send_locks = WeakKeyDictionary()
    gw.sessions = SessionManager(agent_name="test")

    sess = StreamSession(_FakeAgent([]), client_id="device-1", session_id="s1")
    # Keep has_active_turn() true even though the persisted run below says the
    # turn has already completed, mirroring the stale holder race.
    sleeper = asyncio.create_task(asyncio.sleep(60))
    sess._current_turn = sleeper

    stuck_payloads: list[dict] = []

    async def dead_send(payload: dict) -> bool:
        stuck_payloads.append(payload)
        return False

    channel = RealtimeChannel(sess, dead_send)
    gw._stream_sessions[("alice", "s1")] = _StreamHolder(
        session=sess,
        channel=channel,
    )
    gw._record_live_input(
        "s1",
        {"type": "text_final", "session_id": "s1", "text": "hello"},
        owner="alice",
    )
    sess.outbound.put_nowait(OutReasoning(
        session_id="s1",
        seq=1,
        ts_ms=now_ms(),
        active=True,
    ))
    await channel.start()
    await _wait_for(lambda: len(stuck_payloads) >= 1, timeout=2.0)

    db.runs = [{
        "status": "COMPLETED",
        "created_at": gw._live_replays["s1"].started_at + 1,
        "input": {"input_content": "hello"},
        "content": "done",
    }]

    class _FreshWS:
        closed = False
        def __init__(self):
            self.sent: list[dict] = []
        async def send_json(self, payload):
            self.sent.append(payload)

    fresh_ws = _FreshWS()
    try:
        await gw._adopt_sessions_to_ws("device-1", fresh_ws, handle="alice")
        await asyncio.sleep(0.05)
        live = await gw.active_live_session_ids(client_id="device-1", handle="alice")
        await gw._handle_stream_frame(
            fresh_ws,
            "device-1",
            {
                "type": "session_open",
                "session_id": "s2",
                "profile": "batched",
                "client_kind": "webapp-chat",
                "speak": False,
            },
            handle="alice",
        )
        await asyncio.sleep(0.05)
    finally:
        sleeper.cancel()
        try:
            await sleeper
        except asyncio.CancelledError:
            pass

    assert fresh_ws.sent == [], fresh_ws.sent
    assert live == set()
    assert ("alice", "s1") not in gw._stream_sessions
    assert "s1" not in gw._live_replays
    assert ("alice", "s2") in gw._stream_sessions
    assert channel._pump_task is None or channel._pump_task.done()
    await gw._close_stream_session(("alice", "s2"))


@test("stream", "RealtimeChannel.rebind preserves frame ordering across the swap")
async def t_realtime_rebind_preserves_order(ctx: TestContext) -> None:
    """A rebind during an in-flight pump iteration must not lose or
    reorder frames. The frame that was stuck on dead_send completes
    first on live_send, then subsequent frames follow in order."""
    from src.stream.channel import RealtimeChannel
    from src.stream.session import StreamSession

    agent = _FakeAgent(["a", "b", "c"])
    sess = StreamSession(agent, client_id="c", session_id="s")

    async def dead_send(_payload: dict) -> bool:
        return False

    delivered: list[dict] = []

    async def live_send(payload: dict) -> bool:
        delivered.append(payload)
        return True

    channel = RealtimeChannel(sess, dead_send)
    channel.RETRY_INTERVAL_S = 0.05  # type: ignore[misc]
    channel.UNRECOVERABLE_AFTER_S = 5.0  # type: ignore[misc]
    await channel.start()

    try:
        await sess.run_one_shot("hi", speak=False)
        await asyncio.sleep(0.2)
        channel.rebind(live_send)
        def _drained() -> bool:
            return any(p.get("type") == "turn_complete" for p in delivered)
        await _wait_for(_drained, timeout=3.0)

        types = [p["type"] for p in delivered]
        # Deltas concat to "abc"; final OutTextFinal carries the same;
        # TurnComplete is last.
        assert types[-1] == "turn_complete", types
        deltas_text = "".join(
            p.get("text", "") for p in delivered if p.get("type") == "delta"
        )
        assert deltas_text == "abc", deltas_text
    finally:
        await channel.close()


# ── Barge-in must not stall on the TTS speaker drain ──────────────────


@test("stream", "barge-in cancels promptly even with a slow TTS speaker (no 20s drain stall)")
async def t_barge_in_no_speaker_drain_stall(ctx: TestContext) -> None:
    """A voice/stop barge-in must cancel the live turn promptly even while
    TTS is mid-stream.

    The runner's ``speaker_task`` is a sibling that is NOT cancelled by the
    turn task's own cancellation. The old finally awaited
    ``wait_for(speaker, SPEAKER_DRAIN_TIMEOUT=20s)`` on EVERY exit, so a
    barge-in mid-TTS blocked ``_cancel_active_turn`` — which runs inside the
    single dispatch loop under ``_dispatch_lock`` — for up to 20s, going
    deaf to the next utterance/interrupt. The fix cancels the speaker
    immediately on a cancelled turn. Pin it: the cancel must complete in
    well under the drain timeout.
    """
    from src.channels.tts_base import BaseTTS
    from src.stream.events import Interrupt, TextFinal, now_ms

    speaker_cancelled = asyncio.Event()

    class _BlockingTTS(BaseTTS):
        @property
        def audio_format(self):
            return "mp3", "audio/mpeg"

        @property
        def voice_id(self):
            return "blocking"

        async def synthesize_full(self, text, *, language=None):
            return b""

        async def synthesize_stream(self, text_chunks, *, language=None):
            # Drain the text the runner pipes in, then simulate a long
            # TTS tail. Before the fix this 30s sleep was awaited by the
            # cancel path (capped at the 20s drain timeout).
            try:
                async for _ in text_chunks:
                    pass
                await asyncio.sleep(30)
                yield b"audio"  # pragma: no cover — unreachable
            except asyncio.CancelledError:
                speaker_cancelled.set()
                raise

    async def _tts_factory(_db):
        return _BlockingTTS()

    async def _null(_db):
        return None

    agent = _RecordingAgent(block_first=True)
    sess = _make_session(agent, coalesce_window_ms=0, speak_enabled=True)
    await sess.start(stt_factory=_null, tts_factory=_tts_factory)
    try:
        # Fire a turn; it engages (yields a delta) and blocks, while the
        # speaker_task is spun up and parked on its 30s tail.
        await sess.push_in(TextFinal(
            session_id="s", seq=1, ts_ms=now_ms(), text="hi", source="user_typed",
        ))
        engaged = await _wait_for(lambda: sess._current_turn_started, timeout=3.0)
        assert engaged, "turn never engaged"

        # Barge-in. Time how long until the turn slot frees.
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await sess.push_in(Interrupt(
            session_id="s", seq=2, ts_ms=now_ms(), reason="user_speech",
        ))
        freed = await _wait_for(lambda: not sess.has_active_turn(), timeout=5.0)
        elapsed = loop.time() - t0
        assert freed, "turn still active 5s after barge-in — drain stall regressed"
        assert elapsed < 3.0, (
            f"barge-in took {elapsed:.2f}s — the speaker-drain stall (up to 20s) "
            f"appears to have regressed"
        )
        # And the speaker itself was cancelled, not left to drain.
        was_cancelled = await _wait_for(lambda: speaker_cancelled.is_set(), timeout=2.0)
        assert was_cancelled, "speaker task was not cancelled on barge-in"
    finally:
        await sess.close()
