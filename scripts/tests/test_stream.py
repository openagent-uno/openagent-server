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
    from openagent.stream.events import (
        AudioChunk, Interrupt, OutAudioChunk, OutAudioEnd, OutAudioStart,
        OutTextDelta, OutTextFinal, OutToolStatus, OutVideoFrame, SessionOpen,
        TextDelta, TextFinal, TurnComplete, VideoFrame,
    )
    from openagent.stream.events import OutError
    from openagent.stream.wire import event_to_wire, wire_to_event

    cases = [
        OutTextDelta(session_id="s", seq=1, ts_ms=10, text="hi"),
        OutTextFinal(session_id="s", seq=2, ts_ms=20, text="done", model="m"),
        OutAudioStart(session_id="s", seq=3, ts_ms=30, format="mp3", mime="audio/mpeg"),
        OutAudioChunk(session_id="s", seq=4, ts_ms=40, data=b"\x00\x01"),
        OutAudioEnd(session_id="s", seq=5, ts_ms=50, total_chunks=1),
        OutToolStatus(session_id="s", seq=6, ts_ms=60, text="Using bash"),
        OutVideoFrame(session_id="s", seq=7, ts_ms=70, stream="webcam",
                      image_bytes=b"jpgbytes", width=320, height=240),
        TurnComplete(session_id="s", seq=8, ts_ms=80),
        TextDelta(session_id="s", seq=9, ts_ms=90, text="hel", final=False),
        TextFinal(session_id="s", seq=10, ts_ms=100, text="hello", source="stt"),
        AudioChunk(session_id="s", seq=11, ts_ms=110, data=b"raw",
                   end_of_speech=True, sample_rate=16000, encoding="pcm16"),
        VideoFrame(session_id="s", seq=12, ts_ms=120, stream="screen",
                   image_bytes=b"frame", width=1024, height=768, keyframe=True),
        Interrupt(session_id="s", seq=13, ts_ms=130, reason="user_speech"),
        SessionOpen(session_id="s", seq=14, ts_ms=140, profile="realtime",
                    language="en", client_kind="webapp"),
        OutError(session_id="s", seq=15, ts_ms=150, text="boom"),
    ]
    for evt in cases:
        wire = event_to_wire(evt)
        back = wire_to_event(wire)
        assert back == evt, f"round-trip mismatch: {evt!r} → {wire!r} → {back!r}"


@test("stream", "legacy MESSAGE frame decodes to TextFinal")
async def t_legacy_message_decodes(ctx: TestContext) -> None:
    from openagent.stream.events import TextFinal
    from openagent.stream.wire import wire_to_event

    evt = wire_to_event({"type": "message", "session_id": "s1", "text": "hey"})
    assert isinstance(evt, TextFinal)
    assert evt.text == "hey"
    assert evt.source == "user_typed"


@test("stream", "unknown wire types decode to None")
async def t_unknown_wire(ctx: TestContext) -> None:
    from openagent.stream.wire import wire_to_event

    assert wire_to_event({"type": "auth"}) is None
    assert wire_to_event({"type": "lol_what"}) is None
    assert wire_to_event({}) is None


@test("stream", "session_open emits explicit 0 on the wire (encoder side of the 3-state contract)")
async def t_session_open_emits_explicit_zero(ctx: TestContext) -> None:
    """The decoder distinguishes None / 0 / positive (see
    ``t_session_open_coalesce_default``); pin the encoder side too. A
    future "optimization" that drops 0 from the JSON would silently
    flip explicit opt-out sessions back to the default coalesce."""
    from openagent.stream.events import SessionOpen
    from openagent.stream.wire import event_to_wire

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
    from openagent.stream.events import SessionOpen
    from openagent.stream.wire import wire_to_event

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
    from openagent.stream.session import StreamSession

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
    from openagent.channels.tts_base import LocalPiperTTS, resolve_tts
    from openagent.channels import tts_local

    if not tts_local.is_available():
        from ._framework import TestSkip
        raise TestSkip("piper not installed")

    tts = await resolve_tts(db=None)
    assert isinstance(tts, LocalPiperTTS), f"got {type(tts).__name__}"


@test("stream", "resolve_tts returns ElevenLabsWSTTS when row opts in")
async def t_resolve_tts_elevenlabs_ws(ctx: TestContext) -> None:
    from openagent.channels.tts_base import ElevenLabsWSTTS, resolve_tts

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
                         attachments=None, on_status=None):
        for d in self._deltas:
            yield {"kind": "delta", "text": d}
        yield {"kind": "done", "text": "".join(self._deltas)}

    def last_response_meta(self, session_id: str) -> dict:
        return {"model": "fake-model"}


@test("stream", "StreamSession.run_one_shot pumps deltas and TurnComplete")
async def t_run_one_shot(ctx: TestContext) -> None:
    from openagent.stream.events import OutTextDelta, OutTextFinal, TurnComplete
    from openagent.stream.session import StreamSession

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


@test("stream", "BatchedChannel collapses one turn into a finished reply")
async def t_batched_channel(ctx: TestContext) -> None:
    from openagent.stream.channel import BatchedChannel
    from openagent.stream.session import StreamSession

    agent = _FakeAgent(["foo ", "bar"])
    sess = StreamSession(agent, client_id="c", session_id="s")
    channel = BatchedChannel(sess)

    async def driver():
        return await channel.run_one_shot("ping")

    async def runner():
        # Drive the runner directly — the channel pushes a TextFinal,
        # then we run_one_shot to make the assistant reply land in
        # outbound; BatchedChannel drains it.
        await sess.run_one_shot("ping", speak=False)

    # Push the user message via the channel, run_one_shot in parallel.
    drain = asyncio.create_task(driver())
    await runner()
    reply = await asyncio.wait_for(drain, timeout=5.0)

    assert reply.text == "foo bar", reply
    assert reply.audio_bytes is None
    assert reply.model == "fake-model"


@test("stream", "wire codec drops binary payloads losslessly via base64")
async def t_wire_binary(ctx: TestContext) -> None:
    from openagent.stream.events import OutAudioChunk
    from openagent.stream.wire import event_to_wire, wire_to_event

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
    from openagent.channels.tts_base import BaseTTS
    from openagent.stream.events import OutAudioChunk
    from openagent.stream.session import StreamSession

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

    from openagent.channels.stt_base import BaseSTT, STTEvent

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
    from openagent.channels.stt_base import BaseSTT

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

    from openagent.channels.stt_base import BaseSTT, STTEvent
    from openagent.stream.events import AudioChunk, now_ms
    from openagent.stream.session import StreamSession

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
                             attachments=None, on_status=None):
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
    what real providers (claude-cli, agno) signal once the prompt has
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
                         attachments=None, on_status=None):
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
    from openagent.stream.session import StreamSession
    return StreamSession(agent, client_id="c", session_id="s", **kwargs)


@test("stream", "coalesce explicitly off preserves preempt-on-each-message")
async def t_coalesce_explicitly_off(ctx: TestContext) -> None:
    """Passing ``coalesce_window_ms=0`` explicitly must keep the legacy
    behaviour: each new TextFinal preempts the previous and dispatches
    as its own turn — no buffering, no merging. (The class default is
    now 500 ms; this test guards the explicit-disable escape hatch.)"""
    from openagent.stream.events import TextFinal, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
    from openagent.stream.events import Interrupt, TextFinal, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
    pin the wiring: every ``OutToolStatus`` whose JSON ``tool`` field
    matches one of the known MCP prefixes adds to the per-turn set,
    which fires once on TurnComplete."""
    import json as _json

    from openagent.stream.events import OutToolStatus, TurnComplete, now_ms
    from openagent.stream.session import StreamSession

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
        text=_json.dumps({"tool": "scheduler_add_task", "status": "running"}),
    ))
    await sess._publish(OutToolStatus(
        session_id="s", seq=2, ts_ms=now_ms(),
        text=_json.dumps({"tool": "Bash", "status": "running"}),
    ))
    await sess._publish(OutToolStatus(
        session_id="s", seq=3, ts_ms=now_ms(),
        text=_json.dumps({"tool": "mcp__workflow_manager__run", "status": "done"}),
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
    from openagent.stream.collector import StreamCollector, fold_outbound_event
    from openagent.stream.events import OutError, OutTextFinal, now_ms

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
    from openagent.stream.events import (
        OutTextDelta, OutTextFinal, TextFinal, TurnComplete, now_ms,
    )

    class _ChattyAgent:
        name = "chatty"
        db = None

        def __init__(self) -> None:
            self.calls = 0
            self.allow_finish = asyncio.Event()

        async def run_stream(self, *, message, user_id, session_id,
                             attachments=None, on_status=None):
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
    from openagent.stream.events import TextFinal, now_ms

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
    """🔴 Production regression: claude-cli takes 5–10 s to cold-start
    its subprocess + MCP pool. The runner used to set
    ``_current_turn_started=True`` at the top of ``run()`` (before the
    agent actually had the prompt), so a barge-in arriving during the
    spawn window saw "started" and skipped the salvage path — the
    cancelled turn's user message was lost forever, the next merged
    burst dispatched without it, and the agent only addressed the
    later messages.

    This test simulates the spawn delay with a slow ``run_stream`` that
    awaits before yielding its first event. A second message during the
    spawn must trigger salvage so both messages reach the agent."""
    from openagent.stream.events import TextFinal, now_ms

    spawn_release = asyncio.Event()
    seen_messages: list[str] = []

    class _SlowSpawnAgent:
        name = "slow-spawn"
        db = None

        def __init__(self) -> None:
            self.calls = 0

        async def run_stream(self, *, message, user_id, session_id,
                             attachments=None, on_status=None):
            self.calls += 1
            if self.calls == 1:
                # Simulate claude-cli subprocess spawn — agent has the
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
    from openagent.stream.events import Interrupt, TextFinal, now_ms

    spawn_release = asyncio.Event()
    seen_messages: list[str] = []

    class _SlowSpawnAgent:
        name = "slow-spawn"
        db = None

        def __init__(self) -> None:
            self.calls = 0

        async def run_stream(self, *, message, user_id, session_id,
                             attachments=None, on_status=None):
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
    from openagent.channels.tts_base import BaseTTS
    from openagent.stream.events import (
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
    from openagent.stream.events import OutError, TextFinal, TurnComplete, now_ms

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
    from openagent.stream.events import TextFinal, TurnComplete, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
    from openagent.stream.events import TextFinal, now_ms

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
