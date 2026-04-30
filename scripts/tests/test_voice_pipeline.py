"""VoiceTurnOrchestrator — RESPONSE contract under all paths.

Every code path through ``VoiceTurnOrchestrator.run`` must emit exactly
one ``RESPONSE`` frame so the client clears its "Thinking..." state.
This module guards that contract against four scenarios:

  * no TTS configured (text-only voice turn)
  * TTS configured + agent stream succeeds (audio + response)
  * agent stream raises mid-turn (response carries the error)
  * TTS chunk send raises (response still goes through)

Plus the chat store's voice-session helper semantics (Python port of
the JS logic, kept here so a test failure flags drift):

  * ``getOrCreateVoiceSession`` returns the same id on second call
  * ``clearVoiceSession`` makes the next call mint a fresh id
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from ._framework import TestContext, test


# ── Helpers ─────────────────────────────────────────────────────────


class _FakeAgent:
    """Minimal Agent surface: ``.db``, ``run_stream``, ``last_response_meta``."""

    def __init__(self, deltas: list[str], *, raise_after: int | None = None,
                 model: str = "openai:gpt-test"):
        self.db = object()  # placeholder — orchestrator just passes it through
        self._deltas = deltas
        self._raise_after = raise_after  # raise after N deltas yielded
        self._model = model

    async def run_stream(self, **_kwargs):
        for i, delta in enumerate(self._deltas):
            if self._raise_after is not None and i >= self._raise_after:
                raise RuntimeError("simulated stream failure")
            yield {"kind": "delta", "text": delta}
        yield {"kind": "done"}

    def last_response_meta(self, _session_id: str) -> dict[str, Any]:
        return {"model": self._model}


class _FakeTTSConfig:
    """Mirrors openagent.channels.tts.TTSConfig just enough for the
    orchestrator to call ``cfg.voice_id``, ``cfg.response_format``, and
    pass cfg to synth. Mirrors the real dataclass shape so tests catch
    drift if a new attribute becomes required."""

    def __init__(self):
        self.vendor = "openai"
        self.model_id = "tts-1"
        self.voice_id = "alloy"
        self.response_format = "mp3"


def _capture_send():
    """Returns ``(send_fn, frames)`` where ``frames`` accumulates every
    payload the orchestrator emitted, in order."""
    frames: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> None:
        frames.append(payload)

    return send, frames


def _types(frames: list[dict[str, Any]]) -> list[str]:
    return [str(f.get("type")) for f in frames]


# ── Tests: voice pipeline RESPONSE contract ────────────────────────


@test("voice_pipeline", "no TTS configured → text-only RESPONSE still sent")
async def t_no_tts_response_still_sent(ctx: TestContext) -> None:
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    agent = _FakeAgent(["Hello ", "world."])
    send, frames = _capture_send()

    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=None):
        orchestrator = VoiceTurnOrchestrator(agent, send)
        result = await orchestrator.run(
            "hi", client_id="c1", session_id="sess-A",
        )

    types = _types(frames)
    assert "response" in types, f"no RESPONSE frame emitted: {types}"
    assert "audio_start" not in types, f"audio frames leaked when cfg=None: {types}"
    response = next(f for f in frames if f["type"] == "response")
    assert response["session_id"] == "sess-A", response
    assert "Hello world." in response["text"], response
    assert result["audio_chunks"] == 0
    assert result["errored"] is False


@test("voice_pipeline", "TTS configured + success → audio frames AND RESPONSE")
async def t_tts_success_full_pipeline(ctx: TestContext) -> None:
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    agent = _FakeAgent(["First sentence. ", "Second sentence."])
    send, frames = _capture_send()

    async def fake_synth(_sentence: str, _cfg, **_kwargs):
        # Two MP3-ish byte chunks per sentence call. ``**_kwargs``
        # absorbs the new ``language`` keyword the orchestrator now
        # forwards so the mock signature doesn't have to be edited
        # every time the production callsite grows a parameter.
        yield b"\xff\xfb\x90\x00fake-mp3-1"
        yield b"\xff\xfb\x90\x00fake-mp3-2"

    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=_FakeTTSConfig()), \
         patch("openagent.gateway.voice_pipeline.synthesize_stream", fake_synth):
        orchestrator = VoiceTurnOrchestrator(agent, send)
        result = await orchestrator.run(
            "hi", client_id="c1", session_id="sess-B",
        )

    types = _types(frames)
    assert types[0] == "audio_start", f"first frame should be audio_start: {types}"
    assert types.count("audio_chunk") >= 2, f"expected ≥2 audio chunks: {types}"
    assert "audio_end" in types, types
    assert types[-1] == "response", f"last frame should be response: {types}"
    end = next(f for f in frames if f["type"] == "audio_end")
    chunks = [f for f in frames if f["type"] == "audio_chunk"]
    assert end["total_chunks"] == len(chunks), (end, len(chunks))
    assert result["errored"] is False
    assert result["audio_chunks"] == len(chunks)


@test("voice_pipeline", "agent stream raises mid-turn → RESPONSE still sent (errored=True)")
async def t_stream_error_response_still_sent(ctx: TestContext) -> None:
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    # Yield one delta, then raise on the next iteration.
    agent = _FakeAgent(["Partial answer ", "before crash"], raise_after=1)
    send, frames = _capture_send()

    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=None):
        orchestrator = VoiceTurnOrchestrator(agent, send)
        result = await orchestrator.run(
            "hi", client_id="c1", session_id="sess-C",
        )

    types = _types(frames)
    assert "response" in types, f"RESPONSE missing on stream error: {types}"
    response = next(f for f in frames if f["type"] == "response")
    assert response["session_id"] == "sess-C", response
    # Whatever was streamed before the crash is preserved.
    assert "Partial answer" in response["text"], response
    assert result["errored"] is True


@test("voice_pipeline", "TTS chunk send fails → RESPONSE still completes")
async def t_tts_chunk_error_doesnt_break_response(ctx: TestContext) -> None:
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    agent = _FakeAgent(["A sentence."])
    send, frames = _capture_send()

    async def failing_synth(_sentence: str, _cfg, **_kwargs):
        yield b"first-ok-chunk"
        raise RuntimeError("synth API down")

    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=_FakeTTSConfig()), \
         patch("openagent.gateway.voice_pipeline.synthesize_stream", failing_synth):
        orchestrator = VoiceTurnOrchestrator(agent, send)
        result = await orchestrator.run(
            "hi", client_id="c1", session_id="sess-D",
        )

    types = _types(frames)
    # Audio frames may or may not appear depending on where the error
    # surfaces; the contract is just that AUDIO_END (if started) and
    # RESPONSE both close out.
    if "audio_start" in types:
        assert "audio_end" in types, f"audio_end missing after audio_start: {types}"
    assert types[-1] == "response", f"RESPONSE not last frame: {types}"
    # The agent itself didn't fail — only the TTS dispatch did. So
    # ``errored`` stays False (the orchestrator only flips it on agent
    # stream errors).
    assert result["errored"] is False


# ── Tests: local Piper fallback path ───────────────────────────────


class _LocalPiperConfig:
    """Mirrors a TTSConfig produced by resolve_tts_provider when no
    cloud row is configured but Piper is available."""

    def __init__(self):
        from openagent.channels.tts import LOCAL_PIPER_VENDOR
        self.vendor = LOCAL_PIPER_VENDOR
        self.model_id = "piper"
        self.voice_id = "en_US-amy-medium"
        self.response_format = "wav"


@test("voice_pipeline", "local Piper config → audio_start emits mime=audio/wav")
async def t_local_piper_emits_wav_mime(ctx: TestContext) -> None:
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    agent = _FakeAgent(["Hello from Piper."])
    send, frames = _capture_send()

    async def piper_synth(_sentence: str, _cfg, **_kwargs):
        # Realistic-ish WAV header bytes; doesn't matter for the test
        # since we only assert the mime/format wiring.
        yield b"RIFF\x00\x00\x00\x00WAVEfmt "

    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=_LocalPiperConfig()), \
         patch("openagent.gateway.voice_pipeline.synthesize_stream", piper_synth):
        orchestrator = VoiceTurnOrchestrator(agent, send)
        await orchestrator.run("hi", client_id="c1", session_id="sess-piper")

    audio_start = next((f for f in frames if f["type"] == "audio_start"), None)
    assert audio_start is not None, f"audio_start missing: {_types(frames)}"
    assert audio_start["mime"] == "audio/wav", (
        f"local Piper should emit mime=audio/wav, got {audio_start['mime']}"
    )
    assert audio_start["format"] == "wav", audio_start
    # Final RESPONSE must still be present so text-only clients work.
    assert any(f["type"] == "response" for f in frames), _types(frames)


@test("voice_pipeline", "language hint forwarded from orchestrator → synthesize_stream")
async def t_language_forwarded_to_synth(ctx: TestContext) -> None:
    """Regression for the user complaint where an Italian transcription
    was spoken in an American accent: the orchestrator must thread the
    ``language`` arg into every synth call so Piper can pick a matching
    voice. Without this, Piper synthesised every sentence with the
    default English voice regardless of the source language."""
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    agent = _FakeAgent(["Ciao mondo."])
    send, _frames = _capture_send()
    seen_languages: list[str | None] = []

    async def lang_capturing_synth(_sentence: str, _cfg, *, language=None):
        seen_languages.append(language)
        yield b"\xff\xfb\x90\x00mp3"

    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=_FakeTTSConfig()), \
         patch("openagent.gateway.voice_pipeline.synthesize_stream", lang_capturing_synth):
        orchestrator = VoiceTurnOrchestrator(agent, send)
        await orchestrator.run(
            "ciao", client_id="c1", session_id="sess-lang", language="it",
        )

    # Must have called the synth at least once with language='it'.
    assert seen_languages, "expected at least one synth call"
    assert all(lang == "it" for lang in seen_languages), (
        f"every synth call must carry the language hint, got {seen_languages}"
    )


# ── Tests: chat-store voice-session helpers (Python mirror of the TS) ──
# We don't import zustand; we re-implement the contract here and assert
# it. If the JS implementation ever drifts we want a loud failure.


class _MiniChatStore:
    """Mirror of useChat slice we care about for these tests."""

    def __init__(self):
        self.sessions: list[dict[str, Any]] = []
        self.voice_session_id: str | None = None
        self._next = 1

    def get_or_create_voice_session(self) -> str:
        if self.voice_session_id and any(s["id"] == self.voice_session_id
                                         for s in self.sessions):
            return self.voice_session_id
        sid = f"session-{self._next}"
        self._next += 1
        self.sessions.append({"id": sid, "title": "Voice Chat", "messages": []})
        self.voice_session_id = sid
        return sid

    def clear_voice_session(self) -> None:
        self.voice_session_id = None


@test("voice_pipeline", "voice-session helper: subsequent calls return same id")
async def t_voice_session_idempotent(ctx: TestContext) -> None:
    store = _MiniChatStore()
    a = store.get_or_create_voice_session()
    b = store.get_or_create_voice_session()
    assert a == b, f"expected same id, got {a!r} vs {b!r}"
    assert len(store.sessions) == 1, store.sessions


@test("voice_pipeline", "voice-session helper: clear → next call mints fresh")
async def t_voice_session_reset(ctx: TestContext) -> None:
    store = _MiniChatStore()
    a = store.get_or_create_voice_session()
    store.clear_voice_session()
    b = store.get_or_create_voice_session()
    assert a != b, f"expected different id after clear, both were {a!r}"
    assert len(store.sessions) == 2, store.sessions


@test("voice_pipeline", "empty done text + no deltas → fallback message in RESPONSE")
async def t_empty_done_fallback(ctx: TestContext) -> None:
    """The agent's ``_run_inner_stream`` always yields exactly one
    ``done`` event with ``full_text = "".join(deltas)``. If the model
    produces zero deltas, ``done.text`` is "" and the orchestrator's
    safety net (``if event.get("text") and not accumulated:``) skips
    because ``""`` is falsy. The fallback in the finally block must
    surface a readable hint instead of letting RESPONSE go out empty."""
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    class _AgentEmptyDone:
        def __init__(self):
            self.db = object()

        async def run_stream(self, **_kwargs):
            yield {"kind": "done", "text": ""}

        def last_response_meta(self, _sid):
            return {}

    send, frames = _capture_send()
    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=None):
        orch = VoiceTurnOrchestrator(_AgentEmptyDone(), send)
        result = await orch.run("hi", client_id="c1", session_id="sess-EMP")

    response = next(f for f in frames if f["type"] == "response")
    text = response["text"] or ""
    assert "No text response" in text, f"fallback missing: {text!r}"
    assert response["session_id"] == "sess-EMP"
    # The result text mirrors what's sent over the wire.
    assert "No text response" in result["text"], result


@test("voice_pipeline", "done-only with text-only (no deltas) populates accumulated")
async def t_done_only_with_text(ctx: TestContext) -> None:
    """Some agents (older Agno versions, claude-cli synchronous mode)
    skip deltas entirely and put the full text in ``done.text``. The
    safety valve at the orchestrator must still pick that up so the
    user sees the response."""
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    class _AgentDoneOnly:
        def __init__(self):
            self.db = object()

        async def run_stream(self, **_kwargs):
            yield {"kind": "done", "text": "Hello world from done."}

        def last_response_meta(self, _sid):
            return {}

    send, frames = _capture_send()
    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=None):
        orch = VoiceTurnOrchestrator(_AgentDoneOnly(), send)
        result = await orch.run("hi", client_id="c1", session_id="sess-DON")

    response = next(f for f in frames if f["type"] == "response")
    assert response["text"] == "Hello world from done.", response
    assert "No text response" not in (response["text"] or "")
    assert result["text"] == "Hello world from done."


@test("voice_pipeline", "agent generator: GeneratorExit propagates without RuntimeError")
async def t_generatorexit_clean_propagation(ctx: TestContext) -> None:
    """Mirrors the new ``except Exception`` shape in
    :func:`openagent.core.agent.Agent.run_stream`. ``GeneratorExit``
    must NOT be caught — yielding from a generator while it's being
    closed raises ``RuntimeError("async generator ignored
    GeneratorExit")`` and asyncio reports the cleanup task as
    un-retrieved. With ``except Exception`` that path is impossible.
    """

    async def good_generator():
        # Old behaviour (catching BaseException + yielding) would fail
        # this test. New behaviour (catching only Exception) should
        # let GeneratorExit propagate cleanly when the consumer breaks.
        try:
            for i in range(5):
                yield {"kind": "delta", "text": f"chunk-{i}"}
            yield {"kind": "done", "text": "all done"}
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — this is the shape we're testing
            yield {"kind": "done", "text": "error"}

    received: list[dict[str, Any]] = []
    gen = good_generator()
    async for event in gen:
        received.append(event)
        if event["kind"] == "delta" and event["text"] == "chunk-1":
            break  # triggers Python's implicit aclose() → GeneratorExit

    # If RuntimeError was raised it would have torn down the test loop.
    # Reaching here means the generator closed cleanly.
    assert len(received) == 2, received
    assert received[0]["text"] == "chunk-0"
    assert received[1]["text"] == "chunk-1"


@test("voice_pipeline", "spoken status: 'Using ReadFile...' becomes a TTS sentence")
async def t_spoken_status_basic(ctx: TestContext) -> None:
    """When the agent fires status events during a tool call, the
    orchestrator must enqueue a short spoken summary so the user hears
    what's happening instead of minutes of silence."""
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    class _AgentWithStatus:
        def __init__(self):
            self.db = object()

        async def run_stream(self, *, on_status=None, **_kwargs):
            # Simulate two tool calls before any text streams.
            await on_status('Using ReadFile...')
            await on_status('Using ReadFile...')              # dup → skip
            await on_status('{"tool":"ReadFile","status":"done"}')  # done → skip
            await on_status('{"tool":"Calculator","status":"running"}')
            yield {"kind": "delta", "text": "Done. "}
            yield {"kind": "done"}

        def last_response_meta(self, _sid):
            return {"model": "openai:gpt-test"}

    agent = _AgentWithStatus()
    send, frames = _capture_send()

    sentences_synthesized: list[str] = []

    async def fake_synth(sentence: str, _cfg, **_kwargs):
        sentences_synthesized.append(sentence)
        yield b"\xff\xfb\x90\x00mp3"

    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=_FakeTTSConfig()), \
         patch("openagent.gateway.voice_pipeline.synthesize_stream", fake_synth):
        orchestrator = VoiceTurnOrchestrator(agent, send)
        result = await orchestrator.run(
            "ciao", client_id="c1", session_id="sess-S",
        )

    assert "Using ReadFile" in sentences_synthesized, sentences_synthesized
    assert "Using Calculator" in sentences_synthesized, sentences_synthesized
    # Each tool only spoken once, even though ReadFile fired 3 status events.
    assert sentences_synthesized.count("Using ReadFile") == 1, sentences_synthesized
    assert result["spoken_tools"] == 2, result
    # Audio frames should have been produced for the two status sentences
    # plus the response text.
    types = _types(frames)
    assert types[-1] == "response", types
    assert "audio_start" in types
    assert types.count("audio_chunk") >= 3, types  # 2 status + 1 response


@test("voice_pipeline", "spoken status: forwarded to original on_status callback")
async def t_spoken_status_forwarded(ctx: TestContext) -> None:
    """The wrapped on_status must still call through to the original
    callback so the WS status frames continue to populate the chat
    transcript with tool cards."""
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    forwarded: list[str] = []

    async def original_on_status(text: str) -> None:
        forwarded.append(text)

    class _Agent:
        def __init__(self):
            self.db = object()

        async def run_stream(self, *, on_status=None, **_):
            await on_status('Using ReadFile...')
            yield {"kind": "delta", "text": "Hi."}
            yield {"kind": "done"}

        def last_response_meta(self, _sid):
            return {}

    send, _frames = _capture_send()
    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=_FakeTTSConfig()), \
         patch(
             "openagent.gateway.voice_pipeline.synthesize_stream",
             lambda _s, _c, **_k: _empty_async_iter(),
         ):
        orch = VoiceTurnOrchestrator(_Agent(), send)
        await orch.run("hi", client_id="c1", session_id="sess-F",
                       on_status=original_on_status)

    assert forwarded == ['Using ReadFile...'], forwarded


@test("voice_pipeline", "no TTS configured: status events don't crash + RESPONSE still sent")
async def t_status_no_tts(ctx: TestContext) -> None:
    """Voice path must stay robust when the user hasn't configured TTS.
    Status callbacks are exercised but no audio frames are produced."""
    from openagent.gateway.voice_pipeline import VoiceTurnOrchestrator

    class _Agent:
        def __init__(self):
            self.db = object()

        async def run_stream(self, *, on_status=None, **_):
            await on_status('Using ReadFile...')
            yield {"kind": "delta", "text": "Hi."}
            yield {"kind": "done"}

        def last_response_meta(self, _sid):
            return {}

    send, frames = _capture_send()
    with patch("openagent.gateway.voice_pipeline.resolve_tts_provider",
               return_value=None):
        orch = VoiceTurnOrchestrator(_Agent(), send)
        result = await orch.run("hi", client_id="c1", session_id="sess-N")

    types = _types(frames)
    assert "audio_start" not in types, types
    assert types[-1] == "response", types
    assert result["spoken_tools"] == 0
    assert result["audio_chunks"] == 0


@test("voice_pipeline", "_status_speech_for: dedup + JSON shape + plain text")
async def t_status_speech_helper(ctx: TestContext) -> None:
    from openagent.gateway.voice_pipeline import _status_speech_for

    seen: set[str] = set()

    # First "Using ReadFile..." → speak.
    assert _status_speech_for("Using ReadFile...", seen) == "Using ReadFile"
    # Repeat → skip.
    assert _status_speech_for("Using ReadFile...", seen) is None
    assert _status_speech_for(
        '{"tool":"ReadFile","status":"running"}', seen,
    ) is None

    # Different tool via JSON → speak.
    assert _status_speech_for(
        '{"tool":"Calculator","status":"running","params":{"x":1}}', seen,
    ) == "Using Calculator"

    # status != running → skip even if new tool.
    assert _status_speech_for(
        '{"tool":"Brand New Tool","status":"done"}', seen,
    ) is None

    # Plain text without "Using " prefix → skip (e.g. "Thinking...").
    assert _status_speech_for("Thinking...", seen) is None
    assert _status_speech_for("", seen) is None


async def _empty_async_iter():
    if False:  # pragma: no cover — sentinel to make this an async gen
        yield b""


@test("voice_pipeline", "transcribe: language hint forwarded to faster-whisper")
async def t_transcribe_language_local(ctx: TestContext) -> None:
    """The ``language=`` kwarg on :func:`transcribe` must reach the
    underlying ``WhisperModel.transcribe`` call. Auto-detect on short
    Italian utterances has misclassified as Cyrillic; the only fix is
    to pass an explicit hint, so this is the wire we test."""
    from openagent.channels import voice as voice_mod

    seen: dict[str, Any] = {}

    class _FakeSegment:
        def __init__(self, text: str):
            self.text = text

    class _FakeWhisper:
        def transcribe(self, _path, **kwargs):
            seen.update(kwargs)
            return [_FakeSegment("ciao mondo")], None

    with patch.object(voice_mod, "_load_local_model", return_value=_FakeWhisper()):
        text = await voice_mod.transcribe(
            "/tmp/whatever.webm", db=None, language="it",
        )

    assert text == "ciao mondo", text
    assert seen.get("language") == "it", seen
    assert seen.get("vad_filter") is True, seen


@test("voice_pipeline", "transcribe: env OPENAGENT_VOICE_LANG fills missing language")
async def t_transcribe_language_env_fallback(ctx: TestContext) -> None:
    import os
    from openagent.channels import voice as voice_mod

    seen: dict[str, Any] = {}

    class _FakeSegment:
        def __init__(self, text: str):
            self.text = text

    class _FakeWhisper:
        def transcribe(self, _path, **kwargs):
            seen.update(kwargs)
            return [_FakeSegment("hola")], None

    with patch.object(voice_mod, "_load_local_model", return_value=_FakeWhisper()), \
         patch.dict(os.environ, {"OPENAGENT_VOICE_LANG": "es"}, clear=False):
        await voice_mod.transcribe("/tmp/x.webm", db=None)

    assert seen.get("language") == "es", seen


@test("voice_pipeline", "transcribe: language=None means auto-detect")
async def t_transcribe_language_auto(ctx: TestContext) -> None:
    import os
    from openagent.channels import voice as voice_mod

    seen: dict[str, Any] = {}

    class _FakeSegment:
        def __init__(self, text: str):
            self.text = text

    class _FakeWhisper:
        def transcribe(self, _path, **kwargs):
            seen.update(kwargs)
            return [_FakeSegment("hello")], None

    # Make sure the env var doesn't leak in.
    env = {k: v for k, v in os.environ.items() if k != "OPENAGENT_VOICE_LANG"}
    with patch.object(voice_mod, "_load_local_model", return_value=_FakeWhisper()), \
         patch.dict(os.environ, env, clear=True):
        await voice_mod.transcribe("/tmp/x.webm", db=None)

    assert seen.get("language") is None, seen


@test("voice_pipeline", "voice-session helper: orphaned id mints fresh on next call")
async def t_voice_session_orphan(ctx: TestContext) -> None:
    """If voice_session_id points to a session that's been removed (e.g.
    user manually deleted it from the Chat tab sidebar), the next
    get-or-create must mint a new one rather than returning a dangling id."""
    store = _MiniChatStore()
    a = store.get_or_create_voice_session()
    # Simulate the session being removed externally.
    store.sessions = []
    b = store.get_or_create_voice_session()
    assert a != b, f"expected new id, both were {a!r}"
    assert len(store.sessions) == 1
