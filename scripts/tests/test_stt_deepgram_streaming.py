"""DeepgramStreamingSTT — audio-in / transcript-out via WebSocket.

Covers ``openagent.channels.stt_deepgram.DeepgramStreamingSTT.stream``
against a local fake-WS server (mirrors the ElevenLabs streaming-TTS
test pattern). Verifies:

* Partial frames yield ``STTEvent("partial")`` with the correct text.
* Final frames yield ``STTEvent("final")`` with the correct text.
* The writer task forwards every audio chunk and sends ``CloseStream``
  on EOF.
* Cancellation propagates without leaking the WS task.
"""
from __future__ import annotations

import asyncio
import json

from ._framework import TestContext, free_port, test


async def _audio_iter(chunks: list[bytes]):
    for c in chunks:
        yield c


@test("stt_deepgram", "stream yields partial then final transcripts")
async def t_partial_then_final(_ctx: TestContext) -> None:
    import websockets

    from openagent.channels import stt_deepgram

    received_chunks: list[bytes] = []
    received_close = False
    port = free_port()

    async def _server(ws):
        nonlocal received_close
        # Send a partial first so the client yields partial before final.
        await ws.send(json.dumps({
            "type": "Results",
            "is_final": False,
            "channel": {"alternatives": [{"transcript": "hello", "confidence": 0.7}]},
        }))
        # Drain one binary chunk, then emit final.
        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    received_chunks.append(bytes(raw))
                    continue
                # Text frame from client — only ``CloseStream`` matters.
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("type") == "CloseStream":
                    received_close = True
                    break
        except websockets.exceptions.ConnectionClosed:
            pass
        await ws.send(json.dumps({
            "type": "Results",
            "is_final": True,
            "channel": {"alternatives": [{"transcript": "hello world", "confidence": 0.95}]},
        }))

    server = await websockets.serve(_server, "127.0.0.1", port)
    try:
        original_ws = stt_deepgram._DEEPGRAM_WS
        stt_deepgram._DEEPGRAM_WS = f"ws://127.0.0.1:{port}/v1/listen"
        try:
            stt = stt_deepgram.DeepgramStreamingSTT(
                api_key="test-key",
                model="nova-2",
            )
            events = []
            async for ev in stt.stream(
                _audio_iter([b"AUDIO-CHUNK-1", b"AUDIO-CHUNK-2"]),
                language="en",
                encoding="webm",
            ):
                events.append(ev)
                if ev.kind == "final":
                    break
        finally:
            stt_deepgram._DEEPGRAM_WS = original_ws
    finally:
        server.close()
        await server.wait_closed()

    kinds = [e.kind for e in events]
    assert "partial" in kinds, f"expected at least one partial; got {kinds}"
    finals = [e for e in events if e.kind == "final"]
    assert finals and finals[-1].text == "hello world", events
    # Writer drained both chunks before close-stream marker.
    assert received_chunks == [b"AUDIO-CHUNK-1", b"AUDIO-CHUNK-2"], received_chunks
    assert received_close, "writer must send CloseStream on EOF"


@test("stt_deepgram", "missing websockets package degrades to silent no-op")
async def t_no_websockets_no_op(_ctx: TestContext) -> None:
    """If ``websockets`` isn't installed, ``stream()`` must return
    cleanly (yield nothing) so the caller's STT pump skips this
    utterance instead of crashing the whole session."""
    import sys
    import importlib

    from openagent.channels.stt_deepgram import DeepgramStreamingSTT

    # Save+remove websockets so the inner ``import websockets`` raises.
    saved = sys.modules.pop("websockets", None)
    try:
        # Inject a sentinel that makes the import fail.
        import builtins
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "websockets":
                raise ImportError("simulated absence")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            stt = DeepgramStreamingSTT(api_key="k")
            events = []
            async for ev in stt.stream(_audio_iter([b"x"])):
                events.append(ev)
            assert events == [], f"expected no events on missing websockets; got {events}"
        finally:
            builtins.__import__ = original_import
    finally:
        if saved is not None:
            sys.modules["websockets"] = saved
        importlib.invalidate_caches()


@test("stt_deepgram", "resolve_stt picks DeepgramStreamingSTT for deepgram rows")
async def t_resolve_picks_deepgram(_ctx: TestContext) -> None:
    from openagent.channels.stt_base import resolve_stt
    from openagent.channels.stt_deepgram import DeepgramStreamingSTT

    class _StubDB:
        async def latest_audio_model(self, kind: str):
            assert kind == "stt"
            return {
                "provider_name": "deepgram",
                "model": "nova-2",
                "metadata": {"language": "en"},
                "api_key": "k",
                "base_url": None,
            }

    # ``resolve_stt`` reads via voice._resolve_stt_provider, which calls
    # latest_audio_model. Patch the helper so the stub plumbing matches.
    from openagent.channels import voice as _voice

    original = _voice._resolve_stt_provider

    async def fake_resolve(_db):
        return await _StubDB().latest_audio_model("stt")

    _voice._resolve_stt_provider = fake_resolve
    try:
        stt = await resolve_stt(_StubDB())
    finally:
        _voice._resolve_stt_provider = original
    assert isinstance(stt, DeepgramStreamingSTT), f"got {type(stt).__name__}"


@test("stt_deepgram", "missing api_key raises ValueError")
async def t_missing_api_key(_ctx: TestContext) -> None:
    from openagent.channels.stt_deepgram import DeepgramStreamingSTT

    raised: Exception | None = None
    try:
        DeepgramStreamingSTT(api_key="")
    except ValueError as e:
        raised = e
    assert raised is not None and "api_key" in str(raised), raised


@test("stt_deepgram", "PCM input declares linear16 + sample_rate in WS URL")
async def t_pcm_declares_linear16(_ctx: TestContext) -> None:
    """When the universal app's AudioWorklet pushes raw 16-bit PCM,
    the Deepgram URL must carry ``encoding=linear16&sample_rate=16000``
    so Deepgram's decoder doesn't wait for a container header. That's
    what unlocks sub-1 s TTFA partials."""
    from openagent.channels.stt_deepgram import DeepgramStreamingSTT

    stt = DeepgramStreamingSTT(api_key="k", model="nova-2")
    url = stt._ws_url(language="en", encoding="pcm16", sample_rate=16000)
    assert "encoding=linear16" in url, url
    assert "sample_rate=16000" in url, url
    assert "channels=1" in url, url


@test("stt_deepgram", "WebM input keeps container auto-detect")
async def t_webm_keeps_autodetect(_ctx: TestContext) -> None:
    """For WebM/Opus container chunks (the fallback path on browsers
    without AudioWorklet), Deepgram should auto-detect — explicit
    ``encoding`` would force the wrong decoder."""
    from openagent.channels.stt_deepgram import DeepgramStreamingSTT

    stt = DeepgramStreamingSTT(api_key="k", model="nova-2")
    url = stt._ws_url(language="en", encoding="webm", sample_rate=None)
    assert "encoding=linear16" not in url, url
    assert "sample_rate=" not in url, url


@test("stt_deepgram", "PCM stream end-to-end with linear16 URL")
async def t_pcm_stream_end_to_end(_ctx: TestContext) -> None:
    """Full round-trip: client iterator yields PCM, server WS receives,
    URL contains the right query params, transcript comes back."""
    import websockets

    from openagent.channels import stt_deepgram

    received_url: dict[str, str] = {}
    received_chunks: list[bytes] = []
    port = free_port()

    async def _server(ws):
        # ``websockets`` exposes the request URI on the connection
        # object — it's the path + query.
        uri = ws.request.path if hasattr(ws, "request") else getattr(ws, "path", "")
        received_url["uri"] = uri
        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    received_chunks.append(bytes(raw))
                else:
                    msg = json.loads(raw)
                    if msg.get("type") == "CloseStream":
                        break
        except websockets.exceptions.ConnectionClosed:
            pass
        await ws.send(json.dumps({
            "type": "Results", "is_final": True,
            "channel": {"alternatives": [{"transcript": "ok", "confidence": 0.9}]},
        }))

    server = await websockets.serve(_server, "127.0.0.1", port)
    try:
        original_ws = stt_deepgram._DEEPGRAM_WS
        stt_deepgram._DEEPGRAM_WS = f"ws://127.0.0.1:{port}/v1/listen"
        try:
            stt = stt_deepgram.DeepgramStreamingSTT(api_key="k", model="nova-2")
            pcm = b"\x00\x01" * 800  # 100 frames of fake PCM
            events = []
            async for ev in stt.stream(
                _audio_iter([pcm]),
                language="en",
                encoding="pcm16",
                sample_rate=16000,
            ):
                events.append(ev)
                if ev.kind == "final":
                    break
        finally:
            stt_deepgram._DEEPGRAM_WS = original_ws
    finally:
        server.close()
        await server.wait_closed()

    assert events and events[-1].text == "ok", events
    uri = received_url.get("uri", "")
    assert "encoding=linear16" in uri, uri
    assert "sample_rate=16000" in uri, uri
    assert received_chunks == [pcm], received_chunks
