"""ElevenLabs WebSocket streaming TTS — token-in / audio-out path.

Covers ``openagent.channels.tts_streaming.synthesize_token_stream`` and
``supports_token_stream``. Uses a real ``websockets`` server bound to a
free port to exercise the full BOS / text-frame / EOS protocol without
hitting the real ElevenLabs endpoint (no API key, no quota burn, no
flaky network).
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from ._framework import TestContext, free_port, test


# ── Helpers ─────────────────────────────────────────────────────────


def _make_cfg(**overrides: Any):
    """Build a minimal TTSConfig that opts into the WS streaming path."""
    from openagent.channels.tts import TTSConfig

    base = dict(
        vendor="elevenlabs",
        model_id="eleven_flash_v2_5",
        voice_id="test-voice-id",
        api_key="test-key",
        base_url=None,
        response_format="mp3",
        stream_input=True,
    )
    base.update(overrides)
    return TTSConfig(**base)


async def _async_iter(items: list[str]):
    """Wrap a list of strings as an async iterator the synth can consume."""
    for item in items:
        yield item


# ── Tests ───────────────────────────────────────────────────────────


@test("tts_streaming", "supports_token_stream gates correctly on vendor + flag")
async def t_supports_gate(_ctx: TestContext) -> None:
    from openagent.channels.tts_streaming import supports_token_stream

    assert supports_token_stream(None) is False
    # Right vendor + flag → True
    assert supports_token_stream(_make_cfg()) is True
    # Right vendor, flag off → False (today's per-sentence path)
    assert supports_token_stream(_make_cfg(stream_input=False)) is False
    # Wrong vendor, flag on → False (no WS endpoint to call)
    assert supports_token_stream(
        _make_cfg(vendor="openai", stream_input=True),
    ) is False


@test("tts_streaming", "synthesize_token_stream returns immediately for non-WS cfg")
async def t_no_ws_returns_quickly(_ctx: TestContext) -> None:
    """When the cfg doesn't opt in, the token-stream surface must
    return without yielding anything (and without trying to connect).
    The turn_runner branch relies on this for a fast no-op when
    stream_input isn't set."""
    from openagent.channels.tts_streaming import synthesize_token_stream

    cfg = _make_cfg(stream_input=False)
    chunks: list[bytes] = []
    async for c in synthesize_token_stream(_async_iter(["hello"]), cfg):
        chunks.append(c)
    assert chunks == [], f"expected no audio when stream_input is False, got {chunks}"


@test("tts_streaming", "WS path: BOS sent, text frames forwarded, audio yielded, EOS closes")
async def t_ws_full_roundtrip(_ctx: TestContext) -> None:
    """Spin up a real websockets server; the synth client connects,
    sends BOS + 2 text frames + EOS, and receives 2 audio frames + a
    final marker. We assert the wire-protocol contract (BOS first,
    EOS last, our deltas in order) AND the bytes-out behaviour."""
    import websockets

    received_frames: list[dict] = []
    port = free_port()

    async def _server(ws):
        # Drain everything the client sends; respond with two audio
        # frames mid-stream, then a final marker after EOS.
        async def _read_all():
            try:
                async for raw in ws:
                    received_frames.append(json.loads(raw))
                    if received_frames[-1].get("text") == "":
                        # EOS — stop reading.
                        break
            except websockets.exceptions.ConnectionClosed:
                pass

        # Send 2 audio frames as soon as the client connects, regardless
        # of input — ElevenLabs may emit audio early on greeting frames.
        # Then send isFinal AFTER EOS arrives.
        async def _emit():
            await ws.send(json.dumps({
                "audio": base64.b64encode(b"AUDIO-CHUNK-1").decode("ascii"),
                "isFinal": False,
            }))
            await ws.send(json.dumps({
                "audio": base64.b64encode(b"AUDIO-CHUNK-2").decode("ascii"),
                "isFinal": False,
            }))

        await _emit()
        await _read_all()
        await ws.send(json.dumps({"audio": "", "isFinal": True}))

    server = await websockets.serve(_server, "127.0.0.1", port)
    try:
        # Patch the WS URL template so the synth connects to our test
        # server instead of the real ElevenLabs endpoint.
        from openagent.channels import tts_streaming
        original_url = tts_streaming._ELEVENLABS_WS
        tts_streaming._ELEVENLABS_WS = f"ws://127.0.0.1:{port}/{{voice}}"
        try:
            cfg = _make_cfg()
            chunks: list[bytes] = []
            async for audio in tts_streaming.synthesize_token_stream(
                _async_iter(["Hello", " world"]), cfg,
            ):
                chunks.append(audio)
        finally:
            tts_streaming._ELEVENLABS_WS = original_url
    finally:
        server.close()
        await server.wait_closed()

    # Wire-protocol assertions — BOS first, deltas in order, EOS last.
    assert len(received_frames) >= 4, received_frames
    bos = received_frames[0]
    assert "voice_settings" in bos and "xi_api_key" in bos, (
        f"first frame must be BOS with voice_settings + xi_api_key: {bos}"
    )
    # Text frames in order — non-EOS frames after BOS that have a `text`.
    text_frames = [
        f for f in received_frames[1:]
        if f.get("text") and f.get("text") != ""
    ]
    assert [f["text"] for f in text_frames] == ["Hello", " world"], text_frames
    # EOS — last frame with text=="".
    assert received_frames[-1].get("text") == "", received_frames[-1]

    # Audio bytes — both server chunks reached the caller.
    assert chunks == [b"AUDIO-CHUNK-1", b"AUDIO-CHUNK-2"], chunks


@test("tts_streaming", "WS connection refused → raises so caller can fall back")
async def t_ws_connection_refused(_ctx: TestContext) -> None:
    """The voice pipeline relies on raised exceptions to fall back to
    the per-sentence REST path. A silent return on connection failure
    would leave the user with no audio AND no diagnostic."""
    from openagent.channels import tts_streaming

    # Point at a port nothing is listening on. ``free_port`` returns
    # an unbound port, which will refuse connection cleanly.
    closed_port = free_port()
    original_url = tts_streaming._ELEVENLABS_WS
    tts_streaming._ELEVENLABS_WS = f"ws://127.0.0.1:{closed_port}/{{voice}}"

    raised = False
    try:
        cfg = _make_cfg()
        try:
            async for _ in tts_streaming.synthesize_token_stream(
                _async_iter(["hi"]), cfg,
            ):
                pass
        except Exception:
            raised = True
    finally:
        tts_streaming._ELEVENLABS_WS = original_url

    assert raised, (
        "WS connection refused must raise so turn_runner can fall "
        "back to per-sentence REST path"
    )


@test("tts_streaming", "missing voice_id raises before connecting")
async def t_missing_voice_id(_ctx: TestContext) -> None:
    """ElevenLabs WS requires voice_id in the URL path. Catch this
    early instead of trying to connect to ``.../v1/text-to-speech//
    stream-input`` which would 400."""
    from openagent.channels.tts_streaming import synthesize_token_stream

    cfg = _make_cfg(voice_id="")
    raised: Exception | None = None
    try:
        async for _ in synthesize_token_stream(_async_iter(["hi"]), cfg):
            pass
    except ValueError as e:
        raised = e
    assert raised is not None and "voice_id" in str(raised), raised


@test("tts_streaming", "missing api_key raises before connecting")
async def t_missing_api_key(_ctx: TestContext) -> None:
    """ElevenLabs WS auth is the BOS frame's xi_api_key. No key →
    immediate ValueError so we don't burn a connection that would
    fail on the first response."""
    from openagent.channels.tts_streaming import synthesize_token_stream

    cfg = _make_cfg(api_key=None)
    raised: Exception | None = None
    try:
        async for _ in synthesize_token_stream(_async_iter(["hi"]), cfg):
            pass
    except ValueError as e:
        raised = e
    assert raised is not None and "api_key" in str(raised), raised
