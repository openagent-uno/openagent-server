"""Voice / audio handling — is_audio_file + upload-endpoint path.

``transcribe`` itself needs either ``faster-whisper`` or an OpenAI Whisper
key, both optional, so we don't assert actual transcription. We verify
that: (a) the audio detector recognises common formats, and (b) the
upload endpoint still succeeds when given an audio file (transcription
is best-effort and gracefully omitted if no backend is available).
"""
from __future__ import annotations

from ._framework import TestContext, TestSkip, test


@test("voice", "is_audio_file recognises common formats")
async def t_is_audio_file(ctx: TestContext) -> None:
    from openagent.channels.voice import is_audio_file
    for good in ("voice.mp3", "note.ogg", "recording.webm", "clip.m4a", "speech.wav"):
        assert is_audio_file(good), f"{good!r} should count as audio"
    for bad in ("photo.png", "doc.pdf", "script.py", None, ""):
        assert not is_audio_file(bad), f"{bad!r} should NOT count as audio"


@test("voice", "POST /api/upload accepts audio, returns transcription or not")
async def t_upload_audio(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    # A tiny MP3 frame header — the file isn't real audio, but the server
    # branches on extension, not byte inspection. We're verifying the path
    # (extension detected → transcription attempt → graceful omission on
    # failure, still 200) — not that we can transcribe garbage.
    fake_mp3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 128
    async with aiohttp.ClientSession() as http:
        data = aiohttp.FormData()
        data.add_field("file", fake_mp3, filename="clip.mp3", content_type="audio/mpeg")
        async with http.post(f"http://127.0.0.1:{port}/api/upload", data=data) as r:
            if r.status == 401:
                raise TestSkip("upload behind auth")
            assert r.status in (200, 201), f"status {r.status}"
            body = await r.json()
            assert "path" in body, body
            # `transcription` is optional — present if a whisper backend
            # worked, absent if it failed or wasn't configured. Both are OK.
