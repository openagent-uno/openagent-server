"""Voice receive — end-to-end regression suite.

Closes the gap the suite had before: every voice path was unit-tested
with the layer below mocked out, so production failures could hide
between the seams.

These tests use REAL components wired in real composition:

  * ``t_local_stt_real_audio`` — faster-whisper actually transcribes
    a real (synthetic) speech file. Catches a broken local STT
    install before the bridge or gateway path can mask it.
  * ``t_local_stt_handles_telegram_ogg`` — same, but on the OGG/OPUS
    container Telegram delivers voice notes in. Catches container-
    handling regressions in faster-whisper / ffmpeg.
  * ``t_bridge_fallback_with_dead_gateway`` — real ``TelegramBridge``
    pointed at an unreachable gateway URL; ``transcribe_with_fallback``
    must route through ``transcribe_via_gateway`` → fail fast → local
    Whisper → real text. Catches a bridge fallback that silently
    returns VOICE_FALLBACK when local STT works.
  * ``t_gateway_stt_route_transcribes_upload`` — boot a minimal
    ``aiohttp`` app with the REAL ``Gateway._handle_stt_transcribe``
    handler, POST the real audio bytes, assert text comes back.
    Catches the route-side regression the bridge would hit through
    the InProc loopback.
  * ``t_telegram_voice_extract_calls_transcribe`` — real
    ``TelegramBridge._extract_files`` with a fake ``msg.voice`` that
    writes a real audio file to disk; the extracted text must contain
    the recognised words. Catches the receive-handler regression
    where the bridge silently drops voice attachments.

Each test generates its fixture audio via macOS ``say`` (already
installed; falls back to skipping if not on a Mac). Tests skip
gracefully when faster-whisper / ffmpeg isn't installed in the run
environment so they don't false-fail on slim CI images, but they
RUN on the developer machine where the bug actually lives.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ._framework import TestContext, TestSkip, test


# ── Fixture audio generation ─────────────────────────────────────────


_TEST_PHRASE = (
    "Hello, this is a test of speech-to-text. Can you transcribe this?"
)
# Words the model must spit back (case-insensitive substring match)
# for the test to consider transcription "working." We don't pin the
# exact transcript because Whisper drifts on punctuation / casing
# between versions — pinning whole strings would make the test brittle
# without adding value.
_EXPECTED_WORDS = ("test", "speech", "transcribe")


def _make_wav_fixture(path: str) -> bool:
    """Generate ``path`` as a 16kHz mono WAV of ``_TEST_PHRASE``.

    Returns False (and the caller should skip) when macOS ``say`` is
    missing — Linux / Windows runners can't easily synth a test clip
    inline and there's no point pretending; the bridge-fallback test
    can still skip without false-failing.
    """
    if not shutil.which("say"):
        return False
    try:
        subprocess.run(
            ["say", "--data-format=LEI16@16000", "-o", path, _TEST_PHRASE],
            check=True, capture_output=True, timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return Path(path).exists() and Path(path).stat().st_size > 0


def _make_ogg_fixture(wav_path: str, ogg_path: str) -> bool:
    """Transcode WAV to OGG/OPUS (Telegram's voice-note format) via ffmpeg."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", ogg_path],
            check=True, capture_output=True, timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return Path(ogg_path).exists() and Path(ogg_path).stat().st_size > 0


def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _assert_transcript_recognised(text: str) -> None:
    """Pin the substring contract — leave room for Whisper drift but
    fail loudly when the transcription regresses to empty / gibberish."""
    assert text, f"transcription returned empty/falsy: {text!r}"
    lower = text.lower()
    missing = [w for w in _EXPECTED_WORDS if w not in lower]
    assert not missing, (
        f"transcription missing expected words {missing}: {text!r}"
    )


# ── 1. Local STT round-trip on a real audio file ─────────────────────


@test("voice_e2e", "faster-whisper transcribes a real WAV → expected words present")
async def t_local_stt_real_audio(_ctx: TestContext) -> None:
    if not _faster_whisper_available():
        raise TestSkip("faster-whisper not installed in this env")

    with tempfile.TemporaryDirectory(prefix="oa_v_") as tmp:
        wav = str(Path(tmp) / "speech.wav")
        if not _make_wav_fixture(wav):
            raise TestSkip("cannot synthesise test audio (no `say` binary)")

        from src.channels.voice import transcribe
        text = await asyncio.wait_for(
            transcribe(wav, db=None, language="en"),
            timeout=30.0,
        )

    _assert_transcript_recognised(text or "")


@test("voice_e2e", "faster-whisper handles the OGG/OPUS container Telegram delivers")
async def t_local_stt_handles_telegram_ogg(_ctx: TestContext) -> None:
    """Production voice notes arrive as OGG/OPUS. If faster-whisper
    or its libsndfile binding regresses on OGG (a known shape of
    breakage when shipping a new wheel) the bridge will fall through
    to VOICE_FALLBACK silently. This pins that path."""
    if not _faster_whisper_available():
        raise TestSkip("faster-whisper not installed in this env")

    with tempfile.TemporaryDirectory(prefix="oa_v_") as tmp:
        wav = str(Path(tmp) / "speech.wav")
        ogg = str(Path(tmp) / "voice.ogg")
        if not _make_wav_fixture(wav):
            raise TestSkip("cannot synthesise test audio (no `say` binary)")
        if not _make_ogg_fixture(wav, ogg):
            raise TestSkip("cannot transcode to OGG (no ffmpeg)")

        from src.channels.voice import transcribe
        text = await asyncio.wait_for(
            transcribe(ogg, db=None, language="en"),
            timeout=30.0,
        )

    _assert_transcript_recognised(text or "")


# ── 2. Bridge fallback path with no reachable gateway ────────────────


@test("voice_e2e", "bridge transcribe_with_fallback returns real text when gateway is dead")
async def t_bridge_fallback_with_dead_gateway(_ctx: TestContext) -> None:
    """The bridge's gateway STT POST must fail FAST against an
    unreachable URL and route through to local Whisper. A regression
    where the bridge hangs on the dead gateway OR swallows the local
    fallback would surface here as a VOICE_FALLBACK return or a
    timeout."""
    if not _faster_whisper_available():
        raise TestSkip("faster-whisper not installed in this env")

    from src.bridges.base import VOICE_FALLBACK
    from src.bridges.telegram import TelegramBridge

    with tempfile.TemporaryDirectory(prefix="oa_v_") as tmp:
        wav = str(Path(tmp) / "speech.wav")
        if not _make_wav_fixture(wav):
            raise TestSkip("cannot synthesise test audio (no `say` binary)")

        # Port 1 (TCP echo on most systems) so the connection refuses
        # immediately rather than hanging — keeps the test fast even
        # when the gateway path is genuinely broken.
        bridge = TelegramBridge(
            token="fake-token",
            allowed_users=None,
            gateway_url="ws://127.0.0.1:1/ws",
        )
        try:
            text = await asyncio.wait_for(
                bridge.transcribe_with_fallback(wav),
                timeout=30.0,
            )
        finally:
            # Close the aiohttp client session the bridge lazily opened
            # for the gateway probe so the loop doesn't whine about
            # unclosed connections after the test.
            if bridge._http_session is not None:
                await bridge._http_session.close()

    assert text != VOICE_FALLBACK, (
        "bridge returned VOICE_FALLBACK even though local STT works — "
        "the fallback chain is broken"
    )
    _assert_transcript_recognised(text)


# ── 3. Real gateway STT route end-to-end ─────────────────────────────


@test("voice_e2e", "gateway /api/stt/transcribe returns text for a real audio upload")
async def t_gateway_stt_route_transcribes_upload(_ctx: TestContext) -> None:
    """Boot a minimal aiohttp app with the real
    ``Gateway._handle_stt_transcribe`` mounted, POST a real audio
    file as multipart form, assert the JSON response carries the
    transcribed text.

    Bypasses the cert middleware because we're testing the ROUTE,
    not auth. Auth coverage lives in test_bridge_session.py.
    """
    if not _faster_whisper_available():
        raise TestSkip("faster-whisper not installed in this env")

    from aiohttp import web, ClientSession, FormData, ClientTimeout
    from src.gateway.server import Gateway

    with tempfile.TemporaryDirectory(prefix="oa_v_") as tmp:
        wav = str(Path(tmp) / "speech.wav")
        if not _make_wav_fixture(wav):
            raise TestSkip("cannot synthesise test audio (no `say` binary)")

        # Minimal Gateway stub: only need ``self.agent.db`` for the
        # handler's ``getattr(self.agent, "db", None)`` lookup. db=None
        # makes the handler skip the DB-resolved STT vendor and fall
        # through to local Whisper — exactly the user's deployment.
        class _FakeAgent:
            db = None

        gw = Gateway.__new__(Gateway)
        gw.agent = _FakeAgent()

        app = web.Application()
        app.router.add_post("/api/stt/transcribe", gw._handle_stt_transcribe)

        runner = web.AppRunner(app)
        await runner.setup()
        from ._framework import free_port
        port = free_port()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()

        try:
            async with ClientSession() as http:
                form = FormData()
                with open(wav, "rb") as fh:
                    form.add_field(
                        "file", fh, filename="speech.wav", content_type="audio/wav",
                    )
                    async with http.post(
                        f"http://127.0.0.1:{port}/api/stt/transcribe?lang=en",
                        data=form,
                        timeout=ClientTimeout(total=30),
                    ) as resp:
                        assert resp.status == 200, (
                            f"gateway STT returned {resp.status}: "
                            f"{await resp.text()}"
                        )
                        body = await resp.json()
        finally:
            await site.stop()
            await runner.cleanup()

    _assert_transcript_recognised(body.get("text") or "")


# ── 4. Telegram receive path with a real audio file ──────────────────


@test("voice_e2e", "telegram _extract_files transcribes a real voice attachment")
async def t_telegram_voice_extract_calls_transcribe(_ctx: TestContext) -> None:
    """The bridge's voice-receive handler: given a ``msg.voice`` whose
    ``download_to_drive`` writes a real OGG to disk, ``_extract_files``
    must call ``transcribe_with_fallback`` (not bypass it) AND return
    the recognised text in ``text_addition`` AND flip ``voice_detected``.

    This is what the user's Telegram voice flow exercises end-to-end
    on the bridge side. A regression here would silently drop voice
    notes back to VOICE_FALLBACK without local STT ever running.
    """
    if not _faster_whisper_available():
        raise TestSkip("faster-whisper not installed in this env")

    from src.bridges.telegram import TelegramBridge

    with tempfile.TemporaryDirectory(prefix="oa_v_") as tmp:
        wav = str(Path(tmp) / "speech.wav")
        ogg = str(Path(tmp) / "voice.ogg")
        if not _make_wav_fixture(wav):
            raise TestSkip("cannot synthesise test audio (no `say` binary)")
        if not _make_ogg_fixture(wav, ogg):
            raise TestSkip("cannot transcode to OGG (no ffmpeg)")
        ogg_bytes = Path(ogg).read_bytes()

        # Telegram-python-bot's voice object exposes `.get_file()` →
        # `File.download_to_drive(path)` which writes bytes. Mirror the
        # surface with a fake that copies our prepared OGG into the
        # bridge's tmp dir at the exact path the handler picks.
        class _FakeFile:
            async def download_to_drive(self, target: str) -> None:
                Path(target).write_bytes(ogg_bytes)

        class _FakeVoice:
            file_unique_id = "test_unique_id"

            async def get_file(self) -> _FakeFile:
                return _FakeFile()

        class _FakeMessage:
            def __init__(self) -> None:
                self.photo = None
                self.voice = _FakeVoice()
                self.audio = None
                self.document = None
                self.video = None

            async def reply_text(self, *_a, **_k) -> None:
                pass

        bridge = TelegramBridge(token="fake-token", allowed_users=None)
        try:
            with tempfile.TemporaryDirectory(prefix="oa_tg_") as bridge_tmp:
                extracted = await asyncio.wait_for(
                    bridge._extract_files(_FakeMessage(), bridge_tmp),
                    timeout=30.0,
                )
        finally:
            if bridge._http_session is not None:
                await bridge._http_session.close()

    assert extracted.voice_detected is True, (
        f"voice flag not set on extraction result: {extracted}"
    )
    _assert_transcript_recognised(extracted.text_addition or "")
    assert extracted.files_info == [], (
        f"voice file should not leak into files_info: {extracted.files_info}"
    )
