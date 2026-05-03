"""Deepgram streaming-STT adapter.

Audio-in / transcript-out over Deepgram's ``/v1/listen`` WebSocket. This
is the streaming counterpart to :class:`LiteLLMSTT`'s one-shot REST path
— interim partials land in ~150–300 ms and the final commits inside the
tail of the user's last syllable. Used by :class:`StreamSession`'s STT
pump when the active ``kind='stt'`` row in the SQLite catalog has
``provider_name='deepgram'``.

Mirrors the WebSocket-orchestration pattern from
:mod:`openagent.channels.tts_streaming`: a writer task drains audio
bytes from the inbound iterator, a reader loop parses JSON frames and
yields :class:`STTEvent`. The default :meth:`BaseSTT.stream` (tempfile
buffering + one-shot) stays as the fallback for vendors without native
streaming.

REST :meth:`transcribe_file` is provided for parity (used by the
gateway's ``/api/stt/transcribe`` REST endpoint when a Deepgram row is
selected). It uses the same model id and language as the streaming
path, so a single DB row drives both surfaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator
from urllib.parse import urlencode

from openagent.channels.stt_base import BaseSTT, STTEvent
from openagent.core.logging import elog

logger = logging.getLogger(__name__)


_DEEPGRAM_WS = "wss://api.deepgram.com/v1/listen"
_DEEPGRAM_REST = "https://api.deepgram.com/v1/listen"


class DeepgramStreamingSTT(BaseSTT):
    """Streaming Deepgram STT via the ``/v1/listen`` WebSocket."""

    supports_streaming = True

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-2",
        base_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not api_key:
            raise ValueError("DeepgramStreamingSTT requires api_key")
        self._api_key = api_key
        self._model = model or "nova-2"
        self._base_url = (base_url or "").strip() or None
        self._metadata = dict(metadata or {})

    def _ws_url(
        self,
        *,
        language: str | None,
        encoding: str,
        sample_rate: int | None,
    ) -> str:
        params: dict[str, Any] = {
            "model": self._model,
            "interim_results": "true",
            "smart_format": "true",
            "punctuate": "true",
        }
        if language and language != "auto":
            params["language"] = language
        elif self._metadata.get("language"):
            params["language"] = self._metadata["language"]
        # When the universal app's AudioWorklet pushes raw 16-bit PCM
        # (the live-streaming path), declare it explicitly so Deepgram
        # decodes per-frame instead of waiting for a container header.
        # That's what unlocks sub-1s TTFA: partials land in ~150 ms
        # because Deepgram never has to wait for an EBML cluster boundary.
        enc = (encoding or "").lower()
        if enc in ("pcm16", "pcm", "linear16"):
            params["encoding"] = "linear16"
            params["sample_rate"] = str(sample_rate or 16000)
            params["channels"] = "1"
        else:
            # Container path (webm/opus from MediaRecorder.stop()) —
            # Deepgram auto-detects when the bytes carry an EBML header.
            # The metadata override is for native clients that push raw
            # bytes in some other format (mulaw, etc).
            meta_enc = self._metadata.get("encoding")
            if meta_enc:
                params["encoding"] = meta_enc
            meta_sr = self._metadata.get("sample_rate") or sample_rate
            if meta_sr:
                params["sample_rate"] = str(meta_sr)
            meta_channels = self._metadata.get("channels")
            if meta_channels:
                params["channels"] = str(meta_channels)
        # Per-row passthrough for any deepgram knob we haven't normalised
        # explicitly (e.g. ``utterance_end_ms``, ``vad_events``, ``tier``).
        for k, v in (self._metadata.get("extra") or {}).items():
            params[k] = v if isinstance(v, str) else str(v)

        base = (self._base_url or _DEEPGRAM_WS).rstrip("/")
        if base.endswith("/v1/listen"):
            url_root = base
        else:
            url_root = f"{base}/v1/listen"
        return f"{url_root}?{urlencode(params)}"

    async def transcribe_file(
        self,
        path: str,
        *,
        language: str | None = None,
    ) -> str | None:
        """One-shot REST transcription for the bridges' ``/api/stt/transcribe``."""
        try:
            import httpx
        except ImportError:
            logger.warning(
                "httpx not installed — required for Deepgram REST transcribe"
            )
            return None

        params: dict[str, Any] = {
            "model": self._model,
            "smart_format": "true",
            "punctuate": "true",
        }
        if language and language != "auto":
            params["language"] = language
        elif self._metadata.get("language"):
            params["language"] = self._metadata["language"]

        rest_root = (self._base_url or _DEEPGRAM_REST).rstrip("/")
        if rest_root.endswith("/v1/listen"):
            rest_url = rest_root
        else:
            rest_url = f"{rest_root}/v1/listen"

        try:
            with open(path, "rb") as fh:
                payload = fh.read()
        except OSError as e:
            logger.warning("deepgram REST: read %s failed: %s", path, e)
            return None

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    rest_url,
                    params=params,
                    headers={
                        "Authorization": f"Token {self._api_key}",
                        "Content-Type": "audio/*",
                    },
                    content=payload,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("deepgram REST: request failed: %s", e)
                return None

        if resp.status_code != 200:
            logger.warning(
                "deepgram REST %s: %s",
                resp.status_code, resp.text[:200],
            )
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        try:
            return (
                data["results"]["channels"][0]["alternatives"][0]["transcript"]
                or None
            )
        except (KeyError, IndexError):
            return None

    async def stream(
        self,
        audio_in: AsyncIterator[bytes],
        *,
        language: str | None = None,
        encoding: str = "webm",
        sample_rate: int | None = None,
    ) -> AsyncIterator[STTEvent]:
        """Open the WS, fan out a writer + reader, yield STT events.

        Concurrent shape mirrors :func:`tts_streaming._elevenlabs_stream`:
        a writer task drains the inbound audio iterator into the WS, the
        outer loop reads JSON frames and yields :class:`STTEvent`. On
        cancel or early exit, the writer is cancelled and the WS closes
        cleanly so we don't leak a connection.
        """
        try:
            import websockets
        except ImportError:
            logger.warning("websockets package missing — Deepgram WS disabled")
            return

        url = self._ws_url(
            language=language, encoding=encoding, sample_rate=sample_rate,
        )
        elog("stt.deepgram_ws.connect", model=self._model)
        try:
            ws = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Token {self._api_key}"},
                max_size=None,
            )
        except Exception as e:  # noqa: BLE001
            elog(
                "stt.deepgram_ws.connect_failed",
                level="warning",
                model=self._model,
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
            return

        async def _writer() -> None:
            """Forward audio bytes from the inbound iterator to the WS.

            On EOF, send the ``CloseStream`` control message so Deepgram
            flushes the last partial as a final and closes the read side.
            """
            try:
                async for chunk in audio_in:
                    if chunk:
                        await ws.send(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug("deepgram writer drain error: %s", e)
            finally:
                try:
                    await ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass

        writer_task = asyncio.create_task(_writer())

        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    # Deepgram never sends binary frames on this endpoint.
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                # Deepgram message types: "Results" (transcript),
                # "UtteranceEnd", "SpeechStarted", "Metadata", "Error".
                msg_type = msg.get("type", "Results")
                if msg_type != "Results":
                    continue
                channel = msg.get("channel") or {}
                alts = channel.get("alternatives") or []
                if not alts:
                    continue
                top = alts[0]
                text = (top.get("transcript") or "").strip()
                if not text:
                    continue
                confidence = top.get("confidence")
                if msg.get("is_final"):
                    yield STTEvent(kind="final", text=text, confidence=confidence)
                else:
                    yield STTEvent(kind="partial", text=text, confidence=confidence)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            elog(
                "stt.deepgram_ws.read_error",
                level="warning",
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
        finally:
            elog("stt.deepgram_ws.done", model=self._model)
            if not writer_task.done():
                writer_task.cancel()
                try:
                    await writer_task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await ws.close()
            except Exception:
                pass


__all__ = ["DeepgramStreamingSTT"]
