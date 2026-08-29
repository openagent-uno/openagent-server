"""Model media-capability routing, metadata hydration, and extraction tests."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from ._framework import TestContext, test


def _providers(*, include_image: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{
        "id": 1,
        "name": "alpha",
        "framework": "api-based",
        "enabled": True,
        "models": [{
            "id": 1,
            "model": "text-only",
            "enabled": True,
            "is_classifier": True,
            "metadata": {"input_modalities": ["text"]},
        }],
    }]
    if include_image:
        rows.append({
            "id": 2,
            "name": "beta",
            "framework": "api-based",
            "enabled": True,
            "models": [{
                "id": 2,
                "model": "vision",
                "enabled": True,
                "metadata": {"input_modalities": ["text", "image"]},
            }],
        })
    return rows


class _CaptureModel:
    def __init__(self, runtime_id: str):
        self.runtime_id = runtime_id
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def generate(self, messages, **kwargs):
        from src.models.base import ModelResponse

        self.generate_calls.append({"messages": messages, **kwargs})
        return ModelResponse(content=f"handled:{self.runtime_id}", model=self.runtime_id)

    async def stream(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, **kwargs})
        yield f"streamed:{self.runtime_id}"


def _dispatcher(config: list[dict[str, Any]]):
    from src.models.dispatcher import ModelDispatcher

    dispatcher = ModelDispatcher(config)
    captures = {
        "alpha:text-only": _CaptureModel("alpha:text-only"),
        "beta:vision": _CaptureModel("beta:vision"),
    }
    dispatcher._get_team_provider = lambda runtime_id: captures[runtime_id]
    return dispatcher, captures


def _pdf_with_text(text: str) -> bytes:
    """Create a tiny, deterministic PDF whose content pypdf can extract."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
    })
    stream = DecodedStreamObject()
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 200 Td ({safe}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@test("model_media_capabilities", "metadata persists, hydrates, and legacy rows backfill")
async def t_metadata_roundtrip_and_backfill(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.models.catalog import iter_configured_models

    with tempfile.TemporaryDirectory(prefix="oa-media-meta-") as tmp:
        path = Path(tmp) / "agent.db"
        db = MemoryDB(str(path))
        await db.connect()
        pid = await db.upsert_provider(
            name="custom-local", framework="api-based", api_key="test",
        )
        mid = await db.upsert_model(provider_id=pid, model="unknown-llm")
        row = await db.get_model(mid)
        assert row is not None
        assert row["metadata"]["input_modalities"] == ["text"], row

        configured = await db.materialise_providers_config(enabled_only=True)
        entries = iter_configured_models(configured)
        assert entries[0].metadata["input_modalities"] == ["text"], entries[0]

        # Simulate a pre-feature row and reconnect: migration persists the
        # conservative declaration instead of relying on an in-memory guess.
        conn = await db._ensure_connected()
        await conn.execute("UPDATE models SET metadata_json='{}' WHERE id=?", (mid,))
        await conn.commit()
        await db.close()

        reopened = MemoryDB(str(path))
        await reopened.connect()
        try:
            backfilled = await reopened.get_model(mid)
            assert backfilled["metadata"]["input_modalities"] == ["text"], backfilled
            await reopened.upsert_model(
                provider_id=pid,
                model="unknown-llm",
                metadata={"input_modalities": ["text", "image"]},
            )
            explicit = await reopened.get_model(mid)
            assert explicit["metadata"]["input_modalities"] == ["text", "image"]
        finally:
            await reopened.close()


@test("model_media_capabilities", "generate falls back from pinned text model to vision model")
async def t_generate_media_fallback(_ctx: TestContext) -> None:
    from src.stream.media import Image

    dispatcher, captures = _dispatcher(_providers())

    class _PinnedDB:
        async def get_session_pin(self, session_id):
            assert session_id == "media-fallback-generate"
            return "alpha:text-only"

    # Exercise the durable-pin branch, not merely the default-leader branch.
    # A media fallback is per-turn and must not mutate that pin.
    dispatcher._db = _PinnedDB()
    image = Image(content=b"not-a-real-png", mime_type="image/png")
    response = await dispatcher.generate(
        [{"role": "user", "content": "inspect"}],
        session_id="media-fallback-generate",
        images=[image],
    )
    assert response.content == "handled:beta:vision", response
    assert captures["alpha:text-only"].generate_calls == []
    call = captures["beta:vision"].generate_calls[0]
    assert call["images"] == [image]
    assert dispatcher.effective_model_id("media-fallback-generate") == "beta:vision"


@test("model_media_capabilities", "stream uses the same compatible-model fallback")
async def t_stream_media_fallback(_ctx: TestContext) -> None:
    from src.stream.media import Image

    dispatcher, captures = _dispatcher(_providers())
    image = Image(content=b"image", mime_type="image/png")
    chunks = [
        chunk async for chunk in dispatcher.stream(
            [{"role": "user", "content": "inspect"}],
            session_id="media-fallback-stream",
            images=[image],
        )
    ]
    assert chunks == ["streamed:beta:vision"], chunks
    assert captures["alpha:text-only"].stream_calls == []
    assert captures["beta:vision"].stream_calls[0]["images"] == [image]


@test("model_media_capabilities", "incompatible binary modality returns an explicit error")
async def t_explicit_incompatible_error(_ctx: TestContext) -> None:
    from src.core.agent import take_run_failure
    from src.stream.media import Audio

    dispatcher, captures = _dispatcher(_providers(include_image=False))
    audio = Audio(content=b"RIFF", mime_type="audio/wav", format="wav")
    response = await dispatcher.generate(
        [{"role": "user", "content": "transcribe"}],
        session_id="media-no-compatible",
        audio=[audio],
    )
    assert response.stop_reason == "error", response
    assert "No enabled configured model" in response.content, response.content
    assert "audio" in response.content, response.content
    assert captures["alpha:text-only"].generate_calls == []
    assert take_run_failure() == "router: media_incompatible"


@test("model_media_capabilities", "non-extractable Telegram-style PDF fails explicitly")
async def t_explicit_incompatible_file_error(_ctx: TestContext) -> None:
    from src.core.agent import take_run_failure
    from src.stream.media import File
    from pypdf import PdfWriter

    dispatcher, captures = _dispatcher(_providers(include_image=False))
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(output)
    binary = File(
        content=output.getvalue(),
        mime_type="application/pdf",
        filename="scan.pdf",
    )
    response = await dispatcher.generate(
        [{"role": "user", "content": "inspect the attachment"}],
        session_id="media-file-no-compatible",
        files=[binary],
    )
    assert response.stop_reason == "error", response
    assert "Cannot process the attached media" in response.content, response.content
    assert "No enabled configured model" in response.content, response.content
    assert "scan.pdf" in response.content, response.content
    assert captures["alpha:text-only"].generate_calls == []
    assert take_run_failure() == "router: media_incompatible"


@test("model_media_capabilities", "text file is bounded-extracted into text-only model input")
async def t_text_extraction_reaches_model(_ctx: TestContext) -> None:
    from src.stream.media import File

    dispatcher, captures = _dispatcher(_providers(include_image=False))
    file = File(
        content=b'{"needle":"orchid-text-content"}',
        mime_type="application/json",
        filename="payload.json",
    )
    response = await dispatcher.generate(
        [{"role": "user", "content": "read it"}],
        session_id="media-text-extraction",
        files=[file],
    )
    assert response.content == "handled:alpha:text-only"
    call = captures["alpha:text-only"].generate_calls[0]
    assert call["files"] is None
    prompt = str(call["messages"][-1]["content"])
    assert "orchid-text-content" in prompt, prompt
    assert 'filename="payload.json"' in prompt, prompt


@test("model_media_capabilities", "PDF extraction reaches generate and stream model input")
async def t_pdf_extraction_reaches_both_paths(_ctx: TestContext) -> None:
    from src.stream.media import File

    dispatcher, captures = _dispatcher(_providers(include_image=False))
    pdf = File(
        content=_pdf_with_text("orchid-pdf-content"),
        mime_type="application/pdf",
        filename="report.pdf",
    )
    await dispatcher.generate(
        [{"role": "user", "content": "summarize"}],
        session_id="media-pdf-generate",
        files=[pdf],
    )
    gen_call = captures["alpha:text-only"].generate_calls[-1]
    assert gen_call["files"] is None
    assert "orchid-pdf-content" in str(gen_call["messages"][-1]["content"])

    chunks = [
        chunk async for chunk in dispatcher.stream(
            [{"role": "user", "content": "summarize"}],
            session_id="media-pdf-stream",
            files=[pdf],
        )
    ]
    assert chunks == ["streamed:alpha:text-only"]
    stream_call = captures["alpha:text-only"].stream_calls[-1]
    assert stream_call["files"] is None
    assert "orchid-pdf-content" in str(stream_call["messages"][-1]["content"])


@test("model_media_capabilities", "media Team catalog excludes incompatible members")
async def t_team_member_filter(_ctx: TestContext) -> None:
    from src.models.dispatcher import TeamRouterProvider

    provider = TeamRouterProvider(
        entry_runtime_id="beta:vision", providers_config=_providers(),
    )
    ids = {
        entry.runtime_id
        for entry in provider._enabled_llm_models(frozenset({"image"}))
    }
    assert ids == {"beta:vision"}, ids


@test("model_media_capabilities", "model-pinned Team lane also falls back for media")
async def t_pinned_team_media_fallback(_ctx: TestContext) -> None:
    from src.models.base import BaseModel, ModelResponse
    from src.models.dispatcher import TeamRouterProvider
    from src.stream.media import Image

    class _Runtime(BaseModel):
        def __init__(self):
            self.generate_calls: list[dict[str, Any]] = []
            self.stream_calls: list[dict[str, Any]] = []

        async def generate(self, messages, **kwargs):
            self.generate_calls.append({"messages": messages, **kwargs})
            return ModelResponse(content="pinned-fallback", model="beta:vision")

        async def stream(self, messages, **kwargs):
            self.stream_calls.append({"messages": messages, **kwargs})
            yield "pinned-stream-fallback"

    provider = TeamRouterProvider(
        entry_runtime_id="alpha:text-only", providers_config=_providers(),
    )
    runtime = _Runtime()
    requirements: list[frozenset[str]] = []

    def _ensure(sid, _system, required_modalities=frozenset()):
        requirements.append(required_modalities)
        provider._session_runtime_entry[sid] = (
            "beta:vision" if "image" in required_modalities else "alpha:text-only"
        )
        return runtime

    provider._ensure_runtime = _ensure
    image = Image(content=b"image", mime_type="image/png")
    response = await provider.generate(
        [{"role": "user", "content": "inspect"}],
        session_id="pinned-media-generate",
        images=[image],
    )
    assert response.content == "pinned-fallback"
    assert runtime.generate_calls[0]["images"] == [image]
    assert provider.effective_model_id("pinned-media-generate") == "beta:vision"

    chunks = [
        chunk async for chunk in provider.stream(
            [{"role": "user", "content": "inspect"}],
            session_id="pinned-media-stream",
            images=[image],
        )
    ]
    assert chunks == ["pinned-stream-fallback"]
    assert runtime.stream_calls[0]["images"] == [image]
    assert requirements == [frozenset({"image"}), frozenset({"image"})]


@test("model_media_capabilities", "typed runtime media becomes bounded attachment carriers")
async def t_typed_runtime_output_carriers(_ctx: TestContext) -> None:
    import os
    import stat

    import src.models.native_provider as native_provider
    from src.core._run_state.agent import ToolCallCompletedEvent
    from src.stream.media import Audio, File, Image, Video

    media = SimpleNamespace(
        images=[Image(id="image-one", content=b"image-bytes", mime_type="image/png")],
        audio=[Audio(id="audio-one", content=b"audio-bytes", mime_type="audio/ogg")],
        videos=[Video(id="video-one", content=b"video-bytes", mime_type="video/mp4")],
        files=[File(
            id="file-one", content=b"%PDF-output", mime_type="application/pdf",
            filename="report.pdf",
        )],
    )
    with tempfile.TemporaryDirectory(prefix="oa-output-media-") as raw:
        with patch.object(native_provider, "_agno_image_tmpdir", return_value=raw):
            emitted: set[tuple[str, str]] = set()
            markers = native_provider._output_media_markers(media, emitted=emitted)
            assert native_provider._output_media_markers(media, emitted=emitted) == []

            # URL-only media is an untrusted MCP/provider reference, not an
            # instruction for the server to fetch arbitrary addresses. This
            # rejects public URLs, private IPs, localhost, and redirectors at
            # the same boundary, before any HTTP client is called.
            for url in (
                "https://example.test/image.png",
                "https://127.0.0.1/private.png",
                "https://localhost/private.png",
                "https://example.test/redirect-to-metadata",
            ):
                assert native_provider._output_media_bytes(
                    SimpleNamespace(url=url),
                ) is None
            unsafe = SimpleNamespace(
                images=[SimpleNamespace(id="remote-only", url="https://example.test/x")],
            )
            unsafe_seen: set[tuple[str, str]] = set()
            unsafe_result = native_provider._output_media_markers(
                unsafe, emitted=unsafe_seen,
            )
            assert unsafe_result == [
                "\n(Output image could not be saved as an attachment.)\n"
            ]
            assert native_provider._output_media_markers(
                unsafe, emitted=unsafe_seen,
            ) == []

            # Tool-controlled names cannot exceed the filesystem component
            # limit and staging bytes are never group/world-readable.
            long_media = SimpleNamespace(
                id="long-name", filename=("x" * 1_000) + ".pdf",
                content=b"private", mime_type="application/pdf", format="pdf",
            )
            long_marker = native_provider._save_agno_output_media(long_media, "file")
            assert long_marker is not None
            long_path = Path(long_marker.strip()[1:-1].split(":", 1)[1])
            assert len(long_path.name.encode("utf-8")) <= os.pathconf(raw, "PC_NAME_MAX")
            assert stat.S_IMODE(long_path.stat().st_mode) == 0o600

        kinds: dict[str, bytes] = {}
        for marker in markers:
            kind, path = marker.strip()[1:-1].split(":", 1)
            kinds[kind] = Path(path).read_bytes()
            assert stat.S_IMODE(Path(path).stat().st_mode) == 0o600
        assert kinds == {
            "IMAGE": b"image-bytes",
            "VOICE": b"audio-bytes",
            "VIDEO": b"video-bytes",
            "FILE": b"%PDF-output",
        }, kinds

    # Event persistence is part of the typed boundary: File used to be lost
    # from BaseRunOutputEvent.to_dict/from_dict while other media survived.
    text_file = File(
        id="text-file", content=b"file", mime_type="text/plain",
        filename="note.txt",
    )
    event = ToolCallCompletedEvent(files=[media.files[0], text_file])
    restored = ToolCallCompletedEvent.from_dict(event.to_dict())
    assert restored.files and restored.files[0].get_content_bytes() == b"%PDF-output"
    assert restored.files[1].get_content_bytes() == b"file"


@test("model_media_capabilities", "MCP typed media and resources remain structured")
async def t_mcp_typed_media_resources(_ctx: TestContext) -> None:
    import base64

    from mcp.types import (
        AudioContent,
        BlobResourceContents,
        CallToolResult,
        EmbeddedResource,
        ImageContent,
        TextContent,
        TextResourceContents,
        Tool,
    )
    from src.core._runner.utils.mcp import get_entrypoint_for_tool

    def b64(value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")

    class _Session:
        async def send_ping(self):
            return None

        async def call_tool(self, _name, _args, meta=None):
            return CallToolResult(content=[
                TextContent(type="text", text="tool summary"),
                ImageContent(type="image", data=b64(b"image"), mimeType="image/png"),
                AudioContent(type="audio", data=b64(b"audio"), mimeType="audio/ogg"),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="file:///output/movie.mp4", mimeType="video/mp4",
                        blob=b64(b"video"),
                    ),
                ),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="file:///output/%2Funsafe%2Freport.pdf", mimeType="application/pdf",
                        blob=b64(b"pdf"),
                    ),
                ),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="file:///output/note.txt", mimeType="text/plain",
                        text="embedded note",
                    ),
                ),
            ])

    tool = Tool(name="media", inputSchema={"type": "object", "properties": {}})
    result = await get_entrypoint_for_tool(tool, _Session())()
    assert result.images and result.images[0].get_content_bytes() == b"image"
    assert result.audios and result.audios[0].get_content_bytes() == b"audio"
    assert result.videos and result.videos[0].get_content_bytes() == b"video"
    assert result.files and result.files[0].get_content_bytes() == b"pdf"
    assert result.files[0].filename == "report.pdf"
    assert "tool summary" in result.content
    assert "embedded note" in result.content


@test("model_media_capabilities", "MCP payload limits reject malformed media and bound text")
async def t_mcp_payload_limits(_ctx: TestContext) -> None:
    import base64

    from mcp.types import (
        AudioContent,
        BlobResourceContents,
        CallToolResult,
        EmbeddedResource,
        ImageContent,
        TextResourceContents,
        Tool,
    )
    import src.core._runner.utils.mcp as mcp_utils

    class _Session:
        async def send_ping(self):
            return None

        async def call_tool(self, _name, _args, meta=None):
            return CallToolResult(content=[
                ImageContent(
                    type="image",
                    data=base64.b64encode(b"123").decode("ascii"),
                    mimeType="image/png",
                ),
                AudioContent(
                    type="audio",
                    data=base64.b64encode(b"456").decode("ascii"),
                    mimeType="audio/ogg",
                ),
                ImageContent(type="image", data="%%%malformed", mimeType="image/png"),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="file:///too-big.bin",
                        mimeType="application/octet-stream",
                        blob=base64.b64encode(b"abcdef").decode("ascii"),
                    ),
                ),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="file:///long.txt", mimeType="text/plain",
                        text="0123456789ABCDEFGHIJ",
                    ),
                ),
            ])

    tool = Tool(name="bounded", inputSchema={"type": "object", "properties": {}})
    with (
        patch.object(mcp_utils, "_MCP_MEDIA_MAX_BYTES", 4),
        patch.object(mcp_utils, "_MCP_MEDIA_MAX_TOTAL_BYTES", 4),
    ):
        result = await mcp_utils.get_entrypoint_for_tool(tool, _Session())()

    assert result.images and result.images[0].get_content_bytes() == b"123"
    assert not result.audios and not result.files
    assert "Rejected MCP image" in result.content
    assert "Rejected MCP audio" in result.content
    assert "Rejected MCP embedded resource" in result.content

    class _TextSession:
        async def send_ping(self):
            return None

        async def call_tool(self, _name, _args, meta=None):
            return CallToolResult(content=[EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///long.txt", mimeType="text/plain",
                    text="0123456789ABCDEFGHIJ",
                ),
            )])

    with (
        patch.object(mcp_utils, "_MCP_TEXT_MAX_CHARS", 10),
        patch.object(mcp_utils, "_MCP_TEXT_MAX_BYTES", 10),
    ):
        text_result = await mcp_utils.get_entrypoint_for_tool(tool, _TextSession())()

    assert "0123456789" in text_result.content
    assert "truncated by OpenAgent" in text_result.content


@test("model_media_capabilities", "Team collect and stream propagate typed output media")
async def t_team_runtime_output_media_both_paths(_ctx: TestContext) -> None:
    import src.models.native_provider as native_provider
    from src.core._run_state.agent import RunCompletedEvent
    from src.models.dispatcher import _arun_runtime_collect, _arun_runtime_stream
    from src.stream.media import Audio, File, Video

    def _outputs():
        return {
            "audio": [Audio(
                id="team-audio", content=b"team-audio", mime_type="audio/ogg",
            )],
            "videos": [Video(
                id="team-video", content=b"team-video", mime_type="video/mp4",
            )],
            "files": [File(
                id="team-file", content=b"team-file", mime_type="application/pdf",
                filename="team.pdf",
            )],
        }

    class _Runtime:
        def arun(self, _prompt, **kwargs):
            if kwargs.get("stream"):
                async def _events():
                    yield RunCompletedEvent(
                        session_id=kwargs["session_id"],
                        content="stream-complete",
                        **_outputs(),
                    )
                return _events()

            async def _result():
                return SimpleNamespace(
                    content="collect-complete",
                    tools=[],
                    messages=[],
                    metrics=SimpleNamespace(input_tokens=0, output_tokens=0),
                    **_outputs(),
                )
            return _result()

    with tempfile.TemporaryDirectory(prefix="oa-team-output-") as raw:
        with patch.object(native_provider, "_agno_image_tmpdir", return_value=raw):
            collected = await _arun_runtime_collect(
                _Runtime(),
                prompt="probe",
                session_id="typed-output-collect",
                user_id="user",
                error_event="test.collect.error",
                entry_runtime_id="alpha:text-only",
            )
            assert collected.content.startswith("collect-complete")
            assert "[VOICE:" in collected.content
            assert "[VIDEO:" in collected.content
            assert "[FILE:" in collected.content

            chunks = [
                chunk async for chunk in _arun_runtime_stream(
                    _Runtime(),
                    prompt="probe",
                    session_id="typed-output-stream",
                    user_id="user",
                    on_status=None,
                    error_event="test.stream.error",
                )
            ]
            assert chunks[0] == "stream-complete", chunks
            joined = "".join(chunks[1:])
            assert "[VOICE:" in joined
            assert "[VIDEO:" in joined
            assert "[FILE:" in joined
