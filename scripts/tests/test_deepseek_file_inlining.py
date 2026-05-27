"""DeepSeek file attachments get inlined as text — never sent as ``{type:"file"}``.

DeepSeek's chat completions API rejects ``{type:"file"}`` content parts
with ``Failed to deserialize the JSON body into the target type:
messages[N]: unknown variant 'file', expected 'text'``. Without this
shim, attaching a .json file to a session whose leader (or any member)
is DeepSeek crashes every turn until the file scrolls out of history.

Pins:
  - text-readable files (json, text/*) get their full content inlined
  - the inlined block reaches the parent ``_format_message`` so the
    serialized dict's ``content`` carries the file text (not a binary
    file part)
  - binary files (application/pdf) get an omitted placeholder, never
    a base64 blob in the prompt
  - the original ``Message.files`` / ``Message.content`` are restored
    after serialization so subsequent reads aren't affected
  - the wrapper is wired up in ``AGNO_PROVIDER_CLASSES`` so live runs
    actually use it
"""
from __future__ import annotations

import json

from ._framework import TestContext, test


def _make_file(*, content: bytes, mime: str, filename: str):
    """Build a minimal Agno ``File`` from in-memory bytes — keeps the
    test off-disk and independent of any tempfile cleanup."""
    from agno.media import File as _AgnoFile
    return _AgnoFile(
        content=content,
        mime_type=mime,
        filename=filename,
    )


def _make_image(*, content: bytes, mime: str = "image/png"):
    from agno.media import Image as _AgnoImage
    return _AgnoImage(content=content, mime_type=mime)


def _make_audio(*, content: bytes, fmt: str = "wav"):
    from agno.media import Audio as _AgnoAudio
    return _AgnoAudio(content=content, format=fmt, mime_type=f"audio/{fmt}")


def _make_message(*, content: str, files: list | None = None,
                  images: list | None = None, audios: list | None = None):
    from agno.models.message import Message
    return Message(role="user", content=content, files=files,
                   images=images, audio=audios)


def _format(message):
    """Run the patched DeepSeek's serializer against ``message`` and
    return the resulting OpenAI-wire dict."""
    from src.models._provider_overrides import DeepSeekTextOnly
    model = DeepSeekTextOnly(id="deepseek-chat", api_key="sk-test")
    return model._format_message(message)


def _content_text(serialized: dict) -> str:
    """Flatten the OpenAI-wire ``content`` field to a single string for
    substring assertions. Handles both plain-string and parts-list
    shapes — DeepSeek's parent serializer chooses between them based
    on whether ``files`` / ``images`` / ``audio`` is set."""
    content = serialized.get("content")
    if isinstance(content, list):
        return " ".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict)
        )
    return str(content or "")


def _assert_no_disallowed_parts(content) -> None:
    """Critical wire-level pin: DeepSeek rejects file/image_url/input_audio
    content parts. Whatever the serializer emitted, it MUST NOT contain any
    of those types."""
    if not isinstance(content, list):
        return
    for part in content:
        if isinstance(part, dict):
            ptype = part.get("type")
            assert ptype not in ("file", "image_url", "input_audio"), (
                f"disallowed content-part leaked into DeepSeek payload: {part}"
            )


@test("deepseek_file_inlining", "json file gets inlined as text content")
async def t_json_file_inlined(ctx: TestContext) -> None:
    payload = {"hello": "world", "count": 3}
    blob = json.dumps(payload).encode("utf-8")
    file = _make_file(content=blob, mime="application/json", filename="data.json")
    message = _make_message(content="What's in this file?", files=[file])

    serialized = _format(message)

    assert serialized.get("role") == "user", serialized
    _assert_no_disallowed_parts(serialized.get("content"))
    text_blob = _content_text(serialized)
    assert "data.json" in text_blob, f"filename missing from inlined content: {text_blob!r}"
    assert '"hello": "world"' in text_blob, \
        f"json body missing from inlined content: {text_blob!r}"
    assert "What's in this file?" in text_blob, \
        f"original user prompt dropped: {text_blob!r}"


@test("deepseek_file_inlining", "binary file becomes omitted placeholder, no base64 blob")
async def t_binary_file_placeholder(ctx: TestContext) -> None:
    # ``%PDF-1.4`` magic + a few non-text bytes — Agno's File doesn't
    # validate PDF structure, only that one of content/url/filepath is set.
    blob = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>endobj\n"
    file = _make_file(content=blob, mime="application/pdf", filename="report.pdf")
    message = _make_message(content="Summarize this PDF.", files=[file])

    serialized = _format(message)
    _assert_no_disallowed_parts(serialized.get("content"))
    text_blob = _content_text(serialized)

    assert "report.pdf" in text_blob, f"filename missing: {text_blob!r}"
    assert 'omitted="true"' in text_blob, \
        f"binary file should be marked omitted, got: {text_blob!r}"
    # Pin: the raw PDF bytes / base64 must not appear inline — DeepSeek
    # would treat them as garbage tokens and burn context.
    assert "%PDF" not in text_blob, f"binary bytes leaked into prompt: {text_blob!r}"


@test("deepseek_file_inlining", "image attachment stripped → media-omitted placeholder")
async def t_image_stripped(ctx: TestContext) -> None:
    # Even tiny ``content=b'\\x89PNG...'`` triggers the parent path; we just
    # need an Agno Image instance — its serializer would normally emit
    # an image_url content part DeepSeek can't parse.
    img = _make_image(content=b"\x89PNG\r\n\x1a\nfake-image-bytes", mime="image/png")
    message = _make_message(content="What's in this picture?", images=[img])

    serialized = _format(message)
    _assert_no_disallowed_parts(serialized.get("content"))
    text_blob = _content_text(serialized)

    assert "media-omitted" in text_blob, \
        f"image should be stripped to media-omitted block, got: {text_blob!r}"
    assert "image" in text_blob.lower(), f"image label missing: {text_blob!r}"
    # Pin: no base64 / image_url string leaks
    assert "base64" not in text_blob.lower(), f"base64 leaked: {text_blob!r}"
    assert "image_url" not in text_blob, f"image_url leaked: {text_blob!r}"
    assert "What's in this picture?" in text_blob, \
        f"original prompt dropped: {text_blob!r}"


@test("deepseek_file_inlining", "audio attachment stripped → media-omitted placeholder")
async def t_audio_stripped(ctx: TestContext) -> None:
    au = _make_audio(content=b"RIFF\x00\x00\x00\x00WAVEfmt ", fmt="wav")
    message = _make_message(content="Transcribe this.", audios=[au])

    serialized = _format(message)
    _assert_no_disallowed_parts(serialized.get("content"))
    text_blob = _content_text(serialized)

    assert "media-omitted" in text_blob, \
        f"audio should be stripped to media-omitted block, got: {text_blob!r}"
    assert "audio" in text_blob.lower(), f"audio label missing: {text_blob!r}"


@test("deepseek_file_inlining", "mixed files + images coexist in one prefix")
async def t_mixed_attachments(ctx: TestContext) -> None:
    """Pin the rehydration case where a session accumulated a JSON file
    AND a screenshot (from different turns / tools). Both must reach
    DeepSeek's payload safely on a single serialization pass."""
    file = _make_file(
        content=b'{"a":1}', mime="application/json", filename="data.json",
    )
    img = _make_image(content=b"fake", mime="image/jpeg")
    message = _make_message(
        content="Look at both attachments.", files=[file], images=[img],
    )

    serialized = _format(message)
    _assert_no_disallowed_parts(serialized.get("content"))
    text_blob = _content_text(serialized)

    assert "data.json" in text_blob, f"file missing: {text_blob!r}"
    assert '"a":1' in text_blob, f"file content missing: {text_blob!r}"
    assert "media-omitted" in text_blob, f"image placeholder missing: {text_blob!r}"
    assert "Look at both attachments." in text_blob, \
        f"original prompt dropped: {text_blob!r}"


@test("deepseek_file_inlining", "original message state restored across files/images/audio")
async def t_message_state_restored(ctx: TestContext) -> None:
    """Critical for Agno's team mode: a member might serialize the same
    in-memory Message instance multiple times. Mutating any of files /
    images / audio / content to ``None`` permanently would silently
    drop attachments on the second pass."""
    file = _make_file(content=b'{"k":"v"}', mime="application/json", filename="x.json")
    img = _make_image(content=b"img", mime="image/png")
    au = _make_audio(content=b"aud", fmt="mp3")
    message = _make_message(content="hi", files=[file], images=[img], audios=[au])

    original = (message.files, message.images, message.audio, message.content)

    _format(message)

    assert message.files is original[0], "message.files was not restored"
    assert message.images is original[1], "message.images was not restored"
    assert message.audio is original[2], "message.audio was not restored"
    assert message.content == original[3], \
        f"message.content mutated: {message.content!r} != {original[3]!r}"


@test("deepseek_file_inlining", "no attachments → fall through to parent serializer unchanged")
async def t_no_attachments_passthrough(ctx: TestContext) -> None:
    # files=None (not an empty list) — the parent serializer reshapes
    # content into a parts-array whenever any of files/images/audio is set
    # (even ``[]``), so this case pins the truly-no-attachments fall-through.
    from agno.models.message import Message
    message = Message(role="user", content="plain text turn")

    serialized = _format(message)

    assert serialized.get("content") == "plain text turn", serialized


@test("deepseek_file_inlining", "AGNO_PROVIDER_CLASSES wires the patched class for deepseek")
async def t_provider_wired(ctx: TestContext) -> None:
    """Without this wiring the fix would compile but not run in production."""
    from src.models.agno_provider import AGNO_PROVIDER_CLASSES

    entry = AGNO_PROVIDER_CLASSES.get("deepseek")
    assert entry is not None, "deepseek entry missing from AGNO_PROVIDER_CLASSES"
    module_name, class_name, _kwargs = entry
    assert class_name == "DeepSeekTextOnly", \
        f"deepseek entry not pointed at the text-only subclass: {entry}"
    # Import path must resolve — catches typos before they hit a live turn.
    import importlib
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    assert cls is not None, f"{class_name} not importable from {module_name}"
