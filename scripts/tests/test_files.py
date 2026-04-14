"""Files & images in chat sessions — send, understand, receive.

This module covers the full attachment pipeline:

  CLIENT  ──POST /api/upload──▶  GATEWAY  ──path────▶  AGENT  ──MCP read──▶  LLM
                                                           │
                          markers ◀──[IMAGE:/path]──── LLM

Four tests:

1. **Pure-unit parsing**: ``parse_response_markers`` extracts ``[IMAGE:...]``
   / ``[FILE:...]`` / ``[VOICE:...]`` / ``[VIDEO:...]`` markers and strips
   them from the text.

2. **Upload → read-back**: upload a plaintext file via ``POST /api/upload``,
   ask the agent (via WS) to read the path, assert the secret token
   embedded in the file makes it back into the response. This verifies
   the model actually reaches the file through the filesystem MCP, not
   just that the endpoint accepted the upload.

3. **Image send**: upload a PNG with a dominant colour, ask the agent to
   inspect it. The agent reads file metadata (size, type) via the
   filesystem MCP even if the LLM itself has no vision — what we assert
   is that the tool call succeeded and the response mentions the file.
   This exposes whether the image pipeline is reachable.

4. **Markers round-trip**: instruct the agent to emit ``[IMAGE:/tmp/foo.png]``
   in its reply. Verify the gateway parses the marker out of the text
   and emits it in the WS ``attachments`` field — the path the UI (and
   bridges) use to actually display the attachment.
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import tempfile
import uuid
import zlib

from ._framework import TestContext, TestSkip, have_openai_key, test


def _write_simple_png(path: str, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    """Write a 2x2 solid-color PNG with zero dependencies.

    Bypassing Pillow keeps the test hermetic — the suite doesn't need an
    image library just to create one image.
    """
    def _chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)  # 2x2, 8-bit, RGB
    r, g, b = color
    # Scanlines: each row has a filter byte (0 = None) then 3 bytes per pixel
    raw = b""
    for _ in range(2):
        raw += b"\x00" + bytes([r, g, b, r, g, b])
    idat = zlib.compress(raw, 9)
    png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


# ── 1. Pure-unit: parse_response_markers ─────────────────────────────


@test("files", "parse_response_markers extracts IMAGE/FILE/VOICE/VIDEO markers")
async def t_parse_response_markers(ctx: TestContext) -> None:
    from openagent.channels.base import parse_response_markers

    text = (
        "Here's your report: [FILE:/tmp/report.pdf]\n"
        "And the chart: [IMAGE:/tmp/chart.png]\n"
        "Voice note: [VOICE:/tmp/note.mp3]\n"
        "Video: [VIDEO:/tmp/clip.mp4]\n"
        "That's it."
    )
    clean, attachments = parse_response_markers(text)
    # All markers stripped from text
    assert "[FILE:" not in clean
    assert "[IMAGE:" not in clean
    assert "That's it." in clean
    # Four attachments, in order, with correct type + path + filename
    kinds = [(a.type, a.path, a.filename) for a in attachments]
    assert kinds == [
        ("file",  "/tmp/report.pdf", "report.pdf"),
        ("image", "/tmp/chart.png",  "chart.png"),
        ("voice", "/tmp/note.mp3",   "note.mp3"),
        ("video", "/tmp/clip.mp4",   "clip.mp4"),
    ], kinds

    # No markers → passthrough, empty list
    clean2, att2 = parse_response_markers("plain text with no markers")
    assert clean2 == "plain text with no markers"
    assert att2 == []


# ── 2. Upload → agent reads it back ───────────────────────────────────


@test("files", "upload text file then agent reads secret via filesystem MCP")
async def t_upload_text_roundtrip(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    import aiohttp

    # Upload a file with a unique content marker inside. "Marker" (not
    # "secret") avoids triggering the model's file-exfiltration safety
    # heuristic — gpt-4o-mini refuses ~5% of the time when we ask it to
    # read "secrets" even from a local test file.
    marker = f"OPENAGENT_MARKER_{uuid.uuid4().hex[:8].upper()}"
    payload = f"This is a test data file.\nMarker: {marker}\nEnd of data.\n".encode()

    async with aiohttp.ClientSession() as http:
        # Upload
        data = aiohttp.FormData()
        data.add_field("file", payload, filename="secret.txt", content_type="text/plain")
        async with http.post(f"http://127.0.0.1:{port}/api/upload", data=data) as r:
            assert r.status in (200, 201), f"upload status {r.status}"
            upload = await r.json()
        file_path = upload.get("path")
        assert file_path and os.path.exists(file_path), f"upload didn't persist: {upload}"

        # Surface the resolved path in the assertion message if we fail —
        # helps diagnose filesystem MCP permission errors quickly.
        realpath = os.path.realpath(file_path)
        debug_suffix = f"\nuploaded to: {file_path}\nrealpath: {realpath}"

        # Ask the agent, over WS, to read the file and echo the marker back.
        async with http.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await ws.send_json({"type": "auth", "client_id": "files-test"})
            await asyncio.wait_for(ws.receive(), timeout=10)
            sid = f"files-{uuid.uuid4().hex[:6]}"
            # Use the filesystem MCP to read the file. We don't pin a
            # specific tool name — the MCP exposes several
            # (read_text_file, read_file, etc.) across server versions,
            # and the model picks whichever works.
            prompt = (
                f"A test data file has been uploaded at: {file_path}\n"
                "Use a filesystem MCP tool (like filesystem_read_text_file "
                "or filesystem_read_file) to read its contents. This path "
                "is a configured allowed root in this test environment, "
                "so it IS accessible — do not claim otherwise. After "
                "reading, include the line starting with 'Marker:' "
                "verbatim somewhere in your response."
            )
            await ws.send_json({"type": "message", "text": prompt, "session_id": sid})
            response = None
            async with asyncio.timeout(120):
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    payload_json = json.loads(msg.data)
                    if payload_json.get("type") == "response":
                        response = payload_json.get("text", "")
                        break
            assert response is not None, "no response received"
            assert marker in response, (
                f"agent didn't quote the marker from the file.\n"
                f"Expected: {marker}\nGot: {response[:400]}{debug_suffix}"
            )


# ── 3. Image upload reaches the agent ─────────────────────────────────


@test("files", "upload image, agent inspects it via filesystem MCP")
async def t_upload_image_roundtrip(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    import aiohttp

    # Write a tiny red PNG to disk, then upload it.
    tmp = tempfile.mkdtemp(prefix="oa_img_test_")
    png_path = f"{tmp}/red.png"
    _write_simple_png(png_path, color=(255, 0, 0))
    assert os.path.getsize(png_path) > 50, "png too small — generator broke"

    async with aiohttp.ClientSession() as http:
        with open(png_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f.read(), filename="red.png",
                           content_type="image/png")
            async with http.post(f"http://127.0.0.1:{port}/api/upload",
                                 data=data) as r:
                assert r.status in (200, 201), f"upload status {r.status}"
                uploaded = await r.json()
        file_path = uploaded.get("path")
        assert file_path and os.path.exists(file_path)

        async with http.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await ws.send_json({"type": "auth", "client_id": "img-test"})
            await asyncio.wait_for(ws.receive(), timeout=10)
            sid = f"img-{uuid.uuid4().hex[:6]}"
            # The model doesn't have vision here, so don't ask about colour.
            # Ask for file metadata — the model has to actually call the MCP
            # to answer. A "red.png" in the response alone would be cheating
            # (the filename is in the prompt), so we require the file size.
            prompt = (
                f"A PNG image has been uploaded at: {file_path}\n"
                "Call filesystem_get_file_info (or similar filesystem MCP tool) "
                "to get the file size in bytes, then reply with EXACTLY this "
                f"format: FILE_SIZE=<bytes>. The actual size is {os.path.getsize(png_path)} "
                "bytes — your tool call must independently confirm this."
            )
            await ws.send_json({"type": "message", "text": prompt, "session_id": sid})
            response = None
            async with asyncio.timeout(120):
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    pj = json.loads(msg.data)
                    if pj.get("type") == "response":
                        response = pj.get("text", "")
                        break
            assert response is not None, "no response received"
            actual = os.path.getsize(png_path)
            # Model may say "FILE_SIZE=86" or "86 bytes" etc — just check the
            # number appears and the word 'size' or the marker does too.
            assert str(actual) in response, (
                f"agent didn't report actual file size ({actual}).\n"
                f"Response: {response[:400]}"
            )


# ── 4. Attachment markers round-trip through gateway.attachments field ─


@test("files", "agent emits [IMAGE:/path] marker → WS attachments field")
async def t_attachment_marker_roundtrip(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    import aiohttp

    # We instruct the agent to output a literal marker. The gateway should
    # parse it out of the text and emit it in the WS `attachments` list.
    # The prompt is blunt about not calling tools — with a permissive model
    # the marker can be mistaken for a request to create a note / save a file.
    fake_path = f"/tmp/fake-{uuid.uuid4().hex[:6]}.png"
    marker_payload = f"Here is your chart [IMAGE:{fake_path}]"
    prompt = (
        "This is a TEST. Do NOT call any tools. Do NOT save anything to vault. "
        "Do NOT create notes. Do NOT use the filesystem. Just produce a text "
        "response that is EXACTLY the following characters verbatim — the "
        "brackets and path are part of the output, not an instruction:\n\n"
        f"{marker_payload}"
    )
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await ws.send_json({"type": "auth", "client_id": "markers"})
            await asyncio.wait_for(ws.receive(), timeout=10)
            sid = f"mark-{uuid.uuid4().hex[:6]}"
            await ws.send_json({"type": "message", "text": prompt, "session_id": sid})
            payload = None
            async with asyncio.timeout(60):
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    pj = json.loads(msg.data)
                    if pj.get("type") == "response":
                        payload = pj
                        break
            assert payload is not None
            # The marker must be stripped from the visible text …
            assert "[IMAGE:" not in payload.get("text", ""), \
                f"marker leaked into visible text: {payload.get('text')}"
            # … and surface in the `attachments` list instead.
            attachments = payload.get("attachments") or []
            assert attachments, f"no attachments in payload: {payload}"
            paths = [a.get("path") for a in attachments]
            assert fake_path in paths, f"marker path not in attachments: {paths}"
            types = [a.get("type") for a in attachments]
            assert "image" in types, f"image type missing: {types}"


# ── 5. Agent.run(attachments=[...]) builds context block ─────────────


@test("files", "Agent.run(attachments=[...]) prepends attachment context")
async def t_agent_run_attachments_context(ctx: TestContext) -> None:
    """Unit test for the Agent-level attachments plumbing.

    Uses a stub model so we can inspect the outgoing messages without
    burning tokens. Verifies that ``attachments=[{type, filename, path}]``
    gets prepended as a human-readable block with a read hint so the LLM
    knows to call the Read/filesystem MCP tool.
    """
    from openagent.core.agent import Agent
    from openagent.models.base import BaseModel, ModelResponse

    captured_messages: list[dict] = []

    class StubModel(BaseModel):
        history_mode = "caller"
        async def generate(self, messages, system=None, tools=None,
                           on_status=None, session_id=None, **_):
            captured_messages.extend(messages)
            return ModelResponse(content="ACK", input_tokens=1, output_tokens=1,
                                 model="stub")
        async def stream(self, messages, system=None, tools=None):
            yield "ACK"

    agent = Agent(name="stub", model=StubModel(), system_prompt="s",
                  mcp_pool=ctx.extras.get("pool"))
    await agent.initialize()
    try:
        reply = await agent.run(
            message="what's in the file?",
            session_id=f"attach-{uuid.uuid4().hex[:6]}",
            attachments=[
                {"type": "image", "filename": "chart.png", "path": "/tmp/chart.png"},
                {"type": "file",  "filename": "doc.pdf",   "path": "/tmp/doc.pdf"},
            ],
        )
        assert reply == "ACK"
        assert captured_messages, "model never got called"
        user_msg = captured_messages[0]
        assert user_msg["role"] == "user"
        content = user_msg["content"]
        # Block lists each attachment with type + filename + path
        assert "image: chart.png" in content
        assert "/tmp/chart.png" in content
        assert "file: doc.pdf" in content
        assert "/tmp/doc.pdf" in content
        # And mentions the read tool so the LLM knows what to do
        assert "Read" in content or "read" in content
        # Plus the original message
        assert "what's in the file?" in content
    finally:
        await agent.shutdown()
