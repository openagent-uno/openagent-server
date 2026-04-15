"""POST /api/upload endpoint tests.

Covers the HTTP-level behaviour the desktop/browser clients depend on
without burning tokens on the model. End-to-end tests (upload → agent
reads file → marker echoed) live in ``test_files.py``.

Tests:

1. **Plain text upload** — the original smoke test. Proves the endpoint
   accepts multipart uploads and returns a usable server-side path.
2. **Binary upload preserves bytes verbatim** — required by the Electron
   flow, which reads arbitrary file types (PDFs, images, zips) through
   the ``dialog:readFile`` IPC and posts them unchanged.
3. **Filename with spaces, unicode, and punctuation round-trips** — the
   multipart parser on aiohttp's side is known to mangle certain filenames
   if the Content-Disposition header isn't encoded; this locks down the
   expected behaviour so desktop users picking a file named
   "Relazione finale — 2026.pdf" don't end up with ``upload.bin``.
4. **Empty file is accepted** — iOS Expo and some web browsers occasionally
   produce 0-byte form fields when the user cancels mid-pick. The server
   should still persist them as empty files, not 500.
5. **Concurrent uploads don't collide** — each call gets its own tempdir,
   so two simultaneous uploads of the same filename both succeed and
   resolve to different paths.
"""
from __future__ import annotations

import asyncio
import os
import uuid

from ._framework import TestContext, TestSkip, test


async def _post_file(port: int, content: bytes, filename: str, content_type: str = "application/octet-stream") -> dict:
    import aiohttp
    async with aiohttp.ClientSession() as http:
        data = aiohttp.FormData()
        data.add_field("file", content, filename=filename, content_type=content_type)
        async with http.post(f"http://127.0.0.1:{port}/api/upload", data=data) as r:
            if r.status == 401:
                raise TestSkip("upload endpoint behind auth")
            assert r.status in (200, 201), f"unexpected status {r.status}: {await r.text()}"
            return await r.json()


@test("upload", "POST /api/upload accepts a file + returns a path")
async def t_file_upload(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    payload = b"hello from openagent test upload"
    body = await _post_file(port, payload, "test.txt", "text/plain")
    assert "path" in body, body
    assert "filename" in body, body
    assert os.path.exists(body["path"]), f"upload didn't persist: {body['path']}"
    # Server must keep bytes verbatim — no whitespace trimming, no re-encoding.
    with open(body["path"], "rb") as f:
        assert f.read() == payload


@test("upload", "binary upload preserves bytes verbatim")
async def t_binary_upload_preserves_bytes(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")

    # Every byte value 0x00-0xFF, repeated — exercises the full byte range
    # so we'd notice any UTF-8-aware middleware that silently "fixes"
    # invalid sequences. Also includes NUL and CRLF, the two bytes most
    # frequently mishandled by broken multipart parsers.
    payload = bytes(range(256)) * 64  # 16 KB

    body = await _post_file(port, payload, "binary.bin", "application/octet-stream")
    with open(body["path"], "rb") as f:
        got = f.read()
    assert len(got) == len(payload), f"size mismatch: {len(got)} vs {len(payload)}"
    assert got == payload, "bytes differ — middleware must not touch the payload"


@test("upload", "filename with spaces/unicode/punctuation round-trips")
async def t_filename_special_chars(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")

    # Mix: latin-1 space, U+2014 em dash, apostrophe, parentheses — all
    # things a real user can produce from the macOS picker without
    # realising. ``aiohttp`` encodes Content-Disposition per RFC 6266 so
    # these must come back as-sent.
    filename = f"Relazione finale — 2026 (v2) '{uuid.uuid4().hex[:4]}'.pdf"
    payload = b"%PDF-1.4\n%not-a-real-pdf-just-a-marker"

    body = await _post_file(port, payload, filename, "application/pdf")
    # The server currently returns whatever filename it decided on; we
    # require that the *path* reflects the original so the LLM gets a
    # meaningful hint. Accept either exact match or a sanitized form that
    # preserves the recognisable parts.
    assert "Relazione" in body["path"] or "Relazione" in body.get("filename", ""), (
        f"original filename lost: returned={body}"
    )
    assert os.path.exists(body["path"])
    with open(body["path"], "rb") as f:
        assert f.read() == payload


@test("upload", "empty file is accepted, not 500")
async def t_empty_file_upload(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    body = await _post_file(port, b"", "empty.txt", "text/plain")
    assert os.path.exists(body["path"])
    assert os.path.getsize(body["path"]) == 0


@test("upload", "concurrent uploads don't collide on the same filename")
async def t_concurrent_uploads(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")

    async def _one(i: int) -> dict:
        return await _post_file(
            port,
            f"payload-{i}".encode(),
            "same.txt",  # All 5 uploads use the SAME filename on purpose.
            "text/plain",
        )

    results = await asyncio.gather(*[_one(i) for i in range(5)])
    paths = [r["path"] for r in results]
    # Each upload must land in its own file — otherwise two concurrent
    # chat attachments with the same filename would clobber each other.
    assert len(set(paths)) == len(paths), f"duplicate paths: {paths}"
    for i, p in enumerate(paths):
        with open(p, "rb") as f:
            assert f.read() == f"payload-{i}".encode()
