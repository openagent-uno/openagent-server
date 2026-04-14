"""POST /api/upload smoke test (endpoint-level only).

For full upload → agent → response coverage see ``test_files.py``.
"""
from __future__ import annotations

from ._framework import TestContext, TestSkip, test


@test("upload", "POST /api/upload accepts a file + returns a path")
async def t_file_upload(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    payload = b"hello from openagent test upload"
    async with aiohttp.ClientSession() as http:
        data = aiohttp.FormData()
        data.add_field("file", payload, filename="test.txt", content_type="text/plain")
        async with http.post(f"http://127.0.0.1:{port}/api/upload", data=data) as r:
            if r.status == 401:
                raise TestSkip("upload endpoint behind auth")
            assert r.status in (200, 201), f"unexpected status: {r.status}"
            body = await r.json()
            assert "path" in body or "url" in body or "filename" in body, body
