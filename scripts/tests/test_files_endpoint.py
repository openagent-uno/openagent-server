"""GET /api/files endpoint tests.

Covers the outbound-file serving path the agent uses to hand local
attachments back to remote clients (desktop app, CLI).

When the agent includes ``[IMAGE:/tmp/foo.png]`` / ``[FILE:/path]`` in
its response, the gateway strips the marker and delivers
``{path, filename, type}`` to the client via the WS ``response``
message. For a local install the client can read ``path`` directly;
for a remote install (app on laptop, agent on VPS) it fetches the
bytes via ``GET /api/files?path=<abs>`` — the endpoint these tests
lock down.

Tests:

1. **Happy path** — serve an on-disk file with the expected bytes and
   Content-Disposition filename.
2. **Missing path param** — 400.
3. **File not found** — 404.
4. **Symlink resolves through realpath** — a symlink to a real file
   serves the file rather than 404'ing.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

from ._framework import TestContext, TestSkip, test


async def _get_file(port: int, path: str) -> tuple[int, bytes, dict[str, str]]:
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/files", params={"path": path}) as r:
            body = await r.read()
            return r.status, body, dict(r.headers)


@test("files_endpoint", "GET /api/files returns the file bytes + Content-Disposition")
async def t_files_happy(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")

    tmp_dir = tempfile.mkdtemp(prefix="oa_files_test_")
    fpath = pathlib.Path(tmp_dir) / "agent-report.txt"
    payload = b"hello from the agent's filesystem"
    fpath.write_bytes(payload)

    status, body, headers = await _get_file(port, str(fpath))
    assert status == 200, f"expected 200, got {status}: {body[:200]!r}"
    assert body == payload, "body bytes don't match the on-disk file"
    # aiohttp FileResponse should advertise the filename so browsers save
    # with the original name rather than a random URL slug.
    disp = headers.get("Content-Disposition", "")
    assert "agent-report.txt" in disp, f"unexpected Content-Disposition: {disp!r}"


@test("files_endpoint", "GET /api/files without path returns 400")
async def t_files_missing_path(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/files") as r:
            assert r.status == 400, f"expected 400, got {r.status}"


@test("files_endpoint", "GET /api/files for a nonexistent path returns 404")
async def t_files_not_found(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    status, body, _ = await _get_file(port, "/tmp/openagent-does-not-exist-xyz123.bin")
    assert status == 404, f"expected 404, got {status}: {body[:200]!r}"


@test("files_endpoint", "GET /api/files resolves symlinks via realpath")
async def t_files_symlink(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")

    tmp_dir = tempfile.mkdtemp(prefix="oa_files_symlink_")
    real = pathlib.Path(tmp_dir) / "real.bin"
    payload = b"real-file-content-42"
    real.write_bytes(payload)
    link = pathlib.Path(tmp_dir) / "link.bin"
    os.symlink(real, link)

    status, body, _ = await _get_file(port, str(link))
    assert status == 200, f"expected 200 following symlink, got {status}"
    assert body == payload, "symlink target served wrong bytes"
