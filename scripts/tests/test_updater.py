"""Auto-updater module — read-only sanity check.

Does not hit the network. Just confirms the symbols exist and the
current package reports a sane ``__version__``.
"""
from __future__ import annotations

import io
import json
import tarfile
from unittest.mock import patch

from ._framework import TestContext, test


@test("updater", "updater symbols exist + current __version__ is sane")
async def t_updater_callable(ctx: TestContext) -> None:
    import openagent
    from openagent.updater import check_for_update, UpdateInfo, perform_self_update_sync
    assert openagent.__version__ and isinstance(openagent.__version__, str)
    assert callable(check_for_update)
    assert callable(perform_self_update_sync)
    fields = getattr(UpdateInfo, "_fields", None)
    assert fields and len(fields) >= 1


class _FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@test("updater", "check_for_update prefers server asset over CLI asset")
async def t_updater_prefers_server_asset(ctx: TestContext) -> None:
    import openagent
    import openagent.updater as updater

    payload = {
        "tag_name": "v0.5.17",
        "assets": [
            {
                "name": "openagent-cli-0.5.17-linux-x64.tar.gz",
                "browser_download_url": "https://example.invalid/openagent-cli.tgz",
            },
            {
                "name": "openagent-cli-0.5.17-linux-x64.tar.gz.sha256",
                "browser_download_url": "https://example.invalid/openagent-cli.tgz.sha256",
            },
            {
                "name": "openagent-0.5.17-linux-x64.tar.gz",
                "browser_download_url": "https://example.invalid/openagent.tgz",
            },
            {
                "name": "openagent-0.5.17-linux-x64.tar.gz.sha256",
                "browser_download_url": "https://example.invalid/openagent.tgz.sha256",
            },
        ],
    }

    with (
        patch.object(openagent, "__version__", "0.5.16"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(
            updater,
            "urlopen",
            return_value=_FakeHTTPResponse(json.dumps(payload).encode()),
        ),
    ):
        info = updater.check_for_update()

    assert info is not None
    assert info.download_url.endswith("/openagent.tgz"), info
    assert info.checksum_url and info.checksum_url.endswith("/openagent.tgz.sha256"), info


@test("updater", "download_update rejects archives without server binary")
async def t_updater_rejects_cli_only_archive(ctx: TestContext) -> None:
    import openagent.updater as updater

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"fake-cli-binary"
        info = tarfile.TarInfo("bin/openagent-cli")
        info.size = len(data)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(data))
    archive = buf.getvalue()

    with (
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=_FakeHTTPResponse(archive)),
    ):
        try:
            updater.download_update("https://example.invalid/openagent-0.5.17-linux-x64.tar.gz")
        except RuntimeError as exc:
            assert "did not contain the OpenAgent server executable" in str(exc)
        else:
            raise AssertionError("download_update should reject archives without the server binary")
